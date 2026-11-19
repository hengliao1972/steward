"""Fusion feedback preserves sensor ownership, map coordinates and unwrapped control headings."""

import math
import copy
from pathlib import Path

import numpy as np
import pytest

from steward.ego.localization_feedback import LocalizationFeedback
from steward.ego.runtime import EgoRuntime, RuntimeConfig
from steward.ego.segmenter import Segmenter, Segmentation, SegmenterError
from steward.assets.generators import steward_bot
from steward.world.sim import WorldSim


def test_feedback_projects_a_copy_and_does_not_double_correct_or_cross_episodes():
    feedback = LocalizationFeedback()
    raw = {"episode": 7, "stamp": 2.0, "odometry": {"x": 3.0, "y": 4.0, "yaw": 4*math.pi+0.1,
                                                      "sigma": {"pos": 0.003, "yaw": 0.003}}}
    assert feedback.project(raw) is raw

    feedback.publish({"episode": 7, "stamp": 1.0, "mapFromOdometry": [1, 2, math.pi/2],
                      "filterCovariance": np.diag([0.01**2, 0.02**2, 0.03**2]).tolist()})
    mapped = feedback.project(raw)
    assert mapped["stamp"] == 2.0
    assert mapped["odometry"]["x"] == pytest.approx(-3)
    assert mapped["odometry"]["y"] == pytest.approx(5)
    assert mapped["odometry"]["yaw"] == pytest.approx(raw["odometry"]["yaw"]+math.pi/2)
    assert mapped["odometry"]["sigma"] == {"pos": 0.02, "yaw": 0.03}
    assert feedback.project(mapped) is mapped
    assert raw["odometry"]["x"] == 3 and raw["odometry"]["sigma"]["pos"] == 0.003
    other = dict(raw, episode=8)
    assert feedback.project(other) is other
    feedback.reset()
    assert feedback.project(raw) is raw


def _runtime(monkeypatch, link):
    class Provider:
        calls = 0
        def segment(self, rgb):
            self.calls += 1
            return Segmentation(np.zeros(rgb.shape[:2], np.uint8), {})
    provider = Provider()
    monkeypatch.setattr(Segmenter, "from_files", classmethod(lambda cls, **kwargs: provider))
    cfg = RuntimeConfig(localization_config=str(Path(__file__).resolve().parents[1]/"config/localization.toml"),
                        localization_apply=True)
    return EgoRuntime(link, steward_bot().to_dict(), cfg), provider


def test_semantic_frames_are_reused_and_anonymous_masks_are_not_trusted(monkeypatch):
    runtime, provider = _runtime(monkeypatch, object())
    header = {"stamp": 0.0, "episode": 1}
    mask = np.ones((8, 8), np.uint8)
    frame = {"header": header, "camera": "wide_camera", "mask": mask, "rgb": np.zeros((8, 8, 3), np.uint8)}
    localized = runtime._localization_frames(header, [frame])
    assert provider.calls == 1 and not localized[0]["mask"].any()
    assert mask.all(), "the main perception frame must retain its original mask"
    runtime._next_localization = 0
    semantic = {**frame, "labels": {}}
    assert runtime._localization_frames(header, [semantic]) == [semantic]
    assert provider.calls == 1, "even an empty SAM prediction must not be requested twice"
    assert runtime._localization_frames({**header, "stamp": 0.1}, []) == []


def test_feedback_reaches_body_detections_live_state_and_keeps_existing_map_fixed(monkeypatch, kitchen_scene):
    sim = WorldSim(kitchen_scene)
    try:
        sim.step(250)
        header = sim.sensors("w:robot-1")
    finally:
        sim.stop()
    class Link:
        def sensors(self):
            return copy.deepcopy(header)
    raw = EgoRuntime(Link(), steward_bot().to_dict())
    raw._poll_once()
    fused, _provider = _runtime(monkeypatch, Link())
    correction = [0.04, -0.03, 0.02]
    result = {"episode": header["episode"], "stamp": header["stamp"], "mapFromOdometry": correction,
              "filterCovariance": np.diag([0.005**2, 0.005**2, 0.004**2]).tolist()}
    monkeypatch.setattr(fused, "_localization_frames", lambda *_args: [])
    monkeypatch.setattr(fused, "_advance_localization", lambda *_args: fused._feedback.publish(result))
    fused._poll_once()
    expected = fused._feedback.project(header)["odometry"]
    assert fused.estimator.odometry["x"] == pytest.approx(expected["x"])
    assert fused._motion_state()["odometry"]["x"] == pytest.approx(expected["x"])
    assert len(raw.estimator.tracks) == len(fused.estimator.tracks) > 0
    c, s = math.cos(correction[2]), math.sin(correction[2])
    for key, track in raw.estimator.tracks.items():
        xy = np.array([[c, -s], [s, c]]) @ track.pos[:2] + correction[:2]
        np.testing.assert_allclose(fused.estimator.tracks[key].pos[:2], xy, atol=1e-10)
    before = {key: track.pos.copy() for key, track in fused.estimator.tracks.items()}
    fused._feedback.publish({**result, "mapFromOdometry": [0.08, 0, 0.03]})
    fused.ego_state()
    for key, position in before.items():
        np.testing.assert_array_equal(fused.estimator.tracks[key].pos, position)
    assert fused._fusion_stats["mappedHeaders"] > 0
    fused._restart()
    assert fused._feedback.snapshot is None


def test_feedback_cannot_be_enabled_without_a_localization_configuration():
    with pytest.raises(ValueError, match="requires localization_config"):
        EgoRuntime(object(), steward_bot().to_dict(), RuntimeConfig(localization_apply=True))


def test_segmentation_failure_is_visible_and_preserves_prediction(monkeypatch):
    runtime, provider = _runtime(monkeypatch, object())
    def fail(_rgb):
        raise SegmenterError("service unavailable")
    monkeypatch.setattr(provider, "segment", fail)
    alerts = []
    monkeypatch.setattr(runtime, "alert", lambda *args: alerts.append(args))
    header = {"episode": 1, "stamp": 0.0, "odometry": {"x": 0., "y": 0., "yaw": 0.}}
    frame = {"header": header, "camera": "wide_camera", "rgb": np.zeros((8, 8, 3), np.uint8)}
    assert runtime._localization_frames(header, [frame]) == []
    runtime._advance_localization(header, [])
    status = runtime.localization_status()
    assert status["applied"] and status["state"] == "predicted"
    assert status["visualError"] == "service unavailable"
    assert runtime._fusion_stats["visualErrors"] == 1 and len(alerts) == 1
    assert runtime._feedback.snapshot is not None
