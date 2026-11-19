"""Semantic-depth selection, gated correction, prediction continuity and runtime reuse."""

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.assets.generators import steward_bot
from steward.ego.distance_localization import (
    DistanceFieldLocalizer, LocalizationConfig, SemanticPointSelector,
)
from steward.ego.runtime import EgoRuntime, RuntimeConfig
from steward.ego.server import ego_api
from steward.net import call
from steward.ego.pose_ekf import EkfConfig


def header(stamp=0.0, episode=1, pose=(0, 0, 0)):
    return {"stamp": stamp, "episode": episode, "odometry": dict(zip(("x", "y", "yaw"), pose)),
            "joints": {}, "gripper": {"state": "open", "holding": False}}


def room_points(pose=(0, 0, 0)):
    values = np.linspace(-1.4, 1.4, 90)
    world = np.vstack([np.c_[np.full(90, 2.0), values], np.c_[values, np.full(90, 2.5)],
                       np.c_[np.full(90, -2.0), values]])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    local = (world - pose[:2]) @ np.array([[c, -s], [s, c]])
    return np.c_[local, np.full(len(local), 0.9)]


def test_corrects_drifting_pose_and_keeps_map_from_odometry_during_prediction():
    tracker = DistanceFieldLocalizer(LocalizationConfig(ekf=EkfConfig(enabled=False)))
    assert tracker.update([0, 0, 0], 0, 1, room_points())["state"] == "initializing"
    truth = np.array([0.08, -0.04, 0.02])
    raw = truth + [0.08, -0.06, 0.025]
    result = tracker.update(raw, 0.2, 1, room_points(truth))
    assert result["accepted"], result
    assert np.linalg.norm(np.array(result["pose"])[:2] - truth[:2]) < 0.035
    assert abs(result["pose"][2] - truth[2]) < 0.015
    correction = tracker.correction.copy()
    version = tracker.map_version
    future = tracker.update(raw + [0.1, 0, 0], 0.3, 1)
    assert future["state"] == "predicted" and not future["accepted"]
    assert np.linalg.norm(np.array(future["pose"]) - result["pose"]) > 0.09
    np.testing.assert_array_equal(tracker.correction, correction)
    assert tracker.map_version == version
    assert tracker.update(raw, 4, 1)["stale"]


def test_single_wall_is_rejected_and_cannot_contaminate_map():
    tracker = DistanceFieldLocalizer()
    wall = room_points()[:90]
    tracker.update([0, 0, 0], 0, 1, wall)
    before = tracker.occupied.copy()
    result = tracker.update([0.04, 0.1, 0], 0.2, 1, wall)
    assert not result["accepted"], result
    assert result["reason"] == "degenerate_geometry", result
    np.testing.assert_array_equal(tracker.occupied, before)
    np.testing.assert_array_equal(tracker.correction, np.zeros(3))


def test_unknown_area_cannot_win_and_duplicate_time_cannot_mutate_map():
    tracker = DistanceFieldLocalizer()
    tracker.update([0, 0, 0], 0, 1, room_points())
    before = tracker.occupied.copy()
    result = tracker.update([50, 50, 0], 1, 1, room_points())
    assert result["reason"] == "insufficient_overlap"
    with pytest.raises(ValueError, match="strictly increase"):
        tracker.update([0, 0, 0], 1, 1, room_points())
    np.testing.assert_array_equal(before, tracker.occupied)
    result = tracker.update([0, 0, 0], 0, 2)
    assert result["state"] == "predicted" and tracker.map_version == 0
    assert tracker.occupied is None and tracker.last_correction_stamp is None


def test_selector_requires_semantics_excludes_movable_labels_and_limits_region_points(box_depth):
    cfg = LocalizationConfig(camera="head_camera", points_per_region=50)
    selector = SemanticPointSelector(steward_bot().to_dict(), cfg)
    depth, owners = box_depth((120, 160), 60, [((2, -2, 0), (2.2, 2, 3))])
    mask = np.where(owners >= 0, 1, 0).astype(np.uint8)
    frame = {"header": header(), "camera": cfg.camera, "depth": depth, "mask": mask,
             "fovy": 60, "rgb": np.zeros((120, 160, 3), np.uint8)}
    assert not len(selector.select(frame)[0]), "anonymous simulator masks are not semantic evidence"
    frame["labels"] = {1: {"person": 0.9, "wall": 0.8}}
    assert not len(selector.select(frame)[0])
    frame["labels"] = {1: {"wall": 0.9}}
    points, selection = selector.select(frame)
    assert 10 < len(points) <= 50 and selection["regions"] == 1
    assert np.isfinite(points).all()
    assert (points[:, 2] >= cfg.height_edges_m[0]).all()
    frame["mask"] = np.ones((2, 2), np.uint8)
    with pytest.raises(ValueError, match="dimensions"):
        selector.select(frame)


def test_runtime_consumes_existing_mask_once_without_calling_segmenter(tmp_path, box_depth):
    source = Path(__file__).resolve().parents[1] / "config/localization.toml"
    path = tmp_path / "config.toml"
    path.write_text(source.read_text().replace('camera = "wide_camera"', 'camera = "head_camera"'))
    runtime = EgoRuntime(object(), steward_bot().to_dict(), RuntimeConfig(localization_config=str(path)))
    depth, owners = box_depth((120, 160), 60, [((2, -2, 0), (2.2, 2, 3))])
    frame = {"header": header(), "camera": "head_camera", "depth": depth,
             "mask": np.where(owners >= 0, 1, 0), "labels": {1: {"wall": 0.9}}, "fovy": 60}
    # Any accidental segmentation request would fail: this object exposes no model methods.
    runtime.segmenter = object()
    runtime._advance_localization(header(), [frame])
    result = runtime.localization_status()
    assert result["state"] == "initializing" and result["selection"]["points"] >= 40
    runtime._advance_localization(header(), [frame])
    assert runtime.localization_status() == result
    runtime._advance_localization(header(0.2), [])
    assert runtime.localization_status()["state"] == "predicted"
    assert runtime.localizer.tracker.map_version == 1


@pytest.mark.parametrize("values", [{"resolution_m": 0}, {"height_edges_m": (1, 0)},
                                      {"min_overlap": 2}, {"max_points": 2}, {"max_depth_m": float("nan")}])
def test_invalid_configuration_fails_before_processing(values):
    with pytest.raises(ValueError):
        LocalizationConfig(**values)


def test_dynamic_grid_grows_only_with_accepted_observations(monkeypatch):
    tracker = DistanceFieldLocalizer(LocalizationConfig(map_size_m=1.0, dynamic_map=True, min_points=1,
                                                        ekf=EkfConfig(enabled=False)))
    seed = np.array([[0., 0., 0.4], [0., 0., 1.]])
    first = tracker.update([0., 0., 0.], 0.0, 1, seed)
    assert first["mapGrid"]["shape"] == [21, 21]
    additional = np.array([[-1.2, 0., 0.4]])
    monkeypatch.setattr(tracker, "_match", lambda _p, _b, pose: (pose, False, "poor_alignment", {}))
    rejected = tracker.update([0.2, 0., 0.], 1.0, 1, additional)
    assert rejected["mapGrid"] == first["mapGrid"]
    monkeypatch.setattr(tracker, "_match", lambda _p, _b, pose: (pose, True, "matched", {}))
    accepted = tracker.update([0.2, 0., 0.], 2.0, 1, additional)
    assert accepted["mapGrid"]["origin"][0] < first["mapGrid"]["origin"][0]
    assert accepted["mapGrid"]["shape"][0] > first["mapGrid"]["shape"][0]
    points = np.array([[0., 0.], [0., 0.], [-1., 0.]])
    np.testing.assert_allclose(tracker._distances(points, np.array([0, 1, 0]), np.zeros(3)), 0, atol=1e-12)
    assert tracker._distances(points[-1:], np.array([1]), np.zeros(3))[0] == pytest.approx(0.35)
    limited = tracker.update([0.4, 0., 0.], 3.0, 1, np.array([[-100., 0., 0.4]]))
    assert limited["mapGrid"] == accepted["mapGrid"]
    assert limited["metrics"]["mapInsertSkipped"] == "map_grid_limit"


def test_signed_wall_field_uses_observed_side_and_absolute_capture_distance():
    tracker = DistanceFieldLocalizer(LocalizationConfig(map_size_m=1.0, dynamic_map=True,
                                                        min_points=1, signed_walls=True))
    samples = np.array([[0., 0., 0.4], [0., 0., 1.]])
    normals = np.array([[-1., 0.], [0., 0.]])
    tracker.update([0., 0., 0.], 0.0, 1, samples, wall_normals=normals)
    query = np.array([[-0.15, 0.], [0.15, 0.], [0.15, 0.], [0.4, 0.]])
    np.testing.assert_allclose(tracker._distances(query, np.array([0, 0, 1, 0]), np.zeros(3)),
                               [0.15, -0.15, 0.15, -0.35], atol=1e-12)
    _, accepted, reason, _ = tracker._match(query[-1:], np.array([0]), np.zeros(3))
    assert not accepted and reason == "insufficient_overlap"
    assert tracker._insert(samples, np.array([0, 1]), np.zeros(3), -normals)
    assert tracker._distances(query[1:2], np.array([0]), np.zeros(3))[0] == pytest.approx(0.15)


def test_map_clears_visible_old_surfaces_after_separate_accepted_insertions(monkeypatch):
    cfg = LocalizationConfig(map_size_m=1., dynamic_map=True, clear_observations=3, signed_walls=True,
                             min_points=1, ekf=EkfConfig(enabled=False))
    tracker = DistanceFieldLocalizer(cfg)
    old = np.array([[1., 0., .4], [1., .3, .62]])
    wall = np.array([[-1., 0.], [-1., 0.]])
    tracker.update([0., 0., 0.], 0., 1, old, wall_normals=wall, camera_origin=np.array([0., 0., .4]))
    replacement = np.array([[1.1, 0., .4], [1.11, 0., .4], [1.1, .33, .4], [-1.5, 0., .4]])
    zeros = np.zeros(3)

    def cells():
        return tracker._distances(old[:, :2], np.zeros(2, dtype=int), zeros)

    def observe(stamp, samples):
        x = stamp * .2
        return tracker.update([x, 0., 0.], stamp, 1, samples - [x, 0., 0.], camera_origin=np.array([-x, 0., .4]))

    monkeypatch.setattr(tracker, "_match", lambda _p, _b, pose: (pose, False, "poor_alignment", {}))
    assert not observe(1., replacement)["accepted"]
    np.testing.assert_allclose(cells(), [0, 0], atol=1e-12)
    monkeypatch.setattr(tracker, "_match", lambda _p, _b, pose: (pose, True, "matched", {}))
    assert observe(2., replacement)["mapClearedCells"] == 0
    assert observe(3., np.concatenate((replacement, old[:1])))["mapClearedCells"] == 0
    for stamp in (4., 5.):
        assert observe(stamp, replacement)["mapClearedCells"] == 0
        np.testing.assert_allclose(cells(), [0, 0], atol=1e-12)
    assert observe(6., replacement)["mapClearedCells"] == 1
    np.testing.assert_allclose(cells(), [.1, 0], atol=1e-12)


def test_shipped_config_and_runtime_reuse_publish_independent_pose(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config/localization.toml"
    cfg = LocalizationConfig.load(source)
    model = steward_bot().to_dict()
    runtime = EgoRuntime(object(), model, RuntimeConfig(localization_config=str(source)), out_dir=tmp_path)
    raw_scene = runtime.estimator.to_scene().to_dict()
    runtime._advance_localization(header(), [])
    runtime._advance_localization(header(0.1), [])
    assert runtime.localization_status()["state"] == "predicted"
    assert runtime.estimator.to_scene().to_dict() == raw_scene
    original = runtime.localization_status()
    original["pose"][0] = 123
    assert runtime.localization_status()["pose"][0] == 0
    runtime._advance_localization(header(0.1), [])
    assert len((tmp_path / "localization.jsonl").read_text().splitlines()) == 2
    server = ego_api(runtime, port=0).start()
    try:
        reply = call("GET", server.url + "/ego/localization")
        assert reply["enabled"] and reply["stamp"] == 0.1
    finally:
        server.stop()
    runtime._restart()
    assert runtime.localization_status() == {"enabled": True, "state": "waiting"}
    assert runtime.localizer.tracker.map_version == 0
    assert cfg.ekf.min_visual_interval_s == 0.5
    assert cfg.dynamic_map


def test_disabled_runtime_needs_no_semantic_observations():
    runtime = EgoRuntime(object(), steward_bot().to_dict())
    runtime._advance_localization(header(), [])
    assert runtime.localization_status() == {"enabled": False}


def test_cli_selects_opt_in_configuration_and_rejects_unsupported_runtime(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from steward import cli

    seen = []
    monkeypatch.setattr(cli, "cmd_up", lambda args: seen.append(args.localization_config) or 0)
    assert cli.main(["up", "--localization-config", "config/localization.toml"]) == 0
    assert cli.main(["up"]) == 0
    assert seen == ["config/localization.toml", None]
    body = "test-private-runtime"
    monkeypatch.setitem(cli.EMBODIMENTS, body, {"ego": "cobot-ego"})
    with pytest.raises(ValueError, match="shared Ego Runtime"):
        cli._EgoProcess(SimpleNamespace(localization_config="config/localization.toml"),
                        "http://127.0.0.1:1", tmp_path, embodiment=body, api_port=0)
