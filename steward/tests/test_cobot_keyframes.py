"""Dual-gripper state between segmented RGB-D keyframes."""

import copy

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.cobot_arm import PiperArm
from steward.ego.estimator import Track
from steward.ego.runtime import EgoRuntime
from steward.world.sim import WorldSim

from test_cobot_skills import _scene
from test_segmenter import StubSegmenter


@pytest.fixture()
def frame():
    sim = WorldSim(_scene())
    try:
        sim.perception.mode = "rgbd"
        sim.perception.masks = False
        header = sim.sensors("w:cobot")
        assert "gripper" not in header
        return header
    finally:
        sim.stop()


class RecordedLink:
    """Replay a sensor header without imaging or advancing the world."""

    def __init__(self, frame):
        self.frame = frame

    def sensors(self):
        return copy.deepcopy(self.frame)


def test_cobot_perception_updates_the_body_between_segmented_frames(frame):
    model = cobot_magic().to_dict()
    runtime = EgoRuntime(RecordedLink(frame), model, arm=PiperArm(model))
    runtime.perception_mode = "rgbd-segmenter"
    last = copy.deepcopy(frame)
    last["stamp"] -= 0.1
    runtime._keyframe = {camera: last for camera in runtime._cameras(frame)}

    runtime._poll_once()

    assert runtime.estimator.posed
    assert runtime.estimator.stamp == frame["stamp"]
    assert runtime.estimator.grippers == frame["grippers"]
    assert runtime.segmentation["frames"] == 0
    assert runtime._view_scene is not None


@pytest.mark.parametrize("segmentation_fails", [False, True])
def test_cobot_rgbd_perception_continues_after_segmenting_a_keyframe(segmentation_fails):
    sim = WorldSim(_scene())
    try:
        sim.perception.mode = "rgbd"
        sim.perception.masks = False

        class ImagingLink(RecordedLink):
            def rgbd(self, camera):
                return sim.rgbd("w:cobot", camera)

        link = ImagingLink(sim.sensors("w:cobot"))
        stub = StubSegmenter()
        stub.fail = segmentation_fails
        model = cobot_magic().to_dict()
        runtime = EgoRuntime(link, model, arm=PiperArm(model), segmenter=stub)
        runtime._poll_once()
        assert runtime.segmentation["failures" if segmentation_fails else "frames"] > 0
        calls = stub.calls
        link.frame["stamp"] += 0.1

        runtime._poll_once()

        assert runtime.estimator.stamp == link.frame["stamp"]
        assert runtime.estimator.grippers == link.frame["grippers"]
        assert stub.calls == calls, "unchanged cameras reuse the segmented keyframe"
        assert not any("感知循环出现异常" in alert["message"] for alert in runtime.alerts)
    finally:
        sim.stop()


@pytest.mark.parametrize("gripper", ["L_link6", "R_link6"])
@pytest.mark.parametrize("field,value", [("state", "closed"), ("holding", True), ("holdingHandle", True)])
def test_each_gripper_change_requests_only_a_near_camera_keyframe(frame, gripper, field, value):
    model = cobot_magic().to_dict()
    runtime = EgoRuntime(RecordedLink(frame), model, arm=PiperArm(model))
    runtime._keyframe = {camera: copy.deepcopy(frame) for camera in runtime._cameras(frame)}
    changed = copy.deepcopy(frame)
    changed["stamp"] += 0.1
    changed["grippers"][gripper][field] = value

    assert runtime._keyframe_due(changed, "wide_camera")
    assert not runtime._keyframe_due(changed, "head_camera")


def test_a_gripper_change_is_visible_even_when_the_other_remains_closed(frame):
    model = cobot_magic().to_dict()
    runtime = EgoRuntime(RecordedLink(frame), model, arm=PiperArm(model))
    for grip in frame["grippers"].values():
        grip["state"] = "closed"
    runtime._keyframe = {camera: copy.deepcopy(frame) for camera in runtime._cameras(frame)}
    changed = copy.deepcopy(frame)
    changed["stamp"] += 0.1
    changed["grippers"]["L_link6"]["state"] = "open"

    assert runtime._keyframe_due(changed, "wide_camera")


def test_body_updates_keep_the_load_on_its_holding_arm_until_that_arm_releases(frame):
    model = cobot_magic().to_dict()
    runtime = EgoRuntime(RecordedLink(frame), model, arm=PiperArm(model))
    est = runtime.estimator
    est.update(frame)
    grip = "R_link6"
    pos = est.gripper_position(grip)
    track = Track(id="e:1", category_counts={"bowl": 1.0}, pos=pos.copy(), var=np.full(3, 1e-4),
                  yaw=0.0, yaw_var=1e-4, size=np.array([0.15, 0.15, 0.06]), observations=1,
                  first_seen=0.0, last_seen=0.0)
    est.tracks[track.id] = track
    est.mark_held(track.id, since=frame["stamp"], gripper=grip, body=frame)
    moved = copy.deepcopy(frame)
    moved["stamp"] += 0.1
    moved["odometry"]["x"] += 0.2
    moved["grippers"][grip].update(state="closed", holding=True)

    est.update_body(moved)

    assert est.held_id == track.id
    assert est.grippers == moved["grippers"]
    assert track.pos == pytest.approx(pos + np.array([0.2, 0.0, 0.0]))
    moved["stamp"] += 0.1
    moved["grippers"][grip].update(state="open", holding=False)
    moved["grippers"]["L_link6"].update(state="closed", holding=True)
    est.update_body(moved)
    assert est.held_id is None
