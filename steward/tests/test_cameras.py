"""The robot's two head cameras: the narrow head camera and the short-focal-length wide camera."""

from __future__ import annotations

import threading
import time

import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator, _box_corners
from steward.ego.perception import decode_rgbd
from steward.ego.runtime import EgoRuntime, RobotLink, RuntimeConfig
from steward.net import HttpError, call
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def _pump_until(host, done, timeout=60.0):
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def _off_main(host, fn):
    """Run ``fn`` on a worker thread while the main thread serves the world's render requests."""
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("value", fn()), daemon=True)
    thread.start()
    _pump_until(host, lambda: not thread.is_alive())
    return out["value"]


def test_each_camera_gives_its_own_frame_and_an_unknown_camera_is_refused(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=1)
    sim.perception.mode = "rgbd"
    assert [c["id"] for c in sim.sensors("w:robot-1")["cameras"]] == ["head_camera", "wide_camera"]
    head, wide = (decode_rgbd(sim.rgbd("w:robot-1", c)) for c in ("head_camera", "wide_camera"))
    assert head["camera"] == "head_camera" and head["rgb"].shape == (240, 320, 3) and head["fovy"] == 60.0
    assert wide["camera"] == "wide_camera" and wide["rgb"].shape == (360, 480, 3) and wide["fovy"] == 100.0
    assert decode_rgbd(sim.rgbd("w:robot-1"))["camera"] == "head_camera", "no camera named: the primary one"
    with pytest.raises(KeyError):
        sim.rgbd("w:robot-1", "tail_camera")


def test_the_robot_port_names_its_cameras_and_refuses_others(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, phase="phase1").start()
    try:
        url = host.robot_servers["w:robot-1"].url
        with pytest.raises(HttpError) as err:
            call("GET", f"{url}/sensors/rgbd?camera=tail_camera")
        assert err.value.status == 404
    finally:
        host.stop()


class OneCameraLink:
    """A robot port whose robot has only the head camera, as in a scene built with an older model."""

    def __init__(self, link: RobotLink):
        self.link = link

    def __getattr__(self, name):
        return getattr(self.link, name)

    def sensors(self):
        frame = self.link.sensors()
        frame["cameras"] = [c for c in frame["cameras"] if c["id"] == "head_camera"]
        frame["detections"] = [d for d in frame.get("detections", []) if d["camera"] == "head_camera"]
        return frame


def test_a_camera_the_world_lacks_is_reported_once_and_perception_goes_on_without_it(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, speed=4.0).start()
    runtime = EgoRuntime(OneCameraLink(RobotLink(host.robot_servers["w:robot-1"].url)), steward_bot().to_dict())
    runtime.start()
    try:
        _pump_until(host, lambda: len(runtime.estimator.tracks) > 3)
        for _ in range(20):
            host.pump(0.05)
        alerts = [a for a in runtime.alerts if "wide_camera" in a["message"]]
        assert len(alerts) == 1 and alerts[0]["count"] == 1
    finally:
        runtime.stop()
        host.stop()


def test_perception_cameras_must_be_cameras_of_the_robot():
    with pytest.raises(ValueError, match="tail_camera"):
        EgoRuntime(RobotLink("http://127.0.0.1:9"), steward_bot().to_dict(), RuntimeConfig(cameras=("tail_camera",)))


def _offline(config: RuntimeConfig | None = None) -> EgoRuntime:
    return EgoRuntime(RobotLink("http://127.0.0.1:9"), steward_bot().to_dict(), config)


def _header(stamp=1.0, x=0.0, yaw=0.0, lift=0.5, gripper="open", holding=False):
    return {"stamp": stamp, "odometry": {"x": x, "y": 0.0, "yaw": yaw},
            "joints": {"lift": lift, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
            "gripper": {"state": gripper, "holding": holding}}


def test_the_widest_camera_is_in_charge_of_near_regions_unless_one_is_named():
    assert _offline().near_camera == "wide_camera"
    assert _offline(RuntimeConfig(cameras=("head_camera",))).near_camera is None, "one camera: no division"
    assert _offline(RuntimeConfig(near_camera="head_camera")).near_camera == "head_camera"
    with pytest.raises(ValueError, match="near_camera"):
        _offline(RuntimeConfig(cameras=("head_camera",), near_camera="wide_camera"))


def test_a_look_around_samples_a_heading_where_a_long_table_is_fully_in_frame():
    """The head camera's horizontal view is about 1.3 rad. A long table up close is fully inside
    it for only ~0.22 rad of heading. A look-around at the navigation rate turns 0.36 rad between
    frames and can land only on the truncated views, which do not start a track. While perception
    keeps keyframes the look-around steps 0.1 rad, and a stop also becomes a keyframe."""
    cfg = RuntimeConfig()
    old_step = 3.6 / cfg.perception_hz
    new_step = max(cfg.motion.look_angular_speed / cfg.perception_hz, cfg.keyframe_turn_rad)
    window = 0.22
    assert old_step > window and new_step <= window
    runtime = _offline()
    runtime._keyframe = {"head_camera": _header()}
    assert not runtime._keyframe_due(_header(stamp=1.2, yaw=0.05), "head_camera"), "0.05 rad is still one view"
    assert runtime._keyframe_due(_header(stamp=1.3, yaw=0.12), "head_camera")
    runtime._keyframe = {"head_camera": _header()}
    runtime._pose_sample = None
    assert not runtime._keyframe_due(_header(stamp=1.1, yaw=0.05), "head_camera")
    assert runtime._keyframe_due(_header(stamp=1.6, yaw=0.05), "head_camera"), "stopped on a new heading"


def test_arm_motion_with_the_base_still_makes_a_keyframe_of_the_near_camera_only():
    runtime = _offline()
    runtime._keyframe = {c: _header() for c in ("head_camera", "wide_camera")}
    lifted, gripped = _header(stamp=2.0, lift=0.6), _header(stamp=2.0, gripper="closed", holding=True)
    assert runtime._keyframe_due(lifted, "wide_camera") and not runtime._keyframe_due(lifted, "head_camera")
    assert runtime._keyframe_due(gripped, "wide_camera") and not runtime._keyframe_due(gripped, "head_camera")
    assert not runtime._keyframe_due(_header(stamp=2.0, lift=0.52), "wide_camera"), "a small lift is no new view"
    assert runtime._keyframe_due(_header(stamp=2.0, x=0.2), "head_camera"), "base motion still makes keyframes"


def test_of_one_instant_each_camera_fuses_the_regions_of_its_range_that_the_other_has_in_view():
    runtime = _offline()
    head, wide = ({"camera": c, "header": _header()} for c in ("head_camera", "wide_camera"))
    batch = [head, wide]
    head_defers, wide_defers = runtime._handoff(head, batch), runtime._handoff(wide, batch)
    cam = np.asarray(runtime.estimator.camera_pose(_header()["odometry"], "head_camera")["p"])

    def box(x, y, z, size=0.1):
        return _box_corners(np.array([x, y, z]), np.full(3, size), 0.0)

    near, far = box(cam[0] + 0.7, 0.0, 0.4), box(cam[0] + 3.0, 0.0, 0.5)
    side = box(cam[0] + 2.0 * math.cos(math.radians(50)), 2.0 * math.sin(math.radians(50)), 0.5)
    long_table = _box_corners(np.array([cam[0] + 1.6, 0.0, 0.4]), np.array([1.2, 2.4, 0.75]), 0.0)
    assert head_defers(near) and not head_defers(far), "near regions go to the wide camera"
    assert wide_defers(far) and not wide_defers(near), "far regions go to the head camera"
    assert not wide_defers(side), "a far region beside the head camera's view stays with the wide camera"
    assert not wide_defers(long_table), "the head camera sees only part of the long table: the wide camera keeps it"
    assert runtime._handoff(wide, [wide]) is None, "a lone frame fuses everything"


def test_a_deferred_region_keeps_its_track_but_adds_no_observation_or_track(kitchen_scene):
    sim = WorldSim(kitchen_scene, seed=1)
    sim.perception.mode = "rgbd"
    head, wide = (decode_rgbd(sim.rgbd("w:robot-1", c)) for c in ("head_camera", "wide_camera"))
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update_rgbd(head, defer=lambda corners: True)
    assert not est.tracks, "every region deferred: no track"
    est.update_rgbd(wide)
    seen = {tid: t.observations for tid, t in est.tracks.items()}
    assert seen
    est.update_rgbd(head, defer=lambda corners: True)
    assert set(est.tracks) <= set(seen)
    assert all(est.tracks[tid].observations == seen[tid] for tid in est.tracks)


def test_points_on_the_robots_own_links_are_told_apart_except_the_load_in_the_gripper():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 1.0, "y": 2.0, "yaw": 0.5}
    est.joints = {"lift": 0.8, "reach": 0.4, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    grip = est.gripper_position()
    mast = np.array([1.0 + 0.18 * math.cos(0.5), 2.0 + 0.18 * math.sin(0.5), 1.0])
    ahead = grip + np.array([0.4 * math.cos(0.5), 0.4 * math.sin(0.5), -0.05])
    points = np.stack([mast, grip, ahead])
    assert est.on_body(points).tolist() == [True, True, False]
    est.gripper = {"state": "closed", "holding": True}
    assert est.on_body(points).tolist() == [True, False, False], "the load is not the robot"


def test_at_the_table_edge_the_wide_camera_reads_the_face_the_head_camera_cannot_see(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-kitchen", run_dir=tmp_path, phase="phase1", speed=4.0).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    try:
        plate = host.sim.entity_obb("w:plate-1").center
        host.sim.command("w:robot-1", {"base": [float(plate[0]) - 0.62, float(plate[1]), 0.0]})

        def plate_track():
            return next((t for t in runtime.estimator.tracks.values() if t.category == "plate"
                         and abs(t.pos[0] - plate[0]) < 0.1 and abs(t.pos[1] - plate[1]) < 0.1), None)

        _pump_until(host, lambda: abs(runtime.estimator.odometry["x"] - (plate[0] - 0.62)) < 0.03 and plate_track())
        track_id = plate_track().id
        seen = _off_main(host, lambda: runtime.inspect(track_id, 8))
        assert seen["camera"] == "wide_camera" and seen["coverage"] > 0.8
        head = RuntimeConfig(cameras=("head_camera",))
        only_head = EgoRuntime(runtime.link, steward_bot().to_dict(), head)
        only_head.estimator = runtime.estimator
        narrow = _off_main(host, lambda: only_head.inspect(track_id, 8))
        assert narrow["camera"] == "head_camera" and narrow["coverage"] < seen["coverage"] / 1.5
        assert narrow["hint"].startswith("too close"), "the head camera alone cannot see the strip next to the edge"
    finally:
        runtime.stop()
        host.stop()


def test_depth_on_the_robot_body_is_left_out_of_the_fill():
    runtime = _offline()
    intr = runtime.estimator.camera_intrinsics("wide_camera")
    depth = np.full((intr["height"], intr["width"]), 0.45, np.float32)
    cleaned = runtime._depth_without_body(
        {"depth": depth, "fovy": intr["fovy"], "camera": "wide_camera", "header": _header()})
    assert (cleaned == 0).any(), "the wide camera sees the mast and the arm"
    assert (cleaned == 0.45).any()
    assert runtime.estimator.odometry["x"] == 0.0 and runtime.estimator.joints == {}


def _posed_state() -> dict:
    return {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "joints": {},
            "gripper": {"holding": False}, "entities": {}, "toolOffset": 0.3, "posed": True}


def test_a_look_around_leaves_drawers_alone_unless_asked():
    from steward.ego.skills import Job, SkillExecutor

    called = []
    executor = SkillExecutor(lambda _cmd: None, _posed_state, lambda track: track,
                             closed_drawers=lambda: called.append("list") or ["e:drawer"],
                             drawer_plan=lambda skill, target: called.append(skill) or ({}, ""))
    executor._turn_to = lambda _yaw: None
    executor._wait = lambda _seconds: None
    job = Job(id="job-1", skill="look_around", receipt="look-around")
    executor._look_around(job)
    assert called == []
    assert sum(line.startswith("looked toward") for line in job.log) == 4


def test_a_look_around_opens_a_closed_drawer_looks_inside_and_closes_it():
    """The four turns still happen. A closed drawer is then opened, watched, and closed."""
    from steward.ego.skills import Job, MotionConfig, SkillExecutor

    visited = []
    asked = []

    def plan(skill: str, target: str, distance: float | None = None,
             handle: list[float] | None = None, axis: list[float] | None = None) -> tuple[dict | None, str]:
        asked.append((skill, distance, handle, axis))
        return {"drawer": target, "which": skill, "handle": [1.0, 2.0, 0.7], "axis": [1.0, 0.0]}, ""

    pins = []
    executor = SkillExecutor(lambda _cmd: None, _posed_state, lambda track: track,
                             MotionConfig(inspect_closed_drawers=True),
                             closed_drawers=lambda: ["e:drawer"], drawer_plan=plan,
                             pin=lambda tid, on: pins.append((tid, on)), furniture=lambda _tid: "e:cabinet")
    executor._turn_to = lambda _yaw: None
    executor._wait = lambda _seconds: None
    executor._translate_to = lambda _x, _y, _yaw, speed=None: None

    def slide(_job, drawer_plan):
        visited.append(drawer_plan["which"])
        return 0.29 if drawer_plan["which"] == "open" else -0.29

    executor._drawer = slide
    job = Job(id="job-1", skill="look_around", receipt="look-around")
    executor._look_around(job)
    assert sum(line.startswith("looked toward") for line in job.log) == 4
    assert visited == ["open", "close"]
    assert asked[-1][0] == "close" and asked[-1][1] == pytest.approx(0.29)
    assert asked[-1][2] == pytest.approx([1.29, 2.0, 0.7])
    assert asked[-1][3] == pytest.approx([1.0, 0.0])
    assert "looked inside e:drawer" in job.log
    assert pins == [("e:cabinet", True), ("e:drawer", True), ("e:drawer", False), ("e:cabinet", False)]
    assert "looked along e:drawer" in job.log


def test_an_uncertain_handle_is_approached_before_the_drawer_is_opened():
    """A far drawer has a safe path but too much handle uncertainty. Walk there, then open."""
    from steward.ego.skills import Job, MotionConfig, SkillExecutor

    calls = {"n": 0}
    driven = []

    def plan(skill: str, target: str) -> tuple[dict | None, str]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"path": [[0.0, 0.0], [1.0, 0.0]], "yaw": 0.0, "drawer": target}, "87% miss"
        return {"drawer": target, "which": skill}, ""

    executor = SkillExecutor(lambda _cmd: None, _posed_state, lambda track: track,
                             MotionConfig(inspect_closed_drawers=True),
                             closed_drawers=lambda: ["e:drawer"], drawer_plan=plan)
    executor._turn_to = lambda _yaw: None
    executor._wait = lambda _seconds: None
    executor._translate_to = lambda _x, _y, _yaw, speed=None: None
    executor._drive = lambda _job, path, _yaw: driven.append(path)
    visited = []
    executor._drawer = lambda _job, drawer_plan: visited.append(drawer_plan["which"])
    job = Job(id="job-1", skill="look_around", receipt="look-around")
    executor._look_around(job)
    assert driven == [[[0.0, 0.0], [1.0, 0.0]]]
    assert "approached e:drawer" in job.log
    assert visited == ["open", "close"]


def test_a_drawer_that_cannot_be_opened_is_skipped_and_the_look_around_continues():
    from steward.ego.skills import Job, MotionConfig, SkillExecutor, SkillFailed

    visited = []

    def plan(skill: str, target: str) -> tuple[dict | None, str]:
        if target == "e:far":
            return None, "no path"
        if skill == "open":
            return {"drawer": target, "which": skill}, ""
        return None, "already closed"

    executor = SkillExecutor(lambda _cmd: None, _posed_state, lambda track: track,
                             MotionConfig(inspect_closed_drawers=True),
                             closed_drawers=lambda: ["e:far", "e:near"], drawer_plan=plan)

    def open_or_fail(_job, drawer_plan):
        if drawer_plan["drawer"] == "e:near":
            raise SkillFailed("missed the handle")
        visited.append(drawer_plan["which"])

    executor._turn_to = lambda _yaw: None
    executor._wait = lambda _seconds: None
    executor._drawer = open_or_fail
    job = Job(id="job-1", skill="look_around", receipt="look-around")
    executor._look_around(job)
    assert visited == []
    assert any("left e:far closed" in line for line in job.log)
    assert any("could not open e:near" in line for line in job.log)
    assert job.state == "running"


def test_closed_drawers_are_the_ones_already_seen_and_still_shut():
    from steward.ego.estimator import Track

    runtime = _offline()
    runtime.estimator.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def track(oid, category, pos, observations=6, existence=1.0):
        return Track(id=oid, category_counts={category: 5.0}, pos=np.array(pos, float),
                     var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.5, 0.4, 0.2]),
                     observations=observations, first_seen=0.0, last_seen=0.0, existence=existence)

    near = track("e:near", "drawer", [0.4, 0.1, 0.5])
    far = track("e:far", "drawer", [2.0, 0.0, 0.5])
    young = track("e:young", "drawer", [0.2, 0.0, 0.5], observations=2)
    cup = track("e:cup", "cup", [0.1, 0.1, 0.9])
    runtime.estimator.tracks = {t.id: t for t in (far, young, cup, near)}
    assert runtime._closed_drawer_ids() == ["e:near", "e:far"]
