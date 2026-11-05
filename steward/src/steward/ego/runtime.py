"""Ego Runtime (ER): perception loop, Ego-State store, ESim receipts, skill execution and the
operator view of the Ego-State (rendered on its own thread, see :mod:`steward.viewloop`).

The runtime talks to the world only through the robot port (sensors, actuator commands,
telemetry). It never holds World State identifiers or ground truth.
"""

from __future__ import annotations

import base64
import collections
import io
import contextlib
import itertools
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from steward.ego.perception import (box_of, decode_rgbd, match_face, resample_face, segments, top_face, top_face_array,
                                    world_up_face)
from steward.ego.kinematics import gripper_pose
from steward.ego.providers.geometric_masks import _backproject
from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import SELF_ID, EgoEstimator
from steward.ego.roles import RoleConfig, assign_roles
from steward.ego.segmenter import Segmenter, SegmenterError
from steward.ego.skills import MotionConfig, SkillExecutor, SkillFailed
from steward.ego.view import EgoViewRenderer
from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_rotate, wrap_angle, yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState
from steward.viewloop import ViewLoop


@dataclass
class RuntimeConfig:
    perception_hz: float = 10.0
    telemetry_hz: float = 2.0
    appearance_cell_m: float = 0.005  # cell size of the remembered top faces
    appearance_max_m: float = 0.6  # larger objects' faces are remembered only when asked for
    appearance_range_m: float = 1.5  # faces farther from the camera are not looked at
    appearance_interval_s: float = 0.5  # world time between two automatic looks at one face
    appearance_poll_s: float = 1.0  # wall time between camera images fetched for faces with detections
    appearance_min_coverage: float = 0.25  # a view showing less of a face is not kept
    max_risk: float = 0.35
    # A grasp verifies itself and a miss costs one retreat, so a pick may be tried at a risk that
    # would be reckless for a move through the room or a release over a surface.
    max_pick_risk: float = 0.6
    # A fresh look taken for an imminent pick may restate the target's position when the new
    # reading agrees with the belief this closely; farther readings are different objects.
    relocate_max_jump_m: float = 0.35
    relocate_min_points: int = 30
    relocate_fallback_m: float = 0.6  # beyond the agree-gate: accept the nearest confident reading
    relocate_fallback_score: float = 0.55
    relocate_max_dz_m: float = 0.12  # an instance this far off in height is a different thing
    fresh_look_every_s: float = 10.0  # a retried pick of the same target reuses the last look
    redock_max_m: float = 0.12  # a correction this small is folded into the plan; larger re-plans
    # Robot-side segmentation (frames without instance masks): a frame is segmented when the camera
    # moved or turned this much since the last segmented frame and at least ``keyframe_min_s`` of
    # world time passed, or after ``keyframe_max_s`` in any case. Other perception periods only
    # follow the robot's own state; the world waits while a frame is segmented.
    keyframe_move_m: float = 0.15
    keyframe_turn_rad: float = 0.1
    keyframe_arm_turn_rad: float = 0.3
    keyframe_min_s: float = 0.0
    keyframe_max_s: float = 4.0
    keyframe_still_s: float = 0.4
    """Segment again after the base has been still this long, when the pose has changed at all
    since the last segmented frame. A look-around stops on a heading the fast turn did not sample."""
    # Arm motion also makes the near camera's frame a keyframe, while the base stands still: a
    # prismatic joint moved ``keyframe_arm_m``, a revolute joint turned ``keyframe_arm_turn_rad``, or
    # the gripper opened, closed, took or let go of something.
    keyframe_arm_m: float = 0.05
    hold_renew_s: float = 0.5  # below the world's hold lease (HOLD_LEASE_S)
    # Cameras perception uses (sensor ids of the robot model); None: every camera the model has.
    # A frame from each is fused every perception period, so each camera adds its rendering (and,
    # with robot-side segmentation, its segmentation requests) to the period's cost.
    cameras: tuple[str, ...] | None = None
    # Division of labour between the frames of one instant: the near camera (None: the camera with
    # the widest vertical field of view when perception uses several) is in charge of regions
    # nearer than ``near_range_m``, the other cameras of the farther ones. A region is fused from a
    # camera not in charge of its range only when no camera in charge has it in view (inside its
    # image by ``handoff_margin`` of the image size), so each object is one observation per instant.
    near_camera: str | None = None
    near_range_m: float = 1.5  # from the head at 1.5 m: a table top up to ~1.3 m ahead
    handoff_margin: float = 0.05
    esim: EsimConfig = field(default_factory=EsimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)


def body_ok(category: str) -> bool:
    """A text prompt needs a real name; unknown things have nothing to ask for."""
    return bool(category) and category != "unknown"


# A category is one word of the robot's vocabulary; what a person would call the thing may be
# another. The class name alone misses a mug asked for as a cup, so a second name is tried.
PICK_PROMPTS: dict[str, tuple[str, ...]] = {
    "cup": ("cup", "mug"),
    "mug": ("mug", "cup"),
    "wine_glass": ("wine glass", "glass"),
    "bowl": ("bowl", "dish"),
}


def _fresh_look_prompts(category: str, color: str | None) -> tuple[str, ...]:
    """Text prompts to try for one track, best first.

    SAM3's text head needs more than a category to find tableware from above: at 0.95 m "red
    bowl" scores 0.89 and "bowl" nothing at all on the same frame, and the same holds for "black
    mug" against "mug". The colour is the robot's own reading of the object, so putting it in
    front of the name is not being told the answer — it is using the description it already has.
    """
    name = category.replace("_", " ")
    out = []
    if color and color not in ("clear", "unknown"):
        out.append(f"{color} {name}")
    out.append(name)
    out.extend(p for p in PICK_PROMPTS.get(category, ()) if p != name)
    return tuple(dict.fromkeys(out))[:3]  # three prompts at most: each one is a segmenter round trip


class RobotLink:
    """HTTP client of one robot port: the only path between the Ego System and the world."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.closed = False  # set on shutdown: clock waits stop retrying

    def sensors(self) -> dict:
        return call("GET", f"{self.url}/sensors")

    def proprioception(self) -> dict:
        """Joint encoders and gripper states only, without drawing a perception sample."""
        return call("GET", f"{self.url}/sensors/proprioception")

    def rgbd(self, camera: str | None = None) -> bytes:
        """One RGB-D frame (``npz``) of ``camera`` (None: the primary one) when the world provides
        raw imaging instead of detections."""
        return call("GET", f"{self.url}/sensors/rgbd" + ("" if camera is None else f"?camera={camera}"))

    def command(self, cmd: dict) -> None:
        call("POST", f"{self.url}/command", cmd)

    def telemetry(self, scene: dict) -> None:
        call("POST", f"{self.url}/telemetry", scene)

    def camera_png(self, width: int = 320, height: int = 240, camera: str | None = None) -> bytes:
        return call("GET", f"{self.url}/camera/rgb?width={width}&height={height}" + ("" if camera is None else f"&camera={camera}"))

    def cameras(self) -> dict:
        """Synchronized three-camera RGB with relative rig calibration."""
        return call("GET", f"{self.url}/sensors/cameras")

    def kinematics(self) -> dict:
        """Joint encoders and relative camera calibration from one sensor timestamp."""
        return call("GET", f"{self.url}/sensors/kinematics")

    def camera_calibration(self) -> dict:
        """Fixed RGB intrinsics and camera-to-middle-RGB extrinsics, without image capture."""
        return call("GET", f"{self.url}/sensors/cameras/calibration")

    def rgb_recording_start(self, body: dict) -> dict:
        return call("POST", f"{self.url}/sensors/rgb/record/start", body)

    def rgb_recording_status(self, id: str) -> dict:
        from urllib.parse import urlencode
        return call("GET", f"{self.url}/sensors/rgb/record/status?{urlencode({'recording_id': id})}")

    def rgb_recording_stop(self, id: str, cancel: bool) -> dict:
        return call("POST", f"{self.url}/sensors/rgb/record/stop", {"recording_id": id, "cancel": cancel}, timeout=25)

    def rgb_recording_result(self, id: str) -> dict:
        from urllib.parse import urlencode
        return call("GET", f"{self.url}/sensors/rgb/record/result?{urlencode({'recording_id': id})}")

    def rgb_pose(self) -> dict:
        return call("GET", f"{self.url}/sensors/rgb/pose")

    def wait_until(self, t: float | None = None, hold: str | None = None, rank: int = 0, pulls: bool = True) -> float:
        """Wait until the world clock reaches ``t`` (``None``: now) and, with ``hold``, the world lets
        this control loop or input run (:meth:`WorldSim.wait_until`); return the world time then."""
        body = {"hold": hold, "rank": rank, "pulls": pulls, **({} if t is None else {"until": t})}
        while True:
            try:
                return call("POST", f"{self.url}/clock/wait", body)["t"]
            except HttpError as exc:
                if exc.status != 504 or self.closed:
                    raise

    def arm(self, hold: str, t: float | None = None, rank: int = 0, pulls: bool = True) -> None:
        """Register ``hold`` at world time ``t`` (``None``: now) without waiting, so the world stops for
        it before its owner first waits."""
        try:
            call("POST", f"{self.url}/clock/wait", {"hold": hold, "rank": rank, "pulls": pulls, "timeoutS": 0.0,
                                                     **({} if t is None else {"until": t})})
        except HttpError as exc:
            if exc.status != 504:
                raise

    def sleep(self, seconds: float) -> float:
        """Let ``seconds`` of world time pass and return the world time then."""
        return self.wait_until(self.wait_until() + seconds)

    def release(self, hold: str) -> None:
        call("POST", f"{self.url}/clock/release", {"hold": hold})

    def renew(self, hold: str) -> None:
        call("POST", f"{self.url}/clock/renew", {"hold": hold})


ALERT_TEXT_MAX = 1500
QUERY_MIN_PROBABILITY = 0.2
MATCH_REFERENCE_CELLS_MAX = 96  # cells along a reference side for match_face
MATCH_MIN_COVERAGE = 0.5  # share of a face that must have been seen to match it
ACTIVITY_MAX = 300
PLAN_STEP_STATES = frozenset({"pending", "active", "done", "failed", "skipped"})
VIEW_HINT_COVERAGE = 0.95  # a face seen less than this gets a hint saying why


def view_hint(cam: dict, intrinsics: dict, centre: np.ndarray, top: float, range_m: float) -> str:
    """Why the camera at ``cam`` (looking along its +x) does not see all of a top face at
    ``centre`` whose top is at height ``top``: behind or beside the view, below it (too close for the
    camera's downward reach at that height), farther than ``range_m``, or hidden or cut off."""
    x, y, z = pose_compose(pose_inverse(cam), {"p": [float(v) for v in centre], "q": [1.0, 0.0, 0.0, 0.0]})["p"]
    half_v = math.radians(intrinsics["fovy"]) / 2
    half_h = math.atan(math.tan(half_v) * intrinsics["width"] / intrinsics["height"])
    ahead = math.hypot(float(centre[0]) - cam["p"][0], float(centre[1]) - cam["p"][1])
    if x <= 0.05:
        return "it is behind the camera: turn towards it"
    side = math.atan2(y, x)
    if abs(side) > half_h:
        return (f"it lies {math.degrees(abs(side)):.0f}° to the {'left' if side > 0 else 'right'} of the camera's "
                f"{math.degrees(half_h):.0f}° half view: turn towards it")
    if math.atan2(z, x) < -half_v:
        forward = quat_rotate(cam["q"], [1.0, 0.0, 0.0])
        lowest = math.asin(max(-1.0, min(1.0, -forward[2]))) + half_v
        nearest = (cam["p"][2] - top) / math.tan(lowest) if lowest < math.pi / 2 else 0.0
        return (f"too close: {ahead:.2f} m ahead of the camera it lies below the view, which reaches a surface at "
                f"this height only from {nearest:.2f} m ahead: step back")
    if math.atan2(z, x) > half_v:
        return "it lies above the camera's view"
    distance = math.dist(cam["p"], [float(v) for v in centre])
    if distance > range_m:
        return f"{distance:.2f} m from the camera, farther than {range_m:.2f} m: move closer"
    return "in view but hidden by something nearer or cut off at the image edge: look from another side"


def in_view(cam: dict, intrinsics: dict, point: np.ndarray, margin: float) -> bool:
    """Whether ``point`` projects into the image of the pinhole camera at ``cam`` (looking along
    its +x), at least ``margin`` of the image width and height inside its edges."""
    x, y, z = pose_compose(pose_inverse(cam), {"p": [float(v) for v in point], "q": [1.0, 0.0, 0.0, 0.0]})["p"]
    if x <= 0.05:
        return False
    w, h = intrinsics["width"], intrinsics["height"]
    f = (h / 2.0) / math.tan(math.radians(intrinsics["fovy"]) / 2.0)
    u, v = w / 2.0 - y * f / x, h / 2.0 - z * f / x
    return margin * w <= u <= (1.0 - margin) * w and margin * h <= v <= (1.0 - margin) * h


class EgoRuntime:
    def __init__(self, link: RobotLink, robot_model: dict, config: RuntimeConfig | None = None,
                 out_dir: Path | None = None, segmenter: Segmenter | None = None):
        self.link = link
        self.config = config or RuntimeConfig()
        self.estimator = EgoEstimator(robot_model=robot_model)
        unknown = [c for c in self.config.cameras or () if c not in self.estimator.cameras]
        if unknown:
            raise ValueError(f"RuntimeConfig.cameras {unknown} are not cameras of the robot model "
                             f"({', '.join(self.estimator.cameras)})")
        self.near_camera = self._resolve_near_camera()
        self._arm_joints = {j["id"]: j["type"]["value"] for j in robot_model["body"]["joints"]
                            if j["type"]["value"] in ("prismatic", "revolute")}
        self.segmenter = segmenter
        if segmenter is not None:
            self.estimator.config.semantic_shape_weight = segmenter.config.shape_weight
            self.estimator.config.semantic_label_floor = segmenter.config.label_floor
        self._keyframe: dict[str, dict] = {}  # camera → sensor header of its last segmented frame
        self._pose_sample: tuple[float, float, float] | None = None
        self._still_since: float | None = None
        self._missing_cameras: frozenset[str] = frozenset()
        self.segmentation = {"frames": 0, "cached": 0, "failures": 0, "lastSeconds": None, "lastError": None}
        self.esim = Esim(robot_model, self.config.esim)
        self.out_dir = out_dir
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.receipts: dict[str, dict] = {}
        self.last_error: str | None = None
        self.control_owner: str | None = None
        self.perception_mode: str | None = None
        self.alerts: list[dict] = []
        self.plan: dict | None = None
        self._looked_at: dict[str, float] = {}  # target id -> body stamp of its last fresh look
        self._roles: dict[str, str] = {}  # the latest role assignment, kept unless clearly outweighed
        self.activity: collections.deque[dict] = collections.deque(maxlen=ACTIVITY_MAX)
        self._activity_ids = itertools.count(1)
        self._input_lock = threading.Lock()
        self.inputs: list[dict] = []  # Ego API requests as applied, with their world time (:meth:`apply`)
        self._alert_ids = itertools.count(1)
        self.executor = SkillExecutor(self.link.command, self._motion_state, self._mark_held, self.config.motion,
                                      shift=self._shift_track, on_error=lambda msg: self.alert(msg, "skill"),
                                      on_finish=self._skill_finished, on_release=self._released,
                                      on_begin=self._skill_begin, clock=self.link)
        self._view_scene: tuple[dict, int, float] | None = None  # (ES dict, version, stamp) for the view
        self.face_error: str | None = None  # why the latest camera image for faces could not be fetched
        self._face_thread: threading.Thread | None = None
        renderer = EgoViewRenderer()
        self.view = ViewLoop("ego", lambda settings: self._render_view(renderer, settings), renderer.close)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Start perception in a worker thread. The first frame is taken there too: RGB-D frames
        are rendered on the world's main thread, which may be the caller."""
        self._running = True
        self.link.arm("perception", pulls=False)
        self._thread = threading.Thread(target=self._loop, name="ego-perception", daemon=True)
        self._thread.start()
        self._face_thread = threading.Thread(target=self._face_loop, name="ego-faces", daemon=True)
        self._face_thread.start()
        self.view.start()

    def stop(self) -> None:
        self._running = False
        self.link.closed = True
        self.view.stop()
        for thread in (self._thread, self._face_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def _poll_once(self) -> None:
        frame = self.link.sensors()
        if frame.get("imaging") != "rgbd":
            mode = "detections"
        else:
            mode = "rgbd" if frame.get("masks", True) else "rgbd-segmenter"
        # A fast look-around steps farther than the heading window in which a long table is fully
        # inside the head camera, and a border-truncated region does not start a track. Slow the
        # turn only while perception keeps keyframes; fusing every frame already samples that window.
        self.executor._keyframed = mode == "rgbd-segmenter"
        if mode != self.perception_mode:
            with self._lock:
                if self.perception_mode is not None:
                    # The sensors changed kind (a phase switch): objects are perceived afresh with
                    # the new pipeline; what the gripper holds and the running skill are kept.
                    self.estimator.forget_objects()
                self.perception_mode = mode
                self._keyframe = {}
                self._pose_sample = None
                self._still_since = None
        cameras = self._cameras(frame)
        if mode in ("rgbd", "rgbd-segmenter"):
            due = [c for c in cameras if mode == "rgbd" or self._keyframe_due(frame, c)]
            frames = []
            for camera in due:
                rgbd = decode_rgbd(self.link.rgbd(camera))
                if rgbd["mask"] is not None or self._segment(rgbd):
                    frames.append(rgbd)
            with self._lock:
                if frame["stamp"] < self.estimator.stamp - 1e-6:
                    self._restart()
                if not frames:
                    self.estimator.update_body(frame)
                for rgbd in frames:
                    self.estimator.update_rgbd(rgbd, defer=self._handoff(rgbd, frames))
                    self._look_at_faces(rgbd)
                self._publish_view_scene()
            return
        frame = {**frame, "cameras": [c for c in frame.get("cameras", []) if c["id"] in cameras],
                 "detections": [d for d in frame.get("detections", []) if d.get("camera") in cameras]}
        with self._lock:
            if frame["stamp"] < self.estimator.stamp - 1e-6:
                self._restart()
            self.estimator.update(frame)
            self._publish_view_scene()

    def _cameras(self, frame: dict) -> list[str]:
        """The cameras to perceive with: those of the robot model (or ``RuntimeConfig.cameras``) that
        the world's robot provides, primary first. A camera the world lacks (its scene was built
        with an older robot model) is reported once as an operator alert."""
        wanted = [c for c in self.estimator.cameras if self.config.cameras is None or c in self.config.cameras]
        provided = {c["id"] for c in frame.get("cameras", [])}
        missing = frozenset(c for c in wanted if c not in provided)
        if missing and missing != self._missing_cameras:
            self.alert(f"世界中的机器人没有相机 {', '.join(sorted(missing))}：场景可能按旧的机器人模型构建，"
                       "运行 steward build-kitchen 重建", "runtime")
        self._missing_cameras = missing
        return [c for c in wanted if c in provided]

    def _resolve_near_camera(self) -> str | None:
        """The camera in charge of near regions (:attr:`RuntimeConfig.near_camera`); None when
        perception uses a single camera. Raises ``ValueError`` for a camera perception does not use."""
        used = [c for c in self.estimator.cameras if self.config.cameras is None or c in self.config.cameras]
        if self.config.near_camera is not None:
            if self.config.near_camera not in used:
                raise ValueError(f"RuntimeConfig.near_camera {self.config.near_camera!r} is not a camera perception "
                                 f"uses ({', '.join(used)})")
            return self.config.near_camera
        if len(used) < 2:
            return None
        return max(used, key=lambda c: self.estimator.camera_intrinsics(c)["fovy"])

    def _note_pose(self, frame: dict) -> None:
        """Remember whether the base has stayed still since the previous polled pose."""
        odo = frame["odometry"]
        pose = (float(odo["x"]), float(odo["y"]), float(odo["yaw"]))
        prev, stamp = self._pose_sample, float(frame["stamp"])
        self._pose_sample = pose
        if prev is None or stamp < (self._still_since or stamp):
            self._still_since = stamp
            return
        shifted = math.hypot(pose[0] - prev[0], pose[1] - prev[1]) >= 0.01 \
            or abs(wrap_angle(pose[2] - prev[2])) >= 0.02
        if shifted:
            self._still_since = stamp

    def _keyframe_due(self, frame: dict, camera: str) -> bool:
        cfg, last = self.config, self._keyframe.get(camera)
        self._note_pose(frame)
        if last is None:
            return True
        odo, was = frame["odometry"], last["odometry"]
        elapsed = float(frame["stamp"]) - float(last["stamp"])
        if elapsed < 0 or elapsed >= cfg.keyframe_max_s:
            return True
        moved = math.hypot(odo["x"] - was["x"], odo["y"] - was["y"]) >= cfg.keyframe_move_m
        turned = abs(wrap_angle(odo["yaw"] - was["yaw"])) >= cfg.keyframe_turn_rad
        handled = camera == self.near_camera and self._arm_moved(last, frame)
        changed = math.hypot(odo["x"] - was["x"], odo["y"] - was["y"]) >= 0.02 \
            or abs(wrap_angle(odo["yaw"] - was["yaw"])) >= 0.02
        settled = self._still_since is not None and float(frame["stamp"]) - self._still_since >= cfg.keyframe_still_s \
            and changed
        return elapsed >= cfg.keyframe_min_s and (moved or turned or handled or settled)

    def _arm_moved(self, last: dict, frame: dict) -> bool:
        """Whether the arm or gripper changed enough since ``last`` for a new near-camera keyframe."""
        grip, was = frame["gripper"], last["gripper"]
        if grip.get("state") != was.get("state") or bool(grip.get("holding")) != bool(was.get("holding")):
            return True
        for jid, kind in self._arm_joints.items():
            if jid not in frame["joints"] or jid not in last["joints"]:
                continue
            dq = frame["joints"][jid] - last["joints"][jid]
            if (abs(dq) >= self.config.keyframe_arm_m if kind == "prismatic"
                    else abs(wrap_angle(dq)) >= self.config.keyframe_arm_turn_rad):
                return True
        return False

    def _handoff(self, rgbd: dict, batch: list[dict]):
        """The regions of ``rgbd`` that a camera in charge of their range fuses instead
        (:attr:`RuntimeConfig.near_camera`): a predicate on the corners of a region's box, true
        when a camera in charge has the whole box in view; or None to fuse all. A camera in charge
        that sees only part of the box (a long table up close) leaves the region to this one."""
        others = [f for f in batch if f is not rgbd]
        if self.near_camera is None or not others:
            return None
        cfg, est = self.config, self.estimator
        near = rgbd["camera"] == self.near_camera
        cam_p = np.asarray(est.camera_pose(rgbd["header"]["odometry"], rgbd["camera"])["p"], float)
        views = [(f["camera"] == self.near_camera, est.camera_pose(f["header"]["odometry"], f["camera"]),
                  est.camera_intrinsics(f["camera"])) for f in others]

        def defer(corners: np.ndarray) -> bool:
            near_region = float(np.linalg.norm(corners.mean(axis=0) - cam_p)) < cfg.near_range_m
            if near_region == near:
                return False
            return any(all(in_view(pose, intrinsics, c, cfg.handoff_margin) for c in corners)
                       for is_near, pose, intrinsics in views if is_near == near_region)
        return defer

    def _segment(self, rgbd: dict) -> bool:
        """Fill ``rgbd["mask"]`` and ``rgbd["labels"]`` from the robot's segmenter; False (and an
        operator alert on the first failure in a row) when the frame could not be segmented."""
        header = rgbd["header"]
        self._keyframe[rgbd["camera"]] = header
        stats = self.segmentation
        if self.segmenter is None:
            error = "this phase level sends no instance masks and the robot has no segmenter (steward ego --segmentation-config)"
        else:
            try:
                with self._holding("perception"):
                    result = self.segmenter.segment(rgbd["rgb"])
                with self._lock:
                    cam = self.estimator.camera_pose(header["odometry"], rgbd["camera"])
                up_cam = matrix_from_quat(cam["q"]).T @ np.array([0.0, 0.0, 1.0])
                result = self.segmenter.fill(result, self._depth_without_body(rgbd), rgbd["fovy"], up_cam)
            except SegmenterError as exc:
                error = str(exc)
            else:
                rgbd["mask"], rgbd["labels"] = result.mask, result.labels
                stats["frames"] += 1
                stats["cached"] += int(result.cached)
                if not result.cached:
                    stats["lastSeconds"] = round(result.seconds, 2)
                stats["lastError"] = None
                return True
        stats["failures"] += 1
        if stats["lastError"] is None:
            self.alert(f"图像分割失败，本帧只更新机器人自身状态：{error}", "runtime")
        stats["lastError"] = error
        return False

    def _depth_without_body(self, rgbd: dict) -> np.ndarray:
        """``rgbd`` depth with pixels on the robot's own links set to 0.

        Depth clustering, unlike a world mask, does not know the body. The copy is only for the
        geometric fill; the frame's depth still measures the scene."""
        depth = np.array(rgbd["depth"], dtype=np.float32, copy=True)
        header = rgbd["header"]
        est = self.estimator
        saved = (est.odometry, est.joints, est.gripper)
        est.odometry = header["odometry"]
        est.joints = header["joints"]
        est.gripper = header["gripper"]
        try:
            valid = np.isfinite(depth) & (depth > 0.05)
            if not valid.any():
                return depth
            pts, ys, xs = _backproject(depth, float(rgbd["fovy"]), valid)
            cam = est.camera_pose(header["odometry"], rgbd.get("camera"))
            rot = matrix_from_quat(cam["q"])
            world = pts @ rot.T + np.asarray(cam["p"], float)
            inside = est.on_body(world)
        finally:
            est.odometry, est.joints, est.gripper = saved
        depth[ys[inside], xs[inside]] = 0.0
        return depth

    @contextlib.contextmanager
    def _holding(self, hold: str):
        """Renew ``hold`` every ``hold_renew_s`` wall seconds while the block runs, so the world
        stays stopped for a woken loop that works longer than its hold lease."""
        done = threading.Event()

        def renew() -> None:
            while not done.wait(self.config.hold_renew_s):
                try:
                    self.link.renew(hold)
                except (HttpError, OSError):
                    pass  # the world is gone or restarting; the next wait reports it

        thread = threading.Thread(target=renew, name=f"renew-{hold}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    def _face_loop(self) -> None:
        """With detections, which carry no colours, fetch a camera image every ``appearance_poll_s``
        wall seconds while an object whose face is remembered lies within range, and remember the
        faces it shows. It holds no world clock: perception and control never wait for it."""
        while self._running:
            time.sleep(self.config.appearance_poll_s)
            if self.perception_mode != "detections":
                continue
            with self._lock:
                cam_p = np.asarray(self.estimator.camera_pose()["p"], float)
                due = any(not t.held and t.footprint <= self.config.appearance_max_m
                          and float(np.linalg.norm(t.pos - cam_p)) <= self.config.appearance_range_m
                          for t in self.estimator.tracks.values())
            if not due:
                continue
            try:
                frames = self._camera_frames()
            except (HttpError, OSError) as exc:
                self.face_error = f"camera image for faces unavailable: {exc}"
                continue
            with self._lock:
                for rgbd in frames:
                    self._look_at_faces(rgbd)
            self.face_error = None

    def _camera_frames(self) -> list[dict]:
        """One fresh decoded RGB-D frame from each camera perception uses, primary first."""
        return [decode_rgbd(self.link.rgbd(c)) for c in self._cameras(self.link.sensors())]

    def _telemetry_scene(self) -> dict:
        """The Ego-State sent as telemetry, each object carrying its scene-level role as the state
        variable ``role`` (``src: inferred``) next to its perceived category."""
        roles = {o["id"]: (o["role"], o["roleConfidence"]) for o in self.summary()["objects"]}
        scene = self.ego_state().to_dict()
        for ent in scene["entities"]:
            if ent["id"] in roles:
                role, p = roles[ent["id"]]
                ent.setdefault("stateVars", {})["role"] = {"value": role, "unc": {"kind": "categorical", "p": {role: p}},
                                                           "src": "inferred"}
        return scene

    def _publish_view_scene(self) -> None:
        """Hand the view a copy of the Ego-State taken right after a perception update (the caller
        holds the lock), so rendering never waits for the next update to finish."""
        self._view_scene = (self.estimator.to_scene().to_dict(), self.estimator.version, self.estimator.stamp)

    def _render_view(self, renderer: EgoViewRenderer, settings) -> tuple[bytes, str, dict]:
        scene = self._view_scene
        if scene is None:
            with self._lock:
                self._publish_view_scene()
            scene = self._view_scene
        ego, version, stamp = scene
        return renderer.render(ego, settings), "image/jpeg", {"esVersion": version, "t": round(stamp, 3)}

    def _restart(self) -> None:
        """The world clock went backwards: a new episode started, which the robot experiences as a
        reboot. Forget the Ego-State, receipts and any running skill."""
        self.executor.cancel()
        self.estimator = EgoEstimator(robot_model=self.estimator.robot_model, config=self.estimator.config)
        self.receipts.clear()

    def _loop(self) -> None:
        """Perceive every ``1 / perception_hz`` seconds of world time, from the time the runtime started;
        the world waits for each update but does not step for it alone (``freeze_idle``)."""
        period = 1.0 / self.config.perception_hz
        telemetry_every = max(1, int(self.config.perception_hz / self.config.telemetry_hz))
        k = 0
        t: float | None = None
        while self._running:
            try:
                t = self.link.wait_until(None if t is None else t + period, "perception", pulls=False)
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
                time.sleep(period)
                continue
            try:
                self._poll_once()
                k += 1
                if k % telemetry_every == 0:
                    self.link.telemetry(self._telemetry_scene())
                self.last_error = None
            except (HttpError, OSError) as exc:
                self.last_error = f"robot port unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - perception must keep running; the operator is alerted
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.alert(f"感知循环出现异常：{type(exc).__name__}: {exc}", "runtime")
        try:
            self.link.release("perception")
        except (HttpError, OSError):
            pass  # the world is gone or restarting; its hold lease expires on its own

    def apply(self, request: dict, fn, at: float | None = None, then: float | None = None):
        """Apply one Ego API request (``method``, ``path``, ``query``, ``body``) at a world instant, after
        the control loops due then, and append it with that time to :attr:`inputs`. A replay
        (:func:`steward.ego.server.replay`) passes the recorded time ``at`` and the next one, ``then``,
        at which the world stops for the next request."""
        with self._input_lock:
            try:
                t = self.link.wait_until(at, "input", rank=1)
            except (HttpError, OSError):
                return fn()  # no world clock (the robot port is unreachable): applied now, not recorded
            try:
                return fn()
            finally:
                self.inputs.append({"t": t, **request})
                if then is None:
                    self.link.release("input")
                else:
                    self.link.arm("input", then, rank=1)

    # --------------------------------------------------------------- state
    def ego_state(self) -> SceneState:
        with self._lock:
            return self.estimator.to_scene()

    def _motion_state(self) -> dict:
        """What the body is doing now, for the motion loop.

        Odometry, joints and the gripper come from the robot's live sensor header, not from the
        belief: with a real segmenter the next RGB-D frame can be seconds or tens of seconds
        away, and a motion loop that grades its progress on the perception clock either stalls
        or reads a frame rendered before it started and calls the turn failed. Objects stay the
        estimate they are — the body only needs to know where it is."""
        with self._lock:
            est = self.estimator
            entities = {t.id: {"p": t.pos.tolist()} for t in est.tracks.values()}
            state = {"stamp": est.stamp, "odometry": dict(est.odometry), "joints": dict(est.joints),
                     "gripper": dict(est.gripper), "entities": entities, "toolOffset": self.esim.tool_offset,
                     "posed": est.posed}
        try:
            live = self.link.sensors()
        except Exception:  # noqa: BLE001 - the belief is still a usable answer when the body cannot be reached
            return state
        for key in ("stamp", "odometry", "joints", "gripper"):
            if live.get(key) is not None:
                state[key] = live[key]
        if live.get("stamp") is not None and live["stamp"] < state["stamp"] - 1e-6:
            state["stamp"] = est.stamp  # an episode reset race: prefer the belief's own clock
        return state

    def _mark_held(self, track_id: str) -> str:
        at = self.estimator.stamp
        try:
            at = float(self.link.sensors().get("stamp", at))  # the body's clock, now
        except Exception:  # noqa: BLE001 - the belief's own clock is a usable second best
            pass
        with self._lock:
            return self.estimator.mark_held(track_id, at)

    def _released(self, track_id: str, spot: list[float]) -> None:
        """A place knows where it let go; the belief is told, not left to infer it later."""
        with self._lock:
            self.estimator.release_at(track_id, spot)

    def _shift_track(self, track_id: str, delta: list[float]) -> None:
        with self._lock:
            self.estimator.shift(track_id, delta)

    # ------------------------------------------------------------- Ego API
    def summary(self) -> dict:
        es = self.ego_state()
        me = es.entity(SELF_ID)
        p = me.pose.value
        held = next((r.a["entity"] for r in es.relations if r.type == "attached"), None)
        supports = {r.a["entity"]: r.b["entity"] for r in es.relations if r.type == "supported-by"}
        objects = []
        for ent in es.entities:
            if ent.id == SELF_ID:
                continue
            cov = ent.pose.unc["cov"]
            size = ent.rigid.links[0].geometry["collision"][0]["size"]
            objects.append({
                "id": ent.id,
                "category": ent.category.value,
                "categoryConfidence": round(ent.category.unc["p"].get(ent.category.value, 1.0), 3),
                "categories": ent.category.unc["p"],
                "position": [round(v, 3) for v in ent.pose.value["p"]],
                "yawDeg": round(math.degrees(yaw_from_quat(ent.pose.value["q"])), 1),
                "uncertaintyM": round(math.sqrt((cov[0] + cov[7]) / 2), 4),
                "sizeM": [round(v, 3) for v in size],
                "on": supports.get(ent.id),
                "existence": ent.existence,
                "lastSeenAgoS": round(es.stamp - (ent.pose.t or es.stamp), 2),
                "affordances": [a.value for a in ent.rigid.links[0].affordances],
                "color": ent.state_vars["color"].value if "color" in ent.state_vars else None,
                "extentM": self._extent(ent.id),
                **({"openingM": round(self._opening(ent.id), 3)} if ent.category.value == "drawer" else {}),
                **self._front(ent),
                **self._appearance(ent.id),
            })
        with self._lock:
            roles = assign_roles(objects, self.config.roles, self._roles)
            self._roles = {oid: role for oid, (role, _) in roles.items()}
        for obj in objects:
            obj["role"], confidence = roles[obj["id"]]
            obj["roleConfidence"] = round(confidence, 3)
        job = self.executor.current.to_dict() if self.executor.current else None
        dropped = self.dropped()[-20:]
        return {
            "version": es.version,
            "t": round(es.stamp, 3),
            "self": {"x": round(p["p"][0], 3), "y": round(p["p"][1], 3),
                     "yawDeg": round(math.degrees(yaw_from_quat(p["q"])), 1),
                     "holding": held, "joints": {j.id: round(j.q.value[0], 3) for j in me.rigid.joints if j.q.value}},
            "objects": objects,
            "dropped": dropped,
            "currentSkill": job,
            "perception": self.perception_mode,
            **({"segmentation": dict(self.segmentation)} if self.perception_mode == "rgbd-segmenter" else {}),
            "runtimeError": self.last_error,
            "faceCaptureError": self.face_error,
        }

    def dropped(self) -> list[dict]:
        """Every recent removal from the Ego-State, oldest first, with its reason."""
        with self._lock:
            return list(self.estimator.dropped)

    def _extent(self, track_id: str) -> list[float] | None:
        """Estimated physical size of a track, without the uncertainty margin added to ``sizeM``."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return [round(float(v), 3) for v in track.size] if track is not None else None

    def _front(self, ent) -> dict:
        """Heading of the face looking into the room, for furniture and drawers."""
        affordances = [a.value for a in ent.rigid.links[0].affordances]
        if "support-surface" not in affordances and "pullable" not in affordances:
            return {}
        with self._lock:
            track = self.estimator.tracks.get(ent.id)
            if track is None:
                return {}
            box = self.estimator.front(track)
        return {"frontDeg": round(math.degrees(box.yaw), 1), "frontHalfM": round(box.hx, 3)}

    def _opening(self, track_id: str) -> float:
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return self.estimator.drawer_opening(track) if track is not None else 0.0

    def query(self, category: str | None = None, near: str | None = None, radius: float | None = None) -> list[dict]:
        """Objects matching a category by role or, failing that, by probability (best match first),
        optionally within ``radius`` of another object."""
        objs = self.summary()["objects"]
        if category:
            score = lambda o: 2.0 if o["role"] == category else o["categories"].get(category, 0.0)  # noqa: E731
            objs = sorted((o for o in objs if score(o) >= QUERY_MIN_PROBABILITY), key=score, reverse=True)
        if near:
            anchor = next((o for o in self.summary()["objects"] if o["id"] == near), None)
            if anchor is None:
                raise ValueError(f"unknown entity {near!r}")
            r = radius if radius is not None else 1.0
            objs = [o for o in objs if o["id"] != near and math.dist(o["position"][:2], anchor["position"][:2]) <= r]
        return objs

    def capture_cameras(self) -> dict:
        """Capture the existing robot RGB views without moving or changing the perception loop.

        The packet contains sensor images, intrinsics, fixed camera-to-reference calibration
        and gripper kinematics computed from the same encoder sample. It does not provide an
        absolute camera pose.
        """
        packet = self.link.cameras()
        pose = gripper_pose(self.estimator.robot_model, packet)
        return {key: value for key, value in packet.items() if key != "joint_encoders"} | {"gripper_pose": pose}

    def gripper_pose(self) -> dict:
        """Current gripper-link pose relative to the base and the robot cameras, without imaging."""
        return gripper_pose(self.estimator.robot_model, self.link.kinematics())

    def camera_calibration(self) -> dict:
        """Read fixed camera calibration through the robot sensor port, with no absolute poses."""
        return self.link.camera_calibration()

    def prepare_novelanything(self, id: str, source_count: int | None = None) -> dict:
        """Export measured evidence and a current gripper target; no motion or inference is performed."""
        from steward.ego.novelanything_scene import prepare_gripper_scene
        return prepare_gripper_scene(self.estimator.robot_model, self.link.rgb_recording_result(id),
                                     self.link.rgb_pose(), source_count)

    def recording_evidence(self, id: str) -> dict:
        """Export all measured RGB evidence and a synchronized current gripper target frame."""
        from steward.ego.recording_evidence import recording_evidence
        return recording_evidence(self.estimator.robot_model, self.link.rgb_recording_result(id), self.link.rgb_pose())

    def inspect(self, entity: str, px: int = 24) -> dict:
        """The top face of an Ego-State object as the head camera sees it now, resampled to a
        ``px`` × ``px`` grid in the object's own frame: column = local +x to the right, row 0 = local
        +y (the far edge seen from above). Each cell is ``[r, g, b]`` or ``None`` where the face is
        out of view or hidden by something nearer. ``worldRows`` is the same grid turned
        ``worldTurnDeg`` counter-clockwise (:func:`world_up_face`) so that row 0 faces the map's +y.
        Reads one fresh RGB-D frame; no World State.

        :param entity: Ego-State object id.
        :param px: grid size, 4 to 64.
        :returns: ``{entity, px, yawDeg, sizeM, coverage, rows, worldRows, worldTurnDeg, stamp}``, and
            ``hint`` (:func:`view_hint`) when less than :data:`VIEW_HINT_COVERAGE` of the face is seen.
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range."""
        if not 4 <= px <= 64:
            raise ValueError("px must be between 4 and 64")
        with self._lock:
            track = self.estimator.tracks.get(entity)
            if track is None:
                raise KeyError(f"unknown object {entity!r}")
            centre, size, yaw, top = track.pos.copy(), track.size.copy(), float(track.yaw), track.top()
        frames = self._camera_frames()
        best = None
        with self._lock:
            for frame in frames:
                self._look_at_faces(frame, [entity])
        for frame in frames:
            cam = self.estimator.camera_pose(frame["header"]["odometry"], frame["camera"])
            rows = top_face(frame, cam, centre, size, yaw, top, px)
            coverage = sum(cell is not None for row in rows for cell in row) / (px * px)
            if best is None or coverage > best[0]:
                best = (coverage, rows, frame)
        coverage, rows, frame = best
        world_rows, turn = world_up_face(rows, yaw)
        hint = {} if coverage >= VIEW_HINT_COVERAGE else {"hint": self._view_hint(frame["header"]["odometry"], centre, top)}
        return {"entity": entity, "px": px, "camera": frame["camera"], "yawDeg": round(math.degrees(yaw), 2),
                "sizeM": [round(float(v), 3) for v in size], "coverage": round(coverage, 3), **hint,
                "rows": rows, "worldRows": world_rows, "worldTurnDeg": turn, "stamp": frame["header"]["stamp"]}

    def _view_hint(self, odometry: dict, centre: np.ndarray, top: float) -> str:
        """:func:`view_hint` for every camera perception uses at ``odometry``, one clause each."""
        est = self.estimator
        cameras = [c for c in est.cameras if c not in self._missing_cameras
                   and (self.config.cameras is None or c in self.config.cameras)]
        hints = [(c, view_hint(est.camera_pose(odometry, c), est.camera_intrinsics(c), centre, top,
                               self.config.appearance_range_m)) for c in cameras]
        return hints[0][1] if len(hints) == 1 else "; ".join(f"{c}: {h}" for c, h in hints)

    def _look_at_faces(self, frame: dict, ids: list[str] | None = None) -> None:
        """Remember the top faces one decoded RGB-D frame shows (the caller holds the lock): of
        ``ids`` of any size, or of every object no larger than ``appearance_max_m`` whose face was
        not looked at for ``appearance_interval_s``. Only objects within ``appearance_range_m`` are
        looked at (a distant view would overwrite a sharp one), and a view showing less than
        ``appearance_min_coverage`` of a face is not kept."""
        cfg, est = self.config, self.estimator
        cam = est.camera_pose(frame["header"]["odometry"], frame["camera"])
        cam_p = np.asarray(cam["p"], float)
        stamp = float(frame["header"]["stamp"])
        for track in list(est.tracks.values()) if ids is None else [est.tracks[i] for i in ids if i in est.tracks]:
            if track.held:
                continue
            known = track.appearance
            if float(np.linalg.norm(track.pos - cam_p)) > cfg.appearance_range_m:
                continue
            if ids is None and (track.footprint > cfg.appearance_max_m
                                or (known is not None and not known.stale and stamp - known.stamp < cfg.appearance_interval_s)):
                continue
            cols, rows = est.face_grid(track, cfg.appearance_cell_m)
            rgb = top_face_array(frame, cam, track.pos, track.size, float(track.yaw), track.top(), cols, rows)
            if np.isfinite(rgb[..., 0]).mean() >= cfg.appearance_min_coverage:
                est.remember_face(track.id, rgb, stamp)

    def _appearance(self, entity: str) -> dict:
        with self._lock:
            track = self.estimator.tracks.get(entity)
            known = track.appearance if track is not None else None
            if known is None:
                return {"appearance": None}
            return {"appearance": {"seenAgoS": round(self.estimator.stamp - known.stamp, 1),
                                   "coverage": round(known.coverage, 2), "spread": round(known.spread, 1),
                                   "stale": known.stale}}

    def match_face(self, entities: list[str], reference: str, px: int = 12) -> dict:
        """Where the top face of each of ``entities`` appears on the top face of ``reference``
        printed at the same scale (puzzle pieces on the picture of their box, tiles on a pattern),
        from the faces the robot remembers (:meth:`_look_at_faces`), refreshed first with one fresh
        RGB-D frame for those in view. Each face is resampled to ``px`` cells across its shorter
        side and the reference to cells of the same size, then compared by :func:`match_face`.

        :param entities: Ego-State ids of the objects whose faces are sought.
        :param reference: Ego-State id of the object showing the whole picture.
        :param px: cells across each object's shorter side, 4 to 16.
        :returns: ``{reference, referenceCoverage, results, stamp}``; each result holds ``entity``,
            ``cellM``, ``spread``, ``faceCoverage``, ``seenAgoS`` and ``matches`` (``offsetM``: the
            face centre's place in the reference frame from its centre; ``turnDeg``: quarter turns
            counter-clockwise that align the face with the reference; ``placeYawDeg``: the object
            heading that does so with the reference's heading; ``score``: correlation, 1 =
            identical), or ``entity`` and ``error`` when its face is not known well enough.
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range or the reference
            face not known well enough."""
        if not isinstance(entities, list) or not entities or not all(isinstance(e, str) for e in entities):
            raise ValueError("entities must be a non-empty list of object ids")
        if not 4 <= px <= 16:
            raise ValueError("px must be between 4 and 16")
        frames = self._camera_frames()
        frame = frames[0]
        with self._lock:
            est = self.estimator
            missing = [eid for eid in [*entities, reference] if eid not in est.tracks]
            if missing:
                raise KeyError(f"unknown object {missing[0]!r}")
            for each in frames:
                self._look_at_faces(each, [*entities, reference])
            odometry = frame["header"]["odometry"]

            def why(track) -> str:
                return self._view_hint(odometry, track.pos, track.top())

            ref_track = est.tracks[reference]
            ref = ref_track.appearance
            if ref is None or ref.stale or ref.coverage < MATCH_MIN_COVERAGE:
                raise ValueError(f"the face of {reference} is not known well enough "
                                 f"({0 if ref is None else round(ref.coverage, 2)} seen): {why(ref_track)}")
            ref_rgb, ref_size, ref_yaw = ref.rgb.copy(), ref_track.size.copy(), float(ref_track.yaw)
            faces = {}
            for eid in entities:
                track = est.tracks[eid]
                known = track.appearance
                if known is None or known.stale:
                    faces[eid] = f"its face has not been seen since it was last put down: {why(track)}"
                elif known.coverage < MATCH_MIN_COVERAGE:
                    faces[eid] = f"only {known.coverage:.0%} of its face has been seen: {why(track)}"
                else:
                    faces[eid] = (known.rgb.copy(), track.size.copy(), round(est.stamp - known.stamp, 1), known.spread)
        results = []
        for eid in entities:
            face = faces[eid]
            if isinstance(face, str):
                results.append({"entity": eid, "error": face})
                continue
            rgb, size, ago, spread = face
            cell = float(min(size[0], size[1])) / px
            ox, oy = max(1, round(size[0] / cell)), max(1, round(size[1] / cell))
            rx, ry = round(ref_size[0] / cell), round(ref_size[1] / cell)
            if rx <= max(ox, oy) or ry <= max(ox, oy):
                results.append({"entity": eid, "error": f"{reference} is not larger than {eid}"})
                continue
            if max(rx, ry) > MATCH_REFERENCE_CELLS_MAX:
                results.append({"entity": eid, "error": f"{reference} spans more than {MATCH_REFERENCE_CELLS_MAX} "
                                                        f"cells of {cell:.3f} m; use a smaller px"})
                continue
            found = match_face(resample_face(rgb, ox, oy), resample_face(ref_rgb, rx, ry))
            matches = [{"offsetM": [round(float((m["col"] - rx / 2) * ref_size[0] / rx), 3),
                                    round(float((ry / 2 - m["row"]) * ref_size[1] / ry), 3)],
                        "turnDeg": 90 * m["turns"],
                        "placeYawDeg": round(math.degrees(wrap_angle(ref_yaw + m["turns"] * math.pi / 2)), 1),
                        "score": m["score"]} for m in found["matches"]]
            results.append({"entity": eid, "cellM": round(cell, 4), "spread": round(spread, 1),
                            "faceCoverage": round(float(np.isfinite(rgb[..., 0]).mean()), 2), "seenAgoS": ago,
                            "matches": matches})
        return {"reference": reference, "referenceCoverage": round(float(np.isfinite(ref_rgb[..., 0]).mean()), 2),
                "results": results, "stamp": frame["header"]["stamp"]}

    def observe(self) -> dict:
        frame = self.link.sensors()
        cameras = self._cameras(frame)
        if frame.get("imaging") == "rgbd":
            visible = []
            for camera in cameras:
                rgbd = decode_rgbd(self.link.rgbd(camera))
                if rgbd["mask"] is None and not self._segment(rgbd):
                    raise RuntimeError(f"the {camera} image could not be segmented: {self.segmentation['lastError']}")
                with self._lock:
                    cam = self.estimator.camera_pose(rgbd["header"]["odometry"], camera)
                for seg in segments(rgbd, cam):
                    centre, size, _ = box_of(seg.points)
                    probs = self.estimator.category_probs(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac,
                                                          True, seg.semantics)
                    best = max(probs, key=probs.get)
                    visible.append({"category": best, "confidence": round(probs[best], 2), "rangeM": round(seg.range, 2),
                                    "camera": camera})
        else:
            visible = [{"category": d["category"], "rangeM": round(d["range"], 2), "camera": d.get("camera")}
                       for d in frame["detections"] if d.get("camera") in cameras]
        out = {"t": frame["stamp"], "visible": visible}
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            out["imagePaths"] = {}
            for camera in cameras:
                path = self.out_dir / f"observe-{frame['stamp']:.2f}-{camera}.png"
                path.write_bytes(self.link.camera_png(camera=camera))
                out["imagePaths"][camera] = str(path)
        return out

    def predict(self, skill: str, params: dict) -> dict:
        es = self.ego_state()
        receipt = self.esim.predict(es, skill, params)
        limit = self.config.max_pick_risk if receipt.get("skill") == "pick" else self.config.max_risk
        receipt["withinRiskLimit"] = receipt["feasible"] and receipt["risk"] <= limit
        receipt["maxRisk"] = limit
        with self._lock:
            self.receipts[receipt["id"]] = receipt
        return receipt

    def annotate(self, request: dict) -> dict:
        """Name an Ego-State object as the robot's agent identified it (``label``, ``affordances``).

        :param request: ``{entity, label?, affordances?}``.
        :returns: the object's summary entry.
        :raises ValueError: malformed fields; :raises KeyError: no such object."""
        entity, label, affordances = request.get("entity"), request.get("label"), request.get("affordances")
        if not isinstance(entity, str):
            raise ValueError("annotate needs an entity id")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError("label must be a non-empty string")
        if affordances is not None and (not isinstance(affordances, list) or not all(isinstance(a, str) for a in affordances)):
            raise ValueError("affordances must be a list of strings")
        with self._lock:
            if entity not in self.estimator.tracks:
                raise KeyError(f"unknown object {entity!r}")
            self.estimator.annotate(entity, label, affordances)
            self._publish_view_scene()
        return next(o for o in self.summary()["objects"] if o["id"] == entity)

    def receipt(self, receipt_id: str) -> dict:
        if receipt_id not in self.receipts:
            raise KeyError(f"unknown receipt {receipt_id!r}")
        return self.receipts[receipt_id]

    # ------------------------------------------------------------------ alerts
    def alert(self, message: str, source: str, level: str = "error", goal: str = "", step: str = "",
              diagnosis: str = "") -> dict:
        """Record a situation the robot cannot resolve by itself (an unexpected exception, or a
        problem its agent or mission reports) with the goal and step it was pursuing and the
        reporter's own diagnosis. A goal or step left empty is taken from the published plan. A
        repeat of an unacknowledged message only bumps its count."""
        message = message[:ALERT_TEXT_MAX]
        with self._lock:
            plan = self.plan or {}
            for existing in reversed(self.alerts):
                if not existing["acknowledged"] and existing["message"] == message:
                    existing["count"] += 1
                    existing["t"] = round(self.estimator.stamp, 2)
                    return existing
            record = {"id": next(self._alert_ids), "t": round(self.estimator.stamp, 2), "level": level,
                      "source": source, "message": message, "count": 1, "acknowledged": False,
                      "goal": (goal or plan.get("goal", ""))[:ALERT_TEXT_MAX],
                      "step": (step or plan.get("current", ""))[:ALERT_TEXT_MAX],
                      "diagnosis": diagnosis[:ALERT_TEXT_MAX]}
            self.alerts.append(record)
            del self.alerts[:-50]
        self._log("alert", message)
        return record

    def alerts_view(self) -> dict:
        with self._lock:
            return {"alerts": [dict(a) for a in self.alerts],
                    "active": [dict(a) for a in self.alerts if not a["acknowledged"]]}

    def acknowledge(self, alert_id: int | None = None) -> dict:
        with self._lock:
            for record in self.alerts:
                if alert_id is None or record["id"] == alert_id:
                    record["acknowledged"] = True
        return self.alerts_view()

    # ---------------------------------------------------------------- activity
    def _log(self, kind: str, text: str) -> None:
        """Append to the activity record the console shows below the Ego-State view: ``step``,
        ``difficulty``, ``state``, ``skill`` and ``alert`` entries, newest last."""
        with self._lock:
            self.activity.append({"id": next(self._activity_ids), "t": round(self.estimator.stamp, 1),
                                  "kind": kind, "text": text[:ALERT_TEXT_MAX]})

    def _skill_begin(self, skill: str, receipt: dict) -> None:
        """Run inside the job thread, right before the skill moves.

        A pick takes one fresh look at its target from where the robot stands; that is worth a
        few seconds, but not on the request path of ``run``. The look moves the belief, and the
        receipt's dock pose was derived from the old position: a small correction is folded into
        the plan here, a large one invalidates it — the mission re-plans from the corrected
        belief instead of the robot driving to a pose that belongs to an object that is no
        longer there."""
        plan = (receipt.get("plan") or {}) if skill == "pick" else {}
        target, before = plan.get("target"), plan.get("targetP")
        if not target or before is None:
            return
        if not self._fresh_look_once(target):
            return  # no new observation, so the plan is still as good as the belief
        track = self.estimator.tracks.get(target)
        if track is None:
            return
        moved = float(np.linalg.norm(track.pos[:2] - np.asarray(before[:2])))
        if moved <= 0.02:
            return
        if moved > self.config.redock_max_m:
            raise SkillFailed(f"target {target} moved {moved * 100:.0f} cm since planning; re-plan")
        heading = (math.cos(plan["yaw"]), math.sin(plan["yaw"]))
        backoff = math.hypot(plan["dock"][0] - plan["predock"][0], plan["dock"][1] - plan["predock"][1])
        reach = float(plan.get("reach", 0.35)) + self.esim.tool_offset
        dock = [float(track.pos[0]) - heading[0] * reach, float(track.pos[1]) - heading[1] * reach]
        predock = [dock[0] - heading[0] * backoff, dock[1] - heading[1] * backoff]
        plan.update({"targetP": [float(v) for v in track.pos], "dock": dock, "predock": predock})
        if plan.get("path"):
            plan["path"][-1] = predock
        print(f"ego: pick plan for {target} re-docked after a {moved * 100:.0f} cm correction",
              file=sys.stderr, flush=True)

    def _skill_finished(self, job) -> None:
        result = {"succeeded": "完成", "failed": "失败", "stopped": "被停止"}.get(job.state, job.state)
        self._log("skill", f"{job.skill} {result}" + (f"：{job.message}" if job.state != "succeeded" else ""))

    def activity_since(self, since: int = 0) -> list[dict]:
        """Activity entries with an id above ``since``, oldest first.

        :param since: the id of the newest entry the caller already has.
        :returns: the newer entries.
        """
        with self._lock:
            return [dict(e) for e in self.activity if e["id"] > since]

    # -------------------------------------------------------------------- plan
    def set_plan(self, plan: dict) -> dict:
        """Publish what the robot is trying to achieve: a goal, its steps with their status, the
        step in progress and the latest difficulty. A client (a mission script, the robot agent)
        owns its content; the runtime only stores the latest one and stamps it."""
        goal = plan.get("goal")
        steps = plan.get("steps", [])
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("a plan needs a goal")
        if not isinstance(steps, list) or not all(isinstance(st, dict) and isinstance(st.get("name"), str)
                                                   and st.get("status") in PLAN_STEP_STATES for st in steps):
            raise ValueError(f"plan steps need a name and a status in {sorted(PLAN_STEP_STATES)}")
        record = {"goal": goal[:300], "source": str(plan.get("source", "agent"))[:60],
                  "steps": [{"name": st["name"][:200], "status": st["status"]} for st in steps[:30]],
                  "round": plan.get("round"), "current": str(plan.get("current") or "")[:400],
                  "difficulty": str(plan.get("difficulty") or "")[:600], "state": str(plan.get("state") or "running")[:40],
                  "t": round(self.estimator.stamp, 2)}
        with self._lock:
            previous, self.plan = self.plan or {}, record
        if record["current"] and record["current"] != previous.get("current"):
            self._log("step", record["current"])
        if record["difficulty"] and record["difficulty"] != previous.get("difficulty"):
            self._log("difficulty", record["difficulty"])
        if record["state"] != previous.get("state", "running"):
            self._log("state", record["state"])
        return record

    def plan_view(self, since: int = 0) -> dict:
        """The latest published plan (``None`` before any), the skill running now and the activity
        entries newer than ``since``.

        :param since: the id of the newest activity entry the caller already has.
        :returns: ``{"plan", "skill", "activity", "activityLast", "t"}`` (``activityLast`` is the newest
            id, lower than ``since`` after the runtime restarted).
        """
        running = next((j for j in reversed(list(self.executor.jobs.values())) if j.state == "running"), None)
        with self._lock:
            plan = dict(self.plan) if self.plan else None
        with self._lock:
            newest = self.activity[-1]["id"] if self.activity else 0
        return {"plan": plan, "skill": running.to_dict() if running else None, "activity": self.activity_since(since),
                "activityLast": newest, "t": round(self.estimator.stamp, 2)}

    # ----------------------------------------------------------------- control
    def control(self) -> dict:
        return {"owner": self.control_owner}

    def acquire_control(self, owner: str) -> dict:
        """Give ``owner`` exclusive use of skills until it releases control; the running skill stops.
        Clients that are not the owner cannot start skills meanwhile, but anyone may still stop."""
        if not owner:
            raise ValueError("control needs an owner name")
        with self._lock:
            if self.control_owner not in (None, owner):
                raise PermissionError(f"the robot is already controlled by {self.control_owner}")
            self.control_owner = owner
        self.executor.stop()
        return self.control()

    def release_control(self, owner: str) -> dict:
        with self._lock:
            if self.control_owner == owner:
                self.control_owner = None
        return self.control()

    def _check_control(self, owner: str | None) -> None:
        if self.control_owner is not None and owner != self.control_owner:
            raise PermissionError(f"the robot is controlled by {self.control_owner}; wait until it releases control")

    def _fresh_look_once(self, target_id: str) -> bool:
        """At most one fresh look per target per few seconds of body time.

        :returns: whether a look was taken (and so whether the belief may have moved).
        """
        now = self.estimator.stamp
        if now - self._looked_at.get(target_id, -1e9) < self.config.fresh_look_every_s:
            return False
        self._looked_at[target_id] = now
        return self._fresh_look(target_id)

    def _fresh_look(self, target_id: str) -> bool:
        """Re-place the pick's target from a class-prompted segmentation taken right now.

        An accumulated position can be biased by the partial views it was fused from — a rim arc
        read as confidently as the bowl itself. The action deserves the best single look there is,
        and this is it: the robot stands beside the object and the camera points at it. The class
        name is the robot's own prior (another name for the same thing is tried too, because a mug
        is a cup to half the world); the believed position only decides which returned instance is
        the one meant. The track keeps its identity; only where it stands is restated."""
        url = os.environ.get("STEWARD_SEGMENTER_URL")
        track = self.estimator.tracks.get(target_id)
        if url is None or track is None or not body_ok(track.category):
            print(f"ego: fresh look for {target_id} skipped (url={'set' if url else 'unset'}, "
                  f"track={'known' if track else 'unknown'})", file=sys.stderr, flush=True)
            return False
        try:
            frame = decode_rgbd(self.link.rgbd())
            cam = self.estimator.camera_pose()
            buf = io.BytesIO()
            Image.fromarray(frame["rgb"]).save(buf, format="PNG")
            image = base64.b64encode(buf.getvalue()).decode("ascii")
            h, w = frame["depth"].shape
            f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            rot = matrix_from_quat(cam["q"])
            origin = np.asarray(cam["p"], float)

            def found_in(body: dict) -> list[tuple[np.ndarray, float]]:
                """World centre and score of every instance the depth can place."""
                out = []
                if not body.get("mask_png_base64"):
                    return out
                labels = np.array(Image.open(io.BytesIO(base64.b64decode(body["mask_png_base64"]))))
                for info in body.get("instances", []):
                    ys, xs = np.nonzero(labels == int(info["id"]))
                    dd = frame["depth"][ys, xs].astype(np.float32)  # decode_rgbd gives metres
                    ok = (dd > 0.05) & (dd < 6.0)
                    if os.environ.get("STEWARD_FRESH_LOOK_DEBUG"):
                        print(f"ego dbg: instance {info['id']} score {info['score']:.2f} "
                              f"px {len(ys)} depth-valid {int(ok.sum())} "
                              f"depth range {float(dd.min()) if len(dd) else -1:.2f}..{float(dd.max()) if len(dd) else -1:.2f}",
                              file=sys.stderr, flush=True)
                    if int(ok.sum()) < self.config.relocate_min_points:
                        continue
                    cam_pts = np.stack([dd[ok], -(xs[ok] - cx) * dd[ok] / f,
                                        -(ys[ok] - cy) * dd[ok] / f], axis=1)
                    out.append((np.median(cam_pts @ rot.T + origin, axis=0), float(info["score"])))
                return out

            prompts = _fresh_look_prompts(track.category, track.color)
            found: list[tuple[np.ndarray, float]] = []
            asked = []
            for prompt in prompts:
                body = call("POST", f"{url.rstrip('/')}/segment_text",
                            {"image_base64": image, "text_prompt": prompt, "min_score": 0.4},
                            timeout=30.0)
                asked.append(f"{prompt}:{len(body.get('instances', []))}")
                found += found_in(body)
            best, best_d = None, self.config.relocate_max_jump_m
            distances = []
            for centre, score in found:
                dist = float(np.linalg.norm(centre - track.pos[:3]))
                distances.append(f"{dist * 100:.0f}cm z{centre[2]:.2f}(score {score:.2f})")
                if abs(float(centre[2]) - float(track.pos[2])) > self.config.relocate_max_dz_m:
                    continue  # an instance at another height is another thing (a table edge, the floor)
                if dist < best_d:
                    best, best_d = centre, dist
            fallback = self.config.relocate_fallback_m
            if best is None and found:
                # The belief itself can be the wrong thing here (a rim fragment decimetres from
                # the bowl it belongs to). A confidently segmented nearest instance farther away
                # is still the best reading of where this object stands.
                best, best_d = None, fallback
                for centre, score in found:
                    dist = float(np.linalg.norm(centre - track.pos[:3]))
                    if abs(float(centre[2]) - float(track.pos[2])) > self.config.relocate_max_dz_m:
                        continue
                    if dist <= fallback and score >= self.config.relocate_fallback_score and dist < best_d:
                        best, best_d = centre, dist
            if best is None:
                # Text prompts die at dock range, where the robot often stands after a first
                # failed attempt. There the point sweep is at its best instead: the object fills
                # the frame, so prompt the pixels around where the belief says it projects.
                rel = track.pos[:3] - origin
                cam_rel = rel @ rot
                if cam_rel[0] > 0.15:
                    u = int(round(cx - cam_rel[1] * f / cam_rel[0]))
                    v = int(round(cy - cam_rel[2] * f / cam_rel[0]))
                    if 0 <= u < w and 0 <= v < h:
                        points = [[u + du, v + dv] for du in (-10, 0, 10) for dv in (-10, 0, 10)]
                        near = call("POST", f"{url.rstrip('/')}/segment_instances",
                                    {"image_base64": image, "points": points, "min_score": 0.4},
                                    timeout=40.0)
                        best, best_d = None, fallback
                        for centre, _score in found_in(near):
                            dist = float(np.linalg.norm(centre - track.pos[:3]))
                            if dist < best_d:
                                best, best_d = centre, dist
            if best is None:
                print(f"ego: fresh look for {target_id} ({track.category}/{track.color}): "
                      f"{len(found)} text instances from [{', '.join(asked)}], "
                      f"none within {fallback:.2f} m of the belief "
                      f"({', '.join(distances) or 'none returned'})", file=sys.stderr, flush=True)
                return False

            with self._lock:
                moved = self.estimator.relocate(target_id, best)
            if moved:
                print(f"ego: fresh look placed {target_id} at ({best[0]:.3f}, {best[1]:.3f}, {best[2]:.3f})",
                      file=sys.stderr, flush=True)
            return bool(moved)
        except Exception as exc:
            print(f"ego: fresh look for {target_id} failed: {exc}", file=sys.stderr, flush=True)
            return False

    def run(self, receipt_id: str, owner: str | None = None) -> dict:
        """Execute a predicted plan. The runtime re-checks the receipt here, where the motion happens."""
        self._check_control(owner)
        receipt = self.receipt(receipt_id)
        now = self.estimator.stamp
        if receipt.get("used"):
            raise PermissionError(f"receipt {receipt_id} was already executed; predict again")
        if not receipt["feasible"]:
            raise PermissionError(f"receipt {receipt_id} is not feasible: {'; '.join(receipt['issues'])}")
        if now > receipt["expiresAt"]:
            raise PermissionError(f"receipt {receipt_id} expired at t={receipt['expiresAt']:.1f}s; predict again")
        limit = receipt.get("maxRisk", self.config.max_risk)
        if receipt["risk"] > limit:
            raise PermissionError(f"receipt risk {receipt['risk']:.2f} exceeds the limit {limit:.2f}")
        receipt["used"] = True
        job = self.executor.start(receipt["skill"], receipt).to_dict()
        params = receipt.get("params") or {}
        target = params.get("target") or params.get("on") or (f"({params['x']:.2f}, {params['y']:.2f})" if "x" in params else "")
        self._log("skill", f"{receipt['skill']} 开始" + (f"：{target}" if target else ""))
        return job

    def explore(self, owner: str | None = None) -> dict:
        """Look around in place. Rotation without translation needs no receipt."""
        self._check_control(owner)
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
