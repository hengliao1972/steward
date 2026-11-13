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
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from steward.ego.perception import (PerceptionConfig, box_of, decimated, decode_rgbd, face_colour_stats, match_face, resample_face, segments,
                                    top_face,
                                    top_face_array,
                                    world_up_face)
from steward.ego.kinematics import gripper_pose
from steward.ego.arm import Arm
from steward.ego.esim import Esim, EsimConfig, PredictionError
from steward.ego.estimator import SELF_ID, EgoEstimator, _CONTENT
from steward.ego import reflection
from steward.ego.reflection import ReflectionConfig
from steward.ego.roles import RoleConfig, assign_roles
from steward.ego.segmenter import DepthClusters, Segmenter, SegmenterError
from steward.ego.skills import MotionConfig, SkillExecutor, SkillFailed
from steward.ego.view import EgoViewRenderer
from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_rotate, wrap_angle, yaw_from_quat
from steward.net import HttpError, call
from steward.scene.schema import SceneState
from steward.viewloop import ViewLoop


@dataclass(frozen=True)
class FaceMatchSnapshot:
    """Copied robot observations for pure matching after the input/world-clock hold is released."""

    reference: str
    rgb: np.ndarray
    size: np.ndarray
    yaw: float
    faces: dict[str, tuple | str]
    entities: list[str]
    px: int
    stamp: float


@dataclass
class RuntimeConfig:
    perception_hz: float = 10.0
    telemetry_hz: float = 2.0
    appearance_cell_m: float = 0.002  # cell size of the remembered top faces (~2 mm; a 5.75 cm piece is ~29 cells)
    appearance_max_m: float = 0.6  # larger objects' faces are remembered only when asked for
    appearance_range_m: float = 1.5  # faces farther from the camera are not looked at
    detail_px: int = 24  # pixels across a face in the source image below which ego_inspect says to come closer
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
    perception_rows_per_degree: float = 8.0
    """Most image rows per degree of a camera's vertical field of view in an RGB-D frame that is
    segmented and fused; a finer frame is reduced by the smallest whole factor that fits
    (:func:`steward.ego.perception.decimated`). The wide camera's 360 rows over 100 degrees pass
    unchanged and the head camera's 1920 rows over 60 degrees become 480. The head camera's full
    frame is for reading a face up close (:meth:`EgoRuntime.inspect`) and for the segmenter's
    enlarged crops, which both keep it whole; segmenting and fusing every one of its five million
    pixels takes some twenty seconds a frame. The rows left over are not thrown away: each depth
    of the reduced frame is the mean of a block of them, with a quarter of the noise at four rows
    to one."""
    depth_join: float = 0.02
    """Greatest difference in depth, as a share of the depth, between the pixels of a block that
    are averaged into one depth of the reduced frame; farther apart, they are two surfaces."""
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
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)


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
MATCH_REFERENCE_CELLS_MAX = 192  # cells along a reference side for match_face
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
                 out_dir: Path | None = None, segmenter: Segmenter | None = None, arm: Arm | None = None):
        """``arm`` is the body's planar arm (:mod:`steward.ego.arm`); without one the body is the
        mobile kitchen robot, whose lift, reach slide and wrist are the planar arm itself."""
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
        self.segmentation = {"frames": 0, "cached": 0, "skipped": 0, "failures": 0, "lastSeconds": None,
                             "lastError": None}
        self.esim = Esim(robot_model, self.config.esim, arm)
        self.arm = self.esim.arm
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
        self._episode: int | None = None  # the world's episode number of the last sensor frame
        self._generation = 0  # counts restarts: state read in one episode is not applied in the next
        self._receipt_generation: dict[str, int] = {}  # receipt id -> the generation it was predicted in
        self._looked_at: dict[str, float] = {}  # target id -> body stamp of its last fresh look
        self._roles: dict[str, str] = {}  # the latest role assignment, kept unless clearly outweighed
        self.activity: collections.deque[dict] = collections.deque(maxlen=ACTIVITY_MAX)
        self._activity_ids = itertools.count(1)
        self.boot = uuid.uuid4().hex  # a console drops the activity it shows when this changes
        self._input_lock = threading.Lock()
        self.inputs: list[dict] = []  # Ego API requests as applied, with their world time (:meth:`apply`)
        self._alert_ids = itertools.count(1)
        self.executor = SkillExecutor(lambda cmd: self.link.command(self.arm.command(cmd)), self._motion_state,
                                      self._mark_held, self.config.motion,
                                      shift=self._shift_track, on_error=lambda msg: self.alert(msg, "skill"),
                                      on_finish=self._skill_finished, on_release=self._released,
                                      on_begin=self._skill_begin, clock=self.link,
                                      closed_drawers=self._closed_drawer_ids, drawer_plan=self._drawer_plan,
                                      pin=self._pin_track, furniture=self._furniture_id,
                                      contents=self._content_ids, content_positions=self._content_positions,
                                      resettle=self._resettle_contents,
                                      home_empty=lambda tick: self.arm.home_empty(self.link.command, tick),
                                      deploy_working=(lambda tick: self.arm.deploy_working(self.link.command, tick)
                                                      if callable(getattr(self.arm, "deploy_working", None))
                                                      else None),
                                      verify_screw_seat=self._verify_screw_seat,
                                      replan_pick=self._replan_pick,
                                      replan_insert_place=self._replan_insert_place)
        self._view_scene: tuple[dict, int, float] | None = None  # (ES dict, version, stamp) for the view
        # The belief part of the motion state, recorded under ``_lock`` at each read and around each
        # perception update; :meth:`_motion_state` falls back on it while another thread holds the lock.
        self._motion_belief: dict | None = None
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
            whole = [decode_rgbd(self.link.rgbd(camera)) for camera in due]
            if mode == "rgbd-segmenter":
                whole = [rgbd for rgbd in whole if not self._idle_near_frame(rgbd)]
            pairs = [(decimated(rgbd, self._perception_rows(rgbd), self.config.depth_join), rgbd) for rgbd in whole]
            unmasked = [pair for pair in pairs if pair[0]["mask"] is None]
            if unmasked and not self._segment(unmasked):
                pairs = [pair for pair in pairs if pair[0]["mask"] is not None]
            frames = [rgbd for rgbd, _whole in pairs]
            with self._lock:
                if self._new_episode(frame):
                    self._restart()
                self._keep_motion_belief()  # what a motion loop reads while this update runs
                if not frames:
                    self.estimator.update_body(frame)
                for rgbd, full in pairs:
                    self.estimator.update_rgbd(rgbd, defer=self._handoff(rgbd, frames))
                    self._look_at_faces(full)
                self._publish_view_scene()
                self._keep_motion_belief()
            return
        frame = {**frame, "cameras": [c for c in frame.get("cameras", []) if c["id"] in cameras],
                 "detections": [d for d in frame.get("detections", []) if d.get("camera") in cameras]}
        with self._lock:
            if self._new_episode(frame):
                self._restart()
            self._keep_motion_belief()  # what a motion loop reads while this update runs
            self.estimator.update(frame)
            self._publish_view_scene()
            self._keep_motion_belief()

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
        handled = camera == self.near_camera and self._arm_moved(last, frame)
        # A closed look dwells within a degree of a frame taken while the base was still
        # moving. That stop is its own image. A look that goes on to open drawers keeps the
        # coarser stop: those drawers are approached from the estimate the turn already made.
        closed_look = self.executor._look and not self.executor.config.inspect_closed_drawers
        yaw_eps = 0.005 if closed_look else 0.02
        # The same closed look samples the turn every 0.04 rad. The coarser step is the one
        # a drawer visit keeps, so that approach stays on the frames it already measured.
        turn_rad = 0.04 if closed_look else cfg.keyframe_turn_rad
        turned = abs(wrap_angle(odo["yaw"] - was["yaw"])) >= turn_rad
        changed = math.hypot(odo["x"] - was["x"], odo["y"] - was["y"]) >= 0.02 \
            or abs(wrap_angle(odo["yaw"] - was["yaw"])) >= yaw_eps
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

    def _perception_rows(self, rgbd: dict) -> int:
        """Most image rows ``rgbd`` is segmented and fused at
        (:attr:`RuntimeConfig.perception_rows_per_degree`)."""
        return max(1, math.ceil(self.config.perception_rows_per_degree * rgbd["fovy"]))

    def _idle_near_frame(self, rgbd: dict) -> bool:
        """Whether ``rgbd`` is a frame of the near camera with nothing in the range that camera is
        in charge of: fewer pixels than a region needs lie nearer than ``near_range_m`` once the
        floor and the robot's own links are left out. Such a frame is not segmented (it is counted
        as ``skipped``); what it shows farther away is the other cameras' to measure, and they
        see it with several times the pixels."""
        if rgbd["camera"] != self.near_camera:
            return False
        limits = PerceptionConfig()
        _ys, _xs, points, on_body = self._seen_points(rgbd, self.config.near_range_m)
        with self._lock:
            cam = np.asarray(self.estimator.camera_pose(rgbd["header"]["odometry"], rgbd["camera"])["p"], float)
        near = ~on_body & (points[:, 2] > limits.floor_cut) & (np.linalg.norm(points - cam, axis=1) < self.config.near_range_m)
        if int(near.sum()) >= limits.min_points:
            return False
        self._keyframe[rgbd["camera"]] = rgbd["header"]
        self.segmentation["skipped"] += 1
        return True

    def _seen_points(self, rgbd: dict, max_depth: float | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The pixels of ``rgbd`` with a depth (under ``max_depth`` when given) as map-frame points.

        :returns: ``(ys, xs, points, on_body)``: pixel rows and columns, the N×3 points, and which
            of them lie on the robot's own links as the frame's header reports the body."""
        depth = rgbd["depth"]
        valid = np.isfinite(depth) & (depth > 0.05)
        if max_depth is not None:
            valid &= depth < max_depth
        ys, xs = np.nonzero(valid)
        h, w = depth.shape
        f = (h / 2.0) / math.tan(math.radians(rgbd["fovy"]) / 2.0)
        d = depth[ys, xs].astype(np.float64)
        local = np.stack([d, -(xs - (w - 1) / 2.0) * d / f, -(ys - (h - 1) / 2.0) * d / f], axis=1)
        with self._lock:
            cam = self.estimator.camera_pose(rgbd["header"]["odometry"], rgbd["camera"])
            points = local @ matrix_from_quat(cam["q"]).T + np.asarray(cam["p"], float)
            on_body = self.estimator.on_body_at(points, rgbd["header"])
        return ys, xs, points, on_body

    def _depth_without_body(self, rgbd: dict) -> np.ndarray:
        """The depth of ``rgbd`` with the pixels on the robot's own links set to 0, for the
        geometric fill: depth clustering, unlike a world mask, cannot tell the arm from an object
        standing on the table under it. Only pixels within the body's reach of the camera are tested."""
        depth = np.array(rgbd["depth"], dtype=np.float32, copy=True)
        with self._lock:
            reach = self.estimator.body_reach(self.estimator.camera_pose(camera=rgbd["camera"])["p"])
        ys, xs, _points, on_body = self._seen_points(rgbd, reach)
        depth[ys[on_body], xs[on_body]] = 0.0
        return depth

    def _depth_clusters(self, rgbd: dict) -> tuple[np.ndarray, np.ndarray, DepthClusters]:
        """For one frame: its depth without the robot's body, the world's up direction in its
        camera frame, and the segmenter's depth clusters of it (:meth:`Segmenter.clusters`)."""
        with self._lock:
            cam = self.estimator.camera_pose(rgbd["header"]["odometry"], rgbd["camera"])
        up_cam = matrix_from_quat(cam["q"]).T @ np.array([0.0, 0.0, 1.0])
        depth = self._depth_without_body(rgbd)
        return depth, up_cam, self.segmenter.clusters(depth, rgbd["fovy"], up_cam)

    def _segment(self, pairs: list[tuple[dict, dict]]) -> bool:
        """Fill ``mask`` and ``labels`` of the frames of one instant from the robot's segmenter.

        :param pairs: ``(frame, whole)`` per camera: the frame perception fuses and the frame as
            the camera took it, from which the segmenter enlarges its crops.
        :returns: False (and an operator alert on the first failure in a row) when the frames
            could not be segmented; none of them is filled then."""
        for rgbd, _whole in pairs:
            self._keyframe[rgbd["camera"]] = rgbd["header"]
        stats = self.segmentation
        if self.segmenter is None:
            error = "this phase level sends no instance masks and the robot has no segmenter (steward ego --segmentation-config)"
        else:
            try:
                # What the depth alone shows is worked out while the service is asked about the
                # colours: the wait for its answer is otherwise idle, and longer than this work.
                with ThreadPoolExecutor(max_workers=len(pairs), thread_name_prefix="depth") as pool:
                    ahead = [pool.submit(self._depth_clusters, rgbd) for rgbd, _whole in pairs]
                    results = self.segmenter.segment_frames(
                        [(rgbd["rgb"], None if whole is rgbd else whole["rgb"]) for rgbd, whole in pairs])
                    views = [future.result() for future in ahead]
                zoomed = self.segmenter.zoom_frames(
                    [(result, rgbd["rgb"], depth, rgbd["fovy"], up_cam, None if whole is rgbd else whole["rgb"])
                     for (rgbd, whole), result, (depth, up_cam, _clusters) in zip(pairs, results, views)],
                    [clusters for _depth, _up_cam, clusters in views])
                for (rgbd, _whole), result, (depth, up_cam, clusters) in zip(pairs, zoomed, views):
                    result = self.segmenter.fill(result, depth, rgbd["fovy"], up_cam, clusters)
                    result = self.segmenter.peel(result, rgbd["rgb"], depth, rgbd["fovy"])
                    rgbd["mask"], rgbd["labels"] = result.mask, result.labels
            except SegmenterError as exc:
                error = str(exc)
                for rgbd, _whole in pairs:
                    rgbd["mask"] = None
            else:
                stats["frames"] += len(results)
                stats["cached"] += sum(result.cached for result in results)
                asked = [result.seconds for result in results if not result.cached]
                if asked:
                    stats["lastSeconds"] = round(max(asked), 2)
                stats["lastError"] = None
                return True
        stats["failures"] += len(pairs)
        if stats["lastError"] is None:
            self.alert(f"图像分割失败，本帧只更新机器人自身状态：{error}", "runtime")
        stats["lastError"] = error
        return False

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

    def _new_episode(self, frame: dict) -> bool:
        """Whether ``frame`` is from a new episode (the console's 重置 or a scenario load): its
        ``episode`` differs from the last frame's, or, from a world that numbers none, its clock
        went backwards. A reset at the very start of an episode leaves the clock where it was."""
        episode, self._episode = self._episode, frame.get("episode")
        if episode is not None and frame.get("episode") != episode:
            return True
        return frame["stamp"] < self.estimator.stamp - 1e-6

    def _restart(self) -> None:
        """A new episode started, which the robot experiences as a reboot: forget the whole
        Ego-State and everything derived from it (receipts, roles, looks, keyframes) and
        cancel any running skill. Track ids start again at ``e:1``."""
        self.executor.cancel()
        self._generation += 1
        self.estimator = EgoEstimator(robot_model=self.estimator.robot_model, config=self.estimator.config)
        self.receipts.clear()
        self._receipt_generation.clear()  # skills predicted before now are of another episode
        self._roles.clear()
        self._looked_at.clear()
        self._keyframe = {}
        self._pose_sample = None
        self._still_since = None
        self._motion_belief = None

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
                # The whole update runs under the hold, not only the remote request: a fill, peel
                # or fusion longer than the lease would let the world step mid-update, and the
                # frames a look samples would then depend on this machine's speed.
                with self._holding("perception"):
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
        """The belief, with the robot itself as it is now: objects are as old as the last
        perception update, but the body's own pose and joints are read live. With a slow segmenter
        the last update can be tens of seconds and a whole drive old, and a plan made from that
        self pose (a dock, a place height from the load in the hand) belongs to where the robot
        no longer is."""
        # The body is read after waiting out an RGB-D update in progress (seconds under load) --
        # read before it, the self pose and the load in the hand were already that old when
        # composed -- but not while holding the lock, so a slow sensor service does not stall
        # perception. A belief that moved on meanwhile (another update, or a reset) has a body at
        # least as new as the one read: that one is the answer then.
        with self._lock:
            generation = self._generation
        try:
            live = self.link.sensors()
        except Exception:  # noqa: BLE001 - the belief's own self pose is a usable answer when the body cannot be reached
            live = None
        with self._lock:
            est = self.estimator
            if live is not None and (generation != self._generation
                                     or live.get("odometry") is None or live.get("stamp") is None
                                     or live["stamp"] < est.stamp - 1e-6
                                     or (live.get("episode") is not None and self._episode is not None
                                         and live["episode"] != self._episode)):
                live = None  # older than the belief, or another episode's body: the belief's own
            return est.to_scene(live)

    def _motion_state(self) -> dict:
        """What the body is doing now, for the motion loop.

        Odometry, joints and the gripper come from the robot's live sensor header, not from the
        belief: with a real segmenter the next RGB-D frame can be seconds or tens of seconds
        away, and a motion loop that grades its progress on the perception clock either stalls
        or reads a frame rendered before it started and calls the turn failed. Objects stay the
        estimate they are — the body only needs to know where it is.

        Nor may the loop wait for the belief's lock: an RGB-D update holds it for its whole
        duration (seconds on a loaded host), the world clock goes on once the loop's hold lapses,
        and a turn or drive that set its ramp before the wait finds its deadline passed with the
        base still at the first set point. While another thread holds the lock, the objects are
        those recorded when the perception update in progress began (everything changed before
        it is in there) or at the last read, whichever came later."""
        if self._lock.acquire(blocking=False):
            try:
                belief = self._keep_motion_belief()
            finally:
                self._lock.release()
        else:
            belief = self._motion_belief
            if belief is None:  # nothing recorded yet: nothing to fall back on but waiting
                with self._lock:
                    belief = self._keep_motion_belief()
        state = {key: dict(value) if isinstance(value, dict) else value for key, value in belief.items()}
        state.pop("generation", None)
        try:
            live = self.link.sensors()
        except Exception:  # noqa: BLE001 - the belief is still a usable answer when the body cannot be reached
            return self.arm.virtualize(state)
        # A body from another episode than the objects (a reset race: a clock behind the belief's,
        # or another episode number) is not mixed in: the body and the objects of one answer
        # belong to one episode, so the belief is returned whole.
        if (live.get("stamp") is not None and belief.get("stamp") is not None and live["stamp"] < belief["stamp"] - 1e-6) \
                or (live.get("episode") is not None and belief.get("episode") is not None
                    and live["episode"] != belief["episode"]) or belief.get("generation") != self._generation:
            return self.arm.virtualize(state)
        for key in ("stamp", "odometry", "joints", "gripper", "grippers"):
            if live.get(key) is not None:
                state[key] = live[key]
        return self.arm.virtualize(state)

    def _keep_motion_belief(self) -> dict:
        """Record (and return) the belief part of the motion state; the caller holds ``_lock``."""
        est = self.estimator
        self._motion_belief = {
            "generation": self._generation, "episode": self._episode, "stamp": est.stamp, "odometry": dict(est.odometry), "joints": dict(est.joints),
            "gripper": dict(est.gripper), "grippers": dict(est.grippers), "posed": est.posed,
            "entities": {t.id: {"p": t.pos.tolist()} for t in est.tracks.values()}}
        return self._motion_belief

    def _mark_held(self, track_id: str) -> str:
        """Record the grasp the skill just felt, measured against the body as it is now: the last
        perception update can be a slow segmentation old, from before the base docked and the arm
        reached down, and an offset taken against it leaves the load believed half a metre from
        the hand (every place then plans its height from there)."""
        generation, at, body = self._skill_generation(), self.estimator.stamp, None
        try:
            body = self.link.sensors()
            at = float(body.get("stamp", at))  # the body's clock, now
        except Exception:  # noqa: BLE001 - the belief's own clock and pose are a usable second best
            body = None
        with self._lock:
            if not self._same_episode(generation, body):  # the skill or the body is another episode's
                return track_id
            return self.estimator.mark_held(track_id, at, gripper=self.arm.geometry.gripper, body=body)

    def _released(self, track_id: str, spot: list[float], tip_roll: float | None = None) -> None:
        """A place knows where it let go; the belief is told, not left to infer it later -- at the
        height the hand is at now, not the carry height the last perception update saw."""
        generation = self._skill_generation()
        try:
            body = self.link.sensors()
        except Exception:  # noqa: BLE001 - the last estimated height is a usable second best
            body = None
        with self._lock:
            if not self._same_episode(generation, body):  # the skill or the body is another episode's
                return
            self.estimator.release_at(track_id, spot, body=body, tip_roll=tip_roll)

    def _skill_generation(self) -> int:
        """The generation the running skill was predicted in: a callback reached after a reset (the
        old skill thread paused past it) belongs to that skill's episode, not to the current one.
        Outside a skill thread, the current generation."""
        name = threading.current_thread().name
        job = self.executor.jobs.get(name[len("skill-"):]) if name.startswith("skill-") else None
        if job is None:
            return self._generation
        return self._receipt_generation.get(job.receipt, -1)

    def _same_episode(self, generation: int, body: dict | None) -> bool:
        """Whether a skill event of ``generation``, with ``body`` read for it, belongs to the
        current episode; the caller holds ``_lock``."""
        if generation != self._generation:
            return False
        return not (body is not None and body.get("episode") is not None and self._episode is not None
                    and body["episode"] != self._episode)

    def _shift_track(self, track_id: str, delta: list[float]) -> None:
        with self._lock:
            self.estimator.shift(track_id, delta)

    def _pin_track(self, track_id: str, pinned: bool) -> None:
        """Keep a track's box still while a drawer in it is opened and closed.

        A drawer keeps the name ``drawer`` while it is pinned, so a view of the open drawer
        cannot rename it and make the close refuse. Furniture is only held still: its name stays."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            if track is None:
                return
            track.pinned = pinned
            if pinned and track.category == "drawer":
                track.label = "drawer"

    def _furniture_id(self, drawer_id: str) -> str | None:
        """The floor-standing piece whose footprint holds ``drawer_id``, if one is known."""
        with self._lock:
            drawer = self.estimator.tracks.get(drawer_id)
            if drawer is None:
                return None
            host = self.estimator._furniture_of(drawer)
            return None if host is None else host.id

    def _content_ids(self, drawer_id: str) -> list[str]:
        """Small tracks on a drawer's floor, held still while the robot looks along the drawer."""
        with self._lock:
            drawer = self.estimator.tracks.get(drawer_id)
            if drawer is None:
                return []
            # Only the bear, pistol and bottles. A bowl born on the first side step
            # is not held: pinning it froze a wrong name onto the pistol.
            return [tid for tid in self.estimator._near_drawer(drawer)
                    if self.estimator.tracks[tid].category in _CONTENT]

    def _content_positions(self, drawer_id: str) -> dict[str, list[float]]:
        """Positions of those tracks just before the drawer is pushed back."""
        with self._lock:
            return self.estimator.content_positions(drawer_id)

    def _resettle_contents(self, drawer_id: str, delta: list[float], before: dict[str, list[float]]) -> list[str]:
        """Bring back contents a late view of the open drawer left in front of it."""
        with self._lock:
            return self.estimator.resettle(drawer_id, np.asarray(delta, float), before)

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
            aff = list(ent.rigid.links[0].affordances)
            strings = [a.value for a in aff if isinstance(a.value, str)]
            holes = self._screw_holes_view(aff)
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
                "affordances": strings,
                "color": ent.state_vars["color"].value if "color" in ent.state_vars else None,
                "extentM": self._extent(ent.id),
                **({"screwHoles": holes} if holes else {}),
                **({"unsupported": True} if self._unsupported(ent.id) else {}),
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
                     "holding": held, "joints": {k: round(v, 3) for k, v in self.arm.joints_of(
                         {j.id: j.q.value[0] for j in me.rigid.joints if j.q.value},
                         {"x": p["p"][0], "y": p["p"][1], "yaw": yaw_from_quat(p["q"])}).items()}},
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

    @staticmethod
    def _screw_holes_view(affordances) -> list[dict]:
        """Public screw-hole markers for ego_state: id, body-frame ``p``/``n``, and whether they were observed."""
        out = []
        for aff in affordances:
            value = aff.value
            if not (isinstance(value, dict) and "screw-hole" in value):
                continue
            hole = value["screw-hole"]
            item = {"id": str(hole["id"]), "p": [round(float(v), 4) for v in hole["p"][:3]],
                    "source": aff.src or "prior"}
            if hole.get("n") is not None:
                item["n"] = [round(float(v), 4) for v in hole["n"][:3]]
            out.append(item)
        return out

    def screw_holes(self, entity: str) -> dict:
        """Locate an object's screw holes in the world frame and report which face upward.

        Pulls one fresh sensor frame first so tip comes from the detected pose, not from a place
        ``wristRoll`` claim (arm roll is not body tip). Facing uses the observed tip only
        (``tipSource=observed``); until then ``holesFaceUp`` is false.

        :param entity: Ego-State object id.
        :returns: ``{entity, tipRollDeg, tipSource, holesFaceUp, holes:[...]}``.
        :raises KeyError: unknown object.
        """
        from steward.math3d import quat_rotate

        # Refresh belief from the cameras before reading tip / hole facing.
        try:
            self._poll_once()
        except Exception:  # noqa: BLE001 - still answer from the last belief when sensors fail
            pass
        es = self.ego_state()
        ent = es.find(entity)
        if ent is None:
            raise KeyError(f"unknown object {entity!r}")
        with self._lock:
            track = self.estimator.tracks.get(entity)
            tip = float(track.tip_roll) if track is not None else 0.0
            tip_source = track.tip_roll_source if track is not None else "unknown"
        q = ent.pose.value["q"]
        origin = ent.pose.value["p"]
        holes = []
        for aff in ent.rigid.links[0].affordances:
            value = aff.value
            if not (isinstance(value, dict) and "screw-hole" in value):
                continue
            hole = value["screw-hole"]
            body_p = [float(v) for v in hole["p"][:3]]
            body_n = [float(v) for v in hole["n"][:3]] if hole.get("n") is not None else [0.0, 0.0, 1.0]
            through = bool(hole.get("through"))
            world_p = [origin[i] + quat_rotate(q, body_p)[i] for i in range(3)]
            world_n = quat_rotate(q, body_n)
            up_dot = float(world_n[2])
            # Through-holes are insertable from either face when the axis is vertical.
            if through:
                facing = "up" if abs(up_dot) > 0.7 else "side"
            elif up_dot > 0.7:
                facing = "up"
            elif up_dot < -0.7:
                facing = "down"
            else:
                facing = "side"
            item = {
                "id": str(hole["id"]),
                "bodyP": [round(v, 4) for v in body_p],
                "worldP": [round(v, 4) for v in world_p],
                "worldN": [round(v, 4) for v in world_n],
                "upDot": round(up_dot, 3),
                "facing": facing,
                "source": aff.src or "prior",
            }
            if through:
                item["through"] = True
            holes.append(item)
        observed = tip_source == "observed"
        any_up = observed and any(h["facing"] == "up" for h in holes)
        if not holes:
            hint = None
        elif not observed:
            hint = "tip not yet confirmed by camera; wait/observe again before trusting holesFaceUp"
        elif not any_up:
            if any(h.get("through") for h in holes):
                hint = "through-hole axis not vertical; lay the part flat (either face), then look again"
            else:
                hint = "no screw hole faces up; tip the part onto its side with place roll=±π/2, then look again"
        else:
            hint = None
        return {
            "entity": entity,
            "category": ent.category.value,
            "tipRollDeg": round(math.degrees(tip), 1),
            "tipSource": tip_source,
            "holesFaceUp": any_up,
            "holes": holes,
            "hint": hint,
        }

    def _drive_screw_tip_gate(self, target: object) -> str | None:
        """Refuse drive_screw until the screw's tip is camera-confirmed upright (head up).

        :param target: screw entity id from predict params.
        :returns: refusal message, or None when the tip is observed upright.
        """
        if not isinstance(target, str) or not target:
            return "drive_screw needs target: id of the seated screw"
        with self._lock:
            track = self.estimator.tracks.get(target)
            if track is None:
                return f"unknown screw target {target!r}"
            tip_source = track.tip_roll_source
            tip = float(track.tip_roll)
        if tip_source != "observed":
            return (
                f"{target} tip not yet confirmed by camera (tipSource={tip_source}); "
                "ego_observe / ego_screw_holes until tipSource=observed and the screw stands "
                "tip-down in the hole with head up, then drive_screw again")
        if abs(tip) > 0.35:
            return (
                f"{target} is not upright tip-down (tipRollDeg={math.degrees(tip):.0f}); "
                "insert_screw_into_hole (or place tip-down) into the hole with head up, then observe again")
        return None

    def _insert_screw_tip_gate(self, screw_id: object | None = None) -> str | None:
        """Refuse insert until the held screw's tip end is camera-confirmed.

        Without ``tipSource=observed``, ES treats an unknown tip as upright and skips the stand
        roll — the screw is then placed sideways / head-down. Observe after pick first.

        :param screw_id: optional expected screw id (defaults to the held track).
        :returns: refusal message, or None when the held screw tip is observed.
        """
        with self._lock:
            held_id = self.estimator.held_id
            if not held_id:
                return "insert_screw_into_hole needs a toy_screw in the gripper"
            track = self.estimator.tracks.get(held_id)
            if track is None:
                return f"unknown held screw {held_id!r}"
            if track.category != "toy_screw":
                return (
                    f"insert_screw_into_hole needs a toy_screw in the gripper, got {held_id} "
                    f"({track.category})")
            if isinstance(screw_id, str) and screw_id and screw_id != held_id:
                return f"gripper holds {held_id}, not {screw_id}"
            tip_source = track.tip_roll_source
        if tip_source != "observed":
            return (
                f"{held_id} tip not yet confirmed by camera (tipSource={tip_source}); "
                "after pick, ego_observe / ego_screw_holes until tipSource=observed so the skill "
                "knows which end is the tip and can stand it tip-down into the hole")
        return None

    def _verify_screw_seat(self, screw_id: str) -> tuple[bool, str]:
        """After an insert place: poll cameras and confirm the screw stands tip-down in a hole.

        :param screw_id: released screw track id.
        :returns: ``(ok, message)`` — ok when seated head-up in a hole; message explains a miss.
        """
        try:
            self._poll_once()
        except Exception:  # noqa: BLE001 - verify from last belief when sensors fail
            pass
        es = self.ego_state()
        target = es.find(screw_id)
        if target is None:
            return False, f"unknown screw {screw_id!r} after place"
        with self._lock:
            track = self.estimator.tracks.get(screw_id)
            tip_source = track.tip_roll_source if track is not None else "unknown"
            tip = float(track.tip_roll) if track is not None else 0.0
        try:
            _mate, issues = self.esim._ensure_screw_seated_for_drive(es, target)
        except PredictionError as exc:
            return False, str(exc)
        if tip_source == "observed" and abs(tip) > 0.35:
            return False, (
                f"{screw_id} tipped over after place (tipRollDeg={math.degrees(tip):.0f}); "
                "re-pick and insert tip-down again")
        return True, issues[0] if issues else f"{screw_id} seated tip-down in hole"

    def _replan_pick(self, screw_id: str) -> dict | None:
        """Fresh pick plan for a screw that missed its hole (insert retry).

        :param screw_id: screw to grasp again.
        :returns: pick plan dict, or None when pick is infeasible.
        """
        receipt = self.esim.predict(self.ego_state(), "pick", {"target": screw_id})
        if not receipt.get("feasible"):
            return None
        return receipt.get("plan") or None

    def _replan_insert_place(self, insert: dict) -> dict | None:
        """Fresh tip-down insert place after re-picking a missed screw.

        Polls cameras and requires ``tipSource=observed`` on the held screw so the stand roll
        knows which end is the tip.

        :param insert: ``insertParams`` from the original plan (``object``, ``on``, optional ``hole``).
        :returns: insert plan dict, or None when insert predict is infeasible.
        """
        try:
            self._poll_once()
        except Exception:  # noqa: BLE001 - replan from last belief when sensors fail
            pass
        if self._insert_screw_tip_gate(insert.get("object")) is not None:
            return None
        params: dict = {"on": insert.get("on")}
        if insert.get("hole") is not None:
            params["hole"] = insert["hole"]
        if insert.get("maxAttempts") is not None:
            params["maxAttempts"] = insert["maxAttempts"]
        receipt = self.esim.predict(self.ego_state(), "insert_screw_into_hole", params)
        if not receipt.get("feasible"):
            return None
        return receipt.get("plan") or None

    def _unsupported(self, track_id: str) -> bool:
        """Whether the object floats where it is believed to be (:meth:`EgoEstimator.unsupported`)."""
        with self._lock:
            track = self.estimator.tracks.get(track_id)
            return track is not None and self.estimator.unsupported(track)

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

    def _naming(self) -> tuple[list[dict], dict[str, reflection.Evidence], dict[str, list[str]]]:
        """Summary objects, the perception evidence of each track and the segmenter prompt words per category."""
        objects = self.summary()["objects"]
        with self._lock:
            evidence = {}
            for tid, track in self.estimator.tracks.items():
                total = sum(track.category_counts.values())
                perceived = {k: v / total for k, v in track.category_counts.items()} if total else {}
                segmenter = ({k: v / track.semantic_views for k, v in track.semantic_counts.items()}
                             if track.semantic_views else {})
                evidence[tid] = reflection.Evidence(perceived, segmenter, track.label)
        words: dict[str, list[str]] = {}
        if self.segmenter is not None:
            for prompt, category in self.segmenter.config.vocabulary.items():
                words.setdefault(category, []).append(prompt)
        return objects, evidence, words

    def candidates(self, request: dict) -> dict:
        """Objects that could be what a name and approximate size refer to, with the naming evidence
        and doubts of each (see :mod:`steward.ego.reflection`).

        :param request: ``{name?, size?, exclude?, limit?}``; without ``name`` and ``size`` the
            doubtfully named objects.
        :returns: ``{candidates}``.
        :raises ValueError: malformed fields."""
        name, size, exclude, limit = (request.get(k) for k in ("name", "size", "exclude", "limit"))
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a non-empty string")
        if size is not None and (not isinstance(size, list) or not 1 <= len(size) <= 3
                                 or not all(isinstance(v, (int, float)) and v > 0 for v in size)):
            raise ValueError("size must list one to three positive dimensions in metres")
        if exclude is not None and (not isinstance(exclude, list) or not all(isinstance(e, str) for e in exclude)):
            raise ValueError("exclude must be a list of entity ids")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        objects, evidence, words = self._naming()
        return {"candidates": reflection.candidates(objects, evidence, words, self.config.reflection,
                                                    name, size, exclude or (), limit)}

    def reflect(self, failure: dict) -> dict:
        """Evidence for reflecting on a failed step: its cause, the objects it was about and
        candidates for the object the instruction meant (see :func:`steward.ego.reflection.reflect`).

        :param failure: ``{tool?, arguments?, message, name?, size?}``.
        :returns: ``{cause, sought, targets, missing, candidates}``; a ``reflection`` activity entry
            records it for the console.
        :raises ValueError: malformed fields."""
        if failure.get("arguments") is not None and not isinstance(failure["arguments"], dict):
            raise ValueError("arguments must be an object")
        size = failure.get("size")
        if size is not None and (not isinstance(size, list) or not all(isinstance(v, (int, float)) and v > 0 for v in size)):
            raise ValueError("size must list positive dimensions in metres")
        objects, evidence, words = self._naming()
        result = reflection.reflect(objects, evidence, words, self.config.reflection, failure)
        self._log("reflection", reflection.activity_text(failure, result))
        return result

    def capture_cameras(self) -> dict:
        """Capture the existing robot RGB views without moving or changing the perception loop.

        The packet contains sensor images, intrinsics, fixed camera-to-reference calibration
        and gripper kinematics computed from the same encoder sample. It does not provide an
        absolute camera pose.
        """
        packet = self.link.cameras()
        return {key: value for key, value in packet.items() if key != "joint_encoders"} | {"gripper_pose": self._gripper_pose(packet)}

    def gripper_pose(self) -> dict:
        """Current gripper-link pose relative to the base and the robot cameras, without imaging."""
        return self._gripper_pose(self.link.kinematics())

    def _gripper_pose(self, packet: dict) -> dict:
        """The pose of the gripper the arm works with, its joints read as the arm's virtual joints."""
        return gripper_pose(self.estimator.robot_model, packet, self.arm.geometry.gripper,
                            self.arm.joints_of(packet["joint_encoders"], {"x": 0.0, "y": 0.0, "yaw": 0.0}))

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
        :param px: grid size, 4 to 128.
        :returns: ``{entity, px, yawDeg, sizeM, coverage, sourcePx, spread, interiorSpread,
            interiorCoverage, rows, worldRows, worldTurnDeg, stamp}``, ``hint`` (:func:`view_hint`)
            when less than :data:`VIEW_HINT_COVERAGE` of the face is seen, and ``detailHint`` when the
            face's shorter side spans fewer than ``detail_px`` pixels of the camera image
            (``sourcePx``): a finer grid than that only interpolates. ``plain`` / ``interiorColors``
            tell a cardboard back from a printed face (lighting makes ``interiorSpread`` look high
            on a plain back; do not use spread alone).
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range."""
        if not 4 <= px <= 128:
            raise ValueError("px must be between 4 and 128")
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
        intrinsics = self.estimator.camera_intrinsics(frame["camera"])
        focal = intrinsics["height"] / (2 * math.tan(math.radians(intrinsics["fovy"]) / 2))
        cam_p = np.asarray(self.estimator.camera_pose(frame["header"]["odometry"], frame["camera"])["p"], float)
        side, distance = float(min(size[0], size[1])), float(np.linalg.norm(np.array([*centre[:2], top]) - cam_p))
        source_px = focal * side / max(distance, 1e-3)
        if source_px < self.config.detail_px:
            within = focal * side / self.config.detail_px
            hint["detailHint"] = (f"the face spans only {source_px:.0f} px of the {frame['camera']} image from "
                                  f"{distance:.2f} m; its picture is reliable within {within:.2f} m of the camera")
        return {"entity": entity, "px": px, "camera": frame["camera"], "yawDeg": round(math.degrees(yaw), 2),
                "sizeM": [round(float(v), 3) for v in size], "coverage": round(coverage, 3),
                "sourcePx": round(source_px, 1), **face_colour_stats(rows), **hint,
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
                                   "interiorSpread": round(known.interior_spread, 1),
                                   "interiorColors": known.interior_colors, "plain": known.plain,
                                   "stale": known.stale}}

    def match_face(self, entities: list[str], reference: str, px: int = 24) -> dict:
        """Where the top face of each of ``entities`` appears on the top face of ``reference``
        printed at the same scale (puzzle pieces on the picture of their box, tiles on a pattern),
        from the faces the robot remembers (:meth:`_look_at_faces`), refreshed first with one fresh
        RGB-D frame for those in view. Each face is resampled to ``px`` cells across its shorter
        side and the reference to cells of the same size, then compared by :func:`match_face`.

        :param entities: Ego-State ids of the objects whose faces are sought.
        :param reference: Ego-State id of the object showing the whole picture.
        :param px: cells across each object's shorter side, 4 to 32.
        :returns: ``{reference, referenceCoverage, results, stamp}``; each result holds ``entity``,
            ``cellM``, ``spread``, ``interiorSpread``, ``faceCoverage``, ``seenAgoS`` and ``matches``
            (``offsetM``: the face centre's place in the reference frame from its centre; ``turnDeg``:
            quarter turns counter-clockwise that align the face with the reference; ``placeYawDeg``:
            the object heading that does so with the reference's heading; ``score``: correlation, 1 =
            identical), or ``entity`` and ``error`` when its face is not known well enough.
        :raises KeyError: unknown object; :raises ValueError: ``px`` out of range or the reference
            face not known well enough."""
        return self.compute_face_match(self.prepare_face_match(entities, reference, px))

    def prepare_face_match(self, entities: list[str], reference: str, px: int = 24) -> FaceMatchSnapshot:
        """Refresh and copy observed faces at one input instant, without doing correlation.

        :param entities: nonempty Ego IDs to compare.
        :param reference: Ego ID with the reference print.
        :param px: cells across a piece's shorter side, 4 to 32.
        :returns: independent pixel/pose copies; later perception and motion cannot change them.
        :raises KeyError: unknown ID; :raises ValueError: invalid input or insufficient reference.
        """
        if not isinstance(entities, list) or not entities or not all(isinstance(e, str) for e in entities):
            raise ValueError("entities must be a non-empty list of object ids")
        if not 4 <= px <= 32:
            raise ValueError("px must be between 4 and 32")
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
                    faces[eid] = (known.rgb.copy(), track.size.copy(), round(est.stamp - known.stamp, 1),
                                  known.spread, known.interior_spread)
        return FaceMatchSnapshot(reference, ref_rgb, ref_size, ref_yaw, faces, list(entities), px, frame["header"]["stamp"])

    @staticmethod
    def compute_face_match(snapshot: FaceMatchSnapshot, cancelled: Callable[[], bool] | None = None) -> dict:
        """Correlate frozen observations without holding input, estimator or world-clock locks.

        :param snapshot: copied observations returned by :meth:`prepare_face_match`.
        :param cancelled: peer-disconnect check, or None to run to completion (including replay).
        :returns: matches in the reference pose at ``snapshot.stamp``, not its later pose.
        :raises CancelledError: cancellation observed between pieces/rotations; no partial result.
        """
        reference, ref_rgb, ref_size, ref_yaw = snapshot.reference, snapshot.rgb, snapshot.size, snapshot.yaw
        px = snapshot.px
        results = []
        for eid in snapshot.entities:
            face = snapshot.faces[eid]
            if cancelled is not None and cancelled():
                raise CancelledError("face matching cancelled")
            if isinstance(face, str):
                results.append({"entity": eid, "error": face})
                continue
            rgb, size, ago, spread, interior = face
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
            found = match_face(resample_face(rgb, ox, oy), resample_face(ref_rgb, rx, ry), cancelled=cancelled)
            matches = [{"offsetM": [round(float((m["col"] - rx / 2) * ref_size[0] / rx), 3),
                                    round(float((ry / 2 - m["row"]) * ref_size[1] / ry), 3)],
                        "turnDeg": 90 * m["turns"],
                        "placeYawDeg": round(math.degrees(wrap_angle(ref_yaw + m["turns"] * math.pi / 2)), 1),
                        "score": m["score"]} for m in found["matches"]]
            results.append({"entity": eid, "cellM": round(cell, 4), "spread": round(spread, 1),
                            "interiorSpread": round(interior, 1),
                            "faceCoverage": round(float(np.isfinite(rgb[..., 0]).mean()), 2), "seenAgoS": ago,
                            "matches": matches})
        return {"reference": reference, "referenceCoverage": round(float(np.isfinite(ref_rgb[..., 0]).mean()), 2),
                "results": results, "stamp": snapshot.stamp}

    def observe(self) -> dict:
        frame = self.link.sensors()
        cameras = self._cameras(frame)
        if frame.get("imaging") == "rgbd":
            visible = []
            for camera in cameras:
                whole = decode_rgbd(self.link.rgbd(camera))
                rgbd = decimated(whole, self._perception_rows(whole), self.config.depth_join)
                if rgbd["mask"] is None and not self._segment([(rgbd, whole)]):
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
            visible = []
            for d in frame["detections"]:
                if d.get("camera") not in cameras:
                    continue
                item = {"category": d["category"], "rangeM": round(d["range"], 2), "camera": d.get("camera")}
                if d.get("screwHoles"):
                    item["screwHoles"] = [{"id": h["id"], "p": list(h["p"])} for h in d["screwHoles"]]
                visible.append(item)
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
        if skill in ("drive_screw", "insert_screw_into_hole"):
            # Refresh cameras before trusting screw tip / head-up (same as ego_screw_holes).
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 - still predict from the last belief when sensors fail
                pass
        es = self.ego_state()
        if skill == "close":
            # The drawer the robot pulled open is closed by that pull (handle, axis, distance)
            # unless the caller names another: views of the open drawer box it deeper than it is.
            params = {**self.executor.pull_of(str(params.get("target", ""))), **params}
        if skill == "drive_screw":
            gate = self._drive_screw_tip_gate(params.get("target"))
            if gate is not None:
                receipt = self.esim._receipt(
                    es, "drive_screw", params, 1.0, [gate], {}, feasible=False)
                limit = self.config.max_risk
                receipt["withinRiskLimit"] = False
                receipt["maxRisk"] = limit
                with self._lock:
                    self.receipts[receipt["id"]] = receipt
                    self._receipt_generation[receipt["id"]] = self._generation
                return receipt
        if skill == "insert_screw_into_hole":
            gate = self._insert_screw_tip_gate(params.get("object") or params.get("target"))
            if gate is not None:
                receipt = self.esim._receipt(
                    es, "insert_screw_into_hole", params, 1.0, [gate], {}, feasible=False)
                limit = self.config.max_risk
                receipt["withinRiskLimit"] = False
                receipt["maxRisk"] = limit
                with self._lock:
                    self.receipts[receipt["id"]] = receipt
                    self._receipt_generation[receipt["id"]] = self._generation
                return receipt
        receipt = self.esim.predict(es, skill, params)
        limit = self.config.max_pick_risk if receipt.get("skill") == "pick" else self.config.max_risk
        receipt["withinRiskLimit"] = receipt["feasible"] and receipt["risk"] <= limit
        receipt["maxRisk"] = limit
        with self._lock:
            self.receipts[receipt["id"]] = receipt
            self._receipt_generation[receipt["id"]] = self._generation
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

    def forget(self, request: dict) -> dict:
        """Remove an Ego-State object the robot's agent judged a false perception: implausible where
        it is believed to be and not perceived again on a fresh look at its place.

        :param request: ``{entity, reason}``.
        :returns: the removal record as it appears in ``dropped``.
        :raises ValueError: malformed fields, or the object is held or was seen recently;
            :raises KeyError: no such object."""
        entity, reason = request.get("entity"), request.get("reason")
        if not isinstance(entity, str):
            raise ValueError("forget needs an entity id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("forget needs a reason: what made the object implausible and how it was looked for again")
        with self._lock:
            if entity not in self.estimator.tracks:
                raise KeyError(f"unknown object {entity!r}")
            self.estimator.forget(entity, reason.strip())
            self._publish_view_scene()
            return dict(self.estimator.dropped[-1])

    def receipt(self, receipt_id: str) -> dict:
        """A Prediction Receipt with the robot clock now, so a caller can see whether it has expired."""
        if receipt_id not in self.receipts:
            raise KeyError(f"unknown receipt {receipt_id!r}")
        receipt = dict(self.receipts[receipt_id])
        now = float(self.estimator.stamp)
        receipt["now"] = round(now, 3)
        receipt["expired"] = now > float(receipt["expiresAt"])
        return receipt

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
        ``difficulty``, ``state``, ``skill``, ``alert`` and ``reflection`` entries, newest last."""
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
        side = self.esim.lateral  # the arm's line is that far left of the base centre
        ox, oy = plan.get("graspOffset") or (0.0, 0.0)  # an edge grasp closes off the centre
        dock = [float(track.pos[0]) + ox - heading[0] * reach + heading[1] * side,
                float(track.pos[1]) + oy - heading[1] * reach - heading[0] * side]
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
        :returns: ``{"plan", "skill", "activity", "activityLast", "boot", "t"}`` (``activityLast`` is
            the newest id, lower than ``since`` after the runtime restarted; ``boot`` names this
            runtime process, so a client tells a new process from the one it was reading).
        """
        running = next((j for j in reversed(list(self.executor.jobs.values())) if j.state == "running"), None)
        with self._lock:
            plan = dict(self.plan) if self.plan else None
        with self._lock:
            newest = self.activity[-1]["id"] if self.activity else 0
        return {"plan": plan, "skill": running.to_dict() if running else None, "activity": self.activity_since(since),
                "activityLast": newest, "boot": self.boot, "t": round(self.estimator.stamp, 2)}

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
        """Look around in place. Rotation without translation needs no receipt.

        When ``MotionConfig.inspect_closed_drawers`` is set, each closed drawer is opened, looked
        into, and closed afterwards. Those visits are planned from the ego state at the time and
        skipped when they are not feasible."""
        self._check_control(owner)
        return self.executor.start("look_around", {"id": "look-around", "plan": {}}).to_dict()

    def _closed_drawer_ids(self) -> list[str]:
        """Closed pullable drawers already seen enough times to open, nearest first."""
        with self._lock:
            est = self.estimator
            ox, oy = float(est.odometry.get("x", 0.0)), float(est.odometry.get("y", 0.0))
            found = [t for t in est.tracks.values()
                     if t.category == "drawer" and "pullable" in t.affordances
                     and t.existence >= 0.5 and t.observations >= 3
                     and est.drawer_opening(t) < 0.08]
            found.sort(key=lambda t: math.hypot(float(t.pos[0]) - ox, float(t.pos[1]) - oy))
            return [t.id for t in found]

    def _drawer_plan(self, skill: str, target: str, distance: float | None = None,
                     handle: list[float] | None = None, axis: list[float] | None = None) -> tuple[dict | None, str]:
        """Plan for opening or closing ``target``, or why that plan cannot be run.

        ``distance`` is how far a close must travel when the track no longer shows the opening
        that was just pulled. ``handle`` and ``axis`` are that pull, so the close does not follow
        a box the look has shifted."""
        from steward.ego.esim import PredictionError
        params: dict = {"target": target}
        if distance is not None:
            params["distance"] = distance
        if handle is not None:
            params["handle"] = handle
        if axis is not None:
            params["axis"] = axis
        try:
            receipt = self.predict(skill, params)
        except PredictionError as exc:
            return None, str(exc)
        plan = receipt.get("plan")
        if not receipt.get("feasible") or not plan:
            return None, "; ".join(receipt.get("issues") or ["not feasible"])
        receipt["used"] = True
        if receipt.get("withinRiskLimit"):
            return plan, ""
        return plan, "; ".join(receipt.get("issues") or ["risk above the limit"])

    def job(self, job_id: str) -> dict:
        if job_id not in self.executor.jobs:
            raise KeyError(f"unknown job {job_id!r}")
        return self.executor.jobs[job_id].to_dict()

    def wait(self, job_id: str, timeout: float = 120.0) -> dict:
        return self.executor.wait(job_id, timeout).to_dict()

    def stop_motion(self) -> dict:
        return self.executor.stop()
