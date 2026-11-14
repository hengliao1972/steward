"""Ego-State estimation: fuse detections into robot-owned object tracks and publish the ES.

Each detection is transformed from the camera mount frame into the robot's ``map`` frame with
the robot's own odometry and body model, associated to an existing track (nearest compatible
category within a gate) and fused with a per-axis Kalman update, so position variance shrinks
as observations accumulate. Unmatched detections start new tracks with robot-assigned ``e:``
ids. A track that should have been visible but was not detected loses existence probability
and is dropped below a threshold.

Phase 0 assumption: the ``map`` frame coincides with the world frame because odometry is
nearly ideal; the Observer still estimates the alignment explicitly.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from steward.ego.classify import (OVER, SHAPES, _color_score, classify, clear_stem_name, content_color_name,
                                  fuse_respecting_color, max_long_extent, max_plan_aspect, partial_pistol,
                                  part_semantics, perfume_without_bottle_label, prototype, shape_over_bottle,
                                  sideboard_not_counter)
from steward.ego.perception import (
    _largest_compact, box_at_yaw, box_of, oversized_bowl_core, raised_object_points, segments,
)
from steward.ego.planner import Box2D, front_box, protrusion, room_centre
from steward.ego.priors import SCREW_HOLE_FRAMES, compatible, default_affordances, prior, screw_holes
from steward.ego.assembly_memory import AssemblyMemory
from steward.ego.stationary_assembly import StationaryAssemblies
from steward.ego.hole_registration import assembly_posture, register_hole_template, coherent_interface_box
from steward.math3d import (matrix_from_quat, pose_compose, pose_inverse, quat_from_axis_angle, quat_from_yaw,
                            quat_mul, quat_rotate, wrap_angle, yaw_from_quat)
from steward.scene.schema import Entity, Est, Link, Relation, RigidTree, SceneState, primitive

SELF_ID = "e:self"
ASSEMBLY_PARTS = {"toy_screw", "airplane_fuselage", "airplane_wing", "airplane_propeller",
                  "airplane_landing_gear", "car_body", "car_wheel", "electric_screwdriver"}


@dataclass
class Appearance:
    """What the top face of an object looked like: colours on a fixed grid of cells in the object's
    frame (column = +x, row 0 = +y), each from the latest view that saw it. ``stale`` once the object
    was put down again (it may have been turned over): the next view replaces the whole face."""

    rgb: np.ndarray  # (rows, cols, 3), NaN where no view has seen the cell
    size: tuple[float, float]  # the object's extent along x and y when the grid was laid out
    stamp: float  # world time of the latest view
    stale: bool = False

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.rgb[..., 0]).mean())

    @property
    def spread(self) -> float:
        """Colour standard deviation of the seen cells; near 0 the face is plain."""
        seen = self.rgb[np.isfinite(self.rgb[..., 0])]
        return float(seen.std()) if len(seen) else 0.0

    @property
    def interior_spread(self) -> float:
        """Colour standard deviation of seen cells excluding a one-cell border: edge cells often
        mix in a neighbour's colour when pieces lie packed. Lighting still inflates this on a
        plain back; prefer :attr:`interior_colors` / :attr:`plain` to tell face-down from face-up."""
        h, w = self.rgb.shape[:2]
        margin = 1 if min(h, w) > 2 else 0
        region = self.rgb[margin:h - margin or None, margin:w - margin or None] if margin else self.rgb
        seen = region[np.isfinite(region[..., 0])]
        return float(seen.std()) if len(seen) else 0.0

    @property
    def interior_colors(self) -> int:
        """Distinct interior colours after 16-level quantisation; a plain back stays near 1–4."""
        from steward.ego.perception import FACE_COLOR_STEP
        h, w = self.rgb.shape[:2]
        margin = 1 if min(h, w) > 2 else 0
        region = self.rgb[margin:h - margin or None, margin:w - margin or None] if margin else self.rgb
        seen = region[np.isfinite(region[..., 0])]
        if len(seen) == 0:
            return 0
        step = FACE_COLOR_STEP
        q = (seen // step).astype(int)
        return len({(int(c[0]), int(c[1]), int(c[2])) for c in q})

    @property
    def plain(self) -> bool:
        """True when the interior looks like a uniform back (few distinct colours)."""
        from steward.ego.perception import FACE_PLAIN_COLORS
        return self.interior_colors <= FACE_PLAIN_COLORS


@dataclass
class Track:
    id: str
    category_counts: dict[str, float]
    pos: np.ndarray
    var: np.ndarray
    yaw: float
    yaw_var: float
    size: np.ndarray
    observations: int
    first_seen: float
    last_seen: float
    existence: float = 0.5
    held: bool = False
    held_offset: np.ndarray | None = None  # while held: its centre from the pinch point, in the gripper's frame
    color_counts: dict[str, float] = field(default_factory=dict)
    voxels: np.ndarray | None = None  # 1 cm voxels of every surface seen of this object (RGB-D mode)
    span: np.ndarray | None = None  # corners of each view's full-resolution box; the voxel cloud is only a sample
    completion: float = 0.0  # how far the box was extended beyond what was seen (shape prior)
    seen_across: int = 0  # bit 0/1 once that plan axis was seen within SEEN_ACROSS_DOT of broadside
    settled: bool = True  # False right after the robot put it down, until it is seen again
    mean_rgb: np.ndarray | None = None  # running mean colour of its regions (RGB-D mode)
    mean_invalid: float = 0.0  # running mean share of its pixels without depth
    bottom_seen: bool = False  # some view showed its base (not cut by the image edge or hidden)
    label: str | None = None  # a category named by the robot's agent; it outranks the classifier
    annotated_affordances: list[str] | None = None  # affordances named with it, in place of the category prior
    # Screw holes last seen on this object (body frame), from detections; None until perception reports any.
    screw_holes: list[dict] | None = None
    held_yaw: float = 0.0  # while held: its heading relative to the gripper link's heading
    grasp_world_yaw: float | None = None  # estimated frame before transport, not a measured directed heading
    grasp_association: dict | None = None  # frozen perception before the load follows the gripper
    held_up: np.ndarray | None = None  # while held: the gripper-link axis that was vertical at the grasp
    carried_offset: np.ndarray | None = None  # while inside the held container: its offset from that container, in the gripper's frame
    yaw_conflicts: int = 0  # views in a row at a clearly different heading from its settled one
    absent_views: list[tuple[float, float, float, float]] = field(default_factory=list)
    # (t, camera x, camera y, camera yaw) of views that saw through its place since it was last observed
    recent: collections.deque = field(default_factory=lambda: collections.deque(maxlen=RECENT_VIEWS))
    # (centre, size) of its latest single-view measurements, for the robust box of small objects
    appearance: Appearance | None = None
    semantic_counts: dict[str, float] = field(default_factory=dict)
    # summed segmenter scores per category over the views that named it (robot-side segmentation)
    semantic_views: int = 0  # views whose region the segmenter named
    pinned: bool = False  # an active manipulation protects this estimate from occluding masks
    pin_observed_age: float | None = None  # freshness when protection began, not a new observation
    front: np.ndarray | None = None  # drawers: map-frame xy of the front panel in the latest fused view
    front_seen: float = -1.0  # stamp of that view
    # A release prediction and the independent RGB-D measurement after it. Never fused into
    # geometry or copied across track merges: uncertain identity must remain unknown.
    placement: dict | None = None
    assembly_observation: dict | None = None
    interface_axis_consistent: bool | None = None
    interface_views: collections.deque = field(default_factory=lambda: collections.deque(maxlen=48))
    interface_sample_status: dict = field(default_factory=dict)
    interface_best_near: dict | None = None
    interface_estimate_at: float | None = None
    grasp_estimate_pending: bool = False
    grasp_estimate_correction: dict | None = None
    interface_motion_at: float = -math.inf
    interface_last_geometry: dict | None = None
    manipulation_view: dict | None = None  # independent full-resolution RGB-D sample
    tool_feature: dict | None = None
    held_tool_feature: dict | None = None  # measured at grasp, in actual gripper-link frame
    held_interface: dict | None = None  # grasp-time component geometry in the TCP frame

    @property
    def category(self) -> str:
        return self.label or self.perceived_category

    @property
    def perceived_category(self) -> str:
        """The category perception accumulated, regardless of a label: what detections of the same
        object will keep reporting, so association compares with it."""
        if not self.category_counts:
            return "unknown"
        return max(self.category_counts, key=self.category_counts.get)

    @property
    def carried(self) -> bool:
        """Whether it lies in the container the robot holds and moves with it."""
        return self.carried_offset is not None

    @property
    def affordances(self) -> list[str]:
        """Those its agent named, else what common sense gives its category and size."""
        if self.annotated_affordances is not None:
            return list(self.annotated_affordances)
        return default_affordances(self.category, self.footprint, self.bottom(), self.top())

    @property
    def color(self) -> str | None:
        return max(self.color_counts, key=self.color_counts.get) if self.color_counts else None

    @property
    def footprint(self) -> float:
        return float(max(self.size[0], self.size[1]))

    @property
    def sigma(self) -> float:
        return float(math.sqrt(float(np.mean(self.var[:2]))))

    def top(self) -> float:
        return float(self.pos[2] + self.size[2] / 2)

    def bottom(self) -> float:
        return float(self.pos[2] - self.size[2] / 2)


@dataclass
class _HandleHold:
    """A drawer handle in the gripper: the drawer and its contents move with the gripper
    (:meth:`EgoEstimator._follow_handle`) until the gripper lets go."""

    drawer_id: str
    grip: np.ndarray  # pinch point at the last frame, map frame
    travel: np.ndarray  # how far the hold has carried the drawer so far
    opening: float  # how far the drawer stuck out when the hold began
    pinned: set[str]  # tracks this hold pinned, to unpin when it ends
    front: np.ndarray | None  # the drawer's observed front panel (:attr:`Track.front`) when the hold began


@dataclass
class _ShutFollow:
    """A drawer the robot pushed shut: its contents are placed by the panel's observed travel.

    The push is measured at the base, and the drawer slides on after the gripper lets go. The
    front panel, seen again once shut, says how far it really went
    (:meth:`EgoEstimator._follow_shut_front`)."""

    front_at_grasp: np.ndarray  # the front panel before the push, map frame xy
    push: np.ndarray  # unit xy direction of the push
    applied: float  # travel the contents have been moved by so far
    contents: list[str]  # the tracks that rode in the drawer
    since: float  # stamp of the push; only front views after it count


@dataclass
class EstimatorConfig:
    hole_registration_extent_ratio: float = 1.4  # outline measurement compatibility, not task tolerance
    hole_registration_square_ratio: float = .9  # below this, long/short axes are distinguishable
    hole_registration_extent_slack_m: float = 0.03  # absolute outline error of a segmented small part
    placement_axis_noise_deg: float = 5.0  # measurement repeatability, not a task success tolerance
    placement_extent_ratio: float = 1.4  # reject partial/merged/tilted shapes before suggesting a turn
    placement_observation_max_age_s: float = 2.0  # stale measurements cannot prescribe a new motion
    pos_sigma_base: float = 0.01
    pos_sigma_per_m: float = 0.02
    pos_sigma_rows: int = 240  # image rows the pos_sigma_* figures hold for: a camera with more rows localises finer in proportion
    process_noise: float = 1e-6
    gate_m: float = 0.30
    size_gate_ratio: float = 0.25
    size_match_log: float = 0.15
    nested_bowl_log: float = math.log(1.4)
    """Footprint log-ratio above which two bowls are different objects. Nesting sizes step by 1.5."""
    voxels_max: int = 6000
    points_per_cloud: int = 150
    missed_decay: float = 0.97
    """Existence factor per view for a young track (fewer than ESTABLISHED_VIEWS observations) that
    should be in view but was not segmented."""
    absence_free_share: float = 0.9
    """Share of a track's depth probes that must lie beyond its box for a view to count as free space."""
    absence_interval_s: float = 0.5
    """Minimum world time between two counted free-space views of one track."""
    absence_views: int = 3
    """Free-space views that drop a young track."""
    absence_views_established: int = 8
    """Free-space views that drop an established track."""
    absence_viewpoints: int = 2
    """Clearly different camera poses among the free-space views that drop an established track."""
    viewpoint_spread_m: float = 0.3
    viewpoint_spread_deg: float = 20.0
    drop_below: float = 0.05
    max_range: float = 8.0
    yaw_confirm_views: int = 3
    """Views in a row at a heading more than YAW_MOVED from a settled track's that turn the track:
    fewer are taken for detections of a neighbour lying at another heading."""
    lost_content_s: float = 4.0
    """Seconds a track carried in the held container may go unseen while the container is seen
    before it is taken to have fallen out (:meth:`EgoEstimator._drop_lost_contents`)."""
    poured_roll: float = 1.5
    """Tilt (rad) of the held container from how it was taken beyond which its contents have slid
    out: their tracks are dropped, and the contents are perceived afresh where they land. Below the
    least tip a pour uses (ESim's ``pour_roll_min``); a walled box holds its contents until it is
    nearly on its side (measured: at 1.5 rad nothing came out)."""
    rest_gap_m: float = 0.03
    """Largest gap (m), beyond the measurement uncertainty, between a newly seen object's bottom and
    the floor, or the top of a track or detection below it, at which it is taken to rest there; one
    clear of everything is in flight and
    gets no track until it is seen where it lands (:meth:`EgoEstimator._newcomer`)."""
    content_margin_m: float = 0.02
    """Margin (m) around the held container within which a newly seen object is taken to lie in it."""
    grasp_shift_m: float = 0.01
    """Standard deviation (m) of where a grasp leaves the load in the hand relative to where the
    estimate put it before the grasp: the uncertainty views of the load in the gripper start from
    (:meth:`EgoEstimator._refine_held`)."""
    held_view_gate_m: float = 0.05
    """Largest distance (m) between a detection and the held load's predicted position at which it
    corrects where the load sits in the gripper; a detection as near another track corrects nothing."""
    forget_unseen_s: float = 3.0
    """Seconds a track must have gone unperceived before the agent may forget it (:meth:`EgoEstimator.forget`)."""
    wall_like_top: float = 1.25
    """Robot knowledge of the home: nothing but walls reaches this high, so a region topping it is
    part of a wall whatever its own classification says, and a non-wall track this tall is not real."""
    semantic_shape_weight: float = 0.5
    """With robot-side segmentation: exponent of the shape-prior probability (:func:`fuse_semantics`)."""
    semantic_label_floor: float = 0.03
    """With robot-side segmentation: semantic likelihood of a category the segmenter did not name."""
    semantic_named_score: float = 0.6
    """With robot-side segmentation: mean segmenter score of a non-wall category, over at least
    NAMED_VIEWS views, from which a track counts as a real object however hollow it looks."""
    body_margin_m: float = 0.04
    """Margin around the robot's own links (:meth:`EgoEstimator.body_boxes`) within which depth
    points are the robot itself; world masks leave the robot out, robot-side segmentation does not."""
    body_share: float = 0.5
    """Share of a region's points on the robot's own links that makes the whole region the robot."""


@dataclass
class EgoEstimator:
    robot_model: dict
    config: EstimatorConfig = field(default_factory=EstimatorConfig)

    def __post_init__(self) -> None:
        self.tracks: dict[str, Track] = {}
        self.assembly_memory = AssemblyMemory()
        self.stationary_assemblies = StationaryAssemblies()
        self.manipulation_envelopes = {}
        self.flip_capture = None
        self.rigid_carry = {}  # perceptually corroborated members, relative to the held gripper
        self.fastening_attempts = {}  # robot actions and sensory evidence; never mechanical truth
        self.released_interfaces = {}  # immutable release FK, independent of track merging
        self.fastener_origins = {}  # perceived pickup spots, discarded with the sensor session
        self._assembly_views = {}
        self._ids = itertools.count(1)
        self.dropped: collections.deque = collections.deque(maxlen=500)
        self._shed: list[np.ndarray] = []
        self._opened_drawers: set[str] = set()
        self._pulled_open: set[str] = set()
        """Drawers the robot pulled out and has not pushed back (:meth:`shift`)."""
        self._pushed_shut: set[str] = set()
        """Drawers the robot pushed back and has not pulled out again (:meth:`shift`)."""
        self._handle: _HandleHold | None = None
        """The drawer whose handle the gripper holds now (:meth:`_follow_handle`)."""
        self._handle_travel: dict[str, tuple[np.ndarray, float, np.ndarray | None]] = {}
        """Per drawer: how far the last handle hold carried it, which :meth:`shift` subtracts, how
        far it stuck out when that hold began, and where its front panel was then."""
        self._shut_follow: dict[str, _ShutFollow] = {}
        """Drawers pushed shut whose contents still follow the panel's observed travel."""
        self.content_hits: list[tuple] = []
        """The most recent removals: ``{t, id, category, observations, reason, position, size, robot, into?}``."""
        self.version = 0
        self.stamp = 0.0
        self.perception_cycles = 0
        self.perception_stamps = collections.deque(maxlen=64)
        self.held_since = 0.0  # body time the current load was grasped at
        self.posed = False
        self.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
        self.joints: dict[str, float] = {}
        self.gripper = {"state": "open", "holding": False}
        self.grippers: dict[str, dict] = {}
        self.held_id: str | None = None
        self.held_gripper: str | None = None
        self._rng = np.random.default_rng(0)
        self._self_template = self._make_self_template()
        self.cameras = self._camera_models()
        self.primary_camera = next(iter(self.cameras))

    # ------------------------------------------------------------- body model
    def _make_self_template(self) -> dict:
        data = copy.deepcopy(self.robot_model)
        data["id"] = SELF_ID
        data.pop("asset", None)
        return data

    def _camera_models(self) -> dict[str, tuple[str, dict]]:
        """Sensor id → (id of the link carrying it, sensor description) of every camera, in model
        order; the first is the primary camera. The link's pose comes from the tree walk, so a camera
        on a lift or an arm follows the joint encoders."""
        tree = self.robot_model["body"]
        cameras = {}
        for link in tree["links"]:
            for sensor in link.get("sensors", []):
                if sensor["kind"] in ("rgb", "depth", "rgbd"):
                    cameras[sensor["id"]] = (link["id"], sensor)
        if not cameras:
            raise ValueError("robot model has no camera")
        return cameras

    # ------------------------------------------------------- generic kinematics
    def _tree(self) -> tuple[dict[str, list[tuple[dict, str]]], str]:
        """The robot's joint graph: children[parent] = [(joint, child link)] and the root link.

        Any link's pose is one walk from the root; that is what lets a wrist camera move with the
        arm, instead of the fixed head-mount assumption the mobile robot used."""
        cached = getattr(self, "_joint_tree", None)
        if cached is not None:
            return cached
        tree = self.robot_model["body"]
        children: dict[str, list[tuple[dict, str]]] = {}
        child_ids = set()
        for joint in tree["joints"]:
            children.setdefault(joint["parent"], []).append((joint, joint["child"]))
            child_ids.add(joint["child"])
        root_id = tree["root"]
        if isinstance(root_id, dict):
            root_id = root_id.get("value")
        if root_id not in children and root_id not in child_ids:
            roots = [link["id"] for link in tree["links"] if link["id"] not in child_ids]
            root_id = roots[0]
        self._joint_tree = (children, root_id)
        return self._joint_tree

    def link_pose(self, link_id: str, joints: dict | None = None, odometry: dict | None = None) -> dict:
        """Pose of any link in the world frame, from joint encoders.

        ``joints`` overrides the estimator's own last-seen encoders: the Ego-State view asks for
        the pose the arm is in *now*, which is a read, not a perception update. ``odometry`` does
        the same for where the base stands: a read of the body now needs both, or the arm is put
        where it is now on a base where it stood at the last look."""
        children, root_id = self._tree()
        path: list[tuple[dict, str]] = []
        found = self._path_to(children, root_id, link_id, path, set())
        if not found:
            raise KeyError(f"robot model has no link {link_id!r} reachable from {root_id!r}")
        q = self.joints if joints is None else joints
        pose = self.base_pose(odometry)
        for joint, _ in path:
            pose = _apply_joint(pose, joint, q.get(joint["id"], 0.0))
        return pose

    def _path_to(self, children, here: str, target: str, path: list, seen: set[str]) -> bool:
        if here == target:
            return True
        if here in seen:
            return False
        seen.add(here)
        for joint, child in children.get(here, []):
            path.append((joint, child))
            if self._path_to(children, child, target, path, seen):
                return True
            path.pop()
        return False

    def camera_poses(self) -> dict[str, dict]:
        """World pose of every camera the robot carries, keyed by sensor id."""
        return {camera: self.camera_pose(camera=camera) for camera in self.cameras}

    def base_pose(self, odometry: dict | None = None) -> dict:
        o = self.odometry if odometry is None else odometry
        return {"p": [o["x"], o["y"], 0.0], "q": quat_from_yaw(o["yaw"])}

    def camera_pose(self, odometry: dict | None = None, camera: str | None = None) -> dict:
        """World pose of a camera mount (looks along +x; ``camera`` None: the primary one), at the
        current odometry or a given one. Raises ``KeyError`` for a camera the robot model lacks.

        The body's own kinematics place every sensor, which is what a body whose cameras ride on a
        lift or its arms (the twin) needs. With ``odometry`` the same camera is carried to a robot
        standing there, keeping the current joint state."""
        link_id, sensor = self.cameras[camera or self.primary_camera]
        pose = pose_compose(self.link_pose(link_id), sensor["mount"])
        if odometry is None:
            return pose
        base = {"p": [odometry["x"], odometry["y"], 0.0], "q": quat_from_yaw(odometry["yaw"])}
        in_base = pose_compose(pose_inverse(self.base_pose()), pose)
        return pose_compose(base, in_base)

    def camera_intrinsics(self, camera: str | None = None) -> dict:
        """``width``, ``height`` and ``fovy`` (degrees) of a camera (None: the primary one)."""
        return self.cameras[camera or self.primary_camera][1]["intrinsics"]

    def on_body(self, points: np.ndarray) -> np.ndarray:
        """Which of ``points`` (N×3, map frame) lie on the robot's own links (within
        ``body_margin_m``); while the gripper holds something, none near the gripper, which are the load."""
        inside = _in_boxes(points, self.body_boxes(), self.config.body_margin_m)
        if self.gripper.get("holding"):
            held = self.tracks.get(self.held_id) if self.held_id else None
            reach = max(HELD_FALLBACK_M, held.footprint if held is not None else 0.0)
            inside &= np.linalg.norm(points - self.gripper_position(self.held_gripper), axis=1) > reach
        return inside

    def on_body_at(self, points: np.ndarray, header: dict) -> np.ndarray:
        """:meth:`on_body` for the body as a sensor header reports it, before its frame is fused.

        :param points: N×3 map-frame points.
        :param header: sensor header with ``odometry``, ``joints`` and the gripper state.
        :returns: boolean mask of the points on the robot's own links."""
        kept = (self.odometry, self.joints, self.gripper, self.grippers)
        self.odometry, self.joints = header["odometry"], header["joints"]
        try:
            self._read_grippers(header)
            return self.on_body(points)
        finally:
            self.odometry, self.joints, self.gripper, self.grippers = kept

    def body_reach(self, point) -> float:
        """Distance from ``point`` (map frame) within which every point :meth:`on_body` accepts lies.

        :param point: (3,) map-frame position, a camera's for instance.
        :returns: metres to the farthest corner of the robot's links, plus ``body_margin_m``."""
        at = np.asarray(point, float)
        return self.config.body_margin_m + max(
            float(np.linalg.norm(centre - at) + np.linalg.norm(half)) for centre, half, _rot in self.body_boxes())

    def body_boxes(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(centre, half extents, rotation matrix)`` in the map frame of every visual primitive of
        the robot's own links, from odometry and joint encoders."""
        tree = self.robot_model["body"]
        children = {j["child"] for j in tree["joints"]}
        poses = {link["id"]: self.base_pose() for link in tree["links"] if link["id"] not in children}
        pending = list(tree["joints"])
        while pending:
            joint = next((j for j in pending if j["parent"] in poses), None)
            if joint is None:
                raise ValueError(f"robot model joints {[j['id'] for j in pending]} hang from no reachable link")
            pending.remove(joint)
            kind = joint["type"]["value"]
            q = float(self.joints.get(joint["id"], (joint["q"]["value"] or [0.0])[0]))
            axis = joint.get("axis", {}).get("value", [0.0, 0.0, 1.0])
            motion = {"p": [a * q for a in axis], "q": [1.0, 0.0, 0.0, 0.0]} if kind == "prismatic" else \
                {"p": [0.0, 0.0, 0.0], "q": quat_from_axis_angle(axis, q) if kind == "revolute" else [1.0, 0.0, 0.0, 0.0]}
            poses[joint["child"]] = pose_compose(pose_compose(poses[joint["parent"]], joint["origin"]["value"]), motion)
        boxes = []
        for link in tree["links"]:
            for geom in link.get("geometry", {}).get("visual", []):
                size = geom.get("size", [])
                half = {"box": lambda s: [s[0] / 2, s[1] / 2, s[2] / 2], "cylinder": lambda s: [s[0], s[0], s[1] / 2],
                        "capsule": lambda s: [s[0], s[0], s[1] / 2 + s[0]], "sphere": lambda s: [s[0]] * 3}.get(geom.get("shape"))
                if half is None:
                    continue
                world = pose_compose(poses[link["id"]], geom.get("pose") or {"p": [0, 0, 0], "q": [1, 0, 0, 0]})
                boxes.append((np.asarray(world["p"], float), np.asarray(half(size), float), matrix_from_quat(world["q"])))
        return boxes

    def _camera_link(self) -> str:
        return self.cameras[self.primary_camera][0]

    def _grasp_link(self, gripper_link: str | None) -> str:
        if gripper_link is not None:
            return gripper_link
        return next(link["id"] for link in self.robot_model["body"]["links"]
                    if any((a if isinstance(a, str) else a.get("value")) == "grasp-point" for a in link.get("affordances", [])))

    def gripper_rotation(self, gripper_link: str | None = None, joints: dict | None = None,
                         odometry: dict | None = None) -> np.ndarray:
        """World rotation matrix of a gripper link, from the robot's own kinematics."""
        return matrix_from_quat(self.link_pose(self._grasp_link(gripper_link), joints, odometry)["q"])

    def gripper_position(self, gripper_link: str | None = None, joints: dict | None = None,
                         odometry: dict | None = None) -> np.ndarray:
        """Pinch point of a gripper, from the robot's own kinematics."""
        tree = self.robot_model["body"]
        gripper_link = self._grasp_link(gripper_link)
        pose = self.link_pose(gripper_link, joints, odometry)
        link = next(link for link in tree["links"] if link["id"] == gripper_link)
        for affordance in link.get("affordances", []):
            value = affordance.get("value") if isinstance(affordance, dict) else affordance
            if isinstance(value, dict) and "tcp-offset" in value:
                return np.asarray(pose_compose(pose, value["tcp-offset"])["p"])
        return np.asarray(pose["p"])

    # ----------------------------------------------------------------- update
    def update(self, frame: dict) -> None:
        self._note_perception(frame['stamp'])
        self.stamp = float(frame["stamp"])
        self.posed = True
        self.odometry = frame["odometry"]
        self.joints = frame["joints"]
        self._read_grippers(frame)
        if not self.gripper.get("holding"):
            self._clear_held()
        # A held object moves with this frame's body pose; update it before association so a
        # turning robot does not see its own load as a new object.
        self._follow_held()
        self._follow_handle()
        cameras = [c["id"] for c in frame.get("cameras", []) if c["id"] in self.cameras] or [self.primary_camera]
        cams = {c: self.camera_pose(camera=c) for c in cameras}
        measurements = []
        for det in frame.get("detections", []):
            if det.get("camera", self.primary_camera) not in cams:
                continue
            world = pose_compose(cams[det.get("camera", self.primary_camera)], det["pose"])
            rng = float(det.get("range", np.linalg.norm(det["pose"]["p"])))
            rows = self.camera_intrinsics(det.get("camera", self.primary_camera))["height"]
            sigma = (self.config.pos_sigma_base + self.config.pos_sigma_per_m * rng) * self.config.pos_sigma_rows / rows
            measurements.append((det, np.asarray(world["p"]), yaw_from_quat(world["q"]), sigma))
        # Global nearest-first assignment: of all detection-track pairs within the gate, the closest
        # pair is joined first, so of two objects side by side neither takes the other's track. An
        # object yields at most one detection per camera, so a track takes one from each camera.
        pairs = []
        for k, (det, pos, _yaw, sigma) in enumerate(measurements):
            for track in self._candidates(det["category"], pos, sigma, float(max(det["size"][0], det["size"][1]))):
                pairs.append((float(np.linalg.norm(track.pos - pos)), k, track.id))
        camera_of = [det.get("camera", self.primary_camera) for det, _p, _y, _s in measurements]
        seen_by: dict[str, set[str]] = {c: set() for c in cams}
        assigned: dict[int, str] = {}
        for _d, k, tid in sorted(pairs):
            if k not in assigned and tid not in seen_by[camera_of[k]]:
                assigned[k] = tid
                seen_by[camera_of[k]].add(tid)
        boxes = [(pos, np.asarray(det["size"], float), yaw) for det, pos, yaw, _sigma in measurements]
        for k, (det, pos, yaw, sigma) in sorted(enumerate(measurements), key=lambda item: item[1][3]):
            if k in assigned:
                self._fuse(self.tracks[assigned[k]], det, pos, yaw, sigma)
                continue
            # Another camera's view of an object first seen in this very frame joins the track its
            # first view started.
            born = [t for t in self._candidates(det["category"], pos, sigma, float(max(det["size"][0], det["size"][1])))
                    if t.first_seen == self.stamp and t.id not in seen_by[camera_of[k]]]
            if born:
                twin = min(born, key=lambda t: float(np.linalg.norm(t.pos - pos)))
                self._fuse(twin, det, pos, yaw, sigma)
                seen_by[camera_of[k]].add(twin.id)
                continue
            entry = self._newcomer(pos, np.asarray(det["size"], float), sigma, boxes[:k] + boxes[k + 1:])
            if entry == "skip":
                continue
            if entry == "carried":
                # Contents slide about in a container being carried and tipped: a view of one is first
                # taken for a known content beyond the association gate, not for another object.
                slid = [t for t in self.tracks.values()
                        if t.carried and t.id not in seen_by[camera_of[k]] and compatible(t.category, det["category"])]
                if slid:
                    nearest = min(slid, key=lambda t: float(np.linalg.norm(t.pos - pos)))
                    self._fuse(nearest, det, pos, yaw, sigma)
                    seen_by[camera_of[k]].add(nearest.id)
                    continue
            track = self._new_track(det, pos, yaw, sigma)
            if entry == "carried":
                track.carried_offset = self.gripper_rotation(self.held_gripper).T @ (track.pos - self.tracks[self.held_id].pos)
            seen_by[camera_of[k]].add(track.id)
        matched = set().union(*seen_by.values())
        self._decay_unseen(cams, matched)
        self._drop_lost_contents(matched)
        self._merge_duplicates(list(seen_by.values()))
        self._follow_held()
        self.version += 1

    def _drop_lost_contents(self, matched: set[str]) -> None:
        """What was in the carried container but has not been seen in it for ``lost_content_s``
        while the container itself is in view has fallen out on the way: it is forgotten, and
        perceived afresh where it lies."""
        container = self.tracks.get(self.held_id) if self.held_id else None
        if container is None or container.id not in matched:
            return
        for track in [t for t in self.tracks.values() if t.carried]:
            if self.stamp - track.last_seen > self.config.lost_content_s:
                self._drop(track, f"no longer seen in {container.id}: fell out while it was carried")

    def update_body(self, header: dict) -> None:
        """Follow the robot's own state (stamp, odometry, joints, gripper) from a sensor header
        without imaging: between two segmented frames, what the gripper holds moves with it."""
        self.stamp = float(header["stamp"])
        self.posed = True
        self.odometry = header["odometry"]
        self.joints = header["joints"]
        self.gripper = header["gripper"]
        if not self.gripper.get("holding"):
            self._clear_held()
        self._follow_held()
        self._follow_handle()
        self.version += 1

    def category_probs(self, size: np.ndarray, bottom: float, color: np.ndarray, invalid_frac: float,
                       bottom_seen: bool, semantics: dict[str, float] | None) -> dict[str, float]:
        """Category probabilities of a measured box: the shape priors alone, or fused with the
        segmenter's category scores when it named the region.

        Short names are not dropped here. A cabinet face narrower than 0.3 m is often labelled as
        a shorter object; dropping that label changes which track the face joins and the cabinet
        grows. A winning ``bottle`` label is replaced by the shape prior
        (:func:`shape_over_bottle`): that name has no template. The filled wine glass is renamed
        in :meth:`_reclassify_filled_height`."""
        if not semantics:
            return classify(size, bottom, color, invalid_frac, bottom_seen=bottom_seen)
        shape = classify(size, bottom, color, invalid_frac, bottom_seen=bottom_seen, min_prob=0.0)
        cfg = self.config
        fused = fuse_respecting_color(part_semantics(semantics, size), shape, color, invalid_frac, cfg.semantic_shape_weight,
                                      cfg.semantic_label_floor)
        return shape_over_bottle(fused, shape)

    # ------------------------------------------------------------- RGB-D path
    def _note_perception(self, stamp):
        """Count imaged instants, not body polls or multiple cameras at one instant."""
        stamp = float(stamp)
        if not self.perception_stamps or stamp != self.perception_stamps[-1][1]:
            self.perception_cycles += 1
            self.perception_stamps.append((self.perception_cycles, stamp))

    def update_rgbd(self, frame: dict, defer: Callable[[np.ndarray], bool] | None = None) -> None:
        """Fuse one decoded RGB-D frame: segment it, measure every region, associate regions with
        tracks by position, size and category, and rebuild each matched track's box from the
        points it has accumulated over recent views.

        :param frame: decoded RGB-D frame with its sensor header and instance mask.
        :param defer: true for the eight corners of the box of a region that another frame of the
            same instant fuses: such a region keeps its track from fading as unseen but adds no
            points, votes or new track.
        """
        header = frame["header"]
        self._note_perception(header['stamp'])
        self.stamp = float(header["stamp"])
        self.posed = True
        self.odometry = header["odometry"]
        self.joints = header["joints"]
        self._read_grippers(header)
        if not self.gripper.get("holding"):
            self._clear_held()
        self._follow_held()
        self._follow_handle()
        cam = self.camera_pose(camera=frame.get("camera"))
        cam_p = np.asarray(cam["p"])
        measurements = []
        for seg in segments(frame, cam, rng=self._rng):
            on_body = self.on_body(seg.points)
            if on_body.mean() >= self.config.body_share:
                continue  # the robot's own arm, mast or base, which robot-side segmentation also names
            if on_body.any():
                seg = dataclasses.replace(seg, points=seg.points[~on_body])
            centre, size, yaw = box_of(seg.points)
            # A clear surface returns depth on only some pixels, and that depth is noisier, so the
            # raw hull grows along the line of sight. Trim that tail before the size is recorded.
            if seg.invalid_frac >= 0.25 and max(float(size[0]), float(size[1])) < SMALL_M:
                centre, size, yaw = box_of(seg.points, trim=0.06)
            probs = self.category_probs(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac,
                                        seg.bottom_seen, seg.semantics)
            footprint = float(max(size[0], size[1]))
            if seg.semantics and "teddy_bear" in seg.semantics and footprint < 0.40 and len(self.content_hits) < 30:
                self.content_hits.append((round(self.stamp, 1),
                                          [round(float(v), 2) for v in centre],
                                          [round(float(v), 2) for v in size],
                                          round(float(seg.semantics["teddy_bear"]), 2)))
            measurements.append({"seg": seg, "centre": centre, "size": size, "yaw": yaw, "probs": probs,
                                 "footprint": footprint, "concave": _concave(seg.points, centre, footprint, cam_p),
                                 "deferred": defer is not None and defer(_box_corners(centre, size, yaw))})
        pairs = []
        held = self.tracks.get(self.held_id) if self.held_id else None
        for i, m in enumerate(measurements):
            # What the robot is carrying moves with it: a region at the load belongs to the load,
            # never to a neighbour it is carried past.
            at_load = held is not None and float(np.linalg.norm(held.pos - m["centre"])) < max(0.15, 0.6 * held.footprint)
            for track in self.tracks.values():
                # A pinned drawer is being pulled: leave it unmatched so its contents can start
                # their own tracks. A pinned cabinet still takes the view, and the fuse then
                # drops the points, so the open drawer does not become a second cabinet.
                if (track.pinned and track.category == "drawer") or (at_load and track is not held):
                    continue
                if not (track.held or track.carried) and _small_observations_apart(track.span, m["seg"].span):
                    continue  # nearby hardware is not another view of this measured piece
                d = float(np.linalg.norm(track.pos - m["centre"]))
                # Neighbouring tableware stands ~0.2 m apart: the gate scales with the object size.
                spread = 3.0 * math.sqrt(float(np.mean(track.var[:2])))
                gate = max(0.10, 0.6 * max(track.footprint, m["footprint"]) + spread, 0.45 if track.held else 0.0)
                if track.category in _CONTENT:
                    # The next bottle in the same drawer is about 20 cm away. A content track
                    # accepts a new view only inside its own prior, so it cannot walk across.
                    proto = prototype(track.category)
                    reach = float(proto[0]) * 0.5 + 0.04 if proto is not None else 0.08
                    # A bottle is a few centimetres across. A wider gate lets it walk onto
                    # the next bottle, one view at a time.
                    if proto is not None and float(proto[0]) < 0.10:
                        reach = min(reach, 0.04)
                    gate = min(gate, reach)
                # A part of a large object (a table top seen up close, a closed drawer front) lies wholly
                # inside its box and below its top; something standing on it starts at or above the top,
                # and an open drawer reaches out of it.
                m_bottom, m_top = m["centre"][2] - m["size"][2] / 2, m["centre"][2] + m["size"][2] / 2
                # A cup, bear or bottle inside an open drawer is its own object. A piece of the
                # drawer front is named drawer and is still part of the drawer.
                named = max(m["probs"], key=m["probs"].get) if m["probs"] else ""
                # A content already has its own track. A later mask of the drawer or the
                # furniture around it is not another view of it, and must not grow or rename it.
                if (track.category in _CONTENT
                        and (named in ("drawer", "cabinet", "sideboard", "counter", "table")
                             or m["footprint"] > 1.5 * track.footprint)):
                    continue
                contents = track.category == "drawer" and m["footprint"] < 0.30 and named != "drawer"
                part_of_large = not contents and track.footprint > 0.3 and m["footprint"] >= 0.3 \
                    and _footprint_inside(track, m, 0.05) \
                    and m_top <= track.top() + 0.1 and m_bottom < track.top() - 0.02
                # (a tableware-sized region is never a part of furniture: a bowl on a table whose
                # top drifted up would join the table and lift its top further, round after round)
                if d > gate and not part_of_large:
                    continue
                if track.held and m["footprint"] > HELD_LARGER * track.footprint + 0.05:
                    continue  # the gripper hides part of what it holds, but a region much larger than the
                    # load (the table under it) is never the load
                size_r = 0.0 if track.held or part_of_large else \
                    abs(math.log(max(m["footprint"], 1e-3) / max(track.footprint, 1e-3)))
                if size_r > 0.9:
                    continue
                # Nested bowls share a centre, and their footprints differ by more than one noisy
                # view of a single bowl. A partial view of that bowl is shifted toward the side
                # the camera saw, so it still joins the track.
                if (track.category == "bowl" and named == "bowl"
                        and self._separate_bowls(m["footprint"], track.footprint, m["centre"], track.pos,
                                                 m["size"], track.size)):
                    continue
                if not track.held and track.observations >= 5 and m["footprint"] > 1.5 * track.footprint:
                    continue  # partial views only shrink a well-seen object; a much larger region is something else
                if track.category == "drawer" and (m["footprint"] > DRAWER_MAX_M or float(m["size"][2]) > DRAWER_MAX_HEIGHT):
                    continue  # the sideboard or cabinet face around a drawer is a different object
                if not track.held and not part_of_large and self._contradicts(track, m):
                    continue
                if not track.held and _stacked(track.bottom(), track.top(), m_bottom, m_top, m["centre"][2],
                                               m["seg"].bottom_seen and track.bottom_seen):
                    continue  # a lid on a box, a book on a book: one rests on the other
                wall_like = m["probs"].get("wall", 0.0) > 0.5 or m_top >= self.config.wall_like_top or m["concave"]
                if (track.category == "wall") != wall_like and (track.footprint > 0.3 or m["footprint"] > 0.3):
                    continue  # walls only join walls
                if not track.held and track.settled and track.footprint < 0.3 and track.observations > 2 \
                        and not (track.bottom() - 0.03 <= m_bottom <= track.bottom() + 0.06):
                    continue  # a small object resting somewhere keeps its base height
                d = min(d, gate * 0.5) if part_of_large else d
                total = sum(track.category_counts.values()) or 1.0
                overlap = sum(min(m["probs"].get(k, 0.0), v / total) for k, v in track.category_counts.items())
                pairs.append((d / gate + 0.8 * size_r + 0.4 * (1.0 - overlap), i, track.id))
        matched, used = set(), set()
        for _, i, tid in sorted(pairs):
            if i in used or tid in matched:
                continue
            used.add(i)
            matched.add(tid)
            if not measurements[i]["deferred"]:
                self._fuse_cloud(self.tracks[tid], measurements[i], cam_p)
        for i, m in enumerate(measurements):
            if m["deferred"]:
                continue
            # A region cut by the image border is only part of something and waits for a full view,
            # except a wall: a near wall always fills the image edge to edge.
            is_wall = m["probs"].get("wall", 0.0) >= 0.5
            if m["concave"] and not is_wall:
                continue  # the inside of a corner is room, not a solid object
            if _mixed_ranges(m["seg"].points, cam_p, m["size"]) and m["probs"].get("wall", 0.0) < 0.5:
                continue  # near surface and far background in one mask: a wide thick blob, not an object
            if i not in used and (not m["seg"].border or is_wall):
                if (not any(name in _CONTENT for name in (m["seg"].semantics or {}))
                        and self._inside_shut_furniture(m["centre"], m["size"], m["footprint"])):
                    continue  # a handle or an edge of a shut cabinet; nothing inside it can be seen
                if self._in_held_drawer(m["centre"], m["footprint"]):
                    continue  # the frame shows the sliding drawer where it was; its contents are tracked
                if self._is_the_pulled_drawer(m):
                    continue  # the drawer that is out already has a track; a second one would be seated shut
                if self._over_several_objects(m):
                    continue  # one mask over pieces that already stand apart is not one more piece
                track = self._new_cloud_track(m, cam_p)
                matched.add(track.id)
        self._decay_unseen_rgbd(cam, matched, frame)
        self._drop_printed_fragments()
        self._drop_assembly_composites()
        self._merge_duplicates()
        self._merge_parts(cam_p)
        self._dedupe_furniture(cam_p)
        self._merge_rim_fragments()
        self._fold_furniture_slices(cam_p)
        self._seat_against_the_wall()
        self._trim_swollen_table()
        self._seat_closed_drawers()
        self._drop_support_top_drawers()
        self._merge_same_surface(cam_p)
        self._merge_same_place(cam_p)
        self._merge_walls(cam_p)
        self._join_abutting_furniture(cam_p)
        self._drawer_in_sideboard()
        self._drop_impossible(cam_p)
        self._drop_orphan_parts()
        self._spawn_shed()
        self._drop_false_tableware()
        self._drop_false_chairs()
        self._contents_of_open_drawers()
        self._merge_same_spot()
        self._follow_held()
        self._seat_floating_tableware()
        self._seat_against_the_wall()
        self._trim_swollen_table()
        self._seat_closed_drawers()
        self._merge_same_place(cam_p)
        self._drop_unnamed_parts()
        self._drop_inside_shut_furniture()
        self._drop_furniture_edges()
        self._drop_pistols_off_the_drawer()
        self._drop_left_in_front()
        self._drop_the_front_bottle()
        self._drop_thin_bears()
        self._bring_back_what_the_close_pushed_through()
        self._drop_wall_edges()
        self._merge_same_spot()
        self._rename_airplanes_called_bears()
        self._name_the_pistol_in_the_sideboard()
        self._seat_contents_in_closed_drawers()
        self._rename_wide_perfumes()
        self._name_pink_pills()
        self._rename_wide_perfumes()
        self._drop_perfume_on_the_pill()
        self._drop_bottles_on_the_airplane()
        self._drop_bottles_beside_the_drawer()
        self._drop_flat_plants()
        self._drop_unknowns_on_the_airplane()
        self._drop_perfume_smears()
        self._rename_flat_mugs_in_drawers()
        self._drop_stray_bears()
        self._drop_contents_on_the_wrong_piece()
        self._name_unknown_bowls()
        self._drop_cups_on_the_bear()
        self._cups_on_the_counter_stay_cups()
        self._drop_false_chairs()
        self._trim_swollen_table()
        self._seat_swollen_drawer()
        self._seat_closed_drawers()
        self._recede_facing_excess(cam_p)
        self._seat_closed_drawers()
        self._prefer_tight_bowl_views()
        self._pull_bowl_off_the_lip(cam_p)
        self._drop_false_chairs()
        self._drop_inside_shut_furniture()
        self._drop_second_airplane()
        self._follow_shut_front()
        self.version += 1

    def finalize_map(self, cam_p: np.ndarray | None = None) -> None:
        """Apply the late seating and false-positive drops once perception has stopped.

        The last look can leave a swollen cabinet and a table-sized chair that the end of
        ``update_rgbd`` did not yet shrink, when that frame's heading did not face them.
        """
        if cam_p is None:
            cam_p = np.array([float(self.odometry.get("x", 0.0)),
                              float(self.odometry.get("y", 0.0)), 1.5], float)
        else:
            cam_p = np.asarray(cam_p, float)
        self._drop_inside_shut_furniture()
        self._cups_on_the_counter_stay_cups()
        self._name_unknown_bowls()
        self._drop_false_chairs()
        self._name_the_pistol_in_the_sideboard()
        self._seat_contents_in_closed_drawers()
        self._drop_second_airplane()
        self._seat_swollen_drawer()
        # Do not snap a drawer the look already pulled open; seating is for shut faces only.
        self._seat_closed_drawers()
        self._seat_against_the_wall()
        self._recede_facing_excess(cam_p)
        self._seat_closed_drawers()
        self._prefer_tight_bowl_views()
        self._pull_bowl_off_the_lip(cam_p)
        self._drop_false_chairs()
        self._drop_second_airplane()
        self.version += 1

    def _drop_impossible(self, cam_p: np.ndarray) -> None:
        """Remove tracks that cannot be solid objects: one whose surfaces lie behind its own box
        centre as seen from the camera (a concave corner of the room boxed as if it were a convex
        block), one whose footprint holds the robot's base (the robot stands there), a non-wall
        track as tall as a wall (wall pieces swept into it), a ``box`` wider than 0.9 m, a new
        track on the far side of a wall (:func:`_behind_wall`), and a
        grazing edge (metres long and only a few decimetres thick, or a few centimetres thick and
        taller than the furniture).
        Such boxes would otherwise fill the room's corners, hide the objects in them and block every path.
        Grazing edges are removed first: a later hollow test must not keep them."""
        for track in list(self.tracks.values()):
            if track.held or track.category == "wall":
                continue
            thin = min(float(track.size[0]), float(track.size[1]))
            # No furniture in the room is a few decimetres thick over a metre, and nothing real
            # is under 8 cm thick and taller than a cabinet. A wall edge boxed on its own is that sheet.
            grazing = (thin < 0.32 and track.footprint > 1.4) or (thin < 0.08 and float(track.size[2]) > 0.50)
            if not grazing and thin < 0.25 and track.footprint >= 0.6 and track.observations < ESTABLISHED_VIEWS:
                # A new box deeper along the line of sight than any furniture is thick, and
                # thinner across it than any piece: the side of a cabinet and its open drawer
                # seen from in front. A real piece shows its front across the line of sight.
                grazing = _edge_on(track, cam_p)
            # A flat unknown wider than a plate is the counter top boxed again.
            flat_unknown = (track.category == "unknown" and float(track.size[2]) < 0.03
                            and track.footprint > 0.45)
            sliver = (track.category == "unknown" and thin < 0.06 and track.footprint > 0.20)
            # The segmenter names a real pair of scissors. A cabinet edge completed to that prior
            # never received the name.
            unnamed_scissors = (track.category == "scissors"
                                and track.semantic_counts.get("scissors", 0.0) <= 0.0)
            if grazing or flat_unknown or sliver or unnamed_scissors:
                self._drop(track, "a grazing edge, not an object" if grazing else
                           "counter top boxed as a flat unknown" if flat_unknown else
                           "a thin unknown strip, not an object" if sliver else
                           "cabinet edge completed into scissors")
        base = np.array([self.odometry["x"], self.odometry["y"], 0.0])
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category == "wall":
                continue
            holds_robot = _inside(track, base, -0.05) and track.bottom() < 1.0
            wall_tall = track.top() >= self.config.wall_like_top and track.footprint >= 0.5
            # A table (open beneath its top) or an open drawer can look hollow from some viewpoints;
            # an object seen many times and identified with confidence, or one the segmenter keeps
            # naming as something other than a wall, is not a boxed room corner.
            hollow = not _identified(track) and not _named(track, self.config.semantic_named_score) \
                and track.voxels is not None \
                and _concave(_points(track.voxels), track.pos, track.footprint, cam_p)
            oversized_box = track.category == "box" and track.footprint > 0.9
            walled_off = track.observations < ESTABLISHED_VIEWS and any(
                wall.category == "wall" and wall.observations >= ESTABLISHED_VIEWS and wall.footprint >= 1.5
                and wall.top() >= self.config.wall_like_top and _behind_wall(track, wall, cam_p)
                for wall in self.tracks.values())
            if holds_robot or wall_tall or hollow or oversized_box or walled_off:
                self._drop(track, "holds the robot's base" if holds_robot else
                           "as tall as a wall" if wall_tall else
                           "hollow: its surfaces lie behind its centre" if hollow else
                           "box wider than any object the segmenter names box" if oversized_box else
                           "on the far side of a wall")

    def _drop_orphan_parts(self) -> None:
        """A drawer is a part of a piece of furniture; one standing on its own is a misread edge.

        A tabletop's apron or the shadow line under its front panel segments as a "drawer", and
        as a support surface it is poison for planning: every approach to something resting on it
        is docked against the phantom's own small box, which puts the robot's body inside the
        real furniture, so all four sides come back as collisions and the piece can never be
        picked. It must also sit within the furniture's own height: a drawer above a tabletop is
        the apron seen from above. The parent is normally established; when it is not, the part
        is re-read later."""
        furniture = [t for t in self.tracks.values()
                     if t.category not in ("drawer", "counter") and t.footprint >= 0.4
                     and "support-surface" in prior(t.category)["affordances"]]
        for track in list(self.tracks.values()):
            if track.category != "drawer" or track.held or track.id in self._opened_drawers:
                continue
            # A real drawer sits inside its furniture piece: it does not rise above its top (a
            # tabletop apron read as a drawer is a long, shallow box along the near edge, and it
            # comes out ABOVE the top it belongs to). Seen from a steeper camera that box is wide
            # enough to count as a support surface, and then every approach to anything on the
            # table is planned against it and reported as colliding.
            inside = any(_inside(f, track.pos, 0.12) and f.bottom() - 0.06 <= track.pos[2] <= f.top() + 0.06
                         and track.top() <= f.top() + 0.05 for f in furniture)
            # Pulling the drawer slides its centre out of the furniture box. It is still that
            # furniture's drawer: only a box that also stands clear of every piece is a false edge.
            pulled_out = bool(furniture) and not inside and self.drawer_opening(track) >= 0.08 \
                and track.top() <= max(f.top() for f in furniture) + 0.05 \
                and any(float(np.linalg.norm(f.pos[:2] - track.pos[:2])) < 0.9 for f in furniture)
            if not inside and not pulled_out:
                self._drop(track, "orphan drawer: not inside any furniture")
        drawers = [t for t in self.tracks.values() if t.category == "drawer" and not t.held]
        for thin in list(drawers):
            if thin.id not in self.tracks or float(thin.size[2]) > 0.15:
                continue
            if any(other.id != thin.id and other.id in self.tracks
                   and float(np.linalg.norm(other.pos[:2] - thin.pos[:2])) < 0.12
                   and abs(float(other.pos[2] - thin.pos[2])) < 0.20
                   and float(other.size[2]) > float(thin.size[2]) + 0.08
                   for other in drawers):
                self._drop(thin, "a thin second box of the same drawer")
        opened = [t for t in drawers if t.id in self.tracks and t.id in self._opened_drawers]
        for other in list(drawers):
            if other.id not in self.tracks or other.id in self._opened_drawers:
                continue
            if float(other.size[2]) <= 0.30:
                continue
            if any(float(np.linalg.norm(one.pos[:2] - other.pos[:2])) < 0.45 for one in opened):
                self._drop(other, "a swollen second box of the open drawer")
        self._drop_second_airplane()

    def _drop_second_airplane(self) -> None:
        """Drop the larger of two airplanes a hand's width apart.

        One toy is seen twice, once as a smear. Merging them walks the tight box onto the smear.
        The tight box is the toy."""
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        for i, a in enumerate(planes):
            for b in planes[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                if float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.20:
                    continue
                named_a = float(a.semantic_counts.get("toy_airplane", 0.0)) > 0.0
                named_b = float(b.semantic_counts.get("toy_airplane", 0.0)) > 0.0
                if named_a != named_b:
                    # The segmenter named one of them. The other is a depth fill beside it.
                    drop = b if named_a else a
                else:
                    drop = a if a.footprint >= b.footprint else b
                    keep = b if drop is a else a
                    # A box under 6 cm is a sliver of the floor, not a tighter view of the toy.
                    if float(max(keep.size[0], keep.size[1])) < 0.06 <= float(max(drop.size[0], drop.size[1])):
                        drop = keep
                self._drop(drop, "a second airplane on the same spot")

    def _decay_unseen_rgbd(self, cam: dict, matched: set[str], frame: dict) -> None:
        """Weigh the evidence against tracks that should be in view but were not segmented.

        Only free space removes an established track: depth readings at its centre and at a 3×3
        grid just below its top (:meth:`_probes`) that lie beyond the whole box show that the camera
        saw through the place it was said to fill. Such a view counts once per
        ``absence_interval_s``; the track is dropped after ``absence_views_established`` of them
        from ``absence_viewpoints`` camera positions, and any new observation of it clears them. A
        surface where the track should be (a table whose regions went to another track, a view
        the segmentation missed) is no evidence of absence. Tracks seen fewer than
        :data:`ESTABLISHED_VIEWS` times are often fragments or noise: they also fade by
        ``missed_decay`` per view and need only ``absence_views`` free views."""
        depth = frame["depth"]
        h, w = depth.shape
        f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
        inv = pose_inverse(cam)
        cam_yaw = yaw_from_quat(cam["q"])

        def reading(point: np.ndarray) -> tuple[float, float] | None:
            x, y, z = pose_compose(inv, {"p": point.tolist(), "q": [1, 0, 0, 0]})["p"]
            if x < 0.3 or x > self.config.max_range * 0.8:
                return None
            u, v = int(round((w - 1) / 2 - y * f / x)), int(round((h - 1) / 2 - z * f / x))
            if not (0 <= u < w and 0 <= v < h):
                return None
            return x, float(depth[v, u])

        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in matched or track.held or track.carried or self._in_closed_drawer(track) \
                    or self._in_open_drawer(track):
                continue
            # A clear glass returns the room behind it. That is not evidence the glass has gone.
            # A cup or mug on a surface does too: the depth reading is the counter behind the glass.
            if track.mean_invalid >= 0.25 or track.category == "wine_glass" \
                    or (track.category in ("cup", "mug") and track.bottom() > 0.40):
                continue
            centre = reading(track.pos)
            if centre is None:
                continue
            tol = max(0.10, track.footprint / 2)
            free = filled = 0
            for probe in self._probes(track):
                r = reading(probe)
                if r is None or r[1] <= 0.05:
                    continue
                d, seen = r
                if seen > d + max(0.15, track.footprint / 2):
                    free += 1
                else:
                    filled += 1  # a surface where the track should be, or something nearer hiding it
            established = track.observations >= ESTABLISHED_VIEWS
            young_in_view = not established and centre[1] > 0.05 and centre[1] >= centre[0] - tol
            if young_in_view:
                track.existence *= self.config.missed_decay
            required = self.config.absence_views_established if established else self.config.absence_views
            counted = free + filled
            if counted >= 5 and free >= self.config.absence_free_share * counted:
                self._note_absence(track, cam["p"], cam_yaw, required)
            if not established and track.existence < self.config.drop_below:
                self._drop(track, "faded: a young track not seen again where it should be visible")
                continue
            views = len(track.absent_views)
            if views:
                track.existence = min(track.existence, 1.0 - 0.5 * min(views, required) / required)
            if views >= required and (not established or self._viewpoints(track) >= self.config.absence_viewpoints):
                self._drop(track, f"seen through from {views} views")

    @staticmethod
    def _probes(track: Track) -> list[np.ndarray]:
        """The centre and a 3×3 grid just below the top of a track's box: rays to an object that is
        there meet it (a table's top hides the room between its legs); rays to a place it has left
        pass on to what lies behind."""
        c, s = math.cos(track.yaw), math.sin(track.yaw)
        z = track.top() - 0.02
        probes = [track.pos.copy()]
        for fx in (-0.3, 0.0, 0.3):
            for fy in (-0.3, 0.0, 0.3):
                dx, dy = fx * track.size[0], fy * track.size[1]
                probes.append(np.array([track.pos[0] + c * dx - s * dy, track.pos[1] + s * dx + c * dy, z]))
        return probes

    def _note_absence(self, track: Track, cam_p: list[float], cam_yaw: float, required: int) -> None:
        """Count a free-space view of a track, at most once per ``absence_interval_s``; once
        ``required`` views are in, only a view from a new camera pose adds to them."""
        if track.absent_views and self.stamp - track.absent_views[-1][0] < self.config.absence_interval_s:
            return
        view = (self.stamp, float(cam_p[0]), float(cam_p[1]), cam_yaw)
        if len(track.absent_views) >= required and not self._new_viewpoint(track.absent_views, view):
            return
        track.absent_views.append(view)

    def _new_viewpoint(self, views: list[tuple[float, float, float, float]], view: tuple[float, float, float, float]) -> bool:
        _, x, y, yaw = view
        return all(math.hypot(x - a, y - b) >= self.config.viewpoint_spread_m
                   or abs(wrap_angle(yaw - c)) >= math.radians(self.config.viewpoint_spread_deg)
                   for _, a, b, c in views)

    def _viewpoints(self, track: Track) -> int:
        """Number of clearly different camera poses among a track's absence views."""
        distinct: list[tuple[float, float, float, float]] = []
        for view in track.absent_views:
            if self._new_viewpoint(distinct, view):
                distinct.append(view)
        return len(distinct)

    def _rename(self, track: Track, new_id: str) -> None:
        del self.tracks[track.id]
        track.id = new_id
        self.tracks[new_id] = track

    def _drop(self, track: Track, reason: str, into: Track | None = None) -> None:
        """Remove a track and record why, for the operator and for diagnosis."""
        if track.pinned:
            return
        del self.tracks[track.id]
        self.dropped.append({"t": round(self.stamp, 2), "id": track.id, "category": track.category,
                             "observations": track.observations, "reason": reason,
                             "position": [round(float(v), 3) for v in track.pos],
                             "size": [round(float(v), 3) for v in track.size],
                             "robot": [round(self.odometry["x"], 3), round(self.odometry["y"], 3)],
                             **({"into": into.id} if into is not None else {})})

    def _dedupe_furniture(self, cam_p: np.ndarray) -> None:
        """Fold furniture-scale duplicates at one spot into the copy that can be trusted.

        A real segmenter answers "table" with a different region every view — a floating top
        slab from one, the table with the glass glued to it from another — and such copies never
        associate (parts of one object are expected to arrive inside its box, not around it), so
        two or three tables pile up on the same footprint and the planner picks any of them. The
        copy to keep is the one that reaches the floor, else the one with the lowest top: things
        glued onto a copy only ever lift its top. The dropped copy's points are not merged — its
        geometry is the untrustworthy one, and folding it in widens the kept box until the
        category flips (a table 1.7 m wide reads as a counter and gets a counter's height)."""
        big = [t for t in self.tracks.values() if not t.held and t.footprint >= 0.5
               and t.category not in ("wall", "drawer", "bowl", "cup", "plate", "mug", "wine_glass")]
        for i, a in enumerate(big):
            for b in big[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                if float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.4 * max(a.footprint, b.footprint):
                    continue
                # A long sideboard's centre falls inside a chair-sized box beside it. Only the
                # smaller centre lying inside the larger piece is one object seen twice.
                larger, smaller = (a, b) if a.footprint >= b.footprint else (b, a)
                if not _inside(larger, smaller.pos, 0.02):
                    continue
                if abs(a.top() - b.top()) > 0.25:
                    continue  # stacked different things, not two readings of one
                # SAM names the cabinet a counter. Prefer the cabinet label so the drawer keeps a host.
                if {a.category, b.category} == {"cabinet", "counter"}:
                    named_cabinet = a if a.category == "cabinet" else b
                    named_counter = b if named_cabinet is a else a
                    if float(max(named_counter.size[0], named_counter.size[1])) < 1.0 \
                            and float(named_counter.size[2]) > 0.60:
                        named_cabinet.observations += named_counter.observations
                        named_cabinet.existence = max(named_cabinet.existence, named_counter.existence)
                        self._drop(named_counter, "merged: furniture duplicate", into=named_cabinet)
                        continue
                a_floor, b_floor = a.bottom() < 0.12, b.bottom() < 0.12
                if a.pinned != b.pinned:
                    # The visit pins the sideboard. A second table on that footprint must go;
                    # dropping the pinned host is a no-op, and the table then blocks the close.
                    keep, drop = (a, b) if a.pinned else (b, a)
                elif self._host_drawer_open(a) and self._host_drawer_open(b) and a.first_seen != b.first_seen:
                    # A drawer here is pulled out. A view of the furniture now takes the drawer
                    # with it: lower, and deeper toward the robot. The piece known before the
                    # pull is the furniture; kept instead, the newer box fills the place the
                    # robot must stand in to push the drawer back.
                    keep, drop = (a, b) if a.first_seen < b.first_seen else (b, a)
                elif a_floor != b_floor:
                    keep, drop = (a, b) if a_floor else (b, a)
                else:
                    keep, drop = (a, b) if a.top() <= b.top() else (b, a)
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: furniture duplicate", into=keep)

    def _merge_rim_fragments(self) -> None:
        """Fold a sparse tableware track into the fuller neighbour that covers it.

        A prompt on a wide bowl's rim returns the rim arc as reliably as another view returns the
        whole bowl, so the arc becomes a track with existence as high as the bowl's — and a grasp
        aimed at its centre misses by the bowl's radius. What marks the arc is its footprint: a
        ring fills a fraction of its own box that a whole vessel fills most of."""
        tracks = [t for t in self.tracks.values()
                  if not t.held and t.footprint < 0.45 and t.bottom() > 0.2 and t.voxels is not None]
        fills = {t.id: _footprint_fill(_points(t.voxels), t.pos, t.size, t.yaw) for t in tracks}
        for thin in tracks:
            if thin.id not in self.tracks or fills.get(thin.id, 1.0) >= 0.45 or _content_named(thin):
                continue  # a pistol is flat, so its box looks empty; it is not a bowl's rim
            host = None
            for other in tracks:
                if other.id == thin.id or other.id not in self.tracks or other.footprint < thin.footprint * 1.2:
                    continue
                if fills.get(other.id, 0.0) < 0.55:
                    continue
                if abs(other.pos[2] - thin.pos[2]) > 0.12:
                    continue
                if float(np.linalg.norm(other.pos[:2] - thin.pos[:2])) > 0.5 * other.footprint:
                    continue
                if not _inside(other, thin.pos, 0.02):
                    continue
                host = other
                break
            if host is not None:
                host.observations += thin.observations
                host.existence = max(host.existence, thin.existence)
                self._drop(thin, "merged: rim fragment into the whole vessel", into=host)

    def _merge_same_surface(self, cam_p: np.ndarray) -> None:
        """Merge tracks whose accumulated points lie on the same surfaces (duplicates from views
        that were not associated). Nested bowls are different surfaces and stay apart."""
        tracks = [t for t in self.tracks.values() if t.voxels is not None and not t.held and not t.carried]
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks or a.pinned or b.pinned:
                    continue
                if float(np.linalg.norm(a.pos - b.pos)) > 0.5 * max(a.footprint, b.footprint) + 0.05:
                    continue
                if _small_observations_apart(a.span, b.span):
                    continue
                receptacles = "receptacle" in a.affordances and "receptacle" in b.affordances
                if receptacles and not self._size_match(a.footprint, b.footprint):
                    continue  # a smaller bowl nested in a larger one
                if _drawer_beside_furniture(a, b) or _content_on_drawer(a, b):
                    continue  # shared front-face points do not make the drawer the sideboard,
                    # and the bear's mask includes the drawer floor without being the drawer
                if ((a.category == "drawer" and b.footprint < 0.35 and b.category not in ("wall", "drawer"))
                        or (b.category == "drawer" and a.footprint < 0.35 and a.category not in ("wall", "drawer"))):
                    continue  # a chair-sized object on the drawer floor is not a second name for the drawer
                if _different_contents(a, b) or _content_in_larger(a, b):
                    continue  # a pill and a perfume share the drawer floor without being one object,
                    # and that floor must not swallow either of them
                bottle = a.category == b.category and a.category in ("pill_bottle", "perfume_bottle")
                apart = 0.06 if bottle else 0.12
                if (_content_named(a) and _content_named(b)
                        and float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > apart):
                    continue  # the drawer floor under two copies of one bottle is not one object
                if (a.category != b.category and max(a.footprint, b.footprint) < 0.25
                        and float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.05):
                    continue  # a pill and the perfume a hand apart share the drawer floor
                if ({a.category, b.category} <= {"cup", "mug", "bowl", "plate", "wine_glass"}
                        and a.category != b.category
                        and float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.08):
                    continue  # the mug and the bowl beside it share the table, not a surface
                shared = {tuple(v) for v in a.voxels} & {tuple(v) for v in b.voxels}
                if len(shared) < 0.5 * min(len(a.voxels), len(b.voxels)):
                    continue
                # An old background smear must not win just by having more views.
                # Require a repeatedly named kit part and actual shared surfaces;
                # never replace furniture or another named object by proximity.
                contaminated = False
                for part, blob in ((a, b), (b, a)):
                    cap = max_long_extent(part.category)
                    if (part.category in ASSEMBLY_PARTS and cap is not None
                            and part.observations >= 3 and part.semantic_views >= 3
                            and part.semantic_counts.get(part.category, 0.) / part.semantic_views
                                >= self.config.semantic_named_score
                            and part.footprint <= cap and blob.category == "unknown"
                            and max(blob.footprint, float(blob.size[2])) > 2 * cap):
                        self._drop(blob, "background smear overlapping a repeatedly named assembly part", into=part)
                        contaminated = True
                        break
                if contaminated:
                    continue
                if a.held or b.held:
                    keep, drop = (a, b) if a.held else (b, a)
                else:
                    keep, drop = (a, b) if a.observations >= b.observations else (b, a)
                cap = max_long_extent(keep.category)
                if (cap is not None and keep.footprint < 0.45 and keep.observations >= ESTABLISHED_VIEWS
                        and max(drop.footprint, float(drop.size[2])) > 2.0 * cap):
                    # One mask ran from the bowl along the table edge and the wall behind it. It
                    # holds the bowl's points and is not the bowl: unioned in, the bowl's cloud
                    # is two metres long and its box is whichever cluster of that survives. A mask
                    # of the open drawer and the cabinet face under it holds the airplane the same
                    # way, and is tall where that one is long.
                    if drop.observations < ESTABLISHED_VIEWS:
                        self._drop(drop, "a region spanning an established object and its surroundings")
                    continue
                keep.voxels = self._union(keep.voxels, drop.voxels)
                _absorb_span(keep, drop)
                _absorb_hardware_views(keep, drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                _absorb_semantics(keep, drop)
                for col, n in drop.color_counts.items():
                    keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: same surfaces", into=keep)
                self._rebuild_geometry(keep, cam_p, 2.0)

    def _merge_same_place(self, cam_p: np.ndarray) -> None:
        """Fold two small raised tracks whose centres are within 8 cm into one.

        A bowl and the glass beside it are further apart than that. Two segmenter names for the
        same object, or the tight object and the wider slab it was cut out of, are not. The tighter
        cloud is the one kept: the wider one is the table that was boxed with it.
        """
        small = [t for t in self.tracks.values()
                 if not t.held and not t.carried and t.footprint < 0.55 and t.bottom() > 0.3]
        for i, a in enumerate(small):
            for b in small[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks or a.pinned or b.pinned:
                    continue
                if _small_observations_apart(a.span, b.span):
                    continue
                if ("receptacle" in a.affordances and "receptacle" in b.affordances
                        and self._separate_bowls(a.footprint, b.footprint, a.pos, b.pos, a.size, b.size)):
                    continue  # a smaller bowl nested in a larger one
                if (_content_on_drawer(a, b)
                        and {a.category, b.category} & {"pistol", "perfume_bottle"}):
                    continue  # a pistol or perfume in the drawer is not a second name for the drawer front
                if float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.08:
                    continue
                if (a.category in _CONTENT and b.category in _CONTENT and a.category != b.category
                        and {a.category, b.category} != {"pill_bottle", "perfume_bottle"}):
                    continue  # the airplane and the pill beside it are two objects
                if ({a.category, b.category} <= {"pill_bottle", "perfume_bottle", "bottle"}
                        and float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.03):
                    continue  # the pill and the perfume sit apart in this drawer, not one bottle
                if (_content_named(a) or _content_named(b)) and not (_content_named(a) and _content_named(b)):
                    content = a if _content_named(a) else b
                    other = b if content is a else a
                    # A bear is not a second name for the chair-sized smear beside it.
                    if (other.footprint > max(content.footprint * 1.8, 0.25)
                            or float(other.size[2]) > float(content.size[2]) + 0.25):
                        continue
                if {a.category, b.category} == {"teddy_bear", "wine_glass"} and not self._beside_drawer(a.pos):
                    # On the table the glass is the object. The bear name is the brown of the stem.
                    keep = a if a.category == "wine_glass" else b
                    drop = b if keep is a else a
                    self._drop(drop, "merged: two names for one object", into=keep)
                    continue
                if abs(a.bottom() - b.bottom()) > 0.08 and abs(a.top() - b.top()) > 0.08:
                    continue
                if _different_contents(a, b) and {a.category, b.category} != {"pill_bottle", "perfume_bottle"}:
                    continue  # a bear and a pistol a few centimetres apart are two objects
                # A lid and the box under it share a centre, and each end is within a few centimetres,
                # but their bodies do not meet. Two names for one object overlap in height.
                if min(a.top(), b.top()) < max(a.bottom(), b.bottom()) - 0.01:
                    continue
                if (a.category in _CONTENT) != (b.category in _CONTENT):
                    # The segmenter can call the airplane a cup as well. The track whose name is
                    # already the content is the object; the other label is dropped.
                    keep = a if a.category in _CONTENT else b
                    drop = b if keep is a else a
                elif _content_named(a) != _content_named(b):
                    # A bowl or a bottle label on the perfume is the second name, not the object.
                    keep = a if _content_named(a) else b
                    drop = b if keep is a else a
                elif {a.category, b.category} == {"pill_bottle", "perfume_bottle"}:
                    # The pill prior fits the perfume's box. The perfume label is the one SAM
                    # added; the pill label on the same spot is that prior. A pink bottle and a
                    # white one are the two bottles, even when the open drawer has slid them together.
                    # A pill no mask ever called a bottle is that prior on a part of the perfume:
                    # its cap is another colour than its body and is not a second bottle.
                    pill = a if a.category == "pill_bottle" else b
                    seen_as_bottle = any(pill.semantic_counts.get(name, 0.0) > 0.0
                                         for name in ("bottle", "pill_bottle"))
                    if (seen_as_bottle and a.mean_rgb is not None and b.mean_rgb is not None
                            and float(np.linalg.norm(a.mean_rgb - b.mean_rgb)) > 0.20):
                        continue
                    keep = a if a.category == "perfume_bottle" else b
                    drop = b if keep is a else a
                elif (a.category == "unknown") != (b.category == "unknown"):
                    # A depth fill has no name. The cup or the plate already named is the object,
                    # and the unknown count is not a vote for a second category.
                    keep = b if a.category == "unknown" else a
                    drop = a if keep is b else b
                elif (a.category == b.category and a.category in _CONTENT
                      and min(a.observations, b.observations) <= 2 and abs(a.observations - b.observations) >= 2):
                    # The front of a drawer hides most of what stands in it, and a view of the cap
                    # of a bottle is tighter than the bottle. The box seen more often has the views
                    # its size and colour are taken from.
                    keep, drop = (a, b) if a.observations > b.observations else (b, a)
                elif a.observations >= ESTABLISHED_VIEWS and b.observations <= 2:
                    keep, drop = a, b  # one view beside an established box is a partial view of it
                elif b.observations >= ESTABLISHED_VIEWS and a.observations <= 2:
                    keep, drop = b, a
                else:
                    keep, drop = (a, b) if _tighter(a, b) else (b, a)
                _absorb_hardware_views(keep, drop)
                if not (_content_named(keep) and not _content_named(drop)):
                    # A chair label copied onto the bear is what renames the bear and loses it.
                    for cat, n in drop.category_counts.items():
                        if cat == "unknown" and keep.category != "unknown":
                            continue
                        keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                    _absorb_semantics(keep, drop)
                    for col, n in drop.color_counts.items():
                        keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: two names for one object", into=keep)
                if keep.voxels is not None:
                    self._rebuild_geometry(keep, cam_p, 2.0)

    def _merge_walls(self, cam_p: np.ndarray) -> None:
        """Join wall tracks that lie in one plane and meet or nearly meet along it: separate views
        of one wall, each cut by the image border."""
        walls = [t for t in self.tracks.values() if t.category == "wall" and t.voxels is not None]
        for i, a in enumerate(walls):
            for b in walls[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                turn = abs(math.remainder(a.yaw - b.yaw, math.pi))
                if turn > 0.2:
                    continue
                normal = np.array([-math.sin(a.yaw), math.cos(a.yaw)])
                along = np.array([math.cos(a.yaw), math.sin(a.yaw)])
                d = b.pos[:2] - a.pos[:2]
                if abs(float(d @ normal)) > 0.1:
                    continue
                if abs(float(d @ along)) - (a.size[0] + b.size[0]) / 2 > 0.4:
                    continue
                if a.held or b.held:
                    keep, drop = (a, b) if a.held else (b, a)
                else:
                    keep, drop = (a, b) if a.observations >= b.observations else (b, a)
                keep.voxels = self._union(keep.voxels, drop.voxels)
                _absorb_span(keep, drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                _absorb_semantics(keep, drop)
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: one wall", into=keep)
                self._rebuild_geometry(keep, cam_p, 2.0)

    def _join_abutting_furniture(self, cam_p: np.ndarray) -> None:
        """Join floor-standing pieces that meet and together are one sideboard.

        A sideboard face seen in two looks is about the size of a chair each time, so one piece
        is named a chair and the other stays unknown. Neither box contains the other, and the
        sideboard is then missing. Two pieces the segmenter both called a chair are left alone:
        those can be two chairs.
        """
        names = ("chair", "sideboard", "unknown")
        bodies = [t for t in self.tracks.values()
                  if t.category in names and t.voxels is not None and not t.held and not t.carried
                  and t.bottom() < 0.08 and float(t.size[2]) > 0.55]
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                if a.semantic_counts.get("chair", 0.0) > 0.0 and b.semantic_counts.get("chair", 0.0) > 0.0:
                    continue
                if not _boxes_meet(a, b, 0.20):
                    continue
                keep, drop = (a, b) if a.observations >= b.observations else (b, a)
                keep.voxels = self._union(keep.voxels, drop.voxels)
                _absorb_span(keep, drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                _absorb_semantics(keep, drop)
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: one sideboard", into=keep)
                self._rebuild_geometry(keep, cam_p, 2.0)
                long = float(max(keep.size[0], keep.size[1]))
                short = float(min(keep.size[0], keep.size[1]))
                if (0.85 < long < 1.70 and 0.35 < short < 0.70 and 0.65 < float(keep.size[2]) < 1.05
                        and keep.bottom() < 0.10 and keep.category in ("chair", "unknown", "cabinet")):
                    keep.category_counts = {"sideboard": float(max(keep.observations, 1))}

    def _drawer_in_sideboard(self) -> None:
        """A sideboard with no drawer track still has its drawer: the top band of the face.

        The handle and the front sit in that band. Copying it out leaves the sideboard in place,
        so the two stay separate objects.
        """
        for host in list(self.tracks.values()):
            if host.category != "sideboard" or host.voxels is None or host.held:
                continue
            if any(t.category == "drawer" and _inside(host, t.pos, 0.15) for t in self.tracks.values()):
                continue
            cloud = _points(host.voxels)
            if len(cloud) < 40:
                continue
            z_hi = float(np.percentile(cloud[:, 2], 95))
            band = cloud[cloud[:, 2] >= z_hi - 0.28]
            if len(band) < 20:
                continue
            # The band runs the whole length of the face. The drawer is the middle of that face,
            # on the side toward the robot, and only as wide as a drawer. Keeping the whole band
            # puts the handle on the wood beside the drawer, and the next view stretches it back.
            yaw = float(host.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            long_axis = np.array([c, s])
            depth_axis = np.array([-s, c])
            away = np.array([float(self.odometry.get("x", 0.0)) - float(host.pos[0]),
                             float(self.odometry.get("y", 0.0)) - float(host.pos[1])])
            if float(away @ depth_axis) < 0.0:
                depth_axis = -depth_axis
            depth = float(min(host.size[0], host.size[1]))
            front = host.pos[:2] + depth_axis * max(0.0, depth / 2 - 0.02)
            rel = band[:, :2] - front
            piece = band[(np.abs(rel @ long_axis) <= 0.28) & (np.abs(rel @ depth_axis) <= 0.22)]
            if len(piece) < 20:
                continue
            height = min(0.26, float(piece[:, 2].max() - piece[:, 2].min()) + 0.02)
            centre = np.array([float(front[0]), float(front[1]), float(np.median(piece[:, 2]))])
            size = np.array([0.56, 0.44, height]) if float(host.size[0]) >= float(host.size[1]) \
                else np.array([0.44, 0.56, height])
            if float(size[2]) > DRAWER_MAX_HEIGHT:
                continue
            track = Track(id=f"e:{next(self._ids)}", category_counts={"drawer": 5.0}, pos=centre.copy(),
                          var=np.full(3, 0.05 ** 2), yaw=yaw, yaw_var=0.1 ** 2, size=size.copy(),
                          observations=5, first_seen=self.stamp, last_seen=self.stamp, existence=0.9)
            track.voxels = _voxelize(piece)
            track.bottom_seen = True
            self.tracks[track.id] = track

    @staticmethod
    def _contradicts(track: Track, m: dict) -> bool:
        """A floor-standing track and a clearly raised region (or the reverse) are not the same
        object, for example a cabinet and the drawer at its top. Category disagreement alone is
        only a cost: partial views of glass are often misclassified."""
        m_bottom = m["centre"][2] - m["size"][2] / 2
        # A lowest point at an image edge or an occluder says nothing about being raised.
        raised_region = m["seg"].bottom_seen and m_bottom > 0.3
        raised_track = track.bottom_seen and track.bottom() > 0.3
        return (track.bottom() < 0.05 and raised_region) or (raised_track and m_bottom < 0.05)

    def _merge_parts(self, cam_p: np.ndarray) -> None:
        """Fold a track into a larger one (furniture, walls) when it lies inside the larger box at
        the same base height: both were partial views of the same object."""
        for big in sorted(self.tracks.values(), key=lambda t: -t.footprint):
            if big.id not in self.tracks or big.footprint <= 0.3 or big.held or big.category == "wall":
                continue
            for small in list(self.tracks.values()):
                if small.id == big.id or small.held or small.carried or small.pinned or big.pinned \
                        or small.footprint >= big.footprint:
                    continue
                if not _inside(big, small.pos, 0.1) or abs(small.bottom() - big.bottom()) > 0.15:
                    continue
                if _stacked(big.bottom(), big.top(), small.bottom(), small.top(), float(small.pos[2]),
                            big.bottom_seen and small.bottom_seen):
                    continue
                if (small.category == "bowl" and big.category == "bowl"
                        and (self._separate_bowls(small.footprint, big.footprint, small.pos[:2], big.pos[:2],
                                                  small.size, big.size)
                             or (small.bottom() > float(big.pos[2]) and small.footprint < 0.8 * big.footprint))):
                    continue  # a smaller bowl sitting in a larger one, even when the upper base was not seen:
                    # its base is above the larger bowl's middle, where a partial view of one bowl never is
                if small.footprint < 0.3 and small.bottom() > 0.05:
                    continue  # tableware on furniture is not part of it
                if big.category == "drawer" and small.footprint < 0.45 and small.bottom() > 0.45 \
                        and float(small.size[2]) < 0.20:
                    continue  # a smear of the contents shares the drawer floor and is still the contents
                if small.footprint >= 0.3 and small.observations >= ESTABLISHED_VIEWS and small.category != big.category:
                    continue  # an established piece of furniture is not a part of something else (a sideboard
                    # in front of a wall piece misread as furniture)
                if _drawer_beside_furniture(small, big):
                    continue  # a drawer shares the furniture's footprint and base, and is still its own object
                if (small.category == "drawer" and big.category == "drawer"
                        and float(small.size[2]) <= DRAWER_MAX_HEIGHT < float(big.size[2])):
                    continue  # the tall box is the furniture face; the short one is the drawer
                if small.voxels is not None:
                    big.voxels = self._union(big.voxels, small.voxels)
                _absorb_span(big, small)
                _absorb_views(big, small)
                for cat, n in small.category_counts.items():
                    big.category_counts[cat] = big.category_counts.get(cat, 0.0) + n
                _absorb_semantics(big, small)
                big_views = big.observations
                big.observations += small.observations
                big.existence = max(big.existence, small.existence)
                self._drop(small, "merged: part of a larger object", into=big)
                if small.observations > big_views:
                    # The better-known object keeps its id; the newer track is the one that ends.
                    self.dropped[-1].update(id=big.id, observations=big_views, into=small.id)
                    self._rename(big, small.id)
                if big.voxels is not None:
                    self._rebuild_geometry(big, cam_p, 2.0)

    def _fold_furniture_slices(self, cam_p: np.ndarray) -> None:
        """Fold a ``box`` that is the upper half of a cabinet, sideboard or counter back into it.

        The segmenter names that half ``box`` while the furniture keeps its own name. The slice's
        centre lies in the furniture's footprint and it does not rise above the furniture; a box
        standing on the top does, and stays a separate track. The ``box`` votes are not copied, so
        they cannot relabel the furniture."""
        hosts = ("cabinet", "sideboard", "counter", "table", "drawer")
        for small in list(self.tracks.values()):
            if small.id not in self.tracks or small.held or small.carried or small.perceived_category != "box":
                continue
            host = None
            for big in self.tracks.values():
                if big.id == small.id or big.held or big.category not in hosts or big.footprint < 0.4:
                    continue
                if small.footprint < 0.45 * big.footprint or not _inside(big, small.pos, 0.08):
                    continue
                if small.top() > big.top() + 0.06 or small.bottom() > big.top() + 0.05:
                    continue  # it stands on the top, or floats clear of it
                if small.top() < big.bottom() + 0.1:
                    continue
                if host is None or big.footprint > host.footprint:
                    host = big
            if host is None:
                continue
            if small.voxels is not None:
                host.voxels = self._union(host.voxels, small.voxels)
            _absorb_span(host, small)
            host.observations += small.observations
            host.existence = max(host.existence, small.existence)
            self._drop(small, "merged: slice of the furniture", into=host)
            if host.voxels is not None:
                self._rebuild_geometry(host, cam_p, 2.0)

    def _drop_support_top_drawers(self) -> None:
        """Drop a short wide ``drawer`` sitting on a support beside an object already tracked.

        A closed drawer is inside the furniture, so its base is well below the top. A shallow slab
        of the support's own top, shorter than a drawer and centred within 25 cm of a cup, bowl,
        plate, mug or glass, is that top boxed again. A taller track on a table, away from that
        tableware, stays: the green bowl is boxed about 22 cm tall and is the object. A tall box
        inside a counter or table and within 25 cm of that tableware is the counter face. A slab
        wider than 0.75 m is dropped only when it sits on that top; an open drawer is that long
        and stays. A drawer completed from the top band of a sideboard, past that sideboard's end
        and up near its top, is dropped; an open drawer is lower and stays."""
        hosts = [t for t in self.tracks.values()
                 if t.category in ("counter", "table", "cabinet", "sideboard") and t.footprint >= 0.4]
        small = [t for t in self.tracks.values()
                 if t.category in ("cup", "bowl", "plate", "mug", "wine_glass") and t.footprint < 0.3]
        for track in list(self.tracks.values()):
            if track.held or track.carried or track.category != "drawer" or track.footprint < 0.3:
                continue
            base = track.bottom()
            on_top = any(_inside(host, track.pos, 0.08) and abs(base - host.top()) <= 0.08
                         for host in hosts if host.id != track.id)
            beside = any(float(np.linalg.norm(track.pos[:2] - other.pos[:2])) <= 0.25 for other in small)
            wide = track.footprint > 0.75
            shallow_or_wide = float(track.size[2]) < 0.16 or wide
            # A tall box inside the counter, next to a bowl, is the counter face. A drawer in a
            # cabinet or sideboard has no cup or bowl beside it, so it stays.
            in_counter = any(host.category in ("counter", "table") and _inside(host, track.pos, 0.05)
                             for host in hosts if host.id != track.id)
            # The sideboard's top band, completed out to a drawer's size, lands past the
            # sideboard's end and up near its top. An open drawer slides out at drawer height.
            # A pulled drawer leaves the carcass. That is the open drawer, not a completed edge.
            opened = self.drawer_opening(track) >= 0.08 or track.id in self._opened_drawers
            past_end = (not opened) and any(
                host.category in ("cabinet", "sideboard", "counter")
                and float(np.linalg.norm(track.pos[:2] - host.pos[:2])) < 1.2
                and host.top() - float(track.pos[2]) <= 0.12
                and not _inside(host, track.pos, 0.05)
                for host in hosts if host.id != track.id)
            if (shallow_or_wide and on_top and (beside or wide)) or (in_counter and beside and not wide) or past_end:
                self._drop(track, "support top boxed as a drawer" if not past_end else
                           "drawer completed past the furniture")

    def _spawn_shed(self) -> None:
        """Turn points peeled off the top of furniture into their own tracks.

        The peel stops a bowl from lifting the counter's box. Those points are the bowl, and
        dropping them is how a look-around loses it.
        """
        spawned = 0
        while self._shed and spawned < 6:
            pts = self._shed.pop(0)
            for core in _shed_cores(pts):
                if spawned >= 6:
                    break
                centre, size, yaw = box_of(core, trim=0.01)
                if any(float(np.linalg.norm(t.pos[:2] - centre[:2])) < 0.15 and t.footprint < 0.35
                       for t in self.tracks.values()):
                    continue
                bottom = float(centre[2] - size[2] / 2)
                probs = self.category_probs(size, bottom, np.zeros(3), 0.0, True, None)
                track = Track(id=f"e:{next(self._ids)}", category_counts=dict(probs), pos=centre.copy(),
                              var=np.full(3, 0.05 ** 2), yaw=float(yaw), yaw_var=0.1 ** 2, size=size.copy(),
                              observations=3, first_seen=self.stamp, last_seen=self.stamp)
                track.voxels = _voxelize(core)
                track.bottom_seen = True
                self.tracks[track.id] = track
                self._rebuild_geometry(track, centre, 2.0)
                spawned += 1

    def _drop_false_tableware(self) -> None:
        """Drop a bowl longer than the bowl prior allows, a bowl thinner in plan than a bowl, and
        tableware the segmenter only called a drawer.

        The counter's near edge can be boxed as a bowl. :func:`max_long_extent` is that prior's
        longest side, not the widest bowl measured in one room. A patch of an open drawer that
        the segmenter names only ``drawer`` is completed to a plate and kept after the drawer
        shuts, because a closed drawer is not evidence of absence."""
        drawers = [t for t in self.tracks.values() if t.category == "drawer"]
        # The far lip of a counter is a wide shallow unknown. A cup on that counter is about 10 cm across.
        for track in list(self.tracks.values()):
            if track.held or track.category != "unknown":
                continue
            long = float(max(track.size[0], track.size[1]))
            if long < 0.32 or float(track.size[2]) > 0.12:
                continue
            if any(other.category in ("counter", "table") and other.id != track.id
                   and _inside(other, track.pos, 0.0)
                   and other.top() - 0.06 <= track.bottom() <= other.top() + 0.08
                   for other in self.tracks.values()):
                self._drop(track, "a strip of the support")
        # Tableware stands on its support, centre and all. A bowl read at the far edge of the
        # frame, with its centre past the edge of the table beside it, is that edge, not a bowl.
        supports = [t for t in self.tracks.values()
                    if t.category in ("counter", "table") and t.observations >= ESTABLISHED_VIEWS]
        for track in list(self.tracks.values()):
            if track.held or track.carried or track.pinned or track.category not in TABLEWARE:
                continue
            if track.bottom() < 0.3:
                continue  # on the floor or a low shelf, not beside a table top
            beside = [s for s in supports if abs(track.bottom() - s.top()) <= 0.12
                      and float(np.linalg.norm(track.pos[:2] - s.pos[:2])) < 0.5 * s.footprint + 0.3
                      and s.observations > track.observations]
            if beside and not any(_inside(s, track.pos, TABLEWARE_EDGE_M) for s in beside):
                self._drop(track, "tableware past the edge of its support")
        # The smallest bowl is 11 cm across. A smaller round box is a cup that the segmenter
        # called a bowl; left as a bowl it is merged with the counter strip beside it.
        for track in list(self.tracks.values()):
            if track.held or track.category != "bowl":
                continue
            long = float(max(track.size[0], track.size[1]))
            short = float(min(track.size[0], track.size[1]))
            height = float(track.size[2])
            if long <= 0.11 and short >= 0.05 and 0.07 <= height <= 0.14:
                track.category_counts = {"cup": float(min(max(track.observations, 1), 20))}
        for track in list(self.tracks.values()):
            if track.held or track.category not in ("bowl", "plate", "cup"):
                continue
            long = float(max(track.size[0], track.size[1]))
            cap = max_long_extent("bowl")
            if (track.category == "bowl" and cap is not None and long > cap and track.voxels is not None
                    and track.semantic_counts.get("bowl", 0.0) > 0.0):
                # The box is the bowl plus a strip of the counter. Cut back to the cluster that
                # stands on that strip before deciding the track is not a bowl.
                pts = _points(track.voxels)
                core = raised_object_points(pts)
                if core is None:
                    core = oversized_bowl_core(pts)
                if core is not None:
                    centre, size, yaw = box_of(core, trim=0.01)
                    if float(max(size[0], size[1])) <= cap:
                        track.voxels = _voxelize(core)
                        track.pos, track.size, track.yaw = centre, size, yaw
                        long = float(max(track.size[0], track.size[1]))
            if track.category == "bowl" and cap is not None and long > cap:
                # A segmenter bowl whose short side is still a bowl's can carry a strip of the
                # table and stay longer than the prior. A counter edge the prior alone called a
                # bowl has no segmenter score, or is only a few centimetres thick, and goes.
                short = float(min(track.size[0], track.size[1]))
                named_bowl = track.semantic_counts.get("bowl", 0.0) > 0.0 and short >= 0.12
                smaller = any(
                    other.category == "bowl" and other.id != track.id
                    and float(max(other.size[0], other.size[1])) <= cap
                    and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) < 0.20
                    for other in self.tracks.values())
                if smaller or not named_bowl:
                    self._drop(track, "bowl longer than the bowl prior allows")
                    continue
                axis = 0 if float(track.size[0]) >= float(track.size[1]) else 1
                track.size = track.size.copy()
                track.size[axis] = float(cap)
                continue
            # A bowl is round in plan. A few centimetres of thickness is the lip of a cabinet
            # seen edge-on, and that lip is close enough to a drawer to take its match. A cup is
            # 8 cm across and a partial view of it measures about 7 cm, so the lip test stays
            # under that.
            if track.category == "bowl" and float(min(track.size[0], track.size[1])) < 0.06:
                self._drop(track, "edge of a cabinet, thinner than a bowl")
                continue
            if track.category == "bowl" and float(track.size[2]) > 0.10 and any(
                    other.category == "table" and _inside(other, track.pos, 0.0)
                    and abs(track.bottom() - other.top()) <= 0.06
                    for other in self.tracks.values()):
                # The bowl is about 6 cm tall. The extra height is the table boxed with it,
                # and that extra lifts the centre.
                bottom = track.bottom()
                track.size = track.size.copy()
                track.size[2] = 0.07
                track.pos = track.pos.copy()
                track.pos[2] = bottom + 0.035
            # A bowl's prior is round. A box more than twice as long as it is wide is a strip of
            # the counter. The cup is often inside that strip; cut it out when nothing else is there.
            short = float(min(track.size[0], track.size[1]))
            if track.category == "bowl" and 0.0 < short <= 0.12 and long > 2.0 * short:
                if not self._recover_object_in_strip(track):
                    self._drop(track, "a strip of the support boxed as a bowl")
                    continue
            if track.category == "bowl" and any(
                    other.category == "cup" and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) < 0.10
                    for other in self.tracks.values()):
                self._drop(track, "a cup's body boxed as a bowl")
                continue
            plate_cap = max_long_extent("plate")
            if track.category == "plate" and plate_cap is not None and long > plate_cap:
                self._drop(track, "plate longer than the plate prior allows")
                continue
            # A bowl about 14 cm across is only a few centimetres tall. The plate prior is wider.
            height = float(track.size[2])
            if (track.category == "plate" and 0.10 <= long <= 0.18 and 0.025 <= height <= 0.08
                    and any("support-surface" in other.affordances and other.id != track.id
                            and _inside(other, track.pos, 0.0)
                            and other.top() - 0.05 <= float(track.pos[2]) <= other.top() + 0.08
                            for other in self.tracks.values())):
                track.category_counts = {"bowl": float(min(max(track.observations, 1), 20))}
                continue
            # A plate is about 2 cm thick, so its top sits that far above the surface.
            # A box whose top is within about a centimetre of the surface is the surface itself.
            if track.category == "plate" and any(
                    "support-surface" in other.affordances and other.id != track.id
                    and _inside(other, track.pos, 0.0) and track.top() < other.top() + 0.012
                    for other in self.tracks.values()):
                # A bowl about 14 cm across is only a few centimetres tall, so its top sits on
                # the table the way a patch of the tabletop does. A tabletop patch is wider.
                height = float(track.size[2])
                if 0.10 <= long <= 0.18 and 0.025 <= height <= 0.08:
                    track.category_counts = {"bowl": float(min(max(track.observations, 1), 20))}
                else:
                    self._drop(track, "the support surface boxed as a plate")
                continue
            if track.category == "plate" and not any(
                    "support-surface" in other.affordances and other.id != track.id
                    and _inside(other, track.pos, 0.0)
                    and other.top() - 0.05 <= float(track.pos[2]) <= other.top() + 0.05
                    for other in self.tracks.values()):
                self._drop(track, "a plate that is not standing on a surface")
                continue
            counts = track.semantic_counts
            if not counts or any(name != "drawer" and score > 0.0 for name, score in counts.items()):
                continue
            if any(float(np.linalg.norm(track.pos[:2] - d.pos[:2])) < 0.45 for d in drawers):
                self._drop(track, "drawer interior boxed as tableware")

    def _drop_false_chairs(self) -> None:
        """Drop a chair that cannot be one: it does not stand on the floor, or its centre lies in furniture.

        The chair prior stands on the floor and is about 0.9 m tall. A few centimetres of cabinet
        top, or the open drawer face, takes that name and then blocks the close."""
        hosts = [t for t in self.tracks.values()
                 if t.category in ("cabinet", "sideboard", "drawer") and t.footprint >= 0.3]
        for track in list(self.tracks.values()):
            if track.held:
                continue
            # The open cabinet, seen from the side, is named a table. That table sits in
            # front of the drawer and the close refuses to shut it.
            if track.category == "table" and any(
                    host.category in ("cabinet", "sideboard")
                    and float(np.linalg.norm(track.pos[:2] - host.pos[:2])) < 0.55
                    for host in hosts):
                self._drop(track, "furniture called a table")
                continue
            if track.category != "chair":
                continue
            if (float(track.size[2]) < 0.15 and track.footprint < 0.35
                    and self._beside_drawer(track.pos)):
                track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}
                continue
            if track.footprint > 0.55:
                self._drop(track, "furniture called a chair")
                continue
            if float(track.size[2]) < 0.35 or track.bottom() > 0.20:
                self._drop(track, "not a chair: it does not stand on the floor")
                continue
            if any(_inside(host, track.pos, 0.05) for host in hosts):
                self._drop(track, "furniture called a chair")
                continue
            # The open drawer and the cabinet face land just in front of the box. That chair
            # blocks the close, and neither scene has a chair of its own there.
            if any(float(np.linalg.norm(track.pos[:2] - host.pos[:2])) < 0.50 for host in hosts):
                self._drop(track, "furniture called a chair")
                continue
            if any(other.category == "table" and other.id != track.id and _inside(other, track.pos, 0.05)
                   for other in self.tracks.values()):
                self._drop(track, "furniture called a chair")

    def _contents_of_open_drawers(self) -> None:
        """Points standing on an open drawer's floor are the things inside it, not the drawer.

        A closed front is a panel. Once the drawer is pulled out, a compact cluster above its
        floor is its own track.
        """
        for drawer in list(self.tracks.values()):
            if drawer.category != "drawer" or drawer.voxels is None or drawer.held:
                continue
            if self.drawer_opening(drawer) >= 0.08:
                self._opened_drawers.add(drawer.id)
            # Seating puts the drawer back on the carcass while the look is still in it, which
            # used to skip the peel and leave the pistol inside the drawer cloud.
            if drawer.id not in self._opened_drawers:
                continue
            cloud = _points(drawer.voxels)
            if len(cloud) < 30:
                continue
            floor = float(np.percentile(cloud[:, 2], 30))
            # The pistol is only a few centimetres thick, so it sits in the floor band.
            raised = cloud[cloud[:, 2] >= floor + 0.008]
            for core in _shed_cores(raised):
                centre, size, yaw = box_of(core, trim=0.01)
                if any(float(np.linalg.norm(t.pos[:2] - centre[:2])) < 0.12 and t.footprint < 0.35
                       and t.id != drawer.id for t in self.tracks.values()):
                    continue
                if float(min(size[0], size[1])) < DRAWER_BOARD_M and not _inside(drawer, centre, -DRAWER_BOARD_M):
                    continue  # the drawer's own back or side board rises above its floor too
                bottom = float(centre[2] - size[2] / 2)
                probs = self.category_probs(size, bottom, np.full(3, 0.5), 0.0, True, None)
                track = Track(id=f"e:{next(self._ids)}", category_counts=dict(probs), pos=centre.copy(),
                              var=np.full(3, 0.04 ** 2), yaw=float(yaw), yaw_var=0.1 ** 2, size=size.copy(),
                              observations=3, first_seen=self.stamp, last_seen=self.stamp)
                track.voxels = _voxelize(core)
                track.bottom_seen = True
                self.tracks[track.id] = track

    def _recover_object_in_strip(self, track: Track) -> bool:
        """Keep the compact object inside a long thin bowl, and drop the strip when one is already tracked.

        :returns: true when ``track`` is now that object.
        """
        near = any(
            other.id != track.id and other.category in ("cup", "mug", "bowl")
            and float(max(other.size[0], other.size[1])) <= 2.0 * float(min(other.size[0], other.size[1]))
            and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) < 0.25
            for other in self.tracks.values())
        if near or track.voxels is None:
            return False
        pts = _points(track.voxels)
        core = raised_object_points(pts)
        if core is None:
            core = oversized_bowl_core(pts)
        if core is None:
            core = _densest_window(pts)
        if core is None:
            return False
        centre, size, yaw = box_of(core, trim=0.01)
        long = float(max(size[0], size[1]))
        short = float(min(size[0], size[1]))
        if short <= 0.0 or long > 2.0 * short or long > 0.20 or float(size[2]) < 0.04:
            return False
        # The cut keeps the rim. A cup-sized plan stands on the surface under it at the
        # cup's own height; the depth behind the glass is not the glass.
        if long <= 0.12 and float(size[2]) < 0.10:
            size = size.copy()
            size[2] = 0.10
            level = self._resting_level(track, centre, float(centre[2] - size[2] / 2))
            if level is not None:
                centre = centre.copy()
                centre[2] = level + float(size[2]) / 2
        track.voxels = _voxelize(core)
        track.pos, track.size, track.yaw = centre, size, yaw
        color = track.mean_rgb if track.mean_rgb is not None else np.full(3, 0.5)
        probs = self.category_probs(size, float(centre[2] - size[2] / 2), color, track.mean_invalid,
                                    True, None)
        if probs:
            weight = float(min(max(track.observations, 1), 20))
            track.category_counts = {k: v * weight for k, v in probs.items()}
        return True

    def _drop_furniture_edges(self) -> None:
        """Drop a sliver of a cabinet and a bowl or unknown painted on its top edge.

        The top rail is a few centimetres of the cabinet. It is not a second cabinet, and a
        bowl-sized patch of it is not a bowl. An object standing on the top sits above the rail.
        """
        hosts = [t for t in self.tracks.values()
                 if t.category in ("cabinet", "sideboard") and float(t.size[2]) > 0.4]
        for track in list(self.tracks.values()):
            if track.held or track.id not in self.tracks:
                continue
            thin = float(min(track.size[0], track.size[1]))
            if track.category in ("cabinet", "sideboard") and (float(track.size[2]) < 0.10 or thin < 0.25):
                fuller = any(other.id != track.id and other.category == track.category
                             and float(other.size[2]) > 0.4 and other.footprint > track.footprint
                             and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) < 0.6
                             for other in self.tracks.values())
                if fuller:
                    self._drop(track, "slice of the furniture")
                continue
            if track.category == "pill_bottle" and float(track.size[2]) < 0.06:
                self._drop(track, "pill flatter than the bottle")
                continue
            if track.category not in ("bowl", "plate", "unknown") or track.footprint >= 0.45:
                continue
            for host in hosts:
                if not _inside(host, track.pos, 0.0):
                    continue
                if track.category == "unknown" and float(track.size[2]) < 0.04:
                    if (self._beside_drawer(track.pos) and float(max(track.size[0], track.size[1])) < 0.25
                            and not self._under_drawer_floor(track)):
                        track.category_counts = {"pistol": float(min(max(track.observations, 1), 20))}
                        continue
                    self._drop(track, "thin unknown on the furniture")
                    break
                gap = host.top() - track.bottom()
                if 0.02 < gap < 0.20:
                    self._drop(track, "edge of the furniture")
                    break

    def _drop_pistols_off_the_drawer(self) -> None:
        """Drop a pistol that sits on the furniture beside a closed drawer, not in it.

        The pistol prompt also fires on the dark ends of the sideboard. Those tracks are
        half a metre from the drawer and match nothing that is there.
        """
        for track in list(self.tracks.values()):
            if track.held or track.id not in self.tracks:
                continue
            # The pistol in a drawer is about 0.63 m up. Higher than that is the rail
            # or the top of the sideboard, including a smear that sticks out past the box.
            if (track.category == "pistol" and float(track.pos[2]) > 0.70
                    and any(other.category == "sideboard"
                            and float(np.linalg.norm(track.pos[:2] - other.pos[:2])) < 0.80
                            for other in self.tracks.values())):
                self._drop(track, "off the drawer, on the furniture around it")
                continue
            if track.category == "pistol" and any(
                    other.category == "cabinet" and _inside(other, track.pos, 0.02)
                    for other in self.tracks.values()):
                # A long flat pistol at the back of the cabinet drawer is the airplane.
                # The false pistols on the carcass are about 11 cm long.
                if float(max(track.size[0], track.size[1])) >= 0.16 and float(track.size[2]) <= 0.06:
                    track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}
                else:
                    self._drop(track, "a pistol inside the cabinet")
                continue
            if track.footprint >= 0.25 and track.category != "pistol":
                continue
            if track.category == "pistol" and track.footprint >= 0.45:
                continue
            if track.category not in ("pistol", "mug", "cup", "bowl", "box", "plate", "unknown", "bottle"):
                continue
            for drawer in self.tracks.values():
                if drawer.category != "drawer" or drawer.held or self.drawer_opening(drawer) >= 0.08:
                    continue
                if float(np.linalg.norm(track.pos[:2] - drawer.pos[:2])) > 0.90:
                    continue
                if not (drawer.bottom() - 0.05 <= float(track.pos[2]) <= drawer.top() + 0.15):
                    continue
                host = self._furniture_of(drawer)
                if host is None or not _inside(host, track.pos, 0.0):
                    continue
                if track.category != "pistol" and float(track.pos[2]) > host.top() - 0.02:
                    continue  # on the top, not in the drawer band. A pistol there is the sideboard end.
                if track.category == "pistol" and float(track.pos[2]) > drawer.top() - 0.08:
                    self._drop(track, "off the drawer, on the furniture around it")
                    break  # the wide drawer prior also covers the rail above the drawer
                if _inside(drawer, track.pos, 0.10):
                    break
                self._drop(track, "off the drawer, on the furniture around it")
                break

    def _name_the_pistol_in_the_sideboard(self) -> None:
        """A flat box in the sideboard drawer is the pistol, and the smears beside it are not.

        The pistol is a few centimetres thick. A taller unknown in that drawer is the face of
        the sideboard. A wine glass there is the same smear named again. The wine glass on
        the table is not beside a drawer.
        """
        for track in list(self.tracks.values()):
            if track.held or track.id not in self.tracks:
                continue
            if track.category == "wine_glass" and self._beside_drawer(track.pos):
                self._drop(track, "a glass in the drawer")
                continue
            if track.category == "unknown" and float(track.size[2]) >= 0.10 \
                    and float(max(track.size[0], track.size[1])) >= 0.18:
                host = next((other for other in self.tracks.values()
                             if other.category == "sideboard" and _inside(other, track.pos, 0.0)), None)
                if host is not None:
                    self._drop(track, "a smear of the sideboard")
                continue
            if track.category not in ("box", "unknown", "plate") or float(track.size[2]) > 0.05:
                continue
            long = float(max(track.size[0], track.size[1]))
            short = float(min(track.size[0], track.size[1]))
            if long > 0.22 or short > 0.12 or not self._beside_drawer(track.pos):
                continue
            host = next((other for other in self.tracks.values()
                         if other.category == "sideboard" and _inside(other, track.pos, 0.05)), None)
            if host is None:
                continue
            if self._under_drawer_floor(track):
                continue
            track.category_counts = {"pistol": float(min(max(track.observations, 1), 20))}

    def _under_drawer_floor(self, track: Track) -> bool:
        """Whether a flat box beside a drawer lies below that drawer's floor: the shadow in the
        gap under a pulled-out drawer, which nothing inside the drawer can reach."""
        return any(other.category == "drawer" and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) <= 0.70
                   and track.bottom() < other.bottom() - 0.03 for other in self.tracks.values())

    def _seat_contents_in_closed_drawers(self) -> None:
        """Pull a drawer content back to the carcass depth when the cloud sits toward the room.

        The same view that swells the furniture toward the camera leaves the bear in front of
        the drawer it is standing in. The along-drawer position is the one the cloud measured.
        """
        for drawer in self.tracks.values():
            if drawer.category != "drawer" or drawer.held or self.drawer_opening(drawer) >= 0.08:
                continue
            host = self._furniture_of(drawer)
            seat = self._closed_centre(drawer)
            if host is None or seat is None:
                continue
            centre, depth_axis = seat
            host_depth = float(host.pos[:2] @ depth_axis)
            for track in self.tracks.values():
                # Bottles and the airplane have their own depth in the drawer. Pulling every
                # content onto the carcass centre stacks the pill on the perfume.
                if track.held or track.category not in ("teddy_bear", "pistol"):
                    continue
                if float(np.linalg.norm(track.pos[:2] - host.pos[:2])) > 0.55:
                    continue
                if not (drawer.bottom() - 0.05 <= float(track.pos[2]) <= drawer.top() + 0.08):
                    continue
                along = float(track.pos[:2] @ depth_axis)
                # The bear was measured on the carcass. The pistol lies on the drawer floor,
                # one panel-thickness toward the room of that face.
                target = host_depth if track.category == "teddy_bear" else float(centre @ depth_axis)
                if along <= target + 0.02:
                    continue
                delta = target - along
                # The pistol is seen in front of the open drawer, about 25 cm out. The bear is not.
                if delta < (-0.30 if track.category == "pistol" else -0.15):
                    continue
                track.pos = track.pos.copy()
                track.pos[:2] = track.pos[:2] + depth_axis * delta
            for track in self.tracks.values():
                if track.held or track.category not in ("toy_airplane", "perfume_bottle", "teddy_bear", "pistol"):
                    continue
                if float(np.linalg.norm(track.pos[:2] - host.pos[:2])) > 0.55:
                    continue
                # A drawer boxed down into the cabinet has a floor below the real one. Setting
                # the airplane on that floor drops it through the drawer.
                if float(drawer.size[2]) > 0.30:
                    continue
                floor = float(drawer.bottom()) + max(0.0, float(drawer.size[2]) - 0.22) / 2.0
                gap = track.bottom() - floor
                if 0.04 < gap <= 0.15:
                    track.pos = track.pos.copy()
                    track.pos[2] -= gap

    def _inside_shut_furniture(self, centre: np.ndarray, size: np.ndarray, footprint: float) -> bool:
        """Whether a small box lies in the carcass band of a cabinet or sideboard whose drawers are
        all shut.

        The camera cannot see into shut furniture. A box there, or hanging on its front below
        the top, is depth noise on the handle, the rail or the drawer gap, not an object.

        :param centre: box centre, metres.
        :param size: full box extents, metres.
        :param footprint: the box's plan extent, metres.
        :returns: true when the box is in that band and no drawer of the piece is pulled out.
        """
        if footprint >= 0.45:
            return False
        top = float(centre[2]) + float(size[2]) / 2.0
        bottom = float(centre[2]) - float(size[2]) / 2.0
        for host in self.tracks.values():
            if host.category not in ("cabinet", "sideboard") or host.footprint < 0.4:
                continue
            if host.observations < ESTABLISHED_VIEWS or float(host.size[2]) < 0.4:
                continue
            if not _inside(host, centre[:2], 0.06):
                continue
            if top > host.top() + 0.03 or bottom > host.top() - 0.03 or bottom < host.bottom() + 0.05:
                continue  # on the top, or standing on the floor beside it
            if self._host_drawer_open(host):
                continue
            return True
        return False

    def _drop_inside_shut_furniture(self) -> None:
        """Drop a small track that sits in the carcass band of shut furniture, unless it is a
        content the drawer closed on (:meth:`_inside_shut_furniture`).

        A drawer closes on a content only after the robot has moved that drawer. A content name
        on furniture whose drawers the robot never moved is the handle or the drawer gap.
        """
        for track in list(self.tracks.values()):
            if track.held or track.carried or track.pinned or track.id not in self.tracks:
                continue
            if track.category in ("drawer", "cabinet", "sideboard", "counter", "table", "wall"):
                continue
            if self._beside_moved_drawer(track.pos):
                if any(name in _CONTENT for name in track.semantic_counts):
                    continue  # the segmenter named it a content; the drawer closed on it
                if track.category in _CONTENT and self._in_closed_drawer(track):
                    continue
            if self._inside_shut_furniture(track.pos, track.size, track.footprint):
                self._drop(track, "inside shut furniture")

    def _beside_moved_drawer(self, centre: np.ndarray) -> bool:
        """Whether a drawer the robot has pulled out or pushed back (:meth:`shift`) is within
        1 m of ``centre`` in plan: the reach inside which :meth:`_host_drawer_open` takes a
        drawer as the furniture's own. A box that only measures as open does not count."""
        return any(drawer.category == "drawer" and self._moved_by_robot(drawer)
                   and float(np.linalg.norm(drawer.pos[:2] - centre[:2])) < 1.0
                   for drawer in self.tracks.values())

    def _host_drawer_open(self, host: Track) -> bool:
        """Whether a drawer of ``host`` is pulled out.

        The open drawer is the near face of the cabinet. Pulling that face back to the prior
        would move the cabinet while the robot is still standing in front of it.
        """
        for drawer in self.tracks.values():
            if drawer.category != "drawer" or drawer.held or drawer.id == host.id:
                continue
            if drawer.id in self._pulled_open or drawer.pinned:
                if float(np.linalg.norm(drawer.pos[:2] - host.pos[:2])) < 1.0:
                    return True
                continue
            if drawer.id in self._pushed_shut:
                continue  # the robot pushed it back; the box it left swollen is not an opening
            if self._furniture_of(drawer) is not host:
                continue
            if self._drawer_open(drawer):
                return True
        return False

    def _seat_swollen_drawer(self) -> None:
        """Put a drawer that includes the carcass back in the band under the top.

        A drawer front is about 22 cm. A box twice that tall has its centre in the carcass,
        and the handle plan then misses. The top slab of the furniture stays above the drawer.
        """
        proto = prototype("drawer")
        if proto is None:
            return
        height = float(proto[2])
        for track in self.tracks.values():
            if track.held or track.category != "drawer" or float(track.size[2]) <= 0.30:
                continue
            if self.drawer_opening(track) >= 0.08:
                continue
            host = self._furniture_of(track)
            if host is None:
                continue
            centre_z = host.top() - 0.03 - height / 2.0
            if not (host.bottom() < centre_z < host.top()):
                continue
            track.size = track.size.copy()
            track.size[2] = height
            track.pos = track.pos.copy()
            track.pos[2] = centre_z

    def _recede_facing_excess(self, cam_p: np.ndarray) -> None:
        """Shift a near face back to the prior, away from the camera.

        Depth noise adds a few centimetres on the side toward the camera, and the centre
        moves with it. The far face is already the object. A side across the view is the
        length of the piece. An open drawer's host is left as the approach measured it.
        """
        cam = np.asarray(cam_p[:2], float)
        for track in list(self.tracks.values()):
            if track.held or track.id not in self.tracks:
                continue
            if track.category in ("cabinet", "sideboard"):
                # Skip only while a drawer is clearly pulled out. A shut drawer seated on the
                # front still reads a few centimetres of opening and must not block the recede.
                if any(d.category == "drawer" and not d.held and self._furniture_of(d) is track
                       and self.drawer_opening(d) >= 0.15
                       for d in self.tracks.values()):
                    continue
            if track.category == "cup":
                if not self._stands_on_counter(track.pos, track.bottom()):
                    continue
            elif track.category not in ("cabinet", "sideboard", "counter", "table", "drawer"):
                continue
            proto = prototype(track.category)
            if proto is None:
                continue
            away = np.asarray(track.pos[:2], float) - cam
            norm = float(np.linalg.norm(away))
            if norm < 0.3:
                continue
            view = away / norm
            yaw = float(track.yaw)
            directions = (
                np.array([math.cos(yaw), math.sin(yaw)]),
                np.array([-math.sin(yaw), math.cos(yaw)]),
            )
            align = [abs(float(view @ direction)) for direction in directions]
            axis = 0 if align[0] >= align[1] else 1
            if align[axis] < 0.7:
                continue
            long_axis = 0 if float(track.size[0]) >= float(track.size[1]) else 1
            # Cabinet and sideboard are seen from the front: the camera axis is the shallow depth.
            if track.category in ("cabinet", "sideboard"):
                want = float(min(proto[0], proto[1]))
                limit = 0.25
            else:
                want = float(proto[0] if axis == long_axis else proto[1])
                limit = 0.10
            excess = float(track.size[axis]) - want
            if excess < 0.02 or excess > limit:
                continue
            direction = directions[axis]
            sign = 1.0 if float(view @ direction) >= 0.0 else -1.0
            track.size = track.size.copy()
            track.size[axis] = want
            track.pos = track.pos.copy()
            track.pos[:2] = track.pos[:2] + direction * sign * (excess / 2.0)

    def _prefer_tight_bowl_views(self) -> None:
        """A bowl grown out to the prior's largest size takes the centre of its smaller views.

        The smaller views are the bowl. The large box includes the table beside it, and that
        centre is what the match then uses. A smaller view is round in plan, as a bowl is
        (:func:`_tight_bowl_views`): a sliver of a large bowl cut by the edge of the frame is
        smaller too, and two of those would put the whole bowl on its rim.
        """
        for track in self.tracks.values():
            if track.held or track.category != "bowl":
                continue
            if float(max(track.size[0], track.size[1])) <= 0.26 or len(track.recent) < 2:
                continue
            tight = _tight_bowl_views(track)
            if len(tight) < 2:
                continue
            track.pos = np.median([row[0] for row in tight], axis=0)
            track.size = np.median([row[1] for row in tight], axis=0)

    def _pull_bowl_off_the_lip(self, cam_p: np.ndarray) -> None:
        """Keep the near face of a bowl that was boxed together with the far lip of its support.

        The wide camera paints that lip and the bowl as one flat slab, so the centre sits on the
        lip. Two tight views already say where the bowl is; two bowl-sized views give the box
        back its size and centre. Without them, the face nearer the camera is the bowl, and the
        side that reaches the far end of the counter or table is the lip.
        """
        cam = np.asarray(cam_p[:2], float)
        kept = 0.17
        for track in self.tracks.values():
            if track.held or track.category != "bowl":
                continue
            if float(max(track.size[0], track.size[1])) <= 0.32:
                continue
            tight = _tight_bowl_views(track)
            if len(tight) >= 2:
                continue
            bowl_sized = [row for row in track.recent if float(max(row[1][0], row[1][1])) <= 0.32]
            if len(bowl_sized) >= 2:
                # The views are a bowl; only the box drawn around their union, or around a cloud
                # merged in from a smear, is wider than one. Those views are the bowl.
                centre, size = _median_of_closer_views(bowl_sized)
                track.pos = track.pos.copy()
                track.pos[:2] = centre[:2]
                track.size = track.size.copy()
                track.size[:2] = size[:2]
                continue
            support = self._support_under_bowl(track)
            if support is None:
                continue
            rel = cam - np.asarray(support.pos[:2], float)
            yaw = float(support.yaw)
            directions = (
                np.array([math.cos(yaw), math.sin(yaw)]),
                np.array([-math.sin(yaw), math.cos(yaw)]),
            )
            short = 0 if float(support.size[0]) <= float(support.size[1]) else 1
            depth = directions[short]
            sign_far = -1.0 if float(rel @ depth) >= 0.0 else 1.0
            lip = np.asarray(support.pos[:2], float) + depth * sign_far * float(support.size[short]) / 2.0
            far = self._bowl_face(track, depth, sign_far)
            if float(depth @ (lip - far)) > 0.10:
                continue
            near = self._bowl_face(track, depth, -sign_far)
            centre = near + depth * sign_far * (kept / 2.0)
            track.pos = track.pos.copy()
            track.pos[:2] = centre
            track.size = track.size.copy()
            track.size[0] = min(float(track.size[0]), kept)
            track.size[1] = min(float(track.size[1]), kept)
            self._fill_bowl_voxels(track)
            track.recent.clear()
            track.recent.append((track.pos.copy(), track.size.copy(), 1.0))

    def _support_under_bowl(self, track: Track) -> Track | None:
        """The counter or table whose top the bowl is standing on."""
        for other in self.tracks.values():
            if other.category not in ("counter", "table") or other.id == track.id:
                continue
            if abs(track.bottom() - other.top()) > 0.08:
                continue
            if _inside(other, track.pos, 0.05):
                return other
        return None

    def _bowl_face(self, track: Track, direction: np.ndarray, sign: float) -> np.ndarray:
        """The centre of one horizontal face of ``track`` along ``direction``."""
        yaw = float(track.yaw)
        axes = (
            np.array([math.cos(yaw), math.sin(yaw)]),
            np.array([-math.sin(yaw), math.cos(yaw)]),
        )
        face = np.asarray(track.pos[:2], float)
        for axis, length in zip(axes, (float(track.size[0]), float(track.size[1]))):
            face = face + axis * sign * abs(float(direction @ axis)) * length / 2.0
        return face

    def _fill_bowl_voxels(self, track: Track) -> None:
        """Replace the cloud with the box just chosen, so the next view does not grow the lip back."""
        yaw = float(track.yaw)
        axes = (
            np.array([math.cos(yaw), math.sin(yaw), 0.0]),
            np.array([-math.sin(yaw), math.cos(yaw), 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        samples = []
        steps = (4, 4, 2)
        for step, axis, length in zip(steps, axes, (float(track.size[0]), float(track.size[1]), float(track.size[2]))):
            samples.append(np.linspace(-length / 2.0, length / 2.0, step))
        grid = np.array(np.meshgrid(*samples, indexing="ij")).reshape(3, -1)
        points = np.asarray(track.pos, float) + grid.T @ np.vstack(axes)
        track.voxels = _voxelize(points)

    def _drop_cups_on_the_bear(self) -> None:
        """Drop a cup sitting on the bear.

        That cup is the bear seen again, and it is near enough to the pistol to take its match.
        """
        bears = [t for t in self.tracks.values() if t.category == "teddy_bear" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.category not in ("cup", "mug"):
                continue
            if any(float(np.linalg.norm(track.pos[:2] - bear.pos[:2])) < 0.16 for bear in bears):
                self._drop(track, "a cup on the bear")

    def _cups_on_the_counter_stay_cups(self) -> None:
        """A wine glass standing on the counter is the cup.

        The stem override names a tall cup a glass. The glass in these rooms stands on the table.
        """
        for track in self.tracks.values():
            if track.held or track.category != "wine_glass":
                continue
            if not self._stands_on_counter(track.pos, track.bottom()):
                continue
            if float(track.size[2]) > 0.10:
                extra = float(track.size[2]) - 0.10
                track.size = track.size.copy()
                track.size[2] = 0.10
                track.pos = track.pos.copy()
                track.pos[2] -= extra / 2.0
            track.category_counts = {"cup": float(min(max(track.observations, 1), 20))}

    def _name_unknown_bowls(self) -> None:
        """An unnamed shallow box on a table is a bowl the segmenter did not label.

        The smallest bowl is still about 8 cm across. A drawer content is not renamed here.
        """
        for track in self.tracks.values():
            if track.held or track.category != "unknown" or self._beside_drawer(track.pos):
                continue
            long = float(max(track.size[0], track.size[1]))
            short = float(min(track.size[0], track.size[1]))
            if (track.bottom() > 0.60 and float(track.size[2]) <= 0.09
                    and 0.08 <= short <= 0.22 and long <= 0.28):
                track.category_counts = {"bowl": float(min(max(track.observations, 1), 20))}
        for track in self.tracks.values():
            # The blue bowl is often labelled a plate. A real plate is thinner than 3 cm.
            if track.held or track.category != "plate":
                continue
            if float(track.size[2]) < 0.035 or track.bottom() < 0.60:
                continue
            if not any(other.category == "table" and _inside(other, track.pos, 0.05)
                       for other in self.tracks.values()):
                continue
            track.category_counts = {"bowl": float(min(max(track.observations, 1), 20))}
        for track in list(self.tracks.values()):
            # A bowl boxed in the cabinet drawer band is the drawer face, not a dish.
            if track.held or track.category != "bowl" or track.id not in self.tracks:
                continue
            if not any(other.category == "cabinet"
                       and float(np.linalg.norm(track.pos[:2] - other.pos[:2])) < 0.55
                       for other in self.tracks.values()):
                continue
            if track.bottom() > 0.55 and float(track.pos[2]) < 0.75:
                self._drop(track, "a bowl in the cabinet drawer")

    def _rename_airplanes_called_bears(self) -> None:
        """A bear or an unknown the segmenter called an airplane, inside a cabinet, is the airplane.

        The shape prior says bear once the drawer floor makes the box tall, or leaves it unknown.
        The segmenter's name is the airplane, and dropping that track loses it.
        """
        cabinets = [t for t in self.tracks.values() if t.category == "cabinet" and not t.held]
        for track in self.tracks.values():
            if track.held or track.category not in ("teddy_bear", "unknown"):
                continue
            air = float(track.semantic_counts.get("toy_airplane", 0.0))
            bear = float(track.semantic_counts.get("teddy_bear", 0.0))
            if air <= 0.0 or air < bear:
                continue
            if not any(_inside(cabinet, track.pos, 0.05) for cabinet in cabinets):
                continue
            track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}

    def _drop_contents_on_the_wrong_piece(self) -> None:
        """Drop a bottle on the sideboard and a second airplane left in front of the cabinet.

        The pill and the perfume stand in the cabinet drawer. A bottle in front of the sideboard
        is a smear. Two airplanes are one object seen from the open drawer and again after the close.
        """
        drawers = [t for t in self.tracks.values() if t.category == "drawer" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category not in ("pill_bottle", "perfume_bottle"):
                continue
            if not drawers:
                continue
            nearest = min(drawers, key=lambda d: float(np.linalg.norm(d.pos[:2] - track.pos[:2])))
            if float(np.linalg.norm(nearest.pos[:2] - track.pos[:2])) > 0.55:
                continue
            host = self._furniture_of(nearest)
            if host is not None and host.category == "sideboard":
                self._drop(track, "a bottle on the sideboard")
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "toy_airplane":
                continue
            if float(max(track.size[0], track.size[1])) > 0.40:
                self._drop(track, "an airplane the size of the furniture")
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category not in ("pill_bottle", "perfume_bottle"):
                continue
            if float(max(track.size[0], track.size[1])) > 0.20:
                self._drop(track, "a bottle the size of the furniture")
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        cabinet_drawers = []
        for drawer in drawers:
            host = self._furniture_of(drawer)
            if host is not None and host.category == "cabinet":
                cabinet_drawers.append(drawer)
        for plane in list(planes):
            if plane.id not in self.tracks:
                continue
            for drawer in cabinet_drawers:
                near = float(np.linalg.norm(plane.pos[:2] - drawer.pos[:2]))
                if near > 0.08:
                    continue
                farther = any(
                    other.id != plane.id and other.category == "toy_airplane" and not other.held
                    and 0.08 < float(np.linalg.norm(other.pos[:2] - drawer.pos[:2])) < 0.35
                    for other in self.tracks.values())
                if farther:
                    self._drop(plane, "an airplane on the drawer centre")
                    break
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        if len(planes) >= 2 and cabinet_drawers:
            def gap(plane: Track) -> float:
                return min(float(np.linalg.norm(plane.pos[:2] - d.pos[:2])) for d in cabinet_drawers)
            planes.sort(key=gap)
            for extra in planes[1:]:
                if extra.id in self.tracks and gap(extra) > gap(planes[0]) + 0.12:
                    self._drop(extra, "an airplane still out in front")

    def _drop_stray_bears(self) -> None:
        """Drop a bear that is not in a drawer when another bear is already in one.

        The second bear is a smear of the sideboard, and it is close enough to the pistol
        to be matched to it.
        """
        drawers = [t for t in self.tracks.values() if t.category == "drawer"]
        bears = [t for t in self.tracks.values() if t.category == "teddy_bear" and not t.held]

        def nearest(track: Track) -> float:
            return min(float(np.linalg.norm(track.pos[:2] - d.pos[:2])) for d in drawers)

        if not drawers or not any(nearest(bear) <= 0.25 for bear in bears):
            return
        for bear in list(bears):
            if bear.id not in self.tracks:
                continue
            if nearest(bear) > 0.35:
                self._drop(bear, "a bear that is not in a drawer")
                continue
            drawer = min(drawers, key=lambda item: float(np.linalg.norm(bear.pos[:2] - item.pos[:2])))
            seat = self._closed_centre(drawer)
            if seat is None:
                continue
            centre, depth_axis = seat
            lateral = np.array([-float(depth_axis[1]), float(depth_axis[0])])
            # Sideways, a drawer is where its seat is, open or shut. The box of an open drawer
            # takes in the furniture top beside it, and its centre moves that way.
            if abs(float((bear.pos[:2] - np.asarray(centre[:2], float)) @ lateral)) > 0.18:
                self._drop(bear, "a bear off to the side of the drawer")

    def _rename_flat_mugs_in_drawers(self) -> None:
        """A flat mug brought back into a drawer is the pistol.

        The name is decided while the drawer is open and the mug is still out in front, where
        it is not beside the drawer. The close puts it back, and that move does not rebuild it.
        """
        for track in self.tracks.values():
            if track.held or track.category not in ("mug", "cup"):
                continue
            if not self._beside_drawer(track.pos):
                continue
            long = float(max(track.size[0], track.size[1]))
            short = float(min(track.size[0], track.size[1]))
            if float(track.size[2]) <= 0.06 and 0.04 <= long <= 0.22 and short <= 0.14:
                track.category_counts = {"pistol": float(min(max(track.observations, 1), 20))}

    def _drop_unknowns_on_the_airplane(self) -> None:
        """Drop an unknown whose centre sits on the airplane.

        That cloud is the airplane seen without a name, and it is near enough to the perfume
        to take its match.
        """
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.category != "unknown" or track.held:
                continue
            if any(float(np.linalg.norm(track.pos[:2] - plane.pos[:2])) < 0.12
                   and abs(float(track.pos[2] - plane.pos[2])) < 0.08 for plane in planes):
                self._drop(track, "an unknown on the airplane")

    def _drop_perfume_smears(self) -> None:
        """Drop a perfume longer than the bottle.

        The bottle is under 7 cm. A 15 cm box with that name is a smear of the drawer front,
        and it is a false object beside the real contents.
        """
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "perfume_bottle":
                continue
            if float(max(track.size[0], track.size[1])) > 0.12:
                self._drop(track, "a smear named perfume")

    def _rename_wide_perfumes(self) -> None:
        """A perfume box as wide as it is long is the airplane.

        The bottle is a few centimetres on its short side. The square box at the back of the
        drawer is the airplane, and leaving the perfume name on it takes the airplane's match.
        """
        for track in self.tracks.values():
            if track.held or track.category != "perfume_bottle" or not self._beside_drawer(track.pos):
                continue
            short = float(min(track.size[0], track.size[1]))
            long = float(max(track.size[0], track.size[1]))
            # The bottle's short side is about 4 cm. Anything wider, beside the drawer, is the airplane.
            if long < 0.28 and short > 0.06:
                track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}
        for track in self.tracks.values():
            if track.held or track.category != "pill_bottle" or not self._beside_drawer(track.pos):
                continue
            # The pill is a few centimetres across. A square box at the back of the drawer is
            # the airplane the shape prior called a pill. A box the size of the cabinet is not.
            short = float(min(track.size[0], track.size[1]))
            long = float(max(track.size[0], track.size[1]))
            if 0.09 < short and long < 0.28:
                track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        for track in self.tracks.values():
            if track.held or track.category != "pill_bottle":
                continue
            if any(track.semantic_counts.get(name, 0.0) > 0.0 for name in ("bottle", "pill_bottle", "perfume_bottle")):
                continue  # the segmenter saw a bottle; a plane box beside it is not a reason to rename it
            if any(float(np.linalg.norm(track.pos[:2] - plane.pos[:2])) < 0.12
                   and abs(float(track.pos[2] - plane.pos[2])) < 0.08 for plane in planes):
                track.category_counts = {"toy_airplane": float(min(max(track.observations, 1), 20))}

    def _name_pink_pills(self) -> None:
        """A pink pill beside a drawer is the perfume the shape prior swallowed.

        The pill is white. Pink glass stays pink after the drawer floor mixes in, and the two
        bottles are then different tracks instead of one pill.
        """
        for track in self.tracks.values():
            if track.held or track.category != "pill_bottle" or track.mean_rgb is None:
                continue
            if not self._beside_drawer(track.pos):
                continue
            red, green, blue = (float(channel) for channel in track.mean_rgb)
            washed = (red - min(green, blue)) / red if red > 0.0 else 0.0
            long = float(max(track.size[0], track.size[1]))
            short = float(min(track.size[0], track.size[1]))
            if red > blue > green and red > 0.55 and washed > 0.08 and long < 0.14 and short < 0.10:
                track.category_counts = {"perfume_bottle": float(min(max(track.observations, 1), 20))}

    def _drop_bottles_beside_the_drawer(self) -> None:
        """Drop a second bottle that is not in the closed drawer.

        The one inside is the bottle. The one on the cabinet top, or out past the side, is a smear.
        """
        for drawer in list(self.tracks.values()):
            if drawer.category != "drawer" or drawer.held or self.drawer_opening(drawer) >= 0.08:
                continue
            for category in ("pill_bottle", "perfume_bottle"):
                group = [t for t in self.tracks.values()
                         if t.category == category and not t.held
                         and float(np.linalg.norm(t.pos[:2] - drawer.pos[:2])) < 0.70
                         and drawer.bottom() - 0.05 <= float(t.pos[2]) <= drawer.top() + 0.08]
                inside = [t for t in group if _inside(drawer, t.pos, 0.02)
                          and float(min(t.size[0], t.size[1])) <= 0.09
                          and not self._perfume_on_a_pill(t, 0.22)]
                if not inside:
                    continue
                for track in group:
                    if track.id in self.tracks and track not in inside:
                        self._drop(track, "a bottle outside the closed drawer")

    def _perfume_on_a_pill(self, track: Track, reach: float = 0.05) -> bool:
        """Whether ``track`` is a perfume sitting on a pill.

        That perfume is the pill's second name. It is not the bottle in the drawer, so it must
        not be the copy that makes the real bottle look like a duplicate. ``reach`` is the plan
        distance, metres, inside which the perfume is the pill's.

        :param track: the perfume candidate.
        :param reach: plan distance to a pill, metres.
        :returns: true when a pill is within ``reach``.
        """
        if track.category != "perfume_bottle":
            return False
        return any(
            pill.category == "pill_bottle" and not pill.held and pill.id != track.id
            and float(np.linalg.norm(track.pos[:2] - pill.pos[:2])) < reach
            and abs(float(track.pos[2] - pill.pos[2])) < 0.06
            for pill in self.tracks.values())

    def _drop_perfume_on_the_pill(self) -> None:
        """Drop a perfume sitting on the pill.

        The shape prior names the pill a perfume. The other perfume, a hand's breadth in front,
        is the bottle. The one on the pill is the pill's second name.
        """
        pills = [t for t in self.tracks.values() if t.category == "pill_bottle" and not t.held]
        perfumes = [t for t in self.tracks.values() if t.category == "perfume_bottle" and not t.held]
        if len(perfumes) < 2:
            return
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "perfume_bottle":
                continue
            if any(float(np.linalg.norm(track.pos[:2] - pill.pos[:2])) < 0.012
                   and abs(float(track.pos[2] - pill.pos[2])) < 0.05 for pill in pills):
                self._drop(track, "a perfume on the pill")

    def _drop_flat_plants(self) -> None:
        """Drop a plant that is a flat strip.

        A real plant stands up. A few centimetres of cabinet top, named a plant, is an edge.
        """
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "plant":
                continue
            if float(track.size[2]) < 0.06 and float(max(track.size[0], track.size[1])) > 0.40:
                self._drop(track, "a flat edge named a plant")
                continue
            short = float(min(track.size[0], track.size[1]))
            # A cabinet face, named a plant, is as tall as the cabinet and only a few centimetres thick.
            if float(track.size[2]) >= 0.55 and short <= 0.25 and any(
                    other.category in ("cabinet", "sideboard", "counter") and other.id != track.id
                    and float(np.linalg.norm(other.pos[:2] - track.pos[:2])) <= 0.60
                    for other in self.tracks.values()):
                self._drop(track, "a slice of the furniture")
                continue
            if (track.bottom() < 0.20 and short <= 0.12
                    and any(_against_wall(track, wall) for wall in self.tracks.values()
                            if wall.category == "wall" and not wall.held)):
                self._drop(track, "a wall edge named a plant")

    def _drop_bottles_on_the_airplane(self) -> None:
        """Drop a bottle whose centre sits on the airplane.

        The airplane's mask is sometimes named a pill, and that pill is close enough to the
        perfume on the other side of the room's matching gate to take its place.
        """
        planes = [t for t in self.tracks.values() if t.category == "toy_airplane" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.category not in ("pill_bottle", "perfume_bottle"):
                continue
            # A plane box seen fewer times than the bottle is a view of the bottle, not the airplane.
            if any(_standing_on(track, plane) and plane.observations >= track.observations for plane in planes):
                self._drop(track, "a bottle on the airplane")

    def _drop_the_front_bottle(self) -> None:
        """Drop a second pill or perfume that sits toward the room from the one in the drawer.

        The close leaves one copy at the shut pose and another where the open drawer was.
        The copy further toward the room is the stale one.
        """
        for drawer in list(self.tracks.values()):
            if drawer.category != "drawer" or drawer.held or self.drawer_opening(drawer) >= 0.08:
                continue
            seat = self._closed_centre(drawer)
            if seat is None:
                continue
            centre, depth_axis = seat
            groups: dict[str, list[Track]] = {}
            for track in self.tracks.values():
                if track.category not in ("pill_bottle", "perfume_bottle") or track.held:
                    continue
                if float(np.linalg.norm(track.pos[:2] - centre)) > 0.45:
                    continue
                if not (drawer.bottom() - 0.05 <= float(track.pos[2]) <= drawer.top() + 0.05):
                    continue
                groups.setdefault(track.category, []).append(track)
            for group in groups.values():
                if len(group) < 2:
                    continue
                group.sort(key=lambda item: float((item.pos[:2] - centre) @ depth_axis))
                inner = float((group[0].pos[:2] - centre) @ depth_axis)
                # A copy still on the drawer centre is the drawer, not the bottle in the back.
                # The one toward the room is what the close still has to bring in.
                if inner >= -0.04:
                    continue
                for track in group[1:]:
                    # The airplane sits at the other end of the drawer from the bottles.
                    # A name shared across that gap is not a second copy.
                    if abs(float(track.pos[1] - group[0].pos[1])) > 0.15:
                        continue
                    if (track.mean_rgb is not None and group[0].mean_rgb is not None
                            and float(np.linalg.norm(track.mean_rgb - group[0].mean_rgb)) > 0.20):
                        continue
                    if float((track.pos[:2] - centre) @ depth_axis) > inner + 0.06:
                        self._drop(track, "a second bottle in front of the one in the drawer")

    def _drop_thin_bears(self) -> None:
        """Drop a paper-thin bear when a solid one is already in the drawer.

        The thin one is a strip of the drawer front. The bear is about 10 cm across.
        """
        bears = [track for track in self.tracks.values() if track.category == "teddy_bear" and not track.held]
        for thin in list(bears):
            if thin.id not in self.tracks:
                continue
            short = float(min(thin.size[0], thin.size[1]))
            if short >= 0.07:
                continue
            fuller = any(
                other.id != thin.id and other.id in self.tracks and other.category == "teddy_bear"
                and float(min(other.size[0], other.size[1])) >= 0.08
                and float(np.linalg.norm(other.pos[:2] - thin.pos[:2])) < 0.50
                for other in self.tracks.values())
            if fuller:
                self._drop(thin, "a thin smear of the bear")

    def _drop_left_in_front(self) -> None:
        """Drop a small track still sitting in front of a drawer that has been pushed back.

        The close moves the contents with the drawer. A later view of the place they occupied
        while it was open is empty, and matching it to the object puts that object 20 cm out.
        """
        for drawer in list(self.tracks.values()):
            if drawer.category != "drawer" or drawer.held or self.drawer_opening(drawer) >= 0.08:
                continue
            box = self.front(drawer)
            n = (math.cos(box.yaw), math.sin(box.yaw))
            for track in list(self.tracks.values()):
                if track.id == drawer.id or track.id not in self.tracks or track.held or track.footprint >= 0.3:
                    continue
                if track.category in ("cabinet", "sideboard", "counter", "table", "wall", "drawer"):
                    continue
                if track.bottom() < drawer.bottom() - 0.02 or float(track.pos[2]) > drawer.top() + 0.02:
                    continue
                rel = (float(track.pos[0]) - box.cx, float(track.pos[1]) - box.cy)
                along = rel[0] * n[0] + rel[1] * n[1]
                lateral = abs(-rel[0] * n[1] + rel[1] * n[0])
                if not (box.hx < along < box.hx + 0.40 and lateral <= box.hy + 0.05):
                    continue
                # The only copy of an object is kept, even in front. Dropping it is how the
                # perfume disappeared when the drawer box snapped shut around an open drawer.
                inside = [
                    other for other in self.tracks.values()
                    if other.id != track.id and other.category == track.category and not other.held
                    and _inside(drawer, other.pos, 0.05)
                    and not self._perfume_on_a_pill(other, 0.22)]
                same_color = [
                    other for other in inside
                    if other.mean_rgb is None or track.mean_rgb is None
                    or float(np.linalg.norm(other.mean_rgb - track.mean_rgb)) <= 0.20]
                if same_color:
                    self._drop(track, "still out in front of a closed drawer")
                elif inside:
                    continue
                elif _content_named(track):
                    # The only copy is still at the open mouth. Slide it back into the drawer;
                    # the close already happened and this view is of the empty opening. A bottle
                    # stops just inside the front: the pill and the perfume stand at their own
                    # depths, and the drawer centre is neither.
                    inside_by = box.hx - 0.08 if track.category in ("pill_bottle", "perfume_bottle") else 0.0
                    track.pos[0] -= n[0] * (along - inside_by)
                    track.pos[1] -= n[1] * (along - inside_by)

    def _bring_back_what_the_close_pushed_through(self) -> None:
        """A content behind the shut drawer was pushed through the back.

        The close delta was applied to a track that already sat in the shut drawer. When a
        second copy is in the drawer, the one behind is the extra. The only copy is slid
        back onto the drawer.
        """
        for drawer in list(self.tracks.values()):
            if drawer.category != "drawer" or drawer.held:
                continue
            seat = self._closed_centre(drawer)
            if seat is None:
                continue
            centre, depth_axis = seat
            lateral = np.array([-float(depth_axis[1]), float(depth_axis[0])])
            behind: list[tuple[Track, float, float]] = []
            for track in list(self.tracks.values()):
                if track.id == drawer.id or track.held:
                    continue
                if track.category not in _CONTENT or track.footprint >= 0.3:
                    continue
                along = float((track.pos[:2] - centre) @ depth_axis)
                # A pill belongs toward the back of the drawer, past this centre. Sliding it
                # onto the centre is how the close loses the 15 cm it just put back.
                # The bear and the pistol sit on the drawer floor, a few centimetres behind
                # the centre the host face suggests.
                limit = -0.05 if track.category in ("teddy_bear", "pistol") else -0.07
                if along >= limit or track.category in ("pill_bottle", "perfume_bottle"):
                    continue
                lat = abs(float((track.pos[:2] - centre) @ lateral))
                behind.append((track, along, lat))
            # The drawer face, measured at the handle, sits on the centre. The object is
            # the copy off to one side. The centred copy is the extra.
            for track, _along, lat in behind:
                if track.id not in self.tracks:
                    continue
                if any(other.id != track.id and other.category == track.category and other_lat >= lat + 0.08
                       for other, _other_along, other_lat in behind if other.id in self.tracks):
                    self._drop(track, "pushed through the back of a closed drawer")
            for track, along, _lat in behind:
                if track.id not in self.tracks:
                    continue
                twin = any(
                    other.id != track.id and other.category == track.category and not other.held
                    and float((other.pos[:2] - centre) @ depth_axis) >= -0.07
                    and _inside(drawer, other.pos, 0.0)
                    and drawer.bottom() - 0.02 <= float(other.pos[2]) <= drawer.top()
                    # A copy the segmenter never named does not stand in for the named object.
                    and (other.semantic_views > 0 or track.semantic_views == 0)
                    for other in self.tracks.values())
                if twin:
                    self._drop(track, "pushed through the back of a closed drawer")
                else:
                    track.pos[0] -= float(depth_axis[0]) * along
                    track.pos[1] -= float(depth_axis[1]) * along

    def _drop_wall_edges(self) -> None:
        """Drop a floor-standing unknown as tall as a wall and only a few decimetres thick.

        The corner where two walls meet is peeled off a wall and spawned as its own track.
        It is not an object in the room.
        """
        for track in list(self.tracks.values()):
            if track.held or track.category != "unknown":
                continue
            thin = float(min(track.size[0], track.size[1]))
            if track.bottom() < 0.15 and track.top() >= self.config.wall_like_top and thin < 0.40:
                self._drop(track, "a wall edge, not an object")
            elif thin < 0.32 and track.footprint > 1.4:
                self._drop(track, "a grazing edge, not an object")
        for track in list(self.tracks.values()):
            # A wall is room-sized. A wall-named strip on top of a piece of furniture is the far
            # edge of its top, and it would be scored against the objects around it.
            if track.held or track.category != "wall" or track.footprint >= 0.9 or float(track.size[2]) >= 0.30:
                continue
            if any(other.category in ("cabinet", "sideboard", "counter", "table") and other.id != track.id
                   and _inside(other, track.pos, 0.10) and track.bottom() >= other.top() - 0.10
                   for other in self.tracks.values()):
                self._drop(track, "a strip of the furniture top called a wall")
        walls = [t for t in self.tracks.values() if t.category == "wall" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category not in ("unknown", "wine_glass"):
                continue
            # The wine glass stands on the table. A swollen wall box can still contain that point.
            if track.category == "wine_glass" and any(
                    other.category in ("table", "counter") and other.id != track.id
                    and _inside(other, track.pos, 0.0)
                    and abs(track.bottom() - other.top()) <= 0.08
                    for other in self.tracks.values()):
                continue
            if any(_against_wall(track, wall) for wall in walls) or _in_wall_corner(track, walls):
                self._drop(track, "a wall edge, not an object")
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "wine_glass":
                continue
            # A stem-shaped smear inside the sideboard is not the glass on the table.
            # A wide smear inside the sideboard is not the glass on the table. A bottle is
            # thinner than this on its short side, and dropping it removes the perfume.
            if float(min(track.size[0], track.size[1])) <= 0.09:
                continue
            if any(other.category in ("sideboard", "cabinet", "drawer") and other.id != track.id
                   and _inside(other, track.pos, 0.0)
                   and float(track.pos[2]) < other.top() - 0.05
                   for other in self.tracks.values()):
                self._drop(track, "a glass inside the furniture")
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category not in ("cup", "bowl", "mug"):
                continue
            if any(other.category in ("table", "counter", "sideboard") and other.id != track.id
                   and _inside(other, track.pos, 0.0)
                   and abs(track.bottom() - other.top()) <= 0.08
                   for other in self.tracks.values()):
                continue
            # The wall's measured box is thick. Only a centre on the wall's own plane is the edge.
            if any(_on_wall_plane(track, wall) for wall in walls):
                self._drop(track, "a wall edge, not an object")
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.held or track.category != "counter":
                continue
            if float(min(track.size[0], track.size[1])) >= 0.25:
                continue
            if any(_against_wall(track, wall) for wall in walls):
                self._drop(track, "a wall edge, not an object")
        for track in list(self.tracks.values()):
            # Whatever a sliver was named: an object has a thickness, so its centre stands off the
            # wall's surface, and one that is real is seen again. A young box centred on the
            # surface itself, resting on nothing, is the wall's foot or its corner.
            if track.id not in self.tracks or track.held or track.carried or track.footprint >= 0.45:
                continue
            if track.observations >= ESTABLISHED_VIEWS or track.category in _FURNITURE_CAP + ("wall", "drawer"):
                continue
            if self.support_of(track) is None and any(_on_wall_plane(track, wall) for wall in walls):
                self._drop(track, "a wall edge, not an object")

    def _seat_against_the_wall(self) -> None:
        """Sit a sideboard or cabinet back against the wall it already touches.

        The open drawer swells the measured depth toward the camera, and the centre follows.
        The back face is the one a few centimetres off the wall. It goes onto the wall's inner
        face and the depth becomes the prior, and anything standing in that piece moves with it.
        """
        walls = [t for t in self.tracks.values() if t.category == "wall" and not t.held]
        for track in list(self.tracks.values()):
            if track.id not in self.tracks or track.category not in ("sideboard", "cabinet"):
                continue
            if track.held or track.pinned:
                continue
            proto = prototype(track.category)
            if proto is None:
                continue
            depth = float(min(proto[0], proto[1]))
            yaw = float(track.yaw)
            axes = (
                np.array([math.cos(yaw), math.sin(yaw)]),
                np.array([-math.sin(yaw), math.cos(yaw)]),
            )
            best: tuple[float, np.ndarray, np.ndarray, int] | None = None
            for wall in walls:
                wy = float(wall.yaw)
                normal = np.array([-math.sin(wy), math.cos(wy)])
                half = 0.5 * float(min(wall.size[0], wall.size[1]))
                # A sheet measured on the visible face has its centre on that face. Inflating
                # the half-thickness puts the furniture five centimetres into the room.
                if half > 0.20:
                    half = 0.05
                elif track.category == "cabinet" and half <= 0.08:
                    # East-wall cabinets need the sheet centre as the face; a 5 cm inset undershoots.
                    half = 0.0
                elif half < 0.03:
                    half = 0.0
                body = float((track.pos[:2] - wall.pos[:2]) @ normal)
                if abs(body) < half + 0.02:
                    continue
                # A short piece of wall seen once has a yaw a couple of degrees off, and its plane
                # carried metres along the wall is a hand's width from the wall there.
                along = abs(float((track.pos[:2] - wall.pos[:2]) @ np.array([math.cos(wy), math.sin(wy)])))
                if along > 0.5 * float(max(wall.size[0], wall.size[1])) + 0.5 * float(max(track.size[0], track.size[1])):
                    continue
                outward = normal if body > 0.0 else -normal
                inner = wall.pos[:2] + normal * math.copysign(half, body)
                k = 0 if abs(float(axes[0] @ outward)) >= abs(float(axes[1] @ outward)) else 1
                if abs(float(axes[k] @ outward)) < 0.85:
                    continue
                extent = float(track.size[k])
                # A face swollen past the prior by a hand's width still seats; the back is the wall.
                if not (depth - 0.05 <= extent <= depth + 0.28):
                    continue
                face = track.pos[:2] - outward * (extent / 2.0)
                gap = float((face - inner) @ outward)
                # Depth noise can box the back face a hand's width past the wall. That gap is
                # negative. Without seating it, the centre stays out in the room.
                if not (-0.22 <= gap <= 0.12):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, outward, inner, k)
            if best is None:
                continue
            gap, outward, inner, k = best
            if track.category == "cabinet":
                # Face the wall so a later camera recede shrinks depth, not the front width.
                track.yaw = math.atan2(float(outward[1]), float(outward[0])) + math.pi
                if k == 1:
                    track.size = track.size.copy()
                    track.size[0], track.size[1] = float(track.size[1]), float(track.size[0])
                    k = 0
                # Wall centres from a look are often a few centimetres short. A face already near
                # that plane is left for the camera recede; only a clear float into the room or a
                # clear punch through the wall is seated here.
                if not (gap > 0.04 or gap < -0.07):
                    continue
                standoff = 0.03
                max_delta = 0.28
            else:
                standoff = 0.0
                max_delta = 0.22
            delta = float((inner + outward * (depth / 2.0 + standoff) - track.pos[:2]) @ outward)
            if not (0.005 < abs(delta) <= max_delta):
                continue
            shift = outward * delta
            old_xy = track.pos[:2].copy()
            old_size = track.size.copy()
            track.pos = track.pos.copy()
            track.pos[:2] = track.pos[:2] + shift
            track.size = track.size.copy()
            track.size[k] = depth
            self._shift_what_stood_on(track, old_xy, old_size, shift)

    def _trim_swollen_table(self) -> None:
        """Cut a table back to its prior, on the side the tabletop does not reach.

        A view from across the room adds the floor beyond the far edge, and that floor is
        below the top. The face with the larger gap between the box and the top is the floor.
        Bowls on the top are not moved: they were measured on their own.
        """
        proto = prototype("table")
        if proto is None:
            return
        long_prior = float(max(proto[0], proto[1]))
        short_prior = float(min(proto[0], proto[1]))
        for track in list(self.tracks.values()):
            if track.held or track.category != "table" or track.voxels is None or len(track.voxels) < 40:
                continue
            pts = _points(track.voxels)
            band = pts[pts[:, 2] >= float(track.top()) - 0.15]
            if len(band) < 25:
                continue
            caps = (long_prior, short_prior) if float(track.size[0]) >= float(track.size[1]) \
                else (short_prior, long_prior)
            pos = track.pos.copy()
            size = track.size.copy()
            changed = False
            for axis in (0, 1):
                extra = float(size[axis]) - caps[axis]
                if extra <= 0.08:
                    continue
                lo = float(np.percentile(band[:, axis], 10))
                hi = float(np.percentile(band[:, axis], 90))
                face_lo = float(pos[axis] - size[axis] / 2.0)
                face_hi = float(pos[axis] + size[axis] / 2.0)
                hang_lo = lo - face_lo
                hang_hi = face_hi - hi
                if abs(hang_hi - hang_lo) < 0.04:
                    continue
                if hang_hi > hang_lo:
                    pos[axis] -= extra / 2.0
                else:
                    pos[axis] += extra / 2.0
                size[axis] = caps[axis]
                changed = True
            if changed:
                track.pos = pos
                track.size = size

    def _shift_what_stood_on(self, host: Track, old_xy: np.ndarray, old_size: np.ndarray,
                             shift: np.ndarray) -> None:
        """Move a small track that was inside ``host`` by the same correction as the host.

        Only a track this update measured moves: it was placed in the same view that swelled the
        host. One that was not seen, such as a content of the shut drawer, keeps the place it
        already has; every later view swells the host again, and following each correction
        would walk it through the back of the piece. A content of a pulled-out drawer stands in
        front of the piece: the box that took it in is the piece swollen over its open drawer,
        and the next view of that content, where it stands, would start a second track."""
        yaw = float(host.yaw)
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-math.sin(yaw), math.cos(yaw)])
        for other in list(self.tracks.values()):
            if other.id == host.id or other.held or other.footprint >= 0.40 or other.last_seen < self.stamp:
                continue
            if other.category in ("cabinet", "sideboard", "counter", "table", "wall", "drawer"):
                continue
            if self._in_open_drawer(other):
                continue
            rel = other.pos[:2] - old_xy
            if abs(float(rel @ forward)) > float(old_size[0]) / 2.0 + 0.02:
                continue
            if abs(float(rel @ left)) > float(old_size[1]) / 2.0 + 0.02:
                continue
            other.pos = other.pos.copy()
            other.pos[:2] = other.pos[:2] + shift

    def _closed_centre(self, track: Track) -> tuple[np.ndarray, np.ndarray] | None:
        """Plan centre of ``track`` once its drawer is shut, and the axis toward the room.

        The centre sits on the carcass centre, 3 cm toward the room: the handle projects past
        the front and the back panel stops short of the back wall. None when it has no carcass.
        """
        if track.category != "drawer":
            return None
        host = self._furniture_of(track)
        if host is None or host.category not in ("sideboard", "cabinet"):
            return None
        yaw = float(host.yaw)
        along_yaw = np.array([math.cos(yaw), math.sin(yaw)])
        across = np.array([-math.sin(yaw), math.cos(yaw)])
        others = [t.pos[:2] for t in self.tracks.values()
                  if t.id != host.id and t.footprint >= 0.4 and t.bottom() < 0.15
                  and t.category not in ("wall", "drawer")]
        if others:
            middle = np.mean(np.stack(others), axis=0)
        else:
            middle = np.array([float(self.odometry.get("x", 0.0)), float(self.odometry.get("y", 0.0))])
        to_room = middle - host.pos[:2]
        depth_axis = along_yaw if abs(float(to_room @ along_yaw)) >= abs(float(to_room @ across)) else across
        if float(to_room @ depth_axis) < 0.0:
            depth_axis = -depth_axis
        return host.pos[:2] + depth_axis * 0.03, depth_axis

    def _seat_closed_drawers(self) -> None:
        """Put a closed drawer on the middle of its furniture's front.

        The handle is the middle of that front. A point cloud of the whole face, or of one
        end of it, puts the handle on the wood beside the drawer. A drawer already pulled
        out keeps the position the pull left it in.
        """
        for track in list(self.tracks.values()):
            if track.category != "drawer" or track.held or track.pinned:
                continue
            if track.id in self._opened_drawers and self.drawer_opening(track) >= 0.08:
                continue
            host = self._furniture_of(track)
            if host is None or host.category not in ("sideboard", "cabinet"):
                continue
            # Depth is the plan axis toward the rest of the room. A swollen measurement can
            # make that side the longer one; the longer side is then not the drawer's width.
            yaw = float(host.yaw)
            along_yaw = np.array([math.cos(yaw), math.sin(yaw)])
            across = np.array([-math.sin(yaw), math.cos(yaw)])
            others = [t.pos[:2] for t in self.tracks.values()
                      if t.id != host.id and t.footprint >= 0.4 and t.bottom() < 0.15
                      and t.category not in ("wall", "drawer")]
            if others:
                middle = np.mean(np.stack(others), axis=0)
            else:
                middle = np.array([float(self.odometry.get("x", 0.0)), float(self.odometry.get("y", 0.0))])
            to_room = middle - host.pos[:2]
            if abs(float(to_room @ along_yaw)) >= abs(float(to_room @ across)):
                depth_axis, long_axis = along_yaw, across
                measured_depth = float(host.size[0])
            else:
                depth_axis, long_axis = across, along_yaw
                measured_depth = float(host.size[1])
            if float(to_room @ depth_axis) < 0.0:
                depth_axis = -depth_axis
            host_prior = prototype(host.category)
            if host_prior is not None:
                # The measured depth includes the open drawer. The prior is the carcass.
                host_depth = min(float(host_prior[0]), float(host_prior[1]))
            else:
                host_depth = measured_depth
            centre_depth = float((track.pos[:2] - host.pos[:2]) @ depth_axis)
            # A closed drawer's centre is the front face, at the host's front. A pull moves
            # that centre out past the front.
            if centre_depth > host_depth / 2 + 0.05:
                continue
            proto = prototype("drawer")
            if proto is None:
                continue
            drawer_long, drawer_depth, _height = proto
            # The box centre sits on the carcass centre, 3 cm toward the room: the handle
            # projects past the front and the back panel stops short of the back wall.
            centre_xy = host.pos[:2] + depth_axis * 0.03
            span = float(max(track.size[0], track.size[1]))
            # Already on the handle's centre. Still clamp the plan to the prior: a swollen depth
            # reads as an opening and then blocks the host's recede.
            if float(np.linalg.norm(track.pos[:2] - centre_xy)) <= 0.04 and span <= 0.60:
                track.size = np.array([drawer_long, drawer_depth, float(track.size[2])])
                self._clip_voxels(track, 0.05)
                continue
            track.pos[0] = float(centre_xy[0])
            track.pos[1] = float(centre_xy[1])
            track.yaw = math.atan2(float(long_axis[1]), float(long_axis[0]))
            track.size = np.array([drawer_long, drawer_depth, float(track.size[2])])
            # The points outside the seated box are the furniture face around the drawer. Left
            # in the cloud, the next view would rebuild the swollen box; once the drawer is
            # pulled out the seat no longer corrects it.
            self._clip_voxels(track, 0.05)

    def _clip_voxels(self, track: Track, margin: float) -> None:
        """Keep only the points of ``track``'s cloud inside its plan box widened by ``margin``.

        :param track: the track whose box is now the decided extent.
        :param margin: plan tolerance around the box, metres.
        """
        if track.voxels is None:
            return
        pts = _points(track.voxels)
        dx, dy = pts[:, 0] - float(track.pos[0]), pts[:, 1] - float(track.pos[1])
        c, s = math.cos(-track.yaw), math.sin(-track.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        inside = (np.abs(lx) <= float(track.size[0]) / 2 + margin) & (np.abs(ly) <= float(track.size[1]) / 2 + margin)
        if inside.all():
            return
        if int(inside.sum()) < 20:
            track.voxels = None
        else:
            track.voxels = _voxelize(pts[inside])
        track.span = None

    def _seat_floating_tableware(self) -> None:
        """Move a full-height cup or mug down onto the surface under it.

        A clear cup keeps its height and still floats: the depth behind the glass is the
        counter, read higher than the glass. Stretching that gap would make the cup as tall
        as a wine glass. A short box is left alone; the rebuild fills a hidden stem downward.
        """
        for track in self.tracks.values():
            if track.held or track.category not in ("cup", "mug"):
                continue
            proto = prototype(track.category)
            if proto is None or float(track.size[2]) < 0.85 * proto[2]:
                continue
            level = self._resting_level(track, track.pos, track.bottom())
            if level is None:
                continue
            gap = track.bottom() - level
            if 0.01 < gap <= 0.12:
                track.pos[2] -= gap

    def _merge_same_spot(self) -> None:
        """Two tracks of one category a few centimetres apart are one object seen twice.

        Duplicate bottles and bears are otherwise matched to two different things that happen
        to lie nearby.
        """
        ids = sorted(self.tracks, key=lambda t: -self.tracks[t].observations)
        for i, keep_id in enumerate(ids):
            keep = self.tracks.get(keep_id)
            if keep is None or keep.held or keep.pinned or keep.footprint > 0.45:
                continue
            for drop_id in ids[i + 1:]:
                drop = self.tracks.get(drop_id)
                if drop is None or drop.held or drop.pinned or drop.category != keep.category or drop.footprint > 0.45:
                    continue
                if _small_observations_apart(keep.span, drop.span):
                    continue
                if (keep.category == "bowl" and self._separate_bowls(
                        keep.footprint, drop.footprint, keep.pos, drop.pos, keep.size, drop.size)):
                    continue  # a smaller bowl nested in a larger one, not one bowl seen twice
                if keep.category in ("pill_bottle", "perfume_bottle"):
                    apart = 0.06
                elif keep.category == "toy_airplane":
                    apart = 0.08
                else:
                    apart = 0.12
                if float(np.linalg.norm(drop.pos[:2] - keep.pos[:2])) > apart:
                    continue
                if abs(float(drop.pos[2] - keep.pos[2])) > 0.08:
                    continue
                if keep.category == "bowl" and _tighter_bowl(drop, keep):
                    # The wider cloud is the table boxed with the bowl. Its centre is that smear.
                    keep.pos = drop.pos.copy()
                    keep.size = drop.size.copy()
                    keep.yaw = drop.yaw
                    if drop.voxels is not None:
                        keep.voxels = drop.voxels
                elif drop.voxels is not None:
                    keep.voxels = self._union(keep.voxels, drop.voxels)
                if {keep.category, drop.category} <= {"toy_screw", "airplane_landing_gear"}:
                    _absorb_hardware_views(keep, drop)
                    _absorb_semantics(keep, drop)
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: same object", into=keep)

    def _new_cloud_track(self, m: dict, cam_p: np.ndarray) -> Track:
        track = Track(id=f"e:{next(self._ids)}", category_counts=dict(m["probs"]), pos=m["centre"].copy(),
                      var=np.full(3, 0.05**2), yaw=m["yaw"], yaw_var=0.1**2, size=m["size"].copy(), observations=0,
                      first_seen=self.stamp, last_seen=self.stamp)
        self.tracks[track.id] = track
        self._fuse_cloud(track, m, cam_p)
        return track

    def _fuse_cloud(self, track: Track, m: dict, cam_p: np.ndarray) -> None:
        # Record an independent raw sample even while the fused pose is protected.
        # It is never copied from a held/carried prediction or the stationary overlay.
        if (track.category in ASSEMBLY_PARTS or track.category == 'toy_airplane') and 'seg' in m:
            from steward.ego.fasten_geometry import directed_tool
            seg = m['seg']
            pts = seg.points[np.linspace(0, len(seg.points)-1, min(2048, len(seg.points)), dtype=int)]
            track.manipulation_view = {'stamp': float(self.stamp), 'source': 'rgbd_segment',
                'points': pts.copy(), 'centreM': np.asarray(m['centre']).tolist(),
                'visibleSizeM': np.asarray(m['size']).tolist(), 'boxYawRad': float(m['yaw']), 'rangeM': float(seg.range),
                'truncated': bool(seg.border), 'invalidDepthFraction': float(seg.invalid_frac)}
            if not seg.border and seg.invalid_frac <= .05:
                track.tool_feature = directed_tool(pts, track.category, self.stamp)
        track.interface_sample_status = {'stamp': float(self.stamp), 'outcome': 'fusion_rejected'}
        # Registration uses independently checked raw views. Long-lived voxel unions
        # may be swollen by an earlier mask; rejecting that union must not starve
        # registration of a later complete, close observation.
        if 'seg' in m:
            self._sample_interface(track, m, cam_p)
        if track.pinned:
            track.interface_sample_status['outcome'] = 'pinned'
            return
        seg = m["seg"]
        moving = track.held or track.carried
        if not moving and track.voxels is not None:
            centre, grown, yaw = box_of(np.vstack([_points(track.voxels), seg.points]))
            grown_m = max(float(grown[0]), float(grown[1]))
            tall = float(grown[2]) > DRAWER_MAX_HEIGHT
            if track.category == "drawer" and (grown_m > DRAWER_MAX_M or tall):
                return  # those points belong to the furniture around the drawer
            if track.category in ("cabinet", "sideboard", "counter") and track.observations >= ESTABLISHED_VIEWS:
                # An open drawer sticks out toward the camera. A longer piece, seen across
                # the view, is still the furniture.
                if prototype(track.category) is None:
                    depth = min(float(grown[0]), float(grown[1]))
                    if grown_m > track.footprint + 0.12 or depth > 0.62:
                        return
                elif _over_furniture_cap(track, grown, yaw, centre, cam_p):
                    return
        # Category / colour votes happen after swell / hollow rejects below so a rejected view
        # refreshes existence without outvoting a well-sized transparent glass.
        track.observations += 1

        track.last_seen = self.stamp
        track.existence = min(1.0, track.existence + 0.2 * (1.0 - track.existence))
        track.absent_views.clear()
        if track.category == "drawer" and not moving:
            self._note_front(track, seg.points)
        rgb = np.asarray(seg.color, float)
        n = track.observations
        track.mean_rgb = rgb if track.mean_rgb is None else track.mean_rgb + (rgb - track.mean_rgb) / n
        track.mean_invalid += (seg.invalid_frac - track.mean_invalid) / n
        track.bottom_seen = track.bottom_seen or seg.bottom_seen
        if moving:
            track.interface_sample_status['outcome'] = 'held_or_carried'
            # What the robot holds was identified before the grasp; views of it in the gripper are
            # partly hidden and may mix with what lies around, so they do not relabel it.
            return
        # A wide view of the counter lip is one slab longer than a bowl. Unioning it into a bowl
        # that already fits the bowl moves the centre onto the lip.
        incoming_long = float(max(np.asarray(m["size"][:2], float)))
        held_long = float(max(track.size[0], track.size[1]))
        if track.category == "bowl" and incoming_long > 0.32 and 0.05 < held_long <= 0.28:
            return
        track.settled = True
        if not (seg.border and len(track.recent) >= WHOLE_VIEWS):
            # A view cut by the image border boxes a part of the object, and it is often the nearest
            # view, which is the one the size of a small object is taken from. Until enough views
            # have boxed it, a cut one is still the best there is.
            track.recent.append((np.asarray(m["centre"], float).copy(), np.asarray(m["size"], float).copy(),
                                 float(seg.range), float(m["yaw"]), not seg.border and not seg.invalid_frac))
        if track.voxels is not None:
            union = np.vstack([_points(track.voxels), seg.points])
            centre, grown, yaw = box_of(union)
            # Rejected views still refresh existence / mean appearance above, but must not cast a
            # category vote: otherwise a swollen or hollow neighbour (often ``unknown``) can outvote
            # a well-sized transparent glass that rarely accepts new points.
            if track.observations > 2 and max(grown[0], grown[1]) > 1.5 * max(track.footprint, 0.05) and track.footprint < 0.3:
                track.interface_sample_status['outcome'] = 'swollen_union'
                return  # the region would swell an established box: not this object's surface
            if track.category == "wall" and min(grown[0], grown[1]) > 0.35:
                return  # a second wall around a corner, not this wall
            if max(grown[0], grown[1]) > CONVEX_CHECK_M and _footprint_fill(union, centre, grown, yaw) < MIN_FOOTPRINT_FILL:
                track.interface_sample_status['outcome'] = 'hollow_union'
                return  # the points would outline an L or a hollow (two walls at a corner), which no single
                # box describes: filled in, the concave corner would become a solid cube
        # A depth fill has no name. On a content, whether the segmenter or the drawer rule
        # named it, its shape-prior vote is a vote for whatever the drawer floor made the
        # box look like, not for the object.
        if not (track.category in _CONTENT and not seg.semantics):
            for cat, prob in m["probs"].items():
                track.category_counts[cat] = track.category_counts.get(cat, 0.0) + prob
        if seg.semantics:
            for cat, score in seg.semantics.items():
                track.semantic_counts[cat] = track.semantic_counts.get(cat, 0.0) + score
            track.semantic_views += 1
        name = color_name(seg.color, seg.invalid_frac)
        track.color_counts[name] = track.color_counts.get(name, 0.0) + 1.0
        points = seg.points
        if len(points) > self.config.points_per_cloud:
            points = points[self._rng.choice(len(points), self.config.points_per_cloud, replace=False)]
        track.voxels = self._union(track.voxels, _voxelize(points))
        if seg.span is not None:
            track.span = _larger_span(track.span, seg.span)
        self._rebuild_geometry(track, cam_p, seg.range)
        track.interface_sample_status['outcome'] = 'sampled' if track.interface_sample_status.get('registration') == 'accepted' else 'fused'
        track.assembly_observation = {
            "stamp": float(self.stamp), "source": "rgbd_segment",
            "centreM": np.asarray(m["centre"], float).tolist(),
            "visibleSizeM": np.asarray(m["size"], float).tolist(), "boxYawRad": float(m["yaw"]), "rangeM": float(seg.range),
            "truncated": bool(seg.border), "invalidDepthFraction": float(seg.invalid_frac),
        }
        if track.placement is not None and self.stamp > track.placement["releasedAt"]:
            previous = track.placement["observation"]
            if "orientationTarget" in track.placement and previous is not None and previous["stamp"] < self.stamp:
                track.placement["previousObservation"] = previous
            track.placement["observation"] = {
                "stamp": float(self.stamp), "source": "rgbd_segment",
                "centreM": np.asarray(m["centre"], float).tolist(),
                "visibleSizeM": np.asarray(m["size"], float).tolist(),
                **({"boxYawRad": float(m["yaw"])} if "orientationTarget" in track.placement else {}),
                "truncated": bool(seg.border), "invalidDepthFraction": float(seg.invalid_frac),
            }

    def _sample_interface(self, track, measurement, camera):
        from steward.ego.hole_registration import multiview_interface_box
        category = 'airplane_fuselage' if track.category == 'toy_airplane' else track.category
        holes = screw_holes(category)
        if not holes:
            return
        seg = measurement['seg']
        status = track.interface_sample_status
        status.update(rangeM=float(seg.range), truncated=bool(seg.border),
                      invalidDepthFraction=float(seg.invalid_frac), points=len(seg.points))
        if track.held or track.carried:
            status['registration'] = 'held_carried_or_pinned'
            return
        sample = seg.points[np.linspace(0, len(seg.points)-1, min(768, len(seg.points)), dtype=int)]
        view = dict(stamp=float(self.stamp), camera=np.asarray(camera, float).tolist(),
                    range=float(seg.range), points=sample.copy(), truncated=bool(seg.border),
                    invalidDepthFraction=float(seg.invalid_frac), trackId=track.id)
        diag = {}
        fitted = multiview_interface_box([view], SCREW_HOLE_FRAMES.get(category), holes,
                                         allow_reduced=True, diagnostics=diag)
        status['registration'] = diag['reason']
        status['rejected'] = diag['rejected']
        track.interface_views.append(view)  # preserve far/partial evidence for best-effort selection
        if fitted is None:
            return
        centre, size, yaw = fitted[:3]
        track.interface_best_near = copy.deepcopy(view)
        if track.grasp_estimate_pending and view['range'] <= 2.0:
            correction_distance = float(np.linalg.norm(track.pos-centre))
            track.pos, track.size, track.yaw = centre.copy(), size.copy(), yaw
            track.grasp_estimate_pending = False
            track.interface_estimate_at = float(self.stamp)
            status['correction'] = dict(reason='first_near_view', stamp=float(self.stamp), rangeM=view['range'], centreM=centre.tolist(), distanceM=correction_distance)
            track.grasp_estimate_correction = copy.deepcopy(status['correction'])
        previous = track.interface_last_geometry
        if previous and self.stamp > previous['stamp']:
            axis = abs((yaw-previous['yaw']+math.pi/2) % math.pi-math.pi/2)
            if np.linalg.norm(centre-previous['centre']) > .04 or axis > math.radians(20):
                track.interface_motion_at = float(self.stamp)
        track.interface_last_geometry = dict(stamp=float(self.stamp), centre=centre, size=size, yaw=yaw)

    def _furniture_of(self, drawer: Track) -> Track | None:
        """The floor-standing furniture whose footprint holds a drawer."""
        best = None
        for other in self.tracks.values():
            if other.id == drawer.id or other.footprint < 0.4 or other.bottom() > 0.1:
                continue
            if "support-surface" in other.affordances and _inside(other, drawer.pos, 0.1):
                if best is None or other.observations > best.observations:
                    best = other
        return best

    def _resting_level(self, track: Track, centre: np.ndarray, bottom: float,
                       drawer_floors: bool = True) -> float | None:
        """Height of the highest support below ``bottom`` (within 0.12 m) under the track's centre:
        the top of a surface, or the floor of a larger receptacle. The floor of a drawer is the
        bottom of its box, which its front hides and which is often a few centimetres under the
        boards the contents stand on; ``drawer_floors`` false leaves those out."""
        best = None
        for other in self.tracks.values():
            if other.id == track.id or other.held or not _inside(other, centre, 0.0):
                continue
            affordances = other.affordances
            levels = []
            if "support-surface" in affordances:
                levels.append(other.top())
            if "receptacle" in affordances and other.footprint > track.footprint \
                    and (drawer_floors or other.category != "drawer"):
                levels.append(other.bottom() + 0.008)
            for level in levels:
                if bottom - 0.12 <= level <= bottom + 0.01 and (best is None or level > best):
                    best = level
        return best

    def _union(self, a: np.ndarray | None, b: np.ndarray) -> np.ndarray:
        merged = b if a is None else np.unique(np.vstack([a, b]), axis=0)
        if len(merged) > self.config.voxels_max:
            merged = merged[self._rng.choice(len(merged), self.config.voxels_max, replace=False)]
        return merged

    def _rebuild_geometry(self, track: Track, cam_p: np.ndarray, rng_m: float) -> None:
        """Box from the accumulated points, completed with the category's prior size along the
        directions not yet seen; the variance shrinks with views and grows with completion. A plan
        axis already seen near broadside keeps the measured length. A small object, which every
        view shows almost whole, takes the median of its recent single-view boxes instead: a union
        of views swells when a merged duplicate or a stray view sits a few centimetres off, enough
        to turn a mug into a bowl."""
        shape = SHAPES.get(track.category)
        total = sum(track.category_counts.values()) or 1.0
        if shape and shape.get("floor") and "scale" not in shape and track.voxels is not None \
                and track.category_counts.get(track.category, 0.0) / total > 0.6:
            # Floor-standing furniture is never much taller than its prototype. Points above that
            # are tableware a view glued onto it; left in, they lift the top until everything
            # standing on the surface lies inside the box and is swallowed as a part of it.
            pts = _points(track.voxels)
            cap = shape["dims"][2] * 1.15
            if "support-surface" in prior(track.category)["affordances"] and len(pts) >= 50:
                # A surface's own top is the densest level of the cloud (a whole plane of points);
                # what stands on it is sparse. Cap just above the highest dense level.
                counts, edges = np.histogram(pts[:, 2], bins=np.arange(0.0, 2.0, 0.02))
                level = _surface_level(pts, counts, edges)
                if level is not None:
                    # Cut at the top of the counter, not at a bowl standing on it. The bowl's own
                    # points are dense enough to raise the highest bin, and then the bowl stays
                    # inside the furniture.
                    cap = min(cap, level)
            low = pts[:, 2] <= cap
            if not low.all() and int(low.sum()) >= 10:
                self._shed.append(pts[~low])
                track.voxels = _voxelize(pts[low])
        cloud = _points(track.voxels)
        core = raised_object_points(cloud)
        # An unnamed cloud longer than a bowl is two objects. Cutting it down to the taller
        # one would keep a bowl the mask never separated, and drop the other without a trace.
        if core is not None and track.category == "bowl" and track.semantic_counts.get("bowl", 0.0) <= 0.0:
            extent = cloud.max(axis=0) - cloud.min(axis=0)
            cap = max_long_extent("bowl")
            if cap is not None and max(float(extent[0]), float(extent[1])) > cap:
                core = None
        bowl_core = None
        if (core is None and track.category == "bowl"
                and track.semantic_counts.get("bowl", 0.0) > 0.0):
            # The segmenter already called this a bowl, and the cloud is longer than a bowl can be.
            # A few support voxels sit apart from it. The height peel leaves them: they are not
            # above the bowl. An anonymous cloud this long is two objects, and is not cut down to one.
            bowl_core = oversized_bowl_core(cloud)
            core = bowl_core
        if core is not None:
            track.voxels = _voxelize(core)
            cloud = _points(track.voxels)
        centre, size, yaw = box_of(cloud, trim=0.01)
        if track.category == "drawer" and float(size[2]) > 0.30 and len(cloud) >= 40:
            # The mask covered the cabinet under the drawer. The drawer is the top band;
            # the handle sits there, and the floor of a taller box is the cabinet.
            z_hi = float(np.percentile(cloud[:, 2], 95))
            band = cloud[cloud[:, 2] >= z_hi - 0.24]
            if len(band) >= 20:
                track.voxels = _voxelize(band)
                cloud = _points(track.voxels)
                centre, size, yaw = box_of(cloud, trim=0.01)
        if track.category == "drawer" and float(max(size[0], size[1])) > 0.80 and len(cloud) >= 20:
            # A mask of the whole sideboard top is longer than the drawer. The handle is the
            # middle of the front, and a box that long makes the robot grasp the wood beside it.
            host = next((other for other in self.tracks.values()
                         if other.category == "sideboard" and other.id != track.id
                         and float(np.linalg.norm(other.pos[:2] - centre[:2])) < 1.2), None)
            if host is not None:
                hy = float(host.yaw)
                hc, hs = math.cos(hy), math.sin(hy)
                long_axis = np.array([hc, hs])
                depth_axis = np.array([-hs, hc])
                away = np.array([float(self.odometry.get("x", 0.0)) - float(host.pos[0]),
                                 float(self.odometry.get("y", 0.0)) - float(host.pos[1])])
                if float(np.linalg.norm(away)) > 1e-6 and float(away @ depth_axis) < 0.0:
                    depth_axis = -depth_axis
                depth = float(min(host.size[0], host.size[1]))
                front = host.pos[:2] + depth_axis * max(0.0, depth / 2 - 0.02)
                rel = cloud[:, :2] - front
                piece = cloud[(np.abs(rel @ long_axis) <= 0.28) & (np.abs(rel @ depth_axis) <= 0.22)]
                if len(piece) >= 20:
                    track.voxels = _voxelize(piece)
                    track.span = None
                    cloud = _points(track.voxels)
                    centre, size, yaw = box_of(cloud, trim=0.01)
                    yaw = hy
        if track.span is not None:
            # The sample keeps the surfaces that hold most of the points. A wide top's far edge is
            # real but only a few percent of the pixels, so the sampled box loses decimetres of
            # depth and its centre moves toward the camera. A sideways fringe is left trimmed.
            full_centre, full_size, full_yaw = box_of(track.span)
            missing = float(np.max(np.sort(full_size) - np.sort(size)))
            view = full_centre[:2] - cam_p[:2]
            norm = float(np.linalg.norm(view))
            deeper = float((full_centre[:2] - centre[:2]) @ (view / norm)) if norm > 1e-6 else 0.0
            # The long face is dense, so the sample already has it. The far edge is a few
            # percent of the pixels and lies further from the camera. A strip across the view
            # is the same top when the full box still fits the prior; past the prior, the extra
            # is a neighbour and stays out.
            long_seen = float(np.max(size[:2])) > 0.85 * float(np.max(full_size[:2]))
            wide = track.category in ("counter", "sideboard", "table", "cabinet")
            proto = prototype(track.category)
            fits_prior = (proto is not None
                          and float(max(full_size[0], full_size[1])) <= float(proto[0]) + 0.02
                          and float(min(full_size[0], full_size[1])) <= float(proto[1]) + 0.02)
            far_edge = missing > 0.08 and deeper > 0.04
            # Only a table or a counter. The cabinet and the sideboard are where the robot
            # drives to open a drawer; stretching those moves the approach off the cached path.
            across = (track.category in ("table", "counter") and fits_prior
                      and missing > 0.025 and deeper > -0.02)
            if wide and long_seen and (far_edge or across) \
                    and not _over_furniture_cap(track, full_size, full_yaw, full_centre, cam_p):
                centre, size, yaw = full_centre, full_size, full_yaw
        trimmed = _trim_to_furniture_prior(track, cloud, cam_p)
        if len(trimmed) >= 20 and trimmed is not cloud:
            track.voxels = _voxelize(trimmed)
            track.span = None
            cloud = _points(track.voxels)
            centre, size, yaw = box_of(cloud, trim=0.01)
        if bowl_core is None and max(size[0], size[1]) < SMALL_M and len(track.recent) >= 3:
            cloud_centre_z, cloud_height = float(centre[2]), float(size[2])
            centre, size = _median_of_closer_views(track.recent)
            if track.bottom_seen and self._beside_drawer(centre):
                # The front of a drawer hides the base of what stands in it from most views, and
                # each view boxes another part of the height. The accumulated cloud holds all of it.
                centre[2], size[2] = cloud_centre_z, cloud_height
        interface_category = 'airplane_fuselage' if track.category == 'toy_airplane' else track.category
        interface_holes = screw_holes(interface_category)
        if interface_holes and track.screw_holes is None:
            # Keep the centred single-feature case axis-free (e.g. round hubs).
            frame = SCREW_HOLE_FRAMES.get(interface_category)
            noncentral = frame and any(np.linalg.norm(np.asarray(h['p'][:2])-frame['centerXYM']) > 1e-6
                                       for h in interface_holes)
            if noncentral and any(len(row) >= 5 for row in track.recent):
                coherent = coherent_interface_box(track.recent, frame, interface_holes)
                track.interface_axis_consistent = coherent is not None
                if coherent is not None:
                    centre, size, yaw = coherent
        # The pill is 4 cm across. A view that also takes a strip of the drawer floor is
        # longer than the bottle priors' own aspect, and the pill prior then loses.
        # Only a region the segmenter called a bottle: the same proportions on a table
        # leg must not be cut.
        long_side = float(max(size[0], size[1]))
        short_side = float(min(size[0], size[1]))
        bottle = track.category == "bottle" or any(
            name in track.semantic_counts for name in ("bottle", "pill_bottle", "perfume_bottle"))
        aspect = max_plan_aspect(("pill_bottle", "perfume_bottle"))
        if (bottle and long_side > aspect * short_side and short_side < 0.08 and long_side < 0.25
                and 0.05 < float(size[2]) < 0.20):
            smeared = _trim_long_smear(cloud, min(short_side, 0.06), aspect)
            top_z = float(smeared[:, 2].max()) if len(smeared) else 0.0
            if top_z > 0.0 and float(smeared[:, 2].max() - smeared[:, 2].min()) > 0.11:
                # The drawer floor under the bottle makes the box too tall for either prior.
                raised = smeared[smeared[:, 2] >= top_z - 0.10]
                if len(raised) >= 15:
                    smeared = raised
            if smeared is not cloud and len(smeared) >= 15:
                track.voxels = _voxelize(smeared)
                cloud = smeared
                centre, size, yaw = box_of(cloud, trim=0.01)
        parent = self._furniture_of(track) if track.category == "drawer" else None
        if parent is not None:
            # A drawer slides square to its furniture: take the furniture's (well-observed) heading.
            k = round(wrap_angle(yaw - parent.yaw) / (math.pi / 2))
            centre, size, yaw = box_at_yaw(_points(track.voxels), parent.yaw + k * math.pi / 2)
        if track.observations > 1:
            previous = float(track.yaw)
            yaw, size = _continuous_heading(yaw, size, previous)
            if _plan_axes_swapped(previous, yaw):
                track.seen_across = _swap_plan_axis(track.seen_across)
        if track.category == "drawer" and track.voxels is not None:
            # The handle is the middle of the front face. A heading a few tens of degrees off
            # slides that point along the drawer, and the gripper misses. These drawers sit
            # square to the walls.
            snapped = round(float(yaw) / (math.pi / 2.0)) * (math.pi / 2.0)
            if abs(wrap_angle(snapped - float(yaw))) > 0.05:
                centre, size, yaw = box_at_yaw(_points(track.voxels), snapped)
        if track.mean_rgb is not None:
            # The accumulated box is the best evidence of what the object is: classify it again,
            # with confidence growing with the number of views behind it.
            mean = {k: v / track.semantic_views for k, v in track.semantic_counts.items()} if track.semantic_views else None
            probs = self.category_probs(size, centre[2] - size[2] / 2, track.mean_rgb, track.mean_invalid,
                                        track.bottom_seen, mean)
            weight = float(min(track.observations, 20))
            # A short track keeps the drawer name when the shape prior prefers the furniture
            # around it. The top band of the sideboard is as wide as the sideboard and still a drawer.
            # A tall face is cut down to that band before this, so height is what decides.
            top = max(probs, key=probs.get) if probs else ""
            furniture = top in ("cabinet", "sideboard", "counter", "table")
            drawer_sized = float(size[2]) <= DRAWER_MAX_HEIGHT and float(max(size[0], size[1])) <= 1.50
            # A later mask of the open drawer must not rename the bear or the pistol.
            # The bear's own mask is wider than the prior because it includes the drawer
            # floor, so the shape prior says drawer even when SAM says teddy bear.
            sam_top = max(mean, key=mean.get) if mean else ""
            sam_content = (sam_top in _CONTENT and top == "drawer"
                           and float(max(size[0], size[1])) < 0.35 and float(size[2]) < 0.25)
            keep_content = (track.category in _CONTENT and top == "drawer"
                            and float(max(size[0], size[1])) < 0.32)
            # A bottle a little taller than its prior, because the mask took the drawer floor,
            # stays that bottle. Replacing the name with unknown lets the next merge swallow it.
            keep_named = (track.category in _CONTENT and top in (
                              "unknown", "bottle", "cup", "box", "", "chair", "table", "sideboard",
                              "cabinet", "wall", "wine_glass", "bowl", "mug", "plate")
                          and float(max(size[0], size[1])) < 0.25 and float(size[2]) < 0.25)
            # An open drawer reads as a box. Renaming it drops the close: the handle plan
            # requires a drawer.
            keep_drawer = (track.category == "drawer" and (
                track.id in self._opened_drawers
                or (drawer_sized and top in ("box", "unknown", "drawer", "", "cabinet", "sideboard",
                                             "counter", "table"))))
            if sam_content:
                track.category_counts = {sam_top: weight}
            elif not keep_content and not keep_named and not keep_drawer:
                track.category_counts = {k: v * weight for k, v in probs.items()}
        # A wine glass is taller than its rim. One already standing on its surface, and wider
        # than it is tall, is the cup; completing it would invent the stem above the rim.
        if (track.category == "wine_glass" and not self._beside_drawer(centre)
                and float(size[2]) < 0.13 and float(max(size[0], size[1])) > float(size[2])):
            bottom = float(centre[2] - size[2] / 2)
            level = self._resting_level(track, centre, bottom)
            if level is None or bottom - level <= 0.05:
                track.category_counts = {"cup": float(min(max(track.observations, 1), 20))}
        proto = prototype(track.category) if track.category_counts else None
        shift = np.zeros(3)
        completion = 0.0
        total = sum(track.category_counts.values())
        measured_bottom = float(centre[2] - size[2] / 2)
        # A shallow cloud resting high up is the table top boxed with a bowl, not a drawer front.
        # Completing it to a drawer's size is what makes that bowl a 22 cm drawer.
        shallow_top = track.category == "drawer" and float(size[2]) < 0.12 and measured_bottom > 0.65
        sheet = track.category in ("manual", "picture")
        # A label (ego_annotate) may name a category perception never accumulated evidence for.
        if proto is not None and not shallow_top and track.category_counts.get(track.category, 0.0) / total > 0.6:
            long_side, short_side, height = proto
            axes = [(0, long_side, np.array([math.cos(yaw), math.sin(yaw), 0.0])),
                    (1, short_side, np.array([-math.sin(yaw), math.cos(yaw), 0.0]))]
            away = centre - cam_p
            view = away[:2] / max(float(np.linalg.norm(away[:2])), 1e-6)
            # The airplane prior is 6 cm tall and the measured wing is 10 cm. Filling the prior
            # width on that box invents a footprint and the track is then merged away.
            too_tall = track.category == "toy_airplane" and float(size[2]) > height
            for k, want, axis in axes:
                across = abs(float(view @ axis[:2])) < SEEN_ACROSS_DOT
                if across and size[k] >= FULL_AXIS * want:
                    track.seen_across |= 1 << k
            for k, want, axis in axes:
                if too_tall or sheet:
                    continue
                # A side remembered from an earlier broadside view keeps that length when the
                # current cloud only shows the end. A facing side already half a metre, and
                # already most of the prior, is the piece: 0.9 m of a 1.2 m table is not a
                # hidden face. A shorter extent facing the camera is still a drawer's depth.
                # A side within a tenth of its prior is still missing its far edge, even
                # when an earlier view saw that side: half a metre of a shorter piece is left.
                near_prior = ((k == 1 or track.category == "counter")
                              and 0.90 * want <= size[k] < 0.98 * want)
                if track.seen_across & (1 << k) and not near_prior:
                    continue
                if not near_prior and size[k] >= SEEN_FACE_M and size[k] >= FULL_AXIS * want:
                    continue
                if abs(float(view @ axis[:2])) < 0.3:
                    continue
                if size[k] < (0.98 if near_prior else 0.85) * want:
                    grow = want - size[k]
                    shift += axis * (grow / 2.0) * (1.0 if float(away @ axis) >= 0 else -1.0)
                    size[k] = want
                    completion = max(completion, grow)
            if size[2] < 0.85 * height:
                grow = height - size[2]
                bottom = centre[2] - size[2] / 2
                if SHAPES.get(track.category, {}).get("floor") and bottom >= 0.15:
                    shift[2] = -grow / 2.0  # floor-standing, seen from above: the top is the seen
                    # surface and the unseen part is below it — growing up would float the box
                elif sheet:
                    shift[2] = -grow / 2.0  # a printed face is the top of the sheet
                elif bottom < 0.15:
                    shift[2] = height / 2.0 - centre[2]  # floor-standing: the base is on the floor
                else:
                    shift[2] = grow / 2.0  # resting on something: the unseen part is above
                size[2] = height
                completion = max(completion, grow)
        # A printed sheet's two faces can quantize to one depth plane. Manual and
        # picture share a thickness prior: ambiguity between them must not erase
        # a repeatedly named, measured surface from the exported object table.
        # Keep its measured outline (a print may be smaller than the prototype).
        # Never turn an unnamed plane, a thin line or a single detection into paper.
        if (sheet and proto is not None and track.observations >= ESTABLISHED_VIEWS
                and track.semantic_views > 0
                and max(track.semantic_counts.get(c, 0.0) for c in ("manual", "picture"))
                    / track.semantic_views >= self.config.semantic_named_score
                and sum(track.category_counts.get(c, 0.0) for c in ("manual", "picture")) / total > 0.6
                and 0.05 <= float(min(size[:2])) <= float(max(size[:2])) <= max_long_extent(track.category)
                and float(size[2]) < proto[2]):
            grow = proto[2] - float(size[2])
            size[2] = proto[2]
            shift[2] -= grow / 2.0  # measured face is the top; missing thickness lies below it
            completion = max(completion, grow)
        centre = centre + shift
        if not track.bottom_seen and SHAPES.get(track.category, {}).get("floor"):
            # Furniture whose base no view has shown (a table seen from close, a wall behind a
            # counter) stands on the floor below its visible part.
            top = centre[2] + size[2] / 2
            size[2], centre[2] = top, top / 2
        bottom = centre[2] - size[2] / 2
        if size[2] < 0.15 and min(size[0], size[1]) >= 0.5 and bottom > 0.3:
            # A wide, thin, level slab in mid-air is the top of furniture whose body no view has
            # joined to it (a real segmenter hands over a tabletop alone); it stands on the floor.
            top = centre[2] + size[2] / 2
            size[2], centre[2] = top, top / 2
            bottom = 0.0
        if track.category == "table":
            proto = prototype("table")
            floor = float(centre[2] - size[2] / 2)
            if proto is not None and float(size[2]) > float(proto[2]) + 0.02 and floor < 0.05:
                size[2] = float(proto[2])
                centre[2] = floor + float(size[2]) / 2
                bottom = floor
        if float(max(size[0], size[1])) < 0.30 and bottom > 0.15:
            # Something small off the floor rests on something: extend the box down to the nearest
            # surface (or receptacle floor) just below its visible part. A view that showed the
            # base of something in a drawer measured where it stands.
            level = self._resting_level(track, centre, bottom, drawer_floors=not track.bottom_seen)
            if level is not None and level < bottom:
                gap = bottom - level
                size[2] += gap
                centre[2] -= gap / 2
                # The name was decided on the visible part. Any gap filled down to the table can
                # push a rim past a bowl's height, so the category is decided again on the filled box.
                if gap >= 0.01 and track.mean_rgb is not None:
                    self._reclassify_filled_height(track, size, centre)
        # Colour is applied after the filled-height rename. That rename uses the shape prior,
        # which calls a yellow airplane a bowl because the airplane prior is only 6 cm tall.
        named = content_color_name(size, track.mean_rgb, track.mean_invalid, track.category)
        if named is None:
            named = sideboard_not_counter(size, track.category, track.mean_rgb, track.mean_invalid)
        if named is None:
            named = partial_pistol(size, track.category, track.mean_rgb, track.mean_invalid)
        if named is None and track.category in ("mug", "cup") and self._beside_drawer(centre):
            # A dark fragment of the pistol is named a mug. The pistol is also a flat
            # mug: a real mug is about 9 cm tall, and this one is only a few centimetres.
            # The airplane beside the same drawer is yellow, and a mug on the table is not here.
            box_long = float(max(size[0], size[1]))
            box_short = float(min(size[0], size[1]))
            dark = track.mean_rgb is None or _color_score("dark", track.mean_rgb, track.mean_invalid) >= 0
            flat = float(size[2]) <= 0.06 and 0.08 <= box_long <= 0.20 and box_short <= 0.12
            if flat or (dark and 0.08 <= box_long <= 0.16 and 0.05 <= box_short <= 0.14 and float(size[2]) <= 0.13):
                named = "pistol"
        if named is None and track.category in ("box", "bowl", "plate") and self._beside_drawer(centre):
            # The pistol prior loses to "box" or "bowl" once the drawer floor is in the cloud.
            # In a drawer, a flat box of that size is the pistol; the floor lightens the colour
            # and adds a few centimetres of height.
            box_long = float(max(size[0], size[1]))
            box_short = float(min(size[0], size[1]))
            if 0.10 <= box_long <= 0.24 and 0.05 <= box_short <= 0.20 and float(size[2]) <= 0.08:
                named = "pistol"
        if named is None and track.category == "unknown" and self._beside_drawer(centre):
            # The perfume's mask is the drawer floor, so the mean is not pink and the name stays
            # unknown. The pill is shorter than this.
            box_long = float(max(size[0], size[1]))
            box_short = float(min(size[0], size[1]))
            if box_long <= 0.12 and box_short <= 0.09 and 0.09 < float(size[2]) < 0.13:
                named = "perfume_bottle"
            elif box_long <= 0.12 and box_short <= 0.08 and 0.06 < float(size[2]) < 0.16:
                named = "pill_bottle"
        if named is None:
            named = perfume_without_bottle_label(size, track.category, track.semantic_counts)
            # The perfume is 4 cm on its short side. A thinner smear is the pill, named below.
            if named is not None and (not self._beside_drawer(centre) or float(min(size[0], size[1])) <= 0.035):
                named = None
            if named is None and track.category == "bottle" and not self._beside_drawer(centre):
                # The wine glass is a bottle to the segmenter. The glass is taller than the
                # perfume, and it stands on the table, not in a drawer.
                box_long = float(max(size[0], size[1]))
                box_short = float(min(size[0], size[1]))
                if 0.15 < float(size[2]) < 0.22 and box_short <= 0.12 and box_long <= 0.16:
                    named = "wine_glass"
            if named is None and track.category == "cup" and not self._beside_drawer(centre):
                # A cup this tall on the table is the wine glass with the stem only partly seen.
                # A cup whose box is 10 cm stays a cup.
                box_long = float(max(size[0], size[1]))
                box_short = float(min(size[0], size[1]))
                if 0.11 < float(size[2]) < 0.16 and box_short <= 0.12 and box_long <= 0.14:
                    named = "wine_glass"
            if named is None:
                named = clear_stem_name(size, track.category, track.mean_rgb, track.mean_invalid)
        # The bear is in a drawer. The same brown, on the table, is the wine glass. The measured
        # box is a hair over the stem gate, so this check is the one that applies.
        if ((named == "teddy_bear") or track.category == "teddy_bear") and not self._beside_drawer(centre):
            box_long = float(max(size[0], size[1]))
            box_short = float(min(size[0], size[1]))
            if box_long <= 0.16 and box_short <= 0.13 and 0.15 < float(size[2]) < 0.22:
                named = "wine_glass"
        if self._beside_drawer(centre):
            box_long = float(max(size[0], size[1]))
            box_short = float(min(size[0], size[1]))
            yellow = False
            if track.mean_rgb is not None:
                red, green, blue = (float(channel) for channel in track.mean_rgb)
                yellow = (red > green > blue and red > 0.7 and green > 0.55
                          and (red - blue) / max(red, 1e-6) > 0.4)
            # A thin smear of the drawer floor beside the perfume is the pill. The perfume
            # itself is wider than 3 cm on its short side.
            if ((named or track.category) in (
                    "bottle", "unknown", "wine_glass", "cup")
                    and box_long <= 0.08 and box_short <= 0.035 and float(size[2]) > 0.05):
                named = "pill_bottle"
            # The pistol is flat, but the drawer front behind it makes a tall thin box, and
            # that box is named a wine glass. A glass is not this thin.
            elif (not yellow and box_short <= 0.06 and 0.09 <= box_long <= 0.22
                  and 0.05 < float(size[2]) < 0.22
                  and (named or track.category) in (
                      "wine_glass", "cup", "bowl", "unknown", "box", "mug", "bottle")):
                named = "pistol"
            # A flat pistol returns one edge. The box is a line, and the furniture-edge
            # rule would otherwise throw that line away.
            elif ((named or track.category) == "unknown" and box_short <= 0.02
                  and 0.10 <= box_long <= 0.22 and float(size[2]) <= 0.04):
                named = "pistol"
            # The pill's mask is named a cup. A cup this narrow, in the drawer, is the bottle.
            # A mug is wider than 10 cm and is not renamed.
            elif ((named or track.category) == "cup" and box_long <= 0.12
                  and box_short <= 0.08 and 0.06 < float(size[2]) < 0.14):
                named = "pill_bottle"
            # The drawer's floor makes the airplane as tall as a glass, and the shape prior
            # then says bear. Yellow, or a stem name on a box too long for a glass, is the airplane.
            elif (yellow and (named or track.category) == "unknown"
                  and box_long >= 0.15 and box_short >= 0.06 and 0.06 <= float(size[2]) < 0.12):
                # Seen flat, the airplane is under the glass-height rule and still unnamed.
                named = "toy_airplane"
            elif yellow and box_long >= 0.11 and box_short >= 0.06 and 0.12 <= float(size[2]) < 0.20:
                if (named or track.category) in (
                        "cup", "wine_glass", "bowl", "box", "unknown", "mug", "plate", "teddy_bear"):
                    named = "toy_airplane"
            elif ((named or track.category) == "wine_glass" and box_long >= 0.12
                  and float(size[2]) >= 0.12):
                named = "toy_airplane"
        # A wine glass is taller than its rim. A wider, shorter box is the cup. So is a box
        # only a centimetre over the cup's height when it already stands on the counter:
        # the glass in this measurement stands on the table.
        on_counter = self._stands_on_counter(centre, float(centre[2] - size[2] / 2))
        if ((named or track.category) == "wine_glass" and not self._beside_drawer(centre)
                and ((float(max(size[0], size[1])) > float(size[2]) and float(size[2]) < 0.15)
                     or (on_counter and float(size[2]) < 0.13
                         and float(max(size[0], size[1])) >= 0.07))):
            named = "cup"
        if named is not None and named != track.category:
            track.category_counts = {named: float(min(max(track.observations, 1), 20))}
        elif (set(track.category_counts) == {"wine_glass"} and track.semantic_counts
              and set(track.semantic_counts) != {"wine_glass"}
              and clear_stem_name(size, "cup", track.mean_rgb, track.mean_invalid) is None):
            # The stem override wiped the segmenter's name. This box is not stemware, so that name
            # applies again: a cup whose filled height briefly looked like a glass.
            track.category_counts = {k: float(v) for k, v in track.semantic_counts.items()}
        if track.category == "pistol" and float(size[2]) > 0.06 and self._beside_drawer(centre):
            # The pistol lies on the drawer floor. The tall part of the cloud is the front
            # behind it, so the bottom of the box is the pistol and the rest is not.
            extra = float(size[2]) - 0.04
            size = size.copy()
            size[2] = 0.04
            centre = centre.copy()
            centre[2] -= extra / 2.0
        if track.category == "pistol" and self._beside_drawer(centre):
            short_k = 0 if float(size[0]) <= float(size[1]) else 1
            if float(size[short_k]) < 0.04:
                # One edge of the pistol. Grow the missing width toward the drawer it lies in.
                axis = (np.array([math.cos(yaw), math.sin(yaw)]) if short_k == 0
                        else np.array([-math.sin(yaw), math.cos(yaw)]))
                drawers = [t for t in self.tracks.values() if t.category == "drawer"]
                if drawers:
                    drawer = min(drawers, key=lambda t: float(np.linalg.norm(t.pos[:2] - centre[:2])))
                    toward = drawer.pos[:2] - centre[:2]
                    if float(toward @ axis) < 0.0:
                        axis = -axis
                    grow = 0.12 - float(size[short_k])
                    centre = centre.copy()
                    centre[:2] = centre[:2] + axis * (grow / 2.0)
                    size = size.copy()
                    size[short_k] = 0.12
        if track.category == "wall":
            # A wall is a 10 cm panel. Depth noise boxes it as a thick slab that fills the
            # floor in front of it, and the base then cannot stand there to close a drawer.
            short_k = 0 if float(size[0]) <= float(size[1]) else 1
            if float(size[short_k]) > 0.15:
                axis = (np.array([math.cos(yaw), math.sin(yaw)]) if short_k == 0
                        else np.array([-math.sin(yaw), math.cos(yaw)]))
                toward_cam = cam_p[:2] - centre[:2]
                if float(toward_cam @ axis) < 0.0:
                    axis = -axis
                extra = float(size[short_k]) - 0.10
                centre = centre.copy()
                centre[:2] = centre[:2] + axis * (extra / 2.0)
                size = size.copy()
                size[short_k] = 0.10
        if (track.category == "wine_glass" and not self._beside_drawer(centre)
                and float(min(size[0], size[1])) >= 0.08 and float(size[2]) > 0.12
                and self._stands_on_counter(centre, float(centre[2] - size[2] / 2))):
            # The cup stands on the counter. A vertical smear makes it as tall as a glass;
            # the glass in this measurement stands on the table.
            extra = float(size[2]) - 0.10
            size = size.copy()
            size[2] = 0.10
            centre = centre.copy()
            centre[2] -= extra / 2.0
            track.category_counts = {"cup": float(min(max(track.observations, 1), 20))}
        if track.category == "toy_airplane" and float(size[2]) > 0.10 and self._beside_drawer(centre):
            # The wing is the top of the cloud. The rest is the drawer floor under it, and that
            # floor pulls the centre down.
            extra = float(size[2]) - 0.08
            size = size.copy()
            size[2] = 0.08
            centre = centre.copy()
            centre[2] += extra / 2.0
        track.pos = centre
        track.size = size
        track.yaw = yaw
        track.completion = completion
        sigma_depth = self.config.pos_sigma_base + self.config.pos_sigma_per_m * 0.5 * rng_m
        views = max(1, min(track.observations, 20))
        var_xy = sigma_depth**2 / views + (completion / 4.0) ** 2 + 0.002**2
        track.var = np.array([var_xy, var_xy, sigma_depth**2 / views + 0.002**2])
        track.yaw_var = 0.05**2 / views

    def _reclassify_filled_height(self, track: Track, size: np.ndarray, centre: np.ndarray) -> None:
        """Replace the category using the box after it was extended down onto its support.

        Labels whose tallest plausible height is below ``size[2]`` are left out. The shape prior
        then names the object from the filled height: a rim extended across a stem becomes a
        wine glass, while the label it was given as a shallow disk does not stay.
        """
        if track.category == "drawer" or track.id in self._opened_drawers:
            return
        mean = {k: v / track.semantic_views for k, v in track.semantic_counts.items()} if track.semantic_views else None
        if mean:
            mean = _labels_within_height(mean, float(size[2]))
        probs = self.category_probs(size, float(centre[2] - size[2] / 2), track.mean_rgb, track.mean_invalid,
                                    track.bottom_seen, mean or None)
        weight = float(min(track.observations, 20))
        track.category_counts = {k: v * weight for k, v in probs.items()}

    def _gate(self, track: Track, sigma: float) -> float:
        """Association gate: measurement and track uncertainty, widened for large objects whose
        detected box centre moves with the visible portion, and for what lies in the held
        container: it slides about in there while the container is carried."""
        gate = max(self.config.gate_m, 3.0 * math.sqrt(float(np.mean(track.var)) + sigma**2),
                   self.config.size_gate_ratio * float(max(track.size[0], track.size[1])))
        if track.carried and self.held_id in self.tracks:
            container = self.tracks[self.held_id]
            gate = max(gate, 0.5 * float(max(container.size[0], container.size[1])))
        return max(gate, 0.45) if track.held else gate

    def _size_match(self, a: float, b: float) -> bool:
        """Footprints of the same object agree within the detector's size noise; nested bowls do not."""
        return abs(math.log(max(a, 1e-3) / max(b, 1e-3))) <= self.config.size_match_log

    def _different_bowls(self, a: float, b: float) -> bool:
        """Whether two bowl footprints differ by more than one noisy view of a single bowl."""
        return abs(math.log(max(a, 1e-3) / max(b, 1e-3))) > self.config.nested_bowl_log

    def _separate_bowls(self, foot_a: float, foot_b: float, xy_a: np.ndarray, xy_b: np.ndarray,
                        size_a: np.ndarray | None = None, size_b: np.ndarray | None = None) -> bool:
        """Whether two bowl regions are a smaller bowl nested in a larger one.

        Nesting shares a centre: the offset stays within :data:`NEST_CENTER_SHARE` of the smaller
        radius. A partial view of one bowl is shifted further, toward the side the camera saw.
        Bowls are round: a box much longer than wide is a smear of one bowl, not a second bowl.

        :param foot_a: footprint of the first region, in metres.
        :param foot_b: footprint of the second region, in metres.
        :param xy_a: centre of the first region.
        :param xy_b: centre of the second region.
        :param size_a: box of the first region, when known.
        :param size_b: box of the second region, when known.
        :returns: true when the footprints differ by more than one view of one bowl and the centres coincide.
        """
        if not self._different_bowls(foot_a, foot_b):
            return False
        small, big = min(foot_a, foot_b), max(foot_a, foot_b)
        if small < 0.10 and small > 0.55 * big:
            return False  # a partial view of one cup, not a second one inside it: nothing nests in a cup
        for size in (size_a, size_b):
            if size is not None and float(max(size[0], size[1])) > 1.4 * float(min(size[0], size[1])):
                return False
        offset = float(np.linalg.norm(np.asarray(xy_a)[:2] - np.asarray(xy_b)[:2]))
        return offset <= NEST_CENTER_SHARE * min(foot_a, foot_b) / 2

    def _candidates(self, category: str, pos: np.ndarray, sigma: float, footprint: float | None = None) -> list[Track]:
        """Tracks a detection may belong to: a compatible category, a matching size and within the gate."""
        out = []
        for track in self.tracks.values():
            if not compatible(category, track.perceived_category):
                continue
            if footprint is not None and not track.held and not self._size_match(footprint, track.footprint):
                continue
            if float(np.linalg.norm(track.pos - pos)) <= self._gate(track, sigma):
                out.append(track)
        return out

    def _forget(self, track_id: str) -> None:
        """Delete a track, and never silently amputate what the gripper is holding: when the load
        is the track that would go, the pair is not merged after all."""
        if track_id == self.held_id:
            return
        track = self.tracks.get(track_id)
        if track is not None:
            self._drop(track, "forgotten")

    def _merge_duplicates(self, apart: list[set[str]] | None = None) -> None:
        """Merge compatible tracks whose centres lie inside each other's uncertainty-expanded
        footprint; the track with more observations keeps its id. Two tracks one camera saw as two
        detections in this frame (``apart``: the tracks each camera's detections went to) are two
        objects, however close: pieces lying edge to edge are not one piece."""
        ids = sorted(self.tracks, key=lambda t: -self.tracks[t].observations)
        for i, keep_id in enumerate(ids):
            keep = self.tracks.get(keep_id)
            if keep is None or keep.held or keep.carried:
                continue
            for drop_id in ids[i + 1:]:
                drop = self.tracks.get(drop_id)
                if drop is None or drop.held or drop.carried or not compatible(drop.perceived_category,
                                                                               keep.perceived_category):
                    continue
                if _small_observations_apart(keep.span, drop.span):
                    continue
                if apart and any(keep_id in seen and drop_id in seen for seen in apart):
                    continue
                if not self._size_match(drop.footprint, keep.footprint):
                    continue
                z_tol = max(0.5 * max(keep.size[2], drop.size[2], 0.1), 3.0 * _sigma_z(keep, drop))
                if abs(drop.pos[2] - keep.pos[2]) > z_tol:
                    continue
                if not (_inside(keep, drop.pos, 3.0 * keep.sigma) or _inside(drop, keep.pos, 3.0 * drop.sigma)):
                    continue
                if (_content_named(keep) and _content_named(drop)
                        and float(np.linalg.norm(drop.pos[:2] - keep.pos[:2])) > 0.12):
                    continue  # two contents in one drawer, not one bottle seen twice
                if _flat_pair(keep, drop) and float(np.linalg.norm(drop.pos[:2] - keep.pos[:2])) > FLAT_PAIR_OVERLAP_M * min(
                        min(keep.size[0], keep.size[1]), min(drop.size[0], drop.size[1])):
                    continue  # two flat pieces edge to edge, not one seen twice
                if (keep.category == "bowl" and drop.category == "bowl" and _tighter_bowl(drop, keep)) \
                        or _slice_of(drop, keep):
                    # Averaging with the wider bowl walks the centre onto the smear, and a slab
                    # cut from a piece of furniture is the whole piece's box, not a second centre.
                    keep.pos = drop.pos.copy()
                    keep.size = drop.size.copy()
                    keep.yaw = drop.yaw
                    keep.var = drop.var.copy()
                elif _slice_of(keep, drop):
                    pass  # the slab adds its views to the piece; its centre is one edge of it
                else:
                    w_keep, w_drop = 1.0 / keep.var, 1.0 / drop.var
                    keep.pos = (keep.pos * w_keep + drop.pos * w_drop) / (w_keep + w_drop)
                    keep.var = 1.0 / (w_keep + w_drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                _absorb_semantics(keep, drop)
                for col, n in drop.color_counts.items():
                    keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                keep.last_seen = max(keep.last_seen, drop.last_seen)
                _absorb_hardware_views(keep, drop)
                self._drop(drop, "merged: duplicate", into=keep)

    def _new_track(self, det: dict, pos: np.ndarray, yaw: float, sigma: float) -> Track:
        track = Track(
            id=f"e:{next(self._ids)}",
            category_counts={det["category"]: 1.0},
            pos=pos.copy(),
            var=np.full(3, sigma**2),
            yaw=yaw,
            yaw_var=0.05**2,
            size=np.asarray(det["size"], float),
            observations=1,
            first_seen=self.stamp,
            last_seen=self.stamp,
            color_counts={det["color"]: 1.0} if det.get("color") else {},
        )
        self._remember_screw_holes(track, det, pos, yaw)
        self.tracks[track.id] = track
        return track

    @staticmethod
    def _remember_screw_holes(track: Track, det: dict, pos: np.ndarray, yaw: float) -> None:
        """Re-express detection-frame holes in the current fused entity frame.

        The sensor's object pose can differ from the filtered pose. Copying local coordinates
        across that difference would move/rotate the observed features a second time.
        """
        holes = det.get("screwHoles")
        if holes is None:
            return
        c, sn = math.cos(yaw), math.sin(yaw)
        observed_r = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        c, sn = math.cos(track.yaw), math.sin(track.yaw)
        entity_r = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        track.screw_holes = [{"id": str(h["id"]),
                              "p": (entity_r.T @ (np.asarray(pos) + observed_r @ np.asarray(h["p"][:3], float)
                                                  - track.pos)).tolist()} for h in holes]

    def _newcomer(self, pos: np.ndarray, size: np.ndarray, sigma: float, others: list[tuple[np.ndarray, np.ndarray, float]]) -> str:
        """How a detection no track explains enters the belief. ``"carried"``: it lies in the held
        container, which is upright, and moves with it. ``"skip"``: it cannot be a resting object, so
        no track is made: it is in the held object that holds nothing, in the held container while
        that is tipped out, or clear of the floor and of every track and detection that could hold it
        (in flight, perceived where it lands). ``"rest"``: anything else.

        Args:
            pos: Centre of the detection in the world frame.
            size: Its extent along its own axes.
            sigma: Standard deviation (m) of its position; the gap allowed below it grows by twice
                that, and by twice the sigma of the track it may rest on.
            others: Centre, extent and heading of the other detections of the same frame, which may be
                what it rests on before their own tracks exist.

        Returns:
            ``"carried"``, ``"skip"`` or ``"rest"``.
        """
        cfg = self.config
        bottom = float(pos[2] - size[2] / 2)
        held = self.tracks.get(self.held_id) if self.held_id else None
        if held is not None and _inside(held, pos, cfg.content_margin_m) and held.bottom() - cfg.content_margin_m <= pos[2] <= held.top() + cfg.content_margin_m:
            holds_contents = "receptacle" in held.affordances or "pourable" in held.affordances
            return "carried" if holds_contents and not self._tipped(held) else "skip"
        gap = cfg.rest_gap_m + 2 * sigma
        if bottom <= gap:
            return "rest"
        for track in self.tracks.values():
            tol = gap + 2 * track.sigma
            if not track.held and _inside(track, pos, tol) and track.bottom() - tol <= bottom <= track.top() + tol:
                return "rest"
        for centre, extent, yaw in others:
            tol = 2 * gap
            c, s = math.cos(-yaw), math.sin(-yaw)
            dx, dy = float(pos[0] - centre[0]), float(pos[1] - centre[1])
            within = abs(c * dx - s * dy) <= extent[0] / 2 + tol and abs(s * dx + c * dy) <= extent[1] / 2 + tol
            low, high = float(centre[2] - extent[2] / 2), float(centre[2] + extent[2] / 2)
            if within and low - tol <= bottom <= high + tol:
                return "rest"
        return "skip"

    def _within(self, container: Track, track: Track) -> bool:
        """Whether ``track`` lies in ``container``'s volume, which holds contents: a piece stacked on
        another in a box is in the box, though the box's floor is not what it rests on."""
        if "receptacle" not in container.affordances and "pourable" not in container.affordances:
            return False
        margin = self.config.content_margin_m
        return (track.footprint < container.footprint and _inside(container, track.pos, margin)
                and container.bottom() - margin <= track.bottom() and track.pos[2] <= container.top() + margin)

    def _tipped(self, track: Track, rot: np.ndarray | None = None) -> bool:
        """Whether the held ``track`` is tilted beyond ``poured_roll`` from how it was taken, so what
        lay in it has slid out. ``rot``: the gripper rotation to judge by (default: the belief's)."""
        if track.held_up is None:
            return False
        up = (self.gripper_rotation(self.held_gripper) if rot is None else rot) @ track.held_up
        return math.acos(min(1.0, max(-1.0, float(up[2])))) > self.config.poured_roll

    def _fuse(self, track: Track, det: dict, pos: np.ndarray, yaw: float, sigma: float) -> None:
        track.category_counts[det["category"]] = track.category_counts.get(det["category"], 0.0) + 1.0
        if det.get("color"):
            track.color_counts[det["color"]] = track.color_counts.get(det["color"], 0.0) + 1.0
        track.observations += 1
        track.last_seen = self.stamp
        track.existence = min(1.0, track.existence + 0.2 * (1.0 - track.existence))
        track.absent_views.clear()
        if track.held:
            self._refine_held(track, pos, sigma)
            self._remember_screw_holes(track, det, pos, yaw)
            return
        if track.carried:
            # Inside the carried container it may have slid: it follows the view, and keeps its
            # new place in the container (in the gripper's frame, as the container moves).
            container = self.tracks.get(self.held_id) if self.held_id else None
            if container is not None:
                track.pos = track.pos + 0.5 * (pos - track.pos)
                track.carried_offset = self.gripper_rotation(self.held_gripper).T @ (track.pos - container.pos)
            return
        r = sigma**2
        prior_var = track.var + self.config.process_noise
        gain = prior_var / (prior_var + r)
        track.pos = track.pos + gain * (pos - track.pos)
        track.var = (1.0 - gain) * prior_var
        turned = abs(wrap_angle(yaw - track.yaw)) > YAW_MOVED
        if turned and track.yaw_var < YAW_MOVED**2:
            # Its heading is settled: one view at another heading is more likely a neighbour's
            # detection than a turn, so a turn needs yaw_confirm_views such views in a row.
            track.yaw_conflicts += 1
            if track.yaw_conflicts >= self.config.yaw_confirm_views:
                track.yaw_var = YAW_MOVED**2  # it was turned: follow the views
        else:
            track.yaw_conflicts = 0
        if not turned or track.yaw_var >= YAW_MOVED**2:
            k_yaw = track.yaw_var / (track.yaw_var + 0.05**2)
            track.yaw = wrap_angle(track.yaw + k_yaw * wrap_angle(yaw - track.yaw))
            track.yaw_var = (1.0 - k_yaw) * track.yaw_var
            track.yaw_conflicts = 0
        n = track.observations
        track.size = track.size + (np.asarray(det["size"], float) - track.size) / n
        self._remember_screw_holes(track, det, pos, yaw)

    def _refine_held(self, track: Track, pos: np.ndarray, sigma: float) -> None:
        """Correct where the held ``track`` sits in the gripper from a detection of it there. The
        grasp takes the load where it really lay, which the estimate before the grasp missed by its
        error, and a place puts down where the estimate holds it; without this correction that error
        carries into every placement. A view from before the gripper closed, one beyond
        ``held_view_gate_m`` of the load, or one as near another track (a neighbour the load is
        lowered beside) corrects nothing."""
        gate = self.config.held_view_gate_m
        if self.stamp < self.held_since - 1e-6 or track.held_offset is None or np.linalg.norm(pos - track.pos) > gate:
            return
        if any(other is not track and not other.carried and np.linalg.norm(other.pos - pos) <= gate
               for other in self.tracks.values()):
            return
        gain = track.var / (track.var + sigma**2)
        track.pos = track.pos + gain * (pos - track.pos)
        track.var = (1.0 - gain) * track.var
        track.held_offset = self.gripper_rotation(self.held_gripper).T @ (track.pos - self.gripper_position(self.held_gripper))

    def _decay_unseen(self, cams: dict[str, dict], matched: set[str]) -> None:
        """Fade every unmatched track that lies well inside the view of at least one of ``cams``
        (camera id → pose): once per frame, however many cameras should have seen it."""
        views = []
        for camera, cam in cams.items():
            intrinsics = self.camera_intrinsics(camera)
            fovy = math.radians(intrinsics["fovy"])
            half_v = fovy / 2 * 0.8
            half_h = math.atan(math.tan(fovy / 2) * intrinsics["width"] / intrinsics["height"]) * 0.8
            views.append((pose_inverse(cam), half_h, half_v))

        def in_view(pos: np.ndarray) -> bool:
            for inv, half_h, half_v in views:
                local = pose_compose(inv, {"p": pos.tolist(), "q": [1, 0, 0, 0]})["p"]
                if (0.3 < local[0] < self.config.max_range * 0.8 and abs(math.atan2(local[1], local[0])) <= half_h
                        and abs(math.atan2(local[2], local[0])) <= half_v):
                    return True
            return False

        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in matched or track.held or track.carried or self._in_closed_drawer(track) \
                    or self._in_open_drawer(track):
                continue
            if not in_view(track.pos):
                continue
            track.existence *= self.config.missed_decay
            if track.existence < self.config.drop_below:
                self._drop(track, "faded: not detected where it should be visible")

    def forget_objects(self) -> None:
        """Drop every perceived object except the one in the gripper (used when the kind of sensor
        data changes and objects must be perceived afresh)."""
        held = self.tracks.get(self.held_id) if self.held_id else None
        self.tracks = {held.id: held} if held is not None else {}
        self.assembly_memory = AssemblyMemory()
        self.stationary_assemblies = StationaryAssemblies()
        self.manipulation_envelopes = {}
        self.flip_capture = None
        self.rigid_carry = {}  # perceptually corroborated members, relative to the held gripper
        self.fastening_attempts = {}  # robot actions and sensory evidence; never mechanical truth
        self.released_interfaces = {}
        self.fastener_origins = {}  # perceived pickup spots, discarded with the sensor session
        self._assembly_views = {}
        self.perception_stamps.clear()
        self.version += 1

    def reboot(self) -> None:
        """Forget everything, as a power cycle would: the world's clock went backwards, so no track,
        load or encoder reading from before is about this episode. The body model stays; the id
        counter goes on so a console label never means two different objects."""
        self.tracks = {}
        self.assembly_memory = AssemblyMemory()
        self.stationary_assemblies = StationaryAssemblies()
        self.manipulation_envelopes = {}
        self.flip_capture = None
        self.rigid_carry = {}  # perceptually corroborated members, relative to the held gripper
        self.fastening_attempts = {}  # robot actions and sensory evidence; never mechanical truth
        self.released_interfaces = {}
        self.fastener_origins = {}  # perceived pickup spots, discarded with the sensor session
        self._assembly_views = {}
        self.perception_stamps.clear()
        self.stamp = 0.0
        self.held_since = 0.0
        self.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
        self.joints = {}
        self.gripper = {"state": "open", "holding": False}
        self.grippers = {}
        self.held_id = None
        self.held_gripper = None
        self.version += 1

    # ---------------------------------------------------------------- holding
    def mark_held(self, track_id: str, since: float | None = None, gripper: str | None = None,
                  at: list[float] | None = None, body: dict | None = None) -> str:
        """Record the grasp, and carry its load with the arm, together with what lies in it.

        ``since`` is the body's clock at the moment the gripper closed: a frame rendered before
        that moment still reports an open gripper, and must not be read as the load having been
        dropped. With a real segmenter frames are seconds apart, so without this every grasp would
        be forgotten by the next frame that arrives.

        ``gripper`` is the link that took the load. A dual-arm body needs it: the load follows the
        arm that is actually holding it, and the other arm's state says nothing about it. ``at`` is
        the pinch point the gripper closed at -- without it the offset is measured against the
        encoders from *before* the approach, and the load hangs half a metre from the hand.
        ``body`` (a sensor header: ``odometry`` and ``joints``) is the body at the grasp: the
        gripper's pose -- its rotation, and its pinch point unless ``at`` is given -- is read there
        rather than at the last perception update, which a slow segmenter leaves seconds behind,
        from before the base docked and the arm reached down.

        A previous mark still standing when a new grasp is recorded cannot still be true (the
        gripper opened to take this load), and no frame may ever arrive to clear it when skills run
        without looks, so it is dropped here. If the tracked object was merged away during the
        approach, the track nearest the gripper takes the load instead.

        :returns: the id of the held track.
        :raises KeyError: no track is near the gripper."""
        requested_id = track_id
        self.rigid_carry.clear()
        if track_id != self.held_id:
            self._unmark_held(self.tracks.get(self.held_id))
        # The body at the grasp, resolved first: the fallback search for the load and its offset
        # both read the gripper there (the last perception update can be a drive old).
        joints, odometry = (body or {}).get("joints"), (body or {}).get("odometry")
        if joints is None or odometry is None:
            joints = odometry = None
        closed_at = self.gripper_position(gripper, joints, odometry) if at is None else np.asarray(at, dtype=float)
        if track_id not in self.tracks:
            grip = closed_at
            near = min(self.tracks.values(), key=lambda t: float(np.linalg.norm(t.pos - grip)), default=None)
            if near is None or float(np.linalg.norm(near.pos - grip)) > HELD_FALLBACK_M:
                raise KeyError(f"{track_id} is no longer in the ego state and nothing else is in the gripper")
            track_id = near.id
        track = self.tracks[track_id]
        from steward.ego.fasten_release import grasp_frame
        track.held_interface = grasp_frame(self._track_entity(track), closed_at,
            self.gripper_rotation(gripper, joints, odometry))
        self.released_interfaces = {eid:f for eid,f in self.released_interfaces.items()
                                    if eid != track_id and f['body'] != track_id}
        self.assembly_memory.invalidate(track_id)
        if self.stationary_assemblies.first_grasp_at is None:
            self.stationary_assemblies.first_grasp_at = float(self.stamp)
        self.stationary_assemblies.invalidate(requested_id, track, self.stamp)
        self.stationary_assemblies.invalidate(track_id, track, self.stamp)
        track.grasp_association = {
            "requested": requested_id, "held": track_id, "category": track.category,
            "arrivalCorrection": copy.deepcopy(track.grasp_estimate_correction),
            "positionAtGrasp": track.pos.tolist(), "gripperAtGrasp": closed_at.tolist(),
            "observedAt": float(track.last_seen), "recordedAt": float(self.stamp),
            "distanceToGripperM": float(np.linalg.norm(track.pos - closed_at)),
            "source": "ego_track_before_transport", "identityVerified": False,
            **({"protectedObservedAgeS": track.pin_observed_age}
               if track.pinned and track.pin_observed_age is not None else {}),
        }
        track.placement = None
        track.interface_motion_at = float(self.stamp)
        track.interface_last_geometry = None
        track.interface_best_near = None
        track.interface_estimate_at = None
        track.interface_views.clear()  # pre-transport camera samples are not a new body frame
        # Offsets are kept in the gripper's frame: what it holds turns with it (a lid taken by its
        # edge swings round the hand when the base turns), and so does what lies in that.
        rot = self.gripper_rotation(gripper, joints, odometry)
        for other in self.tracks.values():
            if other is not track and not other.held and (self.support_of(other) is track or self._within(track, other)):
                other.carried_offset = rot.T @ (other.pos - track.pos)
                other.voxels = None
                other.span = None
        self.held_since = max(self.stamp, float(since) if since is not None else self.stamp)
        self.held_gripper = gripper
        track.held = True
        track.held_offset = rot.T @ (track.pos - closed_at)
        track.var = np.maximum(track.var, self.config.grasp_shift_m**2)
        track.held_yaw = wrap_angle(track.yaw - math.atan2(rot[1, 0], rot[0, 0]))
        track.grasp_world_yaw = float(track.yaw)
        track.held_up = rot.T @ np.array([0.0, 0.0, 1.0])
        from steward.ego.fasten_geometry import transform_feature, nominal_tool
        feature = track.tool_feature or nominal_tool(track.category, track.pos, track.size,
                                                     matrix_from_quat(quat_from_yaw(track.yaw)), self.stamp)
        track.held_tool_feature = transform_feature(feature, rot.T, -rot.T @ closed_at) if feature else None
        if track.held_tool_feature:
            track.held_tool_feature['graspWrist'] = {
                k:float((joints or self.joints).get(k, 0.)) for k in ('wrist_roll','wrist_pitch')}
            if self.stamp-feature['stamp'] > 20:
                track.held_tool_feature.setdefault('warnings', []).append(dict(reason='stale_grasp_endpoint'))
        track.recent.clear()  # it will be put down elsewhere
        self.held_id = track_id
        return track_id

    def release_held(self, at: list[float] | None = None) -> None:
        """Record the release, from the skill that just opened the gripper.

        ``at`` is where the load was let go -- the place target. The robot knows what it did; the
        belief should say so at once instead of waiting for a look that may be seconds away (or,
        at the end of a round, never come). Without ``at`` the last known position stands and the
        track is simply unsettled: the next look places it."""
        track = self.tracks.get(self.held_id) if self.held_id else None
        if track is not None and at is not None:
            track.pos = np.asarray(at, dtype=float)
        self._unmark_held(track)

    def _unmark_held(self, track) -> None:
        self.held_id = None
        self.held_gripper = None
        for other in self.tracks.values():
            if other.carried:
                other.carried_offset = None  # it lies in the container where that was put down
                other.settled = False
                other.var = np.maximum(other.var, 0.02**2)
        if track is None:
            return
        if track.appearance is not None:
            track.appearance.stale = True  # it may have been put down turned over
        track.held = False
        track.held_offset = None
        track.grasp_world_yaw = None
        track.grasp_association = None
        track.voxels = None
        track.span = None
        track.settled = False  # it settles where it was put down; the next view places it
        track.var = np.maximum(track.var, 0.02**2)
        track.yaw_var = max(track.yaw_var, YAW_MOVED**2)

    def remember_face(self, track_id: str, rgb: np.ndarray, stamp: float) -> Appearance | None:
        """Keep one view of a track's top face, sampled on its appearance grid (see
        :meth:`face_grid`): seen cells replace the remembered ones; a stale appearance, or one laid
        out for a clearly different extent, is replaced whole.

        :returns: the track's appearance, or None when the track is gone or held."""
        track = self.tracks.get(track_id)
        if track is None or track.held or track.carried:
            return None
        known = track.appearance
        if known is None or known.stale or known.rgb.shape != rgb.shape:
            track.appearance = Appearance(rgb.copy(), (float(track.size[0]), float(track.size[1])), stamp)
            return track.appearance
        seen = np.isfinite(rgb[..., 0])
        known.rgb[seen] = rgb[seen]
        known.stamp = stamp
        return known

    def face_grid(self, track: Track, cell: float) -> tuple[int, int]:
        """Columns and rows of a track's appearance grid: kept from its appearance while the track's
        extent stays within :data:`APPEARANCE_SIZE_CHANGE` of the extent it was laid out for, else
        ``cell``-sized cells over the current extent."""
        known = track.appearance
        if known is not None and not known.stale and all(
                abs(float(track.size[k]) - known.size[k]) <= APPEARANCE_SIZE_CHANGE * known.size[k] for k in (0, 1)):
            return known.rgb.shape[1], known.rgb.shape[0]
        return max(1, round(float(track.size[0]) / cell)), max(1, round(float(track.size[1]) / cell))

    def _read_grippers(self, frame: dict) -> None:
        """The gripper state this frame reports, for a single gripper or for one per arm.

        A one-armed robot sends ``gripper``. A dual-arm body sends ``grippers`` per link and no
        summary, because "closed" would have to say which arm — so while a load is held this reads
        the arm that holds it, and otherwise reports whether any of them is holding."""
        self.grippers = dict(frame.get("grippers") or {})
        if self.held_gripper in self.grippers:
            self.gripper = self.grippers[self.held_gripper]
        elif frame.get("gripper"):
            self.gripper = frame["gripper"]
        elif self.grippers:
            self.gripper = {"state": "closed" if any(g.get("state") == "closed" for g in self.grippers.values()) else "open",
                            "holding": any(g.get("holding") for g in self.grippers.values()),
                            "holdingHandle": any(g.get("holdingHandle") for g in self.grippers.values())}
        else:
            self.gripper = {"state": "open", "holding": False, "holdingHandle": False}

    def _clear_held(self) -> None:
        if self.stamp < self.held_since - 1e-6:
            return  # this frame was rendered while the gripper was still open
        track = self.tracks.get(self.held_id) if self.held_id else None
        # It is where the hand let go of it, not where it was picked up: the arm carried it
        # across the table, and a look taken after the release must recognise it there rather
        # than find a second bowl at the old place.
        if track is not None and track.held_offset is not None:
            track.pos = self.gripper_position(self.held_gripper) + self.gripper_rotation(self.held_gripper) @ track.held_offset
        self._unmark_held(track)

    def _follow_held(self) -> None:
        if not (self.held_id and self.held_id in self.tracks):
            return
        track = self.tracks[self.held_id]
        rot = self.gripper_rotation(self.held_gripper)
        track.pos = self.gripper_position(self.held_gripper) + rot @ track.held_offset
        track.yaw = wrap_angle(math.atan2(rot[1, 0], rot[0, 0]) + track.held_yaw)
        poured = self._tipped(track)
        for content in [t for t in self.tracks.values() if t.carried]:
            if poured:
                self._drop(content, f"poured out of {track.id}: perceived afresh where it lands")
            else:
                content.pos = track.pos + rot @ content.carried_offset

    def _follow_handle(self) -> None:
        """A drawer whose handle the gripper holds moves with the gripper, and what lies in it
        moves along.

        While the handle is held the drawer and its contents are pinned: a view of the moving
        drawer must not grow its box across the travel, and nothing may drop them. The travel
        is remembered for :meth:`shift`, which the skill calls with the whole distance once the
        handle is let go."""
        if not self.gripper.get("holdingHandle"):
            self._release_handle()
            return
        grip = np.asarray(self.gripper_position(self.held_gripper), float)
        hold = self._handle
        if hold is None:
            drawer = self._drawer_at(grip)
            if drawer is None:
                return
            self._handle_travel.pop(drawer.id, None)
            self._shut_follow.pop(drawer.id, None)
            hold = self._handle = _HandleHold(drawer.id, grip.copy(), np.zeros(3), self.drawer_opening(drawer), set(),
                                              None if drawer.front is None else drawer.front.copy())
        drawer = self.tracks.get(hold.drawer_id)
        if drawer is None:
            self._release_handle()
            return
        step = np.array([grip[0] - hold.grip[0], grip[1] - hold.grip[1], 0.0])
        hold.grip = grip.copy()
        hold.travel = hold.travel + step
        for tid in [drawer.id, *self._drawer_contents(drawer)]:
            track = self.tracks[tid]
            if float(np.linalg.norm(step[:2])) > 1e-4:
                self._translate_track(track, step)
            if not track.pinned:
                track.pinned = True
                hold.pinned.add(tid)

    def _release_handle(self) -> None:
        """End the handle hold: unpin what it pinned and keep its travel for :meth:`shift`."""
        hold = self._handle
        if hold is None:
            return
        for tid in hold.pinned:
            track = self.tracks.get(tid)
            if track is not None:
                track.pinned = False
        self._handle_travel[hold.drawer_id] = (hold.travel, hold.opening, hold.front)
        self._handle = None

    def _note_front(self, track: Track, points: np.ndarray) -> None:
        """Record where a fused view of drawer ``track`` puts its front panel.

        The panel is the view's room-ward edge: the 95th percentile of the points along the
        drawer's outward axis, at the view's lateral centre, so a handle or a stray carcass
        point does not set it.

        :param track: the drawer the view was fused into.
        :param points: the view's map-frame points (N×3).
        """
        if len(points) < 20:
            return
        box = self.front(track)
        outward = np.array([math.cos(box.yaw), math.sin(box.yaw)])
        lateral = np.array([-outward[1], outward[0]])
        xy = np.asarray(points[:, :2], float)
        along = float(np.percentile(xy @ outward, 95))
        across = float(np.median(xy @ lateral))
        track.front = outward * along + lateral * across
        track.front_seen = self.stamp

    def _follow_shut_front(self) -> None:
        """Move the contents of a drawer the robot pushed shut by the travel its front panel shows.

        The push is measured at the base; the drawer slides on to its stop after the gripper
        lets go, and the contents, hidden inside, cannot be seen to have gone with it. The panel
        seen again after the push gives the real travel; the contents take the difference. A
        difference under a centimetre is view noise, and one beyond :data:`SHUT_FOLLOW_MAX_M`
        is not this drawer's panel."""
        for drawer_id, follow in list(self._shut_follow.items()):
            drawer = self.tracks.get(drawer_id)
            if drawer is None:
                del self._shut_follow[drawer_id]
                continue
            if drawer.pinned or drawer.held or drawer.front is None or drawer.front_seen <= follow.since:
                continue
            observed = float((drawer.front - follow.front_at_grasp) @ follow.push)
            residual = observed - follow.applied
            if abs(residual) < 0.01 or abs(residual) > SHUT_FOLLOW_MAX_M:
                continue
            step = np.array([follow.push[0] * residual, follow.push[1] * residual, 0.0])
            for tid in follow.contents:
                track = self.tracks.get(tid)
                if track is not None and not track.held:
                    self._translate_track(track, step)
            follow.applied = observed
            self.version += 1

    def _in_held_drawer(self, centre: np.ndarray, footprint: float) -> bool:
        """Whether a small region lies in the drawer the gripper is sliding, over its travel.

        A frame is rendered before the gripper state it arrives with, so while the handle is
        held the drawer and its contents appear behind where the hold has put them. A new track
        from such a region is a second copy of a content; the pinned tracks are the contents.

        :param centre: the region's box centre, metres.
        :param footprint: the region's plan extent, metres.
        """
        hold = self._handle
        if hold is None or footprint >= 0.3:
            return False
        drawer = self.tracks.get(hold.drawer_id)
        if drawer is None:
            return False
        if not (drawer.bottom() - 0.1 <= float(centre[2]) <= drawer.top() + 0.3):
            return False
        travel = float(np.linalg.norm(hold.travel[:2]))
        return _inside(drawer, centre, 0.1 + travel)

    def _is_the_pulled_drawer(self, m: dict) -> bool:
        """Whether a region the segmenter named a drawer is a drawer the robot pulled out.

        A pinned drawer takes no views, so a mask of it and of the furniture face under it finds
        no track. Born as a drawer it would be seated shut in the carcass, and the contents of
        the open drawer would be pulled back to that seat.

        :param m: a measurement of :meth:`update_rgbd`.
        :returns: true when the region is furniture-sized and its centre is in such a drawer's box.
        """
        if m["footprint"] < SMALL_M or "drawer" not in (m["seg"].semantics or {}):
            return False
        centre = m["centre"]
        bottom, top = float(centre[2] - m["size"][2] / 2), float(centre[2] + m["size"][2] / 2)
        for drawer in self.tracks.values():
            if drawer.category != "drawer" or drawer.id not in self._opened_drawers:
                continue
            if not (drawer.pinned or self.drawer_opening(drawer) >= 0.08):
                continue
            if top >= drawer.bottom() and bottom <= drawer.top() and _inside(drawer, centre, 0.1):
                return True
        return False

    def _drop_printed_fragments(self) -> None:
        """A planar image detail is part of a measured, semantically named sheet.

        Use uncompleted surfaces: the wheel prior itself invents 16 mm thickness
        on a printed wheel. An actual raised wheel fails the coplanarity check.
        """
        sheets = [t for t in self.tracks.values()
                  if not (t.held or t.carried or t.pinned) and t.semantic_views
                  and t.category in ("manual", "picture")
                  and _printed_sheet(t.span, {k: v / t.semantic_views for k, v in t.semantic_counts.items()})]
        for track in list(self.tracks.values()):
            if (track.held or track.carried or track.pinned or track.span is None
                    or track.category not in ASSEMBLY_PARTS | {"scissors", "puzzle_piece"}):
                continue
            for sheet in sheets:
                if _printed_fragment(sheet.span, track.span):
                    self._drop(track, "coplanar printed detail inside a named sheet", into=sheet)
                    break

    def _assembly_composite(self, points: np.ndarray, exclude: str | None = None) -> bool:
        """Several separately observed screws explain one larger mask's surface.

        A box spanning screws is insufficient: most measured points must lie on
        their measured boxes. A solid wing or an elevated object stays independent.
        """
        if len(points) < 20:
            return False
        extent = np.ptp(points, axis=0)
        if not (0.12 <= max(extent[:2]) <= 0.4 and extent[2] <= 0.08):
            return False
        covered = np.zeros(len(points), dtype=bool)
        parts = []
        for track in self.tracks.values():
            if (track.id == exclude or track.held or track.carried or track.pinned
                    or track.category != "toy_screw" or track.observations < 3 or track.span is None):
                continue
            lo, hi = track.span.min(0), track.span.max(0)
            if max((hi - lo)[:2]) > 0.08 or (hi - lo)[2] > 0.05:
                continue
            inside = np.all((points >= lo - 0.012) & (points <= hi + 0.012), axis=1)
            if int(inside.sum()) >= 3:
                parts.append(track)
                covered |= inside
        return (float(covered.mean()) >= 0.8
                and any(_small_observations_apart(a.span, b.span)
                        for i, a in enumerate(parts) for b in parts[i + 1:]))

    def _drop_assembly_composites(self) -> None:
        for track in list(self.tracks.values()):
            if (not (track.held or track.carried or track.pinned) and track.voxels is not None
                    and track.category in ASSEMBLY_PARTS | {"unknown"}
                    and self._assembly_composite(_points(track.voxels), exclude=track.id)):
                self._drop(track, "composite surface already explained by separate screws")

    def _over_several_objects(self, m: dict) -> bool:
        """Whether a region named as tableware is one mask over pieces that stand apart.

        The segmenter sometimes returns a bowl and the glass beside it as one bowl. The box of
        that region holds the centres of two pieces that were each seen on their own. Nested
        bowls share a centre, and a view of the outer one is not such a mask.

        :param m: a measurement of :meth:`update_rgbd`.
        :returns: true when the box holds the centres of two pieces, one outside the other's plan radius.
        """
        # The shape of two bowls in a row is a drawer front's; the segmenter's name is what the track gets.
        semantics = getattr(m.get("seg"), "semantics", None) or {}
        names = {max(m["probs"], key=m["probs"].get) if m["probs"] else "",
                 max(semantics, key=semantics.get) if semantics else ""}
        if not names & set(TABLEWARE):
            return False
        centre, size = m["centre"], m["size"]
        c, s = math.cos(m["yaw"]), math.sin(m["yaw"])
        inside = []
        for track in self.tracks.values():
            if track.held or track.category not in TABLEWARE or track.observations < NAMED_VIEWS:
                continue
            if abs(float(track.pos[2] - centre[2])) > float(size[2]) / 2 + 0.03:
                continue
            dx, dy = float(track.pos[0] - centre[0]), float(track.pos[1] - centre[1])
            if abs(c * dx + s * dy) <= float(size[0]) / 2 and abs(-s * dx + c * dy) <= float(size[1]) / 2:
                inside.append(track)
        return any(float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > 0.5 * max(a.footprint, b.footprint)
                   for i, a in enumerate(inside) for b in inside[i + 1:])

    def _drawer_at(self, grip: np.ndarray) -> Track | None:
        """The drawer whose front the gripper is at: its box, widened by the handle's projection,
        holds the pinch point at the drawer's height. None when no drawer is there."""
        best: tuple[float, Track] | None = None
        for track in self.tracks.values():
            if track.category != "drawer" or track.held:
                continue
            if not (track.bottom() - 0.15 <= float(grip[2]) <= track.top() + 0.15):
                continue
            if not _inside(track, grip, 0.12):
                continue
            dist = float(np.linalg.norm(track.pos[:2] - grip[:2]))
            if best is None or dist < best[0]:
                best = (dist, track)
        return None if best is None else best[1]

    def _drawer_contents(self, drawer: Track) -> list[str]:
        """Tracks a pull or push of ``drawer`` carries: what rests on its floor, including a
        content seen just past its box."""
        carried = [t.id for t in self.tracks.values()
                   if t.id != drawer.id and not t.held and self.support_of(t) is drawer]
        for tid in self._on_drawer_floor(drawer):
            if tid not in carried:
                carried.append(tid)
        return carried

    def held_pose(self, joints: dict | None = None, odometry: dict | None = None) -> dict[str, tuple]:
        """``{track id: (position, yaw)}`` of the load and of what is carried in it, for the given
        joint state and base pose (the formula of :meth:`_follow_held`; the tracks are not changed).
        Empty when nothing is held or the load has no grasp offset."""
        if not self.held_id or self.held_id not in self.tracks:
            return {}
        track = self.tracks[self.held_id]
        if track.held_offset is None:
            return {}
        rot = self.gripper_rotation(self.held_gripper, joints, odometry)
        pos = self.gripper_position(self.held_gripper, joints, odometry) + rot @ track.held_offset
        out = {track.id: (pos, wrap_angle(math.atan2(rot[1, 0], rot[0, 0]) + track.held_yaw))}
        # Tipped or not by the same body: an upright belief with a live wrist rolled over has
        # poured, and its contents are no longer carried along (to_scene drops their relation).
        if not self._tipped(track, rot):
            for content in self.tracks.values():
                if content.carried and content.id != track.id:
                    out[content.id] = (pos + rot @ content.carried_offset, None)
        return out

    def held_position(self, joints: dict | None = None, odometry: dict | None = None):
        """Where the load is, for the given joint state and base pose: in the gripper that took it."""
        if not self.held_id or self.held_id not in self.tracks:
            return None
        track = self.tracks[self.held_id]
        if track.held_offset is None:
            return np.asarray(track.pos, dtype=float)
        return (self.gripper_position(self.held_gripper, joints, odometry)
                + self.gripper_rotation(self.held_gripper, joints, odometry) @ track.held_offset)

    # ------------------------------------------------------------------ output
    def support_of(self, track: Track) -> Track | None:
        """What holds a track: the held container for its contents, else the innermost receptacle
        (drawer, larger bowl, a box its contents can be poured from) whose floor it rests on, else
        the highest support surface whose top it rests on."""
        if track.carried and self.held_id in self.tracks:
            return self.tracks[self.held_id]
        inside, best = None, None
        for other in self.tracks.values():
            if other.id == track.id or other.held:
                continue
            affordances = other.affordances
            z_tol = max(0.06, 2.5 * _sigma_z(track, other))
            local = _to_local_xy(other, track.pos)
            holds_contents = "receptacle" in affordances or "pourable" in affordances
            if holds_contents and track.footprint < other.footprint:
                within = abs(local[0]) <= other.size[0] / 2 and abs(local[1]) <= other.size[1] / 2
                # Nested bowls: the inner bottom lies from the outer floor up through its rim. A
                # rim-only view centres the inner bowl high and thin, so the old 0.5×height cap
                # missed nesting and fell through to the furniture under both.
                floor_delta = track.bottom() - other.bottom()
                max_above_floor = float(other.size[2]) + max(z_tol, 0.06)
                if within and -0.03 <= floor_delta <= max_above_floor:
                    if inside is None or other.footprint < inside.footprint \
                            or (other.footprint == inside.footprint and other.bottom() > inside.bottom()):
                        inside = other
                    continue
            # A resting object's base is often hidden from view, so its estimate may float a little above
            # the surface; it never sinks far into it.
            gap = track.bottom() - other.top()
            if "support-surface" not in affordances or not (-z_tol <= gap <= max(z_tol, 0.12)):
                continue
            margin = max(0.02, 2.0 * track.sigma)
            if abs(local[0]) <= other.size[0] / 2 + margin and abs(local[1]) <= other.size[1] / 2 + margin:
                if best is None or other.top() > best.top():
                    best = other
        return inside or best

    def _in_closed_drawer(self, track: Track) -> bool:
        """Whether a small track lies in a drawer that is shut.

        The drawer front hides the floor, so a shut drawer's box is often taller than the drawer
        and its bottom a few centimetres under the floor the contents stand on; a content whose
        plan lies in the box and whose base is between :data:`DRAWER_FLOOR_TOL` under that
        bottom and the top rim is in the drawer.
        """
        holder = self.support_of(track)
        if holder is not None and holder.category == "drawer":
            return not self._drawer_open(holder)
        for drawer in self.tracks.values():
            if drawer.category != "drawer" or drawer.held or drawer.id == track.id or self._drawer_open(drawer):
                continue
            if track.footprint < drawer.footprint and _inside(drawer, track.pos, 0.0) \
                    and drawer.bottom() - DRAWER_FLOOR_TOL <= track.bottom() <= drawer.top() - 0.05:
                return True
        return False

    def _drawer_open(self, drawer: Track) -> bool:
        """Whether a drawer is pulled out: what the robot last did to it when it moved it, else
        how far its box sticks out of the furniture. A box left swollen by the close measures as
        open while the drawer is shut."""
        if drawer.id in self._pulled_open:
            return True
        if drawer.id in self._pushed_shut:
            return False
        return self.drawer_opening(drawer) >= 0.08

    def _in_open_drawer(self, track: Track) -> bool:
        """A small track on the floor of a drawer that is currently pulled out.

        A side step looks through the place the track occupied a moment ago. That view is not
        evidence the object is gone; the drawer still holds it."""
        if track.footprint >= 0.3:
            return False
        for drawer in self.tracks.values():
            if drawer.category != "drawer" or drawer.id == track.id or not self._drawer_open(drawer):
                continue
            if track.id in self._on_drawer_floor(drawer):
                return True
        return False

    def drawer_opening(self, drawer: Track) -> float:
        """How far a drawer sticks out of the furniture holding it, along the drawer's front axis."""
        furniture = [_box2d(t) for t in self.tracks.values()
                     if t.id != drawer.id and "support-surface" in t.affordances]
        return protrusion(self.front(drawer), furniture)

    def front(self, track: Track) -> Box2D:
        """The track's footprint with +x pointing out of its front (towards the room centre)."""
        return front_box(_box2d(track), room_centre([_box2d(t) for t in self.tracks.values()]))

    def relocate(self, track_id: str, pos: np.ndarray) -> bool:
        """Re-place a track from a fresh close look taken for an imminent action.

        An accumulated position can be biased by the partial views it was fused from; a text-
        prompted segmentation taken beside the object is not. The track keeps its identity,
        category and extent — only where it stands is restated, and confidently: this one view
        is the best evidence the robot has ever had of this position."""
        track = self.tracks.get(track_id)
        if track is None:
            return False
        track.pos = np.asarray(pos, float)
        track.var = np.array([0.015**2, 0.015**2, track.var[2]])
        self.version += 1
        return True

    def release_at(self, track_id: str, spot: list[float], body: dict | None = None,
                   target: dict | None = None) -> bool:
        """Record where a placed object was actually let go.

        While the gripper holds something the track follows the gripper, and the last frame the
        ego saw it holding can precede the release by a whole perception period — long enough
        for the arm to extend and the base to back away. Without this the object stays believed
        where it was still held, often half a metre from where it now stands, and the next pick
        of it aims at the pose the robot had back then.

        The same holds for its height: the last frame saw it carried at the carry height, a few
        tenths of a metre above where the arm then lowered it. With ``body`` (the sensor header
        at the release) the height is the hand's there; without it, it stays the last estimate.
        Left at the carry height, the object put down on a table is believed floating above it,
        and the next view of it there starts a second track."""
        track = self.tracks.get(track_id)
        if track is None:
            return False
        # Capture before clearing held_offset or accepting a new segmentation.
        # Without release encoders this remains unavailable, never a fake FK pose.
        if (track.held and body and body.get('joints') is not None
                and body.get('odometry') is not None):
            from steward.ego.fasten_release import release_frame
            frame = release_frame(track.held_interface,
                self.gripper_position(self.held_gripper, body['joints'], body['odometry']),
                self.gripper_rotation(self.held_gripper, body['joints'], body['odometry']),
                target, body.get('stamp', self.stamp))
            if frame:
                self.released_interfaces[track_id] = frame
        if self.rigid_carry and self.held_id == track_id:
            rotation = self.gripper_rotation(self.held_gripper,
                (body or {}).get('joints'), (body or {}).get('odometry'))
            pinch = self.gripper_position(self.held_gripper,
                (body or {}).get('joints'), (body or {}).get('odometry'))
            for eid, member in self.rigid_carry.items():
                other = self.tracks.get(eid)
                if other is not None:
                    other.pos = pinch+rotation @ member['offset']
                    rr = rotation @ member['rotation']
                    other.yaw = math.atan2(rr[1,0],rr[0,0])
                    other.carried_offset = None
                    other.interface_motion_at = float(self.stamp)
                    other.interface_last_geometry = None
                    other.interface_views.clear()
                    other.interface_best_near = None
                    other.voxels = other.span = None
                    other.settled = False
                    other.var = np.maximum(other.var, .02**2)
            self.rigid_carry.clear()
        capture = self.flip_capture
        if capture and capture['body'] == track_id:
            capture['releaseRotation'] = self.gripper_rotation(self.held_gripper,
                (body or {}).get('joints'), (body or {}).get('odometry')).copy()
            capture['releasedAt'] = float((body or {}).get('stamp', self.stamp))
        z = float(track.pos[2])
        if body and body.get("joints") is not None and body.get("odometry") is not None \
                and track.held and self.held_id == track_id and track.held_offset is not None:
            z = float(self.held_position(body["joints"], body["odometry"])[2])
        track.pos = np.array([float(spot[0]), float(spot[1]), z])
        track.placement = {"releasedAt": float((body or {}).get("stamp", self.stamp)),
                           "expectedXYM": [float(spot[0]), float(spot[1])], "observation": None}
        if target is not None:
            track.placement["orientationTarget"] = {**target, "sizeM": list(target["sizeM"])}
        self.assembly_memory.released(track_id, track.pos, target, track.placement["releasedAt"])
        self.stationary_assemblies.released(track_id, track.pos, target, track.placement["releasedAt"])
        track.var = np.maximum(track.var, 0.02**2)
        track.voxels = None
        track.interface_motion_at = float(self.stamp)
        track.interface_last_geometry = None
        track.interface_views.clear()
        track.interface_best_near = None
        track.interface_estimate_at = None
        track.settled = False  # it drops the last few millimetres when the grip opens
        track.held = False
        track.held_offset = None
        if self.held_id == track_id:
            self.held_id = None
        self.stationary_assemblies.promote_release(self._track_entity(track), target, track.placement['releasedAt'])
        self.version += 1
        return True

    def placement_evidence(self, track_id: str) -> dict:
        """Report a post-release sensor sample, never the position written by release_at.

        A single view's bounding centre is a measurement of visible surfaces, not proof of
        support, orientation, mating or fastening. Without a task-supplied acceptance criterion
        it cannot certify completion, even when its XY residual is zero.
        """
        track = self.tracks.get(track_id)
        record = track.placement if track is not None else None
        result = {"ok": None, "status": "unknown", "entity": track_id,
                  "reason": "no independent post-release observation; observe from a clear view before continuing"}
        if record is None:
            result["reason"] = "release track unavailable or identity changed; re-identify before continuing"
            return result
        if track.held or track.carried:
            result["reason"] = "object is being carried; previous placement evidence no longer applies"
            return result
        result.update({"releasedAt": record["releasedAt"], "expectedXYM": list(record["expectedXYM"])})
        if "orientationTarget" in record:
            result["orientation"] = {"marker": "[RSI_PLACE_YAW]", "status": "unknown", "directedDeltaDeg": None,
                                     "reason": "no independent post-release axis observation"}
        obs = record["observation"]
        if obs is None or obs["stamp"] <= record["releasedAt"]:
            return result
        centre = obs["centreM"]
        if not all(math.isfinite(v) for v in centre):
            result["reason"] = "post-release depth measurement is invalid; obtain another view"
            return result
        result.update({"observation": {**obs, "centreM": list(centre), "visibleSizeM": list(obs["visibleSizeM"])},
                       "deltaXYM": [centre[k] - record["expectedXYM"][k] for k in range(2)],
                       "reason": "visible-surface centre measured; verify identity, pose and required relation separately"})
        if obs["truncated"] or obs["invalidDepthFraction"] > 0:
            result["reason"] = "partial view or missing depth; residual is diagnostic only, obtain another view"
        else:
            result["status"] = "observed"
        if "orientationTarget" in record:
            self._placement_orientation(track, record, result)
        return result

    def _placement_orientation(self, track: Track, record: dict, result: dict) -> None:
        """Compare raw visible long axes modulo pi; never certify a directed pose from a box.

        Two consistent, uncut samples of comparable extents permit a *proposed* regrasp/re-place.
        Neither the filtered yaw nor the release target is a post-release measurement.
        """
        target, obs = record["orientationTarget"], record["observation"]
        orientation = result["orientation"]
        if self.stamp - obs["stamp"] > self.config.placement_observation_max_age_s:
            orientation["reason"] = "post-release axis observation is stale; obtain a fresh view"
            return
        if not target.get("planar", True):
            orientation["reason"] = "tilted placement requires full 3D feature verification"
            return
        expected_size = np.asarray(target["sizeM"], float)

        def axis(sample):
            if sample is None or sample["truncated"] or sample["invalidDepthFraction"] > 0 \
                    or self.stamp - sample["stamp"] > self.config.placement_observation_max_age_s:
                return None
            size = np.asarray(sample["visibleSizeM"], float)
            yaw = sample.get("boxYawRad", float("nan"))
            if not np.isfinite(size).all() or not math.isfinite(yaw) or not np.isfinite(expected_size).all():
                return None
            if min(size[:2]) >= SQUARE_SIDES * max(size[:2]) or min(expected_size[:2]) >= SQUARE_SIDES * max(expected_size[:2]):
                return None
            # Sort only XY: the estimator may keep the equivalent frame with the axes swapped.
            measured = np.r_[np.sort(size[:2]), size[2]]
            expected = np.r_[np.sort(expected_size[:2]), expected_size[2]]
            if np.any(measured <= 0) or np.any(expected <= 0):
                return None
            if np.any(np.maximum(measured / expected, expected / measured) > self.config.placement_extent_ratio):
                return None
            return yaw + (math.pi / 2 if size[1] > size[0] else 0)

        measured = axis(obs)
        if measured is None:
            orientation["reason"] = "axis ambiguous or depth/extent inconsistent; obtain a clear view of asymmetric features"
            return
        expected_axis = target["yawRad"] + (math.pi / 2 if expected_size[1] > expected_size[0] else 0)
        delta = math.remainder(measured - expected_axis, math.pi)
        orientation.update({"status": "axis_only", "observedAxisYawDeg": math.degrees(measured),
                            "expectedAxisYawDeg": math.degrees(expected_axis), "axisDeltaDeg": math.degrees(delta),
                            "periodDeg": 180, "reason": "outer axis only; a half-turn remains unobservable; "
                            "verify an asymmetric feature and the required relation"})
        previous = axis(record.get("previousObservation"))
        noise = math.radians(self.config.placement_axis_noise_deg)
        if previous is None or abs(math.remainder(previous - measured, math.pi)) > noise:
            orientation["reason"] += "; repeat the observation before a correction"
            return
        if abs(delta) <= noise:
            return
        support = self.tracks.get(target.get("support"))
        if support is None or support.held or support.last_seen <= record["releasedAt"] \
                or self.stamp - support.last_seen > self.config.placement_observation_max_age_s:
            orientation["reason"] += "; reobserve the support before replanning"
            return
        # Preserve the measured centre. The desired box-frame yaw is expressed using the latest
        # fitted frame's x/long-axis convention, not a category's body axes.
        centre = np.asarray(obs["centreM"], float)
        dx, dy = centre[:2] - support.pos[:2]
        c, sn = math.cos(support.yaw), math.sin(support.yaw)
        offset = [float(c * dx + sn * dy), float(-sn * dx + c * dy)]
        long_in_frame = math.pi / 2 if track.size[1] > track.size[0] else 0
        desired_yaw = wrap_angle(measured - delta - long_in_frame)
        result["correction"] = {
            "marker": "[RSI_PLACE_YAW]", "rotateByDeg": -math.degrees(delta),
            "requires": "confirm identity and support in a fresh camera view; re-predict each step; "
                        "this corrects the outer axis only, not front/back, seating or fastening",
            "steps": [{"skill": "pick", "target": track.id},
                      {"skill": "place", "on": support.id, "offset": offset,
                       "yaw": desired_yaw, "alignHoles": False, "clearance": 0.0}],
        }

    def _on_drawer_floor(self, drawer: Track) -> list[str]:
        """Small tracks on a drawer's floor, including ones just in front of its box."""
        box = self.front(drawer)
        found = []
        for other in self.tracks.values():
            if other.id == drawer.id or other.held or other.footprint >= 0.3:
                continue
            if other.category in ("cabinet", "sideboard", "counter", "table", "wall", "drawer"):
                continue
            # The drawer box is often taller than the drawer, so bottoms do not line up.
            # A content sits inside that box, not under it.
            if other.bottom() < drawer.bottom() - DRAWER_FLOOR_TOL or float(other.pos[2]) > drawer.top() + 0.02:
                continue
            dx, dy = float(other.pos[0] - box.cx), float(other.pos[1] - box.cy)
            c, s = math.cos(-box.yaw), math.sin(-box.yaw)
            lx, ly = c * dx - s * dy, s * dx + c * dy
            if abs(ly) > box.hy + 0.08:
                continue
            # Either side of the box: the open side is not always the one front() picked.
            if abs(lx) <= box.hx + 0.45:
                found.append(other.id)
        return found

    def _translate_track(self, track: Track, delta: np.ndarray) -> None:
        """Move ``track`` and everything measured of it by ``delta``: the object itself moved.

        The single-view boxes move too. A small object's box is their median
        (:meth:`_rebuild_geometry`), and left behind they put the next view of a content the
        close carried back where the open drawer held it."""
        step = np.asarray(delta, float)
        track.pos = track.pos + step
        if track.voxels is not None:
            track.voxels = _voxelize(_points(track.voxels) + step)
        if track.span is not None:
            track.span = track.span + step
        rows = [(row[0] + step, *row[1:]) for row in track.recent]
        track.recent.clear()
        track.recent.extend(rows)

    def _trim_shut_drawer(self, track: Track, step: np.ndarray) -> None:
        """Drop the open-travel points a shut drawer still carries.

        The voxels moved with the drawer, so the cloud keeps the length it had while open.
        The face the close moved away from is the front. Points further than the drawer
        prior behind that face are the cabinet the open mask took in.

        :param step: the translation just applied, metres. Its horizontal direction
            points into the furniture.
        """
        proto = prototype("drawer")
        if proto is None or track.voxels is None:
            return
        direction = np.array(step[:2], float)  # a copy: the caller reads ``step`` afterwards
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return
        direction /= norm
        cloud = _points(track.voxels)
        proj = cloud[:, 0] * direction[0] + cloud[:, 1] * direction[1]
        front = float(np.min(proj))
        limit = proto[0] + 0.04
        if float(np.max(proj) - front) <= limit:
            return
        kept = cloud[proj <= front + limit]
        if len(kept) < 20:
            return
        track.voxels = _voxelize(kept)
        track.span = None
        centre, size, yaw = box_of(kept, trim=0.01)
        track.pos, track.size, track.yaw = centre, size, float(yaw)

    def _drop_glimpses_in(self, drawer: Track) -> None:
        """Drop what was only glimpsed beside ``drawer`` while it stood open, as it is pushed shut.

        A shut drawer hides its contents, so nothing in it is confirmed or seen through again
        (:meth:`_in_closed_drawer`): the push keeps for good whatever is there. A content stood
        in view for the whole look into the drawer. A box the segmenter never named and that
        fewer than :data:`CONTENT_VIEWS` frames showed is a patch of the drawer's panels or of
        the furniture's front. So is one that a single moment showed, when the regions of that
        moment folded into it are fewer than :data:`ESTABLISHED_VIEWS`. A name that is another
        content's (:func:`_named_only_as_another_content`) does not name the box: a slab of the
        back panel that colour made a bear, and that a mask of the bottle in front of it was
        folded into, is still that panel."""
        for tid in self._near_drawer(drawer):
            track = self.tracks[tid]
            one_moment = track.observations < ESTABLISHED_VIEWS and track.last_seen <= track.first_seen + 1e-6
            unnamed = track.semantic_views == 0 or _named_only_as_another_content(track)
            if unnamed and (track.observations < CONTENT_VIEWS or one_moment):
                self._drop(track, "glimpsed beside an open drawer, not seen again before it shut")

    def _drop_unnamed_parts(self) -> None:
        """Drop a small box no mask ever named whose centre lies inside a small object the
        segmenter named.

        Two solid objects are not in one place. A depth fill over part of the airplane gets the
        name the shape prior gives a box of that size: a bear when the drawer floor makes it
        tall, a pistol when it is flat. The object is the track the segmenter named, at least
        :data:`CONTENT_VIEWS` times and with the name it has now. A bowl, a plate or a cup holds
        other things and a support carries them, so those do not count as the named object, and
        a box that begins at the object's top stands on it."""
        named = [t for t in self.tracks.values()
                 if t.semantic_views >= CONTENT_VIEWS and t.footprint < 0.45 and not t.held and not t.carried
                 and t.category not in ("wall", "drawer", "unknown")
                 and not {"receptacle", "support-surface"} & set(t.affordances)
                 and t.semantic_counts.get(t.category, 0.0) > 0.0]
        if not named:
            return
        for track in list(self.tracks.values()):
            if (track.semantic_views > 0 or track.held or track.carried or track.footprint >= 0.45
                    or track.category in ("wall", "drawer")):
                continue
            for other in named:
                if other.id == track.id or not _inside(other, track.pos, 0.0):
                    continue
                if track.bottom() >= other.top() - 0.02 or track.top() <= other.bottom() + 0.02:
                    continue
                self._drop(track, "an unnamed part of a named object", into=other)
                break

    def _moved_by_robot(self, drawer: Track) -> bool:
        """Whether the robot has pulled ``drawer`` out or pushed it back (:meth:`shift`), or holds
        its handle now. Only such a drawer has shown what it holds: a box that merely measures
        as open, on a drawer nobody has touched, is the front swollen by a grazing view."""
        return (drawer.id in self._pulled_open or drawer.id in self._pushed_shut
                or (self._handle is not None and self._handle.drawer_id == drawer.id))

    def _beside_drawer(self, centre: np.ndarray) -> bool:
        """Whether a small box is at the height of a drawer the robot has moved
        (:meth:`_moved_by_robot`) and within 0.70 m of it: where the contents of a drawer are
        seen. Beside a drawer that was never opened there is only its handle and its gap."""
        for other in self.tracks.values():
            if other.category != "drawer" or not self._moved_by_robot(other):
                continue
            if not (0.45 <= float(centre[2]) <= 0.90):
                continue
            if float(np.linalg.norm(other.pos[:2] - centre[:2])) <= 0.70:
                return True
        return False

    def _stands_on_counter(self, centre: np.ndarray, bottom: float) -> bool:
        """Whether a small box rests on a counter, even when that counter's heading is wrong.

        The counter's long side is known more steadily than which way it faces. A glass whose
        base is at the counter top and within that length of the centre is the cup on it. The
        wine glass stands on the table, a hand's height lower."""
        for other in self.tracks.values():
            if other.category != "counter":
                continue
            if abs(bottom - other.top()) > 0.08:
                continue
            reach = 0.5 * float(max(other.size[0], other.size[1])) + 0.05
            if float(np.linalg.norm(other.pos[:2] - centre[:2])) <= reach:
                return True
        return False

    def _near_drawer(self, drawer: Track) -> list[str]:
        """Small tracks within 0.70 m of a drawer, at the height of its contents.

        The estimated drawer box often does not contain them yet. They are still what the pull
        carried, and a side step must not delete them."""
        found = []
        for other in self.tracks.values():
            if other.id == drawer.id or other.held or other.footprint >= 0.3:
                continue
            if other.category in ("cabinet", "sideboard", "counter", "table", "wall", "drawer"):
                continue
            if not (0.45 <= float(other.pos[2]) <= 0.90):
                continue
            if float(np.linalg.norm(other.pos[:2] - drawer.pos[:2])) <= 0.70:
                found.append(other.id)
        return found

    def _drawer_now(self, drawer_id: str, before: dict[str, list[float]]) -> Track | None:
        """The drawer track after a merge may have changed id. Follow that, or the nearest drawer."""
        seen: set[str] = set()
        current: str | None = drawer_id
        while current and current not in seen:
            seen.add(current)
            track = self.tracks.get(current)
            if track is not None and track.category == "drawer":
                return track
            nxt = None
            for record in reversed(self.dropped):
                if record.get("id") == current and record.get("into"):
                    nxt = str(record["into"])
                    break
            current = nxt
        if not before:
            return None
        centre = np.mean([np.asarray(v, float)[:2] for v in before.values()], axis=0)
        best: tuple[float, Track] | None = None
        for track in self.tracks.values():
            if track.category != "drawer":
                continue
            dist = float(np.linalg.norm(track.pos[:2] - centre))
            if best is None or dist < best[0]:
                best = (dist, track)
        if best is not None and best[0] < 1.0:
            return best[1]
        return None

    def content_positions(self, drawer_id: str) -> dict[str, list[float]]:
        """Where each small track near ``drawer_id`` is, before the drawer is pushed back."""
        drawer = self.tracks.get(drawer_id)
        if drawer is None:
            return {}
        return {tid: self.tracks[tid].pos.tolist() for tid in self._near_drawer(drawer)}

    def resettle(self, drawer_id: str, delta: np.ndarray, before: dict[str, list[float]]) -> list[str]:
        """Move contents that are still where the open drawer left them.

        ``before`` is :meth:`content_positions` from just before the close. A track already carried
        by :meth:`shift` has left that place and is left alone. A track that appeared only in a
        late frame of the open drawer is still out in front, and is brought back by ``delta``."""
        drawer = self._drawer_now(drawer_id, before)
        if drawer is None:
            return []
        direction = np.asarray(delta, float)
        outward = -direction[:2]
        norm = float(np.linalg.norm(outward))
        axis = outward / norm if norm > 1e-6 else None
        moved: list[str] = []
        for tid, old in before.items():
            track = self.tracks.get(tid)
            if track is None or track.held or track.pinned:
                continue
            if float(np.linalg.norm(track.pos - np.asarray(old, float))) > 0.04:
                continue
            # Already inside the closed drawer. Adding the close again pushes it out the back.
            # A bottle in the front half is still in the box; the centre band alone is not.
            if _inside(drawer, track.pos, 0.02):
                continue
            if axis is not None:
                along = float((track.pos[:2] - drawer.pos[:2]) @ axis)
                if along < 0.10:
                    continue
            self._translate_track(track, direction)
            moved.append(tid)
        for tid in self._near_drawer(drawer):
            if tid in before or tid in moved:
                continue
            track = self.tracks[tid]
            if track.pinned or _inside(drawer, track.pos, 0.02):
                continue
            if axis is not None:
                along = float((track.pos[:2] - drawer.pos[:2]) @ axis)
                if along < 0.10:
                    continue
            self._translate_track(track, direction)
            moved.append(tid)
        # The close can set a second copy down beside the one already in the drawer,
        # after the last camera update has already merged.
        self._merge_same_spot()
        self._drop_thin_bears()
        self._drop_pistols_off_the_drawer()
        self._rename_wide_perfumes()
        self._name_pink_pills()
        self._rename_wide_perfumes()
        self._drop_perfume_on_the_pill()
        self._drop_bottles_beside_the_drawer()
        if moved:
            self.version += 1
        return moved

    def shift(self, track_id: str, delta: np.ndarray) -> list[str]:
        """Move a track the robot itself moved (a drawer it pulled) together with what it holds."""
        self.stationary_assemblies.invalidate(track_id, self.tracks.get(track_id), self.stamp)
        track = self.tracks.get(track_id)
        if track is None:
            return []
        if self._handle is not None and self._handle.drawer_id == track_id:
            self._release_handle()  # the skill let go before a frame showed the open gripper
        carried = self._drawer_contents(track)
        whole = np.asarray(delta, float)
        opening_before = self.drawer_opening(track) if track.category == "drawer" else None
        # A handle hold has carried the drawer frame by frame; only the rest of the distance is left.
        hold = self._handle_travel.pop(track_id, None)
        self._shut_follow.pop(track_id, None)
        step = whole
        front_at_grasp = None
        if hold is not None:
            applied, opening_at_grasp, front_at_grasp = hold
            step = whole - applied
            if opening_before is not None:
                opening_before = opening_at_grasp
        norm = float(np.linalg.norm(whole[:2]))
        step_xy = whole[:2] / norm if norm > 1e-6 else None
        # Seating has already put this drawer on the closed carcass. Contents already in that
        # box are where the close leaves them; adding the close again pushes them out the back.
        already_shut = opening_before is not None and opening_before < 0.12 and norm >= 0.05
        seat = self._closed_centre(track) if track.category == "drawer" else None
        for tid in [track_id, *carried]:
            other = self.tracks[tid]
            if tid != track_id and seat is not None and hold is None:
                centre, depth_axis = seat
                # The bear sits to one side of the drawer. Only the depth says whether the
                # close has already been applied. A handle hold carried the contents frame by
                # frame, so they are mid-travel and take the rest of the distance.
                if abs(float((other.pos[:2] - centre) @ depth_axis)) <= 0.10:
                    continue
            if already_shut and tid != track_id and step_xy is not None:
                along = float((other.pos[:2] - track.pos[:2]) @ step_xy)
                if along > -0.05:
                    continue
            if float(np.linalg.norm(step[:2])) > 1e-4:
                self._translate_track(other, step)
        if opening_before is not None and self.drawer_opening(track) < 0.08 and opening_before > self.drawer_opening(track):
            self._trim_shut_drawer(track, whole)
        if track.category == "drawer" and norm >= 0.08:
            self._opened_drawers.add(track_id)
            # The pull direction, not the measured protrusion, says whether the drawer is now
            # out: a swollen box beside the carcass measures as shut while it stands open.
            box = self.front(track)
            outward = float(whole[0] * math.cos(box.yaw) + whole[1] * math.sin(box.yaw))
            if outward > 0:
                self._pulled_open.add(track_id)
                self._pushed_shut.discard(track_id)
            else:
                self._pulled_open.discard(track_id)
                self._pushed_shut.add(track_id)
                self._drop_glimpses_in(track)
                if front_at_grasp is not None and carried:
                    self._shut_follow[track_id] = _ShutFollow(front_at_grasp, whole[:2] / norm, norm, list(carried),
                                                              self.stamp)
        if track.category == "drawer" and norm >= 0.05:
            self._drop_drawers_left_behind(track)
        self.version += 1
        return carried

    def _drop_drawers_left_behind(self, moved: Track) -> None:
        """Drop drawer tracks that stayed put while ``moved`` was pulled.

        A second box completed from the furniture's top band does not slide. The drawer is the
        track that translated with the pull, not the one nearest the furniture's centre.

        :param moved: the drawer track the robot just translated.
        """
        for host in list(self.tracks.values()):
            if host.category not in ("cabinet", "sideboard"):
                continue
            if float(np.linalg.norm(moved.pos[:2] - host.pos[:2])) > 1.2:
                continue
            for other in list(self.tracks.values()):
                if other.id == moved.id or other.held or other.category != "drawer":
                    continue
                if other.id not in self.tracks:
                    continue
                dist = float(np.linalg.norm(other.pos[:2] - host.pos[:2]))
                if dist < 1.0 and host.bottom() - 0.05 <= float(other.pos[2]) <= host.top() + 0.05:
                    self._drop(other, "drawer that did not move with the pull")

    def to_scene(self, self_frame: dict | None = None) -> SceneState:
        """The Ego State as a scene.

        ``self_frame`` is the robot's *current* sensor frame, when the caller has one. Proprioception
        is continuous — the robot knows its joint angles without looking — so a console can show the
        arm where it is now while the object beliefs stay as old as the last look. Object tracks are
        never refreshed by it. Odometry is proprioception too: a frame that carries it places the
        robot (and the load in its hand) where it stands now, not where the last look -- with a
        slow segmenter, tens of seconds and a whole drive ago -- left it.
        """
        self._assembly_views = self.assembly_memory.views(self.tracks, self.stamp)
        entities = [self._self_entity(self_frame)]
        # A held object is not where the last look saw it: it is in the hand. That is a kinematic
        # fact, so it is read here rather than mutating the track from a rendering thread.
        # The load and what lies in it, as the body now holds them: position, heading and the
        # contents all from one body snapshot (the same transform as _follow_held, without
        # writing it into the tracks). A heading left at the last look made a load turned with the
        # base by 90 degrees read as unturned, and the place wrist angle was planned from it.
        live = self.held_pose(*((None, None) if self_frame is None else
                                (self_frame.get("joints"), self_frame.get("odometry"))))
        relations: list[Relation] = []
        # A snapshot of the track objects: the Ego-State is also read from a view thread while the
        # perception thread adds and drops tracks, and a dict that changes size mid-iteration raises.
        for track in list(self.tracks.values()):
            if float(min(track.size)) <= 1e-4:
                continue  # a flat degenerate box is not an object the observer can score
            at, yaw = live.get(track.id, (None, None)) if live else (None, None)
            entities.append(self._track_entity(track, at=at, yaw=yaw))
            # Carried in a load the live body has tipped: it has slid out, so it is not reported
            # as in the container (the next look perceives it where it landed).
            poured_live = bool(live) and track.carried and track.id not in live
            if poured_live:
                continue
            if track.held:
                relations.append(Relation(type="attached", a={"entity": track.id}, b={"entity": SELF_ID, "link": "gripper"},
                                          mode="rigid", derived=False, confidence=1.0))
            elif prior(track.category)["root"].get("fixed", 0.0) < 0.9:
                support = self.support_of(track)
                if support is None:
                    for drawer in self.tracks.values():
                        # The floor only. A mug on the cabinet top sits far above this.
                        if (drawer.category == "drawer" and track.id in self._on_drawer_floor(drawer)
                                and track.bottom() <= drawer.bottom() + 0.12):
                            support = drawer
                            break
                if support is not None:
                    relations.append(Relation(type="supported-by", a={"entity": track.id}, b={"entity": support.id},
                                              derived=True, confidence=round(min(track.existence, support.existence), 3)))
        self.stationary_assemblies.overlay(entities, relations, self.tracks, self.stamp)
        for ent in entities:
            if self.held_id and ent.id in self.rigid_carry:
                from steward.math3d import quat_from_matrix
                member = self.rigid_carry[ent.id]
                rotation = self.gripper_rotation(self.held_gripper,
                    (self_frame or {}).get('joints'), (self_frame or {}).get('odometry'))
                pinch = self.gripper_position(self.held_gripper,
                    (self_frame or {}).get('joints'), (self_frame or {}).get('odometry'))
                ent.pose.value = {'p':(pinch+rotation @ member['offset']).tolist(),
                                  'q':quat_from_matrix(rotation @ member['rotation'])}
                ent.pose.src = 'inferred'
                ent.state_vars['rigidFollowEvidence'] = Est(value={
                    'source':'observed_follow_then_encoder_prediction', 'anchor':self.held_id,
                    'observedAt':member['observedAt']}, src='inferred')
            track = self.tracks.get(ent.id)
            if track is not None:
                feature = track.tool_feature
                if track.held and track.held_tool_feature:
                    from steward.ego.fasten_geometry import transform_feature
                    feature = transform_feature(track.held_tool_feature,
                        self.gripper_rotation(self.held_gripper, (self_frame or {}).get('joints'), (self_frame or {}).get('odometry')),
                        self.gripper_position(self.held_gripper, (self_frame or {}).get('joints'), (self_frame or {}).get('odometry')))
                    feature['source'] = 'grasp_relative_plus_encoders'
                    feature['grip'] = self.gripper_position(self.held_gripper,
                        (self_frame or {}).get('joints'), (self_frame or {}).get('odometry')).tolist()
                if feature and track.held:
                    centre = np.asarray(ent.pose.value['p'])
                    ends = np.array([feature['tip'], feature['head']])
                    ent.rigid.links[0].geometry['collision'][0]['size'] = (2*np.max(abs(ends-centre),axis=0)+2*feature['radiusM']).tolist()
                if feature:
                    ent.state_vars['toolFeature'] = Est(value=feature, src='observed' if
                        not track.held and feature.get('source') == 'rgbd_endpoint_width' else 'inferred')
                if track.manipulation_view:
                    ent.state_vars['contactView'] = Est(value={**track.manipulation_view,
                        'points': track.manipulation_view['points'].tolist()}, src='observed')
            if ent.id in self.manipulation_envelopes and track is not None and track.held:
                ent.rigid.links[0].geometry['collision'][0]['size'] = [self.manipulation_envelopes[ent.id]]*3
            attempts = [copy.deepcopy(a) for a in self.fastening_attempts.values()
                        if ent.id in (a['body'], a['part'], a.get('screw'))]
            if attempts:
                ent.state_vars['fasteningEvidence'] = Est(value=attempts, src='inferred')
            frames = [copy.deepcopy(f) for eid, f in self.released_interfaces.items()
                      if ent.id in (eid, f['body'])]
            if frames:
                ent.state_vars['releasedInterfaces'] = Est(value=frames, src='inferred')
        return SceneState(mode="ego", entities=entities, relations=relations, version=self.version, stamp=self.stamp,
                          self_id=SELF_ID)

    def _self_entity(self, frame: dict | None = None) -> Entity:
        ent = Entity.from_dict(self._self_template)
        joints = self.joints if frame is None else (frame.get("joints") or self.joints)
        gripper = self.gripper if frame is None else (frame.get("gripper") or self.gripper)
        stamp = self.stamp if frame is None else float(frame.get("stamp", self.stamp))
        o = self.odometry if frame is None or frame.get("odometry") is None else {**self.odometry, **frame["odometry"]}
        sigma = o.get("sigma") or self.odometry.get("sigma") or {"pos": 0.01, "yaw": 0.01}
        pos_var = sigma["pos"] ** 2
        yaw_var = sigma["yaw"] ** 2
        ent.pose = Est(value=self.base_pose(o), unc=_gaussian6([pos_var, pos_var, 1e-8, 1e-8, 1e-8, yaw_var]),
                       t=stamp, src="observed")
        tree: RigidTree = ent.body
        for joint in tree.joints:
            if joint.id in joints:
                joint.q = Est(value=[joints[joint.id]], unc={"kind": "gaussian", "cov": [1e-6]}, t=stamp,
                              src="observed")
        ent.state_vars["gripper"] = Est(value=gripper["state"], t=stamp, src="observed")
        ent.state_vars["holding"] = Est(value=gripper.get("holding", False), t=stamp, src="observed")
        return ent

    def annotate(self, track_id: str, label: str | None = None, affordances: list[str] | None = None) -> None:
        """Name a track as the robot's agent has identified it: ``label`` becomes its category and
        ``affordances`` replace the category prior's; ``None`` keeps the current value.

        :raises KeyError: no such track."""
        track = self.tracks[track_id]
        if label is not None:
            track.label = label
        if affordances is not None:
            track.annotated_affordances = list(affordances)
        self.version += 1

    def unsupported(self, track: Track) -> bool:
        """Whether a track is physically implausible where it is believed to be: not held, not in the
        held container, clear of the floor, and resting on and inside nothing the belief holds. What
        tops ``wall_like_top`` is a wall, which stands on the floor however its height is estimated."""
        tol = 2 * self.config.rest_gap_m + 2 * math.sqrt(float(track.var[2]))
        if track.held or track.carried or track.bottom() <= tol or track.top() >= self.config.wall_like_top:
            return False
        for other in self.tracks.values():
            if other is not track and not other.held and _inside(other, track.pos, tol) and other.bottom() - tol <= track.bottom() <= other.top() + tol:
                return False
        return True

    def forget(self, track_id: str, reason: str) -> None:
        """Remove a track the robot's agent judged not real: implausible and not perceived again on
        a look at its place. The removal is recorded in ``dropped`` with ``reason``.

        :raises KeyError: no such track.
        :raises ValueError: it is held or carried, or was seen within ``forget_unseen_s``."""
        track = self.tracks[track_id]
        if track.held or track.carried:
            raise ValueError(f"{track_id} is in the gripper's load; it is perceived through the grasp, not forgotten")
        ago = self.stamp - track.last_seen
        if ago < self.config.forget_unseen_s:
            raise ValueError(f"{track_id} was perceived {ago:.1f} s ago (under {self.config.forget_unseen_s:g} s): it is still seen, so it stays")
        self._drop(track, f"forgotten by the agent: {reason}")
        self.version += 1

    def _track_entity(self, track: Track, at=None, yaw: float | None = None) -> Entity:
        p = prior(track.category)
        total = sum(track.category_counts.values())
        cat_unc = {"kind": "categorical", "p": {track.label: 1.0} if track.label else
                   {k: round(v / total, 4) for k, v in track.category_counts.items()}}
        src = "fused" if track.observations > 1 else "observed"
        inflate = 0.0 if track.held else 2.0 * track.sigma
        size = [float(track.size[0] + inflate), float(track.size[1] + inflate), float(track.size[2])]
        affordances = [Est(value=a, src="prior" if track.annotated_affordances is None else "inferred")
                       for a in track.affordances]
        registration = None
        posture_category = "airplane_fuselage" if track.category == "toy_airplane" else track.category
        posture = assembly_posture(
            track.size, SHAPES.get(posture_category, {}).get("dims"),
            slack_m=float(SHAPES.get(posture_category, {}).get("extent_slack", 0)) + 2.0 * track.sigma)
        if track.screw_holes is not None:
            hole_src, holes = "observed", track.screw_holes
        else:
            template_category = track.category
            # The segmenter sometimes names the family instead of the component.
            # Offer a component interface only when its measured outline registers;
            # never give a broad/merged airplane the fuselage's holes by label alone.
            family_candidate = track.category == "toy_airplane"
            if family_candidate:
                template_category = "airplane_fuselage"
            hole_src, holes = "prior", screw_holes(template_category)
            if holes:
                cfg = getattr(self, "config", None) or EstimatorConfig()
                holes, registration = register_hole_template(
                    holes, SCREW_HOLE_FRAMES.get(template_category), track.size,
                    extent_ratio=cfg.hole_registration_extent_ratio,
                    square_ratio=cfg.hole_registration_square_ratio,
                    extent_slack_m=cfg.hole_registration_extent_slack_m + 2.0 * track.sigma, best_effort=True)
                if track.interface_axis_consistent is False:
                    registration.setdefault('warnings', []).append(dict(reason='incoherent_close_axis', sigmaM=float(track.sigma)))
                    registration['lowConfidence'] = True
                if family_candidate:
                    if registration["status"] != "axis_registered":
                        holes, registration = [], None
                    else:
                        registration = {**registration, "templateCategory": template_category,
                                        "categoryHypothesis": True}
        assembly = getattr(self, "_assembly_views", {}).get(track.id)
        if assembly is not None and assembly['status'] != 'stationary_envelope_observed':
            posture = {**posture, 'status': 'inconsistent',
                       'reason': 'stationary assembly reference unresolved; obtain fresh uncut body/union observations'}
        if assembly is not None and assembly['status'] == 'stationary_envelope_observed':
            ref = assembly['reference']
            at, yaw = ref['position'], ref['yaw']
            size = assembly['collisionSize']
            holes, hole_src, registration = ref['holes'], ref['holeSource'], ref['registration']
            posture = {**posture, 'status': 'unknown', 'upright': None,
                       'dimensionsCompatible': True,
                       'source': 'stationary_reference_and_observed_assembly_envelope',
                       'reason': 'observed body/known-release union matches stationary reference; not roll/pitch or fastening verification'}
        affordances.extend(Est(value={"screw-hole": {**hole, "p": list(hole["p"]),
                                                   "frame": "entity" if hole_src == "observed" else hole["frame"]}}, src=hole_src)
                           for hole in holes)
        link = Link(
            id="body", name=track.category,
            geometry={"collision": [primitive("box", size)]},
            material=Est(value={"fragility": p["fragility"]}, src="prior"),
            affordances=affordances,
        )
        root_value = max(p["root"], key=p["root"].get)
        return Entity(
            id=track.id,
            role=p["role"],
            category=Est(value=track.category, unc=cat_unc, t=track.last_seen, src=src),
            pose=Est(value={"p": (track.pos if at is None else np.asarray(at, dtype=float)).tolist(),
                            "q": quat_from_yaw(track.yaw if yaw is None else yaw)},
                     unc=_gaussian6([*track.var.tolist(), 1e-4, 1e-4, track.yaw_var]),
                     t=track.interface_estimate_at if track.interface_estimate_at is not None else track.last_seen,
                     src="inferred" if assembly is not None else src),
            body=RigidTree(root=Est(value=root_value, unc={"kind": "categorical", "p": p["root"]}, src="prior"),
                           links=[link]),
            existence=round(track.existence, 4),
            state_vars={**({"color": Est(value=track.color, src=src)} if track.color else {}),
                        **({"orientation": Est(value={
                            "marker": "[RSI_HOLE_REGISTER]" if registration else "[RSI_YAW_AMBIGUOUS]",
                            "frame": "estimated_box", "directedYawObserved": False,
                            **({"assembly": assembly} if assembly is not None else {}),
                            "posture": {**posture, "observedAt": track.last_seen,
                                        "reason": ("held shape is a grasp-time reference, not a current upright observation"
                                                   if track.held else posture["reason"])},
                            **({"holeRegistration": registration} if registration else {}),
                            "interfaceAxisConsistent": track.interface_axis_consistent,
                            "axisPeriodDeg": 90 if min(track.size[:2]) >= SQUARE_SIDES * max(track.size[:2]) else 180,
                            **({"heldRotationSource": "grasp_relative_plus_encoders"} if track.held else {}),
                            **({"graspObservedAt": track.grasp_association["observedAt"]}
                               if track.held and track.grasp_association else {}),
                            **({"graspWorldYawRad": track.grasp_world_yaw} if track.held and track.grasp_world_yaw is not None else {}),
                        }, src="inferred")} if holes or assembly is not None or track.interface_axis_consistent is False or (track.placement and "orientationTarget" in track.placement) else {})},
        )


_COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.15, 0.70, 0.25), "blue": (0.15, 0.35, 0.90),
           "black": (0.06, 0.06, 0.07), "white": (0.95, 0.95, 0.96), "gray": (0.60, 0.62, 0.65),
           "brown": (0.58, 0.40, 0.24), "beige": (0.80, 0.67, 0.50), "yellow": (0.90, 0.85, 0.20)}


ACHROMATIC_SATURATION = 0.18
_CHROMATIC = {k: v for k, v in _COLORS.items() if k not in ("black", "white", "gray")}


def _apply_joint(pose: dict, joint: dict, q: float) -> dict:
    """Move ``pose`` across one joint by ``q`` (a joint angle in radians, or a slide in metres)."""
    origin = joint["origin"]["value"]
    kind = joint["type"]["value"] if isinstance(joint["type"], dict) else joint["type"]
    at = pose_compose(pose, {"p": origin["p"], "q": origin["q"]})
    if kind == "fixed":
        return at
    axis = joint["axis"]["value"] if isinstance(joint["axis"], dict) else joint["axis"]
    axis = [float(v) for v in axis]
    if kind == "prismatic":
        offset = quat_rotate(at["q"], [axis[0] * q, axis[1] * q, axis[2] * q])
        return {"p": [at["p"][i] + offset[i] for i in range(3)], "q": list(at["q"])}
    if kind == "revolute":
        spin = quat_from_axis_angle(axis, q)
        return {"p": list(at["p"]), "q": quat_mul(at["q"], spin)}
    raise ValueError(f"unsupported joint type {kind!r} for forward kinematics")


def color_name(rgb: np.ndarray, invalid_frac: float = 0.0) -> str:
    """Nearest named colour of a measured mean colour; surfaces without depth readings that look
    pale are reported as clear."""
    rgb = np.asarray(rgb, float)
    value, low = float(rgb.max()), float(rgb.min())
    if invalid_frac > 0.15 and value > 0.55 and (value - low) / max(value, 1e-6) < 0.25:
        return "clear"
    if (value - low) / max(value, 1e-6) < ACHROMATIC_SATURATION:
        # Without hue, brightness alone tells black, gray and white apart (scaling a reference to the
        # measured brightness, as for hues below, would make all three the same).
        return "black" if value < 0.2 else "gray" if value < 0.55 else "white"
    # Lighting darkens surfaces: compare chromaticity scaled to the measured brightness.
    best, best_d = "gray", math.inf
    for name, ref in _CHROMATIC.items():
        ref = np.asarray(ref)
        scale = max(value, 1e-3) / max(float(ref.max()), 1e-3)
        d = float(np.sum((rgb - ref * min(scale, 1.4)) ** 2))
        if d < best_d:
            best, best_d = name, d
    return best


VOXEL = 0.01
NEST_CENTER_SHARE = 0.35  # nested bowls share a centre, within this share of the smaller radius
BOWL_ROUND_SHARE = 0.75  # a bowl box's short side is at least this share of its long side; less is a part of one
# A quarter-turn look-around includes a heading within 45 degrees of broadside. cos of that
# angle is 0.707, so a view that meets a plan axis less strongly than this can show both ends.
# The memory is kept only when the measured length is already most of the prior: a face a few
# centimetres thick is still the hidden side, and a later view fills it. A facing side is left
# as measured only when it is already half a metre and most of the prior.
SEEN_ACROSS_DOT = 0.75
FULL_AXIS = 0.65
SEEN_FACE_M = 0.50  # a facing side this long, and already most of the prior, is the piece itself
ESTABLISHED_VIEWS = 5
HELD_FALLBACK_M = 0.1
YAW_MOVED = 0.3  # a detected heading this far off a track's means the object was turned
STACK_TOL = 0.005  # a region this little below another's top face still rests on it
IDENTIFIED_SHARE = 0.6
NAMED_VIEWS = 2  # a name from a single view may be a phantom of one mask
HELD_LARGER = 2.0  # a region with a footprint this many times the held object's is not the held object
CONVEX_CHECK_M = 0.8  # boxes longer than this are checked for being filled rather than an outline
MIN_FOOTPRINT_FILL = 0.2
SMALL_M = 0.3
# A bear, pistol or bottle already named from its own mask. A later view that calls the
# same small box a drawer is the drawer behind it, not a new name for the object.
_CONTENT = ("teddy_bear", "pistol", "pill_bottle", "perfume_bottle", "toy_airplane")
_BOTTLE_NAMES = ("bottle", "pill_bottle", "perfume_bottle")  # the segmenter's names for either bottle
TABLEWARE = ("bowl", "cup", "mug", "plate", "wine_glass")  # stands on a counter or table top
TABLEWARE_EDGE_M = 0.03  # a piece's centre may be this far past its support's edge before it is not on it
BOWL_ROUND = 0.6  # short over long plan extent from which a single view of a bowl shows the bowl, not a slice of it
CONTENT_VIEWS = 3  # frames that must have shown an unnamed box beside an open drawer for the push to keep it
DRAWER_MAX_M = 1.05  # an open drawer is about 0.9 m long; the sideboard around it is 1.4 m
DRAWER_BOARD_M = 0.04  # a strip thinner than this, this close to the edge of a drawer's box, is one of its boards
DRAWER_FLOOR_TOL = 0.06  # a content's base this far under a drawer box's bottom still stands on the drawer floor
SHUT_FOLLOW_MAX_M = 0.12  # a pushed drawer slides on at most this far past the base's travel
DRAWER_MAX_HEIGHT = 0.40  # a drawer front is about 0.22 m tall; the sideboard face behind it is 0.85 m
_FURNITURE_CAP = ("cabinet", "sideboard", "counter", "table")
_FURNITURE_LONG_PAD = 0.08  # past the prior long side: still the piece, not the open drawer
_FURNITURE_SHORT_PAD = 0.06


def _trim_long_smear(cloud: np.ndarray, width: float | None = None, aspect: float = 2.2) -> np.ndarray:
    """Drop a one-axis smear from a small raised box.

    A pill bottle is about 4 cm across. A mask that also takes a strip of the drawer floor
    is longer than the bottle priors' aspect, and the pill prior then cannot win. The short
    side is the object; the long side is the strip.

    :param aspect: long/short ratio past which the long side is the strip. The caller passes
        the bottle priors' own ratio.
    """
    if len(cloud) < 20:
        return cloud
    centre, size, yaw = box_of(cloud, trim=0.0)
    long = float(max(size[0], size[1]))
    short = float(min(size[0], size[1]))
    height = float(size[2])
    if long < aspect * short or short >= 0.08 or long >= 0.30 or not (0.05 < height < 0.16):
        return cloud
    band = short if width is None else float(width)
    if band >= 0.08 or band <= 0.0:
        return cloud
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rel = cloud[:, :2] - centre[:2]
    lx = rel[:, 0] * c + rel[:, 1] * s
    ly = -rel[:, 0] * s + rel[:, 1] * c
    # The long side of the point set, not size[0]: the box heading is not always that side.
    local = lx if float(np.ptp(lx)) >= float(np.ptp(ly)) else ly
    keep = np.abs(local - float(np.median(local))) <= band * 0.5
    if int(keep.sum()) < 15:
        return cloud
    return cloud[keep]


def _furniture_axis_caps(size: np.ndarray, proto: tuple[float, float, float]) -> tuple[float, float]:
    """Prior cap for each plan axis, the longer measurement taking the longer prior.

    :param size: full extents of the box, metres.
    :param proto: prior length, depth and height, metres.
    :returns: cap of axis 0 and cap of axis 1, metres.
    """
    long_cap = proto[0] + _FURNITURE_LONG_PAD
    short_cap = proto[1] + _FURNITURE_SHORT_PAD
    if float(size[0]) >= float(size[1]):
        return long_cap, short_cap
    return short_cap, long_cap


def _over_furniture_cap(track: Track, size: np.ndarray, yaw: float, centre: np.ndarray,
                       cam_p: np.ndarray) -> bool:
    """Whether the extent toward the camera is past the furniture prior.

    That extent is an open drawer. A longer piece, whose extra length lies across the view,
    is the furniture itself and is not past the cap.

    :param size: full extents of the box, metres.
    :param yaw: heading of the box x axis, radians.
    :param centre: box centre, metres.
    :param cam_p: camera position, metres.
    :returns: true when a camera-facing axis exceeds the prior pad.
    """
    if track.category not in _FURNITURE_CAP or track.observations < ESTABLISHED_VIEWS:
        return False
    proto = prototype(track.category)
    if proto is None:
        return False
    caps = _furniture_axis_caps(size, proto)
    for axis in (0, 1):
        if float(size[axis]) <= caps[axis]:
            continue
        if _faces_camera(yaw, axis, centre, cam_p):
            return True
    return False


def _faces_camera(yaw: float, axis: int, centre: np.ndarray, cam_p: np.ndarray) -> bool:
    """Whether plan axis ``axis`` of a box points toward the camera.

    :param yaw: heading of the box x axis, radians.
    :param axis: 0 for the box x axis, 1 for the box y axis.
    :param centre: box centre, metres.
    :param cam_p: camera position, metres.
    :returns: true when the axis is within :data:`SEEN_ACROSS_DOT` of the view.
    """
    away = np.asarray(centre[:2], float) - np.asarray(cam_p[:2], float)
    norm = float(np.linalg.norm(away))
    if norm < 1e-6:
        return True
    view = away / norm
    direction = np.array([math.cos(yaw), math.sin(yaw)]) if axis == 0 else np.array([-math.sin(yaw), math.cos(yaw)])
    return abs(float(view @ direction)) >= SEEN_ACROSS_DOT


def _trim_to_furniture_prior(track: Track, cloud: np.ndarray, cam_p: np.ndarray) -> np.ndarray:
    """Drop points that stick out past a furniture prior, on the side toward the camera.

    That side is the open drawer. A longer piece seen across the view is left as measured.
    ``cloud`` is returned unchanged when the box already fits."""
    if len(cloud) < 20:
        return cloud
    centre, size, yaw = box_of(cloud, trim=0.01)
    if not _over_furniture_cap(track, size, yaw, centre, cam_p):
        return cloud
    centre, size, yaw = box_of(cloud, trim=0.0)
    proto = prototype(track.category)
    if proto is None:
        return cloud
    caps = _furniture_axis_caps(size, proto)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rel = cloud[:, :2] - centre[:2]
    local = np.column_stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c])
    cam = np.asarray(cam_p[:2], float) - centre[:2]
    cam_l = np.array([float(cam[0] * c + cam[1] * s), float(-cam[0] * s + cam[1] * c)])
    keep = np.ones(len(cloud), dtype=bool)
    for axis in (0, 1):
        if float(size[axis]) <= caps[axis]:
            continue
        if not _faces_camera(yaw, axis, centre, cam_p):
            continue
        coord = local[:, axis]
        lo, hi = float(coord.min()), float(coord.max())
        if cam_l[axis] >= 0.0:
            keep &= coord <= lo + caps[axis]
        else:
            keep &= coord >= hi - caps[axis]
    if int(keep.sum()) < 20:
        return cloud
    return cloud[keep]


def _median_of_closer_views(recent) -> tuple[np.ndarray, np.ndarray]:
    """Centre and size of a small object from the views taken nearest to it.

    A far view's depth noise stretches the box along the line of sight. Once the robot has
    looked from closer up, those nearer views are the size; a view more than 0.4 m farther
    than the nearest one is left out. Views recorded without a range still count.

    :param recent: ``(centre, size)`` or ``(centre, size, range_m)`` rows.
    :returns: median centre and median size of the rows that count.
    """
    rows = list(recent)
    ranged = [row for row in rows if len(row) > 2]
    if len(ranged) >= 2:
        nearest = min(float(row[2]) for row in ranged)
        close = [row for row in ranged if float(row[2]) <= nearest + 0.4]
        if len(close) >= 2:
            rows = close
    return np.median([row[0] for row in rows], axis=0), np.median([row[1] for row in rows], axis=0)


RECENT_VIEWS = 12
WHOLE_VIEWS = 3  # single-view boxes after which a view cut by the image border no longer counts toward a small box
APPEARANCE_SIZE_CHANGE = 0.2  # an extent this much off the one a face grid was laid out for starts a new grid
SQUARE_SIDES = 0.9  # a footprint whose shorter side is at least this share of the longer is square


def _in_boxes(points: np.ndarray, boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], margin: float) -> np.ndarray:
    """Which of ``points`` (N×3) lie within ``margin`` of any of the oriented ``boxes``."""
    inside = np.zeros(len(points), dtype=bool)
    for centre, half, rot in boxes:
        local = (points - centre) @ rot
        inside |= np.all(np.abs(local) <= half + margin, axis=1)
    return inside


def _box_corners(centre: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """The eight corners (8×3) of a box turned by ``yaw`` about the vertical."""
    c, s = math.cos(yaw), math.sin(yaw)
    signs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    local = signs * np.asarray(size, float)
    return np.asarray(centre, float) + np.stack([c * local[:, 0] - s * local[:, 1], s * local[:, 0] + c * local[:, 1],
                                                 local[:, 2]], axis=1)


def _surface_level(pts: np.ndarray, counts: np.ndarray, edges: np.ndarray) -> float | None:
    """Top of the highest dense height bin that spans the furniture.

    A bowl on the counter is dense in its own column and would otherwise lift the cap up to its
    rim, so the bowl stays inside the counter. A bin counts as the surface only when its points
    stretch across the furniture.

    :returns: metres, or None when no bin is dense.
    """
    dense = np.nonzero(counts >= 0.3 * float(counts.max()))[0]
    if len(dense) == 0:
        return None
    span = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    need = 0.45 * float(max(span[0], span[1], 1e-6))
    chosen = None
    for index in dense:
        band = pts[(pts[:, 2] >= edges[index]) & (pts[:, 2] < edges[index + 1])]
        if len(band) < 10:
            continue
        wide = band[:, :2].max(axis=0) - band[:, :2].min(axis=0)
        if float(max(wide[0], wide[1])) >= need:
            chosen = int(index)
    if chosen is None:
        chosen = int(dense[-1])
    return float(edges[chosen + 1])


def _densest_window(points: np.ndarray) -> np.ndarray | None:
    """The square of points on a smeared cloud's busiest cell, or None when the cloud is even.

    :param points: map-frame points of one track.
    :returns: the object's points, or None.
    """
    if len(points) < 30:
        return None
    cell = 0.03
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], int] = {}
    for key in keys:
        buckets[(int(key[0]), int(key[1]))] = buckets.get((int(key[0]), int(key[1])), 0) + 1
    if len(buckets) < 2:
        return None
    best = max(buckets, key=lambda item: buckets[item])
    if buckets[best] < 2.0 * float(np.median(list(buckets.values()))):
        return None
    centre = (np.array(best, float) + 0.5) * cell
    extent = points.max(axis=0) - points.min(axis=0)
    side = float(min(extent[0], extent[1]))
    if side < 0.04:
        return None
    keep = np.max(np.abs(points[:, :2] - centre), axis=1) <= side / 2 + cell
    if int(keep.sum()) < 15:
        return None
    return points[keep]


def _content_cluster(horiz: float, height: float) -> bool:
    """Whether a peeled cluster is one object.

    A pistol is about 2 cm thick, under the 2.5 cm floor that keeps a counter's noise out.
    A flat cluster counts only when it is also as wide as the pistol, not a speck.

    :param horiz: longer plan extent, metres.
    :param height: vertical extent, metres.
    :returns: true when the cluster is one object.
    """
    if horiz > 0.32:
        return False
    if 0.025 <= height <= 0.25:
        return True
    return 0.012 <= height < 0.025 and 0.08 <= horiz <= 0.24


def _shed_cores(points: np.ndarray) -> list[np.ndarray]:
    """Compact clusters among points that stood above a piece of furniture.

    A bowl and a cup on the same counter are two clusters. A cluster under eight points, or taller
    than a bottle, is not an object.

    :param points: map-frame points peeled off a furniture cloud.
    :returns: the clusters, largest first.
    """
    if len(points) < 8:
        return []
    rest = np.asarray(points, dtype=np.float64)
    found: list[np.ndarray] = []
    for _ in range(6):
        if len(rest) < 8:
            break
        extent = rest.max(axis=0) - rest.min(axis=0)
        horiz = max(float(extent[0]), float(extent[1]))
        height = float(extent[2])
        if _content_cluster(horiz, height):
            found.append(rest)
            break
        idx = _largest_compact(rest, 8)
        if idx is None:
            break
        core = rest[idx]
        extent = core.max(axis=0) - core.min(axis=0)
        if _content_cluster(max(float(extent[0]), float(extent[1])), float(extent[2])):
            found.append(core)
        keep = np.ones(len(rest), dtype=bool)
        keep[idx] = False
        rest = rest[keep]
    return found


def _printed_sheet(span: np.ndarray | None, semantics: dict | None) -> bool:
    if span is None or not semantics or max(semantics.get(c, 0.) for c in ("manual", "picture")) < 0.6:
        return False
    _, size, _ = box_of(span)
    return 0.05 <= min(size[:2]) <= max(size[:2]) <= 0.32 and size[2] <= 0.008


def _printed_fragment(sheet: np.ndarray, points: np.ndarray) -> bool:
    centre, size, yaw = box_of(sheet)
    if len(points) == 0 or np.ptp(points[:, 2]) > 0.008:
        return False
    # Same observed surface, not an object on the sheet or a neighbouring sheet.
    if np.max(np.abs(points[:, 2] - centre[2])) > 0.004:
        return False
    c, s = math.cos(yaw), math.sin(yaw)
    local = (points[:, :2] - centre[:2]) @ np.array([[c, -s], [s, c]])
    if np.prod(np.ptp(local, axis=0)) > 0.75 * np.prod(size[:2]):
        return False
    return bool(np.all(np.abs(local) <= size[:2] / 2 + 0.008))


def _small_observations_apart(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Disjoint measured boxes of small hardware must survive proximity-based fusion.

    Use full-resolution observation corners, before prior completion or voxel rounding.
    A two-centimetre gap allowance accommodates small viewpoint/depth errors; larger
    objects retain the existing fragment-merging policy. Missing geometry is not evidence
    of separation. This also gates association, so a track cannot walk onto its neighbour.
    """
    if a is None or b is None:
        return False
    alo, ahi = a.min(0), a.max(0)
    blo, bhi = b.min(0), b.max(0)
    for extent in (ahi - alo, bhi - blo):
        if max(extent[:2]) > 0.12 or extent[2] > 0.08:
            return False
    gap = np.maximum(alo[:2] - bhi[:2], blo[:2] - ahi[:2])
    return bool(np.any(gap > 0.02))


def _voxelize(points: np.ndarray) -> np.ndarray:
    return np.unique(np.floor(points / VOXEL).astype(np.int32), axis=0)


def _points(voxels: np.ndarray) -> np.ndarray:
    return (voxels.astype(np.float64) + 0.5) * VOXEL


def _box2d(track: Track) -> Box2D:
    return Box2D(track.id, float(track.pos[0]), float(track.pos[1]), track.yaw, track.size[0] / 2, track.size[1] / 2)


def _gaussian6(diag: list[float]) -> dict:
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = float(v)
    return {"kind": "gaussian", "cov": cov}


def _to_local_xy(track: Track, point: np.ndarray) -> tuple[float, float]:
    dx, dy = point[0] - track.pos[0], point[1] - track.pos[1]
    c, s = math.cos(-track.yaw), math.sin(-track.yaw)
    return (c * dx - s * dy, s * dx + c * dy)


FLAT_PAIR_HEIGHT_SHARE = 0.2  # a track no taller than this share of its footprint is flat
FLAT_PAIR_OVERLAP_M = 0.5  # flat pieces whose centres are closer than this share of the smaller side may be one


def _flat_pair(a: Track, b: Track) -> bool:
    """Both tracks are flat (much wider than tall): two of them side by side are two objects, and the
    duplicate-merge must not fold one into the other."""
    return all(t.size[2] <= FLAT_PAIR_HEIGHT_SHARE * float(max(t.size[0], t.size[1])) for t in (a, b))


def _sigma_z(a: Track, b: Track) -> float:
    """Standard deviation of the height difference between two tracks."""
    return math.sqrt(float(a.var[2] + b.var[2]))


CONCAVE_MIN_M = 0.5
SEE_THROUGH_DEG = 5.0
SEE_THROUGH_M = 0.15


def _plan_axes(track: Track) -> tuple[np.ndarray, np.ndarray, float]:
    """Unit axes of a track's longer and shorter plan sides, and the shorter extent.

    ``size[0]`` lies along ``yaw``. The longer of the two plan extents is the first axis.

    :returns: long axis, the perpendicular, and the shorter extent in metres.
    """
    yaw = float(track.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    along_yaw = np.array([c, s])
    across = np.array([-s, c])
    if float(track.size[0]) >= float(track.size[1]):
        return along_yaw, across, float(track.size[1])
    return across, along_yaw, float(track.size[0])


def _tight_bowl_views(track: Track) -> list[tuple]:
    """The single-view boxes of ``track`` that are a bowl on its own: at most 0.24 m across and
    round in plan, the short side at least :data:`BOWL_ROUND` of the long one."""
    return [row for row in track.recent
            if float(max(row[1][0], row[1][1])) <= 0.24
            and float(min(row[1][0], row[1][1])) >= BOWL_ROUND * float(max(row[1][0], row[1][1]))]


def _content_named(track: Track) -> bool:
    """Whether a track is a drawer content, by its current name or the segmenter's."""
    if track.category in _CONTENT:
        return True
    return any(track.semantic_counts.get(name, 0.0) > 0.0 for name in _CONTENT)


def _content_names(track: Track) -> set[str]:
    """Content names from the current category and from the segmenter."""
    names = {track.category} if track.category in _CONTENT else set()
    names.update(name for name in _CONTENT if track.semantic_counts.get(name, 0.0) > 0.0)
    return names


def _named_only_as_another_content(track: Track) -> bool:
    """Whether every name the segmenter gave ``track`` is that of a drawer content of another
    kind than its category. A bottle name is one kind, whichever bottle the category is."""
    if track.category not in _CONTENT:
        return False

    def kind(name: str) -> str:
        return "bottle" if name in _BOTTLE_NAMES else name

    given = [name for name, score in track.semantic_counts.items() if score > 0.0]
    return bool(given) and all((name in _CONTENT or name in _BOTTLE_NAMES) and kind(name) != kind(track.category)
                               for name in given)


def _different_contents(a: Track, b: Track) -> bool:
    """Two drawer contents with no name in common are not one object.

    A cup the segmenter called an airplane, sitting on the airplane, shares that name.
    A pill and a perfume do not.
    """
    left, right = _content_names(a), _content_names(b)
    return bool(left) and bool(right) and left.isdisjoint(right)


def _content_in_larger(a: Track, b: Track) -> bool:
    """A drawer content sharing points with a larger other object is still itself."""
    if _content_named(a) and not _content_named(b) and b.footprint > a.footprint * 1.4:
        return True
    return _content_named(b) and not _content_named(a) and a.footprint > b.footprint * 1.4


def _content_on_drawer(a: Track, b: Track) -> bool:
    """A bear, pistol or bottle that shares points with the drawer is still its own object.

    The mask takes some of the drawer floor with the object. Those shared points are not
    a reason to fold the object into the drawer."""
    return ((a.category in _CONTENT and b.category == "drawer")
            or (b.category in _CONTENT and a.category == "drawer"))


def _drawer_beside_furniture(a: Track, b: Track) -> bool:
    """A drawer and the cabinet, sideboard, counter or table it is set into are not one object."""
    furniture = ("cabinet", "sideboard", "counter", "table")
    return (a.category == "drawer" and b.category in furniture) or (b.category == "drawer" and a.category in furniture)


def _boxes_meet(a: Track, b: Track, gap: float) -> bool:
    """Whether the two boxes' footprints come within ``gap`` metres of each other."""
    d = b.pos[:2] - a.pos[:2]
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return True
    direction = d / dist

    def reach(track: Track) -> float:
        c, s = math.cos(track.yaw), math.sin(track.yaw)
        along = abs(float(direction @ np.array([c, s]))) * float(track.size[0]) / 2
        across = abs(float(direction @ np.array([-s, c]))) * float(track.size[1]) / 2
        return along + across

    return dist - reach(a) - reach(b) < gap


def _stacked(a_bottom: float, a_top: float, b_bottom: float, b_top: float, b_centre: float, bases_seen: bool) -> bool:
    """Whether box b rests on top of box a, or a on b, rather than being a view of the same object:
    with both bases seen, one lies wholly above the other's top face."""
    if not bases_seen:
        return False
    on_a = b_centre > a_top and b_bottom >= a_top - STACK_TOL
    under_a = b_top <= a_bottom + STACK_TOL and a_bottom > b_centre
    return on_a or under_a


def _identified(track: Track) -> bool:
    """Seen at least :data:`ESTABLISHED_VIEWS` times and named: a category other than unknown holding
    at least ``IDENTIFIED_SHARE`` of its accumulated category evidence."""
    total = sum(track.category_counts.values())
    category = track.category
    # A label names the track after the votes that earned it have been replaced.
    share = track.category_counts.get(category, 0.0)
    return track.observations >= ESTABLISHED_VIEWS and category != "unknown" and total > 0 \
        and share / total >= IDENTIFIED_SHARE


def _named(track: Track, min_score: float) -> bool:
    """Named by the robot's segmenter in at least :data:`NAMED_VIEWS` views, with a mean score of
    at least ``min_score`` for some category other than a wall."""
    if track.semantic_views < NAMED_VIEWS:
        return False
    return any(score / track.semantic_views >= min_score
               for cat, score in track.semantic_counts.items() if cat != "wall")


def _mixed_ranges(points: np.ndarray, cam_p: np.ndarray, size: np.ndarray) -> bool:
    """Whether a region mixes a near surface with far background.

    The camera distance between the near and far tenths of its points exceeds 0.9 m, little of the
    cloud sits between them, and the box is wide and thick. A table seen along its length also
    spans that distance, but its points fill the interval. A wall is thin along its normal."""
    if min(float(size[0]), float(size[1])) <= 0.25 or max(float(size[0]), float(size[1])) <= 1.0:
        return False
    dist = np.linalg.norm(points - cam_p, axis=1)
    lo, hi = float(np.percentile(dist, 10)), float(np.percentile(dist, 90))
    if hi - lo <= 0.9:
        return False
    span = hi - lo
    mid = (dist > lo + 0.3 * span) & (dist < hi - 0.3 * span)
    return float(mid.mean()) < 0.15


def _larger_span(old: np.ndarray | None, new: np.ndarray) -> np.ndarray:
    """The full-resolution box that covers more floor, unless the new one is a different object.

    Corners from several views are not unioned: a few centimetres of disagreement between views
    would swell the box the same way a merged duplicate does."""
    if old is None:
        return new
    old_s, new_s = box_of(old)[1], box_of(new)[1]
    old_area, new_area = float(old_s[0] * old_s[1]), float(new_s[0] * new_s[1])
    if old_area * 1.05 < new_area < old_area * 1.5:
        return new
    return old


def _absorb_span(keep: Track, drop: Track) -> None:
    """Keep the fuller of two merged tracks' full-resolution boxes."""
    if drop.span is None:
        return
    keep.span = _larger_span(keep.span, drop.span)


def _absorb_views(keep: Track, drop: Track) -> None:
    """Keep the recent views of both merged tracks, oldest first, within the deque's length.

    When a wide new box swallows the established track of the same object, the established
    views are the object's history: the bowl's size and range come from them, and a single
    long smear from the newcomer would otherwise be all that is left.
    """
    older, newer = (drop, keep) if drop.first_seen <= keep.first_seen else (keep, drop)
    rows = list(older.recent) + list(newer.recent)
    keep.recent.clear()
    keep.recent.extend(rows)


def _labels_within_height(semantics: dict[str, float], height: float) -> dict[str, float]:
    """``semantics`` without tableware categories whose prior height cannot reach ``height``.

    The prior may be overshot by :data:`OVER`, and a scaled category may be as tall as its largest
    scale. A measured box above that is not that category. Floor-standing categories and drawers
    stay: a cabinet slice is often narrower than 0.3 m and taller than a drawer, and dropping
    ``drawer`` there renames the slice so it is folded into the cabinet.
    """
    limit = math.exp(OVER)
    kept = {}
    for cat, score in semantics.items():
        shape = SHAPES.get(cat)
        if shape is None or shape.get("floor") or cat == "drawer":
            kept[cat] = score
            continue
        hi = shape.get("scale", (1.0, 1.0))[1]
        # A bottle mask that includes the surface under it is about a centimetre taller than
        # the bottle. Dropping the name there leaves the generic bottle, which has no shape prior.
        if height <= shape["dims"][2] * hi * limit + 0.015:
            kept[cat] = score
    return kept


def _slice_of(whole: Track, part: Track) -> bool:
    """Whether ``part`` is a slab cut from furniture-tall ``whole``: under a quarter of its height.

    The top band or one edge of a cabinet is boxed on its own a few centimetres tall. It shares
    the piece's footprint and name without being a second measurement of its centre.
    """
    return float(whole.size[2]) >= 0.4 and float(part.size[2]) < 0.25 * float(whole.size[2])


def _tighter_bowl(a: Track, b: Track) -> bool:
    """Whether bowl ``a``'s box should replace bowl ``b``'s: it is :func:`_tighter`, and it is not
    just a sliver of the same bowl.

    A bowl is round. A box of nearly the same long side but a short side under
    :data:`BOWL_ROUND_SHARE` of it is the half of the bowl one view caught, not a tighter box
    than the whole; the wider box it would replace is the bowl.
    """
    if not _tighter(a, b):
        return False
    round_a = float(min(a.size[0], a.size[1])) / max(float(max(a.size[0], a.size[1])), 1e-6)
    sliver = round_a < BOWL_ROUND_SHARE and a.footprint > 0.8 * b.footprint
    return not sliver


def _tighter(a: Track, b: Track) -> bool:
    """Whether ``a`` is the object and ``b`` is a flat or wider duplicate of it.

    A rim a couple of centimetres tall is the top of the cup, not a better box than the cup.
    Otherwise the smaller footprint is the object cut off the table, and the wider one is the table.
    """
    if a.size[2] >= 0.04 > b.size[2]:
        return True
    if b.size[2] >= 0.04 > a.size[2]:
        return False
    return a.footprint <= b.footprint

def _absorb_hardware_views(keep: Track, drop: Track) -> None:
    # Geometry rebuilding uses the recent single-view boxes, not the unioned
    # cloud, for small objects. Copying only a screw's label discards the compact
    # view that supports it, leaving an older elongated gear box in control.
    if {keep.category, drop.category} <= {"toy_screw", "airplane_landing_gear"}:
        _absorb_views(keep, drop)


def _absorb_semantics(keep: Track, drop: Track) -> None:
    """Add the segmenter evidence of a track merged away to the track that keeps it."""
    for cat, n in drop.semantic_counts.items():
        keep.semantic_counts[cat] = keep.semantic_counts.get(cat, 0.0) + n
    keep.semantic_views += drop.semantic_views


def _edge_on(track: Track, cam_p: np.ndarray) -> bool:
    """Whether the box's long side runs along the line of sight from ``cam_p``, within about 30°.

    A face seen edge-on is boxed as a sheet whose length is its depth from the camera; a face
    seen from in front lies across the line of sight.
    """
    yaw = float(track.yaw)
    long_axis = (np.array([math.cos(yaw), math.sin(yaw)]) if float(track.size[0]) >= float(track.size[1])
                 else np.array([-math.sin(yaw), math.cos(yaw)]))
    to_track = np.asarray(track.pos[:2], float) - np.asarray(cam_p[:2], float)
    distance = float(np.linalg.norm(to_track))
    if distance < 0.3:
        return False
    return abs(float(long_axis @ to_track)) / distance > 0.85


def _concave(points: np.ndarray, centre: np.ndarray, footprint: float, cam_p: np.ndarray) -> bool:
    """Whether the box drawn around a large region is seen through: the surfaces the camera sees in
    the direction of the box centre lie beyond that centre. A solid block shows its near face there,
    in front of the centre; the corner where two walls meet, boxed as if it were a block, shows the
    walls behind it, because the box's inside is open room."""
    if footprint < CONCAVE_MIN_M or len(points) < 10:
        return False
    to_centre = centre[:2] - cam_p[:2]
    middle = float(np.linalg.norm(to_centre))
    if middle < 0.5 * footprint + 0.1:
        return False  # the camera is over the box: there is no near face to see from here
    rel = points[:, :2] - cam_p[:2]
    bearing = np.degrees(np.abs(np.arctan2(rel[:, 1], rel[:, 0]) - math.atan2(to_centre[1], to_centre[0])))
    bearing = np.minimum(bearing, 360.0 - bearing)
    along = np.linalg.norm(rel[bearing <= SEE_THROUGH_DEG], axis=1)
    if len(along) < 5:
        return False
    # A solid seen from several sides keeps the faces of the sides it was seen from, and the far
    # face lies beyond the centre as a matter of course. Only an *empty* near side — nothing at
    # all between the camera and the centre — says the box encloses open room.
    near_face = float(np.min(along)) < middle - SEE_THROUGH_M
    return not near_face and float(np.median(along)) > middle + SEE_THROUGH_M


def _continuous_heading(yaw: float, size: np.ndarray, previous: float) -> tuple[float, np.ndarray]:
    """The heading among a fitted box's equivalent ones nearest ``previous``, with its extents.

    A box fits the same points turned half way round; a square footprint (sides within
    :data:`SQUARE_SIDES`) also turned a quarter, where the fit's x (its longer side) swaps with
    noise. Staying nearest the previous heading keeps an object's frame, and the faces reported in
    it, fixed to the object between views; a quarter turn swaps the x and y extents."""
    square = min(size[0], size[1]) >= SQUARE_SIDES * max(size[0], size[1])
    step = math.pi / 2 if square else math.pi
    turns = round(wrap_angle(previous - yaw) / step)
    if square and turns % 2:
        size = size[[1, 0, 2]]
    return wrap_angle(yaw + turns * step), size


def _plan_axes_swapped(previous: float, yaw: float) -> bool:
    """Whether the box x axis now lies where its y axis did.

    A square footprint can turn a quarter to stay near the previous heading, and that swaps the
    plan extents. A memory of which extent was seen in full has to swap with them.

    :param previous: heading before this fit, radians.
    :param yaw: heading after :func:`_continuous_heading`, radians.
    :returns: true when the turn is a quarter turn.
    """
    return abs(abs(wrap_angle(yaw - previous)) - math.pi / 2) < 0.2


def _swap_plan_axis(bits: int) -> int:
    """Exchange the two plan-axis bits of ``seen_across``."""
    return ((bits & 1) << 1) | ((bits >> 1) & 1)


def _footprint_fill(points: np.ndarray, centre: np.ndarray, size: np.ndarray, yaw: float, cell: float = 0.05) -> float:
    """Share of a box's footprint (in ``cell``-sized squares) that the points cover seen from above."""
    c, s = math.cos(yaw), math.sin(yaw)
    rel = points[:, :2] - centre[:2]
    lx, ly = rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c
    nx, ny = max(1, math.ceil(size[0] / cell)), max(1, math.ceil(size[1] / cell))
    ix = np.clip(((lx + size[0] / 2) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((ly + size[1] / 2) / cell).astype(int), 0, ny - 1)
    return len(set(zip(ix.tolist(), iy.tolist(), strict=True))) / (nx * ny)


def _footprint_inside(track: Track, m: dict, margin: float) -> bool:
    """Whether all four corners of a measurement's box lie inside the track's box."""
    c, s = math.cos(m["yaw"]), math.sin(m["yaw"])
    hx, hy = m["size"][0] / 2, m["size"][1] / 2
    return all(_inside(track, np.array([m["centre"][0] + c * dx - s * dy, m["centre"][1] + s * dx + c * dy]), margin)
               for dx in (-hx, hx) for dy in (-hy, hy))


def _inside(track: Track, point: np.ndarray, margin: float) -> bool:
    lx, ly = _to_local_xy(track, point)
    return abs(lx) <= track.size[0] / 2 + margin and abs(ly) <= track.size[1] / 2 + margin


def _standing_on(track: Track, other: Track) -> bool:
    """Whether ``track`` sits on ``other``, and is not a differently coloured bottle beside it."""
    if not _inside(other, track.pos, 0.02):
        return False
    if track.mean_rgb is not None and other.mean_rgb is not None:
        if float(np.linalg.norm(track.mean_rgb - other.mean_rgb)) > 0.25:
            return False
    return True


def _on_wall_plane(track: Track, wall: Track) -> bool:
    """Whether ``track`` sits on the wall's centre plane, not merely inside a swollen wall box.

    :param track: the small object.
    :param wall: the wall track.
    :returns: true when the centre is within 4 cm of the wall plane and along its length.
    """
    lx, ly = _to_local_xy(wall, track.pos)
    if float(wall.size[0]) <= float(wall.size[1]):
        return abs(lx) <= 0.04 and abs(ly) <= float(wall.size[1]) / 2
    return abs(ly) <= 0.04 and abs(lx) <= float(wall.size[0]) / 2


def _behind_wall(track: Track, wall: Track, cam_p: np.ndarray) -> bool:
    """Whether ``track``'s centre lies beyond ``wall`` as seen from ``cam_p``.

    The camera sees nothing through a wall. A box there was completed away from the camera from
    a sliver seen on the near side: the far edge of a cabinet top, with the wall behind it.

    :param cam_p: camera position, map frame.
    :returns: true when the centre is within the wall's length and more than 10 cm past the
        wall's box, on the side the camera is not on.
    """
    lx, ly = _to_local_xy(wall, track.pos)
    cx, cy = _to_local_xy(wall, cam_p)
    if float(wall.size[0]) <= float(wall.size[1]):
        across, along, cam_across, thick, length = lx, ly, cx, float(wall.size[0]), float(wall.size[1])
    else:
        across, along, cam_across, thick, length = ly, lx, cy, float(wall.size[1]), float(wall.size[0])
    return abs(along) <= length / 2 and across * cam_across < 0.0 and abs(across) > thick / 2 + 0.10


def _in_wall_corner(track: Track, walls: list[Track]) -> bool:
    """Whether ``track`` is where two walls meet.

    A measured wall often stops a few centimetres short of the corner, so the corner is past
    the length of both walls. It is still on the face of both.

    :param track: the small object.
    :param walls: the wall tracks.
    :returns: true when the centre is within 8 cm of the faces of two walls that cross.
    """
    faces = []
    for wall in walls:
        lx, ly = _to_local_xy(wall, track.pos)
        thin_x = float(wall.size[0]) <= float(wall.size[1])
        across, thick = (lx, float(wall.size[0])) if thin_x else (ly, float(wall.size[1]))
        if abs(across) <= thick / 2 + 0.08:
            faces.append(float(wall.yaw) + (math.pi / 2 if thin_x else 0.0))
    return any(abs(math.sin(a - b)) > 0.7 for i, a in enumerate(faces) for b in faces[i + 1:])


def _against_wall(track: Track, wall: Track) -> bool:
    """Whether ``track`` stands on ``wall`` rather than in the room.

    The wall is thin along one plan axis. A point on that face, and within the wall's length,
    is the corner peeled off the wall.
    """
    lx, ly = _to_local_xy(wall, track.pos)
    thin_x = float(wall.size[0]) <= float(wall.size[1])
    if thin_x:
        return abs(lx) <= float(wall.size[0]) / 2 + 0.08 and abs(ly) <= float(wall.size[1]) / 2
    return abs(ly) <= float(wall.size[1]) / 2 + 0.08 and abs(lx) <= float(wall.size[0]) / 2
