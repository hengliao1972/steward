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
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from steward.ego.classify import SHAPES, classify, prototype
from steward.ego.perception import box_at_yaw, box_of, segments
from steward.ego.planner import Box2D, front_box, protrusion, room_centre
from steward.ego.priors import compatible, prior
from steward.math3d import pose_compose, pose_inverse, quat_from_yaw, wrap_angle, yaw_from_quat
from steward.scene.schema import Entity, Est, Link, Relation, RigidTree, SceneState, primitive

SELF_ID = "e:self"


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
    held_offset: np.ndarray | None = None
    color_counts: dict[str, float] = field(default_factory=dict)
    voxels: np.ndarray | None = None  # 1 cm voxels of every surface seen of this object (RGB-D mode)
    completion: float = 0.0  # how far the box was extended beyond what was seen (shape prior)
    settled: bool = True  # False right after the robot put it down, until it is seen again
    mean_rgb: np.ndarray | None = None  # running mean colour of its regions (RGB-D mode)
    mean_invalid: float = 0.0  # running mean share of its pixels without depth
    bottom_seen: bool = False  # some view showed its base (not cut by the image edge or hidden)
    label: str | None = None  # a category named by the robot's agent; it outranks the classifier
    annotated_affordances: list[str] | None = None  # affordances named with it, in place of the category prior
    held_yaw: float = 0.0  # while held: its heading relative to the base heading plus the wrist yaw
    absent_views: list[tuple[float, float, float, float]] = field(default_factory=list)
    # (t, camera x, camera y, camera yaw) of views that saw through its place since it was last observed
    recent: collections.deque = field(default_factory=lambda: collections.deque(maxlen=RECENT_VIEWS))
    # (centre, size) of its latest single-view measurements, for the robust box of small objects
    appearance: Appearance | None = None

    @property
    def category(self) -> str:
        return self.label or self.perceived_category

    @property
    def perceived_category(self) -> str:
        """The category perception accumulated, regardless of a label: what detections of the same
        object will keep reporting, so association compares with it."""
        return max(self.category_counts, key=self.category_counts.get)

    @property
    def affordances(self) -> list[str]:
        return list(self.annotated_affordances) if self.annotated_affordances is not None \
            else list(prior(self.category)["affordances"])

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
class EstimatorConfig:
    pos_sigma_base: float = 0.01
    pos_sigma_per_m: float = 0.02
    process_noise: float = 1e-6
    gate_m: float = 0.30
    size_gate_ratio: float = 0.25
    size_match_log: float = 0.15
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
    wall_like_top: float = 1.25
    """Robot knowledge of the home: nothing but walls reaches this high, so a region topping it is
    part of a wall whatever its own classification says, and a non-wall track this tall is not real."""


@dataclass
class EgoEstimator:
    robot_model: dict
    config: EstimatorConfig = field(default_factory=EstimatorConfig)

    def __post_init__(self) -> None:
        self.tracks: dict[str, Track] = {}
        self._ids = itertools.count(1)
        self.dropped: collections.deque = collections.deque(maxlen=500)
        """The most recent removals: ``{t, id, category, observations, reason, position, size, robot, into?}``."""
        self.version = 0
        self.stamp = 0.0
        self.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.01, "yaw": 0.01}}
        self.joints: dict[str, float] = {}
        self.gripper = {"state": "open", "holding": False}
        self.held_id: str | None = None
        self._rng = np.random.default_rng(0)
        self._self_template = self._make_self_template()
        self._head_mount, self._camera = self._camera_model()

    # ------------------------------------------------------------- body model
    def _make_self_template(self) -> dict:
        data = copy.deepcopy(self.robot_model)
        data["id"] = SELF_ID
        data.pop("asset", None)
        return data

    def _camera_model(self) -> tuple[dict, dict]:
        tree = self.robot_model["body"]
        for link in tree["links"]:
            for sensor in link.get("sensors", []):
                if sensor["kind"] in ("rgb", "depth", "rgbd"):
                    mount_joint = next(j for j in tree["joints"] if j["child"] == link["id"])
                    return mount_joint["origin"]["value"], sensor
        raise ValueError("robot model has no camera")

    def base_pose(self) -> dict:
        o = self.odometry
        return {"p": [o["x"], o["y"], 0.0], "q": quat_from_yaw(o["yaw"])}

    def camera_pose(self, odometry: dict | None = None) -> dict:
        """World pose of the head camera mount (looks along +x), at the current odometry or a given one."""
        o = odometry or self.odometry
        base = {"p": [o["x"], o["y"], 0.0], "q": quat_from_yaw(o["yaw"])}
        return pose_compose(pose_compose(base, self._head_mount), self._camera["mount"])

    def gripper_position(self) -> np.ndarray:
        """Gripper position from the robot's own kinematics (base, lift, reach)."""
        tree = self.robot_model["body"]
        lift = next(j for j in tree["joints"] if j["id"] == "lift")
        grip = next(j for j in tree["joints"] if j["id"] == "gripper_mount")
        local_x = lift["origin"]["value"]["p"][0] + self.joints.get("reach", 0.0) + grip["origin"]["value"]["p"][0]
        local = {"p": [local_x, 0.0, self.joints.get("lift", 0.0)], "q": [1, 0, 0, 0]}
        return np.asarray(pose_compose(self.base_pose(), local)["p"])

    # ----------------------------------------------------------------- update
    def update(self, frame: dict) -> None:
        self.stamp = float(frame["stamp"])
        self.odometry = frame["odometry"]
        self.joints = frame["joints"]
        self.gripper = frame["gripper"]
        if not self.gripper.get("holding"):
            self._clear_held()
        # A held object moves with this frame's body pose; update it before association so a
        # turning robot does not see its own load as a new object.
        self._follow_held()
        cam = self.camera_pose()
        measurements = []
        for det in frame.get("detections", []):
            world = pose_compose(cam, det["pose"])
            rng = float(det.get("range", np.linalg.norm(det["pose"]["p"])))
            sigma = self.config.pos_sigma_base + self.config.pos_sigma_per_m * rng
            measurements.append((det, np.asarray(world["p"]), yaw_from_quat(world["q"]), sigma))
        # Global nearest-first assignment: of all detection-track pairs within the gate, the closest
        # pair is joined first, so of two objects side by side neither takes the other's track.
        pairs = []
        for k, (det, pos, _yaw, sigma) in enumerate(measurements):
            for track in self._candidates(det["category"], pos, sigma, float(max(det["size"][0], det["size"][1]))):
                pairs.append((float(np.linalg.norm(track.pos - pos)), k, track.id))
        matched: set[str] = set()
        assigned: dict[int, str] = {}
        for _d, k, tid in sorted(pairs):
            if k not in assigned and tid not in matched:
                assigned[k] = tid
                matched.add(tid)
        for k, (det, pos, yaw, sigma) in sorted(enumerate(measurements), key=lambda item: item[1][3]):
            if k in assigned:
                self._fuse(self.tracks[assigned[k]], det, pos, yaw, sigma)
            else:
                matched.add(self._new_track(det, pos, yaw, sigma).id)
        self._decay_unseen(cam, matched)
        self._merge_duplicates()
        self._follow_held()
        self.version += 1

    # ------------------------------------------------------------- RGB-D path
    def update_rgbd(self, frame: dict) -> None:
        """Fuse one decoded RGB-D frame: segment it, measure every region, associate regions with
        tracks by position, size and category, and rebuild each matched track's box from the
        points it has accumulated over recent views."""
        header = frame["header"]
        self.stamp = float(header["stamp"])
        self.odometry = header["odometry"]
        self.joints = header["joints"]
        self.gripper = header["gripper"]
        if not self.gripper.get("holding"):
            self._clear_held()
        self._follow_held()
        cam = self.camera_pose()
        cam_p = np.asarray(cam["p"])
        measurements = []
        for seg in segments(frame, cam, rng=self._rng):
            centre, size, yaw = box_of(seg.points)
            probs = classify(size, centre[2] - size[2] / 2, seg.color, seg.invalid_frac, bottom_seen=seg.bottom_seen)
            footprint = float(max(size[0], size[1]))
            measurements.append({"seg": seg, "centre": centre, "size": size, "yaw": yaw, "probs": probs,
                                 "footprint": footprint, "concave": _concave(seg.points, centre, footprint, cam_p)})
        pairs = []
        held = self.tracks.get(self.held_id) if self.held_id else None
        for i, m in enumerate(measurements):
            # What the robot is carrying moves with it: a region at the load belongs to the load,
            # never to a neighbour it is carried past.
            at_load = held is not None and float(np.linalg.norm(held.pos - m["centre"])) < max(0.15, 0.6 * held.footprint)
            for track in self.tracks.values():
                if at_load and track is not held:
                    continue
                d = float(np.linalg.norm(track.pos - m["centre"]))
                # Neighbouring tableware stands ~0.2 m apart: the gate scales with the object size.
                spread = 3.0 * math.sqrt(float(np.mean(track.var[:2])))
                gate = max(0.10, 0.6 * max(track.footprint, m["footprint"]) + spread, 0.45 if track.held else 0.0)
                # A part of a large object (a table top seen up close, a closed drawer front) lies wholly
                # inside its box and below its top; something standing on it starts at or above the top,
                # and an open drawer reaches out of it.
                m_bottom, m_top = m["centre"][2] - m["size"][2] / 2, m["centre"][2] + m["size"][2] / 2
                part_of_large = track.footprint > 0.3 and _footprint_inside(track, m, 0.05) \
                    and m_top <= track.top() + 0.1 and m_bottom < track.top() - 0.02
                if d > gate and not part_of_large:
                    continue
                if track.held and m["footprint"] > HELD_LARGER * track.footprint + 0.05:
                    continue  # the gripper hides part of what it holds, but a region much larger than the
                    # load (the table under it) is never the load
                size_r = 0.0 if track.held or part_of_large else \
                    abs(math.log(max(m["footprint"], 1e-3) / max(track.footprint, 1e-3)))
                if size_r > 0.9:
                    continue
                if not track.held and track.observations >= 5 and m["footprint"] > 1.5 * track.footprint:
                    continue  # partial views only shrink a well-seen object; a much larger region is something else
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
            self._fuse_cloud(self.tracks[tid], measurements[i], cam_p)
        for i, m in enumerate(measurements):
            # A region cut by the image border is only part of something and waits for a full view,
            # except a wall: a near wall always fills the image edge to edge.
            is_wall = m["probs"].get("wall", 0.0) >= 0.5
            if m["concave"] and not is_wall:
                continue  # the inside of a corner is room, not a solid object
            if i not in used and (not m["seg"].border or is_wall):
                track = self._new_cloud_track(m, cam_p)
                matched.add(track.id)
        self._decay_unseen_rgbd(cam, matched, frame)
        self._merge_parts(cam_p)
        self._merge_same_surface(cam_p)
        self._merge_walls(cam_p)
        self._drop_impossible(cam_p)
        self._follow_held()
        self.version += 1

    def _drop_impossible(self, cam_p: np.ndarray) -> None:
        """Remove tracks that cannot be solid objects: one whose surfaces lie behind its own box
        centre as seen from the camera (a concave corner of the room boxed as if it were a convex
        block), one whose footprint holds the robot's base (the robot stands there), and a non-wall
        track as tall as a wall (wall pieces swept into it). Such boxes would otherwise fill the
        room's corners, hide the objects in them and block every path."""
        base = np.array([self.odometry["x"], self.odometry["y"], 0.0])
        for track in list(self.tracks.values()):
            if track.held or track.category == "wall":
                continue
            holds_robot = _inside(track, base, -0.05) and track.bottom() < 1.0
            wall_tall = track.top() >= self.config.wall_like_top and track.footprint >= 0.5
            # A table (open beneath its top) or an open drawer can look hollow from some viewpoints;
            # an object seen many times and identified with confidence is not a boxed room corner.
            hollow = not _identified(track) and track.voxels is not None \
                and _concave(_points(track.voxels), track.pos, track.footprint, cam_p)
            if holds_robot or wall_tall or hollow:
                self._drop(track, "holds the robot's base" if holds_robot else
                           "as tall as a wall" if wall_tall else "hollow: its surfaces lie behind its centre")

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
            if tid in matched or track.held or self._in_closed_drawer(track):
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
        del self.tracks[track.id]
        self.dropped.append({"t": round(self.stamp, 2), "id": track.id, "category": track.category,
                             "observations": track.observations, "reason": reason,
                             "position": [round(float(v), 3) for v in track.pos],
                             "size": [round(float(v), 3) for v in track.size],
                             "robot": [round(self.odometry["x"], 3), round(self.odometry["y"], 3)],
                             **({"into": into.id} if into is not None else {})})

    def _merge_same_surface(self, cam_p: np.ndarray) -> None:
        """Merge tracks whose accumulated points lie on the same surfaces (duplicates from views
        that were not associated). Nested bowls are different surfaces and stay apart."""
        tracks = [t for t in self.tracks.values() if t.voxels is not None and not t.held]
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                if a.id not in self.tracks or b.id not in self.tracks:
                    continue
                if float(np.linalg.norm(a.pos - b.pos)) > 0.5 * max(a.footprint, b.footprint) + 0.05:
                    continue
                receptacles = "receptacle" in a.affordances and "receptacle" in b.affordances
                if receptacles and not self._size_match(a.footprint, b.footprint):
                    continue  # a smaller bowl nested in a larger one
                shared = {tuple(v) for v in a.voxels} & {tuple(v) for v in b.voxels}
                if len(shared) < 0.5 * min(len(a.voxels), len(b.voxels)):
                    continue
                keep, drop = (a, b) if a.observations >= b.observations else (b, a)
                keep.voxels = self._union(keep.voxels, drop.voxels)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                for col, n in drop.color_counts.items():
                    keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: same surfaces", into=keep)
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
                keep, drop = (a, b) if a.observations >= b.observations else (b, a)
                keep.voxels = self._union(keep.voxels, drop.voxels)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                self._drop(drop, "merged: one wall", into=keep)
                self._rebuild_geometry(keep, cam_p, 2.0)

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
                if small.id == big.id or small.held or small.footprint >= big.footprint:
                    continue
                if not _inside(big, small.pos, 0.1) or abs(small.bottom() - big.bottom()) > 0.15:
                    continue
                if _stacked(big.bottom(), big.top(), small.bottom(), small.top(), float(small.pos[2]),
                            big.bottom_seen and small.bottom_seen):
                    continue
                if small.footprint < 0.3 and small.bottom() > 0.05:
                    continue  # tableware on furniture is not part of it
                if small.footprint >= 0.3 and small.observations >= ESTABLISHED_VIEWS and small.category != big.category:
                    continue  # an established piece of furniture is not a part of something else (a sideboard
                    # in front of a wall piece misread as furniture)
                if small.voxels is not None:
                    big.voxels = self._union(big.voxels, small.voxels)
                for cat, n in small.category_counts.items():
                    big.category_counts[cat] = big.category_counts.get(cat, 0.0) + n
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

    def _new_cloud_track(self, m: dict, cam_p: np.ndarray) -> Track:
        track = Track(id=f"e:{next(self._ids)}", category_counts=dict(m["probs"]), pos=m["centre"].copy(),
                      var=np.full(3, 0.05**2), yaw=m["yaw"], yaw_var=0.1**2, size=m["size"].copy(), observations=0,
                      first_seen=self.stamp, last_seen=self.stamp)
        self.tracks[track.id] = track
        self._fuse_cloud(track, m, cam_p)
        return track

    def _fuse_cloud(self, track: Track, m: dict, cam_p: np.ndarray) -> None:
        seg = m["seg"]
        if not track.held:
            # What the robot holds was identified before the grasp; views of it in the gripper are
            # partly hidden and may mix with what lies around, so they do not relabel it.
            for cat, prob in m["probs"].items():
                track.category_counts[cat] = track.category_counts.get(cat, 0.0) + prob
            name = color_name(seg.color, seg.invalid_frac)
            track.color_counts[name] = track.color_counts.get(name, 0.0) + 1.0
        track.observations += 1
        track.last_seen = self.stamp
        track.existence = min(1.0, track.existence + 0.2 * (1.0 - track.existence))
        track.absent_views.clear()
        rgb = np.asarray(seg.color, float)
        n = track.observations
        track.mean_rgb = rgb if track.mean_rgb is None else track.mean_rgb + (rgb - track.mean_rgb) / n
        track.mean_invalid += (seg.invalid_frac - track.mean_invalid) / n
        track.bottom_seen = track.bottom_seen or seg.bottom_seen
        if track.held:
            return
        track.settled = True
        track.recent.append((np.asarray(m["centre"], float).copy(), np.asarray(m["size"], float).copy()))
        if track.voxels is not None:
            union = np.vstack([_points(track.voxels), seg.points])
            centre, grown, yaw = box_of(union)
            if track.observations > 2 and max(grown[0], grown[1]) > 1.5 * max(track.footprint, 0.05) and track.footprint < 0.3:
                return  # the region would swell an established box: not this object's surface
            if track.category == "wall" and min(grown[0], grown[1]) > 0.35:
                return  # a second wall around a corner, not this wall
            if max(grown[0], grown[1]) > CONVEX_CHECK_M and _footprint_fill(union, centre, grown, yaw) < MIN_FOOTPRINT_FILL:
                return  # the points would outline an L or a hollow (two walls at a corner), which no single
                # box describes: filled in, the concave corner would become a solid cube
        points = seg.points
        if len(points) > self.config.points_per_cloud:
            points = points[self._rng.choice(len(points), self.config.points_per_cloud, replace=False)]
        track.voxels = self._union(track.voxels, _voxelize(points))
        self._rebuild_geometry(track, cam_p, seg.range)

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

    def _resting_level(self, track: Track, centre: np.ndarray, bottom: float) -> float | None:
        """Height of the highest support below ``bottom`` (within 0.12 m) under the track's centre:
        the top of a surface, or the floor of a larger receptacle."""
        best = None
        for other in self.tracks.values():
            if other.id == track.id or other.held or not _inside(other, centre, 0.0):
                continue
            affordances = other.affordances
            levels = []
            if "support-surface" in affordances:
                levels.append(other.top())
            if "receptacle" in affordances and other.footprint > track.footprint:
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
        directions not yet seen; the variance shrinks with views and grows with completion. A small
        object, which every view shows almost whole, takes the median of its recent single-view
        boxes instead: a union of views swells when a merged duplicate or a stray view sits a few
        centimetres off, enough to turn a mug into a bowl."""
        centre, size, yaw = box_of(_points(track.voxels), trim=0.01)
        if max(size[0], size[1]) < SMALL_M and len(track.recent) >= 3:
            centre = np.median([c for c, _ in track.recent], axis=0)
            size = np.median([z for _, z in track.recent], axis=0)
        parent = self._furniture_of(track) if track.category == "drawer" else None
        if parent is not None:
            # A drawer slides square to its furniture: take the furniture's (well-observed) heading.
            k = round(wrap_angle(yaw - parent.yaw) / (math.pi / 2))
            centre, size, yaw = box_at_yaw(_points(track.voxels), parent.yaw + k * math.pi / 2)
        if track.observations > 1:
            yaw, size = _continuous_heading(yaw, size, track.yaw)
        if track.mean_rgb is not None:
            # The accumulated box is the best evidence of what the object is: classify it again,
            # with confidence growing with the number of views behind it.
            probs = classify(size, centre[2] - size[2] / 2, track.mean_rgb, track.mean_invalid,
                             bottom_seen=track.bottom_seen)
            weight = float(min(track.observations, 20))
            track.category_counts = {k: v * weight for k, v in probs.items()}
        proto = prototype(track.category) if track.category_counts else None
        shift = np.zeros(3)
        completion = 0.0
        total = sum(track.category_counts.values())
        if proto is not None and track.category_counts[track.category] / total > 0.6:
            long_side, short_side, height = proto
            axes = [(0, long_side, np.array([math.cos(yaw), math.sin(yaw), 0.0])),
                    (1, short_side, np.array([-math.sin(yaw), math.cos(yaw), 0.0]))]
            away = centre - cam_p
            view = away[:2] / max(float(np.linalg.norm(away[:2])), 1e-6)
            for k, want, axis in axes:
                # Only the extent along the viewing direction hides behind the visible surface; the
                # extent across it is the silhouette and is seen in full.
                if abs(float(view @ axis[:2])) < 0.3:
                    continue
                if size[k] < 0.85 * want:
                    grow = want - size[k]
                    shift += axis * (grow / 2.0) * (1.0 if float(away @ axis) >= 0 else -1.0)
                    size[k] = want
                    completion = max(completion, grow)
            if size[2] < 0.85 * height:
                grow = height - size[2]
                bottom = centre[2] - size[2] / 2
                if bottom < 0.15:
                    shift[2] = height / 2.0 - centre[2]  # floor-standing: the base is on the floor
                else:
                    shift[2] = grow / 2.0  # resting on something: the unseen part is above
                size[2] = height
                completion = max(completion, grow)
        centre = centre + shift
        if not track.bottom_seen and SHAPES.get(track.category, {}).get("floor"):
            # Furniture whose base no view has shown (a table seen from close, a wall behind a
            # counter) stands on the floor below its visible part.
            top = centre[2] + size[2] / 2
            size[2], centre[2] = top, top / 2
        bottom = centre[2] - size[2] / 2
        if size[0] < 0.3 and bottom > 0.15:
            # Something small off the floor rests on something: extend the box down to the nearest
            # surface (or receptacle floor) just below its visible part.
            level = self._resting_level(track, centre, bottom)
            if level is not None and level < bottom:
                size[2] += bottom - level
                centre[2] -= (bottom - level) / 2
        track.pos = centre
        track.size = size
        track.yaw = yaw
        track.completion = completion
        sigma_depth = self.config.pos_sigma_base + self.config.pos_sigma_per_m * 0.5 * rng_m
        views = max(1, min(track.observations, 20))
        var_xy = sigma_depth**2 / views + (completion / 4.0) ** 2 + 0.002**2
        track.var = np.array([var_xy, var_xy, sigma_depth**2 / views + 0.002**2])
        track.yaw_var = 0.05**2 / views

    def _gate(self, track: Track, sigma: float) -> float:
        """Association gate: measurement and track uncertainty, widened for large objects whose
        detected box centre moves with the visible portion."""
        gate = max(self.config.gate_m, 3.0 * math.sqrt(float(np.mean(track.var)) + sigma**2),
                   self.config.size_gate_ratio * float(max(track.size[0], track.size[1])))
        return max(gate, 0.45) if track.held else gate

    def _size_match(self, a: float, b: float) -> bool:
        """Footprints of the same object agree within the detector's size noise; nested bowls do not."""
        return abs(math.log(max(a, 1e-3) / max(b, 1e-3))) <= self.config.size_match_log

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

    def _merge_duplicates(self) -> None:
        """Merge compatible tracks whose centres lie inside each other's uncertainty-expanded
        footprint; the track with more observations keeps its id."""
        ids = sorted(self.tracks, key=lambda t: -self.tracks[t].observations)
        for i, keep_id in enumerate(ids):
            keep = self.tracks.get(keep_id)
            if keep is None or keep.held:
                continue
            for drop_id in ids[i + 1:]:
                drop = self.tracks.get(drop_id)
                if drop is None or drop.held or not compatible(drop.perceived_category, keep.perceived_category):
                    continue
                if not self._size_match(drop.footprint, keep.footprint):
                    continue
                z_tol = max(0.5 * max(keep.size[2], drop.size[2], 0.1), 3.0 * _sigma_z(keep, drop))
                if abs(drop.pos[2] - keep.pos[2]) > z_tol:
                    continue
                if not (_inside(keep, drop.pos, 3.0 * keep.sigma) or _inside(drop, keep.pos, 3.0 * drop.sigma)):
                    continue
                if _flat_pair(keep, drop) and float(np.linalg.norm(drop.pos[:2] - keep.pos[:2])) > FLAT_PAIR_OVERLAP_M * min(
                        min(keep.size[0], keep.size[1]), min(drop.size[0], drop.size[1])):
                    continue  # two flat pieces edge to edge, not one seen twice
                w_keep, w_drop = 1.0 / keep.var, 1.0 / drop.var
                keep.pos = (keep.pos * w_keep + drop.pos * w_drop) / (w_keep + w_drop)
                keep.var = 1.0 / (w_keep + w_drop)
                for cat, n in drop.category_counts.items():
                    keep.category_counts[cat] = keep.category_counts.get(cat, 0.0) + n
                for col, n in drop.color_counts.items():
                    keep.color_counts[col] = keep.color_counts.get(col, 0.0) + n
                keep.observations += drop.observations
                keep.existence = max(keep.existence, drop.existence)
                keep.last_seen = max(keep.last_seen, drop.last_seen)
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
        self.tracks[track.id] = track
        return track

    def _fuse(self, track: Track, det: dict, pos: np.ndarray, yaw: float, sigma: float) -> None:
        track.category_counts[det["category"]] = track.category_counts.get(det["category"], 0.0) + 1.0
        if det.get("color"):
            track.color_counts[det["color"]] = track.color_counts.get(det["color"], 0.0) + 1.0
        track.observations += 1
        track.last_seen = self.stamp
        track.existence = min(1.0, track.existence + 0.2 * (1.0 - track.existence))
        track.absent_views.clear()
        if track.held:
            return
        r = sigma**2
        prior_var = track.var + self.config.process_noise
        gain = prior_var / (prior_var + r)
        track.pos = track.pos + gain * (pos - track.pos)
        track.var = (1.0 - gain) * prior_var
        if abs(wrap_angle(yaw - track.yaw)) > YAW_MOVED:
            track.yaw_var = max(track.yaw_var, YAW_MOVED**2)  # it was turned (tipped out, put down): follow the views
        k_yaw = track.yaw_var / (track.yaw_var + 0.05**2)
        track.yaw = wrap_angle(track.yaw + k_yaw * wrap_angle(yaw - track.yaw))
        track.yaw_var = (1.0 - k_yaw) * track.yaw_var
        n = track.observations
        track.size = track.size + (np.asarray(det["size"], float) - track.size) / n

    def _decay_unseen(self, cam: dict, matched: set[str]) -> None:
        fovy = math.radians(self._camera["intrinsics"]["fovy"])
        aspect = self._camera["intrinsics"]["width"] / self._camera["intrinsics"]["height"]
        half_v = fovy / 2 * 0.8
        half_h = math.atan(math.tan(fovy / 2) * aspect) * 0.8
        inv = pose_inverse(cam)
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in matched or track.held or self._in_closed_drawer(track):
                continue
            local = pose_compose(inv, {"p": track.pos.tolist(), "q": [1, 0, 0, 0]})["p"]
            if not (0.3 < local[0] < self.config.max_range * 0.8):
                continue
            if abs(math.atan2(local[1], local[0])) > half_h or abs(math.atan2(local[2], local[0])) > half_v:
                continue
            track.existence *= self.config.missed_decay
            if track.existence < self.config.drop_below:
                self._drop(track, "faded: not detected where it should be visible")

    def forget_objects(self) -> None:
        """Drop every perceived object except the one in the gripper (used when the kind of sensor
        data changes and objects must be perceived afresh)."""
        held = self.tracks.get(self.held_id) if self.held_id else None
        self.tracks = {held.id: held} if held is not None else {}
        self.version += 1

    # ---------------------------------------------------------------- holding
    def mark_held(self, track_id: str) -> str:
        """The gripper closed on ``track_id``: carry that track with it from now on. If the track was
        merged into another while the robot approached, the track nearest the gripper (within
        ``HELD_FALLBACK_M``) is the one it holds.

        :returns: the id of the held track.
        :raises KeyError: no track is near the gripper."""
        if track_id not in self.tracks:
            grip = self.gripper_position()
            near = min(self.tracks.values(), key=lambda t: float(np.linalg.norm(t.pos - grip)), default=None)
            if near is None or float(np.linalg.norm(near.pos - grip)) > HELD_FALLBACK_M:
                raise KeyError(f"{track_id} is no longer in the ego state and nothing else is in the gripper")
            track_id = near.id
        track = self.tracks[track_id]
        track.held = True
        track.held_offset = track.pos - self.gripper_position()
        track.held_yaw = wrap_angle(track.yaw - self.odometry["yaw"] - self.joints.get("wrist_yaw", 0.0))
        track.recent.clear()  # it will be put down elsewhere
        self.held_id = track_id
        return track_id

    def remember_face(self, track_id: str, rgb: np.ndarray, stamp: float) -> Appearance | None:
        """Keep one view of a track's top face, sampled on its appearance grid (see
        :meth:`face_grid`): seen cells replace the remembered ones; a stale appearance, or one laid
        out for a clearly different extent, is replaced whole.

        :returns: the track's appearance, or None when the track is gone or held."""
        track = self.tracks.get(track_id)
        if track is None or track.held:
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

    def _clear_held(self) -> None:
        if self.held_id and self.held_id in self.tracks:
            track = self.tracks[self.held_id]
            if track.appearance is not None:
                track.appearance.stale = True  # it may have been put down turned over
            track.held = False
            track.held_offset = None
            track.voxels = None
            track.settled = False  # it drops a little when released; the next view places it
            track.var = np.maximum(track.var, 0.02**2)
            track.yaw_var = max(track.yaw_var, YAW_MOVED**2)
        self.held_id = None

    def _follow_held(self) -> None:
        if self.held_id and self.held_id in self.tracks:
            track = self.tracks[self.held_id]
            track.pos = self.gripper_position() + track.held_offset
            track.yaw = wrap_angle(self.odometry["yaw"] + self.joints.get("wrist_yaw", 0.0) + track.held_yaw)

    # ------------------------------------------------------------------ output
    def support_of(self, track: Track) -> Track | None:
        """What holds a track: the innermost receptacle (drawer, larger bowl) whose floor it rests
        on, else the highest support surface whose top it rests on."""
        inside, best = None, None
        for other in self.tracks.values():
            if other.id == track.id or other.held:
                continue
            affordances = other.affordances
            z_tol = max(0.06, 2.5 * _sigma_z(track, other))
            local = _to_local_xy(other, track.pos)
            if "receptacle" in affordances and track.footprint < other.footprint:
                within = abs(local[0]) <= other.size[0] / 2 and abs(local[1]) <= other.size[1] / 2
                if within and -0.03 <= track.bottom() - other.bottom() <= min(z_tol, 0.5 * other.size[2]):
                    if inside is None or other.bottom() > inside.bottom():
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
        holder = self.support_of(track)
        return holder is not None and holder.category == "drawer" and self.drawer_opening(holder) < 0.08

    def drawer_opening(self, drawer: Track) -> float:
        """How far a drawer sticks out of the furniture holding it, along the drawer's front axis."""
        furniture = [_box2d(t) for t in self.tracks.values()
                     if t.id != drawer.id and "support-surface" in t.affordances]
        return protrusion(self.front(drawer), furniture)

    def front(self, track: Track) -> Box2D:
        """The track's footprint with +x pointing out of its front (towards the room centre)."""
        return front_box(_box2d(track), room_centre([_box2d(t) for t in self.tracks.values()]))

    def shift(self, track_id: str, delta: np.ndarray) -> list[str]:
        """Move a track the robot itself moved (a drawer it pulled) together with what it holds."""
        track = self.tracks.get(track_id)
        if track is None:
            return []
        carried = [t.id for t in self.tracks.values() if t.id != track_id and not t.held and self.support_of(t) is track]
        for tid in [track_id, *carried]:
            moved = self.tracks[tid]
            moved.pos = moved.pos + np.asarray(delta, float)
            if moved.voxels is not None:
                moved.voxels = _voxelize(_points(moved.voxels) + np.asarray(delta, float))
        self.version += 1
        return carried

    def to_scene(self) -> SceneState:
        entities = [self._self_entity()]
        relations: list[Relation] = []
        for track in self.tracks.values():
            entities.append(self._track_entity(track))
            if track.held:
                relations.append(Relation(type="attached", a={"entity": track.id}, b={"entity": SELF_ID, "link": "gripper"},
                                          mode="rigid", derived=False, confidence=1.0))
            elif prior(track.category)["root"].get("fixed", 0.0) < 0.9:
                support = self.support_of(track)
                if support is not None:
                    relations.append(Relation(type="supported-by", a={"entity": track.id}, b={"entity": support.id},
                                              derived=True, confidence=round(min(track.existence, support.existence), 3)))
        return SceneState(mode="ego", entities=entities, relations=relations, version=self.version, stamp=self.stamp,
                          self_id=SELF_ID)

    def _self_entity(self) -> Entity:
        ent = Entity.from_dict(self._self_template)
        o = self.odometry
        pos_var = o["sigma"]["pos"] ** 2
        yaw_var = o["sigma"]["yaw"] ** 2
        ent.pose = Est(value=self.base_pose(), unc=_gaussian6([pos_var, pos_var, 1e-8, 1e-8, 1e-8, yaw_var]),
                       t=self.stamp, src="observed")
        tree: RigidTree = ent.body
        for joint in tree.joints:
            if joint.id in self.joints:
                joint.q = Est(value=[self.joints[joint.id]], unc={"kind": "gaussian", "cov": [1e-6]}, t=self.stamp,
                              src="observed")
        ent.state_vars["gripper"] = Est(value=self.gripper["state"], t=self.stamp, src="observed")
        ent.state_vars["holding"] = Est(value=self.gripper.get("holding", False), t=self.stamp, src="observed")
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

    def _track_entity(self, track: Track) -> Entity:
        p = prior(track.category)
        total = sum(track.category_counts.values())
        cat_unc = {"kind": "categorical", "p": {track.label: 1.0} if track.label else
                   {k: round(v / total, 4) for k, v in track.category_counts.items()}}
        src = "fused" if track.observations > 1 else "observed"
        inflate = 0.0 if track.held else 2.0 * track.sigma
        size = [float(track.size[0] + inflate), float(track.size[1] + inflate), float(track.size[2])]
        link = Link(
            id="body", name=track.category,
            geometry={"collision": [primitive("box", size)]},
            material=Est(value={"fragility": p["fragility"]}, src="prior"),
            affordances=[Est(value=a, src="prior" if track.annotated_affordances is None else "inferred")
                         for a in track.affordances],
        )
        root_value = max(p["root"], key=p["root"].get)
        return Entity(
            id=track.id,
            role=p["role"],
            category=Est(value=track.category, unc=cat_unc, t=track.last_seen, src=src),
            pose=Est(value={"p": track.pos.tolist(), "q": quat_from_yaw(track.yaw)},
                     unc=_gaussian6([*track.var.tolist(), 1e-4, 1e-4, track.yaw_var]), t=track.last_seen, src=src),
            body=RigidTree(root=Est(value=root_value, unc={"kind": "categorical", "p": p["root"]}, src="prior"),
                           links=[link]),
            existence=round(track.existence, 4),
            state_vars={"color": Est(value=track.color, src=src)} if track.color else {},
        )


_COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.15, 0.70, 0.25), "blue": (0.15, 0.35, 0.90),
           "black": (0.06, 0.06, 0.07), "white": (0.95, 0.95, 0.96), "gray": (0.60, 0.62, 0.65),
           "brown": (0.58, 0.40, 0.24), "beige": (0.80, 0.67, 0.50), "yellow": (0.90, 0.85, 0.20)}


ACHROMATIC_SATURATION = 0.18
_CHROMATIC = {k: v for k, v in _COLORS.items() if k not in ("black", "white", "gray")}


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
ESTABLISHED_VIEWS = 5
HELD_FALLBACK_M = 0.1
YAW_MOVED = 0.3  # a detected heading this far off a track's means the object was turned
STACK_TOL = 0.005  # a region this little below another's top face still rests on it
IDENTIFIED_SHARE = 0.6
HELD_LARGER = 2.0  # a region with a footprint this many times the held object's is not the held object
CONVEX_CHECK_M = 0.8  # boxes longer than this are checked for being filled rather than an outline
MIN_FOOTPRINT_FILL = 0.2
SMALL_M = 0.3
RECENT_VIEWS = 12
APPEARANCE_SIZE_CHANGE = 0.2  # an extent this much off the one a face grid was laid out for starts a new grid
SQUARE_SIDES = 0.9  # a footprint whose shorter side is at least this share of the longer is square


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
    return track.observations >= ESTABLISHED_VIEWS and category != "unknown" and total > 0 \
        and track.category_counts[category] / total >= IDENTIFIED_SHARE


def _concave(points: np.ndarray, centre: np.ndarray, footprint: float, cam_p: np.ndarray) -> bool:
    """Whether the box drawn around a large region is seen through: the surfaces the camera sees in
    the direction of the box centre lie beyond that centre. A solid block shows its near face there,
    in front of the centre; the corner where two walls meet, boxed as if it were a block, shows the
    walls behind it, because the box's inside is open room."""
    if footprint < CONCAVE_MIN_M or len(points) < 10:
        return False
    to_centre = centre[:2] - cam_p[:2]
    middle = float(np.linalg.norm(to_centre))
    if middle < 1e-3:
        return False
    rel = points[:, :2] - cam_p[:2]
    bearing = np.degrees(np.abs(np.arctan2(rel[:, 1], rel[:, 0]) - math.atan2(to_centre[1], to_centre[0])))
    bearing = np.minimum(bearing, 360.0 - bearing)
    along = np.linalg.norm(rel[bearing <= SEE_THROUGH_DEG], axis=1)
    return len(along) >= 5 and float(np.median(along)) > middle + SEE_THROUGH_M


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
