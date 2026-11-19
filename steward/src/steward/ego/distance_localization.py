"""CPU localization from synchronized semantic masks and depth; independent of Ego-State fusion.

The map stores measured vertical surfaces in separate height bands on an expanding planar grid.
Rejected observations cannot change the map or map-from-odometry transform. All poses use metres
and radians; this bounded local map has no loop closure or global relocalization.
"""

from __future__ import annotations

import math
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates
from scipy.optimize import least_squares

from steward.ego.estimator import EgoEstimator, _in_boxes
from steward.math3d import matrix_from_quat
from steward.ego.pose_ekf import EkfConfig, PlanarPoseEKF


@dataclass(frozen=True)
class LocalizationConfig:
    """Validated local-map, semantic selection and optimization settings; disabled unless supplied."""

    camera: str = "wide_camera"
    categories: tuple[str, ...] = ("wall", "cabinet", "counter", "sideboard", "table")
    semantic_score: float = 0.5
    erosion_pixels: int = 2
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0
    height_edges_m: tuple[float, ...] = (0.2, 0.7, 1.3, 2.3)
    max_normal_z: float = 0.5
    sample_cell_m: float = 0.06
    points_per_region: int = 180
    max_points: int = 1000
    min_points: int = 40
    resolution_m: float = 0.05
    map_size_m: float = 20.0
    dynamic_map: bool = False
    signed_walls: bool = False
    clear_observations: int = 0
    clear_surface_margin_m: float = 0.025
    clear_height_tolerance_m: float = 0.075
    capture_distance_m: float = 0.35
    inlier_distance_m: float = 0.10
    min_overlap: float = 0.45
    min_inlier_share: float = 0.65
    max_rmse_m: float = 0.08
    translation_bound_m: float = 0.20
    yaw_bound_rad: float = 0.12
    prior_position_sigma_m: float = 0.15
    prior_yaw_sigma_rad: float = 0.10
    robust_scale_m: float = 0.05
    max_evaluations: int = 30
    derivative_step: float = 0.01
    min_information_ratio: float = 0.05
    insert_translation_m: float = 0.10
    insert_yaw_rad: float = 0.10
    stale_after_s: float = 2.0
    ekf: EkfConfig = field(default_factory=EkfConfig)

    def __post_init__(self):
        if not isinstance(self.ekf, EkfConfig):
            raise ValueError("ekf must be an EkfConfig")
        if not self.camera or not self.categories or not all(isinstance(c, str) and c for c in self.categories):
            raise ValueError("camera and categories must be nonempty")
        if type(self.dynamic_map) is not bool:
            raise ValueError("dynamic_map must be boolean")
        if type(self.signed_walls) is not bool:
            raise ValueError("signed_walls must be boolean")
        if type(self.clear_observations) is not int or not 0 <= self.clear_observations <= 255:
            raise ValueError("clear_observations must be an integer between 0 and 255")
        integers = ("erosion_pixels", "points_per_region", "max_points", "min_points", "max_evaluations")
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in asdict(self).items():
            if name in ("ekf", "dynamic_map", "signed_walls", "clear_observations", "camera", "categories", "height_edges_m") or name in integers:
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("semantic_score", "max_normal_z", "min_overlap", "min_inlier_share", "min_information_ratio"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be at most one")
        edges = self.height_edges_m
        if len(edges) < 2 or not all(math.isfinite(v) for v in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("height_edges_m must be finite and strictly increasing")
        if not (self.min_depth_m < self.max_depth_m and self.min_points <= self.max_points
                and max(self.inlier_distance_m, self.clear_surface_margin_m) < self.capture_distance_m
                and self.map_size_m / self.resolution_m <= 1024):
            raise ValueError("inconsistent depth, point, matching or map limits")

    @classmethod
    def load(cls, path: Path) -> LocalizationConfig:
        """Read one flat TOML configuration; unknown keys and invalid values fail at startup."""
        values = tomllib.loads(path.read_text(encoding="utf-8"))
        values["ekf"] = EkfConfig(**values.get("ekf", {}))
        for key in ("categories", "height_edges_m"):
            if key in values:
                if not isinstance(values[key], list):
                    raise ValueError(f"{key} must be an array")
                values[key] = tuple(values[key])
        return cls(**values)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _transform(points, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def _compose(a, b):
    return np.r_[_transform(np.asarray([b[:2]]), a)[0], _wrap(a[2] + b[2])]


def _inverse(a):
    return np.r_[_transform(np.asarray([-a[:2]]), [0, 0, -a[2]])[0], -a[2]]


class SemanticPointSelector:
    """Select measured surfaces only from labelled masks, using same-frame body kinematics."""

    def __init__(self, robot_model: dict, config: LocalizationConfig):
        self.config = config
        self.body = EgoEstimator(robot_model=robot_model)
        if config.camera not in self.body.cameras:
            raise ValueError(f"robot has no camera {config.camera}")

    def select(self, frame: dict, *, include_wall_normals: bool = False) -> tuple[np.ndarray, dict]:
        """Return N×3 base-frame points and selection counts; optionally append two normal columns.

        Anonymous simulator masks and unlabelled depth fills are never treated as semantic evidence.
        The caller supplies labels for this frame.
        """
        cfg = self.config
        empty = np.empty((0, 5 if include_wall_normals else 3))
        if frame["camera"] != cfg.camera:
            raise ValueError("localization frame camera differs from configuration")
        mask, labels, depth = frame.get("mask"), frame.get("labels"), frame["depth"]
        if mask is None or not labels:
            return empty, {"regions": 0, "points": 0, "reason": "missing_semantic_masks"}
        if depth.ndim != 2 or mask.shape != depth.shape or not np.isfinite(frame["fovy"]) or not 0 < frame["fovy"] < 180:
            raise ValueError("invalid depth/mask dimensions or field of view")
        self.body.update_body(frame["header"])
        cam = self.body.camera_pose(odometry={"x": 0, "y": 0, "yaw": 0}, camera=frame["camera"])
        h, w = depth.shape
        f = h / (2 * math.tan(math.radians(frame["fovy"]) / 2))
        v, u = np.mgrid[:h, :w]
        cloud = np.stack([depth, -(u - (w - 1) / 2) * depth / f,
                          -(v - (h - 1) / 2) * depth / f], axis=-1)
        cloud = cloud @ matrix_from_quat(cam["q"]).T + cam["p"]
        valid = np.isfinite(depth) & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            normal = np.cross(cloud[1:-1, 2:] - cloud[1:-1, :-2], cloud[2:, 1:-1] - cloud[:-2, 1:-1])
            length = np.linalg.norm(normal, axis=-1)
            vertical = np.zeros(depth.shape, bool)
            vertical[1:-1, 1:-1] = (length > 1e-10) & (np.abs(normal[..., 2]) <= cfg.max_normal_z * length)
        valid &= vertical & (cloud[..., 2] >= cfg.height_edges_m[0]) & (cloud[..., 2] < cfg.height_edges_m[-1])
        regions = []
        for instance, scores in sorted(labels.items()):
            # An excluded winning label (person, robot, movable object) wins over a weak wall label.
            category = max(scores, key=scores.get) if scores else None
            if category not in cfg.categories or scores[category] < cfg.semantic_score:
                continue
            interior = binary_erosion(mask == int(instance), iterations=cfg.erosion_pixels)
            selected = interior & valid
            points = cloud[selected]
            if not len(points):
                continue
            if include_wall_normals:
                inward = np.zeros((len(points), 2))
                if category == "wall":
                    inward = normal[selected[1:-1, 1:-1], :2]
                    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
                    toward_camera = np.sum(inward * (np.asarray(cam["p"])[:2] - points[:, :2]), axis=1)
                    inward *= np.sign(toward_camera)[:, np.newaxis]
            # Remove the whole body, including gripper vicinity; held regions get no exemption.
            raw = frame["header"]["odometry"]
            world = points.copy()
            world[:, :2] = _transform(points[:, :2], [raw["x"], raw["y"], raw["yaw"]])
            visible = ~_in_boxes(world, self.body.body_boxes(), self.body.config.body_margin_m)
            points = points[visible]
            if not len(points):
                continue
            _, indices = np.unique(np.floor(points / cfg.sample_cell_m).astype(int), axis=0, return_index=True)
            points = points[indices]
            keep = np.linspace(0, len(points) - 1, min(len(points), cfg.points_per_region), dtype=int)
            kept = points[keep]
            if include_wall_normals:
                kept = np.column_stack((kept, inward[visible][indices][keep]))
            regions.append(kept)
        if not regions:
            return empty, {"regions": 0, "points": 0, "reason": "no_static_surfaces"}
        # Round-robin equalizes per-region contribution before applying the total point limit.
        points = np.asarray([region[i] for i in range(max(map(len, regions))) for region in regions if i < len(region)])
        points = points[:cfg.max_points]
        return points, {"regions": len(regions), "points": len(points), "reason": "selected"}


class DistanceFieldLocalizer:
    """Local surface map with odometry prediction and gated SE(2) correction.

    Single-owner use only. Episode changes reset all state; timestamps must strictly increase
    inside an episode. Rejections preserve the map and correction, while prediction continues.
    """

    def __init__(self, config: LocalizationConfig | None = None):
        self.config = config or LocalizationConfig()
        self.reset()

    def reset(self):
        """Discard map, timestamps and map-from-odometry correction for a new episode."""
        self.episode = None
        self.stamp = None
        self.correction = np.zeros(3)
        self.origin = None
        self._grid_anchor = None
        self._grid_offset = np.zeros(2, dtype=int)
        self.occupied = None
        self.fields = None
        self._wall_votes = None
        self._free_observations = self._hit_heights = None
        self.map_cleared_cells = 0
        self.last_insert = None
        self.last_correction_stamp = None
        self.map_version = 0
        self.filter = PlanarPoseEKF(self.config.ekf) if self.config.ekf.enabled else None

    def _distances(self, points, bands, pose):
        xy = (_transform(points, pose) - self._grid_anchor) / self.config.resolution_m - self._grid_offset
        result = np.full(len(points), self.config.capture_distance_m)
        for band in np.unique(bands):
            take = bands == band
            result[take] = map_coordinates(self.fields[band], xy[take].T, order=1, mode="constant",
                                           cval=self.config.capture_distance_m, prefilter=False)
        return result

    def _insert(self, points, bands, pose, wall_normals=None, *, camera_origin=None):
        cfg = self.config
        shape = np.array(self.occupied.shape[1:])
        world = np.column_stack((_transform(points[:, :2], pose), points[:, 2]))
        xy = np.rint((world[:, :2] - self._grid_anchor) / cfg.resolution_m).astype(int) - self._grid_offset
        offset = np.zeros(2, dtype=int)
        if cfg.dynamic_map:
            # Outside this margin a lookup cannot reach any stored surface within capture distance.
            margin = math.ceil(cfg.capture_distance_m / cfg.resolution_m) + 1
            offset = np.minimum(0, xy.min(axis=0) - margin)
            extent = np.maximum(shape, xy.max(axis=0) + margin + 1)
            if (extent - offset > 1025).any():
                return False
        else:
            extent = shape
            inside = ((xy >= 0) & (xy < shape)).all(axis=1)
            xy, bands, world = xy[inside], bands[inside], world[inside]
            if wall_normals is not None:
                wall_normals = wall_normals[inside]
        padding = ((0, 0), *zip(-offset, extent - shape))
        occupied = np.pad(self.occupied, padding)
        wall_votes = None if self._wall_votes is None else np.pad(self._wall_votes, (*padding, (0, 0)))
        indices = xy - offset
        hits = (bands, indices[:, 0], indices[:, 1])
        grid_offset = self._grid_offset + offset
        origin = self._grid_anchor + grid_offset * cfg.resolution_m
        cleared = free = heights = None
        if cfg.clear_observations:
            free = (np.zeros_like(occupied, dtype=np.uint8) if self._free_observations is None
                    else np.pad(self._free_observations, padding))
            heights = (np.zeros_like(occupied, dtype=np.float32) if self._hit_heights is None
                       else np.pad(self._hit_heights, padding))
            if camera_origin is not None:
                eye = np.r_[_transform(camera_origin[np.newaxis, :2], pose)[0], camera_origin[2]]
                visible = self._visible_free_cells(world, eye, origin, occupied, heights)
                free[visible] += 1
            free[hits] = 0
            cleared = free >= cfg.clear_observations
            occupied[cleared] = False
            free[cleared] = 0
            heights[cleared] = 0
            flat, group, counts = np.unique(np.ravel_multi_index(hits, occupied.shape),
                                            return_inverse=True, return_counts=True)
            heights.flat[flat] = np.bincount(group, weights=world[:, 2]) / counts
        occupied[hits] = True
        if cfg.signed_walls:
            if wall_votes is None:
                wall_votes = np.zeros((*occupied.shape, 3))
            if cleared is not None:
                wall_votes[cleared] = 0
            if wall_normals is not None:
                c, s = np.cos(pose[2]), np.sin(pose[2])
                normals = wall_normals @ np.array([[c, s], [-s, c]])
                votes = np.column_stack((normals, np.linalg.norm(normals, axis=1)))
                np.add.at(wall_votes, hits, votes)
        fields = np.asarray([self._field(layer, None if wall_votes is None else wall_votes[i])
                             for i, layer in enumerate(occupied)])
        self._grid_offset, self.origin = grid_offset, origin
        self.occupied, self.fields, self._wall_votes = occupied, fields, wall_votes
        self._free_observations, self._hit_heights = free, heights
        if cleared is not None:
            self.map_cleared_cells += int(cleared.sum())
        self.last_insert = pose.copy()
        self.map_version += 1
        return True

    def _visible_free_cells(self, endpoints, eye, origin, occupied, heights):
        """Return distinct occupied cells crossed near measured surfaces, at their observed heights."""
        cfg = self.config
        direction = eye - endpoints
        ranges = np.linalg.norm(direction, axis=1)
        valid = ranges > cfg.clear_surface_margin_m
        direction = direction[valid] / ranges[valid, np.newaxis]
        steps = np.arange(cfg.clear_surface_margin_m, cfg.capture_distance_m, cfg.resolution_m / 2)
        rays = endpoints[valid, np.newaxis, :] + direction[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
        rays = rays[steps < ranges[valid, np.newaxis]]
        xy = np.rint((rays[:, :2] - origin) / cfg.resolution_m).astype(int)
        bands = np.searchsorted(cfg.height_edges_m, rays[:, 2], side="right") - 1
        inside = (((xy >= 0) & (xy < occupied.shape[1:])).all(axis=1)
                  & (bands >= 0) & (bands < occupied.shape[0]))
        cells = (bands[inside], xy[inside, 0], xy[inside, 1])
        supported = occupied[cells] & (np.abs(heights[cells] - rays[inside, 2]) <= cfg.clear_height_tolerance_m)
        visible = np.zeros_like(occupied)
        visible[tuple(axis[supported] for axis in cells)] = True
        return visible

    def _field(self, occupied, wall_votes):
        cfg = self.config
        if not occupied.any():
            return np.full(occupied.shape, cfg.capture_distance_m)
        if wall_votes is None:
            distance = distance_transform_edt(~occupied, sampling=cfg.resolution_m)
        else:
            distance, nearest = distance_transform_edt(~occupied, sampling=cfg.resolution_m, return_indices=True)
            votes = wall_votes[tuple(nearest)]
            normals, counts = votes[..., :2], votes[..., 2]
            offset = np.moveaxis(np.indices(occupied.shape) - nearest, 0, -1)
            behind = np.sum(offset * normals, axis=2) < 0
            agreed = np.linalg.norm(normals, axis=2) > counts / 2
            distance[behind & agreed] *= -1
        return np.clip(distance, -cfg.capture_distance_m, cfg.capture_distance_m)

    def update(self, raw_pose, stamp: float, episode: int | str, points: np.ndarray | None = None,
               *, wall_normals: np.ndarray | None = None, camera_origin: np.ndarray | None = None) -> dict:
        """Predict from raw SE(2) odometry, optionally correct against historical segmented points.

        :param points: N×3 robot-base measured points, already selected using synchronized semantics.
        :param wall_normals: N×2 camera-facing wall normals in the base frame. Zero rows carry no side evidence.
        :param camera_origin: Base-frame camera centre used to check visibility of old cells.
        :returns: pose, state/reason, map version and per-attempt metrics; never fabricated tracking.
        """
        started = time.perf_counter()
        raw = np.asarray(raw_pose, float)
        pts = np.empty((0, 3)) if points is None else np.asarray(points, float)
        if raw.shape != (3,) or not np.isfinite(raw).all() or not math.isfinite(stamp):
            raise ValueError("raw pose and stamp must be finite")
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("points must be a finite N by 3 array")
        if wall_normals is not None and (wall_normals.shape != (len(pts), 2) or not np.isfinite(wall_normals).all()):
            raise ValueError("wall_normals must contain one finite planar normal per point")
        if camera_origin is not None and (camera_origin.shape != (3,) or not np.isfinite(camera_origin).all()):
            raise ValueError("camera_origin must be a finite 3D vector")
        if self.stamp is not None and episode == self.episode and stamp <= self.stamp:
            raise ValueError("timestamps must strictly increase within an episode")
        if self.stamp is not None and episode != self.episode:
            self.reset()
        self.episode, self.stamp = episode, stamp
        cfg = self.config
        pose = self.filter.predict(raw, stamp) if self.filter is not None else _compose(self.correction, raw)
        valid = (pts[:, 2] >= cfg.height_edges_m[0]) & (pts[:, 2] < cfg.height_edges_m[-1])
        pts = pts[valid]
        if wall_normals is not None:
            wall_normals = wall_normals[valid]
        bands = np.searchsorted(cfg.height_edges_m, pts[:, 2], side="right") - 1
        points = pts[:, :2]
        accepted, state, reason, metrics = False, "predicted", "insufficient_points", {}
        if len(pts) >= cfg.min_points:
            if self.last_insert is None:
                size = math.ceil(cfg.map_size_m / cfg.resolution_m) + 1
                self._grid_anchor = pose[:2] - cfg.map_size_m / 2
                self.occupied = np.zeros((len(cfg.height_edges_m) - 1, size, size), bool)
                if self._insert(pts, bands, pose, wall_normals, camera_origin=camera_origin):
                    state, reason = "initializing", "map_seeded"
                else:
                    reason = "map_grid_limit"
            else:
                pose, accepted, reason, metrics = self._match(points, bands, pose)
                if accepted and self.filter is not None:
                    metrics["candidatePose"] = pose.tolist()
                    accepted, reason, fusion = self.filter.correct(pose, metrics["visualCovariance"])
                    metrics.update(fusion)
                    pose = self.filter.pose.copy()
                if accepted:
                    self.correction = _compose(pose, _inverse(raw))
                    self.last_correction_stamp = stamp
                    state = "corrected"
                    if (np.linalg.norm(pose[:2] - self.last_insert[:2]) >= cfg.insert_translation_m
                            or abs(_wrap(pose[2] - self.last_insert[2])) >= cfg.insert_yaw_rad):
                        if not self._insert(pts, bands, pose, wall_normals, camera_origin=camera_origin):
                            metrics["mapInsertSkipped"] = "map_grid_limit"
        age = None if self.last_correction_stamp is None else stamp - self.last_correction_stamp
        map_grid = ({} if self.origin is None else
                    {"mapGrid": {"origin": self.origin.tolist(), "shape": list(self.occupied.shape[1:]),
                                 "spacingM": [cfg.resolution_m] * 2}})
        return {"stamp": stamp, "episode": episode, "rawPose": raw.tolist(), "pose": pose.tolist(),
                "mapFromOdometry": self.correction.tolist(), "state": state, "reason": reason,
                "accepted": accepted, "mapVersion": self.map_version, "mapClearedCells": self.map_cleared_cells,
                "points": len(pts),
                "secondsSinceCorrection": age, "stale": age is None or age > cfg.stale_after_s,
                "filterCovariance": None if self.filter is None else self.filter.covariance.tolist(),
                "metrics": metrics, "localizationWallSeconds": time.perf_counter() - started, **map_grid}

    def _match(self, points, bands, predicted):
        cfg = self.config
        before = self._distances(points, bands, predicted)
        support = np.abs(before) < cfg.capture_distance_m * 0.95
        overlap = float(support.mean())
        metrics = {"overlap": overlap}
        if support.sum() < cfg.min_points or overlap < cfg.min_overlap:
            return predicted, False, "insufficient_overlap", metrics
        points = points[support]
        bands = bands[support]
        # This point set stays fixed for every trial pose; moving outside the map costs distance.
        def geometry(delta):
            return self._distances(points, bands, predicted + delta)

        sigmas = np.array([cfg.prior_position_sigma_m, cfg.prior_position_sigma_m, cfg.prior_yaw_sigma_rad])
        def residual(delta):
            if cfg.ekf.enabled:
                return geometry(delta)
            return np.r_[geometry(delta), delta / sigmas * cfg.robust_scale_m * math.sqrt(len(points))]

        bounds = np.array([cfg.translation_bound_m, cfg.translation_bound_m, cfg.yaw_bound_rad])
        fit = least_squares(residual, np.zeros(3), bounds=(-bounds, bounds), loss="soft_l1",
                            f_scale=cfg.robust_scale_m, max_nfev=cfg.max_evaluations,
                            diff_step=cfg.derivative_step)
        after = geometry(fit.x)
        step = cfg.derivative_step
        raw_jac = np.column_stack([(geometry(fit.x + np.eye(3)[i] * step)
                                - geometry(fit.x - np.eye(3)[i] * step)) / (2 * step) for i in range(3)])
        jac = raw_jac.copy()
        # Scale angular information by the point lever arm so the ratio has comparable units.
        jac[:, 2] /= max(1.0, float(np.sqrt(np.mean(np.sum(points[:, :2]**2, axis=1)))))
        inlier = np.abs(after) <= cfg.inlier_distance_m
        information = np.linalg.eigvalsh(jac[inlier].T @ jac[inlier]) / max(1, int(inlier.sum()))
        ratio = float(information[0] / max(information[-1], 1e-12))
        rmse = float(np.sqrt(np.mean(after**2)))
        metrics.update(rmseM=rmse, initialRmseM=float(np.sqrt(np.mean(before[support]**2))),
                       inlierShare=float(inlier.mean()), informationRatio=ratio,
                       informationEigenvalues=information.tolist(), delta=fit.x.tolist(), evaluations=fit.nfev)
        if not fit.success:
            reason = "optimizer_limit"
        elif np.any(np.abs(fit.x) >= bounds * 0.98):
            reason = "correction_at_limit"
        elif inlier.mean() < cfg.min_inlier_share or rmse > cfg.max_rmse_m:
            reason = "poor_alignment"
        elif ratio < cfg.min_information_ratio:
            reason = "degenerate_geometry"
        elif rmse > metrics["initialRmseM"] + 1e-6:
            reason = "alignment_worsened"
        else:
            pose = predicted + fit.x
            pose[2] = _wrap(pose[2])
            if cfg.ekf.enabled:
                # Geometry only: the matcher contains no EKF prior to count a second time.
                count = int(inlier.sum())
                data_information = raw_jac[inlier].T @ raw_jac[inlier] / count
                values, vectors = np.linalg.eigh(data_information)
                variance = max(rmse**2, cfg.resolution_m**2 / 12)
                effective = min(count, cfg.ekf.visual_effective_points)
                covariance = (vectors * (variance / (np.maximum(values, 1e-12) * effective))) @ vectors.T
                floors = np.array([cfg.ekf.visual_position_floor_m**2 + cfg.ekf.map_position_floor_m**2]*2
                                  + [cfg.ekf.visual_yaw_floor_rad**2 + cfg.ekf.map_yaw_floor_rad**2])
                metrics["visualCovariance"] = (cfg.ekf.visual_covariance_inflation * covariance + np.diag(floors)).tolist()
            return pose, True, "matched", metrics
        return predicted, False, reason, metrics


class SegmentedLocalizer:
    """Reuse already segmented RGB-D frames; no model requests, GT or simulator identities."""

    def __init__(self, robot_model: dict, config: LocalizationConfig):
        self.selector = SemanticPointSelector(robot_model, config)
        self.tracker = DistanceFieldLocalizer(config)

    def predict(self, header: dict) -> dict:
        """Advance continuous odometry output between semantic observations."""
        raw = header["odometry"]
        return self.tracker.update([raw["x"], raw["y"], raw["yaw"]], header["stamp"], header["episode"])

    def observe(self, frame: dict) -> dict:
        """Correct using this frame's labels, mask, depth and body header."""
        started = time.perf_counter()
        signed = self.selector.config.signed_walls
        points, selection = self.selector.select(frame, include_wall_normals=signed)
        camera_origin = None
        if len(points) and self.selector.config.clear_observations:
            camera = self.selector.body.camera_pose(odometry={"x": 0, "y": 0, "yaw": 0}, camera=frame["camera"])
            camera_origin = np.asarray(camera["p"])
        selection_wall = time.perf_counter() - started
        header, raw = frame["header"], frame["header"]["odometry"]
        result = self.tracker.update([raw["x"], raw["y"], raw["yaw"]], header["stamp"], header["episode"],
                                     points[:, :3], wall_normals=points[:, 3:] if signed else None,
                                     camera_origin=camera_origin)
        return result | {"selection": selection, "selectionWallSeconds": selection_wall}
