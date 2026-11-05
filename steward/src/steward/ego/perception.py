"""Robot-side perception for RGB-D frames: segments → 3D points → geometry and category.

The world only provides raw sensor data: an RGB image, a depth image and anonymous instance masks
(as a class-agnostic segmenter would produce). This module back-projects every masked region into
the robot's ``map`` frame using the robot's own camera pose estimate, measures the visible geometry,
and classifies it against the robot's shape priors. It never sees World State identifiers.

Camera convention: the sensor mount frame looks along +x with +z up; pixel (u, v) at depth d maps
to ``[d, -(u - cx) d / f, -(v - cy) d / f]`` in that frame, with ``f = (H / 2) / tan(fovy / 2)``.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass

import numpy as np

from steward.math3d import matrix_from_quat


@dataclass
class PerceptionConfig:
    max_range: float = 6.0
    min_points: int = 20
    max_points: int = 400
    floor_cut: float = 0.012
    occluder_gap: float = 0.15  # something this much nearer right below a region hides its base
    overhang_gap: float = 0.15  # something this much farther right below a region: its lower edge overhangs
    occluded_share: float = 0.3  # share of the lower edge hidden or overhanging that leaves the base unseen
    support_gap: float = 0.1  # something at most this much nearer or farther right below a region supports it
    supported_share: float = 0.4  # share of the lower edge on a support that shows a base despite any overhang


@dataclass
class Segment:
    """One masked region of a frame, in the robot's map frame."""

    points: np.ndarray  # (N, 3)
    color: np.ndarray  # mean RGB in [0, 1]
    pixels: int
    invalid_frac: float  # share of masked pixels without a depth reading (glass, far)
    border: bool  # the region touches the image border (truncated view)
    range: float  # median distance from the camera
    bottom_seen: bool = True  # the lowest points are the object's base, not an image edge or an occluder
    semantics: dict[str, float] | None = None  # category → segmenter score, when the robot segmented the image
    span: np.ndarray | None = None  # eight corners of the box around every valid pixel, before the point sample


def decode_rgbd(blob: bytes) -> dict:
    """Unpack the robot port's ``/sensors/rgbd`` payload; ``camera`` is the sensor id it was taken
    with, ``mask`` is None when the world sends no instance masks and the robot segments the image
    itself."""
    z = np.load(io.BytesIO(blob))
    return {"camera": str(z["camera"]) if "camera" in z.files else None, "rgb": z["rgb"], "depth": z["depth"].astype(np.float32) / 1000.0,
            "mask": z["mask"] if "mask" in z.files else None,
            "fovy": float(z["fovy"]), "header": json.loads(str(z["header"]))}


def segments(frame: dict, cam_pose: dict, config: PerceptionConfig | None = None,
             rng: np.random.Generator | None = None) -> list[Segment]:
    """Back-project every masked region of an RGB-D frame into map-frame points. With
    ``frame["labels"]`` (instance id → category scores, from the robot's segmenter) each segment
    carries its scores as :attr:`Segment.semantics`."""
    cfg = config or PerceptionConfig()
    rng = rng or np.random.default_rng(0)
    depth, mask, rgb = frame["depth"], frame["mask"], frame["rgb"]
    if mask is None:
        raise ValueError("the frame carries no instance masks: segment it first (steward.ego.segmenter)")
    labels = frame.get("labels")
    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rot = matrix_from_quat(cam_pose["q"])
    origin = np.asarray(cam_pose["p"], float)
    out = []
    for instance in np.unique(mask):
        if instance == 0:
            continue
        ys, xs = np.nonzero(mask == instance)
        d = depth[ys, xs]
        valid = (d > 0.05) & (d < cfg.max_range)
        if int(valid.sum()) < cfg.min_points:
            continue
        u, v, dv = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid].astype(np.float64)
        local = np.stack([dv, -(u - cx) * dv / f, -(v - cy) * dv / f], axis=1)
        measured = local @ rot.T + origin
        above = measured[:, 2] > cfg.floor_cut
        measured = measured[above]
        if len(measured) < cfg.min_points:
            continue
        semantics = None if labels is None else dict(labels.get(int(instance), {})) or None
        border = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1)
        bottom = _bottom_seen(mask, depth, instance, ys, xs, cfg)
        color = rgb[ys, xs].reshape(-1, 3).mean(axis=0) / 255.0
        invalid = float(1.0 - valid.mean())
        # A segmenter mask sometimes covers two things standing side by side, or a table top and the
        # objects on it. Anonymous world masks stay whole: splitting them would cut one object.
        split = _raised_cluster_indices(measured, cfg.min_points) if labels is not None else None
        if split is None:
            # The stored cloud stays a sample, unchanged from before. The corners are the box of every
            # valid pixel: a wide top's far edge is too few of them to survive the sample.
            span = box_corners(*box_of(measured))
            if len(dv) > cfg.max_points:
                keep = rng.choice(len(dv), cfg.max_points, replace=False)
                u, v, dv = u[keep], v[keep], dv[keep]
            local = np.stack([dv, -(u - cx) * dv / f, -(v - cy) * dv / f], axis=1)
            points = local @ rot.T + origin
            points = points[points[:, 2] > cfg.floor_cut]
            if len(points) < cfg.min_points:
                continue
            out.append(Segment(
                points=points, color=color, pixels=len(ys), invalid_frac=invalid, border=border,
                range=float(np.median(dv)), bottom_seen=bottom, semantics=semantics, span=span,
            ))
            continue
        for idx in split:
            cluster = measured[idx]
            if len(cluster) < cfg.min_points:
                continue
            span = box_corners(*box_of(cluster))
            points = cluster
            if len(points) > cfg.max_points:
                points = points[rng.choice(len(points), cfg.max_points, replace=False)]
            out.append(Segment(
                points=points, color=color, pixels=len(idx), invalid_frac=invalid, border=border,
                range=float(np.median(np.linalg.norm(cluster - origin, axis=1))),
                bottom_seen=bottom, semantics=semantics, span=span,
            ))
    if labels is not None:
        for seg in list(out):
            if not seg.semantics or "picture" not in seg.semantics:
                continue
            body = _box_below_picture(depth, mask, cam_pose, frame["fovy"], seg.points, cfg.min_points)
            if body is None:
                continue
            out.append(Segment(
                points=body, color=seg.color, pixels=len(body), invalid_frac=0.0, border=False,
                range=seg.range, bottom_seen=True, semantics={"box": 0.7},
                span=box_corners(*box_of(body)),
            ))
    return out


def _box_below_picture(depth: np.ndarray, mask: np.ndarray, cam_pose: dict, fovy: float,
                       picture: np.ndarray, min_points: int) -> np.ndarray | None:
    """Points of a shallow box standing under a printed lid.

    The segmenter names the printed top ``picture`` and does not return the walls. Unlabelled
    depth inside the lid's footprint, between 1.2 cm and 9 cm below the lid and spread over
    2.5–8 cm of height, is that wall. The returned cloud uses the lid's outline down to the
    wall, because the lid hides the floor of the box. A flat surface or a tall body is not a box.

    :param depth: (H, W) metres.
    :param mask: instance ids; 0 is unlabelled.
    :param picture: map-frame points of the lid.
    :returns: map-frame points, or ``None`` when no such wall is visible."""
    if len(picture) < min_points:
        return None
    z_lid = float(np.percentile(picture[:, 2], 20))
    xy = picture[:, :2]
    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rot = matrix_from_quat(cam_pose["q"])
    origin = np.asarray(cam_pose["p"], float)
    free = (depth > 0.05) & (depth < 6.0) & (mask == 0)
    ys, xs = np.nonzero(free)
    if len(ys) == 0:
        return None
    d = depth[ys, xs].astype(np.float64)
    local = np.stack([d, -(xs.astype(np.float64) - cx) * d / f, -(ys.astype(np.float64) - cy) * d / f], axis=1)
    world = local @ rot.T + origin
    x0, y0 = xy.min(axis=0) - 0.02
    x1, y1 = xy.max(axis=0) + 0.02
    inside = (world[:, 0] >= x0) & (world[:, 0] <= x1) & (world[:, 1] >= y0) & (world[:, 1] <= y1)
    band = inside & (world[:, 2] < z_lid - 0.012) & (world[:, 2] > z_lid - 0.09)
    wall = world[band]
    if len(wall) < min_points:
        return None
    z_span = float(wall[:, 2].max() - wall[:, 2].min())
    if not 0.025 <= z_span <= 0.08:
        return None
    z0 = float(wall[:, 2].min())
    z1 = z_lid - 0.03  # a voxel below the lid, so the box and the lid do not share a surface
    wall = wall[wall[:, 2] <= z1]
    if len(wall) < min_points or float(wall[:, 2].max() - wall[:, 2].min()) < 0.02:
        return None
    corners = [[x, y, z] for x in (float(xy[:, 0].min()), float(xy[:, 0].max()))
               for y in (float(xy[:, 1].min()), float(xy[:, 1].max())) for z in (z0, z1)]
    return np.vstack([wall, np.asarray(corners, float)])


def _raised_cluster_indices(points: np.ndarray, min_points: int) -> list[np.ndarray] | None:
    """Index arrays into ``points`` when a raised region is several objects, not one.

    A region up to 65 cm across splits on one horizontal gap of at least 5 cm. A piece that also
    contains the table around a shallow bowl keeps only what stands 2 cm above that piece's base.
    A wider raised slab keeps the clusters that stand at least 3.5 cm above the slab. Furniture
    that reaches the floor, and a region wider than 1.2 m, stay whole.

    :param points: map-frame points of one mask, already above the floor cut.
    :param min_points: fewest points each piece must keep.
    :returns: ``None`` when the region is one object."""
    if len(points) < 2 * min_points:
        return None
    if float(np.percentile(points[:, 2], 10)) <= 0.2:
        return None
    span = points.max(axis=0) - points.min(axis=0)
    horiz = max(float(span[0]), float(span[1]))
    if float(span[2]) > 0.35 or horiz > 1.2:
        return None
    if horiz <= 0.65 and float(span[2]) <= 0.3:
        return _two_compact_sides(points, min_points)
    return _bumps_above_slab(points, min_points)


def _two_compact_sides(points: np.ndarray, min_points: int) -> list[np.ndarray] | None:
    """Two index arrays when one gap of at least 5 cm divides ``points`` into two small pieces."""
    centred = points[:, :2] - points[:, :2].mean(axis=0)
    axis = np.linalg.eigh(centred.T @ centred / len(points))[1][:, 1]
    order = np.argsort(centred @ axis)
    gaps = np.diff((centred @ axis)[order])
    if len(gaps) < 8:
        return None
    lo, hi = len(gaps) // 5, len(gaps) - len(gaps) // 5
    if hi <= lo:
        return None
    k = lo + int(np.argmax(gaps[lo:hi]))
    if float(gaps[k]) < 0.05:
        return None
    left, right = order[:k + 1], order[k + 1:]
    sides = []
    for idx in (left, right):
        core = _object_core(points, idx, min_points)
        if core is None:
            return None
        sides.append(core)
    return sides


def _object_core(points: np.ndarray, idx: np.ndarray, min_points: int) -> np.ndarray | None:
    """``idx`` when it is already one small object, or the points that stand above its base.

    A side of a gap can include the table around a shallow bowl. Those table points sit at the
    base; the bowl is what remains more than 3 cm above that base. When that part is still wide,
    the largest cluster within 32 cm is the bowl and the rest of the table is left out."""
    if len(idx) < min_points:
        return None
    chosen = idx
    extent = points[chosen].max(axis=0) - points[chosen].min(axis=0)
    if max(float(extent[0]), float(extent[1])) > 0.25:
        base = float(np.percentile(points[idx, 2], 20))
        chosen = idx[points[idx, 2] > base + 0.03]
        if len(chosen) < min_points:
            return None
        extent = points[chosen].max(axis=0) - points[chosen].min(axis=0)
        if max(float(extent[0]), float(extent[1])) > 0.32:
            compact = _largest_compact(points[chosen], min_points)
            if compact is None:
                return None
            chosen = chosen[compact]
            extent = points[chosen].max(axis=0) - points[chosen].min(axis=0)
    if float(extent[2]) > 0.22:
        return None
    return chosen


def oversized_bowl_core(points: np.ndarray) -> np.ndarray | None:
    """Compact cluster of a bowl cloud that also holds a separate stretch of its support.

    A bowl is at most about 32 cm across. A longer cloud is that bowl plus support voxels that do
    not touch it. One connected slab is left whole: the cluster is kept only when it is actually
    smaller than the cloud it was cut from.

    :param points: map-frame points of one accumulated cloud.
    :returns: the bowl's points, or ``None``."""
    if len(points) < 40:
        return None
    extent = points.max(axis=0) - points.min(axis=0)
    horiz = max(float(extent[0]), float(extent[1]))
    if horiz <= 0.36:
        return None
    compact = _largest_compact(points, 20)
    if compact is None:
        return None
    core = points[compact]
    core_extent = core.max(axis=0) - core.min(axis=0)
    core_horiz = max(float(core_extent[0]), float(core_extent[1]))
    if core_horiz > 0.85 * horiz:
        return None
    return core


def raised_object_points(points: np.ndarray) -> np.ndarray | None:
    """The compact cluster standing on a wide shallow slab, or ``None`` when ``points`` is one object.

    A bowl measured together with the table around it is about half a metre across, with its base
    up on the table. The table is the lower surface; what stands at least 3 cm above that surface
    and stays within 32 cm across is the object. A flat top is left unchanged, and so is a cloud
    whose base is below 0.7 m: a drawer front sits down inside the furniture, not on the top.

    :param points: map-frame points of one accumulated cloud.
    :returns: the object's points, or ``None``."""
    if len(points) < 40:
        return None
    extent = points.max(axis=0) - points.min(axis=0)
    horiz = max(float(extent[0]), float(extent[1]))
    base_z = float(np.percentile(points[:, 2], 10))
    if horiz <= 0.35 or float(extent[2]) > 0.25 or base_z <= 0.70:
        return None
    base = float(np.percentile(points[:, 2], 30))
    raised = points[points[:, 2] > base + 0.03]
    if len(raised) < 20:
        return None
    raised_extent = raised.max(axis=0) - raised.min(axis=0)
    if not 0.04 <= float(raised_extent[2]) <= 0.22:
        return None
    if max(float(raised_extent[0]), float(raised_extent[1])) > 0.32:
        compact = _largest_compact(raised, 20)
        if compact is None:
            return None
        raised = raised[compact]
        raised_extent = raised.max(axis=0) - raised.min(axis=0)
    if float(raised_extent[2]) < 0.04 or max(float(raised_extent[0]), float(raised_extent[1])) > 0.85 * horiz:
        return None
    return raised


def _largest_compact(points: np.ndarray, min_points: int, max_extent: float = 0.32) -> np.ndarray | None:
    """Indices of the largest 3 cm cluster that is at most ``max_extent`` across and at most 22 cm tall."""
    if len(points) < min_points:
        return None
    cell = 0.03
    keys = np.floor(points[:, :2] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault((int(key[0]), int(key[1])), []).append(i)
    seen: set[tuple[int, int]] = set()
    best: np.ndarray | None = None
    for start in buckets:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cells: list[tuple[int, int]] = []
        while stack:
            ix, iy = stack.pop()
            cells.append((ix, iy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (ix + dx, iy + dy)
                    if nxt in buckets and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        idx = np.array([i for c in cells for i in buckets[c]], dtype=np.int64)
        if len(idx) < min_points:
            continue
        extent = points[idx].max(axis=0) - points[idx].min(axis=0)
        if max(float(extent[0]), float(extent[1])) > max_extent or float(extent[2]) > 0.22:
            continue
        if best is None or len(idx) > len(best):
            best = idx
    return best


def _bumps_above_slab(points: np.ndarray, min_points: int) -> list[np.ndarray] | None:
    """Index arrays of the clusters standing above the lower surface of a wide raised region.

    The slab itself is not returned: a table already has its own region. Each cluster has to be
    at least 3 cm tall and no wider than a large bowl, so a ripple in the slab does not qualify.
    """
    support = float(np.percentile(points[:, 2], 40))
    above = np.flatnonzero(points[:, 2] > support + 0.035)
    if len(above) < 2 * min_points:
        return None
    raised = points[above]
    centred = raised[:, :2] - raised[:, :2].mean(axis=0)
    axis = np.linalg.eigh(centred.T @ centred / len(raised))[1][:, 1]
    coord = centred @ axis
    order = np.argsort(coord)
    ordered = coord[order]
    groups: list[np.ndarray] = []
    start = 0
    for i in range(len(ordered) - 1):
        if float(ordered[i + 1] - ordered[i]) >= 0.04:
            groups.append(order[start:i + 1])
            start = i + 1
    groups.append(order[start:])
    kept: list[np.ndarray] = []
    for group in groups:
        if len(group) < min_points:
            continue
        extent = raised[group].max(axis=0) - raised[group].min(axis=0)
        if max(float(extent[0]), float(extent[1])) > 0.36 or not 0.03 <= float(extent[2]) <= 0.25:
            continue
        kept.append(above[group])
    return kept if len(kept) >= 2 else None


def _bottom_seen(mask: np.ndarray, depth: np.ndarray, instance: int, ys: np.ndarray, xs: np.ndarray,
                 cfg: PerceptionConfig) -> bool:
    """Whether a region's lowest pixels show the object's base: not when the region reaches the
    lower image edge (the base is out of view), nor when much of its lower edge has something
    clearly nearer right below it (the base is hidden, for example a wall behind a counter) or
    clearly farther (the edge overhangs, for example a table top segmented without its legs). A
    base on the floor has floor right below it, at nearly the same distance. An overhang does not
    hide the base of a region that also rests on a support along much of its lower edge, for
    example a bowl whose rim reaches past the table edge."""
    h = mask.shape[0]
    if ys.max() >= h - 1:
        return False
    sel = np.zeros(mask.shape, dtype=bool)
    sel[ys, xs] = True
    edge_y, edge_x = np.nonzero(sel[:-2] & ~sel[1:-1])
    if len(edge_y) == 0:
        return True
    here = depth[edge_y, edge_x]
    below = depth[edge_y + 2, edge_x]
    other = mask[edge_y + 2, edge_x] != instance
    measured = (here > 0.05) & other
    if not measured.any():
        return True
    total = float(measured.sum())
    read = measured & (below > 0.05)
    supported = float((read & (np.abs(below - here) <= cfg.support_gap)).sum()) / total >= cfg.supported_share
    hidden = read & (below < here - cfg.occluder_gap)
    if not supported:
        hidden |= read & (below > here + cfg.overhang_gap)
    return float(hidden.sum()) / total < cfg.occluded_share


def _hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    pts = np.unique(np.round(xy, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def box_of(points: np.ndarray, trim: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Oriented box of a point set: centre, full extents (x along the longer horizontal side) and
    yaw, from the minimum-area rectangle around the footprint. With ``trim`` the box is fitted
    again without the given share of points at either end of each horizontal axis (scattered depth
    noise)."""
    if trim > 0 and len(points) > 50:
        centre, _, yaw = box_of(points)
        c, s = math.cos(yaw), math.sin(yaw)
        rel = points - centre
        coords = np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c, rel[:, 2]], axis=1)
        lo, hi = np.quantile(coords, trim, axis=0), np.quantile(coords, 1 - trim, axis=0)
        # Trim horizontally only: thin vertical parts such as table legs hold few points but set the base.
        keep = np.all((coords[:, :2] >= lo[:2]) & (coords[:, :2] <= hi[:2]), axis=1)
        if keep.sum() >= 20:
            points = points[keep]
    xy = points[:, :2]
    hull = _hull(xy)
    best = None
    if len(hull) >= 3:
        edges = np.roll(hull, -1, axis=0) - hull
        for angle in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 4)):
            c, s = math.cos(angle), math.sin(angle)
            lx, ly = hull @ np.array([c, s]), hull @ np.array([-s, c])
            area = (lx.max() - lx.min()) * (ly.max() - ly.min())
            if best is None or area < best[0]:
                best = (area, angle)
    yaw = best[1] if best else 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = xy @ np.array([c, s]), xy @ np.array([-s, c])
    ex, ey = lx.max() - lx.min(), ly.max() - ly.min()
    mx, my = (lx.max() + lx.min()) / 2, (ly.max() + ly.min()) / 2
    if ey > ex:
        ex, ey, yaw = ey, ex, yaw + math.pi / 2
    centre_xy = mx * np.array([c, s]) + my * np.array([-s, c])
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return np.array([centre_xy[0], centre_xy[1], (z_lo + z_hi) / 2]), np.array([ex, ey, z_hi - z_lo]), yaw


def box_corners(centre: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """Eight corners of an oriented box.

    :param centre: box centre, metres.
    :param size: full extents, x along ``yaw``.
    :param yaw: heading of the box x axis, radians.
    :returns: ``(8, 3)`` corner positions.
    """
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    ax, ay = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    hx, hy, hz = (float(v) / 2 for v in size)
    up = np.array([0.0, 0.0, 1.0])
    return np.array([centre + sx * hx * ax + sy * hy * ay + sz * hz * up
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])


def box_at_yaw(points: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Box of a point set in a given heading (for parts aligned with a known frame); the longer
    horizontal side is reported as x, turning the heading by a quarter when needed."""
    c, s = math.cos(yaw), math.sin(yaw)
    xy = points[:, :2]
    lx, ly = xy @ np.array([c, s]), xy @ np.array([-s, c])
    ex, ey = lx.max() - lx.min(), ly.max() - ly.min()
    mx, my = (lx.max() + lx.min()) / 2, (ly.max() + ly.min()) / 2
    centre_xy = mx * np.array([c, s]) + my * np.array([-s, c])
    if ey > ex:
        ex, ey, yaw = ey, ex, yaw + math.pi / 2
    z_lo, z_hi = points[:, 2].min(), points[:, 2].max()
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return np.array([centre_xy[0], centre_xy[1], (z_lo + z_hi) / 2]), np.array([ex, ey, z_hi - z_lo]), yaw


def top_face_array(frame: dict, cam_pose: dict, centre: np.ndarray, size: np.ndarray, yaw: float, top: float,
                   px: int, py: int | None = None, depth_tol: float = 0.025) -> np.ndarray:
    """Colours of a box's top face sampled on a grid of ``px`` columns by ``py`` rows (default
    ``px``) in the box frame (column = +x, row 0 = +y), each read from the pixel the grid point
    projects to; NaN where the point is outside the image or the depth there is not the point's
    (something nearer hides it, or the face is not where the box says).

    :returns: array of shape ``(py, px, 3)``."""
    depth, rgb = frame["depth"], frame["rgb"]
    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(frame["fovy"]) / 2.0)
    rot = matrix_from_quat(cam_pose["q"])
    py = px if py is None else py
    lx, ly = np.meshgrid(((np.arange(px) + 0.5) / px - 0.5) * size[0], -((np.arange(py) + 0.5) / py - 0.5) * size[1])
    c, s = math.cos(yaw), math.sin(yaw)
    world = np.stack([centre[0] + c * lx - s * ly, centre[1] + s * lx + c * ly, np.full_like(lx, top)], axis=-1)
    x, y, z = np.moveaxis((world - np.asarray(cam_pose["p"], float)) @ rot, -1, 0)
    ok = x >= 0.1
    xs = np.where(ok, x, 1.0)
    u, v = np.rint((w - 1) / 2 - y * f / xs).astype(int), np.rint((h - 1) / 2 - z * f / xs).astype(int)
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    ok &= np.abs(depth[v, u].astype(float) - x) <= depth_tol
    out = np.full((py, px, 3), np.nan)
    out[ok] = rgb[v[ok], u[ok]]
    return out


def face_rows(face: np.ndarray) -> list[list[list[int] | None]]:
    """A face array as JSON rows: ``[r, g, b]`` integers, ``None`` where unseen."""
    return [[None if not np.isfinite(c[0]) else [int(round(float(v))) for v in c] for c in row] for row in face]


def top_face(frame: dict, cam_pose: dict, centre: np.ndarray, size: np.ndarray, yaw: float, top: float,
             px: int, depth_tol: float = 0.025, py: int | None = None) -> list[list[list[int] | None]]:
    """:func:`top_face_array` as JSON rows (``None`` where unseen)."""
    return face_rows(top_face_array(frame, cam_pose, centre, size, yaw, top, px, py, depth_tol))


def resample_face(face: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """A face array averaged into ``rows`` × ``cols`` cells over the same extent (each fine cell
    counts in the coarse cell its centre falls in); NaN where no fine cell was seen."""
    h, w = face.shape[:2]
    r = np.repeat(((np.arange(h) * rows) // h)[:, None], w, axis=1)
    c = np.repeat(((np.arange(w) * cols) // w)[None, :], h, axis=0)
    seen = np.isfinite(face[..., 0])
    sums, counts = np.zeros((rows, cols, 3)), np.zeros((rows, cols))
    np.add.at(sums, (r[seen], c[seen]), face[seen])
    np.add.at(counts, (r[seen], c[seen]), 1.0)
    out = np.full((rows, cols, 3), np.nan)
    hit = counts > 0
    out[hit] = sums[hit] / counts[hit][:, None]
    return out


def _grid(face) -> np.ndarray:
    if isinstance(face, np.ndarray):
        return face.astype(float)
    return np.array([[c if c is not None else [np.nan] * 3 for c in row] for row in face], float)


def match_face(face: list[list], reference: list[list], top: int = 3) -> dict:
    """Where a face appears on a reference face sampled at the same cell size, and turned how.

    Each grid is in its object's frame (column = +x, row 0 = +y), as JSON rows (cells ``[r, g, b]``
    or ``None``) or as an array with NaN where unseen.
    The face is turned by k quarter turns counter-clockwise (as :func:`world_up_face` turns) and
    slid over the reference cell by cell; each placement scores the normalized correlation of the
    colours both grids hold there.

    :param face: grid of the object's face.
    :param reference: grid of the reference face, larger than ``face``.
    :param top: placements to return, best first; each sits apart from the better ones by at least
        half the face.
    :returns: ``{matches: [{row, col, turns, score}], spread}`` where ``row`` and ``col`` are the
        reference cell under the turned face's centre (fractional), ``turns`` is k and ``spread``
        is the face's colour standard deviation (near 0: a plain face)."""
    obj, ref = _grid(face), _grid(reference)
    spread = float(np.nanstd(obj)) if np.isfinite(obj).any() else 0.0
    candidates = []
    for k in range(4):
        turned = np.rot90(obj, k)  # counter-clockwise seen from above, row 0 at +y
        h, w = turned.shape[:2]
        for i in range(ref.shape[0] - h + 1):
            for j in range(ref.shape[1] - w + 1):
                window = ref[i:i + h, j:j + w]
                both = np.isfinite(turned) & np.isfinite(window)
                if both.sum() < 0.5 * turned.size:
                    continue
                # Each channel against its own mean: a picture's overall tint matches everywhere.
                a = np.where(both, turned - np.nanmean(np.where(both, turned, np.nan), axis=(0, 1)), 0.0)
                b = np.where(both, window - np.nanmean(np.where(both, window, np.nan), axis=(0, 1)), 0.0)
                norm = math.sqrt(float((a * a).sum() * (b * b).sum()))
                score = float((a * b).sum() / norm) if norm > 1e-9 else 0.0
                candidates.append((score, i + h / 2, j + w / 2, k, max(h, w)))
    candidates.sort(key=lambda c: -c[0])
    chosen: list[tuple] = []
    for cand in candidates:
        if all(math.hypot(cand[1] - c[1], cand[2] - c[2]) >= c[4] / 2 for c in chosen):
            chosen.append(cand)
        if len(chosen) == top:
            break
    return {"matches": [{"row": r, "col": c, "turns": k, "score": round(s, 3)} for s, r, c, k, _ in chosen],
            "spread": round(spread, 1)}


def world_up_face(rows: list[list], yaw: float) -> tuple[list[list], int]:
    """A face grid in the object frame (column = +x, row 0 = +y) turned by the object's heading
    rounded to quarter turns, so that, seen from above, row 0 faces the map's +y and columns run
    towards +x; the face stays turned by the remainder, under 45 degrees.

    :param rows: square grid as :func:`top_face` returns it.
    :param yaw: heading of the object frame in the map, radians.
    :returns: the turned grid and the counter-clockwise turn applied, degrees (0, 90, 180 or 270)."""
    quarter = int(round(yaw / (math.pi / 2))) % 4
    turned = rows
    for _ in range(quarter):
        # Counter-clockwise as seen from above: the last column becomes row 0.
        turned = [[row[len(row) - 1 - k] for row in turned] for k in range(len(turned[0]))]
    return turned, 90 * quarter
