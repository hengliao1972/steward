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
    occluded_share: float = 0.3  # share of the lower edge hidden that leaves the base unseen


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


def decode_rgbd(blob: bytes) -> dict:
    """Unpack the robot port's ``/sensors/rgbd`` payload."""
    z = np.load(io.BytesIO(blob))
    return {"rgb": z["rgb"], "depth": z["depth"].astype(np.float32) / 1000.0, "mask": z["mask"],
            "fovy": float(z["fovy"]), "header": json.loads(str(z["header"]))}


def segments(frame: dict, cam_pose: dict, config: PerceptionConfig | None = None,
             rng: np.random.Generator | None = None) -> list[Segment]:
    """Back-project every masked region of an RGB-D frame into map-frame points."""
    cfg = config or PerceptionConfig()
    rng = rng or np.random.default_rng(0)
    depth, mask, rgb = frame["depth"], frame["mask"], frame["rgb"]
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
        if len(dv) > cfg.max_points:
            keep = rng.choice(len(dv), cfg.max_points, replace=False)
            u, v, dv = u[keep], v[keep], dv[keep]
        local = np.stack([dv, -(u - cx) * dv / f, -(v - cy) * dv / f], axis=1)
        points = local @ rot.T + origin
        points = points[points[:, 2] > cfg.floor_cut]
        if len(points) < cfg.min_points:
            continue
        out.append(Segment(
            points=points,
            color=rgb[ys, xs].reshape(-1, 3).mean(axis=0) / 255.0,
            pixels=len(ys),
            invalid_frac=float(1.0 - valid.mean()),
            border=bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
            range=float(np.median(dv)),
            bottom_seen=_bottom_seen(mask, depth, instance, ys, xs, cfg),
        ))
    return out


def _bottom_seen(mask: np.ndarray, depth: np.ndarray, instance: int, ys: np.ndarray, xs: np.ndarray,
                 cfg: PerceptionConfig) -> bool:
    """Whether a region's lowest pixels show the object's base: not when the region reaches the
    lower image edge (the base is out of view), nor when much of its lower edge has something
    clearly nearer right below it (the base is hidden, for example a wall behind a counter). A
    base on the floor has floor right below it, at nearly the same distance."""
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
    hidden = measured & (below > 0.05) & (below < here - cfg.occluder_gap)
    return float(hidden.sum()) / float(measured.sum()) < cfg.occluded_share


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
