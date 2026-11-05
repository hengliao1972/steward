"""Class-agnostic instance masks from depth (no learned segmenter).

Pipeline (Steward camera frame: x forward, y left, z up):
1. RANSAC find near-horizontal support planes (gravity ≈ camera +z, works with mild pitch).
2. For each elevated plane, cluster points *above* it in the plane's tangent (u,v) frame.
3. Mark plane inliers as non-objects; voxel-cluster residuals into furniture / walls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GeometricMaskConfig:
    max_range: float = 5.5
    min_depth: float = 0.08
    ransac_iters: int = 100
    ransac_thresh_m: float = 0.02
    ransac_min_inlier_frac: float = 0.035
    ransac_up_dot: float = 0.78  # allow ~40° camera pitch
    max_planes: int = 6
    min_plane_span_m: float = 0.4
    # tabletop objects above a support plane
    above_min_m: float = 0.008
    above_max_m: float = 0.5
    above_uv_pad_m: float = 0.06
    above_cluster_m: float = 0.045
    above_min_points: int = 28
    above_max_span_m: float = 0.7
    above_min_extent_m: float = 0.03
    # residual furniture / walls
    residual_voxel_m: float = 0.055
    min_voxels: int = 5
    min_pixels: int = 32
    max_instances: int = 48
    wall_min_height_m: float = 0.7
    wall_max_thickness_m: float = 0.45
    furniture_min_height_m: float = 0.28
    furniture_max_span_m: float = 1.8
    residual_min_points: int = 80
    # plane "elevation": mean range of inliers must be closer than floor-ish
    # and signed height along gravity from cloud percentile
    elevated_min_inlier_z_pct: float = 35.0  # plane mean cam-z above this pct of cloud
    seed: int = 0


def _backproject(depth: np.ndarray, fovy_deg: float, valid: np.ndarray):
    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    ys, xs = np.nonzero(valid)
    d = depth[ys, xs].astype(np.float64)
    u, v = xs.astype(np.float64), ys.astype(np.float64)
    pts = np.stack([d, -(u - cx) * d / f, -(v - cy) * d / f], axis=1)
    return pts, ys, xs


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangent axes for plane normal n."""
    a = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n, t1)
    return t1, t2


def _ransac_horizontal_planes(
    pts: np.ndarray, cfg: GeometricMaskConfig, rng: np.random.Generator, up: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    planes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if len(pts) < 50:
        return planes
    available = np.ones(len(pts), dtype=bool)
    for _ in range(cfg.max_planes):
        idxs = np.nonzero(available)[0]
        if len(idxs) < 50:
            break
        cloud = pts[idxs]
        best = None
        for _it in range(cfg.ransac_iters):
            samp = rng.choice(len(cloud), size=3, replace=False)
            p0, p1, p2 = cloud[samp]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            if abs(float(n @ up)) < cfg.ransac_up_dot:
                continue
            if float(n @ up) < 0:
                n = -n
            inl = np.abs((cloud - p0) @ n) <= cfg.ransac_thresh_m
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, inl, n, p0)
        if best is None or best[0] < cfg.ransac_min_inlier_frac * len(pts):
            break
        _, inl_local, n, p0 = best
        sheet = cloud[inl_local]
        span = sheet.max(0) - sheet.min(0)
        t1, t2 = _plane_basis(n)
        uv = np.stack([(sheet - sheet.mean(0)) @ t1, (sheet - sheet.mean(0)) @ t2], axis=1)
        uv_span = uv.max(0) - uv.min(0)
        if max(float(uv_span[0]), float(uv_span[1])) < cfg.min_plane_span_m:
            break
        center = sheet.mean(0)
        _, _, vh = np.linalg.svd(sheet - center, full_matrices=False)
        n2 = vh[-1]
        if abs(float(n2 @ up)) < cfg.ransac_up_dot * 0.95:
            n2 = n
        if float(n2 @ up) < 0:
            n2 = -n2
        dist_all = np.abs((pts - center) @ n2)
        inl_all = available & (dist_all <= cfg.ransac_thresh_m)
        t1, t2 = _plane_basis(n2)
        uv_all = np.stack([(pts - center) @ t1, (pts - center) @ t2], axis=1)
        uv_sheet = np.stack([(sheet - center) @ t1, (sheet - center) @ t2], axis=1)
        mn, mx = uv_sheet.min(0) - 0.12, uv_sheet.max(0) + 0.12
        inl_all = inl_all & (uv_all[:, 0] >= mn[0]) & (uv_all[:, 0] <= mx[0]) & (uv_all[:, 1] >= mn[1]) & (uv_all[:, 1] <= mx[1])
        if int(inl_all.sum()) < 40:
            break
        planes.append((n2.astype(np.float64), center.astype(np.float64), inl_all))
        available[inl_all] = False
    return planes


def _uv_clusters(uv: np.ndarray, cell: float, min_points: int) -> list[np.ndarray]:
    if len(uv) == 0:
        return []
    keys = np.floor(uv / cell).astype(np.int32)
    off = keys.min(0)
    k = keys - off
    packed = k[:, 0].astype(np.int64) * 100_003 + k[:, 1].astype(np.int64)
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(packed):
        buckets.setdefault(int(p), []).append(i)
    key_set = {tuple(map(int, row)) for row in k}
    key_to_pack = {tuple(map(int, row)): int(p) for row, p in zip(k, packed)}
    neigh = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    visited: set[tuple[int, int]] = set()
    comps: list[np.ndarray] = []
    for seed in key_set:
        if seed in visited:
            continue
        stack = [seed]
        visited.add(seed)
        idx: list[int] = []
        while stack:
            cur = stack.pop()
            pk = key_to_pack.get(cur)
            if pk is not None:
                idx.extend(buckets.get(pk, []))
            x, y = cur
            for dx, dy in neigh:
                nxt = (x + dx, y + dy)
                if nxt in key_set and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if len(idx) >= min_points:
            comps.append(np.asarray(idx, dtype=np.int32))
    comps.sort(key=len, reverse=True)
    return comps


def _voxel_components(pts: np.ndarray, cell: float, min_voxels: int, min_points: int) -> list[np.ndarray]:
    if len(pts) == 0:
        return []
    keys = np.floor(pts / cell).astype(np.int32)
    off = keys.min(0)
    k = keys - off
    packed = (
        k[:, 0].astype(np.int64) * 1_000_000_009
        + k[:, 1].astype(np.int64) * 10007
        + k[:, 2].astype(np.int64)
    )
    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(packed):
        buckets.setdefault(int(p), []).append(i)
    key_set = {tuple(map(int, row)) for row in k}
    key_to_pack = {tuple(map(int, row)): int(p) for row, p in zip(k, packed)}
    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    visited: set[tuple[int, int, int]] = set()
    comps: list[np.ndarray] = []
    for seed in key_set:
        if seed in visited:
            continue
        stack = [seed]
        visited.add(seed)
        idx: list[int] = []
        while stack:
            cur = stack.pop()
            pk = key_to_pack.get(cur)
            if pk is not None:
                idx.extend(buckets.get(pk, []))
            x, y, z = cur
            for dx, dy, dz in neigh:
                nxt = (x + dx, y + dy, z + dz)
                if nxt in key_set and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        n_vox = len({tuple(map(int, k[i])) for i in idx}) if idx else 0
        if n_vox >= min_voxels and len(idx) >= min_points:
            comps.append(np.asarray(idx, dtype=np.int32))
    comps.sort(key=len, reverse=True)
    return comps


def masks_from_depth(
    depth: np.ndarray,
    fovy_deg: float,
    config: GeometricMaskConfig | None = None,
    up_cam: np.ndarray | None = None,
    supports_only: bool = False,
) -> np.ndarray:
    cfg = config or GeometricMaskConfig()
    rng = np.random.default_rng(cfg.seed)
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    valid = (depth > cfg.min_depth) & (depth < cfg.max_range) & np.isfinite(depth)
    if int(valid.sum()) < cfg.min_pixels:
        return mask

    pts, ys, xs = _backproject(depth, fovy_deg, valid)
    # Steward head camera pitches ~30° down; world +z ≈ (-sin θ, 0, cos θ) in camera frame.
    default_up = np.array([-0.5, 0.0, math.sqrt(3.0) / 2.0])
    up = np.asarray(up_cam if up_cam is not None else default_up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    planes = _ransac_horizontal_planes(pts, cfg, rng, up)
    # refine gravity from dominant plane normals if available
    if planes:
        grav = np.mean([n for n, _, _ in planes], axis=0)
        if np.linalg.norm(grav) > 1e-6:
            up = grav / np.linalg.norm(grav)

    claimed = np.zeros(len(pts), dtype=bool)
    instances: list[np.ndarray] = []

    # height along gravity for elevation test
    h_along = pts @ up
    h_ref = float(np.percentile(h_along, cfg.elevated_min_inlier_z_pct))

    for n, center, inl in planes:
        plane_h = float(center @ up)
        elevated = plane_h > h_ref + 0.08  # table/counter above lower cloud
        signed = (pts - center) @ n
        t1, t2 = _plane_basis(n)
        uv = np.stack([(pts - center) @ t1, (pts - center) @ t2], axis=1)
        if not elevated:
            claimed[inl] = True
            continue
        sheet_uv = uv[inl]
        mn, mx = sheet_uv.min(0) - cfg.above_uv_pad_m, sheet_uv.max(0) + cfg.above_uv_pad_m
        in_foot = (
            (uv[:, 0] >= mn[0]) & (uv[:, 0] <= mx[0])
            & (uv[:, 1] >= mn[1]) & (uv[:, 1] <= mx[1])
        )
        # Drop support-surface inliers so tabletop objects detach; keep residual furniture separately.
        claimed[inl] = True
        above = (
            (~claimed)
            & (signed >= cfg.above_min_m)
            & (signed <= cfg.above_max_m)
            & in_foot
        )
        if int(above.sum()) < cfg.above_min_points:
            continue
        local_ids = np.nonzero(above)[0]
        for cidx in _uv_clusters(uv[local_ids], cfg.above_cluster_m, cfg.above_min_points):
            idx = local_ids[cidx]
            cuv = uv[idx]
            span_uv = cuv.max(0) - cuv.min(0)
            height = float(signed[idx].max() - signed[idx].min())
            if max(float(span_uv[0]), float(span_uv[1])) > cfg.above_max_span_m:
                continue
            if max(float(span_uv[0]), float(span_uv[1]), height) < cfg.above_min_extent_m:
                continue
            instances.append(idx)
            claimed[idx] = True

    residual = np.zeros(len(pts), dtype=bool) if supports_only else ~claimed
    if int(residual.sum()) >= cfg.min_pixels:
        res_ids = np.nonzero(residual)[0]
        for cidx in _voxel_components(
            pts[res_ids], cfg.residual_voxel_m, cfg.min_voxels, cfg.min_pixels
        ):
            idx = res_ids[cidx]
            cloud = pts[idx]
            sp = cloud.max(0) - cloud.min(0)
            # height along gravity
            hh = float((cloud @ up).max() - (cloud @ up).min())
            xy_span = max(float(sp[0]), float(sp[1]))
            thin = min(float(sp[0]), float(sp[1]))
            is_wall = hh >= cfg.wall_min_height_m and thin <= cfg.wall_max_thickness_m
            is_furniture = (
                hh >= cfg.furniture_min_height_m
                and xy_span <= cfg.furniture_max_span_m
                and xy_span >= 0.25
                and len(idx) >= cfg.residual_min_points
            )
            # compact mid blobs (chairs / floating synthetic objects) without a support plane
            is_compact = (
                xy_span <= 0.9
                and hh >= 0.08
                and len(idx) >= max(cfg.min_pixels, 40)
                and xy_span >= 0.08
            )
            # skip leftover floor/sheet scraps
            if hh < 0.12 and xy_span > 0.6:
                continue
            if is_wall or is_furniture or is_compact:
                instances.append(idx)
                claimed[idx] = True

    used = np.zeros((h, w), dtype=bool)
    kept: list[tuple[np.ndarray, np.ndarray]] = []

    def accept(idx: np.ndarray) -> None:
        py, px = ys[idx], xs[idx]
        sel = ~used[py, px]
        if int(sel.sum()) < cfg.min_pixels:
            return
        py, px = py[sel], px[sel]
        used[py, px] = True
        kept.append((py, px))

    instances.sort(key=len)
    for idx in instances:
        if len(kept) >= cfg.max_instances:
            break
        accept(idx)

    if not kept:
        return mask
    ids = rng.permutation(np.arange(1, len(kept) + 1))
    for (py, px), iid in zip(kept, ids):
        mask[py, px] = int(iid)
    return mask


def replace_frame_mask(frame: dict, config: GeometricMaskConfig | None = None) -> dict:
    out = dict(frame)
    out["mask"] = masks_from_depth(frame["depth"], float(frame["fovy"]), config)
    return out
