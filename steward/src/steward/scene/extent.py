"""Oriented bounding boxes of simulated entities, expressed in each entity's root frame.

Phase 0 perception reports an object as its oriented bounding box (OBB); the Observer measures
Ego-State error against the same OBB computed from the World State, so both sides use this
module.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from steward.math3d import quat_from_matrix
from steward.scene.mjcf import EntityNames


@dataclass
class Obb:
    center: list[float]          # world position of the box center
    quat: list[float]            # world orientation of the box (the entity root orientation)
    size: list[float]            # full extents along the box axes

    def pose(self) -> dict:
        return {"p": list(self.center), "q": list(self.quat)}


_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)


def _geom_half_extents(model: mujoco.MjModel, gid: int) -> np.ndarray:
    gtype = int(model.geom_type[gid])
    s = model.geom_size[gid]
    if gtype in (_BOX, _ELLIPSOID):
        return np.array([s[0], s[1], s[2]])
    if gtype == _SPHERE:
        return np.array([s[0], s[0], s[0]])
    if gtype in (_CYLINDER, _CAPSULE):
        extra = s[0] if gtype == _CAPSULE else 0.0
        return np.array([s[0], s[0], s[1] + extra])
    if gtype == _PLANE:
        return np.array([s[0], s[1], 0.0])
    raise ValueError(f"unsupported geom type {gtype} for bounding boxes")


def entity_body_ids(model: mujoco.MjModel, names: EntityNames, links: list[str] | None = None) -> list[int]:
    selected = names.bodies if links is None else {k: v for k, v in names.bodies.items() if k in links}
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in selected.values()]


def entity_obb(model: mujoco.MjModel, data: mujoco.MjData, names: EntityNames,
               links: list[str] | None = None) -> Obb:
    """OBB of the entity's geoms (or of the given links), aligned with the root body frame."""
    body_ids = set(entity_body_ids(model, names, links))
    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, next(iter(names.bodies.values())))
    r_root = data.xmat[root_bid].reshape(3, 3)
    p_root = data.xpos[root_bid]
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] not in body_ids:
            continue
        r_g = data.geom_xmat[gid].reshape(3, 3)
        p_g = data.geom_xpos[gid]
        if model.geom_type[gid] == _MESH:
            center, half = model.geom_aabb[gid, :3], model.geom_aabb[gid, 3:]
            p_g = p_g + r_g @ center
        else:
            half = _geom_half_extents(model, gid)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corner_w = p_g + r_g @ (half * np.array([sx, sy, sz]))
                    corner_r = r_root.T @ (corner_w - p_root)
                    lo = np.minimum(lo, corner_r)
                    hi = np.maximum(hi, corner_r)
    center_r = (lo + hi) / 2.0
    center_w = p_root + r_root @ center_r
    return Obb(center=center_w.tolist(), quat=quat_from_matrix(r_root), size=(hi - lo).tolist())
