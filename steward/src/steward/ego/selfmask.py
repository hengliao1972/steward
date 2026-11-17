"""Pixels of a robot's own body, projected from its own kinematics.

A wrist camera looks past its own gripper: roughly the lower half of its frame can be the hand,
the wrist and the forearm. The synthetic channel never had this problem (the renderer's
segmentation excludes the observing robot), but a real segmenter does not know what the robot
looks like, and the old workaround — "closer than 30 cm is the arm" — both misses a distant
forearm and throws away a real object held near the lens.

So the robot masks itself: every link's collision geometry is posed by forward kinematics, its
surface sampled, projected through the camera model, and splatted into a pixel mask. Whatever
that mask covers is the robot, and a perception stage drops those pixels before asking what is
in the world.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from scipy.spatial import ConvexHull

import numpy as np

from steward.math3d import matrix_from_quat, quat_rotate


def camera_model(pose: dict, intrinsics: dict) -> dict:
    """Focal length, principal point and the camera's world rotation for one sensor."""
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fovy = math.radians(float(intrinsics.get("fovy", 60.0)))
    return {
        "f": (height / 2.0) / math.tan(fovy / 2.0),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
        "origin": np.asarray(pose["p"], dtype=float),
        "rot": np.asarray(matrix_from_quat(pose["q"]), dtype=float).reshape(3, 3),
    }


def project(points_world: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixel coordinates; returns ``(uv, in_front)``.

    The camera frame follows the sensor convention used everywhere in steward: +x along the
    optical axis, and the image's u grows opposite to +y and v opposite to +z.
    """
    if len(points_world) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    local = (np.asarray(points_world, dtype=float) - cam["origin"]) @ cam["rot"]
    depth = local[:, 0]
    in_front = depth > 1e-3
    safe = np.where(in_front, depth, 1.0)
    u = cam["cx"] - local[:, 1] * cam["f"] / safe
    v = cam["cy"] - local[:, 2] * cam["f"] / safe
    return np.stack([u, v], axis=1), in_front


def _surface_points(geometry: dict, samples: int = 7) -> np.ndarray:
    """Surface sample points of one primitive, in its link frame."""
    kind = geometry.get("kind")
    pose = geometry.get("pose") or {}
    centre = np.asarray(pose.get("p", [0.0, 0.0, 0.0]), dtype=float)
    quat = pose.get("q", [1.0, 0.0, 0.0, 0.0])
    size = [float(s) for s in geometry.get("size", [])]
    local: list[list[float]] = []

    if kind == "mesh":
        path = Path(geometry["ref"])
        stat = path.stat()
        points = _mesh_surface(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        return points @ matrix_from_quat(quat).T + centre

    if kind != "primitive":
        return np.zeros((0, 3))

    shape = geometry["shape"]
    if shape == "sphere":
        radius = size[0]
        for i in range(1, samples + 1):
            for j in range(samples):
                theta = math.pi * i / (samples + 1)
                phi = 2 * math.pi * j / samples
                local.append([radius * math.sin(theta) * math.cos(phi),
                              radius * math.sin(theta) * math.sin(phi),
                              radius * math.cos(theta)])
        local.append([0.0, 0.0, 0.0])
    elif shape in ("cylinder", "capsule"):
        radius, length = size[0], size[1]
        half = length / 2.0
        for axis in np.linspace(-half, half, samples + 1):
            for j in range(samples):
                phi = 2 * math.pi * j / samples
                local.append([radius * math.cos(phi), radius * math.sin(phi), axis])
        local.extend([[0.0, 0.0, -half], [0.0, 0.0, half]])
    elif shape in ("box", "ellipsoid"):
        half = np.asarray(size, dtype=float) / 2.0
        steps = max(3, samples - 2)
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for sign in (-1.0, 1.0):
                for i in np.linspace(-1.0, 1.0, steps):
                    for j in np.linspace(-1.0, 1.0, steps):
                        point = [0.0, 0.0, 0.0]
                        point[axis] = sign * half[axis]
                        point[others[0]] = i * half[others[0]]
                        point[others[1]] = j * half[others[1]]
                        local.append(point)
    else:
        return np.zeros((0, 3))

    points = np.asarray(local, dtype=float)
    return np.asarray([quat_rotate(quat, p) for p in points], dtype=float) + centre


@lru_cache(maxsize=128)
def _mesh_surface(path: str, modified: int, size: int) -> np.ndarray:
    """Convex-hull vertices in asset coordinates; cache invalidates when the asset changes."""
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="part", file=path)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="mesh", mesh="part")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    points = model.mesh_vert @ matrix_from_quat(model.mesh_quat[0]).T + model.mesh_pos[0]
    return points[ConvexHull(points).vertices]


def body_points(robot_model: dict, link_pose, samples: int = 12) -> list[tuple[str, list[np.ndarray]]]:
    """Each link's geometry as per-part surface samples in the world frame.

    Parts stay separate because every primitive steward emits is convex: the silhouette of each
    part is its projected convex hull, and filling the hull of a union would bridge two parts that
    do not actually touch on screen.
    """
    out: list[tuple[str, list[np.ndarray]]] = []
    tree = robot_model["body"]
    for link in tree["links"]:
        geometry = link.get("geometry") or {}
        parts = list(geometry.get("collision") or []) + list(geometry.get("visual") or [])
        if not parts:
            continue
        pose = link_pose(link["id"])
        origin = np.asarray(pose["p"], dtype=float)
        quat = pose["q"]
        world_parts = []
        for part in parts:
            local = _surface_points(part, samples=samples)
            if len(local):
                world_parts.append(local @ matrix_from_quat(quat).T + origin)
        if world_parts:
            out.append((link["id"], world_parts))
    return out


def self_pixels(robot_model: dict, link_pose, sensor_pose: dict, intrinsics: dict,
                dilate: int = 2) -> np.ndarray:
    """Boolean ``[height, width]`` mask of pixels covered by the robot's own body."""
    cam = camera_model(sensor_pose, intrinsics)
    mask = np.zeros((cam["height"], cam["width"]), dtype=bool)
    for _, parts in body_points(robot_model, link_pose):
        for points in parts:
            uv, in_front = project(points, cam)
            uv = uv[in_front]
            if len(uv) >= 3:
                _fill_convex(mask, uv)
    return _dilate(mask, dilate)


def _fill_convex(mask: np.ndarray, uv: np.ndarray) -> None:
    """Set every pixel inside the convex hull of ``uv`` (Andrew's monotone chain + scanline)."""
    pts = np.unique(np.round(uv).astype(float), axis=0)
    if len(pts) < 3:
        return
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return

    h, w = mask.shape
    y0 = max(0, int(np.floor(hull[:, 1].min())))
    y1 = min(h - 1, int(np.ceil(hull[:, 1].max())))
    for y in range(y0, y1 + 1):
        xs: list[float] = []
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
                xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        if len(xs) >= 2:
            x_from = max(0, int(np.ceil(min(xs))))
            x_to = min(w - 1, int(np.floor(max(xs))))
            if x_to >= x_from:
                mask[y, x_from : x_to + 1] = True


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation: the sampled silhouette is a splat, not an exact outline."""
    if radius <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out |= _shift(mask, dy, dx)
    return out


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    ys = slice(max(0, dy), h + min(0, dy))
    xs = slice(max(0, dx), w + min(0, dx))
    ys_src = slice(max(0, -dy), h + min(0, -dy))
    xs_src = slice(max(0, -dx), w + min(0, -dx))
    out[ys, xs] = mask[ys_src, xs_src]
    return out
