"""Rigid-body math on plain lists: positions are [x, y, z], quaternions are MuJoCo-order [w, x, y, z]."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Vec3 = list[float]
Quat = list[float]

IDENTITY_QUAT: Quat = [1.0, 0.0, 0.0, 0.0]


def quat_normalize(q: Sequence[float]) -> Quat:
    arr = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    return (arr / norm).tolist()


def quat_mul(a: Sequence[float], b: Sequence[float]) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_conj(q: Sequence[float]) -> Quat:
    return [q[0], -q[1], -q[2], -q[3]]


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> Vec3:
    p = quat_mul(quat_mul(q, [0.0, v[0], v[1], v[2]]), quat_conj(q))
    return [p[1], p[2], p[3]]


def quat_from_yaw(yaw: float) -> Quat:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> Quat:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2.0)
    return [math.cos(angle / 2.0), a[0] * s, a[1] * s, a[2] * s]


def yaw_from_quat(q: Sequence[float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tip_roll_from_quat(q: Sequence[float]) -> float:
    """Body tip about +x after yaw is removed: 0 upright, ±π/2 lying on a side.

    :param q: world orientation of the body (MuJoCo ``[w, x, y, z]``).
    :returns: tip angle in radians about body +x.
    """
    yaw = yaw_from_quat(q)
    body = matrix_from_quat(quat_conj(quat_from_yaw(yaw))) @ matrix_from_quat(q)
    return math.atan2(float(body[2, 1]), float(body[2, 2]))


def quat_from_matrix(m: np.ndarray) -> Quat:
    """Convert a 3x3 rotation matrix to a unit quaternion."""
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return quat_normalize(q)


def matrix_from_quat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_compose(a: dict, b: dict) -> dict:
    """Return a * b, where each pose is {"p": Vec3, "q": Quat}."""
    p = quat_rotate(a["q"], b["p"])
    return {"p": [a["p"][i] + p[i] for i in range(3)], "q": quat_normalize(quat_mul(a["q"], b["q"]))}


def pose_inverse(a: dict) -> dict:
    qi = quat_conj(a["q"])
    p = quat_rotate(qi, a["p"])
    return {"p": [-p[0], -p[1], -p[2]], "q": qi}


def quat_angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Smallest rotation angle in radians between two orientations."""
    dot = abs(float(np.dot(quat_normalize(a), quat_normalize(b))))
    return 2.0 * math.acos(min(1.0, dot))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
