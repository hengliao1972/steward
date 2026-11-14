"""Fastener geometry and persistent records; deliberately independent of MuJoCo.

All feature coordinates are link-local, metres. Axes point out of a hole/screw,
whereas a powered driver's axis points from its handle towards the bit tip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import os

import numpy as np

from steward.math3d import matrix_from_quat, quat_from_matrix
from steward.scene.schema import RigidTree

STATE_KEY = "fastener"


@dataclass(frozen=True)
class FastenConfig:
    radial: float = 0.020
    gap: float = 0.003
    angle: float = math.radians(12)
    tip_distance: float = 0.015
    dwell: float = 0.2
    engagement: float = 0.001
    driver_angle: float = math.radians(25)
    mode: str = "driver"
    seat_linear_speed: float = 0.01
    seat_angular_speed: float = 0.1

    def __post_init__(self):
        if self.mode not in ("driver", "seat", "off"):
            raise ValueError("fasten mode must be driver, seat or off")
        for name in ("radial", "gap", "tip_distance", "dwell", "engagement",
                     "seat_linear_speed", "seat_angular_speed"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"fasten {name} must be finite and positive")
        for name in ("angle", "driver_angle"):
            if not 0 < getattr(self, name) < math.pi / 2:
                raise ValueError(f"fasten {name} must lie between 0 and pi/2")

    @classmethod
    def from_env(cls, environ=None):
        """Explicit WorldSim(fasten=...) wins; otherwise read environment once at startup.

        Distances/speeds/time use SI; *_ANGLE_DEG uses degrees. No environment mutation.
        """
        env = os.environ if environ is None else environ
        values = {"mode": env.get("STEWARD_FASTEN_MODE", "off")}
        for name in ("radial", "gap", "tip_distance", "dwell", "engagement",
                     "seat_linear_speed", "seat_angular_speed"):
            key = "STEWARD_FASTEN_" + name.upper()
            if key in env:
                values[name] = float(env[key])
        for name in ("angle", "driver_angle"):
            key = "STEWARD_FASTEN_" + name.upper() + "_DEG"
            if key in env:
                values[name] = math.radians(float(env[key]))
        return cls(**values)


def stationary(velocity, cfg: FastenConfig) -> bool:
    """MuJoCo object velocity order: angular, linear (world coordinates)."""
    v = np.asarray(velocity)
    return bool(np.all(np.isfinite(v)) and np.linalg.norm(v[:3]) <= cfg.seat_angular_speed
                and np.linalg.norm(v[3:]) <= cfg.seat_linear_speed)


@dataclass
class Feature:
    entity: str
    link: str
    kind: str
    spec: dict

    @property
    def key(self) -> tuple[str, str, str]:
        return self.entity, self.link, str(self.spec.get("id", self.kind))

    def at(self, pose: dict) -> dict:
        rot = matrix_from_quat(pose["q"])
        out = dict(self.spec, entity=self.entity, link=self.link, key=self.key)
        for key in ("entry", "tip", "head", "seat"):
            if key in self.spec:
                out[key] = np.asarray(pose["p"]) + rot @ np.asarray(self.spec[key])
        out["axis"] = rot @ unit(self.spec["axis"])
        out["pose"] = pose
        return out


def unit(value):
    v = np.asarray(value, float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("fastener axis must be finite and nonzero")
    return v / n


def features(scene) -> dict[str, list[Feature]]:
    out = {k: [] for k in ("screw-hole", "screw", "powered-driver")}
    required = {"screw-hole": ("id", "axis", "entry", "depth"),
                "screw": ("axis", "tip", "head", "seat"), "powered-driver": ("axis", "tip")}
    for ent in scene.entities:
        if (ent.role != "object" or not isinstance(ent.body, RigidTree) or ent.rigid.root.value != "free"
                or any(j.type.value != "fixed" for j in ent.rigid.joints)):
            continue
        for link in ent.rigid.links:
            for aff in link.affordances:
                if not isinstance(aff.value, dict):
                    continue
                for kind, keys in required.items():
                    spec = aff.value.get(kind)
                    if isinstance(spec, dict) and all(k in spec for k in keys):
                        unit(spec["axis"])
                        out[kind].append(Feature(ent.id, link.id, kind, spec))
    return out


def weld_pairs(inventory) -> list[tuple[str, str]]:
    parts = sorted({f.entity for f in inventory["screw-hole"]})
    screws = sorted({f.entity for f in inventory["screw"]})
    if not screws:
        return []
    return sorted({tuple(sorted(pair)) for pair in [*combinations(parts, 2),
                  *((p, s) for p in parts for s in screws if p != s)]})


def records(scene) -> dict[str, dict]:
    """Only complete records survive deletion of a participant. No truth enters Ego."""
    if scene.mode != "world":
        return {}
    ids = {e.id for e in scene.entities}
    return {e.id: e.state_vars[STATE_KEY].value for e in scene.entities
            if STATE_KEY in e.state_vars
            and all(h["entity"] in ids for h in e.state_vars[STATE_KEY].value["holes"])
            and all(w[k] in ids for w in e.state_vars[STATE_KEY].value["welds"] for k in ("a", "b"))}


def active_welds(scene) -> dict[tuple[str, str], dict]:
    return {(w["a"], w["b"]): w["rel"] for r in records(scene).values() for w in r["welds"]}


def component(entity: str, welds) -> set[str]:
    found = {entity}
    while True:
        larger = found | {x for a, b in welds if a in found or b in found for x in (a, b)}
        if larger == found:
            return found
        found = larger


def occupied(scene) -> set[tuple[str, str, str]]:
    return {(h["entity"], h["link"], h["hole"]) for r in records(scene).values() for h in r["holes"]}


def mating(upper: dict, lower: dict, screw: dict, cfg: FastenConfig) -> bool:
    """Directed entry faces: upper bore exits onto lower entry, screw tip at upper entry."""
    if len({upper["entity"], lower["entity"], screw["entity"]}) != 3:
        return False
    axis = upper["axis"]
    cosine = math.cos(cfg.angle)
    if axis @ lower["axis"] < cosine or axis @ screw["axis"] < cosine:
        return False
    delta = upper["entry"] - lower["entry"]
    axial = float(delta @ axis)
    if np.linalg.norm(delta - axial * axis) > cfg.radial:
        return False
    if upper["depth"] <= 0 or abs(axial - upper["depth"]) > cfg.gap:
        return False
    length = float((screw["seat"] - screw["tip"]) @ screw["axis"])
    if length < axial + cfg.engagement:
        return False
    for hole in (upper, lower):
        offset = hole["entry"] - screw["tip"]
        if np.linalg.norm(offset - float(offset @ screw["axis"]) * screw["axis"]) > cfg.radial:
            return False
    tip = screw["tip"] - upper["entry"]
    along = float(tip @ axis)
    return bool(np.linalg.norm(tip - along * axis) <= cfg.radial and abs(along) <= cfg.gap)


def driving(driver: dict, screw: dict, cfg: FastenConfig) -> bool:
    return bool(driver["axis"] @ screw["axis"] <= -math.cos(cfg.driver_angle)
                and np.linalg.norm(driver["tip"] - screw["head"]) <= cfg.tip_distance)


def seated_pose(feature: Feature, upper: dict, screw: dict) -> dict:
    """Align the screw axis exactly, keeping its existing roll; seat its head bottom."""
    old = screw["axis"]
    new = upper["axis"]
    cross = np.cross(old, new)
    skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
    rotate = np.eye(3) + skew + skew @ skew / (1 + float(old @ new))
    rot = rotate @ matrix_from_quat(screw["pose"]["q"])
    return {"p": (upper["entry"] - rot @ np.asarray(feature.spec["seat"])).tolist(),
            "q": quat_from_matrix(rot)}


@dataclass
class Dwell:
    since: dict[tuple, float] = field(default_factory=dict)

    def update(self, keys, now: float, seconds: float) -> list[tuple]:
        self.since = {key: self.since.get(key, now) for key in keys}
        return [key for key, start in self.since.items() if now - start + 1e-12 >= seconds]
