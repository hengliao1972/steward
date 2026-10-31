"""SceneState: the single data model shared by World State (exact) and Ego-State (estimated).

The JSON form uses camelCase keys and is the wire format of both the World Control API and the
Ego API. ``SceneState(mode="world")`` carries exact values under ``WorldEntityId`` ids (``w:``
prefix); ``SceneState(mode="ego")`` carries estimates with uncertainty under robot-assigned
``EgoEntityId`` ids (``e:`` prefix). Leaf unions (geometry, uncertainty, actuator, sensor) stay
plain JSON dictionaries and are checked by :mod:`steward.scene.validate`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from steward.math3d import IDENTITY_QUAT

Mode = Literal["world", "ego"]
WORLD_ID_PREFIX = "w:"
EGO_ID_PREFIX = "e:"

ROLES = ("environment", "object", "robot", "actor")
ROOT_TYPES = ("fixed", "planar", "free")
JOINT_TYPES = ("fixed", "revolute", "prismatic", "ball")
RELATION_TYPES = ("attached", "supported-by", "contained-in")
SHAPES = ("box", "sphere", "capsule", "cylinder", "plane", "ellipsoid")
GEOMETRY_KINDS = ("primitive", "mesh", "pointcloud", "sdf")
GEOMETRY_ROLES = ("visual", "collision", "observed")
UNCERTAINTY_KINDS = ("gaussian", "categorical", "interval", "samples")
SOURCES = ("observed", "fused", "prior", "inferred", "predicted")


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class Est:
    """An estimable quantity. World State sets only ``value``; Ego-State may add the rest."""

    value: Any
    unc: dict | None = None
    t: float | None = None
    src: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"value": copy.deepcopy(self.value), "unc": self.unc, "t": self.t, "src": self.src})

    @staticmethod
    def from_dict(d: dict) -> Est:
        return Est(value=copy.deepcopy(d["value"]), unc=d.get("unc"), t=d.get("t"), src=d.get("src"))


def exact(value: Any) -> Est:
    return Est(value=value)


def _est(d: dict | None) -> Est | None:
    return None if d is None else Est.from_dict(d)


def _est_dict(e: Est | None) -> dict | None:
    return None if e is None else e.to_dict()


def pose(p: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0), q: list[float] | None = None) -> dict:
    return {"p": [float(x) for x in p], "q": list(q) if q is not None else list(IDENTITY_QUAT)}


def primitive(
    shape: str,
    size: list[float],
    at: dict | None = None,
    rgba: list[float] | None = None,
    texture: str | None = None,
) -> dict:
    """Primitive geometry. ``size`` uses full extents: box [x, y, z]; sphere [r];
    cylinder/capsule [r, length]; plane [x, y]; ellipsoid [x, y, z]. ``texture`` is a base64 PNG
    drawn on a box's top face (+z), the rest of the box keeps ``rgba``."""
    g = {"kind": "primitive", "shape": shape, "size": [float(s) for s in size], "pose": at or pose()}
    if rgba is not None:
        g["rgba"] = [float(c) for c in rgba]
    if texture is not None:
        g["texture"] = texture
    return g


@dataclass
class Link:
    id: str
    name: str
    geometry: dict[str, list[dict]] = field(default_factory=dict)
    inertial: Est | None = None
    material: Est | None = None
    affordances: list[Est] = field(default_factory=list)
    sensors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "id": self.id,
                "name": self.name,
                "geometry": copy.deepcopy(self.geometry),
                "inertial": _est_dict(self.inertial),
                "material": _est_dict(self.material),
                "affordances": [a.to_dict() for a in self.affordances],
                "sensors": copy.deepcopy(self.sensors) or None,
            }
        )

    @staticmethod
    def from_dict(d: dict) -> Link:
        return Link(
            id=d["id"],
            name=d.get("name", d["id"]),
            geometry=copy.deepcopy(d.get("geometry", {})),
            inertial=_est(d.get("inertial")),
            material=_est(d.get("material")),
            affordances=[Est.from_dict(a) for a in d.get("affordances", [])],
            sensors=copy.deepcopy(d.get("sensors", [])),
        )


@dataclass
class Joint:
    id: str
    parent: str
    child: str
    type: Est
    origin: Est
    q: Est
    qd: Est
    axis: Est | None = None
    limits: Est | None = None
    dynamics: Est | None = None
    locked: Est | None = None
    actuator: dict | None = None

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "id": self.id,
                "parent": self.parent,
                "child": self.child,
                "type": self.type.to_dict(),
                "origin": self.origin.to_dict(),
                "axis": _est_dict(self.axis),
                "limits": _est_dict(self.limits),
                "q": self.q.to_dict(),
                "qd": self.qd.to_dict(),
                "dynamics": _est_dict(self.dynamics),
                "locked": _est_dict(self.locked),
                "actuator": copy.deepcopy(self.actuator),
            }
        )

    @staticmethod
    def from_dict(d: dict) -> Joint:
        return Joint(
            id=d["id"],
            parent=d["parent"],
            child=d["child"],
            type=Est.from_dict(d["type"]),
            origin=Est.from_dict(d["origin"]),
            axis=_est(d.get("axis")),
            limits=_est(d.get("limits")),
            q=Est.from_dict(d["q"]),
            qd=Est.from_dict(d["qd"]),
            dynamics=_est(d.get("dynamics")),
            locked=_est(d.get("locked")),
            actuator=copy.deepcopy(d.get("actuator")),
        )


@dataclass
class RigidTree:
    root: Est
    links: list[Link]
    joints: list[Joint] = field(default_factory=list)
    root_actuator: dict | None = None
    kind: str = "rigid"

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "kind": "rigid",
                "root": self.root.to_dict(),
                "links": [link.to_dict() for link in self.links],
                "joints": [j.to_dict() for j in self.joints],
                "rootActuator": copy.deepcopy(self.root_actuator),
            }
        )

    @staticmethod
    def from_dict(d: dict) -> RigidTree:
        return RigidTree(
            root=Est.from_dict(d["root"]),
            links=[Link.from_dict(x) for x in d["links"]],
            joints=[Joint.from_dict(x) for x in d.get("joints", [])],
            root_actuator=copy.deepcopy(d.get("rootActuator")),
        )

    def link(self, link_id: str) -> Link:
        for link in self.links:
            if link.id == link_id:
                return link
        raise KeyError(link_id)

    def joint(self, joint_id: str) -> Joint:
        for j in self.joints:
            if j.id == joint_id:
                return j
        raise KeyError(joint_id)


@dataclass
class Deformable:
    model: str
    rest: str
    keypoints: Est
    material: Est
    nodes: Est | None = None
    surface: dict | None = None
    kind: str = "deformable"

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "kind": "deformable",
                "model": self.model,
                "rest": self.rest,
                "nodes": _est_dict(self.nodes),
                "keypoints": self.keypoints.to_dict(),
                "surface": copy.deepcopy(self.surface),
                "material": self.material.to_dict(),
            }
        )

    @staticmethod
    def from_dict(d: dict) -> Deformable:
        return Deformable(
            model=d["model"],
            rest=d["rest"],
            nodes=_est(d.get("nodes")),
            keypoints=Est.from_dict(d["keypoints"]),
            surface=copy.deepcopy(d.get("surface")),
            material=Est.from_dict(d["material"]),
        )


Body = RigidTree | Deformable


def body_from_dict(d: dict) -> Body:
    if d.get("kind") == "rigid":
        return RigidTree.from_dict(d)
    if d.get("kind") == "deformable":
        return Deformable.from_dict(d)
    raise ValueError(f"unknown body kind: {d.get('kind')!r}")


@dataclass
class Entity:
    id: str
    role: str
    category: Est
    pose: Est
    body: Body
    twist: Est | None = None
    state_vars: dict[str, Est] = field(default_factory=dict)
    existence: float = 1.0
    asset: dict | None = None

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "id": self.id,
                "role": self.role,
                "category": self.category.to_dict(),
                "pose": self.pose.to_dict(),
                "twist": _est_dict(self.twist),
                "body": self.body.to_dict(),
                "stateVars": {k: v.to_dict() for k, v in self.state_vars.items()},
                "existence": self.existence,
                "asset": copy.deepcopy(self.asset),
            }
        )

    @staticmethod
    def from_dict(d: dict) -> Entity:
        return Entity(
            id=d["id"],
            role=d["role"],
            category=Est.from_dict(d["category"]),
            pose=Est.from_dict(d["pose"]),
            twist=_est(d.get("twist")),
            body=body_from_dict(d["body"]),
            state_vars={k: Est.from_dict(v) for k, v in d.get("stateVars", {}).items()},
            existence=float(d.get("existence", 1.0)),
            asset=copy.deepcopy(d.get("asset")),
        )

    @property
    def rigid(self) -> RigidTree:
        if not isinstance(self.body, RigidTree):
            raise TypeError(f"entity {self.id} is not a rigid tree")
        return self.body


@dataclass
class Relation:
    type: str
    a: dict
    b: dict
    derived: bool
    mode: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "type": self.type,
                "a": copy.deepcopy(self.a),
                "b": copy.deepcopy(self.b),
                "mode": self.mode,
                "derived": self.derived,
                "confidence": self.confidence,
            }
        )

    @staticmethod
    def from_dict(d: dict) -> Relation:
        return Relation(
            type=d["type"],
            a=copy.deepcopy(d["a"]),
            b=copy.deepcopy(d["b"]),
            derived=bool(d["derived"]),
            mode=d.get("mode"),
            confidence=d.get("confidence"),
        )


@dataclass
class SpatialLayers:
    regions: list[dict] = field(default_factory=list)
    occupancy: dict | None = None
    tsdf: dict | None = None
    esdf: dict | None = None
    costmap2d: dict | None = None
    unassigned: str | None = None

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "regions": copy.deepcopy(self.regions),
                "occupancy": copy.deepcopy(self.occupancy),
                "tsdf": copy.deepcopy(self.tsdf),
                "esdf": copy.deepcopy(self.esdf),
                "costmap2d": copy.deepcopy(self.costmap2d),
                "unassigned": self.unassigned,
            }
        )

    @staticmethod
    def from_dict(d: dict) -> SpatialLayers:
        return SpatialLayers(
            regions=copy.deepcopy(d.get("regions", [])),
            occupancy=copy.deepcopy(d.get("occupancy")),
            tsdf=copy.deepcopy(d.get("tsdf")),
            esdf=copy.deepcopy(d.get("esdf")),
            costmap2d=copy.deepcopy(d.get("costmap2d")),
            unassigned=d.get("unassigned"),
        )


@dataclass
class SceneState:
    mode: Mode
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    layers: SpatialLayers = field(default_factory=SpatialLayers)
    version: int = 0
    stamp: float = 0.0
    self_id: str | None = None

    @property
    def frame(self) -> str:
        return "world" if self.mode == "world" else "map"

    def to_dict(self) -> dict:
        return _drop_none(
            {
                "mode": self.mode,
                "version": self.version,
                "stamp": self.stamp,
                "frame": self.frame,
                "entities": [e.to_dict() for e in self.entities],
                "relations": [r.to_dict() for r in self.relations],
                "layers": self.layers.to_dict(),
                "self": self.self_id,
            }
        )

    @staticmethod
    def from_dict(d: dict) -> SceneState:
        return SceneState(
            mode=d["mode"],
            version=int(d.get("version", 0)),
            stamp=float(d.get("stamp", 0.0)),
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            relations=[Relation.from_dict(r) for r in d.get("relations", [])],
            layers=SpatialLayers.from_dict(d.get("layers", {})),
            self_id=d.get("self"),
        )

    def entity(self, entity_id: str) -> Entity:
        for e in self.entities:
            if e.id == entity_id:
                return e
        raise KeyError(entity_id)

    def find(self, entity_id: str) -> Entity | None:
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None
