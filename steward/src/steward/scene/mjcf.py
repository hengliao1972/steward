"""SceneState → MuJoCo MJCF converter shared by WSim (World State) and ESim (Ego-State).

Only ``Est.value`` is read, so the same converter builds an exact world model or one sample of
an ego belief. Mapping:

* ``RigidTree.root``: ``fixed`` → body welded to the world; ``free`` → ``freejoint``;
  ``planar`` → slide-x, slide-y and hinge-z joints whose positions are absolute x, y and yaw.
* ``Link`` → ``body``; ``geometry.collision`` → colliding ``geom``; ``geometry.visual`` →
  non-colliding massless ``geom`` in group 1; ``geometry.observed`` is never simulated.
* ``Joint`` → ``hinge`` / ``slide`` / ``ball`` (``fixed`` welds the child body).
* ``Joint.actuator`` and ``RigidTree.rootActuator`` → ``position`` / ``velocity`` / ``motor``.
* Camera ``Sensor`` → ``camera``. A sensor mount frame looks along +x with +z up; the
  converter rotates it into MuJoCo's camera convention (looks along -z, +y up).

Deformable bodies are rejected: Phase 0 does not simulate them.
"""

from __future__ import annotations

import base64
import hashlib
import math
import struct
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from steward.math3d import quat_from_matrix, quat_mul, yaw_from_quat
from steward.scene.schema import Entity, RigidTree, SceneState

_MOUNT_TO_CAMERA = quat_from_matrix(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
_JOINT_KIND = {"revolute": "hinge", "prismatic": "slide", "ball": "ball"}
_DEFAULT_AXIS = {"revolute": [0.0, 0.0, 1.0], "prismatic": [1.0, 0.0, 0.0]}


@dataclass
class EntityNames:
    """MuJoCo element names generated for one entity."""

    root_joints: list[str] = field(default_factory=list)
    bodies: dict[str, str] = field(default_factory=dict)
    joints: dict[str, str] = field(default_factory=dict)
    actuators: dict[str, str] = field(default_factory=dict)
    actuator_joints: dict[str, str] = field(default_factory=dict)
    cameras: dict[str, str] = field(default_factory=dict)
    collision_geoms: list[str] = field(default_factory=list)


@dataclass
class MjcfBuild:
    xml: str
    names: dict[str, EntityNames]
    initial_qpos: dict[str, list[float]]
    assets: dict[str, bytes] = field(default_factory=dict)
    """In-memory files the XML references (texture images)."""

    def compile(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string(self.xml, self.assets or None)

    def apply_initial(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Set initial joint positions and hold position actuators at them."""
        for name, values in self.initial_qpos.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr : adr + len(values)] = values
        for names in self.names.values():
            for key, act_name in names.actuators.items():
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.actuator_joints[key])
                    data.ctrl[aid] = data.qpos[model.jnt_qposadr[jid]]
        mujoco.mj_forward(model, data)


def camera_xyaxes(pos: list[float], target: list[float]) -> str:
    """MJCF ``xyaxes`` for a camera at ``pos`` looking at ``target`` with world +z up."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.6f}" for v in [*right, *up])


def build_mjcf(
    scene: SceneState,
    *,
    timestep: float = 0.002,
    extra_cameras: list[dict] | None = None,
    offscreen: tuple[int, int] = (1280, 960),
    model_name: str = "steward",
) -> MjcfBuild:
    root = ET.Element("mujoco", model=model_name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=f"{timestep}", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(offscreen[0]), offheight=str(offscreen[1]))
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="sun", directional="true", pos="0 0 6", dir="0.3 0.2 -1", diffuse="0.6 0.6 0.6")
    for cam in extra_cameras or []:
        ET.SubElement(
            world, "camera", name=cam["name"], pos=_fmt(cam["pos"]), xyaxes=camera_xyaxes(cam["pos"], cam["target"]),
            fovy=f"{cam.get('fovy', 60)}",
        )
    actuators = ET.SubElement(root, "actuator")
    textures = _Textures(ET.Element("asset"))
    names: dict[str, EntityNames] = {}
    initial: dict[str, list[float]] = {}
    for ent in scene.entities:
        if not isinstance(ent.body, RigidTree):
            raise NotImplementedError(f"entity {ent.id}: deformable bodies are not simulated in Phase 0")
        names[ent.id] = _emit_entity(world, actuators, ent, initial, textures)
    if len(actuators) == 0:
        root.remove(actuators)
    if len(textures.element):
        root.insert(list(root).index(world), textures.element)
    ET.indent(root)
    return MjcfBuild(xml=ET.tostring(root, encoding="unicode"), names=names, initial_qpos=initial,
                     assets=textures.files)


@dataclass
class _Textures:
    """Cube textures for boxes with a top-face image: the image on +z, a plain colour elsewhere.
    Identical images and colours share one file."""

    element: ET.Element
    files: dict[str, bytes] = field(default_factory=dict)
    materials: dict[tuple[str, tuple[float, ...]], str] = field(default_factory=dict)

    def material(self, png_b64: str, rgba: list[float]) -> str:
        colour = tuple(round(float(c), 4) for c in rgba[:3])
        key = (hashlib.sha1(png_b64.encode()).hexdigest()[:16], colour)
        if key in self.materials:
            return self.materials[key]
        top = f"tex-{key[0]}.png"
        image = self.files.setdefault(top, base64.b64decode(png_b64))
        width, height = struct.unpack(">II", image[16:24])  # PNG IHDR: every cube face has the image's size
        side = f"side-{width}x{height}-" + "".join(f"{int(c * 255):02x}" for c in colour) + ".png"
        if side not in self.files:
            self.files[side] = _solid_png(colour, width, height)
        name = f"m{len(self.materials)}"
        ET.SubElement(self.element, "texture", name=name, type="cube", fileup=side, filedown=side, fileleft=side,
                      fileright=side, filefront=top, fileback=side)
        ET.SubElement(self.element, "material", name=name, texture=name, rgba="1 1 1 1")
        self.materials[key] = name
        return name


def _solid_png(colour: tuple[float, ...], width: int, height: int) -> bytes:
    """A PNG of one RGB colour."""
    raw = b"".join(b"\x00" + bytes(int(c * 255) for c in colour) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _emit_entity(world: ET.Element, actuators: ET.Element, ent: Entity, initial: dict,
                 textures: _Textures) -> EntityNames:
    tree = ent.rigid
    names = EntityNames()
    pose = ent.pose.value
    root_type = tree.root.value
    root_link = tree.links[0]
    body_name = f"{ent.id}/{root_link.id}"
    if root_type == "planar":
        body = ET.SubElement(world, "body", name=body_name, pos=_fmt([0.0, 0.0, pose["p"][2]]))
        root_axes = [("root_x", "slide", "1 0 0"), ("root_y", "slide", "0 1 0"), ("root_yaw", "hinge", "0 0 1")]
        values = [pose["p"][0], pose["p"][1], yaw_from_quat(pose["q"])]
        gains = (tree.root_actuator or {}).get("gains", {})
        for i, (key, kind, axis) in enumerate(root_axes):
            jname = f"{ent.id}::{key}"
            attrs = {"name": jname, "type": kind, "axis": axis}
            if "damping" in gains:
                attrs["damping"] = f"{gains['damping'][i]}"
            ET.SubElement(body, "joint", **attrs)
            names.root_joints.append(jname)
            initial[jname] = [values[i]]
            if tree.root_actuator is not None:
                act = {**tree.root_actuator, "gains": {"kp": gains.get("kp", [0, 0, 0])[i]}}
                limits = tree.root_actuator.get("limits", {})
                if "effort" in limits:
                    act["limits"] = {"effort": limits["effort"][i]}
                names.actuators[key] = _emit_actuator(actuators, f"{ent.id}::{key}", jname, act, None)
                names.actuator_joints[key] = jname
    else:
        body = ET.SubElement(world, "body", name=body_name, pos=_fmt(pose["p"]), quat=_fmt(pose["q"]))
        if root_type == "free":
            jname = f"{ent.id}::root"
            ET.SubElement(body, "freejoint", name=jname)
            names.root_joints.append(jname)
    names.bodies[root_link.id] = body_name
    _emit_link(body, ent, root_link.id, names, textures)
    _emit_children(body, actuators, ent, root_link.id, names, initial, textures)
    return names


def _emit_children(parent_el: ET.Element, actuators: ET.Element, ent: Entity, parent_id: str, names: EntityNames,
                   initial: dict, textures: _Textures) -> None:
    tree = ent.rigid
    for joint in tree.joints:
        if joint.parent != parent_id:
            continue
        origin = joint.origin.value
        body_name = f"{ent.id}/{joint.child}"
        body = ET.SubElement(parent_el, "body", name=body_name, pos=_fmt(origin["p"]), quat=_fmt(origin["q"]))
        names.bodies[joint.child] = body_name
        jtype = joint.type.value
        if jtype != "fixed":
            jname = f"{ent.id}::{joint.id}"
            attrs = {"name": jname, "type": _JOINT_KIND[jtype]}
            if jtype != "ball":
                attrs["axis"] = _fmt(joint.axis.value if joint.axis else _DEFAULT_AXIS[jtype])
            if joint.limits is not None:
                attrs["range"] = _fmt([joint.limits.value["lo"], joint.limits.value["hi"]])
            dyn = joint.dynamics.value if joint.dynamics else {}
            for key, attr in (("damping", "damping"), ("friction", "frictionloss"), ("stiffness", "stiffness"),
                              ("springRef", "springref")):
                if key in dyn:
                    attrs[attr] = f"{dyn[key]}"
            if joint.actuator and "armature" in joint.actuator.get("gains", {}):
                attrs["armature"] = f"{joint.actuator['gains']['armature']}"
            ET.SubElement(body, "joint", **attrs)
            names.joints[joint.id] = jname
            q0 = list(joint.q.value)
            if q0:
                initial[jname] = q0
            if joint.actuator is not None:
                lim = joint.limits.value if joint.limits else None
                names.actuators[joint.id] = _emit_actuator(actuators, jname, jname, joint.actuator, lim)
                names.actuator_joints[joint.id] = jname
        _emit_link(body, ent, joint.child, names, textures)
        _emit_children(body, actuators, ent, joint.child, names, initial, textures)


def _emit_actuator(actuators: ET.Element, name: str, joint: str, act: dict, limits: dict | None) -> str:
    kind = act["kind"]
    gains = act.get("gains", {})
    effort = act.get("limits", {}).get("effort")
    attrs = {"name": name, "joint": joint}
    if kind == "position":
        attrs["kp"] = f"{gains.get('kp', 100.0)}"
        if limits is not None:
            attrs["ctrlrange"] = _fmt([limits["lo"], limits["hi"]])
        if effort is not None:
            attrs["forcerange"] = _fmt([-effort, effort])
        ET.SubElement(actuators, "position", **attrs)
    elif kind == "velocity":
        attrs["kv"] = f"{gains.get('kv', 10.0)}"
        if effort is not None:
            attrs["forcerange"] = _fmt([-effort, effort])
        ET.SubElement(actuators, "velocity", **attrs)
    elif kind == "torque":
        attrs["gear"] = f"{act.get('gear', 1.0)}"
        if effort is not None:
            attrs["ctrlrange"] = _fmt([-effort, effort])
        ET.SubElement(actuators, "motor", **attrs)
    else:
        raise ValueError(f"unknown actuator kind {kind!r}")
    return name


def _emit_link(body: ET.Element, ent: Entity, link_id: str, names: EntityNames, textures: _Textures) -> None:
    link = ent.rigid.link(link_id)
    material = link.material.value if link.material else {}
    collision = link.geometry.get("collision", [])
    visual = link.geometry.get("visual", [])
    mass = link.inertial.value.get("mass") if link.inertial else None
    volumes = [_volume(g) for g in collision]
    total_volume = sum(volumes) or 1.0
    body_name = body.get("name")
    for k, g in enumerate(collision):
        attrs = _geom_attrs(g, textures)
        attrs["name"] = f"{body_name}::c{k}"
        if "friction" in material:
            attrs["friction"] = f"{material['friction']} 0.005 0.0001"
        if "contactTime" in material:
            # Stiffer contact (MuJoCo solref time constant): carried contents follow a moving container.
            attrs["solref"] = f"{max(float(material['contactTime']), 0.004)} 1"
        if mass is not None:
            attrs["mass"] = f"{mass * volumes[k] / total_volume}"
        elif "density" in material:
            attrs["density"] = f"{material['density']}"
        if "rgba" not in attrs:
            attrs["rgba"] = "0.6 0.6 0.6 1" if visual else "0.7 0.7 0.7 1"
        if visual:
            attrs["group"] = "3"
        ET.SubElement(body, "geom", **attrs)
        names.collision_geoms.append(attrs["name"])
    for k, g in enumerate(visual):
        attrs = _geom_attrs(g, textures)
        attrs.update(name=f"{body_name}::v{k}", contype="0", conaffinity="0", group="1", density="0")
        ET.SubElement(body, "geom", **attrs)
    if not collision and mass is not None:
        inertia = max(mass * 0.005, 1e-5)
        ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{mass}", diaginertia=_fmt([inertia] * 3))
    for sensor in link.sensors:
        if sensor.get("kind") in ("rgb", "depth", "rgbd"):
            mount = sensor["mount"]
            cam_name = f"{ent.id}::{sensor['id']}"
            fovy = sensor.get("intrinsics", {}).get("fovy", 60.0)
            ET.SubElement(body, "camera", name=cam_name, pos=_fmt(mount["p"]),
                          quat=_fmt(quat_mul(mount["q"], _MOUNT_TO_CAMERA)), fovy=f"{fovy}")
            names.cameras[sensor["id"]] = cam_name


def _geom_attrs(g: dict, textures: _Textures) -> dict:
    if g["kind"] != "primitive":
        raise NotImplementedError(f"{g['kind']} geometry is not simulated in Phase 0")
    shape, size = g["shape"], g["size"]
    if shape in ("box", "ellipsoid"):
        mj_size = [s / 2.0 for s in size]
    elif shape == "sphere":
        mj_size = [size[0]]
    elif shape in ("cylinder", "capsule"):
        mj_size = [size[0], size[1] / 2.0]
    else:
        mj_size = [size[0] / 2.0, size[1] / 2.0, 0.1]
    at = g.get("pose") or {"p": [0, 0, 0], "q": [1, 0, 0, 0]}
    attrs = {"type": shape, "size": _fmt(mj_size), "pos": _fmt(at["p"]), "quat": _fmt(at["q"])}
    if "rgba" in g:
        attrs["rgba"] = _fmt(g["rgba"])
    if "texture" in g:
        attrs["material"] = textures.material(g["texture"], g.get("rgba", [0.7, 0.7, 0.7, 1.0]))
        attrs.pop("rgba", None)
    return attrs


def _volume(g: dict) -> float:
    s = g["size"]
    shape = g["shape"]
    if shape == "box":
        return s[0] * s[1] * s[2]
    if shape == "sphere":
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if shape == "cylinder":
        return math.pi * s[0] ** 2 * s[1]
    if shape == "capsule":
        return math.pi * s[0] ** 2 * s[1] + 4.0 / 3.0 * math.pi * s[0] ** 3
    if shape == "ellipsoid":
        return 4.0 / 3.0 * math.pi * (s[0] / 2) * (s[1] / 2) * (s[2] / 2)
    return 1.0


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)
