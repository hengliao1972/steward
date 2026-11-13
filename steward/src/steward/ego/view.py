"""Render the robot's own Ego-State as a 3D scene for the operator console.

The renderer belongs to the robot: the Ego Runtime runs it on its view loop thread and serves the
frames on the Ego API, so it sees nothing but the Ego-State and can run on the robot's computer.

The Ego-State is converted with the same SceneState → MJCF converter that ESim uses, so the
picture is the robot's own model of the world: only perceived entities, coloured by the colour it
perceived (or by category),
fainter when their existence probability is low, the robot body at its own estimated pose, and a
neutral floor for orientation. Walls on the camera side are left out so the camera looks into the
room.

The robot and the floor are compiled into a model together with a pool of box slots; every other
entity (each is one box in an Ego-State) is drawn by a slot whose pose, size and colour are set per
frame, and unused slots are hidden. Tracks appearing and disappearing therefore cost nothing; the
model is compiled again only when the robot model changes or the pool must grow (doubling). All
calls must run on one thread, which owns the OpenGL renderers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from steward.glrender import close_renderer, encode_image, free_camera
from steward.math3d import pose_compose, yaw_from_quat
from steward.scene.mjcf import MjcfBuild, build_mjcf
from steward.scene.schema import SceneState
from steward.viewloop import ViewSettings

CATEGORY_RGB = {
    "table": [0.79, 0.64, 0.42], "counter": [0.54, 0.42, 0.29], "cabinet": [0.63, 0.47, 0.31],
    "sideboard": [0.69, 0.53, 0.38], "cup": [0.50, 0.83, 1.00], "bowl": [0.91, 0.91, 0.92],
    "plate": [0.95, 0.95, 0.95], "wall": [0.55, 0.58, 0.62], "drawer": [0.80, 0.67, 0.50],
    "wine_glass": [0.80, 0.92, 0.98], "mug": [0.12, 0.12, 0.13], "teddy_bear": [0.58, 0.38, 0.20],
    "scissors": [0.70, 0.72, 0.75], "toy_airplane": [0.95, 0.78, 0.10], "pistol": [0.12, 0.12, 0.13],
    "airplane_fuselage": [0.95, 0.78, 0.10], "airplane_wing": [0.85, 0.15, 0.15],
    "airplane_propeller": [0.92, 0.45, 0.10], "airplane_landing_gear": [0.70, 0.72, 0.75],
    "car_body": [0.20, 0.45, 0.85], "car_wheel": [0.12, 0.12, 0.13], "toy_screw": [0.55, 0.58, 0.62],
    "screw_board": [0.86, 0.75, 0.56],
    "electric_screwdriver": [0.90, 0.40, 0.08], "manual": [0.97, 0.95, 0.88], "picture": [0.90, 0.93, 0.97],
    "pill_bottle": [0.94, 0.94, 0.92], "perfume_bottle": [0.95, 0.60, 0.75], "plant": [0.24, 0.52, 0.24],
}
COLOR_RGB = {
    "red": [0.85, 0.15, 0.15], "green": [0.15, 0.70, 0.25], "blue": [0.15, 0.35, 0.90], "black": [0.12, 0.12, 0.13],
    "white": [0.93, 0.93, 0.94], "clear": [0.80, 0.92, 0.98], "yellow": [0.90, 0.85, 0.20],
}
DEFAULT_RGB = [0.62, 0.68, 0.75]
UNKNOWN_ALPHA = 0.3
FLOOR_ID = "e:render-floor"
SLOT_MIN = 32
HIDDEN_GROUP = 4  # not drawn by MuJoCo's default visualisation options


def _slot_entity(k: int) -> dict:
    return {"id": f"e:render-slot-{k}", "role": "object", "category": {"value": "slot"},
            "pose": {"value": {"p": [0.0, 0.0, -10.0], "q": [1, 0, 0, 0]}},
            "body": {"kind": "rigid", "root": {"value": "fixed"}, "links": [{"id": "slot", "name": "slot", "geometry": {
                "collision": [{"kind": "primitive", "shape": "box", "size": [0.1, 0.1, 0.1],
                               "pose": {"p": [0, 0, 0], "q": [1, 0, 0, 0]}, "rgba": [1, 1, 1, 1]}]}}]}}


def ego_render_scene(ego: dict, toward_camera: np.ndarray, room_center: list[float]) -> SceneState:
    """A renderable copy of an Ego-State: colours, a floor, and camera-side walls removed."""
    data = copy.deepcopy(ego)
    entities = []
    for ent in data["entities"]:
        if ent["id"] != data.get("self"):
            if ent["role"] == "environment":
                c = ent["pose"]["value"]["p"]
                offset = np.array([c[0] - room_center[0], c[1] - room_center[1]])
                norm = float(np.linalg.norm(offset))
                if norm > 1e-6 and float(offset @ toward_camera) / norm > 0.35:
                    continue
            seen = (ent.get("stateVars", {}).get("color") or {}).get("value")
            rgb = COLOR_RGB.get(seen) or CATEGORY_RGB.get(ent["category"]["value"], DEFAULT_RGB)
            alpha = 1.0 if ent["role"] == "environment" else 0.1 + 0.9 * float(ent.get("existence", 1.0))
            if ent["category"]["value"] == "unknown":
                alpha *= UNKNOWN_ALPHA  # something unidentified must not hide what is behind it
            for link in ent["body"]["links"]:
                for geom in link["geometry"].get("collision", []):
                    geom["rgba"] = [*rgb, alpha]
        entities.append(ent)
    entities.append({
        "id": FLOOR_ID, "role": "environment", "category": {"value": "floor"},
        "pose": {"value": {"p": [room_center[0], room_center[1], 0.0], "q": [1, 0, 0, 0]}},
        "body": {"kind": "rigid", "root": {"value": "fixed"}, "links": [{"id": "floor", "name": "floor", "geometry": {
            "collision": [{"kind": "primitive", "shape": "box", "size": [12, 12, 0.02],
                           "pose": {"p": [0, 0, -0.01], "q": [1, 0, 0, 0]}, "rgba": [0.24, 0.26, 0.29, 1]}]}}]},
    })
    data["entities"] = entities
    data["relations"] = []
    data["mode"] = "world"
    return SceneState.from_dict(data)


@dataclass
class _Compiled:
    structure: tuple
    build: MjcfBuild
    model: mujoco.MjModel
    data: mujoco.MjData
    slots: list[tuple[int, int]]  # (body id, geom id) per slot
    renderers: dict[tuple[int, int], mujoco.Renderer] = field(default_factory=dict)

    def close(self) -> None:
        for renderer in self.renderers.values():
            close_renderer(renderer)
        self.renderers.clear()


class EgoViewRenderer:
    """Renders one robot's Ego-State from a free camera."""

    def __init__(self) -> None:
        self._compiled: _Compiled | None = None

    def render(self, ego: dict, settings: ViewSettings) -> bytes:
        """Render one JPEG frame of an Ego-State.

        :param ego: the Ego-State as a SceneState dict.
        :param settings: camera and image size.
        :returns: the encoded image.
        """
        camera = settings.camera
        az = math.radians(float(camera["azimuth"]))
        toward_camera = np.zeros(2) if float(camera["elevation"]) < -80 else np.array([-math.cos(az), -math.sin(az)])
        room_center = [float(camera["lookat"][0]), float(camera["lookat"][1])]
        scene = ego_render_scene(ego, toward_camera, room_center)
        fixed = [e for e in scene.entities if e.role == "robot" or e.id == FLOOR_ID or len(e.rigid.links) > 1]
        boxes = [(e, g) for e in scene.entities if e not in fixed for g in e.rigid.links[0].geometry.get("collision", [])]
        capacity = max(SLOT_MIN, 1 << max(0, len(boxes) - 1).bit_length())
        structure = (tuple((e.id, e.rigid.root.value, len(e.rigid.links)) for e in fixed), capacity)
        compiled = self._compiled
        if compiled is None or compiled.structure != structure:
            if compiled is not None:
                compiled.close()
            compiled = self._compile(scene, fixed, capacity, structure)
            self._compiled = compiled
        else:
            self._update(compiled, fixed)
        self._fill_slots(compiled, boxes)
        mujoco.mj_kinematics(compiled.model, compiled.data)
        mujoco.mj_camlight(compiled.model, compiled.data)
        key = (settings.width, settings.height)
        if key not in compiled.renderers:
            compiled.renderers[key] = mujoco.Renderer(compiled.model, settings.height, settings.width)
        renderer = compiled.renderers[key]
        renderer.update_scene(compiled.data, camera=free_camera(camera))
        return encode_image(renderer.render(), "jpeg")

    @staticmethod
    def _compile(scene: SceneState, fixed: list, capacity: int, structure: tuple) -> _Compiled:
        data = scene.to_dict()
        data["entities"] = [e.to_dict() for e in fixed] + [_slot_entity(k) for k in range(capacity)]
        build = build_mjcf(SceneState.from_dict(data))
        model = build.compile()
        mj_data = mujoco.MjData(model)
        build.apply_initial(model, mj_data)
        slots = []
        for k in range(capacity):
            body = build.names[f"e:render-slot-{k}"].bodies["slot"]
            slots.append((mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body),
                          mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}::c0")))
        return _Compiled(structure, build, model, mj_data, slots)

    @staticmethod
    def _fill_slots(compiled: _Compiled, boxes: list) -> None:
        model = compiled.model
        for (bid, gid), item in zip(compiled.slots, [*boxes, *[None] * (len(compiled.slots) - len(boxes))], strict=True):
            if item is None:
                model.geom_group[gid] = HIDDEN_GROUP
                continue
            ent, geom = item
            pose = pose_compose(ent.pose.value, geom.get("pose") or {"p": [0, 0, 0], "q": [1, 0, 0, 0]})
            model.body_pos[bid] = pose["p"]
            model.body_quat[bid] = pose["q"]
            half = [s / 2 for s in geom["size"]]
            model.geom_size[gid] = half
            model.geom_rbound[gid] = float(np.linalg.norm(half))
            model.geom_rgba[gid] = geom.get("rgba", [*DEFAULT_RGB, 1.0])
            model.geom_group[gid] = 0

    @staticmethod
    def _update(compiled: _Compiled, entities: list) -> None:
        model, data = compiled.model, compiled.data
        for ent in entities:
            names = compiled.build.names[ent.id]
            p, q = ent.pose.value["p"], ent.pose.value["q"]
            root = ent.rigid.root.value
            root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[ent.rigid.links[0].id])
            if root == "free":
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.root_joints[0])
                adr = model.jnt_qposadr[jid]
                data.qpos[adr : adr + 3] = p
                data.qpos[adr + 3 : adr + 7] = q
            elif root == "planar":
                for jname, value in zip(names.root_joints, (p[0], p[1], yaw_from_quat(q)), strict=True):
                    data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)]] = value
            else:
                model.body_pos[root_bid] = p
                model.body_quat[root_bid] = q
            for joint in ent.rigid.joints:
                if joint.id in names.joints and joint.q.value:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, names.joints[joint.id])
                    data.qpos[model.jnt_qposadr[jid]] = joint.q.value[0]

    def close(self) -> None:
        if self._compiled is not None:
            self._compiled.close()
            self._compiled = None
