"""WSim: the authoritative World State simulation.

WorldSim owns the MuJoCo model built from a World State scene, advances it at a fixed timestep
(paced against wall time by ``speed``, ``math.inf`` for as fast as possible, or stepped explicitly),
keeps the clock robot control loops and client inputs wait on (:meth:`WorldSim.wait_until`), executes robot actuator commands,
performs Phase 0 ideal grasping, generates each robot's sensor readings, and reads the physics
state back as a World State ``SceneState``.

Phase 0 perception is an *ideal detector*: for every entity (or environment link) visible from a
robot camera, it reports the category and a noisy oriented bounding box in the camera mount
frame. It never reports World State identifiers.
"""

from __future__ import annotations

import copy
import io
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from steward.glrender import close_renderer, encode_image, free_camera
from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_from_yaw, quat_mul, quat_normalize, yaw_from_quat
from steward.scene.extent import entity_body_ids, entity_obb
from steward.scene.mjcf import MjcfBuild, build_mjcf
from steward.scene.schema import Relation, RigidTree, SceneState, exact

_COLLISION_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
HIDDEN_GROUP = 4  # not drawn by MuJoCo's default visualisation options
HOLD_LEASE_S = 2.0  # wall seconds a hold may go unseen (its owner neither waiting nor done) before the world ignores it
_JOINT_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3),
                int(mujoco.mjtJoint.mjJNT_SLIDE): (1, 1), int(mujoco.mjtJoint.mjJNT_HINGE): (1, 1)}


_COLOR_NAMES = {
    "red": (0.85, 0.15, 0.15), "green": (0.15, 0.70, 0.25), "blue": (0.15, 0.35, 0.90), "black": (0.06, 0.06, 0.07),
    "white": (0.95, 0.95, 0.96), "gray": (0.60, 0.62, 0.65), "brown": (0.58, 0.40, 0.24), "beige": (0.80, 0.67, 0.50),
    "yellow": (0.90, 0.85, 0.20),
}


def speed_json(speed: float) -> float | str:
    """``speed`` as JSON: a multiple of real time, or ``"max"`` for as fast as possible."""
    return "max" if speed == math.inf else speed


def color_name(rgba) -> str:
    """Nearest named colour of an RGBA value."""
    rgb = np.asarray(rgba[:3], float)
    return min(_COLOR_NAMES, key=lambda k: float(np.sum((rgb - np.asarray(_COLOR_NAMES[k])) ** 2)))


class EditError(ValueError):
    """A live World State edit was rejected; the world is unchanged."""


@dataclass
class PerceptionConfig:
    # "ideal": the robot receives detections (category and box with noise); "rgbd": the robot
    # receives RGB, depth and anonymous instance masks and must detect and estimate by itself.
    mode: str = "ideal"
    max_range: float = 8.0
    depth_sigma_base: float = 0.0012
    depth_sigma_quadratic: float = 0.0019
    transparent_depth_drop: float = 0.4
    transparent_depth_noise: float = 3.0
    mask_erode_prob: float = 0.3
    min_mask_pixels: int = 25
    pos_sigma_base: float = 0.01
    pos_sigma_per_m: float = 0.02
    size_sigma_rel: float = 0.03
    yaw_sigma: float = 0.05
    odom_pos_sigma: float = 0.003
    odom_yaw_sigma: float = 0.003
    confusion: dict[str, list[str]] = field(default_factory=lambda: {"cup": ["bowl"], "bowl": ["cup"]})
    confusion_prob: float = 0.03
    transparent_detect_prob: float = 0.6


@dataclass
class GraspConfig:
    capture_radius: float = 0.06
    handle_kp: float = 3000.0
    handle_kd: float = 120.0


def _under(obb, point: np.ndarray) -> bool:
    """Whether ``point`` lies over the footprint of an oriented box."""
    local = matrix_from_quat(obb.quat).T @ (np.asarray(point, float) - np.asarray(obb.center))
    return abs(local[0]) <= obb.size[0] / 2 and abs(local[1]) <= obb.size[1] / 2


def _is_pourable(ent) -> bool:
    """A free object whose root link is ``pourable``: a container carried and tipped with its contents."""
    return ent.rigid.root.value == "free" and any(a.value == "pourable" for a in ent.rigid.links[0].affordances)


@dataclass
class _RenderRequest:
    camera: str | dict
    width: int
    height: int
    fmt: str = "png"
    robot: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    png: bytes = b""
    error: Exception | None = None


@dataclass
class _Hold:
    """A robot control loop or client input the world clock waits for (see :meth:`WorldSim.wait_until`)."""
    step: int  # index of the world step it wakes at
    rank: int
    pulls: bool = True  # with ``freeze_idle``, a waiting hold that pulls lets the world step towards it
    woken: bool = False
    seen: float = field(default_factory=time.monotonic)  # wall time its owner last waited or woke


@dataclass
class _RobotState:
    entity_id: str
    closed: bool = False
    attached: str | None = None
    rel: dict | None = None
    saved_contacts: dict[int, tuple[int, int]] = field(default_factory=dict)
    handle: dict | None = None  # articulated grasp: {"entity", "link", "dof", "axis", "q0", "grip0", "lo", "hi"}


class WorldSim:
    def __init__(self, scene: dict, *, timestep: float = 0.002, speed: float = 1.0, seed: int = 0,
                 perception: PerceptionConfig | None = None, grasp: GraspConfig | None = None,
                 overview_camera: dict | None = None, freeze_idle: bool = False):
        self._initial_scene = copy.deepcopy(scene)
        self.timestep = timestep
        self.speed = speed
        # The world stands still while no one waits for world time to pass: no robot acts and no
        # client sleeps, e.g. while the agents think.
        self.freeze_idle = freeze_idle
        self._pulling_waits = 0  # callers of wait_until without a hold, waiting for a later step
        # Steps per locked batch when the loop is behind: short batches keep robot commands within
        # 50 ms of simulated time even when the physics cannot keep up with the requested speed.
        self.max_batch = 25
        self.perception = perception or PerceptionConfig()
        self.grasp = grasp or GraspConfig()
        self._seed = seed
        self._overview = overview_camera
        self.lock = threading.RLock()
        self._clock = threading.Condition(self.lock)
        self._thread: threading.Thread | None = None
        self._running = False
        self.paused = False
        self.events: list[dict] = []
        self._render_requests: queue.Queue[_RenderRequest] = queue.Queue()
        self._load(self._initial_scene)

    # ------------------------------------------------------------------ setup
    def _load(self, scene_dict: dict) -> None:
        self.template = SceneState.from_dict(scene_dict)
        cameras = [self._overview or self._default_overview()]
        self.build: MjcfBuild = build_mjcf(self.template, timestep=self.timestep, extra_cameras=cameras)
        self.model = self.build.compile()
        self.data = mujoco.MjData(self.model)
        self.build.apply_initial(self.model, self.data)
        self.rng = np.random.default_rng(self._seed)
        self.robots = {e.id: _RobotState(e.id) for e in self.template.entities if e.role == "robot"}
        for renderer in getattr(self, "_renderers", {}).values():
            close_renderer(renderer)
        self._epoch = getattr(self, "_epoch", 0) + 1
        self._holds: dict[str, _Hold] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._index_bodies()
        self.step_count = 0

    def _index_bodies(self) -> None:
        self._body_entity: dict[int, str] = {}
        self._body_link: dict[int, str] = {}
        for eid, names in self.build.names.items():
            for bid in entity_body_ids(self.model, names):
                self._body_entity[bid] = eid
            for link_id, body_name in names.bodies.items():
                self._body_link[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)] = link_id

    def _default_overview(self) -> dict:
        span = 6.0
        for ent in self.template.entities:
            if ent.category.value == "building":
                xs = [abs(j.origin.value["p"][0]) for j in ent.rigid.joints]
                ys = [abs(j.origin.value["p"][1]) for j in ent.rigid.joints]
                span = max(max(xs, default=3) * 2, max(ys, default=3) * 2)
        return {"name": "overview", "pos": [0.0, -span * 0.42, span * 1.05], "target": [0.0, 0.15, 0.0], "fovy": 60}

    def reset(self) -> None:
        with self.lock:
            self._load(copy.deepcopy(self._initial_scene))
            self.events = [{"t": 0.0, "type": "episode/reset"}]

    # ------------------------------------------------------------- live edits
    def placement_pose(self, spec: dict, current: dict | None = None) -> dict:
        """Root pose for a placement: ``at`` {x, y, yaw?, z?} on the floor, or ``on`` a support
        entity with ``offset`` [dx, dy] in the support frame. Templates put the root at the bottom
        centre, so the pose z is the floor (0) or the support top. Yaw defaults to the current yaw."""
        yaw = spec.get("yaw")
        if yaw is None:
            yaw = yaw_from_quat(current["q"]) if current is not None else 0.0
        if "on" in spec:
            support = self.template.find(spec["on"])
            if support is None:
                raise EditError(f"unknown support {spec['on']!r}")
            if support.role in ("robot", "actor"):
                raise EditError(f"{support.id} is not a support surface")
            obb = self.entity_obb(support.id)
            dx, dy = (spec.get("offset") or [0.0, 0.0])[:2]
            sp = self._entity_pose(support.id)
            local = pose_compose(sp, {"p": [float(dx), float(dy), 0.0], "q": [1, 0, 0, 0]})
            return {"p": [local["p"][0], local["p"][1], obb.center[2] + obb.size[2] / 2 + 0.002], "q": quat_from_yaw(float(yaw))}
        if "at" in spec:
            at = spec["at"]
            return {"p": [float(at["x"]), float(at["y"]), float(at.get("z", 0.0))], "q": quat_from_yaw(float(yaw))}
        raise EditError("a placement needs 'at' or 'on'")

    def _entity_pose(self, entity_id: str) -> dict:
        ent = self.template.entity(entity_id)
        if ent.rigid.root.value == "free":
            bid = self._root_body(entity_id)
            return {"p": self.data.xpos[bid].tolist(), "q": quat_normalize(self.data.xquat[bid].tolist())}
        if ent.rigid.root.value == "planar":
            x, y, yaw = (self._joint_q(j) for j in self.build.names[entity_id].root_joints)
            return {"p": [x, y, ent.pose.value["p"][2]], "q": quat_from_yaw(yaw)}
        return copy.deepcopy(ent.pose.value)

    def _editable(self, entity_id: str) -> object:
        ent = self.template.find(entity_id)
        if ent is None:
            raise EditError(f"unknown entity {entity_id!r}")
        if ent.role in ("robot", "environment"):
            raise EditError(f"{entity_id} is a {ent.role} and cannot be edited")
        for rs in self.robots.values():
            if rs.attached == entity_id:
                raise EditError(f"{entity_id} is held by {rs.entity_id}")
        return ent

    def _penetrations(self, entity_ids: set[str], tolerance: float = 0.01) -> list[str]:
        out = []
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            e1 = self._body_entity.get(int(self.model.geom_bodyid[con.geom1]))
            e2 = self._body_entity.get(int(self.model.geom_bodyid[con.geom2]))
            if e1 == e2 or (e1 not in entity_ids and e2 not in entity_ids) or con.dist >= -tolerance:
                continue
            out.append(f"{e1} and {e2} would interpenetrate by {-con.dist:.3f} m")
        return out

    def _set_root_pose(self, ent, target: dict) -> None:
        if ent.rigid.root.value == "free":
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[ent.id].root_joints[0])
            qadr, dadr = self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]
            self.data.qpos[qadr : qadr + 3] = target["p"]
            self.data.qpos[qadr + 3 : qadr + 7] = target["q"]
            self.data.qvel[dadr : dadr + 6] = 0.0
        else:
            bid = self._root_body(ent.id)
            self.model.body_pos[bid] = target["p"]
            self.model.body_quat[bid] = target["q"]
        ent.pose = exact({"p": [float(v) for v in target["p"]], "q": quat_normalize(list(target["q"]))})

    def edit_move(self, entity_id: str, placement: dict) -> dict:
        """Move an object or a piece of furniture; objects resting on moved furniture move with it."""
        with self.lock:
            ent = self._editable(entity_id)
            before = self._entity_pose(entity_id)
            target = self.placement_pose(placement, before)
            if placement.get("on") == entity_id:
                raise EditError("an entity cannot be placed on itself")
            delta = pose_compose(target, pose_inverse(before))
            carried = [r.a["entity"] for r in self._relations()
                       if r.type == "supported-by" and r.b["entity"] == entity_id and r.a["entity"] != entity_id]
            carried = [c for c in carried if self.template.entity(c).rigid.root.value == "free"
                       and all(rs.attached != c for rs in self.robots.values())]
            saved = {eid: self._entity_pose(eid) for eid in [entity_id, *carried]}
            self._set_root_pose(ent, target)
            for cid in carried:
                self._set_root_pose(self.template.entity(cid), pose_compose(delta, saved[cid]))
            mujoco.mj_forward(self.model, self.data)
            problems = self._penetrations({entity_id})
            if problems:
                for eid, pose in saved.items():
                    self._set_root_pose(self.template.entity(eid), pose)
                mujoco.mj_forward(self.model, self.data)
                raise EditError("; ".join(problems[:3]))
            self.events.append({"t": self.time, "type": "world/edit", "action": "move", "entity": entity_id,
                                "carried": carried, "plausibility": "god-mode"})
            return {"ok": True, "action": "move", "entity": entity_id, "pose": target, "carried": carried}

    def edit_spawn(self, entity_id: str, template: dict, placement: dict, asset: dict | None = None) -> dict:
        """Add a new entity from an asset template; rebuilds the model and keeps the running state."""
        from steward.assets.library import instantiate

        with self.lock:
            if self.template.find(entity_id) is not None:
                raise EditError(f"entity id {entity_id!r} is already used")
            if not entity_id.startswith("w:"):
                raise EditError("world entity ids start with 'w:'")
            ent = instantiate(template, entity_id, self.placement_pose(placement), asset=asset)
            if ent.role in ("robot", "environment"):
                raise EditError("robots and buildings cannot be added while an episode runs")
            previous = SceneState.from_dict(self.world_state())
            scene = copy.deepcopy(previous)
            scene.entities.append(ent)
            self._rebuild(scene)
            problems = self._penetrations({entity_id})
            if problems:
                self._rebuild(previous)
                raise EditError("; ".join(problems[:3]))
            self.events.append({"t": self.time, "type": "world/edit", "action": "spawn", "entity": entity_id,
                                "asset": (asset or {}).get("id"), "plausibility": "god-mode"})
            return {"ok": True, "action": "spawn", "entity": entity_id, "pose": ent.pose.value}

    def edit_remove(self, entity_id: str) -> dict:
        """Remove an entity; objects resting on it fall. Rebuilds the model and keeps the running state."""
        with self.lock:
            self._editable(entity_id)
            scene = SceneState.from_dict(self.world_state())
            scene.entities = [e for e in scene.entities if e.id != entity_id]
            scene.relations = [r for r in scene.relations if entity_id not in (r.a.get("entity"), r.b.get("entity"))]
            self._rebuild(scene)
            self.events.append({"t": self.time, "type": "world/edit", "action": "remove", "entity": entity_id,
                                "plausibility": "god-mode"})
            return {"ok": True, "action": "remove", "entity": entity_id}

    def _rebuild(self, scene: SceneState) -> None:
        """Recompile the model for a changed entity set, carrying over joint states, actuator
        targets, time and robot grasps by name. Must run on the main thread (renderers)."""
        old_model, old_data = self.model, self.data
        joints, ctrl = {}, {}
        for j in range(old_model.njnt):
            name = mujoco.mj_id2name(old_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            nq, nv = _JOINT_WIDTH[int(old_model.jnt_type[j])]
            qa, da = old_model.jnt_qposadr[j], old_model.jnt_dofadr[j]
            joints[name] = (old_data.qpos[qa : qa + nq].copy(), old_data.qvel[da : da + nv].copy())
        for a in range(old_model.nu):
            ctrl[mujoco.mj_id2name(old_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = float(old_data.ctrl[a])
        sim_time = float(old_data.time)
        robots = self.robots
        for renderer in self._renderers.values():
            close_renderer(renderer)
        self._renderers = {}
        self.template = scene
        self.build = build_mjcf(scene, timestep=self.timestep, extra_cameras=[self._overview or self._default_overview()])
        self.model = self.build.compile()
        self.data = mujoco.MjData(self.model)
        self.build.apply_initial(self.model, self.data)
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in joints:
                qpos, qvel = joints[name]
                qa, da = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
                self.data.qpos[qa : qa + len(qpos)] = qpos
                self.data.qvel[da : da + len(qvel)] = qvel
        for a in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if name in ctrl:
                self.data.ctrl[a] = ctrl[name]
        self.data.time = sim_time
        self._index_bodies()
        self.robots = {e.id: robots.get(e.id, _RobotState(e.id)) for e in scene.entities if e.role == "robot"}
        for rs in self.robots.values():
            rs.saved_contacts = {}
            rs.handle = None
            if rs.attached is not None and scene.find(rs.attached) is None:
                rs.attached, rs.rel = None, None
            if rs.attached is not None:
                self._suspend_contacts(rs)
        self._epoch += 1
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- stepping
    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                self._drive_handles()
                mujoco.mj_step(self.model, self.data)
                self._follow_attached()
                self.step_count += 1

    def step_seconds(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.timestep))))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="wsim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self.lock:
            for r in self._renderers.values():
                close_renderer(r)
            self._renderers.clear()

    def _loop(self) -> None:
        """Pace simulation time against wall time. The anchor is re-taken after a pause, a speed
        change or a reset, so none of them makes the simulation jump forward. The world never steps
        past the earliest wake-up time of a hold, nor at all while a woken hold runs (:meth:`wait_until`)."""
        wall0, sim0, speed0, epoch0 = time.perf_counter(), self.time, self.speed, self._epoch
        while self._running:
            with self._clock:
                now, epoch = self.time, self._epoch
                idle = self.freeze_idle and not self._pulled()
                if self.paused or idle or self.speed != speed0 or epoch != epoch0:
                    wall0, sim0, speed0, epoch0 = time.perf_counter(), now, self.speed, epoch
                n = min(self.max_batch, self._hold_limit() - self._step_now())
                if speed0 != math.inf:
                    n = min(n, self._step_index(sim0 + (time.perf_counter() - wall0) * speed0) - self._step_now())
                if self.paused or idle or n <= 0:
                    self._clock.wait(0.01 if self.paused else 0.001)
                    continue
                self.step(n)
                self._clock.notify_all()
            # Python locks are not fair: running flat out, this thread would take the lock straight
            # back and starve requests that need it (a robot registering its first hold).
            time.sleep(0)

    # ------------------------------------------------------------------ clock
    def wait_until(self, t: float | None = None, hold: str | None = None, timeout: float | None = None,
                   rank: int = 0, seconds: float = 0.0, pulls: bool = True) -> float:
        """Block until the world reaches ``t`` (``None``: ``seconds`` from now); return the simulated time
        then. A ``t`` already passed means now.

        With ``hold``, the caller is a robot control loop or a client input. The world does not step
        past ``t`` before the caller wakes, and not at all while a woken caller runs, until it waits
        again or calls :meth:`release`. Callers wake one at a time in the order of ``(step, rank, hold)``,
        where ``step`` is the first world step at or after ``t``, so what happens depends neither on
        ``speed`` nor on thread timing nor on how ``t`` was rounded. With ``freeze_idle``, the world
        steps only while a caller that ``pulls`` waits; one that does not (perception) only stops it. A hold whose owner is neither
        waiting nor woken for ``HOLD_LEASE_S`` wall seconds is ignored.

        A reset, a stop or a :meth:`release` of the hold ends the wait early. ``timeout`` wall seconds
        without waking raise :class:`TimeoutError`; a hold then stays registered, so ``timeout=0``
        registers it without waiting. Without the clock thread (:meth:`start`) the call steps the
        world itself."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._clock:
            epoch = self._epoch
            step = max(self._step_now(), self._step_index(self.time + seconds if t is None else t))
            if self._thread is None and step > self._step_now():
                self.step(step - self._step_now())
            if hold is not None:
                self._holds[hold] = _Hold(step, rank, pulls)
                self._clock.notify_all()
            pulling = hold is None and pulls and step > self._step_now()
            self._pulling_waits += pulling
            try:
                self._wait_due(step, hold, epoch, end)
            finally:
                self._pulling_waits -= pulling
            if hold is not None and self._epoch == epoch and hold in self._holds:
                self._holds[hold] = _Hold(step, rank, pulls, woken=True)
            return self.time

    def _wait_due(self, step: int, hold: str | None, epoch: int, end: float | None) -> None:
        while (self._running and self._epoch == epoch and (hold is None or hold in self._holds)
               and not self._due(step, hold)):
            left = None if end is None else end - time.monotonic()
            if left is not None and left <= 0:
                raise TimeoutError(f"the world clock did not let {hold or 'the caller'} wake at step {step}")
            if hold is not None:
                self._holds[hold].seen = time.monotonic()
            self._clock.wait(0.05 if left is None else min(0.05, left))

    def release(self, hold: str) -> None:
        """End ``hold``: the world no longer waits for its owner."""
        with self._clock:
            self._holds.pop(hold, None)
            self._clock.notify_all()

    def _step_index(self, t: float) -> int:
        """Index of the first world step at or after time ``t``; a time within a millionth of a step
        of a step counts as on it."""
        return math.ceil(t / self.timestep - 1e-6)

    def _step_now(self) -> int:
        return round(self.time / self.timestep)

    def _live_holds(self) -> dict[str, _Hold]:
        now = time.monotonic()
        return {name: h for name, h in self._holds.items() if now - h.seen < HOLD_LEASE_S}

    def _due(self, step: int, hold: str | None) -> bool:
        """True when the caller of :meth:`wait_until` may wake: ``step`` is reached and, for a hold, no
        other hold runs and none waits ahead of it."""
        if self._step_now() < step:
            return False
        if hold is None:
            return True
        live = self._live_holds()
        if any(h.woken for name, h in live.items() if name != hold):
            return False
        mine = self._holds[hold]
        return min((h.step, h.rank, name) for name, h in live.items() if not h.woken) >= (mine.step, mine.rank, hold)

    def _pulled(self) -> bool:
        """True when a caller waits for world time to pass (see ``freeze_idle``)."""
        return self._pulling_waits > 0 or any(h.pulls and not h.woken for h in self._live_holds().values())

    def _hold_limit(self) -> int | float:
        """Step the world may advance to: the current one while a hold runs, else the earliest step a
        hold waits for (``math.inf`` when none does)."""
        live = self._live_holds().values()
        if any(h.woken for h in live):
            return self._step_now()
        return min((h.step for h in live), default=math.inf)

    # --------------------------------------------------------------- commands
    def command(self, robot_id: str, cmd: dict) -> dict:
        """Apply a robot actuator command: base [x, y, yaw], lift, reach, wrist_yaw, wrist_roll,
        gripper "open"/"close"."""
        with self.lock:
            rs = self._robot(robot_id)
            names = self.build.names[robot_id]
            if "base" in cmd:
                for key, value in zip(("root_x", "root_y", "root_yaw"), cmd["base"], strict=True):
                    self._set_ctrl(names.actuators[key], value)
            for joint in ("lift", "reach", "wrist_yaw", "wrist_roll"):
                if joint in cmd:
                    self._set_ctrl(names.actuators[joint], cmd[joint])
            if "gripper" in cmd:
                if cmd["gripper"] == "close" and not rs.closed:
                    rs.closed = True
                    self._try_attach(rs)
                elif cmd["gripper"] == "open" and rs.closed:
                    rs.closed = False
                    self._release(rs)
            return {"ok": True, "t": self.time}

    def _set_ctrl(self, actuator: str, value: float) -> None:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        lo, hi = self.model.actuator_ctrlrange[aid]
        if self.model.actuator_ctrllimited[aid]:
            value = min(max(value, lo), hi)
        self.data.ctrl[aid] = value

    def _robot(self, robot_id: str) -> _RobotState:
        if robot_id not in self.robots:
            raise KeyError(f"unknown robot {robot_id!r}")
        return self.robots[robot_id]

    def gripper_pose(self, robot_id: str) -> dict:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[robot_id].bodies["gripper"])
        return {"p": self.data.xpos[bid].tolist(), "q": self.data.xquat[bid].tolist()}

    def _try_attach(self, rs: _RobotState) -> None:
        """Ideal grasp: attach the nearest free object within capture range that has nothing resting
        on it (the top of a stack) or is a pourable container (carried with what lies in it), or else grasp the
        nearest drawer handle."""
        grip = np.asarray(self.gripper_pose(rs.entity_id)["p"])
        supporting = {r.b["entity"] for r in self._relations()
                      if r.type == "supported-by" and self.template.entity(r.a["entity"]).rigid.root.value == "free"}
        best, best_d, best_top = None, self.grasp.capture_radius, -math.inf
        for ent in self.template.entities:
            if ent.role != "object" or ent.rigid.root.value != "free":
                continue
            if ent.id in supporting and not (_is_pourable(ent) and not self._covered(ent.id)):
                continue
            if any(other.attached == ent.id for other in self.robots.values()):
                continue
            obb = entity_obb(self.model, self.data, self.build.names[ent.id])
            d = float(np.linalg.norm(np.asarray(obb.center) - grip))
            if d > self.grasp.capture_radius:
                continue
            # Closing from above, the gripper meets the topmost object under it first (a lid before
            # the pieces in the box); elsewhere the nearest object.
            top = obb.center[2] + obb.size[2] / 2 if _under(obb, grip) else -math.inf
            if (top, -d) > (best_top, -best_d):
                best, best_d, best_top = ent.id, d, top
        if best is None:
            handle = self._nearest_handle(grip)
            if handle is not None:
                rs.handle = handle
                self.events.append({"t": self.time, "type": "grasp/handle", "robot": rs.entity_id,
                                    "entity": handle["entity"], "link": handle["link"]})
                return
            self.events.append({"t": self.time, "type": "grasp/miss", "robot": rs.entity_id})
            return
        body = self._root_body(best)
        obj_pose = {"p": self.data.xpos[body].tolist(), "q": self.data.xquat[body].tolist()}
        rs.attached = best
        rs.rel = pose_compose(pose_inverse(self.gripper_pose(rs.entity_id)), obj_pose)
        self._suspend_contacts(rs)
        self.events.append({"t": self.time, "type": "grasp/attach", "robot": rs.entity_id, "entity": best})

    def _suspend_contacts(self, rs: _RobotState) -> None:
        """A carried object moves with the gripper and stops colliding, so it never snags on what it
        is lifted from; a pourable container keeps colliding, so what lies in it stays in it."""
        if _is_pourable(self.template.entity(rs.attached)):
            return
        for gid in range(self.model.ngeom):
            if self._body_entity.get(int(self.model.geom_bodyid[gid])) == rs.attached:
                rs.saved_contacts[gid] = (int(self.model.geom_contype[gid]), int(self.model.geom_conaffinity[gid]))
                self.model.geom_contype[gid] = 0
                self.model.geom_conaffinity[gid] = 0

    def _covered(self, entity_id: str) -> bool:
        """Whether something rests on top of an entity (above its top face), not merely in it."""
        top = self.entity_obb(entity_id)
        z_top = top.center[2] + top.size[2] / 2
        for r in self._relations():
            if r.type == "supported-by" and r.b["entity"] == entity_id:
                other = self.entity_obb(r.a["entity"])
                if other.center[2] - other.size[2] / 2 >= z_top - 0.01:
                    return True
        return False

    def _nearest_handle(self, grip: np.ndarray) -> dict | None:
        """The drawer handle (front-most geom of a ``graspable-handle`` link) within capture range."""
        best, best_d = None, self.grasp.capture_radius
        for ent in self.template.entities:
            for link, joint in self._handle_links(ent):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[ent.id].bodies[link.id])
                geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
                rot = self.data.xmat[bid].reshape(3, 3)
                front = max(geoms, key=lambda g: float((rot.T @ (self.data.geom_xpos[g] - self.data.xpos[bid]))[0]))
                d = float(np.linalg.norm(self.data.geom_xpos[front] - grip))
                if d > best_d:
                    continue
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[ent.id].joints[joint.id])
                lo, hi = self.model.jnt_range[jid]
                best_d = d
                best = {"entity": ent.id, "link": link.id, "dof": int(self.model.jnt_dofadr[jid]),
                        "qadr": int(self.model.jnt_qposadr[jid]), "axis": (rot @ np.asarray(joint.axis.value)).tolist(),
                        "q0": float(self.data.qpos[self.model.jnt_qposadr[jid]]), "grip0": grip.tolist(),
                        "lo": float(lo), "hi": float(hi)}
        return best

    @staticmethod
    def _handle_links(ent):
        if ent.role != "object":
            return []
        joints = {j.child: j for j in ent.rigid.joints if j.type.value == "prismatic"}
        return [(link, joints[link.id]) for link in ent.rigid.links
                if link.id in joints and any(a.value == "graspable-handle" for a in link.affordances)]

    def _drive_handles(self) -> None:
        """A grasped drawer follows the gripper along its slide axis (PD force on the slide joint)."""
        for rs in self.robots.values():
            h = rs.handle
            if h is None:
                continue
            grip = np.asarray(self.gripper_pose(rs.entity_id)["p"])
            target = min(max(h["q0"] + float((grip - np.asarray(h["grip0"])) @ np.asarray(h["axis"])), h["lo"]), h["hi"])
            q, qd = self.data.qpos[h["qadr"]], self.data.qvel[h["dof"]]
            self.data.qfrc_applied[h["dof"]] = self.grasp.handle_kp * (target - q) - self.grasp.handle_kd * qd

    def _release_handle(self, rs: _RobotState) -> None:
        h = rs.handle
        self.data.qfrc_applied[h["dof"]] = 0.0
        self.events.append({"t": self.time, "type": "grasp/handle-release", "robot": rs.entity_id,
                            "entity": h["entity"], "link": h["link"]})
        rs.handle = None

    def _release(self, rs: _RobotState) -> None:
        if rs.handle is not None:
            self._release_handle(rs)
        if rs.attached is None:
            return
        for gid, (ct, ca) in rs.saved_contacts.items():
            self.model.geom_contype[gid] = ct
            self.model.geom_conaffinity[gid] = ca
        jadr = self._free_joint_dof(rs.attached)
        self.data.qvel[jadr : jadr + 6] = 0.0
        self.events.append({"t": self.time, "type": "grasp/release", "robot": rs.entity_id, "entity": rs.attached})
        rs.attached, rs.rel, rs.saved_contacts = None, None, {}

    def _follow_attached(self) -> None:
        """Move each held object rigidly with its gripper. It also takes the gripper's velocity, so
        contacts carry what lies in a held container instead of letting it pass through."""
        for rs in self.robots.values():
            if rs.attached is None:
                continue
            grip = self.gripper_pose(rs.entity_id)
            target = pose_compose(grip, rs.rel)
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[rs.attached].root_joints[0])
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr : qadr + 3] = target["p"]
            self.data.qpos[qadr + 3 : qadr + 7] = target["q"]
            dadr = self.model.jnt_dofadr[jid]
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[rs.entity_id].bodies["gripper"])
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, gid, vel, 0)
            omega, v_grip = vel[:3], vel[3:]
            arm = np.asarray(target["p"]) - np.asarray(grip["p"])
            self.data.qvel[dadr : dadr + 3] = v_grip + np.cross(omega, arm)
            self.data.qvel[dadr + 3 : dadr + 6] = matrix_from_quat(target["q"]).T @ omega

    def _root_body(self, entity_id: str) -> int:
        names = self.build.names[entity_id]
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, next(iter(names.bodies.values())))

    def _free_joint_dof(self, entity_id: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.build.names[entity_id].root_joints[0])
        return int(self.model.jnt_dofadr[jid])

    # ------------------------------------------------------------ world state
    def world_state(self) -> dict:
        """The current World State as a SceneState dictionary (exact values, ``w:`` ids)."""
        with self.lock:
            scene = copy.deepcopy(self.template)
            for ent in scene.entities:
                names = self.build.names[ent.id]
                tree: RigidTree = ent.rigid
                root_type = tree.root.value
                if root_type == "free":
                    bid = self._root_body(ent.id)
                    ent.pose = exact({"p": self.data.xpos[bid].tolist(), "q": quat_normalize(self.data.xquat[bid].tolist())})
                elif root_type == "planar":
                    x, y, yaw = (self._joint_q(j) for j in names.root_joints)
                    z = ent.pose.value["p"][2]
                    ent.pose = exact({"p": [x, y, z], "q": quat_from_yaw(yaw)})
                for joint in tree.joints:
                    if joint.id in names.joints:
                        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, names.joints[joint.id])
                        joint.q = exact([float(self.data.qpos[self.model.jnt_qposadr[jid]])])
                        joint.qd = exact([float(self.data.qvel[self.model.jnt_dofadr[jid]])])
                if ent.id in self.robots:
                    ent.state_vars["gripper"] = exact("closed" if self.robots[ent.id].closed else "open")
            scene.relations = self._relations()
            scene.version = self.step_count
            scene.stamp = self.time
            return scene.to_dict()

    def _joint_q(self, name: str) -> float:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qpos[self.model.jnt_qposadr[jid]])

    def _relations(self) -> list[Relation]:
        rels: list[Relation] = []
        for rs in self.robots.values():
            if rs.attached is not None:
                rels.append(Relation(type="attached", a={"entity": rs.attached}, b={"entity": rs.entity_id, "link": "gripper"},
                                     mode="rigid", derived=False))
        seen = set()
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            e1 = self._body_entity.get(int(self.model.geom_bodyid[con.geom1]))
            e2 = self._body_entity.get(int(self.model.geom_bodyid[con.geom2]))
            if e1 is None or e2 is None or e1 == e2:
                continue
            normal_z = float(con.frame[2])
            if abs(normal_z) < 0.7:
                continue
            upper, lower = (e2, e1) if normal_z > 0 else (e1, e2)
            if (upper, lower) in seen or self._entity_role(upper) != "object":
                continue
            seen.add((upper, lower))
            lower_body = int(self.model.geom_bodyid[con.geom1 if normal_z > 0 else con.geom2])
            b = {"entity": lower}
            link = self._body_link.get(lower_body)
            if link is not None and len(self.template.entity(lower).rigid.links) > 1 and lower != "w:room":
                if self.template.entity(lower).role != "environment":
                    b["link"] = link
            rels.append(Relation(type="supported-by", a={"entity": upper}, b=b, derived=True))
        return rels

    def _entity_role(self, entity_id: str) -> str:
        return self.template.entity(entity_id).role

    def entity_obb(self, entity_id: str, links: list[str] | None = None):
        with self.lock:
            return entity_obb(self.model, self.data, self.build.names[entity_id], links)

    # ---------------------------------------------------------------- sensors
    def camera_pose(self, robot_id: str, sensor_id: str = "head_camera") -> tuple[dict, dict]:
        """World pose of a robot camera mount frame (looks along +x) and its sensor description."""
        ent = self.template.entity(robot_id)
        for link in ent.rigid.links:
            for sensor in link.sensors:
                if sensor["id"] == sensor_id:
                    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.build.names[robot_id].bodies[link.id])
                    body_pose = {"p": self.data.xpos[bid].tolist(), "q": self.data.xquat[bid].tolist()}
                    return pose_compose(body_pose, sensor["mount"]), sensor
        raise KeyError(f"robot {robot_id} has no sensor {sensor_id}")

    def sensors(self, robot_id: str) -> dict:
        """One sensor frame for a robot: odometry, joint encoders, gripper and ideal detections."""
        with self.lock:
            rs = self._robot(robot_id)
            names = self.build.names[robot_id]
            x, y, yaw = (self._joint_q(j) for j in names.root_joints)
            p = self.perception
            odom = {
                "x": x + self.rng.normal(0, p.odom_pos_sigma),
                "y": y + self.rng.normal(0, p.odom_pos_sigma),
                "yaw": yaw + self.rng.normal(0, p.odom_yaw_sigma),
                "sigma": {"pos": p.odom_pos_sigma, "yaw": p.odom_yaw_sigma},
            }
            joints = {jid: self._joint_q(jname) for jid, jname in names.joints.items()}
            cam_pose, sensor = self.camera_pose(robot_id)
            frame = {
                "stamp": self.time,
                "odometry": odom,
                "joints": joints,
                "gripper": {"state": "closed" if rs.closed else "open",
                            "holding": rs.attached is not None or rs.handle is not None,
                            "holdingHandle": rs.handle is not None},
                "camera": {"id": sensor["id"], "intrinsics": sensor["intrinsics"]},
            }
            if self.perception.mode == "rgbd":
                frame["imaging"] = "rgbd"
            else:
                frame["detections"] = self._detections(robot_id, cam_pose, sensor)
            return frame

    def perception_parts(self, ent) -> list[tuple[str, list[str] | None, str]]:
        """What a detector reports for one entity: ``(part id, links, category)``. Building walls and
        drawers are reported as separate parts; everything else as the whole entity."""
        if ent.role == "environment":
            return [(f"{ent.id}/{link.id}", [link.id], "wall") for link in ent.rigid.links if link.id != "floor"]
        drawers = [link.id for link, _ in self._handle_links(ent)]
        if not drawers:
            return [(ent.id, None, ent.category.value)]
        rest = [link.id for link in ent.rigid.links if link.id not in drawers]
        return [(ent.id, rest, ent.category.value)] + [(f"{ent.id}/{d}", [d], "drawer") for d in drawers]

    def rgbd(self, robot_id: str, timeout: float = 10.0) -> bytes:
        """One RGB-D frame of a robot's head camera as compressed ``npz`` bytes: ``rgb`` (H, W, 3)
        uint8, ``depth`` (H, W) uint16 millimetres (0 = no reading), ``mask`` (H, W) uint8 anonymous
        instance ids (0 = background; ids are shuffled every frame and carry no World State
        identity), ``fovy`` and ``header`` (JSON: stamp, odometry, joints, gripper).

        Rendering runs on the main thread like :meth:`render`."""
        if threading.current_thread() is threading.main_thread():
            return self._rgbd_now(robot_id)
        request = _RenderRequest(None, 0, 0, robot=robot_id)
        self._render_requests.put(request)
        if not request.done.wait(timeout):
            raise TimeoutError("rendering needs the main thread to call WorldSim.pump()")
        if request.error is not None:
            raise request.error
        return request.png

    def _rgbd_now(self, robot_id: str) -> bytes:
        with self.lock:
            header = self.sensors(robot_id)
            header.pop("imaging", None)
            header.pop("detections", None)
            _, sensor = self.camera_pose(robot_id)
            width, height = sensor["intrinsics"]["width"], sensor["intrinsics"]["height"]
            key = (width, height)
            if key not in self._renderers:
                self._renderers[key] = mujoco.Renderer(self.model, height, width)
            renderer = self._renderers[key]
            renderer.update_scene(self.data, camera=self.robot_camera_name(robot_id))
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            renderer.enable_segmentation_rendering()
            seg = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            part_of, transparent = self._geom_parts(robot_id)
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        parts = np.where(geom >= 0, part_of[np.clip(geom, 0, None)], -1)
        clear = np.where(geom >= 0, transparent[np.clip(geom, 0, None)], False)
        mask = self._anonymous_masks(parts)
        depth_mm = self._noisy_depth(depth, clear)
        buf = io.BytesIO()
        np.savez_compressed(buf, rgb=rgb, depth=depth_mm, mask=mask, fovy=np.float32(sensor["intrinsics"]["fovy"]),
                            header=np.array(json.dumps(header)))
        return buf.getvalue()

    def _geom_parts(self, robot_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Per geom: the index of the perception part it belongs to (-1 for the floor and the
        observing robot itself), and whether it is transparent."""
        part_of = np.full(self.model.ngeom, -1, dtype=np.int32)
        index = 0
        for ent in self.template.entities:
            if ent.id == robot_id:
                continue
            names = self.build.names[ent.id]
            for _, links, _ in self.perception_parts(ent):
                for bid in entity_body_ids(self.model, names, links):
                    part_of[self.model.geom_bodyid == bid] = index
                index += 1
        transparent = self.model.geom_rgba[:, 3] < 0.5
        return part_of, transparent

    def _anonymous_masks(self, parts: np.ndarray) -> np.ndarray:
        """Instance masks as a class-agnostic segmenter would give them: shuffled ids, ragged
        boundaries, tiny fragments dropped."""
        p = self.perception
        mask = np.zeros(parts.shape, dtype=np.uint8)
        present = [int(v) for v in np.unique(parts) if v >= 0]
        ids = self.rng.permutation(np.arange(1, 256))[: len(present)]
        for part, new_id in zip(present, ids, strict=False):
            sel = parts == part
            if self.rng.random() < p.mask_erode_prob:
                inner = sel.copy()
                inner[1:, :] &= sel[:-1, :]
                inner[:-1, :] &= sel[1:, :]
                inner[:, 1:] &= sel[:, :-1]
                inner[:, :-1] &= sel[:, 1:]
                sel = inner
            if int(sel.sum()) >= p.min_mask_pixels:
                mask[sel] = new_id
        return mask

    def _noisy_depth(self, depth: np.ndarray, clear: np.ndarray) -> np.ndarray:
        """Depth in millimetres with range-dependent noise; transparent surfaces often give no reading."""
        p = self.perception
        valid = (depth > 0.05) & (depth < p.max_range)
        sigma = p.depth_sigma_base + p.depth_sigma_quadratic * np.clip(depth - 0.4, 0.0, None) ** 2
        sigma = np.where(clear, sigma * p.transparent_depth_noise, sigma)
        noisy = depth + self.rng.normal(0.0, 1.0, depth.shape) * sigma
        valid &= ~(clear & (self.rng.random(depth.shape) < p.transparent_depth_drop))
        return np.where(valid, np.round(noisy * 1000.0), 0).astype(np.uint16)

    def _detection_targets(self, robot_id: str):
        for ent in self.template.entities:
            if ent.id == robot_id:
                continue
            for _, links, category in self.perception_parts(ent):
                yield ent, links, category

    def _appearance(self, ent, links: list[str] | None) -> tuple[str, float]:
        """Colour name and opacity of a part, from its largest collision geom."""
        names = self.build.names[ent.id]
        body_ids = set(entity_body_ids(self.model, names, links))
        geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in body_ids and self.model.geom_group[g] != 1]
        if not geoms:
            return "unknown", 1.0
        big = max(geoms, key=lambda g: float(self.model.geom_rbound[g]))
        rgba = self.model.geom_rgba[big]
        return color_name(rgba), float(rgba[3])

    def _detections(self, robot_id: str, cam_pose: dict, sensor: dict) -> list[dict]:
        p = self.perception
        fovy = math.radians(sensor["intrinsics"]["fovy"])
        aspect = sensor["intrinsics"]["width"] / sensor["intrinsics"]["height"]
        half_v = fovy / 2
        half_h = math.atan(math.tan(half_v) * aspect)
        cam_inv = pose_inverse(cam_pose)
        cam_p = np.asarray(cam_pose["p"])
        robot_base = self._root_body(robot_id)
        dets = []
        for ent, links, category in self._detection_targets(robot_id):
            obb = entity_obb(self.model, self.data, self.build.names[ent.id], links)
            target_bodies = set(entity_body_ids(self.model, self.build.names[ent.id], links))
            if not self._visible(obb, cam_inv, cam_p, half_h, half_v, robot_base, target_bodies):
                continue
            rel = pose_compose(cam_inv, obb.pose())
            rng_m = float(np.linalg.norm(rel["p"]))
            sigma = p.pos_sigma_base + p.pos_sigma_per_m * rng_m
            noisy_p = [rel["p"][i] + self.rng.normal(0, sigma) for i in range(3)]
            noisy_q = quat_normalize(quat_mul(rel["q"], quat_from_yaw(self.rng.normal(0, p.yaw_sigma))))
            size = [max(0.005, s * (1 + self.rng.normal(0, p.size_sigma_rel))) for s in obb.size]
            color, alpha = self._appearance(ent, links)
            if alpha < 0.5 and self.rng.random() > p.transparent_detect_prob:
                continue
            label = category
            if category in p.confusion and self.rng.random() < p.confusion_prob:
                label = str(self.rng.choice(p.confusion[category]))
            dets.append({"category": label, "confidence": round(float(0.9 - 0.05 * rng_m / p.max_range), 3),
                         "pose": {"p": noisy_p, "q": noisy_q}, "size": size, "range": rng_m,
                         "color": "clear" if alpha < 0.5 else color})
        return dets

    def _visible(self, obb, cam_inv: dict, cam_p: np.ndarray, half_h: float, half_v: float,
                 robot_base: int, target_bodies: set[int]) -> bool:
        """True when any OBB sample point lies in the camera frustum with an unobstructed ray.

        Rays test collision geometry only (groups 0 and 3), so visual-only parts such as the
        robot's own arm never occlude; the robot base body is excluded explicitly.
        """
        center = np.asarray(obb.center)
        rot = matrix_from_quat(obb.quat)
        half = np.asarray(obb.size) / 2 * 0.9
        samples = [center] + [center + rot @ (half * np.array([sx, sy, sz]))
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        for pt in samples:
            local = pose_compose(cam_inv, {"p": pt.tolist(), "q": [1, 0, 0, 0]})["p"]
            fwd = local[0]
            if fwd < 0.1 or fwd > self.perception.max_range:
                continue
            if abs(math.atan2(local[1], fwd)) > half_h or abs(math.atan2(local[2], fwd)) > half_v:
                continue
            vec = pt - cam_p
            dist = float(np.linalg.norm(vec))
            geomid = np.array([-1], dtype=np.int32)
            hit = mujoco.mj_ray(self.model, self.data, cam_p, vec / dist, _COLLISION_GROUPS, 1, robot_base, geomid)
            if hit < 0 or hit >= dist - 0.02:
                return True
            if int(self.model.geom_bodyid[geomid[0]]) in target_bodies:
                return True
        return False

    # --------------------------------------------------------------- rendering
    @property
    def free_camera_fovy(self) -> float:
        """Vertical field of view in degrees used by free-camera renders."""
        return float(self.model.vis.global_.fovy)

    def render(self, camera: str | dict, width: int = 640, height: int = 480, timeout: float = 10.0,
               fmt: str = "png") -> bytes:
        """Render an RGB image (``fmt`` ``png`` or ``jpeg``) from a named MuJoCo camera, or from a free
        camera given as ``{"lookat": [x, y, z], "distance": d, "azimuth": deg, "elevation": deg}``.

        OpenGL rendering must run on the main thread on macOS. Calls from other threads queue a
        request and wait for the main thread to serve it through :meth:`pump`; without a pumping
        main thread they raise ``TimeoutError`` instead of blocking the simulation.
        """
        if threading.current_thread() is threading.main_thread():
            return self._render_now(camera, width, height, fmt)
        request = _RenderRequest(camera, width, height, fmt)
        self._render_requests.put(request)
        if not request.done.wait(timeout):
            raise TimeoutError("rendering needs the main thread to call WorldSim.pump()")
        if request.error is not None:
            raise request.error
        return request.png

    def pump(self, timeout: float = 0.05) -> None:
        """Serve queued render requests on the calling (main) thread, waiting up to ``timeout``."""
        try:
            request = self._render_requests.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            try:
                if request.robot is not None:
                    request.png = self._rgbd_now(request.robot)
                else:
                    request.png = self._render_now(request.camera, request.width, request.height, request.fmt)
            except Exception as exc:  # noqa: BLE001 - re-raised on the requesting thread
                request.error = exc
            request.done.set()
            try:
                request = self._render_requests.get_nowait()
            except queue.Empty:
                return

    def _render_now(self, camera: str | dict, width: int, height: int, fmt: str = "png") -> bytes:
        with self.lock:
            key = (width, height)
            if key not in self._renderers:
                self._renderers[key] = mujoco.Renderer(self.model, height, width)
            renderer = self._renderers[key]
            if isinstance(camera, dict):
                hidden = self.cutaway_walls(camera)
                saved = {gid: int(self.model.geom_group[gid]) for gid in hidden}
                try:
                    for gid in hidden:
                        self.model.geom_group[gid] = HIDDEN_GROUP
                    renderer.update_scene(self.data, camera=free_camera(camera))
                finally:
                    for gid, group in saved.items():
                        self.model.geom_group[gid] = group
            else:
                renderer.update_scene(self.data, camera=camera)
        return encode_image(renderer.render(), fmt)

    def cutaway_walls(self, spec: dict) -> list[int]:
        """Geoms of environment walls on the camera side of the room, hidden in free-camera views
        so the camera looks into the room (rendering only; physics and robot cameras keep them)."""
        az, el = math.radians(float(spec["azimuth"])), math.radians(float(spec["elevation"]))
        if el < math.radians(-80):
            return []
        toward_camera = np.array([-math.cos(az), -math.sin(az)])
        hidden = []
        for ent in self.template.entities:
            if ent.role != "environment":
                continue
            names = self.build.names[ent.id]
            center = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[ent.rigid.links[0].id])][:2]
            for link in ent.rigid.links[1:]:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, names.bodies[link.id])
                offset = self.data.xpos[bid][:2] - center
                norm = float(np.linalg.norm(offset))
                if norm > 1e-6 and float(offset @ toward_camera) / norm > 0.35:
                    hidden += [gid for gid in range(self.model.ngeom) if int(self.model.geom_bodyid[gid]) == bid]
        return hidden

    def robot_camera_name(self, robot_id: str, sensor_id: str = "head_camera") -> str:
        return self.build.names[robot_id].cameras[sensor_id]

    def status(self) -> dict:
        with self.lock:
            return {
                "t": self.time,
                "paused": self.paused,
                "speed": speed_json(self.speed),
                "freezeIdle": self.freeze_idle,
                "entities": [{"id": e.id, "category": e.category.value, "role": e.role} for e in self.template.entities],
                "robots": {rid: {"holding": rs.attached, "gripper": "closed" if rs.closed else "open"}
                           for rid, rs in self.robots.items()},
                "events": self.events[-20:],
            }
