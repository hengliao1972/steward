"""Piper inverse kinematics on the robot's own kinematic model.

The supplied robot kinematics are compiled to MJCF
and solved with mink, with six revolute joints per arm.

The solver freezes every DOF outside the arm being solved — the other arm's joints and both
fingers — so a left-arm goal never drags the right arm along. Joint limits come from the model
(:class:`mink.ConfigurationLimit`), so a solution is always inside the model's ranges.

Frames: the end-effector frame of an arm is its last revolute link (``L_link6`` / ``R_link6``).
Its +z axis points out through the operational pinch point (the model-declared TCP offset), and
the fingers separate along its x axis (``tests/test_cobot_gripper.py``). The solver's model carries a site
there (``L_pinch`` / ``R_pinch``) for :meth:`PiperIK.solve_body`.

:meth:`PiperIK.solve_body` solves the whole body above the chassis at once -- the torso lift and
the six arm joints -- for the pinch point and the tool orientation, with any tool axis left free
(a tool tilt the task does not fix): the redundant joints settle where they stay nearest the
posture they start from, so successive goals of one motion keep one posture.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from steward.ego.collision_policy import PiperSelfCollisionPolicy, SelfCollisionPolicy
from steward.math3d import matrix_from_quat, quat_angle_between
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState

import mink

WORKING_JOINTS = (0.0, 1.57, -1.35, 0.0, 0.0, 0.0)
ARMS = ("L", "R")
_ENT = "w:robot"  # entity id used to compile the robot-only model; MJCF names are prefixed with it
_DOF_PER_ARM = 6

# Tool poses tried when the requested one is out of the wrist's range: turns about the tool's own
# axis (they move the wrist, not the pinch point), then tilts from a ring of directions.
_TURNS = (0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi)
_TILT_FRACTIONS = (0.35, 0.6, 0.85, 1.0)
_TILT_AZIMUTHS = 8


class IkError(RuntimeError):
    """No joint solution reached the requested pose within the iteration budget."""


class PiperIK:
    """FK/IK for a dual Piper twin, using the robot's own transforms and declared pinch points."""

    def __init__(self, robot: dict, timestep: float = 0.002, self_clearance: float = 0.01,
                 working_joints: tuple[float, ...] = WORKING_JOINTS,
                 collision_policy: SelfCollisionPolicy | None = None) -> None:
        """``self_clearance``: the least distance :meth:`solve_body` keeps between an arm's links
        (beyond its shoulder) and the robot's frame and chassis.
        ``working_joints`` supplies the six-joint fallback seed for either arm, within both arms' limits."""
        self.robot = robot
        self.collision_policy = collision_policy if collision_policy is not None else PiperSelfCollisionPolicy()
        # Only the robot is compiled: IK must not depend on scene furniture.
        ent = dict(self.robot)
        ent["id"] = _ENT
        build = build_mjcf(SceneState(mode="world", entities=[_as_entity(ent)]), timestep=timestep)
        self.tcp_offsets = {arm: next(a["value"]["tcp-offset"]["p"]
                            for link in self.robot["body"]["links"] if link["id"] == f"{arm}_link6"
                            for a in link["affordances"] if isinstance(a["value"], dict) and "tcp-offset" in a["value"])
                            for arm in ARMS}
        self.model = mujoco.MjModel.from_xml_string(_with_pinch_sites(build.xml, self.tcp_offsets), build.assets or None)
        self.data = mink.Configuration(self.model)
        # Asset initial coordinates include the chassis pose, torso and finger opening;
        # MuJoCo's qpos0 alone does not carry those values.
        q = np.array(self.model.qpos0, dtype=float)
        for jname, values in build.initial_qpos.items():
            adr = int(self.model.jnt_qposadr[_joint_id(self.model, jname)])
            for offset, value in enumerate(values):
                q[adr + offset] = float(value)
        self.data.update(q)
        self.dt = timestep
        self._arms = {}
        for arm in ARMS:
            plain = [f"{arm}_joint{i}" for i in range(1, _DOF_PER_ARM + 1)]
            names = [f"{_ENT}::{p}" for p in plain]
            qadrs = []
            dofadrs = []
            for name in names:
                jid = _joint_id(self.model, name)
                qadrs.append(int(self.model.jnt_qposadr[jid]))
                dofadrs.append(int(self.model.jnt_dofadr[jid]))
            self._arms[arm] = {
                "joints": plain,
                "qadr": qadrs,
                "dofadr": dofadrs,
                "ee": f"{_ENT}/{arm}_link6",
                "pinch": f"{_ENT}/{arm}_pinch",
                "limits": [(float(self.model.jnt_range[_joint_id(self.model, n)][0]),
                            float(self.model.jnt_range[_joint_id(self.model, n)][1])) for n in names],
                "geoms": [g for g in range(self.model.ngeom) if mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) in
                    {f"{_ENT}/{arm}_link{k}" for k in range(1, _DOF_PER_ARM + 1)}],
            }
        if len(working_joints) != _DOF_PER_ARM or any(
                not math.isfinite(q) or not lo <= q <= hi
                for arm in ARMS for q, (lo, hi) in zip(working_joints, self.joint_limits(arm))):
            raise ValueError("working_joints must contain six finite angles within both arms' USD limits")
        self.working_joints = tuple(working_joints)
        # Every DOF that is not part of a solved arm is frozen: the base's x, y and yaw, the lift,
        # the other arm and both fingers. Freezing the base matters as much as freeing the arm --
        # an IK that may drive the chassis can "reach" any table by walking the robot there.
        all_dofs = [int(self.model.jnt_dofadr[j]) for j in range(self.model.njnt)]
        lift = _joint_id(self.model, f"{_ENT}::lift")
        self._lift_qadr, self._lift_dof = int(self.model.jnt_qposadr[lift]), int(self.model.jnt_dofadr[lift])
        self._freeze, self._freeze_body = {}, {}
        for arm in ARMS:
            owned = set(self._arms[arm]["dofadr"])
            self._freeze[arm] = mink.DofFreezingTask(
                self.model, dof_indices=[d for d in all_dofs if d not in owned])
            # The torso joins the arm in a whole-body solve; the chassis and the other arm stay put.
            self._freeze_body[arm] = mink.DofFreezingTask(
                self.model, dof_indices=[d for d in all_dofs if d not in owned | {self._lift_dof}])
        self._posture = mink.PostureTask(self.model, cost=1e-2)
        self._limit = mink.ConfigurationLimit(self.model)
        for arm in ARMS:
            pairs = [(self._collidable({f"{_ENT}/{name}" for name in moving}),
                      self._collidable({f"{_ENT}/{name}" for name in obstacles}))
                     for moving, obstacles in self.collision_policy.pairs(robot, arm)]
            self._arms[arm]["avoid"] = mink.CollisionAvoidanceLimit(
                self.model, pairs, minimum_distance_from_collisions=self_clearance,
                collision_detection_distance=self_clearance + 0.05)

    def _collidable(self, bodies: set[str]) -> list[int]:
        """Ids of the geoms of ``bodies`` that take part in contacts."""
        m = self.model
        return [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) in bodies]

    # ------------------------------------------------------------------ queries
    def joint_names(self, arm: str) -> list[str]:
        return list(self._arms[self._check(arm)]["joints"])

    def set_base(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                 lift: float | None = None) -> None:
        """Place the model where the robot is: the chassis pose and the torso height.

        Targets are then given in the world frame, as everything the robot acts on is. A solver
        whose root stayed at the origin would be solving for a robot that never drove.
        """
        q = np.array(self.data.q, dtype=float)
        for name, value in (("root_x", x), ("root_y", y), ("root_yaw", yaw), ("lift", lift)):
            if name == "lift" and value is None:
                continue
            try:
                jid = _joint_id(self.model, f"{_ENT}::{name}")
            except KeyError:  # a fixed-root model has no chassis joints
                continue
            value = float(value)
            lo, hi = self.model.jnt_range[jid]
            if self.model.jnt_limited[jid]:
                value = float(min(max(value, lo), hi))
            q[self.model.jnt_qposadr[jid]] = value
        self.data.update(q)

    def joint_limits(self, arm: str) -> list[tuple[float, float]]:
        return list(self._arms[self._check(arm)]["limits"])

    def neutral(self, arm: str) -> list[float]:
        """A mid-range starting posture: the middle of each joint's allowed range."""
        return [0.5 * (lo + hi) for lo, hi in self.joint_limits(arm)]

    def current(self, arm: str) -> list[float]:
        spec = self._arms[self._check(arm)]
        q = self.data.q
        return [float(q[a]) for a in spec["qadr"]]

    def set_joints(self, arm: str, joints: list[float]) -> None:
        """Seed the solver from measured joint angles (the robot's encoders, not a past solution).

        Measured values are clamped to the URDF ranges. A servo holding against a hard stop
        reports a few ten-thousandths of a radian past the limit, and mink's limit constraint
        reads the whole configuration, not just the arms being solved: one such value in the
        *other* arm makes every solve infeasible (an arm resting on its stop blocked the other
        arm's grasps in the kitchen stacking mission).
        """
        spec = self._arms[self._check(arm)]
        _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], self._within_limits(arm, joints), strict=True):
            q[qadr] = value
        self.data.update(q)

    def fk(self, arm: str, joints: list[float] | None = None, *, lift: float | None = None) -> dict:
        """End-effector pose ``{p: [x, y, z], q: wxyz}`` in the world frame.

        Supplied arm and lift encoders are measured values, including overshoot beyond limits.
        Omitted coordinates retain the current configuration; solver seeds use ``set_joints`` /
        ``set_base`` and their limit clamps instead.
        """
        spec = self._arms[self._check(arm)]
        q = np.array(self.data.q, dtype=float)
        if lift is not None:
            q[self._lift_qadr] = float(lift)
        if joints is not None:
            _check_len(joints, _DOF_PER_ARM, f"{arm} arm joints")
            for qadr, value in zip(spec["qadr"], joints, strict=True):
                q[qadr] = value
        self.data.update(q)
        pose = self.data.get_transform_frame_to_world(spec["ee"], "body")
        p = np.asarray(pose.translation())
        rot = np.asarray(pose.rotation().as_matrix())
        return {"p": [float(v) for v in p], "q": _quat_from_matrix(rot)}

    def lowest(self, arm: str, joints: list[float], beyond_x: float = -math.inf) -> float:
        """Height in the world frame of the lowest point of the arm's links (joint 1 to the flange
        with the gripper housing) at ``joints``; only links reaching past world ``x`` = ``beyond_x``
        count (inf when none does). Spheres and capsules by the part of their axis past it, other
        geometry by its bounding box."""
        spec = self._arms[self._check(arm)]
        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], joints, strict=True):
            q[qadr] = value
        self.data.update(q)
        d, m = self.data.data, self.model
        low = math.inf
        for g in spec["geoms"]:
            rot, pos = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
                # The axis (a point for a sphere) where it lies past beyond_x less the radius,
                # lowered by the radius.
                axis = rot[:, 2] * (size[1] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)
                ends = [pos - axis, pos + axis]
                cut = beyond_x - size[0]
                inside = [e for e in ends if e[0] > cut]
                if not inside:
                    continue
                if len(inside) == 1 and ends[0][0] != ends[1][0]:
                    t = (cut - ends[0][0]) / (ends[1][0] - ends[0][0])
                    inside.append(ends[0] + (ends[1] - ends[0]) * t)
                low = min(low, min(float(e[2]) for e in inside) - float(size[0]))
            else:
                centre = pos + rot @ m.geom_aabb[g][:3]
                half = np.abs(rot) @ m.geom_aabb[g][3:]
                if centre[0] + half[0] > beyond_x:
                    low = min(low, float(centre[2] - half[2]))
        return low

    # --------------------------------------------------------------------- ik
    def solve(self, arm: str, position: list[float], orientation: list[float] | None = None,
              q_init: list[float] | None = None, iterations: int = 120,
              tolerance: float = 5e-3, orientation_tolerance: float = 0.05) -> list[float]:
        """Joint angles (6) that put the arm's end-effector at ``position`` (metres, world).

        ``orientation`` is a wxyz quaternion; when omitted only position is constrained.
        Raises :class:`IkError` when the goal is unreachable within ``tolerance``.
        """
        arm = self._check(arm)
        spec = self._arms[arm]
        if q_init is not None:
            _check_len(q_init, _DOF_PER_ARM, f"{arm} arm seed")
        start = self.current(arm) if q_init is None else self._within_limits(arm, q_init)

        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], start, strict=True):
            q[qadr] = value
        self.data.update(q)

        if orientation is not None:
            _check_len(orientation, 4, "orientation quaternion")
        # A position-only goal still needs a target pose; the zero orientation cost ignores it.
        task = mink.FrameTask(spec["ee"], "body", position_cost=1.0,
                              orientation_cost=0.0 if orientation is None else 1.0,
                              gain=0.6, lm_damping=1e-1)
        task.set_target(_se3(position, orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0]))
        self._posture.set_target(self.data.q)
        tasks = [task, self._posture]

        best, best_err, oerr = None, float("inf"), 0.0
        for _ in range(iterations):
            try:
                velocity = mink.solve_ik(self.data, tasks, self.dt, solver="daqp",
                                         limits=[self._limit], constraints=[self._freeze[arm]])
            except mink.NoSolutionFound as exc:  # solver gave up: report the best pose so far
                raise IkError(f"{arm} arm IK found no solution toward {list(position)}") from exc
            self.data.integrate_inplace(velocity, self.dt)
            err, oerr = self._errors(arm, position, orientation)
            if err <= tolerance and oerr <= orientation_tolerance:
                return self.current(arm)
            if err < best_err:
                best, best_err = self.current(arm), err
        raise IkError(
            f"{arm} arm IK missed {list(position)}: {best_err * 1000:.1f} mm after {iterations} steps"
            + ("" if best is None else f" (best q={[round(v, 3) for v in best]}, rotation {oerr:.3f} rad)")
        )

    def solve_body(self, arm: str, pinch: list[float], tool: list[float], q_init: list[float],
                   lift_init: float, lift_prefer: float, free_axes: tuple[float, float, float] = (1.0, 1.0, 1.0),
                   iterations: int = 300, tolerance: float = 4e-3, orientation_tolerance: float = 0.02,
                   lift_cost: float = 0.01, posture_cost: float = 1e-2, gain: float = 1.0,
                   damping: float = 1e-2, stall: int = 80, settle: int = 80,
                   settled: float = 1e-4) -> tuple[float, list[float]]:
        """Torso height and arm joints putting ``arm``'s pinch point at ``pinch`` (world) with the
        tool at ``tool`` (wxyz quaternion), starting from ``q_init`` and ``lift_init``.

        ``free_axes`` weighs the tool orientation about the tool's own x, y and z axes: 1 holds an
        axis, a small weight leaves it to the solver, which turns the tool about that axis only as
        far as the pinch point needs. The arm joints are drawn to ``q_init`` (``posture_cost``) and
        the torso to ``lift_prefer`` (``lift_cost``). Only the held axes count toward
        ``orientation_tolerance``. A solve whose pinch point has come no 0.2 mm closer in
        ``stall`` steps gives up. Once the goal is met the redundant joints keep settling, within the
        tolerances, for up to ``settle`` steps or until no joint moves ``settled`` rad in a step: a
        posture the goal does not pin down settles where the costs put it, not where convergence
        happened to stop, so neighbouring goals solve to neighbouring postures. The arm's links stay
        ``self_clearance`` clear of the robot's frame and chassis.

        Returns:
            ``(lift, joints)``.

        Raises:
            IkError: The pinch point or a held axis is not met within the tolerances.
        """
        arm = self._check(arm)
        spec = self._arms[arm]
        _check_len(q_init, _DOF_PER_ARM, f"{arm} arm seed")
        q = np.array(self.data.q, dtype=float)
        for qadr, value in zip(spec["qadr"], self._within_limits(arm, q_init), strict=True):
            q[qadr] = value
        lo, hi = self.model.jnt_range[_joint_id(self.model, f"{_ENT}::lift")]
        q[self._lift_qadr] = min(max(float(lift_init), lo), hi)
        self.data.update(q)
        held = np.asarray(free_axes, dtype=float)
        place = mink.FrameTask(spec["pinch"], "site", position_cost=1.0, orientation_cost=0.0, gain=gain, lm_damping=damping)
        turn = mink.FrameTask(spec["pinch"], "site", position_cost=0.0, orientation_cost=held, gain=gain, lm_damping=damping)
        cost = np.full(self.model.nv, posture_cost)
        cost[self._lift_dof] = lift_cost
        posture = mink.PostureTask(self.model, cost=cost)
        target_q = np.array(self.data.q, dtype=float)
        target_q[self._lift_qadr] = min(max(float(lift_prefer), lo), hi)
        posture.set_target(target_q)
        goal = np.asarray(matrix_from_quat(list(tool)), dtype=float).reshape(3, 3)
        best, best_err, since = None, float("inf"), 0
        met: tuple[float, list[float]] | None = None
        steps = 0
        for _ in range(iterations + settle):
            now = self.data.get_transform_frame_to_world(spec["pinch"], "site")
            rot, pos = np.asarray(now.rotation().as_matrix()), np.asarray(now.translation())
            place.set_target(_se3_matrix(rot, pinch))
            turn.set_target(_se3_matrix(goal, pos))
            err = float(np.linalg.norm(pos - np.asarray(pinch, dtype=float)))
            oerr = float(np.max(np.abs(_rotation_vector(rot.T @ goal) * (held >= 0.5))))
            if err <= tolerance and oerr <= orientation_tolerance:
                met = (float(self.data.q[self._lift_qadr]), self.current(arm))
                steps += 1
                if steps > settle:
                    return met
            elif met is not None:
                return met  # settling would leave the goal: the last posture that met it
            elif steps == 0 and _ >= iterations:
                break
            if err < best_err - 2e-4:
                best, best_err, since = self.current(arm), err, 0
            else:
                since += 1
                if since > stall:
                    break
            try:
                velocity = mink.solve_ik(self.data, [place, turn, posture], self.dt, solver="daqp",
                                         limits=[self._limit, spec["avoid"]], constraints=[self._freeze_body[arm]])
            except mink.NoSolutionFound as exc:
                if met is not None:
                    return met
                raise IkError(f"{arm} arm whole-body IK found no solution toward {list(pinch)}") from exc
            self.data.integrate_inplace(velocity, self.dt)
            if met is not None and float(np.max(np.abs(velocity))) * self.dt < settled:
                return met
        if met is not None:
            return met
        raise IkError(f"{arm} arm whole-body IK missed {[round(v, 3) for v in pinch]}: {best_err * 1000:.1f} mm "
                      f"after {iterations} steps" + ("" if best is None else f" (best q={[round(v, 3) for v in best]})"))

    def reachable(self, arm: str, position: list[float], orientation: list[float] | None = None,
                  q_init: list[float] | None = None) -> bool:
        try:
            self.solve(arm, position, orientation, q_init)
        except IkError:
            return False
        return True

    def solve_tool(self, arm: str, position: list[float], orientation: list[float],
                   max_tilt: float = 0.6, q_init: list[float] | None = None) -> tuple[list[float], float]:
        """Reach ``position`` preferring ``orientation``, but accepting a tilt up to ``max_tilt``.

        The Piper wrist has a +-70 deg roll joint (joint5), so a strictly vertical tool is only
        reachable in a narrow band in front of the base; a real pick turns the tool about its own
        axis, or tilts it a little, instead of refusing. Returns ``(joints, tilt_radians)``; the
        tilt is 0 when the strict orientation was reached.

        What is held fixed is the *pinch point* — ``position`` plus the model-declared TCP offset along the tool
        axis — not the end-effector origin. A tilted tool has to move its wrist the other way by
        the TCP offset times ``sin(tilt)``, or the fingers close up to 8.7 cm away from the object while
        every position report still reads "at the target". That displacement is what used to make
        a reachable place miss by 5 cm and then walk its descent into an unreachable waypoint.

        Turning the tool about its own axis moves the wrist a long way and the pinch point not at
        all, so the turn candidates come first; the tilt sweep follows only if every turn fails.
        Seeds are tried in order — the caller's posture, then home — because a 6-DoF solve started
        from a folded posture can miss a reachable goal entirely.
        """
        seeds = [self.current(arm) if q_init is None else list(q_init), list(self.working_joints), self.neutral(arm)]
        strict_errors: list[str] = []
        for seed in seeds:
            try:
                return self.solve(arm, position, orientation, q_init=seed), 0.0
            except IkError as exc:
                strict_errors.append(str(exc))
        pinch = _along(orientation, position, 1.0, self.tcp_offsets[arm])
        failures: list[str] = []
        for turn, tilt, azimuth in _tool_candidates(max_tilt):
            tool = _turn_and_tilt(orientation, turn, tilt, azimuth)
            wrist = _along(tool, pinch, -1.0, self.tcp_offsets[arm])
            for seed in seeds[:2]:
                try:
                    # A tight orientation tolerance here is what keeps the pinch point on target:
                    # the wrist target was computed from the candidate tool pose, so an orientation
                    # that lands 0.05 rad off moves the fingers 7 mm.
                    joints = self.solve(arm, wrist, tool, q_init=seed, iterations=80,
                                        orientation_tolerance=0.02)
                except IkError as exc:
                    failures.append(str(exc))
                    continue
                return joints, float(quat_angle_between(self.fk(arm, joints)["q"], orientation))
        detail = (failures or strict_errors)[-1]
        raise IkError(f"{arm} arm cannot reach {[round(v, 3) for v in position]} in a usable tool "
                      f"orientation: {detail}")

    # ------------------------------------------------------------------ private
    def _within_limits(self, arm: str, joints: list[float]) -> list[float]:
        return [min(max(float(v), lo), hi)
                for v, (lo, hi) in zip(joints, self._arms[arm]["limits"])]

    def _errors(self, arm: str, position: list[float], orientation: list[float] | None) -> tuple[float, float]:
        pose = self.data.get_transform_frame_to_world(self._arms[arm]["ee"], "body")
        err = float(np.linalg.norm(np.asarray(pose.translation()) - np.asarray(position)))
        if orientation is None:
            return err, 0.0
        target = _quat_from_matrix(np.asarray(_se3(position, orientation).rotation().as_matrix()))
        got = _quat_from_matrix(np.asarray(pose.rotation().as_matrix()))
        dot = abs(float(np.dot(target, got)))
        return err, 2.0 * float(np.arccos(min(1.0, dot)))

    @staticmethod
    def _check(arm: str) -> str:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}; expected one of {ARMS}")
        return arm


def _joint_id(model, name: str) -> int:
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"model has no joint {name!r}")
    return jid


def _as_entity(d: dict):
    from steward.scene.schema import Entity

    return Entity.from_dict(d)


def _check_len(values, n: int, what: str) -> None:
    if len(values) != n:
        raise ValueError(f"{what} must have {n} values, got {len(values)}")


def _with_pinch_sites(xml: str, offsets: dict) -> str:
    """The compiled robot's MJCF with a site at each arm's declared pinch offset."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    for arm in ARMS:
        body = next((b for b in root.iter("body") if b.get("name") == f"{_ENT}/{arm}_link6"), None)
        if body is None:
            raise KeyError(f"the robot model has no body {_ENT}/{arm}_link6")
        ET.SubElement(body, "site", name=f"{_ENT}/{arm}_pinch", pos=" ".join(str(v) for v in offsets[arm]), size="0.005")
    return ET.tostring(root, encoding="unicode")


def _se3_matrix(rot: np.ndarray, position) -> "mink.SE3":
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(position, dtype=float)
    return mink.SE3.from_matrix(m)


def _rotation_vector(m: np.ndarray) -> np.ndarray:
    """Axis times angle of rotation matrix ``m``."""
    angle = math.acos(min(1.0, max(-1.0, (float(np.trace(m)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:  # a half turn: the axis is the column of m + I with the largest norm
        cols = m + np.eye(3)
        axis = cols[:, int(np.argmax(np.linalg.norm(cols, axis=0)))]
        norm = float(np.linalg.norm(axis))
    return axis / norm * angle


def _se3(position: list[float], orientation: list[float]) -> "mink.SE3":
    from steward.math3d import matrix_from_quat

    rot = np.asarray(matrix_from_quat(list(orientation)), dtype=float).reshape(3, 3)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(position, dtype=float)
    return mink.SE3.from_matrix(m)


def _quat_from_matrix(m: np.ndarray) -> list[float]:
    """Rotation matrix to a wxyz quaternion, normalised."""
    from steward.math3d import quat_from_matrix as _qf

    return [float(v) for v in _qf(np.asarray(m, dtype=float))]


def _along(orientation: list[float], point: list[float], sign: float, offset: list[float]) -> list[float]:
    """``point`` moved ``sign`` times the tool offset along the orientation's +z axis.

    The pinch point is the model-declared TCP offset along the end-effector's +z, so a tool pose and its pinch
    point are exchanged by this one displacement with ``sign`` +1 or -1.
    """
    from steward.math3d import quat_rotate

    reach = quat_rotate(orientation, offset)
    return [float(p) + sign * float(r) for p, r in zip(point, reach, strict=True)]


def _turn_and_tilt(orientation: list[float], turn: float, tilt: float, azimuth: float) -> list[float]:
    """``orientation`` with the tool turned about its own axis and then tilted off it."""
    from steward.math3d import quat_from_axis_angle, quat_mul

    tool = quat_mul(orientation, quat_from_axis_angle([0.0, 0.0, 1.0], turn))
    if tilt == 0.0:
        return [float(v) for v in tool]
    axis = [math.cos(azimuth), math.sin(azimuth), 0.0]
    return [float(v) for v in quat_mul(tool, quat_from_axis_angle(axis, tilt))]


def _tool_candidates(max_tilt: float):
    """(turn, tilt, azimuth) tool poses to try, cheapest and most upright first."""
    for turn in _TURNS:
        yield turn, 0.0, 0.0
    for fraction in _TILT_FRACTIONS:
        for i in range(_TILT_AZIMUTHS):
            yield 0.0, max_tilt * fraction, 2.0 * math.pi * i / _TILT_AZIMUTHS
