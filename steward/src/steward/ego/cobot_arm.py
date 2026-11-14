"""The Cobot Magic's planar arm (:mod:`steward.ego.arm`): one Piper arm and the torso lift.

The virtual joints place the pinch point in the robot's own frame -- ``tool_offset + reach``
ahead of the base centre on the arm's own line (``lateral`` = the arm mount's y), ``lift`` above
the floor -- with the tool pointing down, tilted ``wrist_pitch`` forward, turned ``wrist_yaw``
about the vertical and tipped ``wrist_roll`` about the forward axis. IK solves in the base frame
with the torso lift, subject to the model's joint limits, including joint6's finite +-180 deg
travel. If the current posture misses a target, the solver tries continuation and working seeds.
Endpoint checks (``reachable``, ``pitch_for``) solve from the target alone; ``reachable_path``
also checks branch continuity and can start from specified physical encoders.

Encoder readings are turned back into virtual joints by forward kinematics, except that an arm
standing within servo tolerance of the joints solved for the last command reports that command:
a solution meets its pose to a few millimetres and hundredths of a radian, and the skills wait for
their own targets, not for the solver's residual.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from steward.assets.cobot import HOME, LIFT_HOME, TCP_OFFSET
from steward.ego.arm import VIRTUAL_JOINTS, Arm, ArmGeometry
from steward.ego.ik import CobotIK, IkError
from steward.ego.skills import OutOfReach, SkillFailed
from steward.math3d import matrix_from_quat, pose_compose, quat_from_axis_angle, quat_mul

TOOL_DOWN = [0.0, 1.0, 0.0, 0.0]  # link6 +z (through the pinch point) along world -z


@dataclass
class PiperArmConfig:
    arm: str = "L"  # the arm that works; the other stays parked
    # The torso is solved with the arm; it is drawn to a shoulder this high above the pinch point
    # with the tool down, lower by ``shoulder_drop_per_pitch`` per radian of forward tilt (measured:
    # tilted 60 deg, a shoulder 0.25 m below the pinch point tips a container 1.2 rad anywhere from
    # 0.35 to 0.65 m out), so it follows the pinch point as smoothly as a skill moves it.
    shoulder_above: float = 0.15
    shoulder_drop_per_pitch: float = 0.38
    pitch_tolerance: float = 0.03  # how close ``pitch_for`` comes to the least tilt that holds the grasp point
    # ``pitch_for`` tilts this much beyond the least tilt, where it does not stand on the edge of
    # the workspace a grasp point re-estimated a few millimetres away would cross.
    pitch_margin: float = 0.1
    torso_seed_inset: float = 0.05  # the low and high torso seeds stand this far inside its travel
    # A pose that the solver does not reach from the seeds is approached from one it reaches from
    # the working posture -- this far ahead, the tool down -- in this many steps of reach and tilt,
    # each seeded by the last (measured: seeded from the working posture alone it misses a third of
    # the poses the arm reaches, and which ones depends on where the arm stands).
    continuation_reach: float = 0.35
    continuation_steps: int = 6
    # Pinch point ahead of the arm mount: from 0.10 (lifted clear of the frame); tool down 0.50,
    # tilted 60 deg 0.70; planned short of that, where a small step does not make the solver turn
    # the wrist over.
    reach: tuple[float, float] = (0.10, 0.66)
    # The pinch point's height range follows the arm mount (measured: no posture reaches more than
    # 0.39 below it with the torso down, nor 0.42 above it with the torso up).
    lift_below_mount: float = 0.39
    lift_above_mount: float = 0.42
    wrist_yaw_max: float = 3.1
    # Bounds only; what a posture allows is solved (``reachable``): the tool straight down tips
    # less than 1 rad, tilted forward a full turn; a half turn turns what it holds over (``turn_over``).
    wrist_roll_max: float = math.pi
    wrist_pitch_max: float = 1.05
    pitch_unseen_cos: float = 0.1  # |cos roll| below which the tool pitch cannot be read from the posture
    # A pitch read this little below zero (a held load bending the arm off its solved posture) is
    # the tool pointing down: planning accepts no backward tilt, so a reading of -0.002 rad would
    # make every motion from the current posture unreachable.
    pitch_read_slack: float = 0.02
    # Holding something, a small step of the grasp point (a ramp, a tip) that the solver can only
    # follow by turning the wrist over to another posture -- a joint moving this far, or the pinch
    # point leaving the step by this much halfway between the two postures -- is refused: the arm
    # would swing what it holds about on the way (measured: tipping a held box past 1.4 rad, joint4
    # meets its limit and the solution jumps 3.5 rad). The torso alone may move any distance: the
    # arm joints keep the pinch point on the step (measured: retracting 5 cm with the tool tilted,
    # the redundant torso sinks 0.12 m).
    max_joint_step: float = 0.6
    max_swing_m: float = 0.03
    # Turning what it holds over (a half roll, tool tilted forward) the arm goes straight from its
    # posture to the turned-over one in joint space: the wrist joints turn half a turn about the
    # tool's own axis and the pinch point stays near its spot (measured: 3 cm forward halfway,
    # height within 2 mm). A turn whose pinch point would leave its spot by more is refused.
    max_turn_swing_m: float = 0.05
    turn_lower_step: float = 0.02  # height step at which lowering what it turned over is checked
    # Turned over, the elbow hangs below the forearm: lowered, it reaches the surface before the
    # pinch point does (the forearm pressed on the table 19 cm above it). No link comes nearer.
    turn_link_clearance: float = 0.02
    self_clearance: float = 0.01  # the least gap the solver keeps between the arm and the robot's own frame
    small_step: tuple[float, float] = (0.05, 0.2)  # metres, radians: a command this close to the last is a step
    arrived_rad: float = 0.04  # encoders this close to the last solution report its command
    # The torso lift lags the arm joints by more than a centimetre under load; a 1 cm gate made
    # joints_of fall through to FK of a mixed posture and report reach several centimetres off,
    # so drawer grasps refused to close and settle checks saw a phantom error.
    arrived_lift_m: float = 0.06
    # mink will park joint2 on its 3.14 rad stop for low tilted grasps; the servo never gets there
    # and the TCP settles several centimetres off. Prefer a seed with this much room at every stop.
    joint_limit_margin: float = 0.03
    home_speed_rad_s: float = 0.6  # maximum setpoint speed while both empty arms return to zero
    home_settle_s: float = 8.0  # time after the ramp for encoder confirmation
    # Hole-free planar tip after HOME unfold. 0.28 m sits next to the 0.32 m tool-down ridge;
    # settle within a few centimetres lands inside it. Prefer a tip past that ridge.
    working_lift: float = 0.85
    working_reach: float = 0.40


class PiperArm(Arm):
    def __init__(self, robot_model: dict, config: PiperArmConfig | None = None):
        self.config = cfg = config or PiperArmConfig()
        if cfg.home_speed_rad_s <= 0 or cfg.home_settle_s <= 0:
            raise ValueError("arm home speed and settling time must be positive")
        self.ik = CobotIK(robot_model, self_clearance=cfg.self_clearance)
        mount = next(j for j in robot_model["body"]["joints"] if j["id"] == f"{cfg.arm}_mount")
        lift = next(j for j in robot_model["body"]["joints"] if j["id"] == "lift")
        mount_at_zero = pose_compose(lift["origin"]["value"], mount["origin"]["value"])["p"]
        self._mount_height_at_zero = float(mount_at_zero[2])
        self._lift_range = (lift["limits"]["value"]["lo"], lift["limits"]["value"]["hi"])
        lift_min = self._mount_height_at_zero + self._lift_range[0] - cfg.lift_below_mount
        lift_max = self._mount_height_at_zero + self._lift_range[1] + cfg.lift_above_mount
        super().__init__(ArmGeometry(tool_offset=float(mount_at_zero[0]), lateral=float(mount_at_zero[1]), reach_min=cfg.reach[0],
                                     reach_max=cfg.reach[1], lift_min=lift_min, lift_max=lift_max,
                                     wrist_yaw_max=cfg.wrist_yaw_max, wrist_roll_max=cfg.wrist_roll_max,
                                     gripper=f"{cfg.arm}_link6", wrist_pitch_max=cfg.wrist_pitch_max))
        self.names = self.ik.joint_names(cfg.arm)
        self._lock = threading.Lock()
        self._target: dict | None = None  # the virtual joints last commanded
        self._solved: tuple[float, list[float]] | None = None  # (torso, arm joints) solved for them
        self._last: dict = {}  # the latest sensor header seen

    # ------------------------------------------------------------- reading
    def virtualize(self, state: dict) -> dict:
        self._last = state
        out = super().virtualize(state)
        out["joints"] = {**(state.get("joints") or {}), **self.joints_of(state.get("joints") or {}, state.get("odometry"))}
        return out

    def resting(self) -> bool:
        """The planar target is unset before the first command and after joint-space homing."""
        with self._lock:
            return self._target is None

    def deploy_working(self, command: Callable[[dict], None], tick: Callable[[], dict]) -> None:
        """Joint-space unfold through ``HOME`` so the next planar command leaves a working posture.

        PARK / all-zero rest has no safe one-shot IK swing to a working tip (the shoulder hits the
        lift carriage; the servo stalls near 0.21 m FK). ``CobotSkills.park`` uses the same HOME
        waypoint for the opposite fold. Sets the planar ``_target`` to the HOME tip afterward.
        """
        state = tick()
        # ``tick`` is virtualized: ``joints["lift"]`` is TCP height. The torso encoder lives in
        # ``_last`` (set by :meth:`virtualize` before the planar overwrite).
        raw = (self._last.get("joints") or {})
        joints = state.get("joints") or {}
        if not all(n in raw for n in self.names) or "lift" not in raw:
            raise SkillFailed("cannot deploy the arm without its encoders")
        start = [float(raw[n]) for n in self.names]
        torso = float(raw["lift"])
        goal = list(HOME)
        cfg = self.config
        if max(abs(a - b) for a, b in zip(start, goal, strict=True)) >= cfg.arrived_rad:
            duration = 1.5 * max(abs(a - b) for a, b in zip(start, goal, strict=True)) / cfg.home_speed_rad_s
            began = state["stamp"]
            while True:
                state = tick()
                elapsed = state["stamp"] - began
                frac = min(1.0, elapsed / duration) if duration > 0 else 1.0
                share = frac * frac * (3.0 - 2.0 * frac)
                q = [a + (b - a) * share for a, b in zip(start, goal, strict=True)]
                command({"joints": {"lift": torso, **dict(zip(self.names, q, strict=True))}})
                if frac >= 1.0:
                    raw_now = (self._last.get("joints") or {})
                    now = [float(raw_now[n]) for n in self.names]
                    if max(abs(a - b) for a, b in zip(now, goal, strict=True)) < cfg.arrived_rad:
                        command({"joints": {"lift": torso, **dict(zip(self.names, goal, strict=True))}})
                        break
                if elapsed > duration + cfg.home_settle_s:
                    raise SkillFailed("the working arm did not reach its HOME posture; stopped deploying")
        # HOME's own FK tip (~0.51 m @ ~1.34 m) sits on an IK ridge the solver cannot rediscover.
        # Clear the planar cache and step to a hole-free working tip from the HOME joint seed.
        with self._lock:
            self._target = self._solved = None
        tick()  # refresh `_last` encoders at HOME
        tip_lift, tip_reach = cfg.working_lift, cfg.working_reach
        out = self.command({"lift": tip_lift, "reach": tip_reach, "wrist_pitch": 0.0})
        if "joints" in out:
            command(out)
            t0 = tick()["stamp"]
            while True:
                st = tick()
                tip = st.get("joints") or {}
                # Tight reach gate: a 5 cm window around 0.28 accepted the neighbouring 0.32 hole.
                if (abs(float(tip.get("lift", 0.0)) - tip_lift) < 0.08
                        and abs(float(tip.get("reach", 0.0)) - tip_reach) < 0.03):
                    return
                if st["stamp"] - t0 > 20.0:
                    return  # planar follow-ups still start from the HOME seed even if settle is slow

    def home_empty(self, command: Callable[[dict], None], tick: Callable[[], dict]) -> bool:
        """Ramp both empty arms' six joints to zero, keeping lift and fingers unchanged.

        Both grippers must report neither an object nor a handle throughout the move. Missing
        feedback, new attachment or failure to settle stops homing. The planar IK cache is cleared
        before moving so the next command seeds from encoders, including after interruption.
        ``tick`` owns cancellation and the robot clock; commands bypass virtual-joint IK.
        """
        state = tick()
        grips = ("L_link6", "R_link6")

        def loaded(st: dict) -> bool:
            reports = st.get("grippers") or {}
            if any(g not in reports or "holding" not in reports[g] or "holdingHandle" not in reports[g] for g in grips):
                raise SkillFailed("cannot return arms home without both grippers' feedback")
            return any(reports[g].get("holding") or reports[g].get("holdingHandle") for g in grips)

        if loaded(state):
            return False
        names = self.ik.joint_names("L") + self.ik.joint_names("R")
        start = {name: float(state["joints"][name]) for name in names}
        with self._lock:
            self._target = self._solved = None
        cfg = self.config
        duration = 1.5 * max(abs(q) for q in start.values()) / cfg.home_speed_rad_s
        began = state["stamp"]
        while True:
            if loaded(state):
                command({"joints": {name: state["joints"][name] for name in names}})
                raise SkillFailed("arm homing stopped because a gripper became occupied")
            elapsed = state["stamp"] - began
            fraction = min(1.0, elapsed / duration) if duration > 0 else 1.0
            share = fraction * fraction * (3.0 - 2.0 * fraction)
            if fraction == 1.0 and all(abs(state["joints"][name]) < cfg.arrived_rad for name in names):
                command({"joints": dict.fromkeys(names, 0.0)})
                return True
            if elapsed > duration + cfg.home_settle_s:
                command({"joints": {name: state["joints"][name] for name in names}})
                raise SkillFailed("both arms did not reach their zero-joint home posture; stopped homing")
            command({"joints": {name: q * (1.0 - share) for name, q in start.items()}})
            state = tick()

    def joints_of(self, joints: dict, odometry: dict | None = None) -> dict:
        if not all(n in joints for n in self.names) or "lift" not in joints:
            return {}
        arm = [float(joints[n]) for n in self.names]
        torso = float(joints["lift"])
        with self._lock:
            if self._target is not None and self._solved is not None and abs(torso - self._solved[0]) < self.config.arrived_lift_m \
                    and max(abs(a - b) for a, b in zip(arm, self._solved[1], strict=True)) < self.config.arrived_rad:
                return dict(self._target)
            return self._fk(torso, arm)

    def _fk(self, torso: float, arm: list[float]) -> dict:
        self.ik.set_base(0.0, 0.0, 0.0, torso)
        tcp = pose_compose(self.ik.fk(self.config.arm, arm), {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})
        # Inverse of :meth:`_solve_from`: R_tcp = Rz(yaw) Ry(-pitch) R_down Rz(roll). Pitch near a
        # quarter turn hides in the matrix; reuse the last commanded pitch then.
        down = matrix_from_quat(TOOL_DOWN)
        r_tcp = matrix_from_quat(tcp["q"])
        m = r_tcp @ down.T
        if math.hypot(m[2, 0], m[2, 2]) < self.config.pitch_unseen_cos:
            pitch = self._target["wrist_pitch"] if self._target is not None else 0.0
        else:
            pitch = math.atan2(m[2, 0], m[2, 2])
            if abs(pitch) > math.pi / 2:
                pitch = math.atan2(-m[2, 0], -m[2, 2])
            if -self.config.pitch_read_slack <= pitch < 0.0:
                pitch = 0.0
        m_yaw = m @ matrix_from_quat(quat_from_axis_angle([0.0, 1.0, 0.0], pitch))
        yaw = math.atan2(m_yaw[1, 0], m_yaw[0, 0])
        rolled = (down.T @ matrix_from_quat(quat_from_axis_angle([0.0, 1.0, 0.0], pitch))
                  @ matrix_from_quat(quat_from_axis_angle([0.0, 0.0, 1.0], -yaw)) @ r_tcp)
        return {"lift": float(tcp["p"][2]), "reach": float(tcp["p"][0]) - self.geometry.tool_offset,
                "wrist_yaw": yaw, "wrist_roll": math.atan2(rolled[1, 0], rolled[0, 0]),
                "wrist_pitch": pitch}

    # ----------------------------------------------------------- commanding
    def command(self, cmd: dict) -> dict:
        out = {k: v for k, v in cmd.items() if k == "base"}
        if "gripper" in cmd:
            grip = cmd["gripper"]
            out["gripper"] = grip if isinstance(grip, dict) else {self.geometry.gripper: grip}
        wanted = {k: float(cmd[k]) for k in VIRTUAL_JOINTS if k in cmd}
        if wanted:
            torso, arm = self._solve(wanted, bool(cmd.get("turnOver")))
            out["joints"] = {"lift": torso, **dict(zip(self.names, arm, strict=True))}
        return out

    def _solve(self, wanted: dict, turn_over: bool = False) -> tuple[float, list[float]]:
        """Torso and arm joints for ``wanted`` merged into the last target. ``turn_over`` solves it
        from the target alone, as :meth:`turn_over` planned, and lets the arm go there in one move
        while it holds something."""
        joints = self._last.get("joints") or {}
        with self._lock:
            target = dict(self._target) if self._target is not None else None
            if target is None:
                if not all(n in joints for n in self.names) or "lift" not in joints:
                    raise SkillFailed("the arm's encoders have not been read yet")
                # PARK rest FK is ~1.74 m lift / ~0 reach — a wrist-only first command used to seed
                # that tip into `_target` and every later reach inherit an unreachable height. Start
                # from the caller's lift/reach (or a hole-free working band) and seed IK from HOME,
                # not from the folded PARK encoders (a singularity with no useful gradient).
                g = self.geometry
                seed_lift = float(wanted["lift"]) if "lift" in wanted else max(g.lift_min, min(g.lift_max, 0.85))
                seed_reach = float(wanted["reach"]) if "reach" in wanted else max(g.reach_min, 0.28)
                target = {"lift": min(max(seed_lift, g.lift_min), g.lift_max),
                          "reach": min(max(seed_reach, g.reach_min), g.reach_max),
                          "wrist_yaw": 0.0, "wrist_roll": 0.0,
                          "wrist_pitch": float(wanted.get("wrist_pitch", 0.0))}
                joints = {"lift": float(joints.get("lift", LIFT_HOME)),
                          **{n: float(h) for n, h in zip(self.names, HOME, strict=True)}}
            previous, prior = dict(target), self._solved
            target.update(wanted)
            holding = ((self._last.get("grippers") or {}).get(self.geometry.gripper) or {}).get("holding")
            continuous = bool(holding and prior is not None and not turn_over and self._step(previous, target))
            solved, last = (self._ik(target, {}) if turn_over else
                            self._ik(target, joints, prior, continuous=continuous))
            if solved is not None and continuous and self._jumps(prior, solved):
                raise SkillFailed(f"the {self.config.arm} arm would have to turn its wrist over to follow its gripper "
                                  f"{target['reach']:.2f} m ahead at {target['lift']:.2f} m (yaw {target['wrist_yaw']:.2f}, "
                                  f"roll {target['wrist_roll']:.2f}, "
                                  f"pitch {target['wrist_pitch']:.2f}); stopped before it swings what it holds")
            if solved is not None:
                self._target, self._solved = target, solved
                return solved
        raise OutOfReach(f"the {self.config.arm} arm cannot put its gripper {target['reach']:.2f} m ahead at "
                          f"{target['lift']:.2f} m height (wrist yaw {target['wrist_yaw']:.2f}, roll "
                          f"{target['wrist_roll']:.2f}, pitch {target['wrist_pitch']:.2f}) at any torso height: {last}")

    def _step(self, a: dict, b: dict) -> bool:
        """Whether virtual-joint targets ``b`` lie a small step from ``a``."""
        metres, radians = self.config.small_step
        return (max(abs(a[k] - b[k]) for k in ("lift", "reach")) <= metres
                and max(abs(a[k] - b[k]) for k in ("wrist_yaw", "wrist_roll", "wrist_pitch")) <= radians)

    def _jumps(self, a: tuple[float, list[float]], b: tuple[float, list[float]]) -> bool:
        """Whether going from solution ``a`` to ``b`` (torso, arm joints) turns the arm over."""
        if max(abs(x - y) for x, y in zip(a[1], b[1], strict=True)) > self.config.max_joint_step:
            return True
        halfway = ((a[0] + b[0]) / 2, [(x + y) / 2 for x, y in zip(a[1], b[1], strict=True)])
        swing = self._pinch(*halfway) - (self._pinch(*a) + self._pinch(*b)) / 2
        return float(np.linalg.norm(swing)) > self.config.max_swing_m

    def _pinch(self, torso: float, arm: list[float]) -> np.ndarray:
        """The pinch point (base frame) of the arm at ``torso`` height with joints ``arm``."""
        self.ik.set_base(0.0, 0.0, 0.0, torso)
        return np.asarray(pose_compose(self.ik.fk(self.config.arm, arm), {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})["p"])

    def _lowest(self, solution: tuple[float, list[float]], beyond: float) -> float:
        """Height above the floor of the lowest point of the arm's links in ``solution`` (torso, arm
        joints) that reach farther than ``beyond`` ahead of the base centre."""
        self.ik.set_base(0.0, 0.0, 0.0, solution[0])
        return self.ik.lowest(self.config.arm, solution[1], beyond)

    def _ik(self, target: dict, joints: dict, prior: tuple[float, list[float]] | None = None, *, continuous: bool = False
            ) -> tuple[tuple[float, list[float]] | None, IkError | None]:
        """Torso height and arm joints for virtual joints ``target`` in one whole-body solve, seeded
        from ``prior`` (the last solution, so a motion keeps one posture), then ``joints``, then
        approached from the working posture (:meth:`_approached`); then the working posture itself.
        Without ``prior`` and ``joints`` the answer depends on ``target`` alone. With ``continuous``,
        a branch jump does not stop the search: remaining seeds may reach a nearby solution. If
        every solution jumps, return the first for the caller's discontinuity rejection.
        The caller holds the lock."""
        encoders = ([(float(joints["lift"]), [float(joints[n]) for n in self.names])]
                    if "lift" in joints and all(n in joints for n in self.names) else [])
        last: IkError | None = None
        self.ik.set_base(0.0, 0.0, 0.0, None)
        def candidates():
            for seed in ([prior] if prior is not None else []) + encoders:
                yield self._solve_from(target, [seed])
            yield self._approached(target), None
            for seed in self._working_seeds(target):
                yield self._solve_from(target, [seed])

        rejected = None
        for solved, error in candidates():
            if error is not None:
                last = error
            if solved is None:
                continue
            if continuous and prior is not None and self._jumps(prior, solved):
                if rejected is None:
                    rejected = solved
                continue
            # Fresh endpoint solves refuse limit-packed postures (the servo never holds them). A
            # continuous tip may ride near a stop; rejecting those breaks an otherwise valid path.
            if not continuous and self._crowded(solved[1]):
                continue
            return solved, None
        return rejected, last

    def _crowded(self, arm: list[float]) -> bool:
        """Whether any arm joint sits inside ``joint_limit_margin`` of its model stop."""
        margin = self.config.joint_limit_margin
        if margin <= 0:
            return False
        return any(q < lo + margin or q > hi - margin
                   for q, (lo, hi) in zip(arm, self.ik.joint_limits(self.config.arm), strict=True))

    def _approached(self, target: dict) -> tuple[float, list[float]] | None:
        """``target`` solved by stepping reach and tilt to it from ``continuation_reach`` ahead with
        the tool down, each step seeded by the last; None where a step misses."""
        cfg = self.config
        start = {**target, "reach": cfg.continuation_reach, "wrist_pitch": 0.0}
        solved, _ = self._solve_from(start, self._working_seeds(start))
        for k in range(1, cfg.continuation_steps + 1):
            if solved is None:
                return None
            share = k / cfg.continuation_steps
            step = {**target, "reach": start["reach"] + (target["reach"] - start["reach"]) * share,
                    "wrist_pitch": target["wrist_pitch"] * share}
            solved, _ = self._solve_from(step, [solved])
        return solved

    def _working_seeds(self, target: dict) -> list[tuple[float, list[float]]]:
        """The working posture at the torso height ``target`` prefers and near either end of the
        torso's travel: one solve moves the torso only so far from where it starts."""
        lo, hi = self._lift_range
        inset = self.config.torso_seed_inset
        return [(min(max(h, lo), hi), list(HOME)) for h in (self._prefer(target), lo + inset, hi - inset)]

    def _prefer(self, target: dict) -> float:
        """The torso height ``target`` draws the torso to."""
        cfg = self.config
        return (target["lift"] + cfg.shoulder_above - cfg.shoulder_drop_per_pitch * target["wrist_pitch"]
                - self._mount_height_at_zero)

    def _solve_from(self, target: dict, seeds: list[tuple[float, list[float]]]
                    ) -> tuple[tuple[float, list[float]] | None, IkError | None]:
        """``target`` solved from the first of ``seeds`` (torso, arm joints) that reaches it."""
        tcp = [self.geometry.tool_offset + target["reach"], self.geometry.lateral, target["lift"]]
        # Yaw about world z, then pitch the tool down about world y, then roll about the tool's own
        # +z (through the pinch). Rolling about world x before the pitch (the pre-USD composition)
        # asked for poses the manufacturer wrist cannot hold once the joint frames match the USD.
        pitched = quat_mul(quat_from_axis_angle([0.0, 1.0, 0.0], -target["wrist_pitch"]), TOOL_DOWN)
        tool = quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], target["wrist_yaw"]),
                        quat_mul(pitched, quat_from_axis_angle([0.0, 0.0, 1.0], target["wrist_roll"])))
        last: IkError | None = None
        for lift, q in seeds:
            try:
                return self.ik.solve_body(self.config.arm, tcp, tool, q, lift, self._prefer(target)), None
            except IkError as exc:
                last = exc
        return None, last

    def tool_rotation(self, yaw: float, roll: float, pitch: float = 0.0):
        pitched = quat_mul(quat_from_axis_angle([0., 1., 0.], -pitch), TOOL_DOWN)
        return matrix_from_quat(quat_mul(quat_from_axis_angle([0., 0., 1.], yaw),
            quat_mul(pitched, quat_from_axis_angle([0., 0., 1.], roll))))

    def reachable(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0,
                  wrist_pitch: float = 0.0) -> bool:
        # Endpoint existence does not imply a continuous path from the current posture.
        # A solution that only exists packed against a joint stop is not reachable in practice:
        # the servo never gets there and the TCP settles centimetres off the command.
        if not super().reachable(lift, reach, wrist_yaw, wrist_roll, wrist_pitch):
            return False
        with self._lock:
            solved, _ = self._ik({"lift": lift, "reach": reach, "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll,
                                  "wrist_pitch": wrist_pitch}, {})
        return solved is not None and not self._crowded(solved[1])

    def _fresh(self, target: dict) -> tuple[float, list[float]] | None:
        """A non-crowded endpoint solve of ``target``, or None (matches :meth:`reachable`)."""
        solved, _ = self._ik(target, {})
        return None if solved is None or self._crowded(solved[1]) else solved

    def pitch_for(self, lift: float, reach: float, wrist_yaw: float = 0.0, wrist_roll: float = 0.0) -> float | None:
        g = self.geometry
        if not super().reachable(lift, reach, wrist_yaw, wrist_roll, 0.0):
            return None
        at = {"lift": lift, "reach": reach, "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll}
        with self._lock:
            if self._fresh({**at, "wrist_pitch": 0.0}) is not None:
                return 0.0
            # A tilt that holds it there, from the most down; the least one lies between it and none.
            # Limit-packed (crowded) solves are rejected: the servo never holds them, so a pitch that
            # only exists packed against a stop is not a usable ``pitch_for`` answer.
            lo, hi = 0.0, None
            for share in (1.0, 0.75, 0.5):
                if self._fresh({**at, "wrist_pitch": g.wrist_pitch_max * share}) is not None:
                    hi = g.wrist_pitch_max * share
                    break
            if hi is None:
                return None
            while hi - lo > self.config.pitch_tolerance:
                mid = (lo + hi) / 2
                if self._fresh({**at, "wrist_pitch": mid}) is not None:
                    hi = mid
                else:
                    lo = mid
            wide = min(g.wrist_pitch_max, hi + self.config.pitch_margin)
            return wide if self._fresh({**at, "wrist_pitch": wide}) is not None else hi

    def min_lift_for(self, reach: float, wrist_pitch: float = 0.0, *,
                     wrist_yaw: float = 0.0, wrist_roll: float = 0.0,
                     lo: float | None = None, hi: float | None = None) -> float | None:
        """Lowest TCP height at which ``reach`` is freshly reachable with the given wrist, or None.

        The geometric lift_max is often outside the non-crowded USD envelope, so the search first
        walks down from ``hi`` to a feasible seed before bisecting."""
        g = self.geometry
        lo = g.lift_min if lo is None else lo
        hi = g.lift_max if hi is None else hi
        if hi < lo or not super().reachable(hi, reach, wrist_yaw, wrist_roll, wrist_pitch):
            # Geometry may still allow a lower band even when lift_max itself is out of range.
            if not super().reachable((lo + hi) / 2, reach, wrist_yaw, wrist_roll, wrist_pitch):
                return None
        seed = hi
        while seed >= lo and not self.reachable(seed, reach, wrist_yaw, wrist_roll, wrist_pitch):
            seed -= 0.05
        if seed < lo:
            return None
        hi = seed
        while hi - lo > 0.01:
            mid = (lo + hi) / 2
            if self.reachable(mid, reach, wrist_yaw, wrist_roll, wrist_pitch):
                hi = mid
            else:
                lo = mid
        return hi

    def turn_over(self, lift: float, reach: float, wrist_pitch: float, floor: float, surface_top: float,
                  surface_edge: float, wrist_yaw: float = 0.0) -> float | None:
        # Planned as the command runs it: both postures solved from the target alone, the move
        # between them straight in joint space, then down step by step from the turned-over one.
        g = self.geometry
        if not (super().reachable(lift, reach, wrist_yaw, 0.0, wrist_pitch) and g.lift_min <= floor <= lift):
            return None
        at = {"lift": lift, "reach": reach, "wrist_yaw": wrist_yaw, "wrist_pitch": wrist_pitch}
        with self._lock:
            upright, _ = self._ik({**at, "wrist_roll": 0.0}, {})
            over, _ = self._ik({**at, "wrist_roll": math.pi}, {})
            if upright is None or over is None:
                return None
            spot = self._pinch(*upright)
            above = surface_top + self.config.turn_link_clearance
            for k in range(1, 11):
                share = k / 10
                between = (upright[0] + (over[0] - upright[0]) * share,
                           [a + (b - a) * share for a, b in zip(upright[1], over[1], strict=True)])
                if (float(np.linalg.norm(self._pinch(*between) - spot)) > self.config.max_turn_swing_m
                        or self._lowest(between, surface_edge) < above):
                    return None
            low, prior = lift, over
            while low - self.config.turn_lower_step >= floor - 1e-9:
                nxt = low - self.config.turn_lower_step
                solved, _ = self._ik({**at, "lift": nxt, "wrist_roll": math.pi}, {}, prior, continuous=True)
                if solved is None or self._jumps(prior, solved) or self._lowest(solved, surface_edge) < above:
                    break
                low, prior = nxt, solved
            return low

    def reachable_path(self, targets, *, start_joints: dict | None = None) -> bool:
        # Each waypoint is solved from its predecessor; a fresh endpoint solve can miss that branch.
        if not all(Arm.reachable(self, t["lift"], t["reach"], t.get("wrist_yaw", 0.0), t.get("wrist_roll", 0.0),
                                 t.get("wrist_pitch", 0.0)) for t in targets):
            return False
        with self._lock:
            joints = start_joints if start_joints is not None else self._last.get("joints") or {}
            prior = ((float(joints["lift"]), [float(joints[n]) for n in self.names])
                     if start_joints is not None else None)
            for t in targets:
                full = {"wrist_yaw": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, **t}
                solved, _ = self._ik(full, joints, prior, continuous=True)
                if solved is None or (prior is not None and self._jumps(prior, solved)):
                    return False
                prior = solved
        return True



def workspace(arm: PiperArm, reaches: np.ndarray, lifts: np.ndarray, wrist_yaw: float = 0.0,
              wrist_roll: float = 0.0, wrist_pitch: float = 0.0) -> np.ndarray:
    """Which (reach, lift) pairs the arm reaches from its parked posture with the tool down, tilted,
    turned and tipped as given: a boolean grid, rows by lift."""
    ok = np.zeros((len(lifts), len(reaches)), dtype=bool)
    for i, lift in enumerate(lifts):
        for j, reach in enumerate(reaches):
            arm._target, arm._solved = None, None
            arm._last = {"joints": {"lift": LIFT_HOME, **dict(zip(arm.names, HOME, strict=True))}}
            try:
                arm._solve({"lift": float(lift), "reach": float(reach), "wrist_yaw": wrist_yaw, "wrist_roll": wrist_roll,
                            "wrist_pitch": wrist_pitch})
            except SkillFailed:
                continue
            ok[i, j] = True
    return ok
