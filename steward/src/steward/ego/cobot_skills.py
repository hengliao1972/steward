"""Skills for the Cobot Magic twin: chassis driving, torso lift and IK-driven arm motion.

The verbs here are the cobot's version of :mod:`steward.ego.skills` (which drives the mobile
kitchen robot's base, lift and reach slide). Both expose the same ``start`` / ``wait`` / ``stop``
job interface, so the Ego Runtime can drive either one. The base verbs (:meth:`driving`,
:meth:`navigate`) are the mobile robot's primitives, ported as they are: turn to a heading,
translate to a point, with a stall detector; the twin adds IK-driven arm motion on top.

Grasp geometry comes from the model, not from a constant: the pinch point is
``assets.cobot.TCP_OFFSET`` along link6's +z, so a grasp targets the pinch point and the IK
target is that pose pulled back by the offset. That is why a 1 cm perception error costs 1 cm
here instead of the mobile robot's 6 cm capture radius.

Arm targets are world poses. The solver is told where the chassis is and how high the torso is
before every solve (:meth:`_sync`), so driving somewhere and then reaching for something works in
the same coordinates the belief uses.

Grasping still uses the simulator's ideal attach (the finger joint closes and the world rigidly
attaches what is between the fingers); :meth:`CobotSkills.pick` is what makes the approach and
retreat real.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from steward.assets.cobot import HOME, LIFT_RANGE, TCP_OFFSET, park
from steward.ego.ik import CobotIK, IkError
from steward.math3d import pose_compose, quat_from_axis_angle, wrap_angle

ARM_OF_GRIPPER = {"L_link6": "L", "R_link6": "R"}


class SkillFailed(RuntimeError):
    """A skill could not be carried out; the message is what the operator sees."""


class SkillStopped(RuntimeError):
    """The skill was cancelled."""


@dataclass
class MotionConfig:
    tick_s: float = 0.02
    joint_tol: float = 0.03        # rad, all six joints
    settle_timeout_s: float = 12.0
    stall_s: float = 1.5
    approach_height: float = 0.12  # straight up from the grasp point before descending
    retreat_height: float = 0.15   # lift after closing the fingers
    line_step: float = 0.04        # straight-line descents and lifts are walked in these increments
    max_tool_tilt: float = 0.7     # rad the tool may tilt off the requested orientation (wrist limits)
    tcp_tolerance: float = 0.015   # a solved pose further than this from the pinch point is refused
    grasp_grip_wait_s: float = 0.4
    # Chassis motion, the mobile robot's numbers: the TRACER is not faster in the twin.
    linear_speed: float = 0.25     # m/s
    angular_speed: float = 0.6     # rad/s
    position_tol: float = 0.02     # m, waypoint
    yaw_tol: float = 0.03          # rad
    drive_stall_s: float = 2.0


@dataclass
class Job:
    id: str
    skill: str
    receipt: str
    state: str = "running"
    message: str = ""
    started: float = 0.0
    finished: float = 0.0
    log: list[str] = field(default_factory=list)


class CobotSkills:
    """Runs at most one cobot skill at a time in a worker thread."""

    def __init__(self, command: Callable[[dict], None], state: Callable[[], dict],
                 ik: CobotIK | None = None, config: MotionConfig | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_finish: Callable[[Job], None] | None = None):
        self._command = command
        self._state = state
        self.ik = ik if ik is not None else CobotIK()
        self.config = config or MotionConfig()
        self._on_error = on_error
        self._on_finish = on_finish
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None

    # ------------------------------------------------------------------ job api
    def start(self, skill: str, receipt: dict) -> Job:
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise RuntimeError(f"skill {self.current.skill} ({self.current.id}) is still running")
            job = Job(id=f"job-{next(self._ids)}", skill=skill, receipt=receipt["id"],
                      started=self._state()["stamp"])
            self.jobs[job.id] = job
            self.current = job
            self._stop.clear()
        threading.Thread(target=self._run, args=(job, skill, receipt), name=f"skill-{job.id}",
                         daemon=True).start()
        return job

    def cancel(self) -> None:
        self._stop.set()

    def stop(self) -> dict:
        self._stop.set()
        self._open_all_grippers()
        return {"stopped": self.current.id if self.current and self.current.state == "running" else None}

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.jobs[job_id]
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        return job

    def _run(self, job: Job, skill: str, receipt: dict) -> None:
        try:
            plan = receipt.get("plan", {})
            if skill == "move_pose":
                self.move_pose(job, plan)
            elif skill == "move_joints":
                self.move_joints(job, plan)
            elif skill == "pick":
                self.pick(job, plan)
            elif skill == "place":
                self.place(job, plan)
            elif skill == "push":
                self.push(job, plan)
            elif skill == "drawer":
                self.drawer(job, plan)
            elif skill == "navigate":
                self.navigate(job, plan)
            elif skill == "lift":
                self.set_lift(job, plan)
            elif skill == "park":
                self.park(job, plan)
            elif skill in ("open", "close"):
                self.gripper(job, plan.get("gripper") or plan.get("grip"), close=(skill == "close"))
            else:
                raise SkillFailed(f"unknown skill {skill}")
            job.state, job.message = "succeeded", job.message or f"{skill} completed"
        except SkillStopped:
            job.state, job.message = "stopped", "stopped on request"
        except SkillFailed as exc:
            job.state, job.message = "failed", str(exc)
            self._retreat_on_failure()
        except Exception as exc:  # noqa: BLE001 - reported through the job record and the alert channel
            job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）出现未预料的异常：{type(exc).__name__}: {exc}")
            self._retreat_on_failure()
        finally:
            job.finished = self._state()["stamp"]
            if self._on_finish is not None:
                self._on_finish(job)

    # ------------------------------------------------------------------- verbs
    def _move_above(self, job: Job, arm: str, at: list[float], tool: list[float]) -> list[float]:
        """Reach a pre-grasp pose above ``at``, lowering the approach until the wrist can hold it.

        A pre-grasp pose is the highest point of the motion, and the far side of the table sits at
        the wrist's limit: an unreachable approach must be lowered, not treated as an unreachable
        object. Nothing has moved when the IK refuses, so retrying lower is safe.
        """
        last: SkillFailed | None = None
        for height in (self.config.approach_height, self.config.approach_height * 0.75,
                       self.config.approach_height * 0.5, 0.03):
            point = [at[0], at[1], at[2] + height]
            try:
                self.move_to(job, arm, point, tool)
            except SkillFailed as exc:
                last = exc
                continue
            if height < self.config.approach_height:
                job.log.append(f"{arm} approach lowered to {height:.2f} m above the target")
            return point
        raise last if last is not None else SkillFailed(f"{arm} arm cannot reach above {_fmt(at)}")

    def pick(self, job: Job, plan: dict) -> None:
        """Reach the pinch point over ``plan['at']``, descend, close, and lift clear."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        grip = plan.get("gripper", f"{arm}_link6")
        tool = _tool_down()
        if (self._state().get("grippers") or {}).get(grip, {}).get("state") != "open":
            self.gripper(job, grip, close=False)
        above = self._move_above(job, arm, at, tool)
        job.log.append(f"{arm} approach above {_fmt(above)}")
        self.move_to(job, arm, at, tool, line=True)
        self.gripper(job, grip, close=True)
        lifted = [at[0], at[1], at[2] + self.config.retreat_height]
        job.log.append(f"{arm} lift to {_fmt(lifted)}")
        self.move_to(job, arm, lifted, tool, line=True)
        st = self._state()
        holding = (st.get("grippers") or {}).get(grip, {}).get("holding", False)
        if not holding:
            raise SkillFailed(f"closed the gripper at {_fmt(at)} but grabbed nothing")

    def place(self, job: Job, plan: dict) -> None:
        """Carry the load to ``plan['at']``, lower it, open, and retreat."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        grip = plan.get("gripper", f"{arm}_link6")
        tool = _tool_down()
        try:
            above = self._move_above(job, arm, at, tool)
        except SkillFailed:
            # Already holding the load at table height: cross over at the height it is carried and
            # come straight down, which is what a person does and what the reach limit allows.
            here = self.tcp(arm)
            above = [at[0], at[1], max(here[2], at[2] + 0.03)]
            self.move_to(job, arm, above, tool)
            job.log.append(f"{arm} approach at carry height {above[2]:.3f}")
        job.log.append(f"{arm} carry to {_fmt(above)}")
        self.move_to(job, arm, at, tool, line=True)
        self.gripper(job, grip, close=False)
        self.move_to(job, arm, [at[0], at[1], at[2] + self.config.retreat_height], tool, line=True)

    def push(self, job: Job, plan: dict) -> None:
        """Push an object side-on: approach behind it, then travel ``plan['by']`` along the surface."""
        arm = self._arm(plan)
        start = _xyz(plan["at"], "plan.at")
        travel = _xyz(plan["by"], "plan.by")
        tool = _tool_sideways(plan.get("direction", [1.0, 0.0, 0.0]))
        self.move_to(job, arm, start, tool)
        self.move_to(job, arm, [start[i] + travel[i] for i in range(3)], tool, line=True)

    def drawer(self, job: Job, plan: dict) -> None:
        """Grasp a drawer handle and pull it along its slide, or push it shut.

        ``plan``: ``handle`` is the pinch point on the handle, ``axis`` the direction the drawer
        travels when it opens (world), ``travel`` the metres to move (negative closes it), and
        ``gripper``. The world makes a grasped drawer follow the pinch point on a PD force along
        its slide, so moving the arm is what moves the drawer -- and the job reports the travel
        it was asked for, while the caller checks the drawer itself in World State.
        """
        grip = plan.get("gripper") or plan.get("grip") or "R_link6"
        arm = ARM_OF_GRIPPER.get(grip)
        if arm is None:
            raise SkillFailed(f"unknown gripper {grip!r}; expected one of {sorted(ARM_OF_GRIPPER)}")
        handle = _xyz(plan["handle"], "plan.handle")
        axis = _xyz(plan["axis"], "plan.axis")
        travel = float(plan["travel"])
        tool = _tool_down()
        if (self._state().get("grippers") or {}).get(grip, {}).get("state") != "open":
            self.gripper(job, grip, close=False)
        above = [handle[0], handle[1], handle[2] + self.config.approach_height]
        self.move_to(job, arm, above, tool)
        self.move_to(job, arm, handle, tool, line=True)
        self.gripper(job, grip, close=True)
        if not (self._state().get("grippers") or {}).get(grip, {}).get("holdingHandle"):
            self.gripper(job, grip, close=False)
            self.move_to(job, arm, above, tool, line=True)
            raise SkillFailed(f"closed at {_fmt(handle)} but no drawer handle was between the fingers")
        job.log.append(f"grasped the handle at {_fmt(handle)}")
        here = self.tcp(arm)
        steps = max(1, int(abs(travel) / 0.03))
        for i in range(1, steps + 1):
            point = [here[j] + axis[j] * travel * i / steps for j in range(3)]
            self.move_to(job, arm, point, tool)
        self.gripper(job, grip, close=False)
        end = [here[j] + axis[j] * travel for j in range(3)]
        self.move_to(job, arm, [end[0], end[1], end[2] + self.config.retreat_height], tool, line=True)
        verb = "opened" if travel >= 0 else "closed"
        job.message = f"{verb} the drawer by {abs(travel) * 100:.0f} cm"

    def gripper(self, job: Job, grip: str, close: bool) -> None:
        if grip not in ARM_OF_GRIPPER:
            raise SkillFailed(f"unknown gripper {grip!r}; expected one of {sorted(ARM_OF_GRIPPER)}")
        self._command({"gripper": {grip: "close" if close else "open"}})
        self._settle_fingers(close)
        job.log.append(f"{grip} {'closed' if close else 'opened'}")

    # ------------------------------------------------------------- chassis verbs
    def navigate(self, job: Job, plan: dict) -> None:
        """Drive to ``plan['at']`` = [x, y, yaw] in the world, through ``plan['path']`` if given.

        The path is a list of ``[x, y]`` waypoints ending at the goal; without one the base goes
        straight there. It is the caller's job to have chosen a collision-free path -- the base
        itself is a collider, so a bad plan shows up as a stall, which is reported as such.
        """
        at = _xyz(plan["at"], "plan.at")
        path = [[float(p[0]), float(p[1])] for p in plan.get("path") or [[at[0], at[1]]]]
        if not path:
            path = [[at[0], at[1]]]
        self._drive(job, path, at[2])
        job.log.append(f"base at {_fmt(self.tcp_base())}")

    def set_lift(self, job: Job, plan: dict) -> None:
        """Set ``plan['height']`` metres of carriage travel above its lowest position."""
        height = float(plan["height"])
        lo, hi = LIFT_RANGE
        if not lo <= height <= hi:
            raise SkillFailed(f"lift height {height:.2f} m is outside [{lo:.2f}, {hi:.2f}]")
        self._command({"joints": {"lift": height}})
        deadline = time.monotonic() + self.config.settle_timeout_s
        while time.monotonic() < deadline:
            self._tick()
            now = self._state()["joints"].get("lift")
            if now is not None and abs(now - height) < 0.01:
                job.log.append(f"lift at {height:.2f} m")
                return
        raise SkillFailed(f"lift did not reach {height:.2f} m")

    def tcp_base(self) -> list[float]:
        o = self._state().get("odometry") or {}
        return [float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("yaw", 0.0))]

    def _turn_to(self, yaw_target: float) -> None:
        """Turn in place to ``yaw_target``, steering the position controller from measured odometry."""
        st = self._state()
        o = st["odometry"]
        x, y, yaw0 = o["x"], o["y"], o["yaw"]
        delta = wrap_angle(yaw_target - yaw0)
        duration = abs(delta) / self.config.angular_speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3.0 + 5.0
        since_progress = {"t": t0, "err": None}
        while True:
            st = self._tick()
            frac = 1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration)
            self._command({"base": [x, y, yaw0 + delta * frac]})
            err = abs(wrap_angle(yaw_target - st["odometry"]["yaw"]))
            if frac >= 1.0 and err < self.config.yaw_tol:
                return
            if _stalled(st["stamp"], err, since_progress, self.config.drive_stall_s):
                raise SkillFailed(f"turn is blocked: no progress for {self.config.drive_stall_s:.0f} s "
                                  f"(yaw error {err:.2f} rad)")
            if st["stamp"] > deadline:
                raise SkillFailed(f"turn did not converge (yaw error {err:.2f} rad)")

    def _translate_to(self, x1: float, y1: float, yaw: float) -> None:
        """Drive straight to ``(x1, y1)`` holding ``yaw`` over the same path."""
        st = self._state()
        x0, y0 = st["odometry"]["x"], st["odometry"]["y"]
        # The yaw joint is continuous: keep the heading in the odometry's unwrapped frame, or the
        # position controller spins the base a whole turn.
        current = st["odometry"]["yaw"]
        yaw = current + wrap_angle(yaw - current)
        length = math.hypot(x1 - x0, y1 - y0)
        duration = length / self.config.linear_speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3.0 + 5.0
        since_progress = {"t": t0, "err": None}
        while True:
            st = self._tick()
            frac = 1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration)
            self._command({"base": [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, yaw]})
            err = math.hypot(x1 - st["odometry"]["x"], y1 - st["odometry"]["y"])
            if frac >= 1.0 and err < self.config.position_tol:
                return
            if _stalled(st["stamp"], err, since_progress, self.config.drive_stall_s):
                raise SkillFailed(f"base is blocked {err:.2f} m from the waypoint "
                                  f"({st['odometry']['x']:.2f}, {st['odometry']['y']:.2f}); no progress for "
                                  f"{self.config.drive_stall_s:.0f} s")
            if st["stamp"] > deadline:
                raise SkillFailed(f"base did not reach the waypoint (error {err:.2f} m)")

    def _drive(self, job: Job, path: list, final_yaw: float) -> None:
        for waypoint in path:
            wx, wy = waypoint[:2]
            o = self._state()["odometry"]
            if math.hypot(wx - o["x"], wy - o["y"]) < self.config.position_tol:
                continue
            heading = math.atan2(wy - o["y"], wx - o["x"])
            if len(waypoint) == 3:
                heading = waypoint[2]
                if abs(wrap_angle(o['yaw']-heading)) > self.config.yaw_tol:
                    raise SkillFailed('escape heading changed; predict again')
            else:
                self._turn_to(heading)
            self._translate_to(wx, wy, heading)
            job.log.append(f"reached waypoint ({wx:.2f}, {wy:.2f})")
        self._turn_to(final_yaw)

    # -------------------------------------------------------------- primitives
    def move_to(self, job: Job, arm: str, tcp: list[float], orientation: list[float] | None = None,
                line: bool = False) -> list[float]:
        """Move the arm so the pinch point reaches ``tcp``; returns the joint targets.

        ``line=True`` walks the pinch point straight there in ``line_step`` increments instead of
        one joint interpolation, which is what a descent or a lift has to be."""
        if line:
            here = self.tcp(arm)
            distance = sum((a - b) ** 2 for a, b in zip(tcp, here, strict=True)) ** 0.5
            steps = max(1, int(distance / self.config.line_step) + 1)
            for i in range(1, steps + 1):
                frac = i / steps
                waypoint = [here[j] + (tcp[j] - here[j]) * frac for j in range(3)]
                self._move_to(job, arm, waypoint, orientation)
        else:
            self._move_to(job, arm, tcp, orientation)
        return self.ik.current(arm)

    def _move_to(self, job: Job, arm: str, tcp: list[float], orientation: list[float] | None) -> None:
        self._sync(arm)
        seed = self.ik.current(arm)
        target = self._link6_target(tcp, orientation)
        try:
            if orientation is None:
                joints = self.ik.solve(arm, target["p"], target["q"], q_init=seed)
            else:
                # The wrist cannot hold a strictly vertical tool everywhere, so a small tilt is
                # accepted rather than declaring a reachable place unreachable.
                joints, tilt = self.ik.solve_tool(arm, target["p"], target["q"],
                                                  max_tilt=self.config.max_tool_tilt, q_init=seed)
                # Report the pinch point, not the wrist: a tilted tool that keeps its wrist on
                # target is exactly how a descent ends up a few centimetres off the plate.
                reached = self._tcp_of(self.ik.fk(arm, joints))
                missed = sum((a - b) ** 2 for a, b in zip(reached, tcp, strict=True)) ** 0.5
                if missed > self.config.tcp_tolerance:
                    raise SkillFailed(f"{arm} tool would land {missed * 100:.1f} cm from {_fmt(tcp)} "
                                      f"at {_fmt(reached)} (tilt {tilt:.2f} rad)")
                if tilt > 0.02:
                    job.log.append(f"{arm} tool tilted {tilt:.2f} rad to reach {_fmt(tcp)}")
        except IkError as exc:
            raise SkillFailed(f"{arm} arm cannot reach {_fmt(tcp)}: {exc}") from exc
        self._command({"joints": dict(zip(self.ik.joint_names(arm), joints, strict=True))})
        self._wait_joints(arm, joints)
        job.log.append(f"{arm} at {_fmt(self.tcp(arm))}")

    def move_pose(self, job: Job, plan: dict) -> None:
        """Put the arm's pinch point at ``plan['at']`` (with an optional wxyz ``orientation``).

        Without an explicit orientation the tool points down, which is what makes "the pinch point
        is at ``plan['at']``" true: a position-only solve may satisfy the wrist anywhere it likes."""
        arm = self._arm(plan)
        at = _xyz(plan["at"], "plan.at")
        orientation = plan.get("orientation")
        if orientation is not None and len(orientation) != 4:
            raise SkillFailed(f"plan.orientation must be a wxyz quaternion, got {orientation!r}")
        self.move_to(job, arm, at, orientation or _tool_down(), line=bool(plan.get("line", False)))

    def park(self, job: Job, plan: dict) -> None:
        """Return one arm to the twin's rest posture, out of the head cameras' view.

        A mission that gives up while holding something over the table leaves its own forearm in
        the camera; parking is a verb because "look again" needs a clear view first.

        Arms boot at the all-zero USD posture (#29). Swinging ``park``'s shoulder yaw from that
        fold collides with the lift carriage, so the arm unfolds through ``HOME`` first."""
        arm = self._arm(plan)
        names = self.ik.joint_names(arm)
        self.move_joints(job, {"arm": arm, "joints": dict(zip(names, HOME, strict=True))})
        self.move_joints(job, {"arm": arm, "joints": dict(zip(names, park(arm), strict=True))})

    def move_joints(self, job: Job, plan: dict) -> None:
        """Drive one arm's six joints to the requested posture."""
        arm = self._arm(plan)
        joints = plan.get("joints") or {}
        names = self.ik.joint_names(arm)
        missing = [n for n in names if n not in joints]
        if missing:
            raise SkillFailed(f"move_joints needs all six {arm} joints; missing {missing}")
        target = [float(joints[n]) for n in names]
        for value, (lo, hi) in zip(target, self.ik.joint_limits(arm), strict=True):
            if not lo <= value <= hi:
                raise SkillFailed(f"{arm} joint target {value:.3f} is outside [{lo:.3f}, {hi:.3f}]")
        self._command({"joints": dict(zip(names, target, strict=True))})
        self._wait_joints(arm, target)

    def _sync(self, arm: str) -> None:
        """Seed the solver from the robot's own encoders, not from the last solution.

        The chassis pose and the torso height go in too: the arm solves toward world targets, so
        the model the solver walks has to stand where the robot stands."""
        names = self.ik.joint_names(arm)
        st = self._state()
        o = st.get("odometry") or {}
        self.ik.set_base(float(o.get("x", 0.0)), float(o.get("y", 0.0)), float(o.get("yaw", 0.0)),
                         st.get("joints", {}).get("lift"))
        if all(n in st.get("joints", {}) for n in names):
            self.ik.set_joints(arm, [st["joints"][n] for n in names])

    def _wait_joints(self, arm: str, target: list[float]) -> None:
        names = self.ik.joint_names(arm)
        deadline = time.monotonic() + self.config.settle_timeout_s
        last, since = None, time.monotonic()
        compensated = False
        while time.monotonic() < deadline:
            self._tick()
            now = [self._state()["joints"][n] for n in names]
            if max(abs(a - b) for a, b in zip(now, target, strict=True)) <= self.config.joint_tol:
                return
            # A joint that stops moving short of the target is either blocked or holding a load
            # against gravity: a position servo settles with a steady-state error. If the residual
            # is small, command the target plus that residual once -- integral action, which is
            # what a real arm's servo loop does -- before calling it blocked.
            if last is not None and max(abs(a - b) for a, b in zip(now, last, strict=True)) < 1e-3:
                if time.monotonic() - since > self.config.stall_s:
                    residual = [t - n for t, n in zip(target, now, strict=True)]
                    if not compensated and max(abs(r) for r in residual) < 0.1:
                        self._command({"joints": dict(zip(names, [t + r for t, r in zip(target, residual, strict=True)],
                                                         strict=True))})
                        compensated, since, last = True, time.monotonic(), None
                        continue
                    raise SkillFailed(f"{arm} arm stalled {_fmt(list(now))} toward {_fmt(target)}")
            else:
                since = time.monotonic()
            last = now
        raise SkillFailed(f"{arm} arm did not reach {_fmt(target)} within {self.config.settle_timeout_s:.0f} s")

    def _settle_fingers(self, close: bool) -> None:
        time.sleep(self.config.grasp_grip_wait_s)

    def tcp(self, arm: str) -> list[float]:
        """Where the pinch point is now, from the robot's own joint encoders."""
        self._sync(arm)
        return self._tcp_of(self.ik.fk(arm))

    def _tcp_of(self, link6_pose: dict) -> list[float]:
        return pose_compose(link6_pose, {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})["p"]

    def _link6_target(self, tcp: list[float], orientation: list[float] | None) -> dict:
        """The link6 pose that puts the pinch point at ``tcp`` with the given tool orientation."""
        tool = orientation if orientation is not None else _tool_down()
        offset = {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]}
        from steward.math3d import pose_inverse

        return pose_compose({"p": list(tcp), "q": tool}, pose_inverse(offset))

    def _tick(self) -> dict:
        if self._stop.is_set():
            raise SkillStopped()
        time.sleep(self.config.tick_s)
        return self._state()

    def _arm(self, plan: dict) -> str:
        grip = plan.get("gripper")
        if grip in ARM_OF_GRIPPER:
            return ARM_OF_GRIPPER[grip]
        arm = plan.get("arm")
        if arm not in ("L", "R"):
            raise SkillFailed(f"plan needs arm L/R or a gripper link, got {plan!r}")
        return arm

    def _retreat_on_failure(self) -> None:
        for arm in ("L", "R"):
            try:
                here = self.tcp(arm)
                self._move_to(Job("retreat", "retreat", ""), arm,
                              [here[0], here[1], here[2] + self.config.retreat_height], None)
            except Exception:  # noqa: BLE001 - best effort after a failure that is already reported
                pass

    def _open_all_grippers(self) -> None:
        try:
            self._command({"gripper": {g: "open" for g in ARM_OF_GRIPPER}})
        except Exception:  # noqa: BLE001 - best effort while stopping
            pass


def _tool_down() -> list[float]:
    """Tool pointing straight down: link6 +z along world -z (180 degrees about x)."""
    return [0.0, 1.0, 0.0, 0.0]


def _tool_sideways(direction: list[float]) -> list[float]:
    """Tool pointing along a horizontal direction (used to push): link6 +z onto world +/-x."""
    sign = 1.0 if direction[0] >= 0 else -1.0
    return [0.7071068, 0.0, sign * 0.7071068, 0.0]


def _fmt(p: list[float]) -> str:
    return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"


def _xyz(value, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SkillFailed(f"{what} must be [x, y, z], got {value!r}")
    return [float(v) for v in value]


def _stalled(now: float, err: float, since: dict, window: float) -> bool:
    """Whether ``err`` has stopped improving for ``window`` seconds of body time.

    Progress means the error got smaller, not that it changed: during a turn the error ticks down
    by millimetres per sample and a "did it change at all" test would call that blocked.
    """
    if since["err"] is None or err < since["err"] - 1e-4:
        since["t"], since["err"] = now, err
        return False
    return now - since["t"] > window
