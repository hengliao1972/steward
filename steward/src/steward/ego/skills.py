"""Skill execution: turn a Prediction Receipt's plan into timed actuator commands.

Motion uses only the robot's own estimates (odometry, encoders, ES). Base motion ramps a
position setpoint along each path segment at a bounded speed in simulation time taken from sensor
stamps; joints settle to targets with one sag correction. Control ticks follow the world clock when
one is given, so a skill moves the same at every simulation speed. A stop request interrupts
the current skill between control ticks and holds the current pose.
"""

from __future__ import annotations

import itertools
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from steward.math3d import wrap_angle
from steward.ego.motion_errors import OutOfReach, SkillFailed, SkillStopped
from steward.ego.motion_policy import MotionPolicy, PlanarMotionPolicy

if TYPE_CHECKING:
    from steward.ego.runtime import RobotLink


@dataclass
class MotionConfig:
    linear_speed: float = 1.4
    angular_speed: float = 3.6
    look_angular_speed: float = 1.0
    """Turn rate of a look-around while perception keeps keyframes. At the perception rate this
    is one keyframe step, so a long table that is fully inside the head camera for only ~0.2 rad
    is sampled. The faster ``angular_speed`` steps ~0.36 rad and can land only on the truncated
    views, which do not start a track. A look-around that fuses every frame keeps
    ``angular_speed``."""
    gentle_speed: float = 0.3
    """Base speed while carrying a container with its contents (plans marked ``gentle``)."""
    gentle_turn_speed: float = 0.5
    gentle_lift_speed: float = 0.1
    """Lift and reach speed with a gentle plan: a container accelerated faster leaves its contents behind."""
    arm_speed: float | None = None
    """Lift and reach speed of an arm whose joints, servoed straight to a solved posture, carry the
    grasp point off the line between the two postures (an arm driven by inverse kinematics sweeps its
    wrist below that line, through a drawer front): the move is ramped so the grasp point stays on the
    line. None: the joint goes straight to its target (a slide moves along the line anyway)."""
    wrist_pitch_speed: float = 1.0
    """Rate at which a tool tilts forward (``wrist_pitch``) on its way out to a grasp, and back after."""
    wrist_pitch_max: float = 0.0
    """Most forward tool tilt this body uses; 0 means the tool stays pointing down (kitchen robot)."""
    drawer_speed: float = 0.35
    """Base speed while pulling or pushing a drawer by its handle: faster, its contents tip over."""
    position_tol: float = 0.03
    yaw_tol: float = 0.04
    joint_tol: float = 0.01
    # mink often parks a commanded shallow pitch on a neighbouring posture a few tenths away; the
    # virtual TCP still works, so wrist_pitch settle uses this wider gate (lift/reach keep joint_tol).
    wrist_pitch_tol: float = 0.25
    tick_s: float = 0.04
    retract_reach: float = 0.10
    # After a failure the arm first rises this far, then turns the wrist upright in place, then draws
    # back: drawn back at once, a box tipped over a table caught its edge and turned the table over.
    failure_raise: float = 0.08
    reach_max: float = 0.70  # the wrist slide's upper joint limit; asking beyond it only fails
    stow_lift: float = 0.60
    # Exit / carry floors queried against the USD envelope: a tilted short retract needs the higher
    # one; an untilted long place tops out near 0.50 m reach so it must draw back while still tilted.
    tilted_exit_lift: float = 1.05
    untilted_exit_lift: float = 0.85
    tilted_carry_floor: float = 0.95
    exit_retilt_min: float = 0.6
    # Drawer-handle docks can ask ~0.40 m reach; tool-down only holds that above this TCP height.
    handle_lift_floor: float = 0.65
    # Farthest tool-down reach at drawer-close height (~0.61 m); creep/probes cap here.
    tool_down_reach_mid: float = 0.36
    # Post-deploy / unfurl seed tip. ``retract_reach`` (0.28) sits next to the 0.32 m tool-down
    # hole: settle within joint_tol lands inside it and the next planar move fails. Prefer a tip
    # the envelope measured as freshly reachable with room past that ridge.
    working_lift: float = 0.85
    working_reach: float = 0.40
    dock_tol: float = 0.06
    dock_position_tol: float = 0.01
    """How near its docking pose the base settles before the arm moves (``position_tol`` leaves it
    2-3 cm short, and the arm reaches that much farther than planned)."""
    dock_yaw_tol: float = 0.12
    align_min: float = 0.01
    align_max: float = 0.15
    stall_s: float = 2.0
    # Offsets from the estimate at which a pick closes the gripper before giving up: the first
    # probe is at the estimate itself, then forward/back, then left/right, then other heights.
    grasp_probe_xy: tuple[float, ...] = (0.035, -0.035)
    grasp_probe_z: tuple[float, ...] = (-0.025, 0.025)
    settle_s: float = 0.4
    """Robot time after a joint command at which a joint still short of its target (sagging under
    a load) gets one corrected setpoint."""
    grasp_wait_s: float = 0.1
    release_wait_s: float = 0.15
    release_joint_tol: float = 0.01
    """Maximum measured lift/reach error before opening, including load sag and IK residual.
    The slide uses its centimetre motion budget; Piper profiles set their own release budget."""
    release_settle_s: float = 0.6
    """Robot-time budget for measured lift/reach to settle before release; expiry keeps the load held."""
    # insert_screw_into_hole: linger after open so the screw can drop into the hole, then rise
    # tip-down before any retract — a 0.15 s wait + snap retract knocks it over.
    insert_release_wait_s: float = 0.75
    insert_clear_dz: float = 0.05
    """Metres to raise the gripper (still tip-down) after insert release before drawing back."""
    handle_release_wait_s: float = 0.08
    look_dwell_s: float = 0.6
    inspect_closed_drawers: bool = False
    """After the four look-around turns, open each closed drawer, look inside, and close it.

    A closed drawer hides its contents from every camera. The visit uses the drawer skill's
    plan, taken at that moment, and is skipped when the plan is not feasible. Closing restores
    the drawer; tracks seen inside it stay, because a closed drawer is not evidence they are gone."""
    drawer_look_s: float = 4.5
    """World time spent looking into an open drawer, long enough for a second keyframe."""
    drawer_peek_m: float = 0.3
    """How far the look along an open drawer steps toward it to see over the front panel."""
    look_into_opened_drawers: bool = True
    """After the ``open`` skill, look along and up to the drawer before reporting it opened.

    The viewing pose the pull ends at frames the drawer floor from about 1.1 m, where a bottle
    is some ten pixels tall in the head camera: too few to name. The look is the one the
    look-around's drawer visit makes, so the contents get the same close views either way."""
    wrist_turn_s: float = 0.6
    """Robot time to turn the wrist yaw to a placing heading."""
    pour_tip_s: float = 2.5
    """Robot time to tip a container over and to set it upright again: slow, so the contents slide
    out rather than fly."""


class TimedLog(list):
    """Keep legacy text and a separate wall-clock trace, including intermediate stages."""
    def __init__(self):
        super().__init__()
        self.started = time.monotonic()
        self.events = []

    def append(self, message):
        self.events.append(dict(wallTime=time.time(), elapsedS=time.monotonic()-self.started,
                                message=message))
        super().append(message)


@dataclass
class Job:
    id: str
    skill: str
    receipt: str
    state: str = "running"
    message: str = ""
    started: float = 0.0
    finished: float | None = None
    log: list[str] = field(default_factory=TimedLog)
    postcondition: dict | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "skill": self.skill, "receipt": self.receipt, "state": self.state,
                "message": self.message, "started": self.started, "finished": self.finished, "log": self.log[-20:],
                "timing": getattr(self.log, "events", [])[-20:],
                **({"postcondition": self.postcondition} if self.postcondition is not None else {})}


class _Stall:
    """Detects a motion that stopped making progress: the error has not improved by ``step``
    within ``window`` seconds of robot time and over at least ``samples`` fresh state updates
    (at a high simulation speed one update can span a large part of the window)."""

    def __init__(self, t0: float, window: float, step: float, samples: int = 8):
        self.window, self.step, self.samples = window, step, samples
        self.best = math.inf
        self.since = t0
        self.last_stamp = t0
        self.fresh = 0

    def stuck(self, stamp: float, err: float) -> bool:
        if stamp != self.last_stamp:
            self.last_stamp = stamp
            self.fresh += 1
        if err < self.best - self.step:
            self.best, self.since, self.fresh = err, stamp, 0
        return stamp - self.since > self.window and self.fresh >= self.samples


class SkillExecutor:
    """Runs at most one skill at a time in a worker thread."""

    def __init__(self, command: Callable[[dict], None], state: Callable[[], dict], mark_held: Callable[[str], str],
                 config: MotionConfig | None = None, shift: Callable[[str, list[float]], None] | None = None,
                 on_error: Callable[[str], None] | None = None, on_finish: Callable[[Job], None] | None = None,
                 on_begin: Callable[[str, dict], None] | None = None,
                 on_release: Callable[..., None] | None = None,
                 clock: RobotLink | None = None,
                 closed_drawers: Callable[[], list[str]] | None = None,
                 drawer_plan: Callable[..., tuple[dict | None, str]] | None = None,
                 pin: Callable[[str, bool], None] | None = None,
                 furniture: Callable[[str], str | None] | None = None,
                 contents: Callable[[str], list[str]] | None = None,
                 content_positions: Callable[[str], dict[str, list[float]]] | None = None,
                 resettle: Callable[[str, list[float], dict[str, list[float]]], list[str]] | None = None,
                 home_empty: Callable[[Callable[[], dict]], bool] | None = None,
                 deploy_working: Callable[[Callable[[], dict]], None] | None = None,
                 placement_evidence: Callable[[str], dict] | None = None,
                 on_release_target: Callable[[str, list[float], dict], None] | None = None,
                 assemble_onto: Callable[[Job, dict], None] | None = None,
                 validate_placement: Callable[[dict], None] | None = None,
                 fasten: Callable[[Job, dict], None] | None = None,
                 verify_screw_seat: Callable[[str], tuple[bool, str]] | None = None,
                 replan_pick: Callable[[str], dict | None] | None = None,
                 replan_insert_place: Callable[[dict], dict | None] | None = None,
                 motion_policy: MotionPolicy | None = None):
        """``clock`` is the robot link whose world clock times the control ticks (one hold per job,
        ``skill/<job id>``); without it a tick sleeps ``tick_s`` of wall time.

        ``motion_policy`` owns body-dependent approach, withdrawal and recovery sequences;
        the executor supplies cancellable motion primitives and owns job state.

        ``closed_drawers`` lists drawers to open after a look-around, and ``drawer_plan`` returns
        the open or close plan for one of them, with the reason when there is no plan. A close may
        be asked with the metres just opened, so the drawer is pushed back the same distance.
        ``furniture`` is the piece that holds a drawer; it is held still for the visit so the
        open drawer is not fused into it.

        ``home_empty``, when provided, returns empty arms home after successful work using the job's
        cancellable tick. Homing failure fails the job without attempting Cartesian recovery; None
        leaves arms as-is.

        ``deploy_working``, when provided, joint-space unfolds a resting arm through HOME before the
        first planar dock (PARK/zero has no safe one-shot IK swing).

        ``verify_screw_seat`` / ``replan_pick`` / ``replan_insert_place`` back
        ``insert_screw_into_hole``: after each place, confirm the screw stands tip-down in a hole;
        on miss, re-pick and re-plan the place.
        """
        self._on_error = on_error
        self._clock = clock
        self._now = 0.0
        self._hold = ""
        self._on_finish = on_finish
        self._on_begin = on_begin
        self._on_release = on_release
        self._home_empty = home_empty
        self._placement_evidence = placement_evidence
        self._on_release_target = on_release_target
        self._assemble_onto = assemble_onto
        self._fasten = fasten
        self._validate_placement = validate_placement
        self._deploy_working = deploy_working
        self._verify_screw_seat = verify_screw_seat
        self._replan_pick = replan_pick
        self._replan_insert_place = replan_insert_place
        self._command = command
        self._state = state
        self._mark_held = mark_held
        self._shift = shift
        self._closed_drawers = closed_drawers
        self._drawer_plan = drawer_plan
        self._pin = pin
        self._furniture = furniture
        self._contents = contents
        self._content_positions = content_positions
        self._resettle = resettle
        self.config = config or MotionConfig()
        self.motion_policy = motion_policy if motion_policy is not None else PlanarMotionPolicy()
        self._ids = itertools.count(1)
        self._gentle = False  # the running plan moves a container with its contents
        self._setpoints: dict[str, float] = {}  # the last target each joint was moved to
        self._look = False  # the running skill is a look-around
        self._keyframed = False  # that look-around turns at look_angular_speed only while perception keeps keyframes
        self._arm_engaged = False  # Cartesian recovery is allowed only after this job docks successfully
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self._pulls: dict[str, dict] = {}  # drawer id -> the pull that left it open (:meth:`pull_of`)

    # ------------------------------------------------------------------ public
    def pull_of(self, drawer_id: str) -> dict:
        """The pull that left ``drawer_id`` open, as close parameters, or ``{}`` when the robot has
        not opened it or has pushed it back since.

        ``handle`` is where the gripper let the handle go, ``axis`` the direction it was pulled
        and ``distance`` the metres it travelled. The robot knows them from its own motion, so a
        close planned from them does not depend on how later views box the open drawer."""
        pull = self._pulls.get(drawer_id)
        return {} if pull is None else {"handle": list(pull["handle"]), "axis": list(pull["axis"]),
                                        "distance": pull["distance"]}

    def start(self, skill: str, receipt: dict) -> Job:
        with self._lock:
            if self.current is not None and self.current.state == "running":
                raise RuntimeError(f"skill {self.current.skill} ({self.current.id}) is still running")
            job = Job(id=f"job-{next(self._ids)}", skill=skill, receipt=receipt["id"], started=self._state()["stamp"])
            self.jobs[job.id] = job
            self.current = job
            self._stop.clear()
        if self._clock is not None:
            self._clock.arm(f"skill/{job.id}")
        threading.Thread(target=self._run, args=(job, skill, receipt), name=f"skill-{job.id}", daemon=True).start()
        return job

    def cancel(self) -> None:
        """Interrupt the current skill without commanding the body (used when the world restarts)."""
        self._stop.set()

    def stop(self) -> dict:
        self._stop.set()
        running = self.current if self.current and self.current.state == "running" else None
        if running is not None and self._clock is not None:
            # A skill waiting on a paused world clock ends now instead of when the world resumes.
            self._release_clock(f"skill/{running.id}")
        st = self._state()
        o = st["odometry"]
        self._command({"base": [o["x"], o["y"], o["yaw"]]})
        return {"stopped": running.id if running else None}

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.monotonic() + timeout
        job = self.jobs[job_id]
        while job.state == "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        return job

    # ---------------------------------------------------------------- runner
    def _begin(self, job: Job, skill: str, receipt: dict) -> None:
        """Run the caller's preparation inside the job thread, so ``start`` returns at once.

        Preparation may take tens of seconds (a fresh segmentation of the scene). The caller that
        waits on the job has a budget for that; the caller that issues the run only has its
        transport timeout, and must not be made to hold it."""
        self._primitive_deadline = time.monotonic() + 3600
        if self._on_begin is None:
            return
        try:
            self._on_begin(skill, receipt)
        except SkillFailed:
            raise  # the skill itself reports it; a preparation that found the plan stale is not an alert
        except Exception as exc:  # noqa: BLE001 - an unexpected preparation error is an alert, and fails the skill
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）准备阶段出错：{type(exc).__name__}: {exc}")
            raise

    def _run(self, job: Job, skill: str, receipt: dict) -> None:
        homing = False
        self._stamp_watchdog = (None, time.monotonic())
        self._job_deadline = time.monotonic() + 14400
        self._primitive_deadline = time.monotonic() + 3600
        self._arm_engaged = False
        try:
            if self._clock is not None:
                self._hold = f"skill/{job.id}"
                self._now = self._clock.wait_until(None, self._hold)
            plan = receipt.get("plan", {})
            self._begin(job, skill, receipt)
            if self._stop.is_set():
                raise SkillStopped()
            self._gentle = bool(plan.get("gentle"))
            if skill == "look_around":
                self._look = True
                try:
                    self._look_around(job)
                finally:
                    self._look = False
            elif skill in ('seat_screw', 'fasten_onto', 'flip_assembly', 'regrasp_upright') and self._fasten is not None:
                self._fasten(job, plan)
            elif skill == "assemble_onto" and self._assemble_onto is not None:
                self._assemble_onto(job, plan)
            elif skill == "navigate":
                self._drive(job, plan["path"], plan["yaw"])
            elif skill == "pick":
                self._pick(job, plan)
            elif skill == "place" and "floor" in plan:
                self._place_floor(job, plan)
            elif skill == "place":
                self._place(job, plan)
            elif skill == "open":
                self._open(job, plan)
            elif skill == "close":
                self._close(job, plan)
            elif skill == "pour":
                self._pour(job, plan)
            elif skill == "drive_screw":
                # Kitchen tip-down plans carry pressLift/spot; assembly fastening uses body/screw.
                if self._fasten is not None and "pressLift" not in plan and "spot" not in plan:
                    self._fasten(job, plan)
                else:
                    self._drive_screw(job, plan)
            elif skill == "insert_screw_into_hole":
                self._insert_screw_into_hole(job, plan)
            else:
                raise SkillFailed(f"unknown skill {skill}")
            if self._home_empty is not None:
                homing = True
                if self._home_empty(self._tick):
                    self._setpoints.clear()
                    job.log.append("both empty arms returned to zero")
            job.state, job.message = "succeeded", job.message or f"{skill} completed"
        except SkillStopped:
            job.state, job.message = "stopped", "stopped on request"
        except SkillFailed as exc:
            job.state, job.message = "failed", (f"{job.message or skill + ' completed'}; homing failed: {exc}"
                                                if homing else str(exc))
            if not homing and self._arm_engaged:
                self._safe_retract()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller through the job record
            detail = f"{type(exc).__name__}: {exc}"
            job.state, job.message = "failed", (f"{job.message or skill + ' completed'}; homing failed: {detail}"
                                                if homing else detail)
            if self._on_error is not None:
                self._on_error(f"技能 {skill}（{job.id}）出现未预料的异常：{type(exc).__name__}: {exc}")
            if not homing and self._arm_engaged:
                self._safe_retract()
        finally:
            job.finished = self._state()["stamp"]
            if self._clock is not None:
                self._release_clock(f"skill/{job.id}")
            if self._on_finish is not None:
                self._on_finish(job)

    def _released(self, object_id: str, spot: list[float], target: dict | None = None,
                  tip_roll: float | None = None) -> None:
        if target is not None and self._on_release_target is not None:
            self._on_release_target(object_id, list(spot), target)
        elif self._on_release is not None:
            try:
                self._on_release(object_id, list(spot), tip_roll=tip_roll)
            except TypeError:
                # Older callbacks take only (id, spot).
                self._on_release(object_id, list(spot))

    def _report_placement(self, job: Job, object_id: str) -> None:
        if self._placement_evidence is None:
            return
        try:
            evidence = self._placement_evidence(object_id)
        except Exception:  # release already happened; failure to read evidence must not imply rollback
            evidence = {"ok": None, "status": "unknown", "entity": object_id,
                        "reason": "post-release evidence unavailable; observe before continuing"}
        job.postcondition = evidence
        job.log.append(f"[RSI_PLACE_EVIDENCE] {object_id} {evidence['status']} "
                       f"deltaXYM={evidence.get('deltaXYM')}; ok=null")
        if "orientation" in evidence:
            angle = evidence["orientation"]
            job.log.append(f"[RSI_PLACE_YAW] {object_id} {angle['status']} "
                           f"axisDeltaDeg={angle.get('axisDeltaDeg')}; directed yaw unverified")
        job.message = f"release executed for {object_id}; placement unverified: {evidence['reason']}"

    def _release_clock(self, hold: str) -> None:
        try:
            self._clock.release(hold)
        except Exception:  # noqa: BLE001 - the world is gone or restarting; its hold lease expires on its own
            pass

    def _safe_retract(self) -> None:
        try:
            self.motion_policy.recover(self)
        except SkillStopped:
            return  # Stop recovery while preserving the skill failure already recorded.

    def _tick(self) -> dict:
        if self._stop.is_set():
            raise SkillStopped()
        if self._clock is None:
            time.sleep(self.config.tick_s)
        else:
            self._now = self._clock.wait_until(self._now + self.config.tick_s, self._hold)
        state = self._state()
        stamp = state['stamp']
        now = time.monotonic()
        last, progressed = getattr(self, '_stamp_watchdog', (None, now))
        if stamp != last:
            progressed = now
        self._stamp_watchdog = (stamp, progressed)
        if now-progressed >= 900:
            raise SkillFailed('robot sensor timestamp stalled for 900 s')
        if now > min(getattr(self, '_primitive_deadline', math.inf), getattr(self, '_job_deadline', math.inf)):
            raise SkillFailed('motion wall timeout; continue after checked recovery')
        return state

    # ------------------------------------------------------------- primitives
    def _turn_to(self, yaw_target: float) -> None:
        st = self._state()
        o = st["odometry"]
        x, y, yaw0 = o["x"], o["y"], o["yaw"]
        delta = wrap_angle(yaw_target - yaw0)
        if self._look and self._keyframed:
            # A closed look turns half as fast. The cup is only a few pixels wide, and a
            # tenth of a radian steps over the heading that separates it from the counter.
            speed = self.config.look_angular_speed if self.config.inspect_closed_drawers else self.config.look_angular_speed * 0.5
        elif self._gentle:
            speed = self.config.gentle_turn_speed
        else:
            speed = self.config.angular_speed
        duration = abs(delta) / speed
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        stall = _Stall(t0, self.config.stall_s, 0.02)
        self._primitive_deadline = time.monotonic() + 3600
        while True:
            st = self._tick()
            frac = self._profile(1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration))
            self._command({"base": [x, y, yaw0 + delta * frac]})
            err = abs(wrap_angle(yaw_target - st["odometry"]["yaw"]))
            if frac >= 1.0 and err < self.config.yaw_tol:
                return
            if stall.stuck(st["stamp"], err):
                raise SkillFailed(f"turn is blocked: no progress for {self.config.stall_s:.0f} s (yaw error {err:.2f} rad)")
            if st["stamp"] > deadline:
                raise SkillFailed(f"turn did not converge (yaw error {err:.2f} rad)")

    def _translate_to(self, x1: float, y1: float, yaw: float, speed: float | None = None,
                      tol: float | None = None) -> None:
        tol = self.config.position_tol if tol is None else tol
        st = self._state()
        x0, y0 = st["odometry"]["x"], st["odometry"]["y"]
        # The base yaw joint is continuous: express the heading in the same unwrapped frame as the
        # odometry, otherwise the position controller spins the base by a full turn.
        current = st["odometry"]["yaw"]
        yaw = current + wrap_angle(yaw - current)
        length = math.hypot(x1 - x0, y1 - y0)
        duration = length / (speed or (self.config.gentle_speed if self._gentle else self.config.linear_speed))
        t0 = st["stamp"]
        deadline = t0 + duration * 3 + 5.0
        stall = _Stall(t0, self.config.stall_s, 0.01)
        self._primitive_deadline = time.monotonic() + 3600
        while True:
            st = self._tick()
            frac = self._profile(1.0 if duration == 0 else min(1.0, (st["stamp"] - t0) / duration))
            self._command({"base": [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, yaw]})
            err = math.hypot(x1 - st["odometry"]["x"], y1 - st["odometry"]["y"])
            if frac >= 1.0 and err < tol:
                return
            if stall.stuck(st["stamp"], err):
                raise SkillFailed(f"base is blocked: no progress for {self.config.stall_s:.0f} s ({err:.2f} m from the waypoint)")
            if st["stamp"] > deadline:
                raise SkillFailed(f"base did not reach the waypoint (error {err:.2f} m)")

    def _drive(self, job: Job, path: list, final_yaw: float) -> None:
        for waypoint in path[1:]:
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

    def _set_joint(self, name: str, target: float, timeout: float = 8.0, ramp: bool = True,
                   tol: float | None = None) -> None:
        """Drive virtual joint ``name`` to ``target``.

        ``tol`` overrides the motion ``joint_tol`` / ``wrist_pitch_tol`` for settle. Use a tight
        value when unfurling from rest: the default 0.10 m accepts a folded FK (~0.19 m) as already
        at ``retract_reach`` (0.28 m)."""
        self._setpoints[name] = target
        speed = self.config.gentle_lift_speed if self._gentle else self.config.arm_speed
        if ramp and speed and name in ("lift", "reach"):
            st = self._state()
            # An arm at rest stands outside the planar arm's workspace: its first move goes straight in.
            q0 = target if st.get("armResting") else st["joints"].get(name, target)
            seconds = abs(target - q0) / speed
            if seconds > self.config.tick_s:
                self._ramp_joint(name, target, seconds)
                return
        self._command({name: target})
        t0 = self._state()["stamp"]
        corrected = False
        if tol is None:
            tol = self.config.wrist_pitch_tol if name == "wrist_pitch" else self.config.joint_tol
        self._primitive_deadline = time.monotonic() + 3600
        while True:
            st = self._tick()
            q = st["joints"][name]
            if abs(q - target) < tol:
                return
            settled = st["stamp"] - t0 > self.config.settle_s
            if settled and not corrected:
                # Re-issue the absolute target instead of target + (target - q). The command
                # takes an HTTP round trip to reach the servo, so at the settle stamp the joint
                # may not have started moving yet. Inflating by the full remaining error then
                # pegs the joint at about twice the intended travel.
                self._command({name: target})
                corrected = True
            if st["stamp"] - t0 > timeout:
                raise SkillFailed(f"joint {name} did not reach {target:.3f} (at {q:.3f})")

    def _settle_for_release(self) -> None:
        """Require measured lift/reach within release tolerance; missing or stalled feedback fails before opening."""
        names = ("lift", "reach")
        if not all(name in self._setpoints for name in names):
            raise SkillFailed("release requires lift and reach setpoints; keeping the load held")
        t0 = self._state()["stamp"]
        self._primitive_deadline = time.monotonic() + 3600
        while True:
            st = self._tick()
            if not all(name in st["joints"] for name in names):
                raise SkillFailed("release requires measured lift and reach; keeping the load held")
            errors = {name: abs(st["joints"][name] - self._setpoints[name]) for name in names}
            if all(error < self.config.release_joint_tol for error in errors.values()):
                return
            if st["stamp"] - t0 > self.config.release_settle_s:
                raise SkillFailed(f"release position did not settle (lift error {errors['lift']:.4f} m, "
                                  f"reach error {errors['reach']:.4f} m); keeping the load held")

    def _profile(self, frac: float) -> float:
        """Share of a move done at time share ``frac``: steady, or for a gentle plan starting and
        ending at rest (smoothstep), so a carried container's contents are not flung."""
        return frac * frac * (3.0 - 2.0 * frac) if self._gentle else frac

    def _ramp_joint(self, name: str, target: float, seconds: float) -> None:
        """Move a joint to ``target`` at a steady rate over ``seconds`` of robot time, then settle."""
        self._ramp_joints({name: target}, seconds)

    def _ramp_joints(self, targets: dict[str, float], seconds: float) -> None:
        """Move joints to ``targets`` together over ``seconds`` of robot time, starting and ending
        at rest, then settle each."""
        st = self._state()
        q0, t0 = {name: st["joints"].get(name, 0.0) for name in targets}, st["stamp"]
        self._primitive_deadline = time.monotonic() + 3600
        while True:
            st = self._tick()
            frac = min(1.0, (st["stamp"] - t0) / seconds) if seconds > 0 else 1.0
            share = frac * frac * (3.0 - 2.0 * frac)
            self._command({name: q0[name] + (target - q0[name]) * share for name, target in targets.items()})
            if frac >= 1.0:
                break
        for name, target in targets.items():
            self._set_joint(name, target, ramp=False)

    def _reach_over(self, plan: dict, xy: list[float], lift: float) -> None:
        self.motion_policy.approach(self, plan, xy, lift)

    def _creep_to(self, xy: list[float], capture: float, reach_cap: float | None = None) -> None:
        self.motion_policy.creep(self, xy, capture, reach_cap)

    def _reach_out(self, xy: list[float], pitch: float | None) -> None:
        self.motion_policy.reach(self, xy, pitch)

    def _extend_then_tilt(self, reach: float, pitch: float) -> None:
        self.motion_policy.extend_then_tilt(self, reach, pitch)

    def _untilt(self) -> None:
        """Turn a tool tilted forward for a grasp back to pointing down, once the arm is retracted."""
        pitch = self._state()["joints"].get("wrist_pitch", 0.0)
        if abs(pitch) >= self.config.joint_tol:
            self._ramp_joints({"wrist_pitch": 0.0}, abs(pitch) / self.config.wrist_pitch_speed)

    def _withdraw(self, plan: dict) -> None:
        self.motion_policy.withdraw(self, plan)

    def _exit_long_reach(self, plan: dict) -> None:
        self.motion_policy.exit(self, plan)

    def _retract(self, plan: dict) -> None:
        """Bring the gripper back to the retracted reach, or only as near as the plan's ``retractReach``
        (a tool tilted forward cannot come all the way back without turning the wrist over)."""
        self._set_joint("reach", max(self.config.retract_reach, plan.get("retractReach", 0.0)))

    def _wait(self, seconds: float) -> None:
        t0 = self._state()["stamp"]
        self._primitive_deadline = time.monotonic() + 3600
        while self._tick()["stamp"] - t0 < seconds:
            pass

    def _dock(self, job: Job, plan: dict, retract: bool = True) -> None:
        """Drive and verify docking with the arm unchanged, then prepare its retracted reach.
        ``retract=False`` leaves the carrying reach unchanged for floor placement. Failure before
        docking does not enable Cartesian arm recovery."""
        self._arm_engaged = False
        self._drive(job, plan["path"], plan["yaw"])
        self._translate_to(plan["dock"][0], plan["dock"][1], plan["yaw"], tol=self.config.dock_position_tol)
        o = self._state()["odometry"]
        dist = math.hypot(o["x"] - plan["dock"][0], o["y"] - plan["dock"][1])
        dyaw = abs(wrap_angle(o["yaw"] - plan["yaw"]))
        if dist > self.config.dock_tol or dyaw > self.config.dock_yaw_tol:
            self._backoff(plan)
            raise SkillFailed(f"docking missed the planned pose by {dist:.2f} m / {math.degrees(dyaw):.0f} deg")
        if self._stop.is_set():
            raise SkillStopped()
        job.log.append("docked")
        self._arm_engaged = True
        self.motion_policy.prepare_dock(self, job, plan, retract)

    def _gripper_xy(self) -> np.ndarray:
        """Gripper position from the robot's own odometry and reach encoder, on the arm's line
        (``armLateral`` left of the base centre)."""
        st = self._state()
        o = st["odometry"]
        ahead, side = st["toolOffset"] + st["joints"]["reach"], st.get("armLateral", 0.0)
        c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
        return np.array([o["x"] + ahead * c - side * s, o["y"] + ahead * s + side * c])

    def _align_lateral(self, target_xy: list[float], tol: float | None = None) -> None:
        """Shift the base sideways so the target lies on the arm's line of reach.

        A long unfurl that creeps the base forward can leave 10+ cm of lateral error; step in
        ``align_max`` chunks so a single large correction is not skipped."""
        for _ in range(6):
            st = self._state()
            o = st["odometry"]
            left = np.array([-math.sin(o["yaw"]), math.cos(o["yaw"])])
            lateral = float((np.array(target_xy) - np.array([o["x"], o["y"]])) @ left) - st.get("armLateral", 0.0)
            if abs(lateral) <= (self.config.align_min if tol is None else tol):
                return
            step = max(-self.config.align_max, min(self.config.align_max, lateral))
            self._translate_to(o["x"] + step * left[0], o["y"] + step * left[1], o["yaw"], tol=tol)
            if abs(lateral) <= self.config.align_max:
                return

    def _backoff(self, plan: dict) -> None:
        self._translate_to(plan["predock"][0], plan["predock"][1], plan["yaw"])

    def _reach_to(self, target_xy: list[float]) -> float:
        """Wrist extension that puts the gripper over ``target_xy``, within the joint's travel.

        The plan's reach and the pose the base actually stopped at differ by a few centimetres; a
        target a little beyond the arm used to fail the whole pick with "joint reach did not
        reach", when all it costs is that the gripper closes short — inside its capture radius."""
        st = self._state()
        o = st["odometry"]
        heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
        rel = np.array(target_xy) - np.array([o["x"], o["y"]])
        return min(self.config.reach_max, float(rel @ heading) - st["toolOffset"])

    def _over_support(self, plan: dict) -> bool:
        """True when the held object is released over the support top.

        Furniture placements require the held footprint (shrunk by ``heldHalf``) to lie on the
        surface. Mating a screw-hole part onto another (``plan["mate"]``) only needs the held
        centre on the mate: a wing spans past a fuselage, so footprint containment can never hold.
        """
        box = plan["supportBox"]
        grip = plan.get("gripSpot", plan["spot"])
        g = self._gripper_xy() + np.array([plan["spot"][0] - grip[0], plan["spot"][1] - grip[1]])
        dx, dy = g[0] - box["cx"], g[1] - box["cy"]
        c, s = math.cos(-box["yaw"]), math.sin(-box["yaw"])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        if "placementToleranceM" in plan:
            if np.linalg.norm(g - np.asarray(plan["spot"])[:2]) > plan["placementToleranceM"]:
                return False
        if plan.get("mate"):
            # Centre on the mate, with a short capture slack for dock error.
            slack = 0.02
            return abs(lx) <= box["hx"] + slack and abs(ly) <= box["hy"] + slack
        margin = plan.get("heldHalf", 0.05) * 0.6
        if abs(lx) > box["hx"] - margin or abs(ly) > box["hy"] - margin:
            return False
        if "releaseHalfXY" in plan:
            ex, ey = plan["releaseHalfXY"]
            return abs(lx) + ex <= box["hx"] and abs(ly) + ey <= box["hy"]
        return True

    # ---------------------------------------------------------------- skills
    def _look_around(self, job: Job) -> None:
        st = self._state()
        # Odometry stays at the origin until the first body frame. A look that commands that point
        # drives the base there. A state without the flag (a test double) is already a pose.
        while st.get("posed") is False:
            st = self._tick()
        yaw0 = st["odometry"]["yaw"]
        for k in range(1, 5):
            self._turn_to(wrap_angle(yaw0 + k * math.pi / 2))
            self._wait(self.config.look_dwell_s)
            job.log.append(f"looked toward {math.degrees(wrap_angle(yaw0 + k * math.pi / 2)):.0f} deg")
        if self.config.inspect_closed_drawers:
            self._inspect_closed_drawers(job)

    def _inspect_closed_drawers(self, job: Job) -> None:
        """Open every closed drawer the ego state still names, look in, and close it.

        A drawer whose plan is not feasible, or whose handle is missed, is left as it is and the
        look-around still succeeds. The other drawers are visited anyway."""
        if self._closed_drawers is None or self._drawer_plan is None:
            return
        pending: list[tuple[str, list[float], dict[str, list[float]]]] = []
        self._pending_resettle = pending
        for drawer_id in self._closed_drawers():
            host_id = self._furniture(drawer_id) if self._furniture is not None else None
            if host_id and self._pin is not None:
                self._pin(host_id, True)
            try:
                self._visit_drawer(job, drawer_id)
            finally:
                if host_id and self._pin is not None:
                    self._pin(host_id, False)
        if pending and self._resettle is not None:
            # Late frames can land after the first close. Bring back anything still left out.
            self._wait(self.config.look_dwell_s)
            for drawer_id, delta, before in pending:
                brought = self._resettle(drawer_id, delta, before)
                if brought:
                    job.log.append(f"brought back {', '.join(brought)}")

    def _visit_drawer(self, job: Job, drawer_id: str) -> None:
        """Open one drawer, look in from the front, and close it by the distance just pulled."""
        plan, why = self._drawer_plan("open", drawer_id)
        if plan is None:
            job.log.append(f"left {drawer_id} closed ({why})")
            return
        if why:
            # The handle is still too uncertain to grasp. The plan's path is safe, so walk
            # there, look, and plan again from the closer view.
            try:
                self._drive(job, plan["path"], plan["yaw"])
            except SkillFailed as exc:
                job.log.append(f"could not approach {drawer_id}: {exc}")
                return
            self._wait(self.config.drawer_look_s)
            job.log.append(f"approached {drawer_id}")
            plan, why = self._drawer_plan("open", drawer_id)
            if plan is None:
                job.log.append(f"left {drawer_id} closed ({why})")
                return
            if why:
                job.log.append(f"opening {drawer_id} with {why}")
        axis = list(plan.get("axis") or [1.0, 0.0])
        open_handle = plan.get("handle")
        try:
            opened = self._drawer(job, plan)
        except SkillFailed as exc:
            job.log.append(f"could not open {drawer_id}: {exc}")
            return
        self._wait(self.config.drawer_look_s)
        job.log.append(f"looked inside {drawer_id}")
        self._look_along_drawer(job, drawer_id, axis)
        # Close by the distance just pulled, on the handle that was pulled.
        if isinstance(opened, (int, float)) and abs(opened) >= 0.05 and open_handle is not None:
            shifted = [float(open_handle[0]) + float(opened) * axis[0],
                       float(open_handle[1]) + float(opened) * axis[1], float(open_handle[2])]
            plan, why = self._drawer_plan("close", drawer_id, abs(opened), shifted, axis)
        elif isinstance(opened, (int, float)) and abs(opened) >= 0.05:
            plan, why = self._drawer_plan("close", drawer_id, abs(opened))
        else:
            plan, why = self._drawer_plan("close", drawer_id)
        if plan is None:
            job.log.append(f"left {drawer_id} open ({why})")
            return
        before = self._content_positions(drawer_id) if self._content_positions is not None else {}
        try:
            closed = self._drawer(job, plan)
        except SkillFailed as exc:
            job.log.append(f"could not close {drawer_id}: {exc}")
            return
        self._wait(self.config.look_dwell_s)
        if self._resettle is not None and isinstance(closed, (int, float)) and abs(closed) >= 0.05:
            delta = [float(closed) * axis[0], float(closed) * axis[1], 0.0]
            pending = getattr(self, "_pending_resettle", None)
            if pending is not None:
                pending.append((drawer_id, delta, before))
            brought = self._resettle(drawer_id, delta, before)
            if brought:
                job.log.append(f"brought back {', '.join(brought)}")

    def _open(self, job: Job, plan: dict) -> None:
        """The ``open`` skill: pull the drawer, remember that pull for the close, and look into it
        as the look-around's visit does.

        The furniture is held still meanwhile: neither the pull nor the look may move the carcass
        or fuse the open drawer into it. A look that cannot be completed leaves the open as it is."""
        drawer_id = plan["drawer"]
        host_id = self._furniture(drawer_id) if self._furniture is not None else None
        if host_id and self._pin is not None:
            self._pin(host_id, True)
        try:
            opened = self._drawer(job, plan)
            self._remember_pull(plan, opened)
            if not self.config.look_into_opened_drawers:
                return
            message = job.message
            try:
                self._wait(self.config.drawer_look_s)
                job.log.append(f"looked inside {drawer_id}")
                self._look_along_drawer(job, drawer_id, list(plan.get("axis") or [1.0, 0.0]))
            except SkillFailed as exc:
                job.log.append(f"could not look along {drawer_id}: {exc}")
            job.message = message
        finally:
            if host_id and self._pin is not None:
                self._pin(host_id, False)

    def _remember_pull(self, plan: dict, opened: float) -> None:
        """Record where a pull of ``opened`` metres left the handle of the plan's drawer."""
        axis, handle = plan.get("axis"), plan.get("handle")
        if axis is None or handle is None or abs(opened) < 0.05:
            return
        self._pulls[plan["drawer"]] = {
            "handle": [float(handle[0]) + opened * float(axis[0]), float(handle[1]) + opened * float(axis[1]),
                       float(handle[2])],
            "axis": [float(axis[0]), float(axis[1])], "distance": abs(opened)}

    def _close(self, job: Job, plan: dict) -> None:
        """The ``close`` skill: push the drawer back and bring its contents with it.

        A view of the open drawer that lands after the push can leave a content in front of the
        shut front; one dwell later it is brought back, as the look-around's visit does."""
        drawer_id = plan["drawer"]
        host_id = self._furniture(drawer_id) if self._furniture is not None else None
        before = self._content_positions(drawer_id) if self._content_positions is not None else {}
        if host_id and self._pin is not None:
            self._pin(host_id, True)
        try:
            closed = self._drawer(job, plan)
            self._pulls.pop(drawer_id, None)
            message = job.message
            self._wait(self.config.look_dwell_s)
            if self._resettle is not None and abs(closed) >= 0.05:
                axis = plan["axis"]
                brought = self._resettle(drawer_id, [closed * float(axis[0]), closed * float(axis[1]), 0.0], before)
                if brought:
                    job.log.append(f"brought back {', '.join(brought)}")
            job.message = message
        finally:
            if host_id and self._pin is not None:
                self._pin(host_id, False)

    def _look_along_drawer(self, job: Job, drawer_id: str, axis: list[float]) -> None:
        """Step sideways in front of an open drawer and back, so both ends of it are in frame.

        The drawer is held still for the step. A moving box would send the close at the wrong handle."""
        if self._pin is None:
            return
        self._pin(drawer_id, True)
        held = self._contents(drawer_id) if self._contents is not None else []
        for tid in held:
            self._pin(tid, True)
        try:
            st = self._state()["odometry"]
            x, y, yaw = float(st["x"]), float(st["y"]), float(st["yaw"])
            side = [-float(axis[1]), float(axis[0])]
            for sign in (1.0, -1.0):
                # 0.24 m puts the pistol on the drawer floor in frame. 0.40 m looks past the
                # front panel, and that step also leaves the close short of the handle.
                self._translate_to(x + side[0] * 0.24 * sign, y + side[1] * 0.24 * sign, yaw)
                self._wait(2.0)
                # The bear is first seen from the side. Hold it before the move back,
                # or that move treats it as something that has left.
                if self._contents is None:
                    continue
                for tid in self._contents(drawer_id):
                    if tid not in held:
                        held.append(tid)
                        self._pin(tid, True)
            # The front panel hides the near part of the drawer floor from the dock. A step
            # toward the drawer looks over the panel; the pull left half a metre of room. The
            # contents are let go for that look, so the closer view can place them, and held
            # again before the step back.
            try:
                self._translate_to(x - axis[0] * self.config.drawer_peek_m, y - axis[1] * self.config.drawer_peek_m, yaw)
                for tid in held:
                    self._pin(tid, False)
                self._wait(2.0)
            except SkillFailed as exc:
                job.log.append(f"could not step up to {drawer_id}: {exc}")
            finally:
                for tid in (self._contents(drawer_id) if self._contents is not None else []):
                    if tid not in held:
                        held.append(tid)
                for tid in held:
                    self._pin(tid, True)
            self._translate_to(x, y, yaw)
            job.log.append(f"looked along {drawer_id}")
        finally:
            for tid in held:
                self._pin(tid, False)
            self._pin(drawer_id, False)

    def _pick(self, job: Job, plan: dict) -> None:
        # Registered small parts can disappear into a gripper/background mask
        # during docking. Keep the last unobstructed estimate for this attempt;
        # tactile capture and the existing spatial gates still decide the grasp.
        protected = bool(plan.get("protectGraspFrame") and self._pin is not None)
        if protected:
            self._pin(plan["target"], True)
        try:
            self._pick_target(job, plan)
        finally:
            if protected:
                self._pin(plan["target"], False)

    def _pick_target(self, job: Job, plan: dict) -> None:
        self._dock(job, plan)
        if plan.get('assemblyGrip'):
            # The planner's grasp path uses roll zero. A previous exit can leave
            # the wrist inverted; set the empty hand before closing, not the load.
            self._set_joint('wrist_roll', 0., tol=.02)
            self._set_joint('wrist_yaw', plan['wristYaw'], tol=.02)
        target = ({'p':list(plan['targetP'])} if plan.get('assemblyGrip')
                  else self._state()['entities'].get(plan['target']))
        if target is None and plan.get("targetP") is not None:
            # The track was renumbered while docking; the predicted position is seconds old and
            # the gripper is already beside it — go on with the position.
            target = {"p": list(plan["targetP"])}
        if target is None:
            raise SkillFailed(f"target {plan['target']} is no longer in the ego state")
        # A large object may be taken by its edge (``graspOffset`` from its centre, in the world), a
        # container on its rim (``graspDz`` above its centre).
        ox, oy = plan.get("graspOffset") or (0.0, 0.0)
        dz = plan.get("graspDz", 0.0)

        def aimed(entity: dict) -> dict:
            return {"p": [entity["p"][0] + ox, entity["p"][1] + oy, entity["p"][2] + dz]}

        target = aimed(target)
        self._align_lateral(target["p"][:2])
        # Match the planned dock reach before the arm moves. A long live reach (base stopped short)
        # asks 0.55 m at the close height with approach tilt — no USD IK (needs ~0.70 m+ at that pitch).
        if float(plan.get("wristPitch") or 0.0) > 0.2 or float(plan.get("closePitch") or 0.0) > 0.2:
            self._restore_planned_reach(plan, target["p"][:2])
            # Hard cap: deep-drawer approaches were sized ≤0.50 m; never extend past that while tilted.
            for _ in range(8):
                if self._reach_to(target["p"][:2]) <= 0.50:
                    break
                o = self._state()["odometry"]
                fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
                self._translate_to(o["x"] + 0.03 * fwd[0], o["y"] + 0.03 * fwd[1], o["yaw"])
        # Off the floor the planned height (its base at floor level) outranks the noisy live estimate.
        # An elevated drawer grasp stores its working height in plan["lift"] (via graspDz); prefer it
        # so the live centre cannot pull the tilt back into the USD hole.
        pick_lift = (plan["lift"] if plan.get("support") == "floor" or "drawerEntry" in plan
                     else max(float(target["p"][2]), float(plan["lift"])))
        # A tilted close at the dock reach may need a raised floor; tool-down closes (closePitch≈0)
        # keep the object's true height so the gripper does not close in air above a short bottle.
        close_pitch = abs(float(plan.get("closePitch") or 0.0))
        if ("drawerEntry" not in plan and float(plan.get("wristPitch") or 0.0) > 0.4 and plan.get("approachLift") is not None
                and close_pitch >= 0.15):
            pick_lift = max(pick_lift, min(float(plan["approachLift"]), float(plan.get("lift", pick_lift) + 0.12)))
        self._reach_over(plan, target["p"][:2], pick_lift)
        self._wait(self.config.settle_s)
        miss = float(np.linalg.norm(self._gripper_xy() - np.array(target["p"][:2])))
        # Tool-down drawer closes creep under a short reach cap; allow a centimetre more than the
        # plan's capture so TCP settle noise does not refuse a grasp that probes would find.
        capture = float(plan.get("captureRadius", 0.05))
        if plan.get("approachLift") is not None and abs(float(plan.get("closePitch") or 0.0)) < 1e-3:
            capture = max(capture, 0.14)
        if miss > capture:
            self._withdraw(plan)
            self._untilt()
            self._backoff(plan)
            raise SkillFailed(f"gripper is {miss * 100:.0f} cm from {plan['target']}; not closing")
        # A real object's estimate is a few centimetres off in any direction as often as not, and
        # the hand only feels the object when it closes. Probe around the estimate the way a person
        # feels for a cup in the dark: at the estimate, then a little forward and back (the arm
        # alone moves), then to either side (the base shifts), then at other heights. The estimate
        # is re-read before each probe -- it is best while the camera is right over the object.
        # Taken by its edge (docked facing that side), forward lies over what is inside it and back
        # over nothing: the probes keep along the edge.
        probes = [(0.0, 0.0, 0.0)]
        if not (ox or oy):
            probes += [(dx, 0.0, 0.0) for dx in self.config.grasp_probe_xy]
        probes += [(0.0, dy, 0.0) for dy in self.config.grasp_probe_xy]
        # Drawer / over-rim grasps often sit a centimetre high after the approach; try down first —
        # but not while still tilted: a centimetre lower at long reach sits in the USD hole.
        # Use the live wrist (approach may have untilted for close) rather than the plan's pitch.
        live_pitch = abs(float(self._state()["joints"].get("wrist_pitch", 0.0)))
        # Mild close tilt (~0.35) still allows centimetre height probes; only skip when steep.
        # Drawer / rim closes: only probe upward. Negative dz untilts to tool-down at ~0.60 m, and
        # that height has no IK past ~0.35 m (grasp missed with pitch=0.00).
        if live_pitch <= 0.5:
            if plan.get("approachLift") is not None:
                z_probes = (0.02, 0.04)
            else:
                z_probes = self.config.grasp_probe_z
            probes += [(0.0, 0.0, dz) for dz in z_probes]
        grasped = False
        for n, (forward, sideways, dz) in enumerate(probes):
            seen = None if plan.get('assemblyGrip') else self._state()['entities'].get(plan['target'])
            latest = aimed(seen) if seen is not None else target
            o = self._state()["odometry"]
            heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
            left = np.array([-heading[1], heading[0]])
            goal = np.array(latest["p"][:2]) + forward * heading + sideways * left
            if n:
                self._command({"gripper": "open"})
                self._wait(self.config.grasp_wait_s)
                self._align_lateral(goal.tolist())
                try:
                    live_pitch = abs(float(self._state()["joints"].get("wrist_pitch", 0.0)))
                    floor = float(plan["lift"]) if live_pitch > 0.5 else float(plan["lift"]) + dz
                    need = max(0.05, self._reach_to(goal.tolist()))
                    # Drawer closes only: 0.52 m @ ~0.65 m @ mild tilt packs joint2. Table / flip
                    # picks keep the planned reach (high tilt at elevated lift needs the full dock).
                    td_cap = self.config.tool_down_reach_mid
                    if plan.get("approachLift") is not None and live_pitch <= 0.5 and need > td_cap:
                        o2 = self._state()["odometry"]
                        fwd = (math.cos(o2["yaw"]), math.sin(o2["yaw"]))
                        for _ in range(6):
                            if self._reach_to(goal.tolist()) <= td_cap:
                                break
                            self._translate_to(o2["x"] + 0.03 * fwd[0], o2["y"] + 0.03 * fwd[1], o2["yaw"])
                            o2 = self._state()["odometry"]
                        need = max(0.05, min(td_cap, self._reach_to(goal.tolist())))
                    # Shorten / untilt before lowering: lift-first at long reach sits in the USD hole.
                    # Keep a mild tilt under a rim — full untilt at drawer-close height fails IK.
                    if dz < 0.0 and live_pitch >= self.config.joint_tol:
                        self._set_joint("reach", min(need, td_cap), ramp=False)
                        if plan.get("approachLift") is not None:
                            keep = 0.25 if live_pitch >= 0.25 else live_pitch
                            if abs(float(self._state()["joints"].get("wrist_pitch", 0.0)) - keep) >= self.config.joint_tol:
                                self._set_joint("wrist_pitch", keep)
                            live_pitch = keep
                        else:
                            self._untilt()
                            live_pitch = 0.0
                        floor = float(plan["lift"]) + dz
                        need = max(0.05, min(td_cap, self._reach_to(goal.tolist())))
                    self._set_joint("reach", need, ramp=False)
                    self._set_joint("lift", max(float(latest["p"][2] + dz), floor))
                except (OutOfReach, SkillFailed) as e:
                    job.log.append(f"skipped probe {n}: {e}")
                    continue
            self._command({"gripper": "close"})
            self._wait(self.config.grasp_wait_s)
            if self._state()["gripper"]["holding"]:
                if n:
                    job.log.append(f"grasped {plan['target']} on probe {n} "
                                   f"(forward {forward * 100:+.0f} cm, left {sideways * 100:+.0f} cm, "
                                   f"height {dz * 100:+.1f} cm)")
                grasped = True
                break
        if not grasped:
            self._command({"gripper": "open"})
            self._wait(0.2)
        if not grasped:
            self._withdraw(plan)
            self._untilt()
            self._backoff(plan)
            raise SkillFailed(f"grasp missed {plan['target']} ({len(probes)} probes)")
        try:
            held = self._mark_held(plan["target"])
        except KeyError as exc:
            raise SkillFailed(str(exc).strip("'")) from None
        job.log.append(f"grasped {held}")
        # Rise at the grasp reach first. Stretching back to the planned dock reach at the low close
        # height (e.g. 0.50 m @ 0.60 m @ ~0.25 rad after a drawer untilt-close) often has no USD IK;
        # the carry/exit floors are what make a tilted retract feasible.
        plan_tilt = max(float(plan.get("wristPitch") or 0.0), float(plan.get("closePitch") or 0.0))
        live_tilt = abs(float(self._state()["joints"].get("wrist_pitch", 0.0)))
        pitched = plan_tilt > 0.5 or live_tilt > 0.5
        mild_tilt = (not pitched) and (plan_tilt > 0.2 or live_tilt > 0.2)
        carry = max(float(plan["carryLift"]),
                    self.config.tilted_carry_floor if (pitched or mild_tilt) else 0.0)
        self.motion_policy.prepare_carry(self, plan)
        self._set_joint("lift", carry)
        if not plan.get("keepReach"):
            if pitched:
                # Short retract while most-tilted needs the exit floor; keep tilt for pour/flip carry.
                self._set_joint("lift", max(carry, self.config.tilted_exit_lift, self.config.stow_lift))
                self._retract(plan)
            elif mild_tilt:
                self._exit_long_reach({**plan, "wristPitch": plan.get("wristPitch") or plan.get("closePitch")
                                       or live_tilt})
            else:
                self._retract(plan)
                if abs(float(self._state()["joints"].get("wrist_pitch", 0.0))) >= self.config.joint_tol:
                    self._untilt()
        if "retreat" in plan:
            self._translate_to(plan["retreat"][0], plan["retreat"][1], plan["yaw"])
        elif not plan.get('stayNear'):
            self._backoff(plan)
        job.message = f"holding {held}"

    def _restore_planned_reach(self, plan: dict, xy: list[float]) -> None:
        """Match the live reach to the plan's: back up when the dock landed short, creep in when it
        landed long. A long dock asks past the tilted envelope the receipt checked (e.g. 0.55 m at
        a pitch that only holds through ~0.50 m)."""
        want = plan.get("reach")
        if want is None:
            return
        for _ in range(12):
            live = self._reach_to(xy)
            if abs(live - want) <= 0.03:
                return
            o = self._state()["odometry"]
            # Short live reach → base too close → back up. Long → base too far → creep forward.
            sign = -1.0 if live < want else 1.0
            step = (sign * math.cos(o["yaw"]), sign * math.sin(o["yaw"]))
            self._translate_to(o["x"] + 0.03 * step[0], o["y"] + 0.03 * step[1], o["yaw"])

    def _unfurl_reach(self, xy: list[float], *, reach_cap: float | None = None) -> None:
        self.motion_policy.unfurl(self, xy, reach_cap=reach_cap)

    def _place(self, job: Job, plan: dict) -> None:
        # Flip-in-air: keep the planned release lift (turn_over already chose the lowest safe drop).
        # Raising it to the carry floor here made pieces bounce onto neighbours on the table.
        if "turnLift" in plan:
            plan["turnLift"] = max(float(plan["turnLift"]),
                                   float(plan["lift"]) + 0.2, self.config.tilted_carry_floor + 0.15)
            if plan.get("wristPitch") is None:
                plan["wristPitch"] = self.config.wrist_pitch_max
        self._dock(job, plan)
        if "wristYaw" in plan:
            self._ramp_joint("wrist_yaw", plan["wristYaw"], self.config.wrist_turn_s)
        grip = plan.get("gripSpot", plan["spot"])
        self._restore_planned_reach(plan, grip)
        if "turnLift" in plan:
            # Turned over in the air above the spot, then lowered as far as the arm reaches so.
            # Reach and tilt first: ramping lift alone through ~0.87 m with a short carrying reach
            # and pitch 1.05 has no USD IK solution (the servo target would sit in the hole).
            self._align_lateral(grip)
            self._reach_out(grip, plan.get("wristPitch"))
            self._set_joint("lift", plan["turnLift"])
            self._command({"wrist_roll": plan["wristRoll"], "turnOver": True})
            self._set_joint("wrist_roll", plan["wristRoll"], ramp=False)
        elif 'assemblyTurnLift' in plan:
            self._set_joint('lift', plan['assemblyTurnLift'])
            self._align_lateral(grip)
            self._reach_out(grip, plan.get('wristPitch'))
            self._ramp_joint('wrist_roll', plan['wristRoll'], self.config.wrist_turn_s)
            self._set_joint('wrist_roll', plan['wristRoll'], ramp=False, tol=.02)
        elif "wristRoll" in plan:
            self._ramp_joint("wrist_roll", plan["wristRoll"], self.config.wrist_turn_s)
        if "placementToleranceM" in plan:
            self._align_lateral(grip, tol=min(self.config.dock_position_tol, plan["placementToleranceM"] / 4))
        else:
            self._align_lateral(grip)
        self._restore_planned_reach(plan, grip)
        self._reach_over(plan, grip, plan["lift"])
        self._settle_for_release()
        if not self._over_support(plan):
            error = float(np.linalg.norm(self._gripper_xy() - np.asarray(grip))) if "placementToleranceM" in plan else None
            self._withdraw(plan)
            self._backoff(plan)
            detail = (f"; [RSI_MATING_RELEASE] pre-retreat XY error {error:.3f} m, "
                      f"limit {plan['placementToleranceM']:.3f} m or release outside support" if error is not None else "")
            raise SkillFailed(f"gripper is not over {plan['support']}; keeping {plan['object']}" + detail)
        if self._validate_placement is not None:
            self._validate_placement(plan)
        self._command({"gripper": "open"})
        tip = float(plan["wristRoll"]) if "wristRoll" in plan else None
        self._released(plan["object"], plan["spot"], target=plan.get("placementTarget"), tip_roll=tip)
        stamp = self._state().get('stamp')
        if stamp is not None:
            plan['releasedAt'] = stamp
        insert_release = bool(plan.get("insert") or plan.get("insertRelease"))
        if insert_release:
            self._wait(self.config.insert_release_wait_s)
        else:
            self._wait(self.config.release_wait_s)
            # Flip-in-air: let the piece finish falling before any exit motion.
            if "turnLift" in plan:
                self._wait(max(0.5, self.config.release_wait_s))
        job.log.append(f"released {plan['object']} on {plan['support']}")
        if plan.get('stayNear') and 'turnLift' not in plan:
            # Clear vertically before retracting, but keep the base at this work
            # station. The next pick still receives its own checked approach.
            self._exit_long_reach(plan)
            for key in ('wrist_roll','wrist_yaw'):
                if abs(self._state()['joints'].get(key,0.)) >= self.config.joint_tol:
                    self._set_joint(key,0.)
            self._report_placement(job, plan['object'])
            return
        # Once the object is released the place has succeeded: an exit that hits a USD hole (common
        # after flip-in-air with a residual wrist composition) must not fail the skill.
        try:
            if insert_release:
                # Tip-down insert: linger, rise clear while still rolled, then draw back — do not
                # snap-retract or unroll beside the screw (that tips it out of the hole).
                try:
                    lift_now = float(self._state()["joints"].get("lift", plan["lift"]))
                    self._set_joint("lift", lift_now + self.config.insert_clear_dz)
                except (SkillFailed, OutOfReach):
                    pass
                self._wait(0.25)
                reach_now = float(self._state()["joints"].get("reach", 0.0))
                target_reach = max(self.config.retract_reach, reach_now - 0.18)
                try:
                    self._ramp_joint("reach", target_reach,
                                    abs(reach_now - target_reach) / max(self.config.gentle_lift_speed, 0.05))
                except (SkillFailed, OutOfReach):
                    try:
                        self._set_joint("reach", self.config.retract_reach, ramp=False)
                    except (SkillFailed, OutOfReach):
                        pass
                if "wristRoll" in plan and abs(float(self._state()["joints"].get("wrist_roll", 0.0))) >= self.config.joint_tol:
                    self._ramp_joint("wrist_roll", 0.0, self.config.wrist_turn_s)
                self._exit_long_reach(plan)
                self._set_joint("lift", self.config.stow_lift)
                if "wristYaw" in plan:
                    self._set_joint("wrist_yaw", 0.0)
                self._backoff(plan)
            elif "turnLift" in plan:
                # Drive the base clear before any arm motion: tip retract/lift/unroll over the
                # release pose shoved flipped pieces onto neighbours on the table.
                try:
                    self._backoff(plan)
                except (SkillFailed, OutOfReach):
                    pass
                try:
                    o = self._state()["odometry"]
                    back = (-math.cos(o["yaw"]), -math.sin(o["yaw"]))
                    self._translate_to(o["x"] + 0.20 * back[0], o["y"] + 0.20 * back[1], o["yaw"])
                except (SkillFailed, OutOfReach):
                    pass
                try:
                    self._set_joint("reach", self.config.retract_reach, ramp=False)
                except (SkillFailed, OutOfReach):
                    pass
                try:
                    self._set_joint("lift", plan["turnLift"])
                except (SkillFailed, OutOfReach):
                    pass
                self._command({"wrist_roll": 0.0, "turnOver": True})
                try:
                    self._set_joint("wrist_roll", 0.0, ramp=False)
                except (SkillFailed, OutOfReach):
                    pass
                self._exit_long_reach(plan)
                if "wristRoll" in plan and abs(float(self._state()["joints"].get("wrist_roll", 0.0))) >= self.config.joint_tol:
                    self._set_joint("wrist_roll", 0.0)
                self._set_joint("lift", self.config.stow_lift)
                if "wristYaw" in plan:
                    self._set_joint("wrist_yaw", 0.0)
                self._backoff(plan)
            else:
                self.motion_policy.after_place(self, plan)
        except (SkillFailed, OutOfReach) as exc:
            job.log.append(f"exit after place: {exc}")
            try:
                self._backoff(plan)
            except (SkillFailed, OutOfReach):
                pass
        job.message = f"placed {plan['object']} on {plan['support']}"
        self._report_placement(job, plan["object"])

    def _insert_screw_into_hole(self, job: Job, plan: dict) -> None:
        """Place the screw tip-down into a hole, verify it stands head-up in the hole, retry on miss.

        Each failed verify re-picks the screw and re-plans the place (via runtime callbacks) until
        ``maxAttempts`` is exhausted.
        """
        insert = dict(plan.get("insertParams") or {})
        screw_id = str(insert.get("object") or plan.get("object") or "")
        attempts = int(plan.get("maxAttempts") or insert.get("maxAttempts") or 3)
        place_plan = plan
        last_msg = "not seated"
        for attempt in range(1, attempts + 1):
            if self._stop.is_set():
                raise SkillStopped()
            holding = bool(self._state().get("gripper", {}).get("holding"))
            if not holding:
                if self._replan_pick is None or self._replan_insert_place is None:
                    raise SkillFailed(
                        f"insert attempt {attempt}: screw not in gripper and no replan hooks")
                pick_plan = self._replan_pick(screw_id)
                if pick_plan is None:
                    raise SkillFailed(
                        f"insert attempt {attempt}: cannot re-pick {screw_id} after a miss ({last_msg})")
                job.log.append(f"insert attempt {attempt}: re-picking {screw_id}")
                self._pick(job, pick_plan)
                place_plan = self._replan_insert_place(insert)
                if place_plan is None:
                    raise SkillFailed(
                        f"insert attempt {attempt}: cannot re-plan tip-down place for {screw_id}")
            self._place(job, place_plan)
            if self._verify_screw_seat is None:
                job.message = f"inserted {screw_id} (no seat verify configured)"
                return
            ok, msg = self._verify_screw_seat(screw_id)
            if ok:
                job.log.append(msg)
                job.message = f"inserted {screw_id} tip-down into hole"
                return
            last_msg = msg
            job.log.append(f"insert attempt {attempt}/{attempts} failed: {msg}")
            place_plan = plan
        raise SkillFailed(
            f"insert_screw_into_hole failed after {attempts} attempts ({last_msg}); "
            "screw must stand tip-down in the hole with head up")

    def _drive_screw(self, job: Job, plan: dict) -> None:
        """Roll the screwdriver bit-down (shaft must already cross the gripper), press onto the
        screw head, hold to tighten, then raise — still holding the driver for the next screw.

        Predict refuses an end-on grasp (shaft on the wrist-roll axis). Re-assert wrist roll after
        reaching out (dock/reach can clear the wrist).
        """
        self._dock(job, plan)
        grip = plan.get("gripSpot", plan["spot"])
        self._align_lateral(grip)
        self._restore_planned_reach(plan, grip)
        if "wristYaw" in plan:
            self._ramp_joint("wrist_yaw", plan["wristYaw"], self.config.wrist_turn_s)
        if "wristRoll" in plan:
            self._ramp_joint("wrist_roll", plan["wristRoll"], self.config.wrist_turn_s)
            job.log.append(f"stood screwdriver bit-down (roll={plan['wristRoll']:.2f})")
        self._set_joint("lift", plan["lift"])
        self._reach_out(grip, plan.get("wristPitch"))
        self._align_lateral(grip)
        # Reach/dock merges often drop wrist setpoints; lock bit-down again before pressing.
        if "wristYaw" in plan:
            self._set_joint("wrist_yaw", plan["wristYaw"])
        if "wristRoll" in plan:
            self._set_joint("wrist_roll", plan["wristRoll"])
        self._set_joint("lift", plan["pressLift"])
        self._wait(float(plan.get("driveS", 1.5)))
        job.log.append(f"drove {plan['target']} with {plan['object']}")
        # Raise while keeping the bit down so the next drive_screw starts upright.
        self._set_joint("lift", plan["lift"])
        try:
            self._exit_long_reach(plan)
        except (SkillFailed, OutOfReach) as exc:
            job.log.append(f"exit after drive_screw: {exc}")
        self._backoff(plan)
        job.message = f"drove {plan['target']} with {plan['object']}"

    def _pour(self, job: Job, plan: dict) -> None:
        """Hold the container low over the spot, tip it slowly about the arm, wait for the contents
        to fall out, raise it ``rise`` off the pile, set it upright again and back away still holding it."""
        self._dock(job, plan)
        grip = plan.get("gripSpot", plan["spot"])
        self._align_lateral(grip)
        self._restore_planned_reach(plan, grip)
        # Rim-tilted picks must keep pitch while reaching the pour spot (tool-down tops out ~0.50 m).
        pitch = plan.get("wristPitch")
        if pitch is None:
            pitch = float(self._state()["joints"].get("wrist_pitch", 0.0)) or None
        self._set_joint("lift", plan["lift"])
        self._reach_out(grip, pitch)
        # Re-assert pitch before tipping: a dock settle can under-report / drop tilt, and a tip
        # started tool-down leaves rim-held contents in the box.
        if pitch is not None and abs(float(pitch)) >= self.config.joint_tol:
            try:
                self._set_joint("wrist_pitch", float(pitch))
            except OutOfReach:
                pass
        self._tip(plan["lift"], plan["pivot"], 0.0, plan["roll"])
        self._wait(plan["pourS"])
        job.log.append(f"poured {plan['object']} over {plan['support']}")
        raised = plan["lift"] + plan["rise"]
        self._set_joint("lift", raised + plan["pivot"] * abs(math.sin(plan["roll"])))
        self._tip(raised, plan["pivot"], plan["roll"], 0.0)
        # A rim-tilted pour still sits past the tool-down envelope; exit like place.
        if pitch is not None and "wristPitch" not in plan:
            plan = {**plan, "wristPitch": pitch}
        self._exit_long_reach(plan)
        self._set_joint("lift", max(plan["lift"], self.config.stow_lift))
        self._backoff(plan)
        job.message = f"poured {plan['object']} over {plan['support']}"

    def _tip(self, lift0: float, pivot: float, roll0: float, roll1: float) -> None:
        """Turn the wrist roll from ``roll0`` to ``roll1`` about the held container's lower edge, ``pivot``
        metres from the gripper: the lift rises by pivot * |sin roll|, so that edge stays just above
        the surface and the contents slide out from low down instead of dropping."""
        # Hold the pour dock's pitch in setpoints while tipping: omitting pitch can let the next
        # whole-body solve drift tool-down and leave rim-held contents in the box.
        pitch = float(self._setpoints.get("wrist_pitch",
                                          self._state()["joints"].get("wrist_pitch", 0.0)))
        if abs(pitch) >= self.config.joint_tol:
            self._setpoints["wrist_pitch"] = pitch
        t0 = self._state()["stamp"]
        while True:
            st = self._tick()
            frac = min(1.0, (st["stamp"] - t0) / self.config.pour_tip_s)
            roll = roll0 + (roll1 - roll0) * frac * frac * (3.0 - 2.0 * frac)
            cmd = {"wrist_roll": roll, "lift": lift0 + pivot * abs(math.sin(roll))}
            if abs(pitch) >= self.config.joint_tol:
                cmd["wrist_pitch"] = pitch
            self._command(cmd)
            if frac >= 1.0:
                break
        self._set_joint("wrist_roll", roll1)

    def _drawer(self, job: Job, plan: dict) -> float:
        """Grasp a drawer handle, move the base along the drawer axis to slide it, release and back away."""
        self._dock(job, plan)
        handle = plan["handle"]
        grip_lift, tip_cap, reach = self.motion_policy.approach_handle(self, job, plan)
        # Brief pause so the USD TCP can catch the planar tip before attach (sim capture is 6 cm).
        # Do not wait a full settle_s here: that alone blew the 150 s skill wait on drawer open.
        self._wait(min(0.45, self.config.settle_s))
        self._command({"gripper": "close"})
        self._wait(self.config.grasp_wait_s)
        if not self._state()["gripper"].get("holdingHandle"):
            # Re-close once at the same pose (attach is evaluated on close).
            self._command({"gripper": "open"})
            self._wait(min(0.15, self.config.grasp_wait_s))
            self._command({"gripper": "close"})
            self._wait(self.config.grasp_wait_s)
        if not self._state()["gripper"].get("holdingHandle"):
            # A swollen front puts the hand past the real handle. Step the arm along its line.
            # When the dock stopped short of the front, the arm is already at its limit, so
            # the base has to come those centimetres closer.
            for delta in (*plan.get("handleProbesAlong", ()), None):
                self._command({"gripper": "open"})
                try:
                    if delta is None:
                        o = self._state()["odometry"]
                        heading = (math.cos(o["yaw"]), math.sin(o["yaw"]))
                        self._translate_to(o["x"] + heading[0] * 0.08, o["y"] + heading[1] * 0.08, o["yaw"])
                        self._align_lateral(handle[:2])
                        reach = max(0.05, min(tip_cap, self._reach_to(handle[:2])))
                        self._set_joint("reach", reach)
                    else:
                        self._set_joint("reach", max(0.02, min(tip_cap, reach + delta)))
                except (SkillFailed, OutOfReach):
                    continue
                self._command({"gripper": "close"})
                self._wait(self.config.grasp_wait_s)
                if self._state()["gripper"].get("holdingHandle"):
                    job.log.append("came closer for the handle" if delta is None
                                   else f"handle of {plan['drawer']} was {delta * 100:.0f} cm along the arm")
                    break
        if not self._state()["gripper"].get("holdingHandle"):
            # The handle estimate is off to one side of the real one. Step the base across and
            # reach again; the pull still ends on the planned pose.
            o = self._state()["odometry"]
            left = (-math.sin(o["yaw"]), math.cos(o["yaw"]))
            for side in plan.get("handleProbesSide", ()):
                self._command({"gripper": "open"})
                try:
                    self._translate_to(o["x"] + left[0] * side, o["y"] + left[1] * side, o["yaw"])
                    self._set_joint("reach", max(0.05, min(tip_cap, self._reach_to(handle[:2]))))
                except (SkillFailed, OutOfReach):
                    continue
                self._command({"gripper": "close"})
                self._wait(self.config.grasp_wait_s)
                if self._state()["gripper"].get("holdingHandle"):
                    job.log.append(f"handle of {plan['drawer']} was {side * 100:.0f} cm to the side")
                    break
        if not self._state()["gripper"].get("holdingHandle"):
            # Last tries: nudge the base in, then height / short-back probes. Close after place can
            # leave the TCP a few centimetres past or above the bar while XY still looks near enough.
            self._command({"gripper": "open"})
            try:
                o = self._state()["odometry"]
                heading = (math.cos(o["yaw"]), math.sin(o["yaw"]))
                self._translate_to(o["x"] + heading[0] * 0.10, o["y"] + heading[1] * 0.10, o["yaw"])
                self._align_lateral(handle[:2])
                self._set_joint("lift", grip_lift)
                self._set_joint("reach", max(0.05, min(tip_cap, self._reach_to(handle[:2]))))
                self._command({"gripper": "close"})
                self._wait(self.config.grasp_wait_s)
            except (SkillFailed, OutOfReach):
                pass
            if self._state()["gripper"].get("holdingHandle"):
                job.log.append("came closer again for the handle")
        if not self._state()["gripper"].get("holdingHandle"):
            here = max(0.05, min(tip_cap, float(self._state()["joints"].get("reach", reach))))
            for dz, dr in ((-0.03, 0.0), (-0.05, 0.0), (0.03, 0.0), (0.0, -0.03), (0.0, -0.06),
                           (-0.03, -0.04), (-0.05, -0.06)):
                self._command({"gripper": "open"})
                try:
                    self._set_joint("lift", max(self.config.handle_lift_floor - 0.02, grip_lift + dz))
                    self._set_joint("reach", max(0.05, min(tip_cap, here + dr)))
                except (SkillFailed, OutOfReach):
                    continue
                self._command({"gripper": "close"})
                self._wait(self.config.grasp_wait_s)
                if self._state()["gripper"].get("holdingHandle"):
                    job.log.append(f"handle of {plan['drawer']} at lift/reach probe "
                                   f"({dz * 100:.0f}/{dr * 100:.0f} cm)")
                    break
        if not self._state()["gripper"].get("holdingHandle"):
            grip = self._gripper_xy()
            self._command({"gripper": "open"})
            self._set_joint("reach", self.config.retract_reach)
            self._backoff(plan)
            raise SkillFailed(
                f"missed the handle of {plan['drawer']} at ({handle[0]:.2f}, {handle[1]:.2f}, {handle[2]:.2f}),"
                f" gripper ({grip[0]:.2f}, {grip[1]:.2f})")
        job.log.append(f"grasped the handle of {plan['drawer']}")
        start = self._state()["odometry"]
        travel = plan['distance'] * plan['direction']
        self._translate_to(start['x'] + travel * plan['axis'][0], start['y'] + travel * plan['axis'][1],
                           plan["yaw"], self.config.drawer_speed)
        end = self._state()["odometry"]
        self._command({"gripper": "open"})
        self._wait(self.config.handle_release_wait_s)
        moved = (end["x"] - start["x"]) * plan["axis"][0] + (end["y"] - start["y"]) * plan["axis"][1]
        if self._shift is not None:
            self._shift(plan["drawer"], [moved * plan["axis"][0], moved * plan["axis"][1], 0.0])
        verb = "opened" if plan["direction"] > 0 else "closed"
        job.log.append(f"{verb} {plan['drawer']} by {abs(moved) * 100:.0f} cm")
        self.motion_policy.after_handle(self, plan)
        job.message = f"{verb} {plan['drawer']}"
        return moved

    def _place_floor(self, job: Job, plan: dict) -> None:
        """Drive in with the held object, lower it to just above the floor, release and back away.

        The arm keeps its carrying reach, or extends to the plan's ``reach`` where the object would
        otherwise land on the base: retracting it would pull the object over the base. ``wristRoll``
        tips the object (for example π/2 onto its side) before the gripper opens."""
        self._dock(job, plan, retract=False)
        if "reach" in plan:
            self._set_joint("reach", plan["reach"])
        if "wristRoll" in plan:
            self._ramp_joint("wrist_roll", plan["wristRoll"], self.config.wrist_turn_s)
        self._set_joint("lift", plan["lift"])
        self._command({"gripper": "open"})
        tip = float(plan["wristRoll"]) if "wristRoll" in plan else None
        self._released(plan["object"], plan["floor"], tip_roll=tip)
        self._wait(self.config.release_wait_s)
        job.log.append(f"released {plan['object']} on the floor at ({plan['floor'][0]:.2f}, {plan['floor'][1]:.2f})")
        if "wristRoll" in plan:
            self._set_joint("wrist_roll", 0.0)
        self._backoff(plan)
        self._set_joint("lift", self.config.stow_lift)
        job.message = f"placed {plan['object']} on the floor"
        self._report_placement(job, plan["object"])
