"""Injectable approach, withdrawal and recovery sequences for planar-arm skills.

Policies use the executor's motion primitives so commands retain its clock, limits and cancellation.
The default policy preserves the existing task paths; a body may replace it or supply its own settings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from steward.ego.motion_errors import OutOfReach, SkillFailed, SkillStopped

if TYPE_CHECKING:
    from steward.ego.skills import Job, SkillExecutor


@dataclass(frozen=True)
class MotionPolicyConfig:
    """Body-dependent recovery budget and intermediate TCP positions, in metres."""

    approach_floor: float = 0.85
    intermediate_reaches: tuple[float, ...] = (0.40, 0.44, 0.48)
    reach_attempts: int = 9
    unfurl_attempts: int = 6
    base_adjustment: float = 0.04
    redock_reach: float = 0.48
    recovery_lift_max: float = 1.28
    handle_lower_reaches: tuple[float, float] = (0.36, 0.42)
    handle_reach_cap: float = 0.44
    handle_lower_clearance: float = 0.10


class MotionPolicy(Protocol):
    """Motion sequences called synchronously inside a skill; cancellation must propagate."""

    def recover(self, driver: SkillExecutor) -> None: ...
    def approach(self, driver: SkillExecutor, plan: dict, xy: list[float], lift: float) -> None: ...
    def creep(self, driver: SkillExecutor, xy: list[float], capture: float, reach_cap: float | None = None) -> None: ...
    def reach(self, driver: SkillExecutor, xy: list[float], pitch: float | None) -> None: ...
    def extend_then_tilt(self, driver: SkillExecutor, reach: float, pitch: float) -> None: ...
    def withdraw(self, driver: SkillExecutor, plan: dict) -> None: ...
    def exit(self, driver: SkillExecutor, plan: dict) -> None: ...
    def unfurl(self, driver: SkillExecutor, xy: list[float], *, reach_cap: float | None = None) -> None: ...

    def prepare_dock(self, driver: SkillExecutor, job: Job, plan: dict, retract: bool) -> None: ...
    def approach_handle(self, driver: SkillExecutor, job: Job, plan: dict) -> tuple[float, float, float]: ...
    def after_place(self, driver: SkillExecutor, plan: dict) -> None: ...
    def after_handle(self, driver: SkillExecutor, plan: dict) -> None: ...
    def prepare_carry(self, driver: SkillExecutor, plan: dict) -> None: ...


class PlanarMotionPolicy:
    """Planar approach paths with configurable Piper workspace waypoints and recovery limits."""

    def __init__(self, settings: MotionPolicyConfig | None = None):
        self.settings = settings if settings is not None else MotionPolicyConfig()

    def after_place(self, driver: SkillExecutor, plan: dict) -> None:
        """Withdraw and restore the wrist, clearing the support before lowering to stow."""
        self.exit(driver, plan)
        if "wristRoll" in plan and abs(float(driver._state()["joints"].get("wrist_roll", 0.0))) >= driver.config.joint_tol:
            driver._set_joint("wrist_roll", 0.0)
        if "wristYaw" in plan:
            driver._set_joint("wrist_yaw", 0.0)
        driver._backoff(plan)
        driver._set_joint("lift", driver.config.stow_lift)

    def after_handle(self, driver: SkillExecutor, plan: dict) -> None:
        """Stow the planar slides and move to the drawer's viewing position."""
        driver._set_joint("lift", driver.config.stow_lift)
        driver._set_joint("reach", driver.config.retract_reach)
        driver._translate_to(plan["viewing"][0], plan["viewing"][1], plan["yaw"])

    def clear_handle_descent(self, driver: SkillExecutor, handle: list[float], reach: float) -> None:
        """Planar slides descend directly at their docked position."""

    def prepare_carry(self, driver: SkillExecutor, plan: dict) -> None:
        """Planar grasps can rise to carry height at their grasp position."""

    def recover(self, driver: SkillExecutor) -> None:
        """Rise ``failure_raise``, turn the wrist upright in place, then draw the arm back; each step
        is tried even where the one before it failed."""
        lift = driver._state()["joints"].get("lift")
        steps = [lambda: driver._set_joint("lift", lift + driver.config.failure_raise)] if lift is not None else []
        steps += [lambda: driver._ramp_joint("wrist_roll", 0.0, driver.config.wrist_turn_s),
                  lambda: driver._set_joint("reach", driver.config.retract_reach)]
        for step in steps:
            try:
                step()
            except SkillStopped:
                raise
            except Exception as exc:  # noqa: BLE001 - best effort after a failure that is already reported
                continue

    def approach(self, driver: SkillExecutor, plan: dict, xy: list[float], lift: float) -> None:
        """Put the gripper over ``xy`` at ``lift``; with the plan's ``approachLift`` (the spot lies
        below the rim of what holds it) the arm reaches out above the rim and goes straight down.

        Lowering onto ``lift`` with the tool still tilted can leave a short dock outside the
        workspace: back up and refresh the reach before each retry."""
        over = plan.get("approachLift")
        pitch = plan.get("wristPitch")
        if over is not None and over > lift:
            driver._set_joint("lift", over)
            driver._reach_out(xy, pitch)
        for attempt in range(self.settings.reach_attempts):
            try:
                driver._set_joint("lift", lift)
                break
            except OutOfReach:
                if attempt == self.settings.reach_attempts - 1:
                    raise
                # A deep dock approached tilted often cannot lower at that pitch: ease toward
                # tool-down (and shorten if tool-down cannot hold the full reach) before raising.
                pitch_now = float(driver._state()["joints"].get("wrist_pitch", 0.0))
                if abs(pitch_now) >= driver.config.joint_tol:
                    # Tool-down tops out near tool_down_reach_mid around drawer-close height; draw in
                    # before untilting.
                    td_cap = driver.config.tool_down_reach_mid
                    reach_now = float(driver._state()["joints"].get("reach", 0.0))
                    if reach_now > td_cap:
                        try:
                            driver._set_joint("reach", td_cap, ramp=False)
                        except OutOfReach:
                            pass
                    driver._set_joint("wrist_pitch", max(0.0, abs(pitch_now) - 0.2) * (1.0 if pitch_now >= 0 else -1.0))
                    continue
                lift = min(self.settings.recovery_lift_max, lift + 0.03)
                o = driver._state()["odometry"]
                back = (-math.cos(o["yaw"]), -math.sin(o["yaw"]))
                driver._translate_to(o["x"] + self.settings.base_adjustment * back[0], o["y"] + self.settings.base_adjustment * back[1], o["yaw"])
                try:
                    td_cap = driver.config.tool_down_reach_mid
                    driver._set_joint("reach", max(0.05, min(td_cap, driver._reach_to(xy))), ramp=False)
                except OutOfReach:
                    pass
        # After approachLift: re-tilt when closePitch is positive; for tool-down closes, creep the
        # base in instead of stretching back to the tilted dock reach (0.48 m @ 0.64 m @ pitch≈0
        # has no IK). Cap extensions at ``tool_down_reach_mid`` so drawer-close heights stay inside
        # the USD envelope.
        if over is not None and over > lift:
            close = plan.get("closePitch")
            if close is not None and abs(float(close)) >= driver.config.joint_tol:
                try:
                    driver._reach_out(xy, float(close))
                except OutOfReach:
                    # 0.45 m @ ~0.64 m @ 0.70 rad often has no USD IK: rise and ease the tilt, but
                    # keep a mild positive pitch — full untilt at mid reach sits in the tool-down hole.
                    driver._set_joint("lift", min(self.settings.recovery_lift_max, float(driver._state()["joints"].get("lift", lift)) + 0.12))
                    eased = max(0.25, abs(float(close)) - 0.25) * (1.0 if float(close) >= 0 else -1.0)
                    try:
                        driver._reach_out(xy, eased)
                    except OutOfReach:
                        td_cap = driver.config.tool_down_reach_mid
                        driver._set_joint("lift", min(self.settings.recovery_lift_max, float(driver._state()["joints"].get("lift", lift)) + 0.08))
                        try:
                            driver._set_joint("reach", min(td_cap, float(driver._state()["joints"].get("reach", td_cap))),
                                           ramp=False)
                        except OutOfReach:
                            pass
                        driver._reach_out(xy, 0.25 if float(close) >= 0 else -0.25)
            else:
                driver._creep_to(xy, plan.get("captureRadius", 0.05),
                              reach_cap=plan.get("closeReach", driver.config.tool_down_reach_mid))
        else:
            driver._reach_out(xy, pitch)

    def creep(self, driver: SkillExecutor, xy: list[float], capture: float, reach_cap: float | None = None) -> None:
        """Translate the base forward until the gripper is within ``capture`` of ``xy``, extending
        the arm only as far as ``reach_cap`` (default ``tool_down_reach_mid``) allows tool-down."""
        last = float("inf")
        stalled = 0
        cap = driver.config.tool_down_reach_mid if reach_cap is None else float(reach_cap)
        for _ in range(16):
            miss = float(np.linalg.norm(driver._gripper_xy() - np.array(xy)))
            if miss <= capture:
                return
            if miss >= last - 0.002:
                stalled += 1
                if stalled >= 2:
                    # Base blocked or TCP not tracking across two samples.
                    return
            else:
                stalled = 0
            last = miss
            o = driver._state()["odometry"]
            forward = (math.cos(o["yaw"]), math.sin(o["yaw"]))
            step = min(0.05, max(0.02, miss - capture + 0.01))
            driver._translate_to(o["x"] + step * forward[0], o["y"] + step * forward[1], o["yaw"])
            driver._align_lateral(xy)
            try:
                driver._set_joint("reach", max(0.05, min(cap, driver._reach_to(xy))), ramp=False)
            except OutOfReach:
                # Cap may still sit on a ridge at this live height: ease a centimetre shorter.
                try:
                    driver._set_joint("reach", max(0.05, min(cap - 0.02, driver._reach_to(xy))), ramp=False)
                except OutOfReach:
                    pass

    def reach(self, driver: SkillExecutor, xy: list[float], pitch: float | None) -> None:
        """Reach the gripper out to ``xy``, tilting the tool to ``pitch`` on the way.

        Prefer tilting at the current reach then extending (a synchronized reach+pitch ramp can sit
        in the tool-down hole at half tilt). When the current reach is too short for that pitch
        (0.28 m @ ~0.63 m @ 0.57 rad has no USD IK), extend first at the current pitch, then tilt.

        A dock that lands short can put a highly tilted grasp just outside the workspace: back the
        base up a few centimetres and try again before giving up."""
        pitch_cmd = pitch
        # Table picks after HOME deploy often sit near 0.81 m, where tool-down has holes around
        # 0.22 m and 0.32–0.35 m; rise into the working / untilted-exit band before extending.
        if pitch_cmd is None or abs(float(pitch_cmd)) < 0.15:
            lift_now = float(driver._state()["joints"].get("lift", 0.0))
            safe = max(driver.config.working_lift, driver.config.untilted_exit_lift, self.settings.approach_floor)
            if lift_now < safe - 0.01:
                try:
                    driver._set_joint("lift", safe)
                except (OutOfReach, SkillFailed):
                    pass
        for attempt in range(self.settings.reach_attempts):
            reach = max(0.05, driver._reach_to(xy))
            joints = driver._state()["joints"]
            try:
                if pitch_cmd is not None and abs(joints.get("wrist_pitch", 0.0) - pitch_cmd) >= driver.config.joint_tol:
                    try:
                        driver._ramp_joints({"wrist_pitch": pitch_cmd},
                                         abs(pitch_cmd - joints.get("wrist_pitch", 0.0)) / driver.config.wrist_pitch_speed)
                    except OutOfReach:
                        # Short retract cannot take a high pitch at mid height, and tool-down may not
                        # reach the full dock either: step through mid reaches, then tilt, then finish.
                        driver._extend_then_tilt(reach, pitch_cmd)
                        return
                # Tool-down: step through envelope-safe mids so a single jump does not land in a hole.
                if pitch_cmd is None or abs(float(pitch_cmd)) < 0.15:
                    here = float(joints.get("reach", 0.0))
                    td = driver.config.tool_down_reach_mid
                    mids = [r for r in (driver.config.working_reach, td, *(max(td, r) for r in self.settings.intermediate_reaches[:2]))
                            if here + 1e-9 < r < reach - 1e-9]
                    for mid in mids:
                        try:
                            driver._set_joint("reach", mid, ramp=False)
                        except OutOfReach:
                            break
                driver._set_joint("reach", reach)
                return
            except OutOfReach:
                if attempt == self.settings.reach_attempts - 1:
                    raise
                # Live reach past what the planned pitch covers: escalate tilt, then shorten the dock
                # (creep in). Backing away lengthens reach and usually deepens the hole.
                cap = driver.config.wrist_pitch_max
                if pitch_cmd is not None and cap > pitch_cmd + driver.config.joint_tol:
                    pitch_cmd = min(cap, pitch_cmd + max(0.15, (cap - pitch_cmd) * 0.5))
                    continue
                if pitch_cmd is None:
                    raise
                o = driver._state()["odometry"]
                if reach > self.settings.redock_reach:
                    fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
                    driver._translate_to(o["x"] + self.settings.base_adjustment * fwd[0], o["y"] + self.settings.base_adjustment * fwd[1], o["yaw"])
                else:
                    back = (-math.cos(o["yaw"]), -math.sin(o["yaw"]))
                    driver._translate_to(o["x"] + self.settings.base_adjustment * back[0], o["y"] + self.settings.base_adjustment * back[1], o["yaw"])

    def extend_then_tilt(self, driver: SkillExecutor, reach: float, pitch: float) -> None:
        """Reach ``reach`` at ``pitch`` when tilt-at-current and extend-at-current-pitch both miss.

        Under the USD mounts a mid height like 0.63 m only reaches ~0.40 m tool-down, while a short
        retract cannot hold a deep-drawer pitch: step out to a mid reach, tilt, then finish."""
        here = float(driver._state()["joints"].get("reach", 0.0))
        td = driver.config.tool_down_reach_mid
        mids = [r for r in (td, *(max(td, r) for r in self.settings.intermediate_reaches))
                if here + 1e-9 < r < reach - 1e-9]
        last: Exception | None = None
        for mid in mids + [reach]:
            try:
                driver._set_joint("reach", mid)
                joints = driver._state()["joints"]
                if abs(joints.get("wrist_pitch", 0.0) - pitch) >= driver.config.joint_tol:
                    driver._ramp_joints({"wrist_pitch": pitch},
                                     abs(pitch - joints.get("wrist_pitch", 0.0)) / driver.config.wrist_pitch_speed)
                if mid + 1e-9 < reach:
                    driver._set_joint("reach", reach)
                return
            except OutOfReach as exc:
                last = exc
                continue
        if last is not None:
            raise last
        raise OutOfReach(f"cannot reach {reach:.2f} m at pitch {pitch:.2f}")

    def withdraw(self, driver: SkillExecutor, plan: dict) -> None:
        """Retract the arm, first back up above the rim it reached over (the plan's ``approachLift``).

        Tool-down tops out near 0.50 m reach under the USD mounts, so a long dock must draw back
        while still tilted (and high enough for ``retract_reach``) before untilting."""
        over = plan.get("approachLift")
        if over is not None and driver._state()["joints"]["lift"] < over:
            driver._set_joint("lift", over)
        driver._exit_long_reach(plan)

    def exit(self, driver: SkillExecutor, plan: dict) -> None:
        """Leave a long dock without crossing the tool-down / short-retract USD holes.

        Rise high enough for a tilted short retract, restore tilt if the wrist already untilted,
        draw back to ``retract_reach``, then untilt.

        FK of a settling USD posture can under-report ``wrist_pitch`` while the virtual target still
        carries tilt: also consult ``_setpoints`` so a short retract does not keep that hidden pitch."""
        pitch_now = abs(float(driver._state()["joints"].get("wrist_pitch", 0.0)))
        pitch_cmd = abs(float(driver._setpoints.get("wrist_pitch", 0.0)))
        plan_pitch = abs(float(plan.get("wristPitch") or 0.0))
        need_tilt = max(pitch_now, pitch_cmd, plan_pitch) >= 0.3
        floor = driver.config.tilted_exit_lift if need_tilt else driver.config.untilted_exit_lift
        driver._set_joint("lift", max(float(driver._state()["joints"].get("lift", 0.0)), floor, driver.config.stow_lift))
        if need_tilt and max(pitch_now, pitch_cmd) < 0.5:
            driver._set_joint("wrist_pitch", max(plan_pitch, pitch_cmd, driver.config.exit_retilt_min))
        driver._set_joint("reach", driver.config.retract_reach)
        if max(abs(float(driver._state()["joints"].get("wrist_pitch", 0.0))),
               abs(float(driver._setpoints.get("wrist_pitch", 0.0)))) >= driver.config.joint_tol:
            driver._untilt()

    def unfurl(self, driver: SkillExecutor, xy: list[float], *, reach_cap: float | None = None) -> None:
        """Extend toward ``xy`` through mid setpoints the USD IK holds from rest.

        A resting PARK posture FK-reads ~1.74 m lift / ~0 reach: commanding only ``reach`` keeps that
        seed height and every solve misses. Leave rest with one ``lift``+``reach`` command into the
        hole-free band (0.85 m × 0.28 m), then step reach with a settle tol tighter than the default
        0.10 m (which accepts the folded FK as already at retract_reach).

        ``reach_cap`` limits the extension (use ~0.40 m at predock so a far handle does not ask for
        an unreachable 0.55 m+ tip while only deploying off PARK)."""
        tight = 0.04
        unfurl_lift = max(driver.config.stow_lift, driver.config.working_lift, self.settings.approach_floor)
        cap = driver.config.reach_max if reach_cap is None else min(driver.config.reach_max, reach_cap)
        target = max(0.05, min(cap, driver._reach_to(xy)))
        here = float(driver._state()["joints"].get("reach", 0.0))
        # Already at the handle reach (e.g. deploy left working_reach): do not raise lift alone —
        # that re-solve can collapse the tip back onto the HOME ridge (~0.22 m).
        if here >= target - tight:
            return
        # First planar command from a folded tip must set lift and reach together (see PiperArm._solve).
        # Dock retract can clear ``armResting`` while FK is still ~0.22 m (joint_tol 0.10 accepted it
        # as retract_reach): still treat that as folded and re-seed.
        first = max(driver.config.retract_reach, min(target, driver.config.working_reach))
        tip_lift = float(driver._state()["joints"].get("lift", 0.0))
        # PARK FK tip ~1.74 m, or a wrist-only seed that inherited it: always re-seed lift+reach.
        folded = (bool(driver._state().get("armResting")) or here < driver.config.retract_reach - 0.04
                  or tip_lift > unfurl_lift + 0.25)
        if folded:
            driver._setpoints["lift"] = unfurl_lift
            driver._setpoints["reach"] = first
            driver._command({"lift": unfurl_lift, "reach": first, "wrist_pitch": 0.0})
            t0 = driver._state()["stamp"]
            while True:
                st = driver._tick()
                if (abs(st["joints"].get("lift", 0.0) - unfurl_lift) < 0.08
                        and abs(st["joints"].get("reach", 0.0) - first) < tight):
                    break
                if st["stamp"] - t0 > 20.0:
                    raise SkillFailed(
                        f"joint reach did not unfurl to {first:.3f} at {unfurl_lift:.2f} m "
                        f"(at {st['joints'].get('reach', 0.0):.3f} / lift {st['joints'].get('lift', 0.0):.2f})")
            here = float(driver._state()["joints"].get("reach", 0.0))
        else:
            # Drawer open already lowered to the handle band: yanking lift back to the deploy tip
            # (0.85 m) undoes that and the next mid-reach step often collapses onto the HOME ridge.
            if tip_lift < unfurl_lift - 0.1:
                pass
            else:
                driver._set_joint("lift", max(tip_lift, unfurl_lift))
        for attempt in range(self.settings.unfurl_attempts):
            target = max(0.05, min(cap, driver._reach_to(xy)))
            here = float(driver._state()["joints"].get("reach", 0.0))
            if here >= target - tight:
                return
            # At handle height, step through shorter mids; the working tip band is for the deploy tip.
            td = driver.config.tool_down_reach_mid
            mid_ladder = ((driver.config.working_reach, td, *(max(td, r) for r in self.settings.intermediate_reaches))
                          if float(driver._state()["joints"].get("lift", 0.0)) >= unfurl_lift - 0.1
                          else (max(0.28, td - 0.06), max(0.30, td - 0.02), td, max(td, 0.40), max(td, 0.42)))
            mids = [r for r in mid_ladder if here + tight < r < target - tight]
            last_err: Exception | None = None
            for mid in mids + [target]:
                try:
                    driver._set_joint("reach", mid, ramp=False, timeout=16.0, tol=tight)
                    here = float(driver._state()["joints"].get("reach", 0.0))
                    last_err = None
                except (SkillFailed, OutOfReach) as exc:
                    last_err = exc
                    break
            here = float(driver._state()["joints"].get("reach", 0.0))
            if last_err is None and here >= target - tight:
                return
            # A mid step can collapse back onto the folded HOME ridge (~0.21 m): re-seed lift+reach
            # before the next attempt instead of walking the base with a dead tip.
            if here < driver.config.retract_reach - 0.04:
                driver._setpoints["lift"] = unfurl_lift
                driver._setpoints["reach"] = first
                driver._command({"lift": unfurl_lift, "reach": first, "wrist_pitch": 0.0})
                driver._wait(driver.config.settle_s)
            if attempt == self.settings.unfurl_attempts - 1:
                if last_err is not None:
                    raise last_err
                raise SkillFailed(f"joint reach did not reach {target:.3f} (at {here:.3f})")
            if last_err is None and here >= target - tight:
                return
            o = driver._state()["odometry"]
            fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
            driver._translate_to(o["x"] + 0.03 * fwd[0], o["y"] + 0.03 * fwd[1], o["yaw"])
            driver._align_lateral(xy)


    def prepare_dock(self, driver: SkillExecutor, job: Job, plan: dict, retract: bool) -> None:
        """Deploy or retract the arm after the base has verified its docking pose."""
        # PARK/zero rest cannot one-shot IK to a working tip (servo stalls ~0.21 m); unfold via HOME.
        deployed = False
        if driver._deploy_working is not None:
            before = float(driver._state()["joints"].get("reach", 0.0))
            if driver._state().get("armResting") or before < driver.config.retract_reach - 0.04:
                job.log.append(f"deploying working posture (reach was {before:.3f})")
                driver._deploy_working(driver._tick)
                # Re-assert a hole-free planar tip and sync setpoints from the live virtual joints.
                # Leaving the arm on the HOME FK ridge (~0.21–0.32 m) makes the next handle dock
                # settle short of its setpoint. Prefer working_reach (past the 0.32 m hole), not
                # retract_reach which sits on that ridge.
                tip_lift = max(driver.config.working_lift, driver.config.stow_lift, self.settings.approach_floor)
                tip_reach = driver.config.working_reach
                driver._command({"lift": tip_lift, "reach": tip_reach, "wrist_pitch": 0.0,
                               "wrist_yaw": 0.0, "wrist_roll": 0.0})
                driver._wait(driver.config.settle_s)
                after = float(driver._state()["joints"].get("reach", 0.0))
                job.log.append(f"deployed working posture (reach now {after:.3f})")
                driver._setpoints.update({k: float(driver._state()["joints"].get(k, 0.0))
                                        for k in ("lift", "reach", "wrist_pitch", "wrist_yaw", "wrist_roll")
                                        if k in driver._state()["joints"]})
                if after < tip_reach - 0.04:
                    raise SkillFailed(
                        f"arm did not leave its folded tip after deploy (reach {after:.3f})")
                deployed = True
        if retract:
            # Only what the arm holds is held at the plan's tilt; an empty gripper points down.
            # Clear ghost tilt before an empty short retract: FK can read ~0 while the virtual
            # target still carries pitch, and 0.28 m @ mid height @ that pitch has no reliable IK.
            held = bool(driver._state()["gripper"].get("holding"))
            parked = plan if held else {}
            pitch_now = max(abs(float(driver._state()["joints"].get("wrist_pitch", 0.0))),
                            abs(float(driver._setpoints.get("wrist_pitch", 0.0))),
                            abs(float(parked.get("wristPitch") or 0.0)))
            if held and pitch_now >= 0.3:
                driver._exit_long_reach({**parked, "wristPitch": parked.get("wristPitch", pitch_now)})
            elif deployed:
                # Just unfolded to the working tip; a further empty retract can throw the arm back
                # onto the folded ridge (FK falls to ~0.22 m). Keep the working tip.
                driver._setpoints["wrist_pitch"] = 0.0
                driver._command({"wrist_pitch": 0.0})
            else:
                # Force the virtual target down even when FK already reads ~0 (otherwise a later
                # lift/reach merge keeps the ghost pitch and hits the short-retract hole).
                driver._setpoints["wrist_pitch"] = 0.0
                driver._command({"wrist_pitch": 0.0})
                if float(driver._state()["joints"].get("reach", 0.0)) > driver.config.retract_reach + 0.05:
                    driver._set_joint("lift", max(float(driver._state()["joints"].get("lift", 0.0)),
                                                driver.config.untilted_exit_lift, driver.config.stow_lift))
                driver._retract(parked if held else {})

    def approach_handle(self, driver: SkillExecutor, job: Job, plan: dict) -> tuple[float, float, float]:
        """Approach a handle, logging recovery on ``job``; return grip height, reach cap and target reach."""
        handle = plan["handle"]
        driver._align_lateral(handle[:2])
        # Lower at a short tip, then extend in the handle band. One-shot 0.85→0.65 at
        # ``working_reach`` (~0.40 m) crosses mid-height holes and leaves FK on the folded ridge
        # (~0.22 m); the next ``_set_joint(reach, ~0.41)`` then fails with "joint reach did not
        # reach". ``tool_down_reach_mid`` is clear across the drawer-close heights.
        grip_lift = max(float(handle[2]), driver.config.handle_lift_floor)
        work_lift = max(driver.config.working_lift, driver.config.stow_lift, self.settings.approach_floor)
        # Lower in the 0.36–0.42 m band (clear at every 0.05 m step from 0.85→0.65). Prefer the
        # live tip after deploy (~working_reach) over yanking to exact 0.36 — the servo often keeps
        # ~0.41 m, and a strict 0.05 m hold then fails as "lower-0.65 tip did not hold (0.69/0.41)".
        short_lo, short_hi = self.settings.handle_lower_reaches

        def _hold_tip(lift: float, reach: float, label: str, *, reach_tol: float = 0.08) -> None:
            driver._setpoints["lift"] = lift
            driver._setpoints["reach"] = reach
            driver._command({"lift": lift, "reach": reach, "wrist_pitch": 0.0})
            driver._wait(driver.config.settle_s)
            live = driver._state()["joints"]
            live_reach = float(live.get("reach", 0.0))
            if abs(float(live.get("lift", 0.0)) - lift) > 0.12:
                raise SkillFailed(f"{label} tip did not hold ({live.get('lift', 0.0):.2f}/"
                                  f"{live_reach:.2f})")
            if live_reach < driver.config.retract_reach - 0.04:
                raise SkillFailed(f"{label} reach collapsed ({live_reach:.2f})")
            if abs(live_reach - reach) > reach_tol and not (short_lo - 0.02 <= live_reach <= short_hi + 0.02):
                raise SkillFailed(f"{label} tip did not hold ({live.get('lift', 0.0):.2f}/"
                                  f"{live_reach:.2f})")

        def _reseed_short() -> None:
            if driver._deploy_working is not None and (
                    driver._state().get("armResting")
                    or float(driver._state()["joints"].get("reach", 0.0)) < driver.config.retract_reach - 0.04):
                driver._deploy_working(driver._tick)
            tip = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", short_lo))))
            _hold_tip(work_lift, tip, "short-working")

        def _step_lower(reach: float) -> None:
            level = float(driver._state()["joints"].get("lift", work_lift))
            while level > grip_lift + 0.03:
                level = max(grip_lift, level - 0.05)
                # Keep commanding the live tip's reach so a 0.41 m FK is not forced back to 0.36.
                tip = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", reach))))
                _hold_tip(level, tip, f"lower-{level:.2f}")
            tip = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", reach))))
            _hold_tip(grip_lift, tip, "handle-short")

        live0 = float(driver._state()["joints"].get("reach", short_lo))
        short = min(short_hi, max(short_lo, live0 if live0 >= short_lo - 0.02 else short_lo))
        try:
            _hold_tip(work_lift, short, "short-working")
        except (OutOfReach, SkillFailed):
            _reseed_short()
            short = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", short_lo))))
        # Creep in so the handle only needs a short-to-mid tip after the lower.
        for _ in range(12):
            need = max(0.05, driver._reach_to(handle[:2]))
            if need <= short_hi + 0.02:
                break
            o = driver._state()["odometry"]
            fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
            driver._translate_to(o["x"] + 0.03 * fwd[0], o["y"] + 0.03 * fwd[1], o["yaw"])
            driver._align_lateral(handle[:2])
        short = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", short))))
        self.clear_handle_descent(driver, handle, short_hi)
        try:
            _step_lower(short)
        except (OutOfReach, SkillFailed) as exc:
            job.log.append(f"handle lower failed ({exc}); reseed and retry")
            _reseed_short()
            short = min(short_hi, max(short_lo, float(driver._state()["joints"].get("reach", short_lo))))
            _step_lower(short)
        driver._align_lateral(handle[:2])
        # Extend at grip height. If FK collapsed, reseed short tip before asking for more reach.
        target = max(0.05, min(self.settings.handle_reach_cap, driver._reach_to(handle[:2])))
        here = float(driver._state()["joints"].get("reach", 0.0))
        if here < driver.config.retract_reach - 0.04:
            _reseed_short()
            try:
                _hold_tip(grip_lift, short, "handle-after-collapse")
            except (OutOfReach, SkillFailed):
                pass
            here = float(driver._state()["joints"].get("reach", 0.0))
        for mid_r in [r for r in (short, 0.38, 0.40, 0.42) if here + 0.02 < r < target - 0.02] + [target]:
            try:
                driver._setpoints["lift"] = float(driver._state()["joints"].get("lift", grip_lift))
                driver._setpoints["reach"] = mid_r
                driver._command({"lift": driver._setpoints["lift"], "reach": mid_r, "wrist_pitch": 0.0})
                driver._wait(min(driver.config.settle_s, 1.5))
                here = float(driver._state()["joints"].get("reach", 0.0))
                if here < mid_r - 0.05:
                    raise SkillFailed(f"joint reach did not reach {mid_r:.3f} (at {here:.3f})")
            except (SkillFailed, OutOfReach):
                o = driver._state()["odometry"]
                fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
                driver._translate_to(o["x"] + 0.03 * fwd[0], o["y"] + 0.03 * fwd[1], o["yaw"])
                driver._align_lateral(handle[:2])
                if float(driver._state()["joints"].get("reach", 0.0)) < driver.config.retract_reach - 0.04:
                    _reseed_short()
                    try:
                        _hold_tip(grip_lift, short, "handle-reseed")
                    except (OutOfReach, SkillFailed):
                        pass
                break
        capture = float(plan.get("captureRadius", 0.05))
        # Cap tip while creeping the base in: USD tool-down holes above ~0.44 m at grip height.
        tip_cap = self.settings.handle_reach_cap
        for _ in range(8):
            miss = float(np.linalg.norm(driver._gripper_xy() - np.array(handle[:2])))
            if miss <= capture:
                break
            o = driver._state()["odometry"]
            fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
            driver._translate_to(o["x"] + 0.03 * fwd[0], o["y"] + 0.03 * fwd[1], o["yaw"])
            driver._align_lateral(handle[:2])
            try:
                driver._set_joint("reach", max(0.05, min(tip_cap, driver._reach_to(handle[:2]))),
                                ramp=False, timeout=12.0, tol=0.04)
            except (SkillFailed, OutOfReach):
                pass
        miss = float(np.linalg.norm(driver._gripper_xy() - np.array(handle[:2])))
        if miss > capture:
            driver._align_lateral(handle[:2])
            try:
                driver._set_joint("reach", max(0.05, min(tip_cap, driver._reach_to(handle[:2]))),
                                ramp=False, timeout=12.0, tol=0.04)
            except (SkillFailed, OutOfReach):
                pass
            miss = float(np.linalg.norm(driver._gripper_xy() - np.array(handle[:2])))
        if miss > capture:
            driver._set_joint("lift", driver.config.stow_lift)
            driver._set_joint("reach", driver.config.retract_reach)
            driver._backoff(plan)
            raise SkillFailed(f"gripper is {miss * 100:.0f} cm from the handle of {plan['drawer']}; not closing")
        reach = max(0.05, min(tip_cap, driver._reach_to(handle[:2])))
        return grip_lift, tip_cap, reach


class PiperMotionPolicy(PlanarMotionPolicy):
    """Articulated Piper paths, resolved with each body's workspace and trajectory settings."""

    def approach(self, driver: SkillExecutor, plan: dict, xy: list[float], lift: float) -> None:
        """Descend in the exposed drawer gap before approaching a grasp under the cabinet top."""
        if "drawerEntry" not in plan:
            return super().approach(driver, plan, xy, lift)
        driver._set_joint("lift", max(plan["approachLift"], driver.config.working_lift))
        driver._set_joint("reach", plan["closeReach"])
        driver._set_joint("wrist_pitch", plan["closePitch"])
        driver._wait(driver.config.settle_s)
        o = driver._state()["odometry"]
        heading = np.array([math.cos(o["yaw"]), math.sin(o["yaw"])])
        left = np.array([-heading[1], heading[0]])
        delta = (np.array(xy) - plan["drawerEntry"] * heading
                 + plan["drawerEntrySide"] * left - driver._gripper_xy())
        driver._translate_to(o["x"] + delta[0], o["y"] + delta[1], o["yaw"], tol=driver.config.dock_position_tol)
        driver._set_joint("lift", lift)
        driver._settle_for_release()
        o = driver._state()["odometry"]
        delta = np.array(xy) - driver._gripper_xy()
        driver._translate_to(o["x"] + delta[0], o["y"] + delta[1], o["yaw"], tol=driver.config.dock_position_tol)

    def prepare_carry(self, driver: SkillExecutor, plan: dict) -> None:
        """Leave the space below the cabinet top before lifting a drawer grasp."""
        if "drawerExit" in plan:
            o = driver._state()["odometry"]
            c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
            driver._translate_to(o["x"] - plan["drawerExit"] * c - plan["drawerEntrySide"] * s,
                                 o["y"] - plan["drawerExit"] * s + plan["drawerEntrySide"] * c, o["yaw"])

    def withdraw(self, driver: SkillExecutor, plan: dict) -> None:
        """Return through the drawer opening before raising after a missed grasp."""
        self.prepare_carry(driver, plan)
        super().withdraw(driver, plan)

    def clear_handle_descent(self, driver: SkillExecutor, handle: list[float], reach: float) -> None:
        """Keep fingers outside the cabinet while descending with the articulated arm extended."""
        back = reach + self.settings.handle_lower_clearance - driver._reach_to(handle[:2])
        if back > 0:
            o = driver._state()["odometry"]
            driver._translate_to(o["x"] - back * math.cos(o["yaw"]),
                                 o["y"] - back * math.sin(o["yaw"]), o["yaw"])
            driver._align_lateral(handle[:2])

    def after_handle(self, driver: SkillExecutor, plan: dict) -> None:
        """Cross the short-reach band at working height before retreating and lowering to stow."""
        driver._set_joint("lift", max(driver.config.working_lift, driver.config.stow_lift,
                                      driver.config.untilted_exit_lift))
        driver._set_joint("reach", driver.config.retract_reach)
        driver._translate_to(plan["viewing"][0], plan["viewing"][1], plan["yaw"])
        driver._set_joint("lift", driver.config.stow_lift)
