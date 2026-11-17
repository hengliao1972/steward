"""Planar task heights derived from a Piper arm's IK workspace."""
from steward.ego.arm import Arm


def piper_envelope(arm: Arm, retract_reach: float) -> dict[str, float]:
    """Lift floors implied by the USD IK envelope for ``arm``, not scene-specific constants.

    ``tilted_exit_lift``: lowest height where a most-tilted short retract is freshly reachable.
    ``untilted_exit_lift``: same for tool-down at a mid place reach (tool-down tops out near 0.50 m).
    ``tilted_carry_floor``: lowest height where a most-tilted mid reach is freshly reachable.
    ``elevated_grasp_floor``: lowest height where a deep dock (~0.52 m) needs a positive pitch.
    """
    pitch = arm.geometry.wrist_pitch_max
    min_lift_for = getattr(arm, "min_lift_for", None)
    pitch_for = arm.pitch_for
    if min_lift_for is None:
        return {"tilted_exit_lift": 1.05, "untilted_exit_lift": 0.85,
                "tilted_carry_floor": 0.95, "elevated_grasp_floor": 0.72, "handle_lift_floor": 0.65,
                "tool_down_reach_mid": 0.36, "working_lift": 0.85, "working_reach": 0.40}
    def floor(value: float | None, default: float, margin: float = 0.02) -> float:
        return (default if value is None else value) + margin

    # Carry/exit use the short-retract and mid-reach envelopes. Exit floors add joint_tol (0.08) so
    # a settle that stops short of the setpoint still clears the IK ridge (1.02 m @ pitch 1.05
    # cannot take retract_reach; 1.05 m can).
    settle_slack = 0.08
    tilted_exit = floor(min_lift_for(retract_reach, pitch), 1.05, margin=0.02 + settle_slack)
    untilted_exit = floor(min_lift_for(0.45, 0.0), 0.85, margin=0.02 + settle_slack)
    # Flip/place carries often keep ~0.40 m reach while most-tilted; that ridge is higher than 0.45 m.
    tilted_carry = floor(min_lift_for(0.40, pitch), 0.95, margin=0.02 + settle_slack)
    # Deep dock: first height where a 0.52 m reach needs a decisive tilt (not a shallow ridge).
    elevated = 0.72
    lo, hi = arm.geometry.lift_min, arm.geometry.lift_max
    for k in range(0, int((hi - lo) / 0.02) + 1):
        h = lo + 0.02 * k
        pf = pitch_for(h, 0.52)
        if pf is not None and pf >= 0.35:
            elevated = h
            break
    # Handle docks ask ~0.40 m tool-down; below this TCP height that reach has no fresh IK.
    handle_lift = floor(min_lift_for(0.42, 0.0), 0.65)
    # Farthest tool-down reach across the drawer-close band (0.58–0.62 m): use the strictest
    # height so creep/probes never ask past a hole that opens a centimetre lower.
    tool_down_mid = 0.36
    for r in (0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.28):
        if all(arm.reachable(h, r, wrist_pitch=0.0) for h in (0.58, 0.60, 0.62)):
            tool_down_mid = r
            break
    # Working tip after HOME: past the 0.32 m tool-down ridge. Prefer ≥0.85 m — untilted_exit can
    # sit near 0.75 m where 0.32 m is a hole, and settle then lands inside that ridge.
    working_lift = max(untilted_exit, 0.85)
    working_reach = max(tool_down_mid, 0.40)
    for r in (0.42, 0.40, tool_down_mid, 0.36):
        if arm.reachable(working_lift, r, wrist_pitch=0.0):
            working_reach = r
            break
    return {"tilted_exit_lift": tilted_exit, "untilted_exit_lift": untilted_exit,
            "tilted_carry_floor": tilted_carry, "elevated_grasp_floor": elevated,
            "handle_lift_floor": handle_lift, "tool_down_reach_mid": tool_down_mid,
            "working_lift": working_lift, "working_reach": working_reach}
