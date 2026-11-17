"""What the Ego Runtime needs to drive each registered body: its model, its planar arm
(:mod:`steward.ego.arm`) and the planner and motion settings that differ between bodies.

Every body runs the same Ego Runtime (``steward ego --body <id>``): perception, the Ego-State,
ESim receipts and the skills are the kitchen robot's; a body changes only what its geometry and
actuators require.
"""

from __future__ import annotations

import dataclasses

from steward.ego.arm import Arm
from steward.ego.piper_workspace import piper_envelope
from steward.ego.runtime import RuntimeConfig
from steward.embodiments import robot_entity


def robot_model(body: str) -> dict:
    """The body's model as its generator builds it (the same one the scenes compile)."""
    return robot_entity(body)


def body_arm(body: str, model: dict) -> Arm | None:
    """The body's planar arm; None for the kitchen robot, whose own joints are the planar arm."""
    if body == "agilex-usd":
        from steward.ego.agilex import configured_arm

        return configured_arm(model)[0]
    if body == "cobot-magic":
        from steward.ego.piper_arm import PiperArm, PiperArmConfig

        # Match MotionConfig working tip so deploy_working and skills land on the same hole-free seed.
        arm = PiperArm(model)
        floors = piper_envelope(arm, retract_reach=0.28)
        return PiperArm(model, PiperArmConfig(working_lift=floors["working_lift"],
                                              working_reach=floors["working_reach"]))
    return None



def runtime_config(body: str, base: RuntimeConfig | None = None) -> RuntimeConfig:
    """``base`` (default settings) with the body's planner and motion settings."""
    cfg = base or RuntimeConfig()
    if body == "agilex-usd":
        from steward.ego.agilex import prepare_agilex

        return prepare_agilex(cfg)[2]
    if body != "cobot-magic":
        return cfg
    from steward.ego.piper_arm import PiperArm

    # The TRACER base is servoed at 0.5 m/s and 1 rad/s (assets.cobot.ROOT_ACTUATOR): commanded faster,
    # it lags its setpoint and every docking misses. The arm's joints, servoed straight to a solved
    # posture, sweep the wrist off the line the grasp point is to move along, so its moves are
    # ramped. A retracted arm stays inside the reach its tool-down workspace has; tilted, the tool
    # reaches 0.70 m. Its reach is still short, so a large object is taken by the edge nearest the
    # robot (it then hangs ahead of the hand) and set down with a 5 cm margin inside the surface's edge.
    # Its gripper cannot point up at a surface, so it turns things over in the air.
    # Under the USD mounts, tool-down at stow height (0.60 m) only reaches from about 0.18 m out:
    # retract_reach 0.15 left every drawer open stuck when drawing the arm back.
    retract_reach = 0.28
    arm = PiperArm(robot_model(body))
    arm.virtualize({"joints": dict(arm.initial),
                    "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0}})
    floors = piper_envelope(arm, retract_reach)
    motion = dataclasses.replace(cfg.motion, linear_speed=0.45, angular_speed=0.9, look_angular_speed=0.6,
                                 gentle_speed=0.25, gentle_turn_speed=0.4, retract_reach=retract_reach, reach_max=0.70,
                                 # Twin arm mount is ~0.30 m off centreline; a short dock can leave
                                 # 15+ cm of lateral error after unfurl creep — allow a wider align.
                                 align_max=0.25,
                                 # Virtual TCP after a whole-body solve can sit a few centimetres off
                                 # the commanded lift/reach while the torso is still settling.
                                 joint_tol=0.10, arm_speed=0.3, settle_s=2.0, release_joint_tol=0.008,
                                 # USD whole-body TCP lags the planar tip report; a short grasp wait
                                 # closes before the pinch is inside the sim's 6 cm handle capture.
                                 # Keep under ~0.4 s: longer plus settle-before-close blew the 150 s
                                 # skill wait on a single drawer open.
                                 grasp_wait_s=0.35,
                                 # Stow above the drawer-handle band: at 0.60 m a retracted reach only
                                 # has limit-packed IK solutions the servo cannot hold.
                                 stow_lift=0.75,
                                 wrist_pitch_max=arm.geometry.wrist_pitch_max,
                                 tilted_exit_lift=floors["tilted_exit_lift"],
                                 untilted_exit_lift=floors["untilted_exit_lift"],
                                 tilted_carry_floor=floors["tilted_carry_floor"],
                                 handle_lift_floor=floors["handle_lift_floor"],
                                 tool_down_reach_mid=floors["tool_down_reach_mid"],
                                 working_lift=floors["working_lift"],
                                 working_reach=floors["working_reach"])
    # handle_reach 0.18 packs joint2 against its 3.14 rad stop at drawer-handle height (~0.61 m),
    # so the TCP settles ~6 cm long and the grasp misses; 0.25 m leaves headroom. capture_radius
    # covers residual TCP report noise after a whole-body solve.
    # handle_reach 0.18 packs joint2; 0.32 asks ~0.45 m at handle height and leaves the tool-down
    # envelope. 0.25 leaves stop headroom without forcing an unreachable dock.
    # Slow whole-body IK + ramped arm moves: a 60 s receipt TTL expires mid-drawer/flip.
    esim = dataclasses.replace(cfg.esim, edge_grasp_first=True, place_edge_margin=0.05, flip_in_air=True,
                               handle_reach=0.25, capture_radius=0.12,
                               handle_probes_side_m=(0.06, -0.06, 0.10, -0.10, 0.14, -0.14),
                               handle_probes_along_m=(0.04, 0.08, 0.12, -0.05, -0.09, 0.16),
                               elevated_grasp_floor=floors["elevated_grasp_floor"],
                               drawer_entry_lift=floors["working_lift"],
                               tilted_carry_floor=floors["tilted_carry_floor"],
                               min_require_tilt=0.25, receipt_ttl=300.0,
                               # Flip-in-air drops from turnLift; allow a wide shift so the lid already
                               # on the table does not exhaust free spots around the requested offset.
                               # Flip-in-air releases from ~turnLift; need clearance past neighbours
                               # and room to shift off a crowded requested offset onto free table.
                               # Soft-exit unrolls then leaves; keep clearance modest so free spots
                               # still exist on a crowded puzzle table after the lid is set down.
                               place_clearance=0.18, spot_shift_max=0.85,
                               # USD tip is slower to track than the kitchen slide; hold longer so
                               # rim-tilted pours actually empty before the wrist rights the box.
                               pour_hold_s=3.0, pour_roll=2.3)
    return dataclasses.replace(cfg, motion=motion, esim=esim)


@dataclasses.dataclass(frozen=True)
class BodySetup:
    """A model, controller and runtime settings resolved together for one embodiment."""

    model: dict
    arm: Arm | None
    config: RuntimeConfig


def prepare_body(body: str, base: RuntimeConfig | None = None) -> BodySetup:
    """Resolve a body once; USD geometry, controller and policy share one profile snapshot."""
    if body == "agilex-usd":
        from steward.ego.agilex import prepare_agilex

        model, arm, config = prepare_agilex(base or RuntimeConfig())
        return BodySetup(model, arm, config)
    model = robot_model(body)
    return BodySetup(model, body_arm(body, model), runtime_config(body, base))
