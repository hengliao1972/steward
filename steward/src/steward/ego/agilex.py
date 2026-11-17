"""Runtime assembly for the imported AgileX USD, using shared Piper control and skills."""
from __future__ import annotations

import dataclasses

from steward.assets.agilex import build_agilex_model
from steward.assets.agilex_profile import load_agilex_profile
from steward.ego.collision_policy import PiperSelfCollisionPolicy
from steward.ego.motion_policy import MotionPolicyConfig, PiperMotionPolicy
from steward.ego.piper_arm import PiperArm, PiperArmConfig
from steward.ego.piper_workspace import piper_envelope
from steward.ego.runtime import RuntimeConfig


def configured_arm(model: dict, settings: dict | None = None) -> tuple[PiperArm, dict[str, float]]:
    """Create the USD controller and task floors from its model and resolved settings."""
    settings = load_agilex_profile() if settings is None else settings
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in settings["arm"].items()}
    config = PiperArmConfig(**values)
    obstacles = tuple(link["id"] for link in model["body"]["links"]
                      if not link["id"].startswith(("L_", "R_")))
    arm = PiperArm(model, config, collision_policy=PiperSelfCollisionPolicy(body_links=obstacles))
    floors = piper_envelope(arm, settings["motion"]["retract_reach"])
    arm.config = dataclasses.replace(config, working_lift=floors["working_lift"],
                                     working_reach=floors["working_reach"])
    trajectory = {k: tuple(v) if isinstance(v, list) else v for k, v in settings["trajectory"].items()}
    arm.motion_policy = PiperMotionPolicy(MotionPolicyConfig(**trajectory))
    return arm, floors


def prepare_agilex(base: RuntimeConfig) -> tuple[dict, PiperArm, RuntimeConfig]:
    """Resolve the real USD model, arm and task settings from one validated profile read."""
    settings = load_agilex_profile()
    model = build_agilex_model(settings)
    arm, floors = configured_arm(model, settings)
    motion_floors = {k: v for k, v in floors.items() if k != "elevated_grasp_floor"}
    motion = dataclasses.replace(base.motion, **settings["motion"], **motion_floors,
                                 wrist_pitch_max=arm.geometry.wrist_pitch_max)
    planning = {k: tuple(v) if isinstance(v, list) else v for k, v in settings["planning"].items()}
    esim = dataclasses.replace(base.esim, **planning, elevated_grasp_floor=floors["elevated_grasp_floor"],
                               drawer_entry_lift=floors["working_lift"],
                               tilted_carry_floor=floors["tilted_carry_floor"])
    return model, arm, dataclasses.replace(base, motion=motion, esim=esim)
