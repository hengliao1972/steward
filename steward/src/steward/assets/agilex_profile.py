"""Resolve the USD body's simulation and task settings from its own TOML profile.

External profiles override the shipped settings, including individual table members. Unknown
keys, non-finite numbers and invalid ranges fail before a model or controller is constructed.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import tomllib

PROFILE = Path(__file__).resolve().parents[3] / "config/agilex-usd.toml"


# Types and array dimensions are fixed even when the shipped TOML is edited directly.
_CAMERA = {"size": (int, 2), "fovy_deg": float, "position": (float, 3), "rate_hz": float}
_SCHEMA = {
    "template": str,
    **dict.fromkeys(("lift_home", "arm_kp", "arm_damping", "arm_effort", "lift_kp", "lift_damping",
                     "lift_effort", "finger_kp", "finger_damping", "finger_effort",
                     "survey_pitch_deg", "work_pitch_deg"), float),
    "base": dict.fromkeys(("kp", "damping", "effort", "velocity"), (float, 3)),
    "cameras": {"head": _CAMERA, "wide": _CAMERA},
    "arm": {
        "arm": str, "deploy_joints": (float, 6), "reach": (float, 2), "small_step": (float, 2),
        "continuation_steps": int,
        **dict.fromkeys(("shoulder_above", "shoulder_drop_per_pitch", "pitch_tolerance", "pitch_margin",
                         "torso_seed_inset", "continuation_reach", "lift_below_mount", "lift_above_mount",
                         "wrist_yaw_max", "wrist_roll_max", "wrist_pitch_max", "pitch_unseen_cos",
                         "pitch_read_slack", "roll_about_x_pitch", "roll_about_x_max", "drive_tip_roll_floor",
                         "max_joint_step", "max_swing_m", "max_turn_swing_m", "turn_lower_step",
                         "turn_link_clearance", "self_clearance", "arrived_rad",
                         "joint_limit_margin", "home_speed_rad_s", "home_settle_s"), float),
    },
    "motion": dict.fromkeys(("linear_speed", "angular_speed", "look_angular_speed", "gentle_speed",
                             "gentle_turn_speed", "retract_reach", "reach_max", "align_max", "joint_tol",
                             "arm_speed", "settle_s", "grasp_wait_s", "stow_lift",
                             "release_joint_tol", "release_settle_s"), float),
    "trajectory": {
        "handle_lower_reaches": (float, 2), "handle_reach_cap": float, "handle_lower_clearance": float,
        "intermediate_reaches": (float, None), "reach_attempts": int, "unfurl_attempts": int,
        **dict.fromkeys(("approach_floor", "base_adjustment", "redock_reach", "recovery_lift_max"), float),
    },
    "planning": {
        "edge_grasp_first": bool, "flip_in_air": bool,
        "handle_probes_side_m": (float, None), "handle_probes_along_m": (float, None),
        **dict.fromkeys(("place_edge_margin", "handle_reach", "capture_radius", "min_require_tilt",
                         "receipt_ttl", "place_clearance", "spot_shift_max", "pour_hold_s",
                         "pour_roll", "pour_roll_min", "drawer_entry_pitch", "drawer_grasp_raise",
                         "drawer_grasp_backoff", "drawer_entry_backoff"), float),
    },
}


def load_agilex_profile(path: Path | None = None) -> dict:
    """Return validated settings with an absolute template path.

    ``path`` selects a profile explicitly; otherwise STEWARD_AGILEX_USD_CONFIG selects it, falling
    back to the shipped profile. An overridden template is relative to the file declaring it.
    Missing fields use shipped USD defaults. Legacy arm_effort also sets finger_effort when the
    latter is omitted; an explicit finger_effort makes the two independent.
    """
    selected = path or Path(os.environ.get("STEWARD_AGILEX_USD_CONFIG", str(PROFILE)))
    defaults = _read(PROFILE)
    _check(_SCHEMA, defaults, PROFILE, "profile")
    supplied = _read(selected)
    _check(_SCHEMA, supplied, selected, "profile", partial=True)
    cfg = _merge(defaults, supplied)
    # Flat profiles assigned the arm effort to fingers unless finger_effort was explicit.
    if "arm_effort" in supplied and "finger_effort" not in supplied:
        cfg["finger_effort"] = supplied["arm_effort"]
    _validate(cfg, selected)
    owner = selected if "template" in supplied else PROFILE
    cfg["template"] = str((owner.parent / cfg["template"]).resolve())
    return cfg


def _read(path: Path) -> dict:
    text = path.read_text()
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid USD profile TOML in {path}: {exc}") from exc


def _merge(defaults: dict, supplied: dict) -> dict:
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        result[key] = _merge(defaults[key], value) if isinstance(value, dict) else value
    return result


def _check(expected, value, path: Path, name: str, *, partial: bool = False) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ValueError(f"Invalid {name} in {path}: expected a table")
        unknown = value.keys() - expected.keys()
        if unknown:
            raise ValueError(f"Unknown USD profile field in {path}: {name}.{sorted(unknown)[0]}")
        missing = expected.keys() - value.keys()
        if missing and not partial:
            raise ValueError(f"Missing USD profile field in {path}: {name}.{sorted(missing)[0]}")
        for key, item in value.items():
            _check(expected[key], item, path, name + "." + key, partial=partial)
        return
    if isinstance(expected, tuple):
        item_type, length = expected
        if type(value) is not list or not value or (length is not None and len(value) != length):
            raise ValueError(f"Invalid {name} in {path}: expected {length or 'one or more'} array members")
        for index, item in enumerate(value):
            _check(item_type, item, path, f"{name}[{index}]")
        return
    valid = (type(value) in (int, float) and math.isfinite(value)) if expected is float else type(value) is expected
    if not valid:
        raise ValueError(f"Invalid {name} in {path}: expected a finite number" if expected is float
                         else f"Invalid {name} in {path}: expected {expected.__name__}")


def _validate(cfg: dict, path: Path) -> None:
    def require(ok: bool, name: str, rule: str) -> None:
        if not ok:
            raise ValueError(f"Invalid {name} in {path}: {rule}")

    require(bool(cfg["template"].strip()), "template", "expected a nonempty path")
    for key in ("arm_kp", "arm_damping", "arm_effort", "lift_kp", "lift_damping", "lift_effort",
                "finger_kp", "finger_damping", "finger_effort"):
        require(cfg[key] > 0, key, "expected a positive number")
    for key in ("survey_pitch_deg", "work_pitch_deg"):
        require(-90 <= cfg[key] <= 90, key, "expected degrees in [-90, 90]")
    for key, values in cfg["base"].items():
        require(all(v > 0 for v in values), "base." + key, "expected three positive x/y/yaw values")
    for camera, values in cfg["cameras"].items():
        require(all(v > 0 for v in values["size"]), f"cameras.{camera}.size", "expected positive width/height")
        require(0 < values["fovy_deg"] < 180, f"cameras.{camera}.fovy_deg", "expected degrees in (0, 180)")
        require(values["rate_hz"] > 0, f"cameras.{camera}.rate_hz", "expected a positive rate")
    arm = cfg["arm"]
    require(arm["arm"] in ("L", "R"), "arm.arm", "expected L or R")
    for key, value in arm.items():
        if type(value) in (int, float):
            require(value > 0, "arm." + key, "expected a positive number")
    require(0 <= arm["reach"][0] < arm["reach"][1], "arm.reach", "expected ordered nonnegative bounds")
    require(all(v > 0 for v in arm["small_step"]), "arm.small_step", "expected positive metres/radians")
    require(arm["pitch_unseen_cos"] <= 1, "arm.pitch_unseen_cos", "expected a cosine in (0, 1]")
    for section in ("motion", "planning", "trajectory"):
        for key, value in cfg[section].items():
            if type(value) in (int, float):
                require(value > 0, section + "." + key, "expected a positive number")
    planning = cfg['planning']
    require(planning['drawer_entry_pitch'] <= arm['wrist_pitch_max'],
            'planning.drawer_entry_pitch', 'exceeds arm.wrist_pitch_max')
    require(0.3 <= planning['pour_roll'] <= arm['wrist_roll_max'],
            'planning.pour_roll', 'expected radians in [0.3, arm.wrist_roll_max]')
    require(planning['pour_roll_min'] <= math.pi,
            'planning.pour_roll_min', 'expected container tilt in (0, pi] radians')
    motion, velocity = cfg["motion"], cfg["base"]["velocity"]
    for key in ("linear_speed", "gentle_speed"):
        require(motion[key] <= min(velocity[:2]), "motion." + key, "exceeds base x/y velocity limits")
    for key in ("angular_speed", "gentle_turn_speed", "look_angular_speed"):
        require(motion[key] <= velocity[2], "motion." + key, "exceeds base yaw velocity limit")
    require(arm["reach"][0] <= motion["retract_reach"] <= arm["reach"][1],
            "motion.retract_reach", "outside arm.reach")
    trajectory = cfg["trajectory"]
    reaches = trajectory["intermediate_reaches"]
    require(all(arm["reach"][0] <= r <= arm["reach"][1] for r in reaches)
            and all(a < b for a, b in zip(reaches, reaches[1:])),
            "trajectory.intermediate_reaches", "expected increasing reaches inside arm.reach")
    require(arm["reach"][0] <= trajectory["redock_reach"] <= arm["reach"][1],
            "trajectory.redock_reach", "outside arm.reach")
    require(trajectory["approach_floor"] <= trajectory["recovery_lift_max"],
            "trajectory.recovery_lift_max", "below trajectory.approach_floor")
    lo, hi = trajectory["handle_lower_reaches"]
    require(arm["reach"][0] <= lo < hi <= trajectory["handle_reach_cap"] <= arm["reach"][1],
            "trajectory.handle_lower_reaches", "expected ordered bounds below handle_reach_cap inside arm.reach")
