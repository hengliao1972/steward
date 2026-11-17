"""Cobot Magic dual-arm insert/drive skills and screw-board mission smoke tests."""

from __future__ import annotations

import time

import pytest

from steward.assets.cobot import ARM_MOUNT_X, cobot_magic
from steward.assets.generators import building, dining_table, electric_screwdriver, screw_board, toy_screw
from steward.ego.cobot_skills import CobotSkills, MotionConfig
from steward.world.sim import WorldSim

TABLE_TOP = 0.75
# Left-arm reachable strip ahead of the mounts (see test_cobot_skills).
SCREW_AT = [0.42 + ARM_MOUNT_X, 0.18, TABLE_TOP + 0.03]
DRIVER_AT = [0.42 + ARM_MOUNT_X, -0.18, TABLE_TOP + 0.02]
HOLE_AT = [0.45 + ARM_MOUNT_X, 0.05, TABLE_TOP + 0.04]


def _scene() -> dict:
    room = building(width=4.0, depth=4.0, height=2.6).to_dict()
    room["id"] = "w:room"
    robot = cobot_magic().to_dict()
    robot["id"] = "w:cobot"
    table = dining_table(width=1.2, depth=0.9, height=TABLE_TOP).to_dict()
    table["id"] = "w:table"
    table["pose"]["value"]["p"] = [0.92 + ARM_MOUNT_X, 0.0, 0.0]
    board = screw_board().to_dict()
    board["id"] = "w:board"
    board["pose"]["value"]["p"] = [0.45 + ARM_MOUNT_X, 0.05, TABLE_TOP]
    screw = toy_screw().to_dict()
    screw["id"] = "w:screw-0"
    screw["pose"]["value"]["p"] = list(SCREW_AT)
    driver = electric_screwdriver().to_dict()
    driver["id"] = "w:driver"
    driver["pose"]["value"]["p"] = list(DRIVER_AT)
    return {"mode": "world", "entities": [room, robot, table, board, screw, driver]}


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    sim.start()
    skills = CobotSkills(command=lambda cmd: sim.command("w:cobot", cmd),
                         state=lambda: sim.sensors("w:cobot"),
                         config=MotionConfig(tick_s=0.01))
    try:
        yield sim, skills
    finally:
        sim.stop()


def _run(skills, sim, skill, plan, timeout=90):
    job = skills.start(skill, {"id": skill, "plan": plan})
    deadline = time.monotonic() + timeout
    while job.state == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert job.state == "succeeded", f"{skill} failed: {job.message}\n" + "\n".join(job.log)
    return job


def test_piper_wrist_roll_stands_screwdriver_bit_down():
    """Near tool-down, drive wrist_roll tips a lateral bit onto world −z (not a flat spin).

    Full ±π/2 is often outside the Piper envelope; the arm softens toward a reachable tip that
    still stands the bit (world-up · tool+y < −0.7).
    """
    import math

    from steward.assets.cobot import HOME, LIFT_HOME, TCP_OFFSET, cobot_magic
    from steward.ego.piper_arm import PiperArm
    from steward.math3d import pose_compose, quat_rotate

    arm = PiperArm(cobot_magic().to_dict())
    arm._last = {"joints": {**{f"L_joint{i}": HOME[i - 1] for i in range(1, 7)},
                            **{f"R_joint{i}": HOME[i - 1] for i in range(1, 7)}, "lift": LIFT_HOME},
                 "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                 "grippers": {"L_link6": {"state": "open", "holding": None}}}
    arm.command({"lift": 0.90, "reach": 0.22, "wrist_yaw": -0.3, "wrist_roll": 0.0, "wrist_pitch": 0.0})
    torso, joints = arm._solved
    arm.ik.set_base(0.0, 0.0, 0.0, torso)
    tcp0 = pose_compose(arm.ik.fk("L", joints), {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})
    assert abs(quat_rotate(tcp0["q"], [0.0, 1.0, 0.0])[2]) < 0.2
    arm.command({"lift": 0.90, "reach": 0.22, "wrist_yaw": -0.3, "wrist_roll": -math.pi / 2,
                 "wrist_pitch": 0.0})
    assert abs(arm._target["wrist_roll"]) >= 1.05
    torso, joints = arm._solved
    arm.ik.set_base(0.0, 0.0, 0.0, torso)
    tcp1 = pose_compose(arm.ik.fk("L", joints), {"p": TCP_OFFSET, "q": [1.0, 0.0, 0.0, 0.0]})
    bit = quat_rotate(tcp1["q"], [0.0, 1.0, 0.0])
    assert bit[2] < -0.7, f"tool +y (lateral bit) should point down after roll, got {bit}"


def test_insert_and_drive_skills_accept_wrist_refine_plans(rig):
    sim, skills = rig
    _run(skills, sim, "pick", {"arm": "L", "at": list(SCREW_AT)})
    refine = [HOLE_AT[0] + 0.01, HOLE_AT[1] - 0.005, HOLE_AT[2]]
    job = _run(skills, sim, "insert_screw_into_hole", {
        "arm": "L", "at": list(HOLE_AT), "refine": True, "wristCamera": "L_wrist_camera",
        "refineAt": refine, "releaseLingerS": 0.1,
    })
    assert any("wrist refine" in line for line in job.log)
    assert any("tip-down" in line for line in job.log)
    # Tip-down insert must leave the screw upright (head up), not laid in the hole.
    from steward.math3d import quat_rotate
    screw = next(e for e in sim.world_state()["entities"] if e["id"] == "w:screw-0")
    up = quat_rotate(screw["pose"]["value"]["q"], [0.0, 0.0, 1.0])[2]
    assert up > 0.7, f"screw not tip-down upright after insert (headUp={up:.2f})"

    _run(skills, sim, "pick", {"arm": "R", "at": list(DRIVER_AT)})
    # Keep the bit-down hover low: a high tip-down pose near the midline exceeds R-arm reach.
    head = [refine[0], refine[1], HOLE_AT[2] + 0.03]
    press = [head[0], head[1], HOLE_AT[2] + 0.008]
    _run(skills, sim, "drive_screw", {
        "arm": "R", "at": head, "pressAt": press, "driveS": 0.15,
        "refine": True, "wristCamera": "R_wrist_camera", "refineAt": head,
    })
    assert sim.sensors("w:cobot")["grippers"]["R_link6"]["state"] == "closed"


def test_cobot_screw_board_spec_lists_dual_arm_parts():
    from steward.authoring.service import cobot_screw_board_spec

    ids = {e["id"] for e in cobot_screw_board_spec()["entities"]}
    assert {"w:cobot-1", "w:board", "w:screwdriver", "w:screw-0", "w:screw-3"} <= ids


def test_free_screws_are_beside_board_not_above_it():
    """Table screws sit below the board top; free = outside the board footprint."""
    import numpy as np
    from steward.ego.cobot_mission import Sighting, _free_screws

    board = Sighting("b", "screw_board", np.array([0.53, 0.0, 0.76]), np.array([0.28, 0.20, 0.018]),
                     1.0, 1.0)
    free_screw = Sighting("s0", "toy_screw", np.array([0.47, 0.27, 0.75]), np.array([0.05, 0.05, 0.066]),
                          1.0, 1.0)
    seated = Sighting("s1", "toy_screw", np.array([0.60, 0.05, 0.76]), np.array([0.05, 0.05, 0.066]),
                      1.0, 1.0)
    free = _free_screws(board, [free_screw, seated], driven=[])
    assert [s.track_id for s in free] == ["s0"]
    assert _free_screws(board, [free_screw], driven=["s0"]) == []
