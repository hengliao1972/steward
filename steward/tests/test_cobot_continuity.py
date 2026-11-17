"""Offline IK regressions for a folded wrist branch; no simulation stepping or services."""

import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.piper_arm import PiperArm
from steward.ego.skills import SkillFailed


def test_reachable_place_endpoint_is_rejected_from_the_folded_branch():
    arm = PiperArm(cobot_magic().to_dict())
    # Encoder posture recorded after the phase0-chores place failure.
    folded = (0.3065262769995, [0.087215417596, 3.141445869797747, -1.3177291913573286,
                               1.1118273317041372, -0.10483940876783834, 2.92840361173118])
    target = {"lift": 0.8148570192568708, "reach": 0.273992621809232, "wrist_yaw": -2.374488769806783,
              "wrist_roll": 0.0, "wrist_pitch": 0.14921875}
    start = {"lift": folded[0], **dict(zip(arm.names, folded[1], strict=True))}
    current = arm.joints_of(start)
    assert arm.reachable(**target)
    path = [{k: current[k] + (target[k] - current[k]) * i / 12 for k in current} for i in range(13)]
    assert not arm.reachable_path(path, start_joints=start)
    assert arm._target is None and arm._solved is None, "prediction must not command a posture"

    arm._last = {"joints": start, "grippers": {arm.geometry.gripper: {"holding": "cup"}}}
    arm._target, arm._solved = current, folded
    with pytest.raises(SkillFailed):
        for step in path[1:]:
            before = arm._target.copy(), arm._solved
            arm.command(step)
    assert (arm._target, arm._solved) == before
    assert arm._last["grippers"][arm.geometry.gripper]["holding"] == "cup"

    solved, _ = arm._ik(target, {})
    start = {"lift": solved[0], **dict(zip(arm.names, solved[1], strict=True))}
    assert arm.reachable_path([target, {**target, "lift": target["lift"] + 0.01}], start_joints=start)


def test_continuous_search_tries_another_seed_after_a_branch_jump(monkeypatch):
    arm = PiperArm(cobot_magic().to_dict())
    target = {"lift": 0.85, "reach": 0.3, "wrist_yaw": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0}
    prior, _ = arm._ik(target, {})
    jumping = prior[0], [*prior[1][:-1], prior[1][-1] + 1.0]
    seeds = iter([(jumping, None), (prior, None)])
    monkeypatch.setattr(arm, "_solve_from", lambda target, seeds_: next(seeds))
    monkeypatch.setattr(arm, "_approached", lambda target: None)
    solved, error = arm._ik(target, {}, prior, continuous=True)
    assert solved == prior and error is None


def test_joint6_limits_do_not_wrap_across_plus_and_minus_pi():
    arm = PiperArm(cobot_magic().to_dict())
    lo, hi = arm.ik.joint_limits("L")[-1]
    assert arm._jumps((0.0, [0.0] * 5 + [lo]), (0.0, [0.0] * 5 + [hi]))
