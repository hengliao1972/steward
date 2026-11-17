"""Synchronous encoder replays for empty-arm homing and skill completion; no running world."""

import json
from pathlib import Path

import pytest

from steward.assets.cobot import cobot_magic
from steward.ego.piper_arm import PiperArm, PiperArmConfig
from steward.ego.skills import Job, SkillExecutor, SkillFailed, SkillStopped


class Feedback:
    def __init__(self, arm):
        self.arm = arm
        self.joints = {"lift": 0.24, "L_joint7": 0.05, "R_joint7": 0.05}
        for side in ("L", "R"):
            self.joints.update(dict(zip(arm.ik.joint_names(side), [0.2, 0.6, -0.5, 0.2, 0.1, -0.3], strict=True)))
        self.grippers = {f"{side}_link6": {"holding": False, "holdingHandle": False} for side in ("L", "R")}
        self.stamp = 0.0
        self.commands = []
        self.follow = True

    def tick(self):
        self.stamp += 0.04
        return self.arm.virtualize({"stamp": self.stamp, "joints": dict(self.joints),
                                    "grippers": self.grippers})

    def command(self, command):
        self.commands.append((self.stamp, command))
        if self.follow:
            self.joints.update(command["joints"])


def test_both_empty_arms_return_to_zero_before_the_place_job_succeeds(monkeypatch):
    arm = PiperArm(cobot_magic().to_dict())
    feedback = Feedback(arm)
    initial = dict(feedback.joints)
    arm._target = arm.joints_of(initial)
    arm._solved = (initial["lift"], [initial[n] for n in arm.names])
    executor = SkillExecutor(lambda cmd: None, feedback.tick, lambda eid: eid,
                             home_empty=lambda tick: arm.home_empty(feedback.command, tick))
    job = Job("job-1", "place", "receipt-1")

    def tick():
        assert job.state == "running"
        return feedback.tick()

    def place(job, plan):
        job.log.append("released cup on table")
        job.message = "placed cup on table"

    monkeypatch.setattr(executor, "_tick", tick)
    monkeypatch.setattr(executor, "_place", place)
    executor._run(job, "place", {"plan": {}})
    names = arm.ik.joint_names("L") + arm.ik.joint_names("R")
    result = {"state": job.state, "message": job.message, "log": job.log,
              "armJoints": {n: feedback.joints[n] for n in names}}
    expected = json.loads((Path(__file__).parent / "expected/cobot-homing.json").read_text())
    assert result == expected
    assert feedback.joints["lift"] == 0.24 and feedback.joints["L_joint7"] == feedback.joints["R_joint7"] == 0.05
    assert arm._target is None and arm._solved is None and arm.resting()
    for (_, a), (_, b) in zip(feedback.commands, feedback.commands[1:]):
        assert all(abs(b["joints"][n] - a["joints"][n]) <= arm.config.home_speed_rad_s * 0.04 + 1e-9 for n in names)
        assert set(b) == {"joints"} and set(b["joints"]) == set(names)
    # A subsequent Cartesian command must start from the returned encoder posture.
    arm.virtualize({"joints": feedback.joints, "grippers": feedback.grippers})
    assert arm.command({"lift": 0.85, "reach": 0.30, "wrist_yaw": 0.0, "wrist_roll": 0.0})["joints"]


@pytest.mark.parametrize("side,attachment", [("L", "holding"), ("R", "holding"), ("R", "holdingHandle")])
def test_either_occupied_gripper_prevents_both_arms_from_homing(side, attachment):
    arm = PiperArm(cobot_magic().to_dict())
    feedback = Feedback(arm)
    feedback.grippers[f"{side}_link6"][attachment] = True
    target = arm.joints_of(feedback.joints)
    arm._target = dict(target)
    assert not arm.home_empty(feedback.command, feedback.tick)
    assert not feedback.commands and arm._target == target


def test_missing_gripper_feedback_does_not_mean_empty():
    arm = PiperArm(cobot_magic().to_dict())
    feedback = Feedback(arm)
    del feedback.grippers["R_link6"]
    with pytest.raises(SkillFailed, match="feedback"):
        arm.home_empty(feedback.command, feedback.tick)
    assert not feedback.commands


def test_cancellation_during_homing_sends_no_further_commands():
    arm = PiperArm(cobot_magic().to_dict())
    feedback = Feedback(arm)

    def tick():
        if feedback.commands:
            raise SkillStopped()
        return feedback.tick()

    with pytest.raises(SkillStopped):
        arm.home_empty(feedback.command, tick)
    assert len(feedback.commands) == 1
    assert arm._target is None and arm._solved is None


def test_a_new_attachment_stops_homing_at_the_measured_joint_positions():
    arm = PiperArm(cobot_magic().to_dict())
    feedback = Feedback(arm)

    def tick():
        if len(feedback.commands) == 2:
            feedback.grippers["R_link6"]["holding"] = True
        return feedback.tick()

    with pytest.raises(SkillFailed, match="became occupied"):
        arm.home_empty(feedback.command, tick)
    assert len(feedback.commands) == 3
    assert feedback.joints["R_joint2"] > 0
    assert feedback.commands[-1][1] == feedback.commands[-2][1]


def test_stalled_homing_fails_without_cartesian_recovery(monkeypatch):
    arm = PiperArm(cobot_magic().to_dict(), PiperArmConfig(home_settle_s=0.1))
    feedback = Feedback(arm)
    feedback.follow = False
    executor = SkillExecutor(lambda cmd: None, feedback.tick, lambda eid: eid,
                             home_empty=lambda tick: arm.home_empty(feedback.command, tick))
    monkeypatch.setattr(executor, "_tick", feedback.tick)
    monkeypatch.setattr(executor, "_place", lambda job, plan: setattr(job, "message", "placed cup on table"))
    monkeypatch.setattr(executor, "_safe_retract", lambda: pytest.fail("no IK recovery after joint-space homing"))
    job = Job("job-1", "place", "receipt-1")
    executor._run(job, "place", {"plan": {}})
    assert job.state == "failed" and "placed cup on table; homing failed" in job.message
    assert "returned to zero" not in " ".join(job.log)
    assert feedback.commands[-1][1]["joints"]["L_joint2"] == feedback.joints["L_joint2"]


def test_failed_skill_does_not_start_homing(monkeypatch):
    executor = SkillExecutor(lambda cmd: None, lambda: {"stamp": 1.0}, lambda eid: eid,
                             home_empty=lambda tick: pytest.fail("failed work must not trigger homing"))

    def failed(job, plan):
        raise SkillFailed("place blocked")

    monkeypatch.setattr(executor, "_place", failed)
    monkeypatch.setattr(executor, "_safe_retract", lambda: None)
    job = Job("job-1", "place", "receipt-1")
    executor._run(job, "place", {"plan": {}})
    assert job.state == "failed" and job.message == "place blocked"
