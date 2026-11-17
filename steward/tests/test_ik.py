"""Both Piper models at matched shoulder heights: FK/IK round trips, limits and per-arm isolation."""

import numpy as np
import pytest

from steward.ego.ik import CobotIK, IkError
from steward.ego.piper_ik import PiperIK
from steward.assets.agilex import agilex_usd
from test_agilex import imported_profile  # noqa: F401 - shared pytest fixture


@pytest.fixture(params=['cobot-magic', 'agilex-usd'])
def ik(request, imported_profile) -> PiperIK:
    solver = CobotIK() if request.param == 'cobot-magic' else PiperIK(agilex_usd(imported_profile))
    # Fixed-torso goals compare the same physical shoulder height; native home is tested separately.
    lift = next(j for j in solver.robot['body']['joints'] if j['id'] == 'lift')
    solver.set_base(lift=lift['limits']['value']['lo'])
    return solver


def test_arms_are_mirrored_across_the_mount_separation(ik: PiperIK) -> None:
    from steward.assets.cobot import HOME

    left, right = ik.fk("L", list(HOME)), ik.fk("R", list(HOME))
    # The same posture on either arm, so the arms differ only by the mount offset.
    assert left["p"][0] == pytest.approx(right["p"][0], abs=1e-6)
    assert left["p"][1] == pytest.approx(-right["p"][1], abs=0.01)
    assert left["p"][2] == pytest.approx(right["p"][2], abs=1e-6)


def test_fk_ik_round_trip_is_exact_at_a_reached_posture(ik: PiperIK) -> None:
    seed = ik.neutral("R")
    target = ik.fk("R", seed)
    solution = ik.solve("R", target["p"], target["q"], q_init=seed)
    got = ik.fk("R", solution)
    assert np.linalg.norm(np.subtract(got["p"], target["p"])) < 1e-3
    assert abs(float(np.dot(got["q"], target["q"]))) > 0.999


def test_ik_reaches_a_tabletop_point_and_stays_inside_the_urdf_limits(ik: PiperIK) -> None:
    goal = [0.45, 0.30, 0.78]
    solution = ik.solve("L", goal, q_init=ik.neutral("L"))
    assert len(solution) == 6
    assert np.linalg.norm(np.subtract(ik.fk("L", solution)["p"], goal)) < 5e-3
    for value, (lo, hi) in zip(solution, ik.joint_limits("L"), strict=True):
        assert lo <= value <= hi


def test_solving_one_arm_leaves_the_other_arm_untouched(ik: PiperIK) -> None:
    # Seed the left arm somewhere distinctive first, then solve the right arm.
    ik.solve("L", [0.45, 0.30, 0.78], q_init=ik.neutral("L"))
    left_before = ik.current("L")
    ik.solve("R", [0.45, -0.30, 0.78], q_init=ik.neutral("R"))
    assert ik.current("L") == pytest.approx(left_before)


def test_unreachable_goal_raises_with_the_distance_missed(ik: PiperIK) -> None:
    with pytest.raises(IkError, match="missed"):
        ik.solve("L", [2.0, 0.0, 1.0], q_init=ik.neutral("L"))
    assert not ik.reachable("L", [2.0, 0.0, 1.0], q_init=ik.neutral("L"))


def test_orientation_constrained_solve_points_the_tool_down(ik: PiperIK) -> None:
    # Tool pointing straight down: -z of the world expressed in the link's own frame.
    # These fixed-torso goals use the lowest carriage position of each body.
    down = [0.0, 0.7071068, 0.7071068, 0.0]  # wxyz: 180 deg about the (1, 1, 0) axis
    goal = [0.45, 0.30, 0.90]
    solution = ik.solve("L", goal, down, q_init=ik.neutral("L"))
    pose = ik.fk("L", solution)
    assert np.linalg.norm(np.subtract(pose["p"], goal)) < 5e-3
    assert abs(float(np.dot(pose["q"], down))) > 0.99


def test_a_measured_joint_past_its_limit_does_not_block_the_other_arm(ik: PiperIK) -> None:
    """An encoder on a hard stop reads a hair past the limit; that must not stop the other arm.

    mink's limit constraint covers the whole configuration, so seeding the measured value verbatim
    made the *other* arm's solve infeasible. Live, the right arm lost the blue bowl's grasp
    whenever the left arm finished a place with its wrist roll against the stop.
    """
    from steward.assets.cobot import HOME

    other = list(HOME)
    other[4] = ik.joint_limits("L")[4][1] + 2e-4  # joint5, a few ten-thousandths past its stop
    ik.set_joints("L", other)
    ik.set_joints("R", list(HOME))
    wrist, tool = _wrist_for(ik, 'R', [0.461, -0.278, 0.806], [0.0, 1.0, 0.0, 0.0])
    joints, _ = ik.solve_tool("R", wrist, tool, max_tilt=0.7)
    reached = _pinch_of(ik, "R", joints)
    assert np.linalg.norm(np.subtract(reached, [0.461, -0.278, 0.806])) < 0.015, reached


def test_bad_arguments_are_rejected(ik: PiperIK) -> None:
    with pytest.raises(KeyError):
        ik.fk("X")
    with pytest.raises(ValueError):
        ik.solve("L", [0.3, 0.2, 0.6], q_init=[0.0, 0.0])


def _wrist_for(ik, arm, pinch: list[float], tool: list[float]) -> tuple[list[float], list[float]]:
    from steward.math3d import pose_compose, pose_inverse

    wrist = pose_compose({"p": pinch, "q": tool},
                         pose_inverse({"p": ik.tcp_offsets[arm], "q": [1.0, 0.0, 0.0, 0.0]}))
    return wrist["p"], wrist["q"]


def _pinch_of(ik: PiperIK, arm: str, joints: list[float]) -> list[float]:
    from steward.math3d import pose_compose

    return pose_compose(ik.fk(arm, joints), {"p": ik.tcp_offsets[arm], "q": [1.0, 0.0, 0.0, 0.0]})["p"]


def test_tilted_tool_keeps_the_pinch_point_where_it_was_asked(ik: PiperIK) -> None:
    """A tilted tool must move the wrist, not the fingers.

    This is the far corner of the plate where the vertical tool runs into the wrist's roll limit:
    the solver has to tilt, and before it compensated, the fingers landed ~5 cm away from the
    target while every report still read "at the target" -- a place that missed.
    """
    from steward.assets.cobot import ARM_MOUNT_X

    pinch = [0.50 + ARM_MOUNT_X, 0.10, 0.82]  # 0.5 m ahead of the arm mount
    wrist, tool = _wrist_for(ik, 'L', pinch, [0.0, 1.0, 0.0, 0.0])  # tool pointing down
    joints, tilt = ik.solve_tool("L", wrist, tool, max_tilt=0.7, q_init=ik.neutral("L"))
    assert tilt > 0.0, "this pose is expected to need a tilt (otherwise the case is not exercised)"
    reached = _pinch_of(ik, "L", joints)
    assert np.linalg.norm(np.subtract(reached, pinch)) < 0.015, f"pinch landed at {reached}"


def test_solve_tool_does_not_tilt_when_the_exact_orientation_works(ik: PiperIK) -> None:
    pinch = [0.30, 0.20, 0.80]
    wrist, tool = _wrist_for(ik, 'L', pinch, [0.0, 1.0, 0.0, 0.0])
    joints, tilt = ik.solve_tool("L", wrist, tool, max_tilt=0.7, q_init=ik.neutral("L"))
    assert tilt == 0.0
    assert np.linalg.norm(np.subtract(_pinch_of(ik, "L", joints), pinch)) < 5e-3
