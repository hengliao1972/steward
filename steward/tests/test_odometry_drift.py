"""Accumulated calibration error, sampling invariance and episode reset."""

import math

import pytest

from steward.authoring.service import AuthoringService
from steward.world.odometry import OdometryTrack
from steward.world.sim import PerceptionConfig, WorldSim


def test_distance_and_heading_bias_accumulate_and_stopping_does_not_add_drift():
    track = OdometryTrack((0, 0, 0), (0, 0, 0))
    for i in range(1, 101):
        track.advance((i / 100, 0, 0), 0.01, 0.005)
    assert track.pose[2] == pytest.approx(0.005)
    assert track.pose[0] == pytest.approx(1.01 * math.sin(0.005) / 0.005, abs=1e-8)
    pose = track.pose
    for _ in range(20):
        track.advance((1, 0, 0), 0.01, 0.005)
    assert track.pose == pose


def test_sampling_does_not_advance_drift_and_reset_forgets_it(kitchen_root):
    sim = WorldSim(AuthoringService(kitchen_root).load_scene("phase0-kitchen"),
                   perception=PerceptionConfig(odom_scale_error=0.01, odom_yaw_drift_per_m=0.005))
    robot = "w:robot-1"
    try:
        sim.command(robot, {"base": [-1.6, -1.8, 0]})
        sim.step(1000)
        reading = sim.odometry(robot)
        for _ in range(10):
            assert sim.odometry(robot) == reading
            assert sim.sensors(robot)["odometry"] == reading
        before = sim._odometry_tracks[robot]
        assert math.dist(before.pose[:2], before.truth[:2]) > 0.001
        sim.reset()
        assert sim._odometry_tracks[robot].pose == sim._odometry_tracks[robot].truth
    finally:
        sim.stop()


@pytest.mark.parametrize("config", [{"odom_scale_error": -1}, {"odom_yaw_drift_per_m": math.nan}])
def test_invalid_calibration_is_rejected(config):
    with pytest.raises(ValueError):
        PerceptionConfig(**config)
