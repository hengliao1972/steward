"""Odometry covariance, wrapped visual innovations, outlier containment and fused map insertion."""

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from steward.ego.distance_localization import DistanceFieldLocalizer, LocalizationConfig
from steward.ego.pose_ekf import EkfConfig, PlanarPoseEKF
from test_distance_localization import room_points


NOISE = np.diag([0.02**2, 0.02**2, 0.015**2])


def test_visual_pose_is_weighted_and_posterior_is_symmetric_positive_definite():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    prior = tracker.covariance.copy()
    measurement = np.array([0.01, -0.005, 0.005])
    gain = np.linalg.solve(prior + NOISE, prior.T).T
    accepted, reason, metrics = tracker.correct(measurement, NOISE)
    assert accepted and reason == "ekf_fused"
    np.testing.assert_allclose(tracker.pose, gain @ measurement, atol=1e-12)
    assert 0 < tracker.pose[0] < measurement[0]
    assert metrics["innovationNis"] < tracker.config.innovation_chi2_limit
    np.testing.assert_allclose(tracker.covariance, tracker.covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(tracker.covariance).min() > 0


def test_stationary_eighteen_centimetre_visual_jump_is_rejected_without_mutation():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    before = tracker.covariance.copy()
    accepted, reason, metrics = tracker.correct([0.18, 0, 0], NOISE)
    assert not accepted and reason == "innovation_outlier"
    assert metrics["innovationNis"] > 70
    np.testing.assert_array_equal(tracker.pose, np.zeros(3))
    np.testing.assert_array_equal(tracker.covariance, before)
    assert tracker.last_visual_stamp is None


def test_heading_innovation_crosses_pi_by_the_short_angle():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.array([0, 0, math.radians(179.8)]), 0)
    accepted, _, metrics = tracker.correct([0, 0, math.radians(-179.9)], NOISE)
    assert accepted
    assert math.degrees(metrics["innovation"][2]) == pytest.approx(0.3)
    assert abs(tracker.pose[2]) <= math.pi


def test_close_repeated_observations_do_not_keep_shrinking_covariance():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    assert tracker.correct([0.005, 0, 0], NOISE)[0]
    tracker.predict(np.zeros(3), 0.1)
    pose, covariance = tracker.pose.copy(), tracker.covariance.copy()
    accepted, reason, _ = tracker.correct([0.005, 0, 0], NOISE)
    assert not accepted and reason == "correlated_observation"
    np.testing.assert_array_equal(tracker.pose, pose)
    np.testing.assert_array_equal(tracker.covariance, covariance)
    assert tracker.last_visual_stamp == 0


def test_large_visual_uncertainty_and_invalid_covariance_are_contained():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    assert tracker.correct(np.zeros(3), np.eye(3))[1] == "visual_uncertainty"
    before = tracker.covariance.copy()
    for invalid in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])):
        with pytest.raises(ValueError, match="positive definite"):
            tracker.correct(np.zeros(3), invalid)
    np.testing.assert_array_equal(tracker.covariance, before)


def test_frequent_stationary_sensor_jitter_does_not_become_metres_of_motion_noise():
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    initial = tracker.covariance.copy()
    for index in range(1, 201):
        raw = np.array([0.002*math.sin(index), 0.002*math.cos(index), 0.002*math.sin(index)])
        tracker.predict(raw, index*0.05)
    assert np.diag(tracker.covariance).max() < np.diag(initial).max() + 1e-6
    assert np.linalg.eigvalsh(tracker.covariance).min() > 0
    moving = PlanarPoseEKF(EkfConfig())
    moving.predict(np.zeros(3), 0)
    moving.predict(np.array([2, 0, 0]), 1)
    assert moving.covariance[0, 0] > tracker.covariance[0, 0]*20


def test_prediction_rotates_body_motion_by_fused_heading():
    tracker = PlanarPoseEKF(EkfConfig(initial_yaw_sigma_rad=0.1))
    tracker.predict(np.zeros(3), 0)
    assert tracker.correct([0, 0, 0.02], NOISE)[0]
    angle = tracker.pose[2]
    result = tracker.predict(np.array([1, 0, 0]), 1)
    np.testing.assert_allclose(result[:2], [math.cos(angle), math.sin(angle)], atol=1e-12)
    assert tracker.covariance[1, 2] > 0


def test_prediction_math_failure_preserves_the_previous_valid_state(monkeypatch):
    tracker = PlanarPoseEKF(EkfConfig())
    tracker.predict(np.zeros(3), 0)
    pose, covariance, anchor = tracker.pose.copy(), tracker.covariance.copy(), tracker.noise_anchor.copy()
    def fail(_covariance):
        raise np.linalg.LinAlgError("test decomposition failure")
    monkeypatch.setattr(tracker, "_floor_covariance", fail)
    with pytest.raises(np.linalg.LinAlgError):
        tracker.predict(np.array([1, 0, 0]), 1)
    np.testing.assert_array_equal(tracker.pose, pose)
    np.testing.assert_array_equal(tracker.covariance, covariance)
    np.testing.assert_array_equal(tracker.noise_anchor, anchor)
    assert tracker.stamp == tracker.noise_stamp == 0


def test_only_fused_pose_inserts_points_and_rejected_outlier_cannot_pollute_map(monkeypatch):
    tracker = DistanceFieldLocalizer(LocalizationConfig(insert_translation_m=0.0001))
    tracker.update([0, 0, 0], 0, 1, room_points())
    candidate = np.array([0.03, 0, 0.01])
    monkeypatch.setattr(tracker, "_match", lambda *args: (candidate.copy(), True, "matched", {"visualCovariance": NOISE.tolist()}))
    inserted = []
    original_insert = tracker._insert
    def insert(points, bands, pose, wall_normals=None, *, camera_origin=None):
        inserted.append(pose.copy())
        return original_insert(points, bands, pose, wall_normals, camera_origin=camera_origin)
    monkeypatch.setattr(tracker, "_insert", insert)
    result = tracker.update([0, 0, 0], 1, 1, room_points())
    assert result["accepted"] and result["reason"] == "ekf_fused"
    assert len(inserted) == 1
    np.testing.assert_allclose(inserted[0], result["pose"], atol=1e-12)
    assert result["pose"][0] < 0.01 and not np.allclose(inserted[0], candidate)
    before, correction, version = tracker.occupied.copy(), tracker.correction.copy(), tracker.map_version
    candidate[:] = [0.18, 0, 0]
    rejected = tracker.update([0, 0, 0], 2, 1, room_points())
    assert not rejected["accepted"] and rejected["reason"] == "innovation_outlier"
    np.testing.assert_array_equal(before, tracker.occupied)
    np.testing.assert_array_equal(correction, tracker.correction)
    assert tracker.map_version == version
    tracker.update([0, 0, 0], 0, 2)
    assert tracker.filter.last_visual_stamp is None and tracker.map_version == 0


@pytest.mark.parametrize("values", [{"enabled": "yes"}, {"innovation_chi2_limit": 0},
                                      {"visual_effective_points": 1.5}, {"initial_position_sigma_m": 0.001},
                                      {"visual_covariance_inflation": 0.5}, {"max_visual_position_sigma_m": 0.001},
                                      {"max_visual_yaw_sigma_rad": 0.001}])
def test_invalid_ekf_settings_fail_at_configuration(values):
    with pytest.raises(ValueError):
        EkfConfig(**values)
