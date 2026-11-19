"""Planar odometry/visual-pose EKF with wrapped innovations and conservative observation noise.

Covariances are an engineering approximation: a self-built map and odometry-initialized matches
are correlated. Floors, capped effective point counts and observation spacing limit confidence;
they do not replace a joint pose/map covariance model.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EkfConfig:
    """Standard deviations use metres/radians; motion coefficients scale with square-root travel."""

    enabled: bool = True
    initial_position_sigma_m: float = 0.005
    initial_yaw_sigma_rad: float = 0.004
    odometry_position_sigma_m: float = 0.003
    odometry_yaw_sigma_rad: float = 0.003
    position_sigma_per_sqrt_m: float = 0.025
    yaw_sigma_per_sqrt_m: float = 0.015
    position_sigma_per_sqrt_rad: float = 0.001
    yaw_sigma_per_sqrt_rad: float = 0.001
    position_sigma_per_sqrt_s: float = 0.0001
    yaw_sigma_per_sqrt_s: float = 0.0001
    motion_noise_interval_s: float = 0.5
    motion_position_deadband_m: float = 0.01
    motion_yaw_deadband_rad: float = 0.01
    visual_position_floor_m: float = 0.02
    visual_yaw_floor_rad: float = 0.015
    map_position_floor_m: float = 0.01
    map_yaw_floor_rad: float = 0.005
    visual_covariance_inflation: float = 2.0
    visual_effective_points: int = 20
    max_visual_position_sigma_m: float = 0.20
    max_visual_yaw_sigma_rad: float = 0.20
    innovation_chi2_limit: float = 11.345
    min_visual_interval_s: float = 0.5

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ekf.enabled must be boolean")
        for key, value in asdict(self).items():
            if key == "enabled":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"ekf.{key} must be finite and positive")
        if type(self.visual_effective_points) is not int:
            raise ValueError("ekf.visual_effective_points must be an integer")
        if (self.initial_position_sigma_m < self.odometry_position_sigma_m
                or self.initial_yaw_sigma_rad < self.odometry_yaw_sigma_rad):
            raise ValueError("initial EKF uncertainty must include odometry sampling noise")
        if self.visual_covariance_inflation < 1:
            raise ValueError("visual covariance inflation must be at least one")
        if (self.max_visual_position_sigma_m < math.hypot(self.visual_position_floor_m, self.map_position_floor_m)
                or self.max_visual_yaw_sigma_rad < math.hypot(self.visual_yaw_floor_rad, self.map_yaw_floor_rad)):
            raise ValueError("visual uncertainty limits must include their noise floors")


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class PlanarPoseEKF:
    """Single-owner three-state EKF; invalid or rejected measurements never mutate the posterior."""

    def __init__(self, config: EkfConfig):
        self.config = config
        self.pose = None
        self.covariance = None
        self.raw = None
        self.stamp = None
        self.noise_anchor = None
        self.noise_stamp = None
        self.last_visual_stamp = None

    def _floor_covariance(self, covariance):
        cfg = self.config
        scale = np.diag([cfg.odometry_position_sigma_m, cfg.odometry_position_sigma_m, cfg.odometry_yaw_sigma_rad])
        normalized = np.linalg.solve(scale, np.linalg.solve(scale, covariance).T).T
        values, vectors = np.linalg.eigh((normalized + normalized.T) / 2)
        return scale @ ((vectors * np.maximum(values, 1.0)) @ vectors.T) @ scale

    def predict(self, raw_pose: np.ndarray, stamp: float) -> np.ndarray:
        """Compose body-relative odometry increments and propagate pose covariance.

        Motion noise is measured over coarse intervals, rather than counting absolute sensor
        jitter as independent travel on every read. Time noise and Jacobians update every read.
        """
        raw = np.asarray(raw_pose, float)
        if raw.shape != (3,) or not np.isfinite(raw).all() or not math.isfinite(stamp):
            raise ValueError("EKF odometry must be a finite pose and timestamp")
        cfg = self.config
        if self.pose is None:
            self.pose = raw.copy()
            self.pose[2] = _wrap(self.pose[2])
            self.covariance = np.diag([cfg.initial_position_sigma_m**2]*2 + [cfg.initial_yaw_sigma_rad**2])
            self.raw, self.noise_anchor = raw.copy(), raw.copy()
            self.stamp = self.noise_stamp = stamp
            return self.pose.copy()
        if stamp <= self.stamp:
            raise ValueError("EKF timestamps must strictly increase")
        c, s = math.cos(self.raw[2]), math.sin(self.raw[2])
        delta = raw[:2] - self.raw[:2]
        body_delta = np.array([[c, s], [-s, c]]) @ delta
        theta = self.pose[2]
        c, s = math.cos(theta), math.sin(theta)
        move = np.array([[c, -s], [s, c]]) @ body_delta
        jacobian = np.eye(3)
        jacobian[0, 2], jacobian[1, 2] = -move[1], move[0]
        pose = self.pose + np.r_[move, _wrap(raw[2] - self.raw[2])]
        pose[2] = _wrap(pose[2])
        dt = stamp - self.stamp
        position_noise = cfg.position_sigma_per_sqrt_s**2 * dt
        yaw_noise = cfg.yaw_sigma_per_sqrt_s**2 * dt
        noise_anchor, noise_stamp = self.noise_anchor, self.noise_stamp
        if stamp - self.noise_stamp >= cfg.motion_noise_interval_s - 1e-9:
            distance = max(0.0, float(np.linalg.norm(raw[:2] - self.noise_anchor[:2])) - cfg.motion_position_deadband_m)
            angle = max(0.0, abs(float(_wrap(raw[2] - self.noise_anchor[2]))) - cfg.motion_yaw_deadband_rad)
            position_noise += cfg.position_sigma_per_sqrt_m**2 * distance + cfg.position_sigma_per_sqrt_rad**2 * angle
            yaw_noise += cfg.yaw_sigma_per_sqrt_m**2 * distance + cfg.yaw_sigma_per_sqrt_rad**2 * angle
            noise_anchor, noise_stamp = raw.copy(), stamp
        covariance = jacobian @ self.covariance @ jacobian.T + np.diag([position_noise, position_noise, yaw_noise])
        self.pose, self.covariance = pose, self._floor_covariance(covariance)
        self.noise_anchor, self.noise_stamp = noise_anchor, noise_stamp
        self.raw, self.stamp = raw.copy(), stamp
        return pose.copy()

    def correct(self, measurement: np.ndarray, covariance: np.ndarray) -> tuple[bool, str, dict]:
        """Gate a visual pose using its NIS, then update with Joseph-form covariance.

        Measurement covariance is map-frame metres/radians, symmetric positive definite.
        A rejection leaves pose, covariance and the previous visual update timestamp unchanged.
        """
        if self.pose is None:
            raise ValueError("predict before correcting")
        observed, noise = np.asarray(measurement, float), np.asarray(covariance, float)
        if (observed.shape != (3,) or noise.shape != (3, 3) or not np.isfinite(observed).all()
                or not np.isfinite(noise).all() or not np.allclose(noise, noise.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(noise).min() <= 0):
            raise ValueError("visual pose must be finite and covariance symmetric positive definite")
        innovation = observed - self.pose
        innovation[2] = _wrap(innovation[2])
        combined = self.covariance + noise
        nis = float(innovation @ np.linalg.solve(combined, innovation))
        metrics = {"innovation": innovation.tolist(), "innovationNis": nis,
                   "visualCovariance": noise.tolist(), "predictedCovariance": self.covariance.tolist()}
        cfg = self.config
        if (np.sqrt(np.linalg.eigvalsh(noise[:2, :2]).max()) > cfg.max_visual_position_sigma_m
                or math.sqrt(noise[2, 2]) > cfg.max_visual_yaw_sigma_rad):
            return False, "visual_uncertainty", metrics
        if nis > cfg.innovation_chi2_limit:
            return False, "innovation_outlier", metrics
        if self.last_visual_stamp is not None and self.stamp - self.last_visual_stamp < cfg.min_visual_interval_s - 1e-9:
            return False, "correlated_observation", metrics
        gain = np.linalg.solve(combined, self.covariance.T).T
        pose = self.pose + gain @ innovation
        pose[2] = _wrap(pose[2])
        remaining = np.eye(3) - gain
        posterior = remaining @ self.covariance @ remaining.T + gain @ noise @ gain.T
        self.pose, self.covariance = pose, self._floor_covariance(posterior)
        self.last_visual_stamp = self.stamp
        metrics.update(kalmanGain=gain.tolist(), fusedDelta=(gain @ innovation).tolist())
        return True, "ekf_fused", metrics
