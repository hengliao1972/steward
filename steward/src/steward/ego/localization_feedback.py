"""Immutable map-from-odometry snapshots for perception and live control reads.

The map frame stays anchored at initialization. A new robot-pose correction does not move
stationary map objects; subsequent observations are fused in that same map frame. This does
not re-optimize historical object geometry or correct accumulated distortion retrospectively.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PoseSnapshot:
    """One episode's immutable correction and conservative scalar pose uncertainty."""

    episode: int | str
    stamp: float
    correction: tuple[float, float, float]
    position_sigma: float | None
    yaw_sigma: float | None


class LocalizationFeedback:
    """Publish under the runtime lock; project from any reader without acquiring that lock."""

    def __init__(self):
        self.snapshot: PoseSnapshot | None = None

    def reset(self):
        """Discard the previous episode's coordinate relation."""
        self.snapshot = None

    def publish(self, result: dict):
        """Publish an estimator result; it must carry the current raw-odometry/map relation."""
        covariance = result["filterCovariance"]
        position_sigma = yaw_sigma = None
        if covariance is not None:
            matrix = np.asarray(covariance)
            position_sigma = math.sqrt(float(np.linalg.eigvalsh(matrix[:2, :2]).max()))
            yaw_sigma = math.sqrt(float(matrix[2, 2]))
        self.snapshot = PoseSnapshot(result["episode"], result["stamp"], tuple(result["mapFromOdometry"]),
                                     position_sigma, yaw_sigma)

    def project(self, header: dict) -> dict:
        """Copy a same-episode raw header into the latest map frame without changing its timestamp.

        This is a coordinate projection, not a second EKF update. Yaw retains its unwrapped
        odometry turn count so position-servo commands do not unexpectedly add a full turn.
        Already mapped headers and different episodes pass through unchanged.
        """
        snapshot = self.snapshot
        odometry = header.get("odometry")
        if (snapshot is None or odometry is None or header.get("episode") != snapshot.episode
                or odometry.get("frame") == "map"):
            return header
        x, y, angle = snapshot.correction
        c, s = math.cos(angle), math.sin(angle)
        mapped = dict(odometry, x=x+c*odometry["x"]-s*odometry["y"],
                      y=y+s*odometry["x"]+c*odometry["y"], yaw=odometry["yaw"]+angle, frame="map")
        if snapshot.position_sigma is not None:
            sigma = dict(odometry.get("sigma", {}))
            sigma["pos"] = max(sigma.get("pos", 0.0), snapshot.position_sigma)
            sigma["yaw"] = max(sigma.get("yaw", 0.0), snapshot.yaw_sigma)
            mapped["sigma"] = sigma
        return {**header, "odometry": mapped}
