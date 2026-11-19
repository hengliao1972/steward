"""Integrate biased planar motion increments without localization correction."""

import math
from dataclasses import dataclass


@dataclass
class OdometryTrack:
    """Noiseless accumulated sensor pose, separate from instantaneous measurement noise."""

    truth: tuple[float, float, float]
    pose: tuple[float, float, float]

    def advance(self, truth: tuple[float, float, float], scale_error: float, yaw_per_m: float) -> None:
        """Integrate true displacement rotated by accumulated heading error.

        :param truth: Current physical x, y and unwrapped yaw.
        :param scale_error: Fractional displacement scale error (0.01 means +1%).
        :param yaw_per_m: Signed heading bias accumulated per metre of travel, in radians.
        """
        dx, dy = truth[0] - self.truth[0], truth[1] - self.truth[1]
        distance = math.hypot(dx, dy)
        yaw_bias = yaw_per_m * distance
        angle = self.pose[2] - self.truth[2] + yaw_bias / 2
        c, s = math.cos(angle), math.sin(angle)
        self.pose = (self.pose[0] + (1 + scale_error) * (c * dx - s * dy),
                     self.pose[1] + (1 + scale_error) * (s * dx + c * dy),
                     self.pose[2] + truth[2] - self.truth[2] + yaw_bias)
        self.truth = truth
