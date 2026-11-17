"""Compatibility constructor for the simplified Cobot; explicit models use the shared Piper IK."""
from steward.ego.piper_ik import ARMS, WORKING_JOINTS, IkError, PiperIK, _ENT


class CobotIK(PiperIK):
    """Use the simplified model when omitted; retain explicit-model callers."""

    def __init__(self, robot: dict | None = None, timestep: float = .002, self_clearance: float = .01,
                 working_joints: tuple[float, ...] = WORKING_JOINTS):
        if robot is None:
            from steward.assets.cobot import cobot_magic

            robot = cobot_magic().to_dict()
        super().__init__(robot, timestep, self_clearance, working_joints)
