"""Failures shared by task execution and body motion policies."""


class SkillStopped(Exception):
    """The caller cancelled the current skill."""


class SkillFailed(Exception):
    """The requested skill could not be completed."""


class OutOfReach(SkillFailed):
    """The requested pose has no accepted arm solution; no command was sent."""
