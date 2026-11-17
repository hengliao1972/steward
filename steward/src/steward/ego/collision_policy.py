"""Model-link groups used by a body's IK self-collision constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SelfCollisionPolicy(Protocol):
    """Select pairs of link groups; the IK solver resolves them to contact geometries."""

    def pairs(self, robot: dict, arm: str) -> list[tuple[set[str], set[str]]]: ...


@dataclass(frozen=True)
class PiperSelfCollisionPolicy:
    """Keep Piper links 2–5 clear of the chassis and torso.

    Explicit body_links replaces the default non-arm links, allowing a body to include additional
    obstacles. Missing declared links fail before solving. Other policies may add arm/arm pairs.
    """

    body_links: tuple[str, ...] | None = None
    arm_links: tuple[int, ...] = (2, 3, 4, 5)

    def pairs(self, robot: dict, arm: str) -> list[tuple[set[str], set[str]]]:
        names = {link['id'] for link in robot['body']['links']}
        body = (set(self.body_links) if self.body_links is not None
                else {name for name in names if not name.startswith(('L_', 'R_'))})
        moving = {f'{arm}_link{k}' for k in self.arm_links}
        missing = (body | moving) - names
        if missing:
            raise ValueError(f'Self-collision policy refers to missing links: {sorted(missing)}')
        if not body or not moving:
            raise ValueError('Self-collision policy requires nonempty body and arm link groups')
        return [(moving, body)]
