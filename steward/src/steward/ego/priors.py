"""Category priors the robot brings to perception: affordances, how likely the object is fixed,
and fragility. They are robot knowledge, not World State facts.

``receptacle`` marks things whose floor holds other objects inside them (drawers, nesting bowls);
``pullable`` marks drawers that open along their front axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORY_PRIORS: dict[str, dict] = {
    "wall": {"role": "environment", "root": {"fixed": 1.0}, "affordances": [], "fragility": 0.0},
    "counter": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface"], "fragility": 0.0},
    "table": {"role": "object", "root": {"free": 0.7, "fixed": 0.3}, "affordances": ["support-surface"], "fragility": 0.0},
    "sideboard": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface", "container"],
                  "fragility": 0.0},
    "cabinet": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface", "container"],
                "fragility": 0.0},
    "drawer": {"role": "object", "root": {"fixed": 1.0}, "affordances": ["container", "receptacle", "pullable"],
               "fragility": 0.0},
    "cup": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.8},
    "bowl": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container", "receptacle", "stackable"],
             "fragility": 0.6},
    "plate": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "support-surface"], "fragility": 0.6},
    "wine_glass": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.95},
    "mug": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.4},
    "teddy_bear": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "scissors": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "toy_airplane": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.1},
    "airplane_fuselage": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.1},
    "airplane_wing": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.1},
    "airplane_propeller": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.15},
    "airplane_landing_gear": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.05},
    "car_body": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.1},
    "car_wheel": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.05},
    "toy_screw": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "electric_screwdriver": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "screwdriver"],
                             "fragility": 0.05},
    "manual": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "flat"], "fragility": 0.1},
    "picture": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "flat"], "fragility": 0.1},
    "pistol": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "pill_bottle": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.2},
    "perfume_bottle": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.8},
    "chair": {"role": "object", "root": {"free": 0.95, "fixed": 0.05}, "affordances": ["graspable", "floor-standing"],
              "fragility": 0.1},
    "plant": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "floor-standing"], "fragility": 0.3},
}

UNKNOWN_PRIOR = {"role": "object", "root": {"free": 0.5, "fixed": 0.5}, "affordances": [], "fragility": 0.5}

# Category hypotheses in template coordinates (metres), NOT measured entity coordinates.
# SCREW_HOLE_FRAMES declares the corresponding outline; registration must check observed
# dimensions and retain both half turns. A category label alone is not registration.
# group names identify catalogue interfaces; mates lists compatible peer groups.
# These are design connections, not measured identity or assembly pose. Missing
# metadata leaves geometric matching available; explicit selectors override it.
SCREW_HOLES: dict[str, list[dict]] = {
    "airplane_fuselage": [
        {"id": "wing_0", "p": [0.02, 0.012, 0.038], "group": "fuselage.wing", "mates": ('wing.mount',)},
        {"id": "wing_1", "p": [0.02, -0.012, 0.038], "group": "fuselage.wing", "mates": ('wing.mount',)},
        {"id": "gear_0", "p": [0.0, 0.012, 0.002], "group": "fuselage.gear", "mates": ('gear.mount',)},
        {"id": "gear_1", "p": [0.0, -0.012, 0.002], "group": "fuselage.gear", "mates": ('gear.mount',)},
    ],
    "airplane_wing": [
        {"id": "mount_0", "p": [0.0, 0.012, 0.008], "group": "wing.mount", "mates": ('fuselage.wing',)},
        {"id": "mount_1", "p": [0.0, -0.012, 0.008], "group": "wing.mount", "mates": ('fuselage.wing',)},
    ],
    "airplane_propeller": [{"id": "hub", "p": [0.0, 0.0, 0.004]}],
    "airplane_landing_gear": [
        {"id": "mount_0", "p": [0.0, 0.012, 0.006], "group": "gear.mount", "mates": ('fuselage.gear',)},
        {"id": "mount_1", "p": [0.0, -0.012, 0.006], "group": "gear.mount", "mates": ('fuselage.gear',)},
    ],
    "car_body": [
        {"id": "wheel_0", "p": [0.05, 0.048, 0.01575], "group": "body.wheel0", "mates": ('wheel.hub',)},
        {"id": "wheel_1", "p": [0.05, -0.048, 0.01575], "group": "body.wheel1", "mates": ('wheel.hub',)},
        {"id": "wheel_2", "p": [-0.05, 0.048, 0.01575], "group": "body.wheel2", "mates": ('wheel.hub',)},
        {"id": "wheel_3", "p": [-0.05, -0.048, 0.01575], "group": "body.wheel3", "mates": ('wheel.hub',)},
    ],
    "car_wheel": [{"id": "hub", "p": [0.0, 0.0, 0.008], "group": "wheel.hub", "mates": ('body.wheel0', 'body.wheel1', 'body.wheel2', 'body.wheel3')}],
}

# Static template geometry, same catalogue provenance as SCREW_HOLES above and
# the shape prototypes in classify.py. XY extents use the template's original
# axes (not sorted long/short). Values describe isolated part outlines from the
# asset design, never a scene pose, mating target, or desired assembly orientation.
# All these designs have their XY outline centre at the template origin.
SCREW_HOLE_FRAMES: dict[str, dict] = {
    "airplane_fuselage": {"sizeXYM": [0.18, 0.045], "centerXYM": [0.0, 0.0]},
    "airplane_wing": {"sizeXYM": [0.055, 0.20], "centerXYM": [0.0, 0.0]},
    "airplane_propeller": {"sizeXYM": [0.06, 0.06], "centerXYM": [0.0, 0.0]},
    "airplane_landing_gear": {"sizeXYM": [0.03, 0.08], "centerXYM": [0.0, 0.0]},
    "car_body": {"sizeXYM": [0.16, 0.101], "centerXYM": [0.0, 0.0]},
    "car_wheel": {"sizeXYM": [0.05, 0.05], "centerXYM": [0.0, 0.0]},
}

# Words that name a listed category in other terms ("desk" for a table).
SYNONYMS: dict[str, str] = {"desk": "table", "bench": "table", "shelf": "cabinet", "cupboard": "cabinet",
                            "dresser": "cabinet", "stool": "chair", "glass": "wine_glass"}


@dataclass(frozen=True)
class CommonSense:
    """Sizes by which an object of an unlisted category gets its affordances: a thing at least
    ``surface_min_m`` across, standing on the floor (base below ``floor_max_m``) with its top at
    least ``surface_top_min_m`` high, is furniture with a ``support-surface``; a thing less than
    ``graspable_max_m`` across is ``graspable``."""

    surface_min_m: float = 0.4
    surface_top_min_m: float = 0.3
    floor_max_m: float = 0.1
    graspable_max_m: float = 0.35


COMMON_SENSE = CommonSense()

COMPATIBLE: dict[str, set[str]] = {"cup": {"cup", "bowl"}, "bowl": {"bowl", "cup"}}


def prior(category: str) -> dict:
    return CATEGORY_PRIORS.get(category, UNKNOWN_PRIOR)


def screw_holes(category: str) -> list[dict]:
    """Known screw-hole markers for ``category`` (``{id, p}`` in the body frame), or [].

    :param category: the part category.
    :returns: hole descriptors the place skill can mate; empty when the category has none.
    """
    return [dict(hole) for hole in SCREW_HOLES.get(category, ())]


def default_affordances(category: str, footprint: float, bottom: float, top: float,
                        sense: CommonSense = COMMON_SENSE) -> list[str]:
    """Affordances an object has by common sense: its category prior's, else those of a listed
    category its name contains ("dinner table", "photo_frame"), else by its size (see
    :class:`CommonSense`); none for an object that is neither furniture nor small.

    :param category: the category perception or the robot's agent gave it.
    :param footprint: its larger horizontal extent in metres.
    :param bottom: the height of its base in metres.
    :param top: the height of its top in metres.
    :param sense: the sizes the size rule uses.
    :returns: the affordance names.
    """
    if category in CATEGORY_PRIORS:
        return list(CATEGORY_PRIORS[category]["affordances"])
    for word in reversed(re.split(r"[\s_\-]+", category.lower())):
        named = SYNONYMS.get(word, word)
        if named in CATEGORY_PRIORS:
            return list(CATEGORY_PRIORS[named]["affordances"])
    if footprint >= sense.surface_min_m and bottom < sense.floor_max_m and top >= sense.surface_top_min_m:
        return ["support-surface"]
    if footprint < sense.graspable_max_m:
        return ["graspable"]
    return []


def compatible(a: str, b: str) -> bool:
    return a == b or b in COMPATIBLE.get(a, set())
