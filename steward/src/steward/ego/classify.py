"""Category inference from measured geometry and colour, against the robot's shape priors.

A measured box can be smaller than the object (only part of it was in view) but rarely larger, so
an observed extent below the prototype costs little and one above it costs a lot. Whether the
object stands on the floor or on something raised separates furniture from tableware. The result
is a probability per category; the estimator accumulates these over observations.
"""

from __future__ import annotations

import math

import numpy as np

# Robot knowledge: typical full extents (long side, short side, height) per category, a scale range
# for categories that come in several sizes, where they stand, and an optional colour test.
SHAPES: dict[str, dict] = {
    "wall": {"dims": (4.0, 0.10, 2.6), "scale": (0.3, 1.6), "floor": True, "min_top": 1.2},
    # The furniture of this home has distinct finishes: a white counter, a walnut sideboard, a birch
    # table and a blue-grey cabinet (deployment knowledge, like the sizes).
    "counter": {"dims": (1.85, 0.62, 0.90), "floor": True, "color": "white"},
    "table": {"dims": (1.20, 0.80, 0.75), "floor": True, "color": "birch"},
    "sideboard": {"dims": (1.42, 0.49, 0.85), "floor": True, "color": "walnut"},
    "cabinet": {"dims": (0.62, 0.52, 0.80), "floor": True, "color": "bluegrey"},
    "chair": {"dims": (0.47, 0.46, 0.90), "floor": True, "color": "wood"},
    "plant": {"dims": (0.26, 0.22, 0.40), "scale": (0.7, 2.5), "floor": True, "color": "green"},
    "drawer": {"dims": (0.56, 0.44, 0.22), "floor": False, "min_bottom": 0.35},
    "bowl": {"dims": (0.17, 0.17, 0.06), "scale": (0.65, 1.85), "floor": False},
    "cup": {"dims": (0.08, 0.08, 0.10), "floor": False},
    "plate": {"dims": (0.24, 0.24, 0.02), "floor": False},
    "wine_glass": {"dims": (0.08, 0.08, 0.19), "floor": False, "color": "clear"},
    "mug": {"dims": (0.11, 0.085, 0.095), "floor": False, "color": "dark"},
    "teddy_bear": {"dims": (0.11, 0.105, 0.18), "floor": False, "color": "brown"},
    "scissors": {"dims": (0.16, 0.07, 0.012), "floor": False},
    "toy_airplane": {"dims": (0.22, 0.20, 0.06), "floor": False, "color": "yellow"},
    "airplane_fuselage": {"dims": (0.18, 0.045, 0.04), "floor": False, "color": "yellow"},
    "airplane_wing": {"dims": (0.20, 0.055, 0.01), "floor": False, "color": "red"},
    "airplane_propeller": {"dims": (0.06, 0.06, 0.01), "floor": False, "color": "orange"},
    "airplane_landing_gear": {"dims": (0.08, 0.03, 0.025), "floor": False},
    "car_body": {"dims": (0.16, 0.10, 0.07), "floor": False, "color": "blue"},
    "car_wheel": {"dims": (0.05, 0.05, 0.016), "floor": False, "color": "dark"},
    "toy_screw": {"dims": (0.048, 0.048, 0.066), "floor": False},
    "screw_board": {"dims": (0.28, 0.20, 0.018), "floor": False, "color": "light"},
    "electric_screwdriver": {"dims": (0.16, 0.035, 0.04), "floor": False, "color": "orange"},
    "manual": {"dims": (0.24, 0.24, 0.004), "floor": False},
    "picture": {"dims": (0.24, 0.24, 0.004), "floor": False},
    "pistol": {"dims": (0.18, 0.12, 0.03), "floor": False, "color": "dark"},
    "pill_bottle": {"dims": (0.046, 0.046, 0.085), "floor": False},
    "perfume_bottle": {"dims": (0.065, 0.04, 0.10), "floor": False, "color": "pink"},
}

FLAT = 0.05  # a region thinner than this that is not on the floor is a top surface, not a height
UNDER = (0.55, 1.2, 0.55)  # tolerance (log ratio) per axis for measuring less than the prototype; the
# short side is usually the depth seen from the front, so it may come out much thinner
OVER = 0.25  # tolerance for measuring more (1 cm voxels, depth noise, overhangs)
UNKNOWN = "unknown"
UNKNOWN_COST = 7.0  # a region that fits no prototype this badly is something the robot has no prior for;
# one broken placement rule (6) on an otherwise fitting shape still names the category


def _color_score(rule: str | None, color: np.ndarray, invalid_frac: float,
                 geom_cost: float | None = None) -> float:
    """Colour agreement for a shape rule; ``geom_cost`` softens a clear miss when size already fits.

    Transparent glass often paints table beige into the mask while depth holes are uneven, so a hard
    clear fail would demote a well-measured stemware box to :data:`UNKNOWN`. A miss with a low
    geometric cost therefore only lightly penalises ``clear``; a miss on a poorly fitting box still
    costs the full −2.
    """
    r, g, b = (float(x) for x in color)
    value, low = max(r, g, b), min(r, g, b)
    saturation = 0.0 if value == 0 else (value - low) / value
    if rule == "clear":
        if invalid_frac > 0.15 or (saturation < 0.12 and value > 0.6):
            return 1.0
        # Partial depth dropouts on glass: weaker than the clear hit, but not a full reject.
        if invalid_frac > 0.08 and value > 0.5:
            return 0.25
        if geom_cost is not None and geom_cost < 2.0:
            return -0.25
        return -2.0
    if rule == "dark":
        return 0.0 if value < 0.3 else -3.0
    if rule == "brown":
        return 0.0 if (r > g > b and 0.25 < value < 0.8 and saturation > 0.3) else -4.0
    if rule == "wood":
        return 0.0 if (r > g > b and saturation > 0.2) else -3.0
    if rule == "white":
        return 0.0 if (saturation < 0.15 and value > 0.55) else -3.0
    if rule == "walnut":
        return 0.0 if (r >= g >= b and value < 0.55 and saturation > 0.2) else -3.0
    if rule == "birch":
        return 0.0 if (r > g > b and value > 0.6 and 0.2 < saturation < 0.5) else -3.0
    if rule == "bluegrey":
        return 0.0 if (b > r and b >= g and saturation > 0.15) else -3.0
    if rule == "green":
        return 0.0 if (g > r and g > b and saturation > 0.25) else -3.0
    if rule == "yellow":
        return 0.0 if (r > 0.45 and g > 0.35 and r >= g > b and saturation > 0.45) else -3.0
    if rule == "pink":
        return 0.0 if (r > 0.45 and r > b > g and saturation > 0.2) else -2.0
    return 0.0


def content_color_name(size: np.ndarray, color: np.ndarray | None, invalid_frac: float,
                       category: str) -> str | None:
    """A more specific name when colour separates two categories the detector confuses.

    A SAM score outweighs the shape prior, so a pink bottle called a pill bottle, a white
    smear called a generic bottle, a yellow airplane called a bowl, and a brown bear called
    a drawer all stay wrong. Colour decides those pairs, and only for a box the size of the
    content. Returns None to keep ``category``.

    :param size: full extents of the measured box, metres.
    :param color: mean RGB in 0..1, or None when the view had no colour.
    :param invalid_frac: fraction of the region whose depth was missing.
    :param category: the name the fused probabilities currently give the track.
    :returns: the replacement category, or None.
    """
    if color is None:
        return None
    long = float(max(size[0], size[1]))
    short = float(min(size[0], size[1]))
    height = float(size[2])
    if long > 0.35 or height < 0.02:
        return None
    red, green, blue = (float(channel) for channel in color)
    # The perfume is pink glass. The mask takes the pale drawer floor with it, so the mean
    # is only faintly pink: red still leads, but the saturation is below the pink rule.
    washed = (red - min(green, blue)) / red if red > 0.0 else 0.0
    if (red > blue > green and red > 0.55 and washed > 0.06
            and category in ("pill_bottle", "bottle", "cup", "unknown")
            and long < 0.12 and short < 0.08 and height > 0.05):
        return "perfume_bottle"
    if (_color_score("pink", color, invalid_frac) >= 0 and category in ("pill_bottle", "bottle", "cup")
            and long < 0.12 and height > 0.05):
        return "perfume_bottle"
    if (_color_score("white", color, invalid_frac) >= 0 and category in ("bottle", "unknown", "cup")
            and short < 0.08 and long < 0.16 and 0.05 < height < 0.15):
        return "pill_bottle"
    if (_color_score("yellow", color, invalid_frac) >= 0 and category in ("bowl", "plate", "box", "cup")
            and long >= 0.12 and short >= 0.07 and height < 0.14):
        return "toy_airplane"
    if (_color_score("brown", color, invalid_frac) >= 0 and category in ("drawer", "box", "unknown")
            and 0.08 < long < 0.22 and height > 0.10):
        return "teddy_bear"
    # A dark cup is the mug. Cup has no colour of its own, so a black mug that SAM calls a cup stays a cup.
    if (category in ("cup", "unknown") and long < 0.16 and short < 0.12 and 0.04 < height < 0.14
            and max(red, green, blue) < 0.45):
        return "mug"
    return None


def clear_stem_name(size: np.ndarray, category: str, color: np.ndarray | None,
                    invalid_frac: float) -> str | None:
    """A wine glass, when the measured box is stemware but the prior said unknown.

    With the base unseen, every category is scored by the height of the surface it stands on, so a
    19 cm glass matches nothing. The box around the glass itself is still about 8 cm by 19 cm.
    A cup is about 10 cm and a bowl about 6 cm, so a box this tall is the stem even when the
    table behind the glass makes the mean colour opaque.

    :param size: full extents of the measured box, metres.
    :param category: the name the fused probabilities currently give the track.
    :param color: mean RGB in 0..1, or None when the view had no colour.
    :param invalid_frac: fraction of the region whose depth was missing.
    :returns: ``wine_glass``, or None to keep ``category``.
    """
    if category not in ("unknown", "bowl", "cup"):
        return None
    long = float(max(size[0], size[1]))
    short = float(min(size[0], size[1]))
    height = float(size[2])
    if long > 0.14 or short > 0.12 or not (0.15 < height < 0.26):
        return None
    return "wine_glass"


def sideboard_not_counter(size: np.ndarray, category: str, color: np.ndarray | None,
                          invalid_frac: float) -> str | None:
    """The sideboard, when a counter label sits on a sideboard-sized walnut piece.

    The counter is about 1.85 m by 0.62 m and white. The sideboard is about 1.42 m by 0.49 m.
    SAM's counter score outweighs that difference. A white piece is left as a counter.

    :param size: full extents of the measured box, metres.
    :param category: the name the fused probabilities currently give the track.
    :param color: mean RGB in 0..1, or None when the view had no colour.
    :param invalid_frac: fraction of the region whose depth was missing.
    :returns: ``sideboard``, or None to keep ``category``.
    """
    if category != "counter":
        return None
    long = float(max(size[0], size[1]))
    short = float(min(size[0], size[1]))
    height = float(size[2])
    if not (1.15 < long < 1.60 and 0.40 < short < 0.56 and 0.70 < height < 0.95):
        return None
    if color is not None and _color_score("white", color, invalid_frac) >= 0:
        return None
    return "sideboard"


def perfume_without_bottle_label(size: np.ndarray, category: str, semantics: dict[str, float]) -> str | None:
    """The perfume, when the detector never called the region a bottle.

    The pill bottle is the region SAM names. The perfume's mask is the drawer around a
    square box about 12 cm tall, and that box fits the pill's footprint, so the shape
    prior says pill. A box this tall that SAM did not call a pill is the perfume.

    :param size: full extents of the measured box, metres.
    :param category: the name the fused probabilities currently give the track.
    :param semantics: summed segmenter scores for the track.
    :returns: ``perfume_bottle``, or None to keep ``category``.
    """
    if category not in ("pill_bottle", "bottle"):
        return None
    if any(semantics.get(name, 0.0) > 0.0 for name in ("pill_bottle", "bottle", "perfume_bottle")):
        return None
    long = float(max(size[0], size[1]))
    height = float(size[2])
    if long > 0.06 or not (0.105 < height < 0.16):
        return None
    return "perfume_bottle"


def classify(size: np.ndarray, bottom: float, color: np.ndarray, invalid_frac: float = 0.0,
             temperature: float = 1.0, bottom_seen: bool = True, min_prob: float = 0.01) -> dict[str, float]:
    """Probability per category for a measured box (full extents, long side first) whose lowest
    point is ``bottom`` metres above the floor; :data:`UNKNOWN` takes the mass when no prototype fits.

    With ``bottom_seen`` false the lowest point is an image edge or an occluder, not the base: it is
    no evidence of standing on the floor or not, and a floor-standing prototype is compared with
    the visible top instead of the visible height (a table seen from close shows only its top).
    Categories below ``min_prob`` are dropped."""
    seen = np.array([max(size[0], size[1]), min(size[0], size[1]), size[2]], float)
    seen = np.maximum(seen, 0.004)
    scores = {}
    on_floor = bottom < 0.06
    top = bottom + seen[2]
    # Seen straight down, a tabletop is a thin plate at table height: its legs are underneath it and
    # its sides are hidden, so the measured height is the top's thickness. For a prototype that
    # stands on the floor, such a region is compared by the height it reaches instead -- without
    # this the one camera that can see a whole tabletop calls the table "unknown", and the Ego
    # State has its tableware resting on nothing.
    flat = seen[2] < FLAT and not on_floor
    for name, shape in SHAPES.items():
        proto = np.array(shape["dims"], float)
        reaches_down = (not bottom_seen or (flat and shape["floor"]))
        obs = np.array([seen[0], seen[1], max(seen[2], top)]) if reaches_down else seen
        lo, hi = shape.get("scale", (1.0, 1.0))
        # One scale for the whole object, set by the axis that fills the prototype most: a partial
        # view only shrinks the other axes, which costs little, while a flat wide region cannot pass
        # for a small deep object scaled per axis.
        scale = min(max(float(np.max(obs / proto)), lo), hi)
        cost = 0.0
        for k in range(3):
            r = math.log(obs[k] / (proto[k] * scale))
            cost += (r / UNDER[k]) ** 2 if r < 0 else (r / OVER) ** 2
        if bottom_seen and not reaches_down and shape["floor"] != on_floor:
            cost += 6.0
        if bottom_seen and bottom < shape.get("min_bottom", -1.0):
            cost += 6.0
        if bottom + obs[2] < shape.get("min_top", -1.0):
            cost += 6.0
        geom_cost = cost
        cost -= _color_score(shape.get("color"), color, invalid_frac,
                             geom_cost if shape.get("color") == "clear" else None)
        scores[name] = -cost / temperature
    scores[UNKNOWN] = -UNKNOWN_COST / temperature
    top = max(scores.values())
    weights = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items() if v / total > min_prob}


_BOTTLE_FAMILY = ("pill_bottle", "perfume_bottle")


def shape_over_bottle(fused: dict[str, float], shape: dict[str, float]) -> dict[str, float]:
    """The bottle prior that fits, when the fused winner is the generic name ``bottle``.

    ``bottle`` has no template. It names a family. The segmenter has said which family; the
    prior picks the member. A cup that merely fits the box is not in that family, so it does
    not inherit the name.

    :param fused: probabilities from :func:`fuse_semantics`.
    :param shape: probabilities from :func:`classify` with ``min_prob=0``.
    :returns: the fitting bottle prior, or ``fused`` when ``bottle`` is not the winner or no
        bottle prior fits better than :data:`UNKNOWN`.
    """
    if not fused or max(fused, key=fused.get) != "bottle":
        return fused
    members = {name: shape.get(name, 0.0) for name in _BOTTLE_FAMILY}
    if not members or max(members.values()) <= shape.get(UNKNOWN, 0.0):
        return fused
    winner = max(members, key=members.get)
    return {winner: 1.0}


def fuse_respecting_color(semantic: dict[str, float], shape: dict[str, float], color: np.ndarray,
                          invalid_frac: float, weight: float, floor: float) -> dict[str, float]:
    """Fused probabilities after a failed colour test drops that segmenter score.

    The score returns to the floor and fusion runs again, until the winner has no colour
    test or the test passes. Each name is dropped at most once. A name with no colour test,
    such as cup, is unchanged: colour does not invent a test it does not have.

    :param semantic: category → score for the names the segmenter gave the region.
    :param shape: probabilities from :func:`classify` with ``min_prob=0``.
    :param color: mean RGB in 0..1.
    :param invalid_frac: fraction of the region whose depth was missing.
    :param weight: exponent of the shape probability.
    :param floor: score used for a name the segmenter did not give.
    :returns: category → probability.
    """
    remaining = dict(semantic)
    banned: set[str] = set()
    fused = fuse_semantics(remaining, shape, weight, floor)
    while fused:
        top = max(fused, key=fused.get)
        rule = SHAPES.get(top, {}).get("color")
        if rule is None or _color_score(rule, color, invalid_frac) >= 0 or top in banned:
            return fused
        banned.add(top)
        remaining.pop(top, None)
        # A prior that was never in the segmenter list still wins the next pass unless it is
        # removed. Dropping it from ``remaining`` alone leaves it in the shape probabilities.
        shaped = {k: v for k, v in shape.items() if k not in banned}
        fused = fuse_semantics(remaining, shaped, weight, floor)
    return fused


def fuse_semantics(semantic: dict[str, float], shape: dict[str, float], weight: float, floor: float,
                   unknown: str = UNKNOWN) -> dict[str, float]:
    """Category probabilities from SAM scores and shape-prior probabilities:
    ``p(c) ∝ L(c) · P_shape(c)^weight`` with ``L(c)`` the score of ``c`` (``floor`` when SAM did not
    name it) and, for a category without a shape template (:data:`SHAPES`),
    ``P_shape(c) = P_shape(unknown)``: shape has no opinion on what it has no template for.

    :param semantic: category → score in (0, 1] for the names SAM gave the region.
    :param shape: category → probability from :func:`classify` with ``min_prob=0``, so that a
        template that does not fit at all (a two-metre bowl) keeps its tiny probability.
    :returns: category → probability, entries below 0.01 dropped."""
    tiny = 1e-6
    neutral = max(shape.get(unknown, 0.0), tiny)
    scores = {c: semantic.get(c, floor) * max(shape.get(c, tiny if c in SHAPES else neutral), tiny) ** weight
              for c in set(semantic) | set(shape)}
    total = sum(scores.values())
    return {c: v / total for c, v in scores.items() if v / total > 0.01}


def prototype(category: str) -> tuple[float, float, float] | None:
    shape = SHAPES.get(category)
    return shape["dims"] if shape and "scale" not in shape else None


def max_plan_aspect(categories: tuple[str, ...]) -> float:
    """The greatest long-side / short-side ratio among those shape priors.

    A measured box longer than every one of these priors, on a region the segmenter called a
    bottle, is a strip of the support rather than one of those bottles. The ratio is the
    priors' own, so it does not follow the longest smear seen in one room.

    :param categories: keys of :data:`SHAPES`.
    :returns: a ratio, or 1 when none of the categories has a prior.
    """
    ratios = []
    for name in categories:
        shape = SHAPES.get(name)
        if not shape:
            continue
        long, short, _height = shape["dims"]
        ratios.append(float(long) / float(short))
    return max(ratios) if ratios else 1.0


def partial_pistol(size: np.ndarray, category: str, color: np.ndarray | None,
                   invalid_frac: float) -> str | None:
    """A pistol, when a dark speck is too small to be a bowl and as flat as the pistol prior.

    A black surface returns depth on only part of itself, so the long side is a lower bound.
    The bowl prior's smallest scale is still larger than this speck, and the height matches
    the pistol prior. A full-length pistol is left to the ordinary prior.

    :param size: full extents of the measured box, metres.
    :param category: the name the fused probabilities currently give the track.
    :param color: mean RGB in 0..1, or None when the view had no colour.
    :param invalid_frac: fraction of the region whose depth was missing.
    :returns: ``pistol``, or None to keep ``category``.
    """
    if color is None or category not in ("bowl", "unknown"):
        return None
    if _color_score("dark", color, invalid_frac) < 0:
        return None
    long = float(max(size[0], size[1]))
    height = float(size[2])
    bowl = SHAPES["bowl"]
    min_long = float(bowl["dims"][0]) * float(bowl["scale"][0])
    pistol_h = float(SHAPES["pistol"]["dims"][2])
    if long >= min_long:
        return None
    if not (pistol_h * math.exp(-UNDER[2]) <= height <= pistol_h * math.exp(OVER)):
        return None
    return "pistol"


def max_long_extent(category: str) -> float | None:
    """The longest side a box of ``category`` can have and still be that category.

    The prior's long side times its largest scale times :data:`OVER`. Past this, the
    measurement is a different object. The bound is the prior, so it does not grow when
    one room contains a larger false bowl.

    :param category: a key of :data:`SHAPES`.
    :returns: metres, or None when ``category`` has no shape prior.
    """
    shape = SHAPES.get(category)
    if not shape:
        return None
    hi = float(shape.get("scale", (1.0, 1.0))[1])
    return float(shape["dims"][0]) * hi * math.exp(OVER)
