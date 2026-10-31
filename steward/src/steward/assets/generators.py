"""Procedural asset generators. Each returns an entity template: a World State ``Entity`` whose id
is ``w:template`` and whose pose is the identity; :func:`steward.assets.library.instantiate`
assigns the scene id and pose.

Sizes are full extents in metres. Every template origin sits on the floor (z = 0) at the
footprint center, so a placement pose only needs x, y and yaw.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from steward.math3d import quat_from_axis_angle, quat_from_yaw
from steward.scene.schema import Entity, Est, Joint, Link, RigidTree, exact, pose, primitive

TEMPLATE_ID = "w:template"

WOOD = [0.62, 0.45, 0.30, 1.0]
LIGHT_WOOD = [0.80, 0.67, 0.50, 1.0]
# Furniture of this kitchen in clearly different finishes, so that a robot seeing only part of a piece
# can still tell the counter, sideboard, table and cabinet apart by colour.
COUNTER_WHITE = [0.93, 0.93, 0.92, 1.0]
COUNTER_TOP = [0.78, 0.79, 0.81, 1.0]
WALNUT = [0.36, 0.22, 0.13, 1.0]
WALNUT_DARK = [0.27, 0.16, 0.09, 1.0]
BIRCH = [0.86, 0.75, 0.56, 1.0]
BIRCH_LEG = [0.74, 0.62, 0.44, 1.0]
CABINET_BLUE = [0.40, 0.52, 0.66, 1.0]
WALL = [0.92, 0.91, 0.88, 1.0]
FLOOR = [0.55, 0.52, 0.48, 1.0]
STEEL = [0.70, 0.72, 0.75, 1.0]
CERAMIC = [0.95, 0.95, 0.97, 1.0]
GLASS = [0.75, 0.88, 0.95, 0.6]
CLEAR_GLASS = [0.85, 0.93, 0.98, 0.3]
BLACK = [0.06, 0.06, 0.07, 1.0]
TEDDY = [0.58, 0.38, 0.20, 1.0]
TEDDY_DARK = [0.30, 0.18, 0.10, 1.0]
SCISSOR_HANDLE = [0.80, 0.12, 0.12, 1.0]
TOY_YELLOW = [0.95, 0.78, 0.10, 1.0]
TOY_RED = [0.85, 0.15, 0.15, 1.0]
PILL_WHITE = [0.94, 0.94, 0.92, 1.0]
PILL_CAP = [0.95, 0.45, 0.10, 1.0]
PERFUME_GLASS = [0.95, 0.60, 0.75, 0.8]
GOLD = [0.85, 0.70, 0.30, 1.0]
COLORS = {"red": [0.85, 0.15, 0.15, 1.0], "green": [0.15, 0.70, 0.25, 1.0], "blue": [0.15, 0.35, 0.90, 1.0]}
ROBOT_BODY = [0.20, 0.32, 0.55, 1.0]
ROBOT_ARM = [0.85, 0.55, 0.15, 1.0]


def _fixed(jid: str, parent: str, child: str, at: dict) -> Joint:
    return Joint(id=jid, parent=parent, child=child, type=exact("fixed"), origin=exact(at), q=exact([]), qd=exact([]))


def _entity(category: str, role: str, root: str, links: list[Link], joints: list[Joint] | None = None,
            state_vars: dict[str, Est] | None = None, root_actuator: dict | None = None) -> Entity:
    return Entity(
        id=TEMPLATE_ID,
        role=role,
        category=exact(category),
        pose=exact(pose()),
        body=RigidTree(root=exact(root), links=links, joints=joints or [], root_actuator=root_actuator),
        state_vars=state_vars or {},
    )


def building(width: float = 6.0, depth: float = 5.0, height: float = 2.6, wall: float = 0.1) -> Entity:
    """Floor plus four walls; every wall is a link welded to the floor by a fixed joint."""
    floor = Link(id="floor", name="floor",
                 geometry={"collision": [primitive("box", [width, depth, 0.1], pose([0, 0, -0.05]), FLOOR)]},
                 material=exact({"friction": 0.9}))
    walls = {
        "wall_north": ([0, depth / 2 + wall / 2, height / 2], [width + 2 * wall, wall, height]),
        "wall_south": ([0, -depth / 2 - wall / 2, height / 2], [width + 2 * wall, wall, height]),
        "wall_east": ([width / 2 + wall / 2, 0, height / 2], [wall, depth, height]),
        "wall_west": ([-width / 2 - wall / 2, 0, height / 2], [wall, depth, height]),
    }
    links = [floor]
    joints = []
    for name, (center, size) in walls.items():
        links.append(Link(id=name, name=name, geometry={"collision": [primitive("box", size, None, WALL)]}))
        joints.append(_fixed(f"{name}_mount", "floor", name, pose(center)))
    return _entity("building", "environment", "fixed", links, joints)


def counter(width: float = 1.8, depth: float = 0.6, height: float = 0.9) -> Entity:
    """Fixed kitchen counter without openable parts."""
    top = 0.04
    carcass = primitive("box", [width, depth, height - top], pose([0, 0, (height - top) / 2]), COUNTER_WHITE)
    slab = primitive("box", [width + 0.02, depth + 0.02, top], pose([0, 0, height - top / 2]), COUNTER_TOP)
    link = Link(id="body", name="counter", geometry={"collision": [carcass, slab]},
                affordances=[exact("support-surface")])
    return _entity("counter", "object", "fixed", [link])


def sideboard(width: float = 1.4, depth: float = 0.45, height: float = 0.85, drawer_width: float = 0.0,
              drawer_height: float = 0.22) -> Entity:
    """Fixed low storage cabinet with a flat top; its long side runs along local y and its front faces +x.

    With ``drawer_width > 0`` the top section holds one centred drawer on a prismatic joint along +x.
    """
    top = 0.03
    plinth = 0.06
    if drawer_width <= 0:
        geoms = [
            primitive("box", [depth - 0.04, width - 0.04, plinth], pose([0, 0, plinth / 2]), WALNUT_DARK),
            primitive("box", [depth, width, height - top - plinth], pose([0, 0, plinth + (height - top - plinth) / 2]), WALNUT),
            primitive("box", [depth + 0.02, width + 0.02, top], pose([0, 0, height - top / 2]), WALNUT_DARK),
        ]
        link = Link(id="body", name="sideboard", geometry={"collision": geoms}, affordances=[exact("support-surface")])
        return _entity("sideboard", "object", "fixed", [link])
    wall = 0.02
    lower_top = height - top - drawer_height
    side_w = (width - drawer_width) / 2
    geoms = [
        primitive("box", [depth - 0.04, width - 0.04, plinth], pose([0, 0, plinth / 2]), WALNUT_DARK),
        primitive("box", [depth, width, lower_top - plinth], pose([0, 0, plinth + (lower_top - plinth) / 2]), WALNUT),
        primitive("box", [depth + 0.02, width + 0.02, top], pose([0, 0, height - top / 2]), WALNUT_DARK),
        primitive("box", [depth, side_w, drawer_height], pose([0, drawer_width / 2 + side_w / 2, lower_top + drawer_height / 2]), WALNUT),
        primitive("box", [depth, side_w, drawer_height], pose([0, -drawer_width / 2 - side_w / 2, lower_top + drawer_height / 2]), WALNUT),
        primitive("box", [wall, drawer_width, drawer_height], pose([-depth / 2 + wall / 2, 0, lower_top + drawer_height / 2]), WALNUT),
    ]
    body = Link(id="body", name="sideboard", geometry={"collision": geoms}, affordances=[exact("support-surface")])
    drawer, slide = _drawer("body", depth, drawer_width, drawer_height, wall, lower_top)
    return _entity("sideboard", "object", "fixed", [body, drawer], [slide])


def _drawer(parent: str, depth: float, opening_w: float, drawer_height: float, wall: float,
            floor_z: float) -> tuple[Link, Joint]:
    """Drawer link and its prismatic slide along +x for an opening ``opening_w`` wide whose floor is at ``floor_z``."""
    inner_w = opening_w - 2 * wall - 0.01
    inner_d = depth - wall - 0.02
    dh = drawer_height - 0.02
    geoms = [
        primitive("box", [inner_d, inner_w, 0.01], pose([0, 0, 0.005]), LIGHT_WOOD),
        primitive("box", [0.02, inner_w + 0.008, dh], pose([inner_d / 2, 0, dh / 2]), LIGHT_WOOD),
        primitive("box", [0.012, inner_w, dh * 0.7], pose([-inner_d / 2, 0, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [inner_d, 0.012, dh * 0.7], pose([0, inner_w / 2 - 0.006, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [inner_d, 0.012, dh * 0.7], pose([0, -inner_w / 2 + 0.006, dh * 0.35]), LIGHT_WOOD),
        primitive("box", [0.03, 0.12, 0.02], pose([inner_d / 2 + 0.025, 0, dh / 2]), STEEL),
    ]
    drawer = Link(id="drawer", name="drawer", geometry={"collision": geoms}, inertial=exact({"mass": 1.5}),
                  affordances=[exact("pullable"), exact("graspable-handle"), exact("container")])
    slide = Joint(
        id="drawer_slide", parent=parent, child="drawer", type=exact("prismatic"),
        origin=exact(pose([wall / 2 + 0.005, 0, floor_z + 0.006])), axis=exact([1.0, 0.0, 0.0]),
        limits=exact({"lo": 0.0, "hi": inner_d * 0.85}), q=exact([0.0]), qd=exact([0.0]),
        dynamics=exact({"damping": 8.0, "friction": 0.5}),
    )
    return drawer, slide


def dining_table(width: float = 1.2, depth: float = 0.8, height: float = 0.75, mass: float = 25.0) -> Entity:
    """Free-standing table: heavy but movable."""
    top_t, leg = 0.04, 0.05
    geoms = [primitive("box", [width, depth, top_t], pose([0, 0, height - top_t / 2]), BIRCH)]
    leg_h = height - top_t
    for sx in (-1, 1):
        for sy in (-1, 1):
            at = pose([sx * (width / 2 - leg), sy * (depth / 2 - leg), leg_h / 2])
            geoms.append(primitive("box", [leg, leg, leg_h], at, BIRCH_LEG))
    link = Link(id="body", name="table", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8}), affordances=[exact("support-surface")])
    return _entity("table", "object", "free", [link])


def cabinet_with_drawer(width: float = 0.6, depth: float = 0.5, height: float = 0.8,
                        drawer_height: float = 0.2) -> Entity:
    """Fixed cabinet whose top section holds one drawer on a prismatic joint along +x (front)."""
    wall = 0.02
    lower_h = height - drawer_height - wall
    carcass = [
        primitive("box", [depth, width, lower_h], pose([0, 0, lower_h / 2]), CABINET_BLUE),
        primitive("box", [depth, width, wall], pose([0, 0, height - wall / 2]), CABINET_BLUE),
        primitive("box", [depth, wall, drawer_height], pose([0, width / 2 - wall / 2, lower_h + drawer_height / 2]), CABINET_BLUE),
        primitive("box", [depth, wall, drawer_height], pose([0, -width / 2 + wall / 2, lower_h + drawer_height / 2]), CABINET_BLUE),
        primitive("box", [wall, width, drawer_height], pose([-depth / 2 + wall / 2, 0, lower_h + drawer_height / 2]), CABINET_BLUE),
    ]
    base = Link(id="carcass", name="carcass", geometry={"collision": carcass}, affordances=[exact("support-surface")])
    drawer, slide = _drawer("carcass", depth, width, drawer_height, wall, lower_h)
    return _entity("cabinet", "object", "fixed", [base, drawer], [slide])


def cup(radius: float = 0.04, height: float = 0.10, mass: float = 0.25, fragile: bool = True) -> Entity:
    link = Link(id="body", name="cup",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), GLASS if fragile else CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 1.0, "fragility": 0.8 if fragile else 0.2}),
                affordances=[exact("graspable"), exact("container")])
    return _entity("cup", "object", "free", [link])


def plate(radius: float = 0.12, height: float = 0.02, mass: float = 0.4) -> Entity:
    link = Link(id="body", name="plate",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 0.9, "fragility": 0.6}),
                affordances=[exact("graspable"), exact("support-surface")])
    return _entity("plate", "object", "free", [link])


def bowl(radius: float = 0.08, height: float = 0.06, mass: float = 0.3) -> Entity:
    link = Link(id="body", name="bowl",
                geometry={"collision": [primitive("cylinder", [radius, height], pose([0, 0, height / 2]), CERAMIC)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 0.9, "fragility": 0.6}),
                affordances=[exact("graspable"), exact("container")])
    return _entity("bowl", "object", "free", [link])


def nesting_bowl(radius: float = 0.08, height: float = 0.06, mass: float = 0.3, color: str = "white",
                 wall: float = 0.006, segments: int = 12) -> Entity:
    """Open bowl with a thin wall, so a smaller bowl can nest inside a larger one.

    The wall segments start just above the base, so only the base disc touches what the bowl
    stands on (fewer contacts per bowl).
    """
    rgba = COLORS.get(color, CERAMIC)
    base_t = 0.008
    lift = 0.002
    geoms = [primitive("cylinder", [radius, base_t], pose([0, 0, base_t / 2]), rgba)]
    seg_w = 2 * math.pi * radius / segments * 1.12
    for k in range(segments):
        a = 2 * math.pi * k / segments
        r = radius - wall / 2
        geoms.append(primitive("box", [wall, seg_w, height - lift],
                               pose([r * math.cos(a), r * math.sin(a), lift + (height - lift) / 2], quat_from_yaw(a)), rgba))
    link = Link(id="body", name="bowl", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.9, "fragility": 0.6}),
                affordances=[exact("graspable"), exact("container"), exact("stackable")])
    return _entity("bowl", "object", "free", [link])


CARDBOARD = [0.78, 0.66, 0.48, 1.0]


def printed_board(width: float = 0.12, depth: float = 0.12, thickness: float = 0.006, texture: str | None = None,
                  back: list[float] | None = None, mass: float = 0.02, category: str = "board") -> Entity:
    """Thin rigid board with a printed top face (``texture``: base64 PNG, image top towards local
    +y) and a plain back and edges (``back``): a puzzle piece, a box lid, a card."""
    link = Link(id="body", name=category.replace("_", " "),
                geometry={"collision": [primitive("box", [width, depth, thickness], pose([0, 0, thickness / 2]),
                                                  back or CARDBOARD, texture=texture)]},
                inertial=exact({"mass": mass}), material=exact({"friction": 0.8, "fragility": 0.1, "contactTime": 0.005}),
                affordances=[exact("graspable"), exact("flat")])
    return _entity(category, "object", "free", [link])


def open_box(width: float = 0.42, depth: float = 0.42, height: float = 0.05, wall: float = 0.01,
             color: list[float] | None = None, mass: float = 0.3) -> Entity:
    """Open-top box (a base and four walls) that can be carried with what lies in it and tipped
    to pour it out."""
    rgba = color or CARDBOARD
    base_t = 0.015
    geoms = [primitive("box", [width, depth, base_t], pose([0, 0, base_t / 2]), rgba)]
    h = height - base_t
    for x, y, sx, sy in ((width / 2 - wall / 2, 0, wall, depth), (-width / 2 + wall / 2, 0, wall, depth),
                         (0, depth / 2 - wall / 2, width - 2 * wall, wall), (0, -depth / 2 + wall / 2, width - 2 * wall, wall)):
        geoms.append(primitive("box", [sx, sy, h], pose([x, y, base_t + h / 2]), rgba))
    link = Link(id="body", name="box", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.1, "contactTime": 0.005}),
                affordances=[exact("graspable"), exact("container"), exact("pourable")])
    return _entity("box", "object", "free", [link])


def wine_glass(bowl_radius: float = 0.04, bowl_height: float = 0.09, stem_height: float = 0.09,
               base_radius: float = 0.036, mass: float = 0.2) -> Entity:
    """Clear stemmed wine glass: base disc, thin stem and a cup-shaped bowl."""
    stem_r = 0.005
    geoms = [
        primitive("cylinder", [base_radius, 0.006], pose([0, 0, 0.003]), CLEAR_GLASS),
        primitive("cylinder", [stem_r, stem_height], pose([0, 0, 0.006 + stem_height / 2]), CLEAR_GLASS),
        primitive("cylinder", [bowl_radius, bowl_height], pose([0, 0, 0.006 + stem_height + bowl_height / 2]), CLEAR_GLASS),
    ]
    link = Link(id="body", name="wine glass", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.95}), affordances=[exact("graspable"), exact("container")])
    return _entity("wine_glass", "object", "free", [link])


def mug(radius: float = 0.042, height: float = 0.095, mass: float = 0.3, color: str = "black") -> Entity:
    """Coffee mug with a side handle along +y."""
    rgba = BLACK if color == "black" else COLORS.get(color, CERAMIC)
    geoms = [
        primitive("cylinder", [radius, height], pose([0, 0, height / 2]), rgba),
        primitive("box", [0.012, 0.022, 0.012], pose([0, radius + 0.011, height * 0.78]), rgba),
        primitive("box", [0.012, 0.022, 0.012], pose([0, radius + 0.011, height * 0.25]), rgba),
        primitive("box", [0.012, 0.012, height * 0.65], pose([0, radius + 0.022, height * 0.515]), rgba),
    ]
    link = Link(id="body", name="mug", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.9, "fragility": 0.4}), affordances=[exact("graspable"), exact("container")])
    return _entity("mug", "object", "free", [link])


def teddy_bear(scale: float = 1.0, mass: float = 0.15) -> Entity:
    """Sitting toy bear built from spheres on a flat seat (rigid in Phase 0)."""
    s = scale
    geoms = [
        primitive("cylinder", [0.045 * s, 0.02 * s], pose([0, 0, 0.01 * s]), TEDDY),
        primitive("sphere", [0.05 * s], pose([0, 0, 0.055 * s]), TEDDY),
        primitive("sphere", [0.038 * s], pose([0, 0, 0.13 * s]), TEDDY),
        primitive("sphere", [0.014 * s], pose([0, 0.028 * s, 0.163 * s]), TEDDY_DARK),
        primitive("sphere", [0.014 * s], pose([0, -0.028 * s, 0.163 * s]), TEDDY_DARK),
        primitive("sphere", [0.016 * s], pose([0.033 * s, 0, 0.125 * s]), TEDDY_DARK),
        primitive("sphere", [0.02 * s], pose([0.035 * s, 0.035 * s, 0.02 * s]), TEDDY),
        primitive("sphere", [0.02 * s], pose([0.035 * s, -0.035 * s, 0.02 * s]), TEDDY),
    ]
    link = Link(id="body", name="teddy bear", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 1.2, "fragility": 0.0}), affordances=[exact("graspable"), exact("deformable")])
    return _entity("teddy_bear", "object", "free", [link])


def scissors(length: float = 0.17, mass: float = 0.08) -> Entity:
    """Articulated scissors: two blades joined by a revolute pivot about the vertical axis."""
    blade, t = length * 0.55, 0.004
    ring_x = -length * 0.3

    def half(link_id: str, z: float, side: float) -> Link:
        geoms = [
            primitive("box", [blade, 0.012, t], pose([blade / 2, 0, z]), STEEL),
            primitive("box", [length * 0.3, 0.01, t], pose([ring_x / 2, side * 0.008, z]), STEEL),
            primitive("cylinder", [0.018, t * 1.5], pose([ring_x, side * 0.016, z]), SCISSOR_HANDLE),
        ]
        return Link(id=link_id, name=link_id, geometry={"collision": geoms}, inertial=exact({"mass": mass / 2}),
                    material=exact({"friction": 0.7, "fragility": 0.0}), affordances=[exact("graspable")])

    pivot = Joint(id="pivot", parent="blade_a", child="blade_b", type=exact("revolute"), origin=exact(pose([0, 0, 0])),
                  axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": 0.0, "hi": 0.5}), q=exact([0.0]), qd=exact([0.0]),
                  dynamics=exact({"damping": 0.05, "friction": 0.02}))
    return _entity("scissors", "object", "free", [half("blade_a", t / 2, 1.0), half("blade_b", t * 1.5 + 0.0005, -1.0)], [pivot])


def toy_airplane(length: float = 0.22, span: float = 0.20, mass: float = 0.12) -> Entity:
    """Toy airplane resting on its belly: fuselage along +x, wings across, tail fin at the back."""
    body_h = 0.045
    geoms = [
        primitive("box", [length, 0.04, body_h], pose([0, 0, body_h / 2]), TOY_YELLOW),
        primitive("box", [0.07, span, 0.008], pose([0.01, 0, body_h * 0.6]), TOY_RED),
        primitive("box", [0.04, span * 0.4, 0.006], pose([-length / 2 + 0.025, 0, body_h * 0.8]), TOY_RED),
        primitive("box", [0.035, 0.006, 0.035], pose([-length / 2 + 0.02, 0, body_h + 0.0175]), TOY_RED),
    ]
    link = Link(id="body", name="toy airplane", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.1}), affordances=[exact("graspable")])
    return _entity("toy_airplane", "object", "free", [link])


def pistol(length: float = 0.18, grip: float = 0.12, thickness: float = 0.028, mass: float = 0.6) -> Entity:
    """Pistol lying flat on its side: slide along +x, grip towards -y at the rear, trigger guard."""
    t = thickness
    geoms = [
        primitive("box", [length, 0.03, t], pose([0, 0, t / 2]), BLACK),
        primitive("box", [0.035, grip - 0.03, t], pose([-length / 2 + 0.03, -(grip - 0.03) / 2 - 0.015, t / 2]), BLACK),
        primitive("box", [0.03, 0.006, t * 0.6], pose([-length / 2 + 0.07, -0.035, t / 2]), BLACK),
    ]
    link = Link(id="body", name="pistol", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.0}), affordances=[exact("graspable")])
    return _entity("pistol", "object", "free", [link])


def pill_bottle(radius: float = 0.022, height: float = 0.07, mass: float = 0.06) -> Entity:
    """Small white pill bottle with an orange cap."""
    cap = 0.015
    geoms = [
        primitive("cylinder", [radius, height], pose([0, 0, height / 2]), PILL_WHITE),
        primitive("cylinder", [radius + 0.002, cap], pose([0, 0, height + cap / 2]), PILL_CAP),
    ]
    link = Link(id="body", name="pill bottle", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.2}), affordances=[exact("graspable"), exact("container")])
    return _entity("pill_bottle", "object", "free", [link])


def perfume_bottle(width: float = 0.065, depth: float = 0.04, height: float = 0.08, mass: float = 0.2) -> Entity:
    """Flat pink glass perfume bottle with a gold cap."""
    cap = 0.02
    geoms = [
        primitive("box", [width, depth, height], pose([0, 0, height / 2]), PERFUME_GLASS),
        primitive("cylinder", [0.012, cap], pose([0, 0, height + cap / 2]), GOLD),
    ]
    link = Link(id="body", name="perfume bottle", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.7, "fragility": 0.8}), affordances=[exact("graspable"), exact("container")])
    return _entity("perfume_bottle", "object", "free", [link])


def chair(seat: float = 0.46, seat_height: float = 0.46, back_height: float = 0.9, mass: float = 5.0) -> Entity:
    """Dining chair standing on four legs; the backrest is on the -x side, so the chair faces +x."""
    leg, t = 0.04, 0.04
    geoms = [primitive("box", [seat, seat, t], pose([0, 0, seat_height - t / 2]), LIGHT_WOOD)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            geoms.append(primitive("box", [leg, leg, seat_height - t],
                                   pose([sx * (seat / 2 - leg / 2), sy * (seat / 2 - leg / 2), (seat_height - t) / 2]), WOOD))
    back_h = back_height - seat_height
    geoms.append(primitive("box", [t, seat, back_h], pose([-seat / 2 + t / 2, 0, seat_height + back_h / 2]), WOOD))
    link = Link(id="body", name="chair", geometry={"collision": geoms}, inertial=exact({"mass": mass}),
                material=exact({"friction": 0.8, "fragility": 0.1}),
                affordances=[exact("graspable"), exact("support-surface")])
    return _entity("chair", "object", "free", [link])


def steward_bot(camera_pitch_deg: float = 30.0, camera_fovy_deg: float = 60.0) -> Entity:
    """Phase 0 mobile manipulator: planar base, vertical lift, horizontal reach, a two-axis wrist
    (yaw about the vertical, roll about the arm) and an ideal gripper.

    The base collides with the world. The lift carriage, arm and gripper are visual-only links,
    so an approaching gripper never pushes the target: Phase 0 grasping is an ideal attach
    performed by WSim when the gripper closes within capture range.
    """
    base = Link(
        id="base", name="base",
        geometry={
            "collision": [primitive("box", [0.5, 0.44, 0.28], pose([0, 0, 0.16]), ROBOT_BODY)],
            "visual": [
                primitive("box", [0.5, 0.44, 0.28], pose([0, 0, 0.16]), ROBOT_BODY),
                primitive("box", [0.05, 0.05, 1.15], pose([0.18, 0, 0.87]), STEEL),
                primitive("box", [0.05, 0.05, 1.10], pose([-0.12, 0, 0.85]), STEEL),
            ],
        },
        inertial=exact({"mass": 60.0}),
    )
    pitch = math.radians(camera_pitch_deg)
    head = Link(
        id="head", name="head",
        geometry={"visual": [primitive("box", [0.08, 0.16, 0.08], None, [0.15, 0.15, 0.15, 1.0])]},
        inertial=exact({"mass": 0.5}),
        sensors=[{
            "id": "head_camera", "kind": "rgbd", "mount": pose([0.05, 0, 0], quat_from_axis_angle([0, 1, 0], pitch)),
            "intrinsics": {"width": 320, "height": 240, "fovy": camera_fovy_deg}, "rateHz": 10,
        }],
    )
    carriage = Link(id="carriage", name="carriage",
                    geometry={"visual": [primitive("box", [0.08, 0.16, 0.05], None, STEEL)]},
                    inertial=exact({"mass": 2.0}))
    arm = Link(id="arm", name="arm",
               geometry={"visual": [primitive("box", [0.5, 0.04, 0.03], pose([-0.25, 0, 0]), ROBOT_ARM)]},
               inertial=exact({"mass": 1.5}))
    gripper = Link(id="gripper", name="gripper",
                   geometry={"visual": [
                       primitive("box", [0.06, 0.012, 0.05], pose([0.0, 0.045, 0]), [0.1, 0.1, 0.1, 1.0]),
                       primitive("box", [0.06, 0.012, 0.05], pose([0.0, -0.045, 0]), [0.1, 0.1, 0.1, 1.0]),
                       primitive("box", [0.02, 0.10, 0.03], pose([-0.03, 0, 0]), [0.1, 0.1, 0.1, 1.0]),
                   ]},
                   inertial=exact({"mass": 0.5}), affordances=[exact("grasp-point")])
    joints = [
        _fixed("head_mount", "base", "head", pose([0.16, 0, 1.50])),
        Joint(id="lift", parent="base", child="carriage", type=exact("prismatic"), origin=exact(pose([0.22, 0, 0])),
              axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": 0.30, "hi": 1.30}), q=exact([0.60]), qd=exact([0.0]),
              dynamics=exact({"damping": 250.0}),
              actuator={"kind": "position", "limits": {"effort": 1600.0, "velocity": 1.2}, "gains": {"kp": 6000.0}}),
        Joint(id="reach", parent="carriage", child="arm", type=exact("prismatic"), origin=exact(pose([0.0, 0, 0])),
              axis=exact([1.0, 0.0, 0.0]), limits=exact({"lo": 0.05, "hi": 0.70}), q=exact([0.10]), qd=exact([0.0]),
              dynamics=exact({"damping": 150.0}),
              actuator={"kind": "position", "limits": {"effort": 800.0, "velocity": 1.2}, "gains": {"kp": 2500.0}}),
        _fixed("gripper_mount", "arm", "wrist", pose([0.04, 0, 0])),
        # Both wrist axes pass through the gripper point, so turning them never moves what it holds
        # away from the gripper: yaw turns a held object about the vertical, roll tips it (pouring).
        Joint(id="wrist_yaw", parent="wrist", child="hand", type=exact("revolute"), origin=exact(pose()),
              axis=exact([0.0, 0.0, 1.0]), limits=exact({"lo": -math.pi, "hi": math.pi}), q=exact([0.0]), qd=exact([0.0]),
              dynamics=exact({"damping": 2.0}),
              actuator={"kind": "position", "limits": {"effort": 40.0, "velocity": 6.0}, "gains": {"kp": 40.0, "armature": 0.02}}),
        Joint(id="wrist_roll", parent="hand", child="gripper", type=exact("revolute"), origin=exact(pose()),
              axis=exact([1.0, 0.0, 0.0]), limits=exact({"lo": -3.2, "hi": 3.2}), q=exact([0.0]), qd=exact([0.0]),
              dynamics=exact({"damping": 2.0}),
              actuator={"kind": "position", "limits": {"effort": 40.0, "velocity": 6.0}, "gains": {"kp": 40.0, "armature": 0.02}}),
    ]
    root_actuator = {
        "kind": "position",
        "limits": {"effort": [3600.0, 3600.0, 1000.0], "velocity": [2.0, 2.0, 4.0]},
        "gains": {"kp": [8000.0, 8000.0, 900.0], "damping": [1500.0, 1500.0, 130.0]},
    }
    wrist = Link(id="wrist", name="wrist", geometry={"visual": [primitive("cylinder", [0.02, 0.03], None, STEEL)]},
                 inertial=exact({"mass": 0.1}))
    hand = Link(id="hand", name="hand", geometry={"visual": [primitive("box", [0.03, 0.03, 0.03], None, STEEL)]},
                inertial=exact({"mass": 0.1}))
    return _entity("steward-bot", "robot", "planar", [base, head, carriage, arm, wrist, hand, gripper], joints,
                   state_vars={"gripper": exact("open")}, root_actuator=root_actuator)


GENERATORS: dict[str, Callable[..., Entity]] = {
    "building": building,
    "counter": counter,
    "sideboard": sideboard,
    "dining_table": dining_table,
    "cabinet_with_drawer": cabinet_with_drawer,
    "cup": cup,
    "plate": plate,
    "bowl": bowl,
    "nesting_bowl": nesting_bowl,
    "wine_glass": wine_glass,
    "mug": mug,
    "teddy_bear": teddy_bear,
    "scissors": scissors,
    "toy_airplane": toy_airplane,
    "pistol": pistol,
    "pill_bottle": pill_bottle,
    "perfume_bottle": perfume_bottle,
    "chair": chair,
    "printed_board": printed_board,
    "open_box": open_box,
    "steward_bot": steward_bot,
}
