"""ESim (Phase 0): consequence prediction over the Ego-State, producing Prediction Receipts.

Phase 0 ESim is geometric and probabilistic rather than a physics rollout: it plans the exact
motion the skill would execute, checks the robot footprint and path against ES obstacles, and
converts ES uncertainty into a risk score (for example the probability that the gripper misses
a target whose position is uncertain). The receipt carries that plan; running a receipt executes
the predicted plan, not a fresh one. Physics rollouts with belief sampling arrive in Phase 2.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from steward.ego.arm import Arm, slide_arm
from steward.math3d import quat_from_yaw, wrap_angle, matrix_from_quat
from steward.ego.planner import (
    FLOOR_LYING_M,
    Box2D,
    astar,
    build_grid,
    entity_box,
    floor_obstacles,
    path_clearance,
    front_box,
    path_length,
    room_centre,
    protrusion,
    rects_overlap,
    escape_prefix,
)
from steward.math3d import yaw_from_quat
from steward.scene.schema import Entity, SceneState


@dataclass
class EsimConfig:
    capture_radius: float = 0.05
    hole_fit_tolerance_m: float = 0.003  # sensor-fit tolerance, not a task acceptance criterion
    hole_assignment_limit: int = 4096  # bound work; ask for explicit feature selection above it
    # Where the drawer skill closes the gripper again when the planned handle point held nothing:
    # reach offsets along the arm, then base steps across it (m). A swollen drawer front puts the
    # estimate past or short of the real handle, and a miss only costs the next closing.
    handle_probes_along_m: tuple[float, ...] = (0.04, 0.08, 0.12, -0.05, -0.09)
    handle_probes_side_m: tuple[float, ...] = (0.06, -0.06)
    # The grasp point passes this far above the rim of what it reaches into (a drawer's front, a
    # bowl's wall), and what it carries out clears the rim by as much: the arm comes in over the wall
    # and goes straight down, instead of sweeping through it.
    rim_clearance: float = 0.05
    edge_grasp_min_m: float = 0.25  # an object this large may be taken by an edge when its centre is out of reach
    # How far inside its extent an edge grasp closes. Fingers close on an edge that lies between
    # them: from a gripper's width (3 cm) inside it to a finger's thickness (1.5 cm) outside. The
    # grasp aims at the middle of that window, where an estimate off by as much either way still
    # takes the edge (aimed deeper, a box was missed or a piece beside its wall was taken instead).
    edge_grasp_inset: float = 0.008
    # How far below its top an edge grasp of a container closes: on the rim, not halfway down the
    # wall beside what lies in it (there the fingers meet a piece by the wall before the wall).
    edge_grasp_depth: float = 0.008
    # Take a large object by its edge even when its centre is in reach: held so, it hangs ahead of
    # the gripper and can be set down (or poured) deeper on a surface than an arm of short reach gets.
    edge_grasp_first: bool = False
    # Turned over (place with ``flip``), what a gripper pinches from above ends with the tool
    # pointing up. An arm whose tool cannot point up at the surface turns it over above the spot,
    # its tool tilted forward so that it points up only so far, and lowers it as far as it reaches
    # turned over (Arm.turn_over), the rest of the way it drops flat. It needs the object taken
    # tilted at least ``flip_pitch_min`` (pick with ``flip``), turns it ``flip_turn_above`` above
    # where it is set down and lets it drop at most ``flip_drop_max``.
    flip_in_air: bool = False
    flip_pitch_min: float = 1.0
    flip_turn_above: float = 0.2
    flip_drop_max: float = 0.2
    # Below this TCP height a deep dock has no tool-down IK under the USD mounts; grasps raise into
    # it, and docks that must tilt stretch until ``pitch_for`` returns at least ``min_require_tilt``.
    elevated_grasp_floor: float = 0.72
    # Least wrist pitch ``require_tilt`` accepts: a shallower ridge (≈0.12 rad) often has a nearby
    # mink posture the servo holds instead, so the virtual pitch never settles on the plan.
    min_require_tilt: float = 0.35
    # Carrying or placing a most-tilted grasp below this height sits in the short-reach USD hole.
    tilted_carry_floor: float = 0.95
    receipt_ttl: float = 60.0
    dock_gap: float = 0.07
    predock_backoff: float = 0.35
    dock_backoff_step: float = 0.05  # how much farther back to try when a dock pose collides
    reach_min: float = 0.08
    reach_max: float = 0.65
    lift_min: float = 0.0  # the gripper reaches down to the floor
    lift_max: float = 1.28
    place_edge_margin: float = 0.10
    place_clearance: float = 0.18
    spot_shift_max: float = 0.3  # an occupied requested spot moves to the nearest free one at most this far
    spot_shift_step: float = 0.01
    nav_margin: float = 0.03
    floor_furniture_height: float = 0.35
    handle_reach: float = 0.18
    drawer_pull: float = 0.30
    drawer_pull_max: float = 0.33
    # The head camera looks 30° down and its lower edge is 60° down. At 0.25 m the open
    # drawer's floor is below that edge, so a flat pistol has no pixels. 0.55 m puts the floor
    # near 45° down, inside the frame.
    drawer_viewing_backoff: float = 0.55
    nest_fit: float = 0.92
    floor_carry_lift: float = 0.05
    floor_lift_max_m: float = 0.3  # an object off the floor no larger than this is lifted clear of the base to carry
    floor_release_gap: float = 0.015
    carry_retreat_max: float = 1.2
    park_keep_clear: float = 0.9
    park_spacing: float = 0.5
    # Belief-based risks: a pick is attempted rather than refused when the estimator's picture is
    # incomplete, and the mission layer's risk limit (0.35) decides whether to look closer first.
    blind_support_risk: float = 0.15  # no known surface under the target; the gripper checks the grasp
    phantom_pick_cost: float = 0.5  # share of a target's non-existence that counts as pick risk
    nesting_risk_each: float = 0.15  # pick risk per thing the estimator reads as inside the target
    nesting_risk_cap: float = 0.35
    graspable_max_m: float = 0.35  # a raised thing this small is tableware whatever the classifier says
    miscategorised_risk: float = 0.15
    wrist_yaw_max: float = 3.1
    wrist_roll_max: float = math.pi  # a half turn for flip; slide_arm also enforces the model's joint stops
    pour_roll: float = 1.9
    # Tipped less, the contents stay in the container: its walls hold them until it is past upright
    # (measured with the puzzle box: at 1.5 rad nothing came out, at 1.75 everything did).
    pour_roll_min: float = 1.7
    pour_check_step: float = 0.1  # roll step at which the whole tip is checked for a continuous motion
    wrist_check_step: float = 0.1  # angular step for a held object's approach, including yaw and roll
    lift_check_step: float = 0.05  # height step at which a grasp's way down and up is checked likewise
    dock_reach_margin: float = 0.04  # a dock lands this far off its pose, and the reach with it
    retract_floor: float = 0.10  # the nearest a tilted tool is checked to come back to (a slide retracts to 0.10)
    retract_step: float = 0.05
    pour_clearance: float = 0.02
    # Poured contents slide up to this many times their largest side beyond the container (measured:
    # 5.75 cm puzzle pieces poured from a box at the rim of a table slid 0.11 m beyond it); the spot
    # keeps that from every edge.
    pour_spill_sizes: float = 2.0
    pour_hold_s: float = 1.5
    # Tipped out, the container rises this far before it turns upright again: the contents pile up
    # under its lower edge, and the last pieces rest on that pile against its wall, where the wall
    # turning back would scoop them in again.
    pour_rise: float = 0.06


class PredictionError(ValueError):
    pass


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Esim:
    def __init__(self, robot_model: dict, config: EsimConfig | None = None, arm: Arm | None = None):
        """``arm`` is the body's planar arm (:mod:`steward.ego.arm`); without one the model is the
        mobile kitchen robot's. Reach and lift use ``config``; wrist caps also respect the model's
        declared joint stops."""
        self.config = config or EsimConfig()
        tree = robot_model["body"]
        base = next(link for link in tree["links"] if link["id"] == "base")
        base_geom = base["geometry"]["collision"][0]
        bx, by, bz = base_geom["size"]
        self.half_x, self.half_y = bx / 2, by / 2
        self.ground_clearance = base_geom["pose"]["p"][2] - bz / 2
        self.radius = math.hypot(self.half_x, self.half_y)
        cfg = self.config
        self.arm = arm or slide_arm(robot_model, (cfg.reach_min, cfg.reach_max), (cfg.lift_min, cfg.lift_max),
                                    cfg.wrist_yaw_max, cfg.wrist_roll_max)
        self.reach_min, self.reach_max = self.arm.geometry.reach_min, self.arm.geometry.reach_max
        self.lift_min, self.lift_max = self.arm.geometry.lift_min, self.arm.geometry.lift_max
        self.tool_offset, self.lateral = self.arm.geometry.tool_offset, self.arm.geometry.lateral
        self._ids = itertools.count(1)
        self._blockers: set[str] = set()

    # ---------------------------------------------------------------- helpers
    def _self(self, es: SceneState) -> Entity:
        return es.entity(es.self_id)

    def _arm_joints(self, es: SceneState) -> dict:
        """The planar arm's joints (lift, reach, wrist yaw and roll) for the self entity's encoders."""
        me = self._self(es)
        joints = {j.id: j.q.value[0] for j in me.rigid.joints if j.q.value}
        x, y, yaw = self._robot_xy_yaw(es)
        return self.arm.joints_of(joints, {"x": x, "y": y, "yaw": yaw})

    def _robot_xy_yaw(self, es: SceneState) -> tuple[float, float, float]:
        me = self._self(es)
        p = me.pose.value
        return p["p"][0], p["p"][1], yaw_from_quat(p["q"])

    def _held(self, es: SceneState) -> Entity | None:
        for rel in es.relations:
            if rel.type == "attached" and rel.b.get("entity") == es.self_id:
                return es.find(rel.a["entity"])
        return None

    @staticmethod
    def _sigma_xy(ent: Entity) -> float:
        cov = (ent.pose.unc or {}).get("cov")
        if not cov:
            return 0.0
        return math.sqrt((cov[0] + cov[7]) / 2)

    @staticmethod
    def _affords(ent: Entity, name: str) -> bool:
        return any(a.value == name for a in ent.rigid.links[0].affordances)

    @staticmethod
    def _screw_holes(ent: Entity) -> list[dict]:
        """``{id, p}`` screw-hole markers on ``ent``'s first link, from structured affordances."""
        out = []
        for aff in ent.rigid.links[0].affordances:
            value = aff.value
            if isinstance(value, dict) and "screw-hole" in value:
                hole = value["screw-hole"]
                out.append({**hole, "id": str(hole["id"]), "p": [float(v) for v in hole["p"][:3]],
                            "source": aff.src or "unknown",
                            "frame": hole.get("frame", "entity" if aff.src == "observed" else "unregistered")})
        return out

    def _anchor(self, es: SceneState, support: Entity) -> Entity:
        """The floor-standing furniture to dock at for a support: a bowl on a sideboard docks at the
        sideboard; a drawer (raised, held by no surface) docks at its own front."""
        seen, cur = set(), support
        while cur.id not in seen:
            seen.add(cur.id)
            _, bottom, _ = entity_box(cur)
            below = self._support_of(es, cur)
            if bottom < self.config.floor_furniture_height or below is None:
                return cur
            cur = below
        return support

    def _contents(self, es: SceneState, holder: Entity, exclude: str | None = None) -> list[Entity]:
        out = []
        for rel in es.relations:
            if rel.type == "supported-by" and rel.b.get("entity") == holder.id and rel.a["entity"] != exclude:
                ent = es.find(rel.a["entity"])
                if ent is not None:
                    out.append(ent)
        return out

    @staticmethod
    def _relation_confidence(es: SceneState, inner: str, holder: str) -> float:
        """How sure the estimator is that ``inner`` rests in or on ``holder`` (1.0 when unstated)."""
        for rel in es.relations:
            if rel.type == "supported-by" and rel.a.get("entity") == inner and rel.b.get("entity") == holder:
                return float(rel.confidence if rel.confidence is not None else 1.0)
        return 1.0

    def _covering(self, es: SceneState, target: Entity) -> list[str]:
        """Objects closing ``target`` from above (a lid on a box), judged by geometry: their base is
        level with its top and their footprint, overlapping it, is at least half as large (smaller
        objects seen near the rim lie in it)."""
        tbox, _, top = entity_box(target)
        out = []
        for ent in es.entities:
            if ent.id in (target.id, es.self_id) or ent.role == "environment":
                continue
            box, bottom, _ = entity_box(ent)
            if top - 0.015 <= bottom <= top + 0.03 and rects_overlap(box, tbox) \
                    and box.hx * box.hy >= 0.5 * tbox.hx * tbox.hy:
                out.append(ent.id)
        return out

    def _front(self, es: SceneState, ent: Entity) -> Box2D:
        """The entity's footprint with +x pointing out of its front (the face looking into the room)."""
        boxes = [entity_box(e)[0] for e in es.entities if e.id != es.self_id]
        return front_box(entity_box(ent)[0], room_centre(boxes))

    def _drawer_furniture(self, es: SceneState, drawer: Entity) -> Entity | None:
        """The sideboard or cabinet the drawer sits in, or None when it is still an orphan box."""
        centre = entity_box(drawer)[0]
        found, best = None, math.inf
        for ent in es.entities:
            if ent.category.value not in ("sideboard", "cabinet"):
                continue
            box = entity_box(ent)[0]
            if not box.contains(centre.cx, centre.cy, inflate=0.20):
                continue
            gap = math.hypot(box.cx - centre.cx, box.cy - centre.cy)
            if gap < best:
                found, best = ent, gap
        return found

    def _opening(self, es: SceneState, drawer: Entity) -> float:
        furniture = [entity_box(e)[0] for e in es.entities
                     if e.id not in (drawer.id, es.self_id) and self._affords(e, "support-surface")]
        return protrusion(self._front(es, drawer), furniture)

    def _support_of(self, es: SceneState, ent: Entity) -> Entity | None:
        for rel in es.relations:
            if rel.type == "supported-by" and rel.a.get("entity") == ent.id:
                return es.find(rel.b["entity"])
        return None

    def _receipt(self, es: SceneState, skill: str, params: dict, risk: float, issues: list[str], plan: dict,
                 feasible: bool = True) -> dict:
        warnings = list(plan.get('targetEvidence', {}).get('warnings', []))
        requested = {params.get(k) for k in ('target', 'body', 'part', 'on')}
        for entity in es.entities:
            orientation = entity.state_vars.get('orientation')
            info = (orientation.value or {}) if orientation else {}
            registration = info.get('holeRegistration') or {}
            warnings.extend(dict(w, entity=entity.id) for w in registration.get('warnings', []))
            holes = self._screw_holes(entity)
            if any(h.get('frame') != 'entity' for h in holes):
                warnings.append(dict(reason='unregistered_interface', entity=entity.id, featureCount=len(holes)))
            if info.get('posture', {}).get('status') == 'inconsistent':
                warnings.append(dict(reason='suspect_posture', entity=entity.id,
                    posture=info['posture'], extentXM=entity_box(entity)[0].hx*2))
            if entity.id in requested and entity.pose.t is not None and es.stamp-entity.pose.t > 20:
                warnings.append(dict(reason='stale_pose_estimate', entity=entity.id, ageS=es.stamp-entity.pose.t))
        return {
            "id": f"rcpt-{next(self._ids)}",
            "skill": skill,
            "params": params,
            "esVersion": es.version,
            "createdAt": es.stamp,
            "expiresAt": es.stamp + self.config.receipt_ttl,
            "risk": round(float(min(1.0, max(0.0, risk))), 4),
            "feasible": feasible,
            "issues": issues,
            "warnings": warnings,
            "plan": plan,
        }

    # -------------------------------------------------------------- approach
    def _approach(self, es: SceneState, support: Entity,
                  target_xy: tuple[float, float] | Callable[[float], tuple[float, float]], height: float,
                  exclude: set[str], load: Entity | None = None,
                  heading_issue: Callable[[float], str | None] | None = None,
                  pitches: Sequence[float | None] = (0.0,), lifts: Sequence[float] | None = None,
                  wrist: Callable[[float], tuple[float, float]] | None = None,
                  approach_path: Callable[[float, float, float], bool] | None = None,
                  require_tilt: bool = False) -> dict:
        """Choose the support side to dock at; return path, dock pose and reach, or raise.

        ``target_xy`` is where the gripper goes, or a function of the docked base heading giving it
        (a held object taken off-centre hangs off the gripper in a direction that turns with the
        base). With ``load`` (a floor-standing object about to be picked) a side is only accepted
        when the robot can still turn at the pre-dock pose while carrying the load in front of it,
        so a pick never leaves the robot boxed in. ``heading_issue`` rejects a side by the base
        heading docked there (a wrist turn beyond its travel), with the reason it returns. The arm
        must hold its gripper there at one of ``pitches`` (tried in order; None: the least tilt the
        arm needs there), moving it through the heights ``lifts`` in turn (default: ``height``
        alone) without turning over; a pitch other than 0 is returned as ``wristPitch``.
        ``wrist`` supplies tool yaw and roll for each dock heading. ``approach_path`` checks the
        motion to that heading and reach via a candidate retracted reach, from the held posture."""
        sbox, _, _ = entity_box(support)
        rx, ry, _ = self._robot_xy_yaw(es)
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)
        candidates = []
        for normal in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
            heading = math.atan2(-(s * normal[0] + c * normal[1]), -(c * normal[0] - s * normal[1]))
            txy = target_xy(heading) if callable(target_xy) else target_xy
            dx, dy = txy[0] - sbox.cx, txy[1] - sbox.cy
            tx, ty = c * dx + s * dy, -s * dx + c * dy
            if normal[0]:
                candidates.append(((normal[0] * sbox.hx, ty), normal, sbox.hx - normal[0] * tx, txy))
            else:
                candidates.append(((tx, normal[1] * sbox.hy), normal, sbox.hy - normal[1] * ty, txy))
        target_xy = candidates[0][3]
        cfg = self.config
        obstacles_all = self._floor_obstacles(es, exclude)
        # A floor-standing load stays an obstacle for the path to it (only its own back-out ignores it).
        loaded = self._floor_obstacles(es, exclude - {load.id}) if load is not None else None
        if loaded is not None:
            obstacles_all, load_free = loaded, obstacles_all
        carry = self._carry(es)
        grid = build_grid(obstacles_all, self._sweep(carry) + cfg.nav_margin, [(rx, ry), target_xy])
        best, reasons = None, []
        if not (self.lift_min <= height <= self.lift_max):
            raise PredictionError(f"height {height:.2f} m is outside the lift range [{self.lift_min}, {self.lift_max}]")
        for edge, normal, depth, _ in candidates:
            # A target nearer the edge than the arm's shortest reach is worked from farther back.
            standoff = max(self.half_x + cfg.dock_gap, self.reach_min + self.tool_offset - depth)
            reach = standoff + depth - self.tool_offset
            if not (self.reach_min <= reach <= self.reach_max):
                reasons.append(f"side {normal}: reach {reach:.2f} m out of range")
                continue
            nx, ny = c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]
            yaw = math.atan2(-ny, -nx)
            # The arm works on its own line, ``lateral`` left of the base centre: the base stands
            # that far to the right of the target.
            ex = sbox.cx + c * edge[0] - s * edge[1] + math.sin(yaw) * self.lateral
            ey = sbox.cy + s * edge[0] + c * edge[1] - math.cos(yaw) * self.lateral
            issue = heading_issue(yaw) if heading_issue is not None else None
            if issue:
                reasons.append(f"side {normal}: {issue}")
                continue
            # Where the nearest dock pose collides, stand farther back and reach further.
            while True:
                dock = (ex + nx * standoff, ey + ny * standoff)
                blocked = self._footprint_blocked(obstacles_all, dock, yaw, carry)
                farther = standoff + cfg.dock_backoff_step
                if not blocked or farther + depth - self.tool_offset > self.reach_max:
                    break
                standoff, reach = farther, farther + depth - self.tool_offset
            predock = (dock[0] + nx * cfg.predock_backoff, dock[1] + ny * cfg.predock_backoff)
            if blocked:
                reasons.append(f"side {normal}: dock pose collides with {blocked}")
                continue
            wrist_yaw, wrist_roll = wrist(yaw) if wrist is not None else (0.0, 0.0)
            # Deep-drawer docks must leave the tool-down envelope with enough tilt: stretch until
            # pitch_for ≥ min_tilt and ``reachable`` accepts the posture (pitch_for alone can land
            # on a limit-packed joint2 that the servo never holds).
            if require_tilt and any(p is None or (p is not None and p > 0.0) for p in pitches):
                min_tilt = cfg.min_require_tilt
                # Cap near 0.50 m: deeper docks often cannot close at the object height.
                tilt_reach_cap = min(self.reach_max, 0.50)
                while reach < tilt_reach_cap:
                    trial = self.arm.pitch_for(height, reach, wrist_yaw, wrist_roll)
                    if (trial is not None and trial >= min_tilt
                            and self.arm.reachable(height, reach, wrist_yaw, wrist_roll, trial)):
                        break
                    farther = standoff + cfg.dock_backoff_step
                    new_reach = farther + depth - self.tool_offset
                    if new_reach > tilt_reach_cap:
                        break
                    new_trial = self.arm.pitch_for(height, new_reach, wrist_yaw, wrist_roll)
                    if (new_trial is None
                            or not self.arm.reachable(height, new_reach, wrist_yaw, wrist_roll, new_trial)):
                        break
                    standoff, reach = farther, new_reach
                    dock = (ex + nx * standoff, ey + ny * standoff)
                    if self._footprint_blocked(obstacles_all, dock, yaw, carry):
                        break
                    predock = (dock[0] + nx * cfg.predock_backoff, dock[1] + ny * cfg.predock_backoff)
            pitch = self._grasp_pitch(reach, height, lifts or [height], pitches, wrist_yaw, wrist_roll)
            # ``_grasp_pitch`` may still pick the shallow least tilt; bump to a decisive one the
            # stretched dock was sized for, otherwise the servo parks on a neighbouring posture.
            if require_tilt and pitch is not None and pitch < cfg.min_require_tilt:
                lifts_here = lifts or [height]
                for candidate in (cfg.min_require_tilt, 0.45, 0.55, self.arm.geometry.wrist_pitch_max):
                    if candidate > pitch and self._holds_there(
                            reach, height, lifts_here, candidate, cfg.dock_reach_margin, wrist_yaw, wrist_roll):
                        pitch = candidate
                        break
            if pitch is None:
                reasons.append(f"side {normal}: the arm cannot hold its gripper {reach:.2f} m out at {height:.2f} m "
                               f"(yaw {wrist_yaw:.2f}, roll {wrist_roll:.2f})")
                continue
            accept = (lambda near: approach_path(yaw, reach, near)) if approach_path is not None else None
            retract = (self._retract_reach(reach, (lifts or [height])[-1], pitch, wrist_yaw, wrist_roll, accept)
                       if pitch or approach_path is not None else None)
            if (pitch or approach_path is not None) and retract is None:
                reasons.append(f"side {normal}: no continuous arm approach and retraction at yaw {wrist_yaw:.2f}, "
                               f"roll {wrist_roll:.2f}, pitch {pitch:.2f}; try another placement or heading")
                continue
            ryaw = self._robot_xy_yaw(es)[2]
            reuse_dock = (abs(wrap_angle(ryaw-yaw)) < .02
                and math.hypot(dock[0]-rx, dock[1]-ry) <= cfg.predock_backoff
                and all(self._footprint_blocked(obstacles_all,
                    (rx+(dock[0]-rx)*t, ry+(dock[1]-ry)*t), yaw, carry) is None
                    for t in np.linspace(0., 1., max(2, math.ceil(math.hypot(dock[0]-rx,dock[1]-ry)/.01)+1))))
            blocked = None if reuse_dock else self._turn_blocked(obstacles_all, predock, self._sweep(carry))
            if blocked:
                reasons.append(f"side {normal}: no room to turn at the pre-dock pose ({blocked})")
                continue
            retreat = None
            if load is not None:
                lbox, _, _ = entity_box(load)
                # The load already stands where it is: check the back-out with its physical size (the ES
                # box carries a 2-sigma margin) less a small tolerance for sliding past a neighbour.
                sigma = self._sigma_xy(load)
                # Sliding past a neighbour may use a small tolerance; the turning room must be judged
                # with the same box the carry predictions will use after the grasp.
                slide = (self.tool_offset + reach, self.lateral, lbox.yaw - yaw, max(lbox.hx - sigma - 0.02, 0.05),
                         max(lbox.hy - sigma - 0.02, 0.05))
                turn = (self.tool_offset + reach, self.lateral, lbox.yaw - yaw, max(lbox.hx - sigma, 0.05),
                        max(lbox.hy - sigma, 0.05))
                retreat = self._retreat(load_free, dock, (nx, ny), yaw, slide, cfg.predock_backoff, turn)
                if retreat is None:
                    reasons.append(f"side {normal}: carrying {load.id} from here the robot could not back out to "
                                   "a spot with room to turn")
                    continue
            path = [[rx,ry]] if reuse_dock else self._navigation_path(es, obstacles_all, predock, carry, grid)
            if path is None:
                reasons.append(f"side {normal}: no collision-free path to the pre-dock pose")
                continue
            length = path_length(path)
            if best is None or length < best["pathLength"]:
                best = {"path": [list(p) for p in path], "predock": list(predock), "dock": list(dock), "yaw": yaw,
                        "reach": reach, "pathLength": length}
                if pitch:
                    best["wristPitch"] = pitch
                if retract is not None:
                    best["retractReach"] = retract
                if len(path) > 1:
                    # Standing at the pre-dock pose already, there is no path to keep clear.
                    best["clearance"] = path_clearance(path, obstacles_all)
                if retreat is not None:
                    best["retreat"] = list(retreat)
        if best is None:
            hint = self._boxed_in(es, obstacles_all, carry)
            raise PredictionError("no feasible approach: " + "; ".join(reasons) + (f"; {hint}" if hint else ""))
        return best

    def _retract_reach(self, reach: float, lift: float, pitch: float, yaw: float = 0.0, roll: float = 0.0,
                       accept: Callable[[float], bool] | None = None) -> float | None:
        """The nearest checked retracted reach at ``lift`` and tool yaw, roll and pitch. The search
        stops at ``retract_floor`` or the arm's minimum reach, and may keep ``reach`` unchanged;
        None when no continuous retraction and optional approach ``accept`` are feasible."""
        cfg = self.config
        near = min(reach, max(cfg.retract_floor, self.reach_min))
        while near <= reach:
            steps = max(1, math.ceil((reach - near) / cfg.lift_check_step))
            if self.arm.reachable_path([{"lift": lift, "reach": reach - (reach - near) * k / steps,
                                         "wrist_yaw": yaw, "wrist_roll": roll, "wrist_pitch": pitch}
                                        for k in range(steps + 1)]) and (accept is None or accept(near)):
                return near
            if near == reach:
                break
            near = min(reach, near + cfg.retract_step)
        return None

    def _grasp_pitch(self, reach: float, height: float, lifts: Sequence[float],
                     candidates: Sequence[float | None], yaw: float = 0.0, roll: float = 0.0) -> float | None:
        """The first tool tilt of ``candidates`` (None: the least the arm needs there) at which
        :meth:`_holds_there`; best one that also holds the gripper as far off as a dock lands."""
        for margin in (self.config.dock_reach_margin, 0.0):
            for wanted in candidates:
                if wanted is None:
                    # Cover a dock that lands short or long: take the most tilt needed across the
                    # margin band so a centimetre of base error does not outrun the planned pitch.
                    needed = [self.arm.pitch_for(height, reach + d, yaw, roll)
                              for d in (-margin, 0.0, margin, margin + 0.02) if reach + d > 0]
                    usable = [p for p in needed if p is not None]
                    pitch = max(usable) if usable else None
                else:
                    pitch = wanted
                if pitch is not None and self._holds_there(reach, height, lifts, pitch, margin, yaw, roll):
                    return pitch
        return None

    def _holds_there(self, reach: float, height: float, lifts: Sequence[float], pitch: float, margin: float,
                     yaw: float = 0.0, roll: float = 0.0) -> bool:
        """Whether the arm, its tool tilted ``pitch``, moves its gripper through ``lifts`` at ``reach``
        without turning over, and holds it at ``height`` ``margin`` nearer and farther too."""
        # Include the exact reach: ±margin alone can accept a posture whose centre sits in an IK hole.
        return (all(self.arm.reachable(height, r, yaw, roll, pitch)
                    for r in {reach - margin, reach, reach + margin} if r > 0)
                and self.arm.reachable_path([{"lift": h, "reach": reach, "wrist_pitch": pitch,
                                             "wrist_yaw": yaw, "wrist_roll": roll} for h in self._lift_steps(lifts)]))

    def _lift_steps(self, lifts: Sequence[float]) -> list[float]:
        """The heights ``lifts`` in turn, with steps of at most ``lift_check_step`` between them."""
        out = [lifts[0]]
        for h in lifts[1:]:
            n = max(1, math.ceil(abs(h - out[-1]) / self.config.lift_check_step))
            out += [out[-1] + (h - out[-1]) * k / n for k in range(1, n + 1)]
        return out

    @staticmethod
    def _edge_facing(offset: tuple[float, float]) -> Callable[[float], str | None] | None:
        """For an edge grasp (``offset`` from the object's centre), a ``heading_issue`` refusing the
        sides that edge does not face: taken by the edge nearest it, the object hangs ahead of the
        gripper and can be set down or poured deeper than the arm reaches, while one taken by a
        side edge hangs beside it. Used with ``edge_grasp_first`` (an arm of short reach)."""
        if not any(offset):
            return None
        length = math.hypot(*offset)

        def issue(heading: float) -> str | None:
            facing = -(math.cos(heading) * offset[0] + math.sin(heading) * offset[1]) / length
            return None if facing > math.cos(math.pi / 4) else "the edge taken does not face this side"
        return issue

    def _grasp_points(self, es: SceneState, target: Entity, tp: list[float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Where a pick may close on ``target``: ``(xy, offset from its centre)``, its centre first,
        then -- for an object ``edge_grasp_min_m`` or larger -- the middle of each side,
        ``edge_grasp_inset`` inside its extent, the side nearest the robot first; with
        ``edge_grasp_first`` the sides come before the centre. A pourable container (an open box,
        carried with what lies in it) is taken by its rim only, whatever its size: closing over its
        middle the fingers meet its contents."""
        cfg = self.config
        points = [((tp[0], tp[1]), (0.0, 0.0))]
        box, _, _ = entity_box(target)
        pourable = self._affords(target, "pourable")
        if 2 * max(box.hx, box.hy) < cfg.edge_grasp_min_m and not pourable:
            return points
        rx, ry, _ = self._robot_xy_yaw(es)
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        edges = []
        for lx, ly in ((box.hx, 0.0), (-box.hx, 0.0), (0.0, box.hy), (0.0, -box.hy)):
            scale = max(math.hypot(lx, ly) - cfg.edge_grasp_inset, 0.0) / math.hypot(lx, ly)
            dx, dy = (c * lx - s * ly) * scale, (s * lx + c * ly) * scale
            edges.append(((tp[0] + dx, tp[1] + dy), (dx, dy)))
        edges.sort(key=lambda e: math.hypot(e[0][0] - rx, e[0][1] - ry))
        if pourable:
            return edges
        return edges + points if cfg.edge_grasp_first else points + edges

    def _turn_blocked(self, obstacles: list[Box2D], xy: tuple[float, float], radius: float | None = None) -> str | None:
        """The robot turns in place at the pre-dock pose, sweeping its circumscribed circle (or the
        larger circle swept by an object it carries low)."""
        r = self.radius if radius is None else radius
        for box in obstacles:
            if box.distance(xy[0], xy[1]) < r + self.config.nav_margin:
                self._blockers.add(box.id)
                return box.id
        return None

    def _footprint_blocked(self, obstacles: list[Box2D], xy: tuple[float, float], yaw: float,
                           carry: tuple | None = None) -> str | None:
        me = Box2D("robot", xy[0], xy[1], yaw, self.half_x + 0.01, self.half_y + 0.01)
        held = self._carried_box(carry, xy, yaw) if carry is not None else None
        for box in obstacles:
            if rects_overlap(me, box) or (held is not None and rects_overlap(held, box)):
                self._blockers.add(box.id)
                return box.id
        return None

    # ---------------------------------------------------------------- blockers
    def _describe_blockers(self, es: SceneState) -> list[dict]:
        """Entities that made the prediction infeasible. Floor-standing graspable ones (a chair) are
        movable and come with parking spots out of everyone's way; other things lying on the floor
        are reported as blockers only (a flat thing there may be a mat read as tableware)."""
        held = self._held(es)
        out = []
        for eid in sorted(self._blockers):
            ent = es.find(eid)
            if ent is None or ent.role == "environment" or ent.id == es.self_id:
                continue
            _, bottom, _ = entity_box(ent)
            movable = (self._affords(ent, "graspable") and self._affords(ent, "floor-standing") and bottom < FLOOR_LYING_M
                       and (held is None or held.id != ent.id))
            out.append({"id": ent.id, "category": ent.category.value, "movable": movable,
                        "parkAt": self._parking_spots(es, ent) if movable else []})
        return out

    def _parking_spots(self, es: SceneState, obj: Entity, count: int = 4) -> list[list[float]]:
        """Free floor spots for a movable object: inside the room, at least ``park_keep_clear`` from
        furniture (the robot needs that room to work there), preferring spots along a wall."""
        cfg = self.config
        walls = [entity_box(e)[0] for e in es.entities if e.role == "environment"]
        if not walls:
            return []
        x_lo, x_hi = min(w.cx for w in walls) + 0.06, max(w.cx for w in walls) - 0.06
        y_lo, y_hi = min(w.cy for w in walls) + 0.06, max(w.cy for w in walls) - 0.06
        box, _, _ = entity_box(obj)
        shrink = self._sigma_xy(obj)
        hx, hy = max(box.hx - shrink, 0.05), max(box.hy - shrink, 0.05)
        reach = math.hypot(hx, hy)
        others = []
        for ent in es.entities:
            if ent.id in (obj.id, es.self_id) or ent.role == "environment" or ent.existence < 0.3:
                continue
            ebox, bottom, _ = entity_box(ent)
            if bottom < cfg.floor_furniture_height:
                furniture = self._affords(ent, "support-surface") or self._affords(ent, "pullable")
                others.append((ebox, cfg.park_keep_clear if furniture else 0.3))
        candidates = []
        for x in np.arange(x_lo + reach + 0.05, x_hi - reach - 0.05 + 1e-9, 0.1):
            for y in np.arange(y_lo + reach + 0.05, y_hi - reach - 0.05 + 1e-9, 0.1):
                if all(o.distance(x, y) - reach >= clear for o, clear in others):
                    wall = min(x - x_lo, x_hi - x, y - y_lo, y_hi - y) - reach
                    near = math.hypot(x - box.cx, y - box.cy)
                    candidates.append((wall + 0.1 * near, float(x), float(y)))
        spots: list[list[float]] = []
        for _, x, y in sorted(candidates):
            if all(math.hypot(x - sx, y - sy) >= cfg.park_spacing for sx, sy in spots):
                spots.append([round(x, 2), round(y, 2)])
            if len(spots) == count:
                break
        return spots

    # ---------------------------------------------------------------- carrying
    def _held_relative(self, es: SceneState) -> tuple | None:
        """Held object's footprint in the robot frame: (x, y, yaw, half x, half y)."""
        held = self._held(es)
        if held is None:
            return None
        box, _, _ = entity_box(held)
        rx, ry, ryaw = self._robot_xy_yaw(es)
        dx, dy = box.cx - rx, box.cy - ry
        c, s = math.cos(-ryaw), math.sin(-ryaw)
        return (c * dx - s * dy, s * dx + c * dy, box.yaw - ryaw, box.hx, box.hy)

    def _hold_dz(self, es: SceneState, held: Entity) -> float:
        """How far above the held object's centre the gripper holds it (a box taken by its rim): the
        gripper goes that much higher for the object to land where it is set down."""
        lift = self._arm_joints(es).get("lift")
        return 0.0 if lift is None else float(lift) - float(held.pose.value["p"][2])

    def _carry(self, es: SceneState) -> tuple | None:
        """The held object's footprint when it is carried low enough to hit floor obstacles (a chair
        lifted a few centimetres); objects carried above furniture height are ignored."""
        held = self._held(es)
        if held is None or entity_box(held)[1] >= self.config.floor_furniture_height:
            return None
        return self._held_relative(es)

    def _retreat(self, obstacles: list[Box2D], start: tuple[float, float], direction: tuple[float, float],
                 yaw: float, carry: tuple | None, min_d: float, turn_carry: tuple | None = None) -> tuple[float, float] | None:
        """Nearest point at least ``min_d`` along ``direction`` (heading unchanged) that the robot can
        reach in a straight line with its load and where it has room to turn with it (judged with
        ``turn_carry`` when given)."""
        sweep = self._sweep(turn_carry or carry)
        d = 0.0
        while d <= self.config.carry_retreat_max + 1e-9:
            xy = (start[0] + direction[0] * d, start[1] + direction[1] * d)
            if self._footprint_blocked(obstacles, xy, yaw, carry):
                return None
            if d >= min_d and self._turn_blocked(obstacles, xy, sweep) is None:
                return xy
            d += 0.05
        return None

    def _navigation_path(self, es, obstacles, goal, carry=None, grid=None):
        rx, ry, yaw = self._robot_xy_yaw(es)
        radius = self._sweep(carry)
        if grid is None:
            grid = build_grid(obstacles, radius+self.config.nav_margin, [(rx, ry), goal])
        if self._turn_blocked(obstacles, (rx, ry), radius) is None:
            return astar(grid, (rx, ry), goal, start_clear=0.)
        exit_point = escape_prefix(obstacles, (rx, ry), yaw,
            (self.half_x+.01, self.half_y+.01),
            carry_box=(lambda xy: self._carried_box(carry, xy, yaw)) if carry is not None else None,
            turn_radius=radius, margin=self.config.nav_margin+grid.res,
            max_distance=self.config.carry_retreat_max)
        if exit_point is None:
            return None
        tail = astar(grid, tuple(exit_point[:2]), goal, start_clear=0.)
        if tail is None:
            return None
        # A third value specifies fixed yaw for this translation only. It survives
        # receipt serialization; the executor must not turn before this waypoint.
        return [[rx, ry], exit_point] + tail[1:]

    def _boxed_in(self, es: SceneState, obstacles: list[Box2D], carry: tuple | None) -> str | None:
        """When a carried load leaves no room to turn where the robot stands, say how to get out:
        put the load down where it is, or, where it would then trap the base, on a surface."""
        if carry is None:
            return None
        rx, ry, ryaw = self._robot_xy_yaw(es)
        blocked = self._turn_blocked(obstacles, (rx, ry), self._sweep(carry))
        if blocked is None:
            return None
        held = self._held(es)
        box, _, _ = entity_box(held)
        _, trapped = self._put_down_here(es, held, obstacles)
        if trapped:
            return (f"carrying {held.id} the robot has no room to turn here ({blocked}), and putting it down here "
                    f"would trap the base ({trapped}); set it on a surface within reach with place on=<surface id>")
        return (f"carrying {held.id} the robot has no room to turn here ({blocked}); put it down where it is with "
                f"place at=[{box.cx:.2f}, {box.cy:.2f}] yaw={box.yaw:.2f}, then move without it")

    def _release_reach(self, es: SceneState, held: Entity, rel: tuple) -> tuple[tuple, float | None]:
        """The held footprint (robot frame) and wrist reach for setting ``held`` down on the floor.
        An object that overhangs the base while carried would land on it, so the arm extends until
        the object clears the front of the base, within its travel; the reach is None when the arm
        stays as it is (the object already clears the base, the base drives over it, or the arm
        cannot extend that far, which the release check then reports)."""
        lx, ly, dyaw, hx, hy = rel
        _, bottom, top = entity_box(held)
        along = abs(math.cos(dyaw)) * hx + abs(math.sin(dyaw)) * hy
        extend = self.half_x + 0.02 + along - lx
        now = self._arm_joints(es).get("reach")
        if top - bottom <= self.ground_clearance or extend <= 0 or now is None or now + extend > self.reach_max:
            return rel, None
        return (lx + extend, ly, dyaw, hx, hy), now + extend

    def _put_down_here(self, es: SceneState, held: Entity, obstacles: list[Box2D]) -> tuple[dict, str | None]:
        """Plan setting ``held`` down on the floor where it is now: the base backs off straight by
        as much as the arm extends (see ``_release_reach``), so the object lands below where it is
        held. Returns the plan's dock, pre-dock (the retreat after release) and reach, or the
        reason it cannot be done."""
        rel = self._held_relative(es)
        rx, ry, ryaw = self._robot_xy_yaw(es)
        box, _, _ = entity_box(held)
        extended, reach = self._release_reach(es, held, rel)
        back = (-math.cos(ryaw), -math.sin(ryaw))
        shift = extended[0] - rel[0]
        dock = (rx + back[0] * shift, ry + back[1] * shift)
        placed = Box2D(held.id, box.cx, box.cy, box.yaw, box.hx, box.hy)
        landing = next((o.id for o in obstacles if rects_overlap(placed, o)), None)
        if landing:
            return {}, f"{held.id} would land against {landing}"
        hit = self._footprint_blocked(obstacles, dock, ryaw, rel) if shift > 0 else None
        if hit:
            return {}, f"backing off {shift:.2f} m to extend the arm would hit {hit}"
        trapped = self._release_blocked(obstacles, held, (box.cx, box.cy), box.yaw, dock, ryaw)
        if trapped:
            return {}, trapped
        retreat = self._retreat(obstacles + [placed], dock, back, ryaw, None, self.config.predock_backoff)
        here = {"dock": list(dock), "predock": list(retreat)}
        return (here | {"reach": reach}) if reach is not None else here, None

    def _release_blocked(self, obstacles: list[Box2D], held: Entity, xy: tuple[float, float], yaw: float,
                         dock: tuple[float, float], dock_yaw: float) -> str | None:
        """Why the robot, standing at ``dock``, could not drive away after setting ``held`` down on
        the floor at ``xy`` and heading ``yaw``, or None. A carried object may overhang the base; on
        the floor it must lie clear of the base, and the robot must back out to room to turn."""
        box, bottom, top = entity_box(held)
        if top - bottom <= self.ground_clearance:
            return None
        placed = Box2D(held.id, xy[0], xy[1], yaw, box.hx, box.hy)
        if self._footprint_blocked([placed], dock, dock_yaw):
            return f"{held.id} would lie under the robot's base"
        back = (-math.cos(dock_yaw), -math.sin(dock_yaw))
        if self._retreat(obstacles + [placed], dock, back, dock_yaw, None, self.config.predock_backoff) is None:
            return f"with {held.id} down there the robot could not back out to room to turn"
        return None

    @staticmethod
    def _carried_box(carry: tuple, xy: tuple[float, float], yaw: float) -> Box2D:
        lx, ly, dyaw, hx, hy = carry
        c, s = math.cos(yaw), math.sin(yaw)
        return Box2D("carried", xy[0] + c * lx - s * ly, xy[1] + s * lx + c * ly, yaw + dyaw, hx, hy)

    def _sweep(self, carry: tuple | None) -> float:
        """Radius of the circle the robot sweeps when turning in place, including a carried object."""
        if carry is None:
            return self.radius
        lx, ly, dyaw, hx, hy = carry
        c, s = math.cos(dyaw), math.sin(dyaw)
        far = max(math.hypot(lx + c * sx * hx - s * sy * hy, ly + s * sx * hx + c * sy * hy)
                  for sx in (-1, 1) for sy in (-1, 1))
        return max(self.radius, far)

    # ----------------------------------------------------------------- skills
    def predict(self, es: SceneState, skill: str, params: dict) -> dict:
        self._blockers = set()
        try:
            if skill == '_simple_driver':
                from steward.ego.fasten_simple import driver_stroke
                return driver_stroke(self, es, params)
            if skill in ('seat_screw', 'drive_screw', 'fasten_onto', 'flip_assembly', 'regrasp_upright', '_fastener_contact', '_fastener_reset', '_stage_screw', '_upright_tool'):
                from steward.ego.fasten_plan import predict, contact, reset_tool
                if skill == '_fastener_reset':
                    return reset_tool(self, es, params)
                return contact(self, es, params) if skill in ('_fastener_contact', '_stage_screw', '_upright_tool') else predict(self, es, skill, params)
            if skill == "assemble_onto":
                return self._predict_assemble_onto(es, params)
            if skill == "navigate":
                return self._predict_navigate(es, params)
            if skill == "pick":
                return self._predict_pick(es, params)
            if skill == "place":
                return self._predict_place(es, params)
            if skill in ("open", "close"):
                return self._predict_drawer(es, skill, params)
            if skill == "pour":
                return self._predict_pour(es, params)
            raise PredictionError(f"unknown skill {skill!r}; known: assemble_onto, navigate, pick, place, pour, open, close")
        except PredictionError as exc:
            blockers = self._describe_blockers(es)
            issues = [str(exc)] + [
                f"{b['id']} ({b['category']}) is in the way and can be moved: pick it, put it down out of the way "
                f"(for example place at={b['parkAt'][0]}), then predict again"
                for b in blockers if b["movable"] and b["parkAt"]]
            receipt = self._receipt(es, skill, params, 1.0, issues, {}, feasible=False)
            receipt["blockers"] = blockers
            return receipt

    def _predict_assemble_onto(self, es: SceneState, params: dict) -> dict:
        part, body = es.find(str(params.get('part', ''))), es.find(str(params.get('body', '')))
        if part is None or body is None or part is body:
            raise PredictionError(f'assemble_onto needs distinct perceived part and body ids: '
                                  f'part={params.get("part")!r} present={part is not None}; '
                                  f'body={params.get("body")!r} present={body is not None}')
        if self._held(es) is not None:
            raise PredictionError('assemble_onto starts with an empty hand; park the held part first')
        admission_warnings = []
        for obj in (part, body):
            if not self._screw_holes(obj):
                raise PredictionError(f'{obj.id} has no known interface; observe/name it before assembly')
            info = obj.state_vars.get('orientation')
            frozen = (info.value or {}).get('stationaryAssembly', {}) if info else {}
            if frozen.get('status') == 'invalidated':
                admission_warnings.append(dict(reason='stationary_reference_estimated', entity=obj.id))
            if obj is part and frozen:
                raise PredictionError('the loose part cannot be a frozen assembly body')
        body_info = body.state_vars.get('orientation')
        ledger = (body_info.value or {}).get('stationaryAssembly', {}) if body_info else {}
        for installed in ledger.get('parts', []):
            if installed['entity'] == part.id or (installed.get('category') == part.category.value
                    and np.linalg.norm(np.asarray(installed['position'])[:2]-part.pose.value['p'][:2]) < .08):
                raise PredictionError('part is already released at this body; inspect instead of assembling it again')
        anchor = self._anchor(es, body)
        if anchor is body or not self._affords(anchor, 'support-surface'):
            from steward.ego.fasten_plan import support_anchor
            anchor = support_anchor(self, es, body, admission_warnings)
        for obj, selector in ((body, params.get('hole')), (part, params.get('heldHole'))):
            holes = self._screw_holes(obj)
            chosen = self._select_holes(holes, selector, obj.id)
            groups = {h.get('group') for h in chosen if h.get('group')}
            if any(h.get('group') in groups and h not in chosen for h in holes):
                raise PredictionError('assemble_onto requires a complete interface prefix, not one hole of a group')
        source = lambda o: {'id': o.id, 'category': o.category.value, 'position': list(o.pose.value['p'])}
        plan = {'part': part.id, 'body': body.id, 'support': anchor.id,
                'partSource': source(part), 'bodySource': source(body),
                **{k: params[k] for k in ('hole', 'heldHole', 'deferVerify', 'closeRelease') if params.get(k) is not None},
                'stages': ['observe_multiview', 'freeze', 'pick', 'fit_and_place', 'retreat_verify'],
                'stagedPrediction': True}
        receipt = self._receipt(es, 'assemble_onto', params, .02,
                             ['Staged controller: every motion is freshly predicted and risk-gated before execution; '
                              'later stages may fail. Frozen geometry is a stationary assumption, not fresh verification.'], plan)
        receipt['warnings'].extend(admission_warnings)
        return receipt

    def _predict_navigate(self, es: SceneState, params: dict) -> dict:
        rx, ry, ryaw = self._robot_xy_yaw(es)
        goal = (float(params["x"]), float(params["y"]))
        yaw = float(params.get("yaw", ryaw))
        held = self._held(es)
        obstacles = self._floor_obstacles(es, {held.id} if held is not None else set())
        carry = self._carry(es)
        grid = build_grid(obstacles, self._sweep(carry) + self.config.nav_margin, [(rx, ry), goal])
        blocked = self._footprint_blocked(obstacles, goal, yaw, carry)
        if blocked:
            raise PredictionError(f"goal pose collides with {blocked}")
        path = self._navigation_path(es, obstacles, goal, carry, grid)
        if path is None:
            hint = self._boxed_in(es, obstacles, carry)
            raise PredictionError("no collision-free path to the goal" + (f"; {hint}" if hint else ""))
        clearance = path_clearance(path, obstacles) if len(path) > 1 else math.inf
        issues, risk = [], 0.02
        if clearance < self._sweep(carry) + 0.05:
            issues.append(f"path passes within {clearance:.2f} m of an obstacle")
            risk = 0.2
        plan = {"path": [list(p) for p in path], "yaw": yaw, "pathLength": path_length(path)}
        if held is not None and self._affords(held, "pourable"):
            plan["gentle"] = True  # a container with its contents: driven fast, they slide out
        return self._receipt(es, "navigate", params, risk, issues, plan)

    def _floor_obstacles(self, es: SceneState, exclude: set[str]) -> list[Box2D]:
        return floor_obstacles(es, exclude, ground_clearance=self.ground_clearance)

    def _predict_pick(self, es: SceneState, params: dict) -> dict:
        target = es.find(params.get("target", ""))
        if target is None:
            raise PredictionError(f"unknown target {params.get('target')!r}")
        if self._held(es) is not None:
            raise PredictionError("the gripper is already holding an object")
        assembly_grip = None
        if params.get('assemblyGrip'):
            from copy import deepcopy
            info = target.state_vars.get('orientation')
            ledger = (info.value or {}).get('stationaryAssembly', {}) if info else {}
            ref = ledger.get('reference')
            if ref:
                assembly_grip = ref
                target = deepcopy(target)
                target.pose.value = dict(p=list(ref['position']), q=quat_from_yaw(ref['yaw']))
                if 'graspHeightM' in ref:
                    target.pose.value['p'][2] = ref['graspHeightM']
                target.rigid.links[0].geometry['collision'][0]['size'] = list(ref['size'])
        tbox, target_bottom, _ = entity_box(target)
        tableware_sized = max(tbox.hx, tbox.hy) * 2 < self.config.graspable_max_m and target_bottom > 0.3
        miscategorised = not self._affords(target, "graspable")
        if miscategorised and not tableware_sized:
            raise PredictionError(f"{target.id} ({target.category.value}) is not graspable")
        support = self._support_of(es, target)
        # Upstream's floor rules: a small object off the floor is lifted clear of the base and
        # carried like one off a table; a large one (a chair) stays low in front of the robot.
        # The fork's rule stands on top of them: with no known support the pick is weighed, not
        # refused -- refusing kept the robot circling the table for hours.
        target_box, target_bottom, target_top = entity_box(target)
        on_floor = support is None and target_bottom < FLOOR_LYING_M
        blind = support is None and not on_floor
        carried_low = on_floor and 2 * max(target_box.hx, target_box.hy) > self.config.floor_lift_max_m
        tp = list(target.pose.value["p"])
        if on_floor:
            # Resting on the floor, its base is at floor level whatever the noisy height estimate says.
            tp[2] = max((target_top - target_bottom) / 2, self.lift_min)
        elif support is not None and self._affords(support, "support-surface") and not (assembly_grip and 'graspHeightM' in assembly_grip):
            # Resting on a surface, its centre is no lower than the top plus half its height: a grasp
            # closed below that presses what it holds into the surface and tips it (an edge grasp
            # tips a box enough to spill what lies in it).
            tp[2] = max(tp[2], entity_box(support)[2] + (target_top - target_bottom) / 2)
        contents = self._contents(es, target)
        if self._affords(target, "pourable"):
            covering = self._covering(es, target)
            if covering:
                raise PredictionError(f"{', '.join(covering)} lies on top of {target.id}; take it off first")
        elif contents and self._affords(target, "receptacle"):
            raise PredictionError(f"{target.id} holds {', '.join(e.id for e in contents)}; take those out first")
        if carried_low:
            carry_lift = tp[2] + self.config.floor_carry_lift
        elif on_floor:
            carry_lift = self.config.floor_furniture_height + 0.10
        else:
            carry_lift = tp[2] + 0.10
        over = None
        if support is not None and not on_floor:
            rim = entity_box(support)[2]
            if rim > tp[2]:  # it lies inside its support, below the walls
                over = min(self.lift_max, rim + self.config.rim_clearance)
                carry_lift = max(carry_lift, rim + (target_top - target_bottom) / 2 + self.config.rim_clearance)
        carry_lift = min(self.lift_max, carry_lift)
        if params.get("flip") and self.config.flip_in_air:
            # Carrying a most-tilted grasp below the tilted-carry floor with a short reach has no USD IK.
            carry_lift = min(self.lift_max, max(carry_lift, self.config.tilted_carry_floor))
        anchor = target if (on_floor or blind) else self._anchor(es, support)
        # Tilted most, the wrist tips a container farthest; anything else is taken with the least tilt,
        # or tilted most where the least does not also hold a dock that lands short or long.
        # Taken to be turned over (``flip``) by an arm that does so in the air, it is tilted most.
        most = self.arm.geometry.wrist_pitch_max
        if params.get("flip") and self.config.flip_in_air:
            pitches = [most]
        elif self._affords(target, "pourable"):
            pitches = [most, None]
        else:
            # Drawer picks use ``require_tilt`` so the least pitch is still positive; most-tilt at the
            # low close height is often outside the USD envelope (needs ~0.70 m+ at 0.50 m reach).
            pitches = [None, most] if most > 0 else [None]
        driver_heading = None
        if params.get('prepareDriver') or params.get('prepareUpright'):
            from steward.ego.fasten_plan import driver_grasp_heading
            driver_heading = driver_grasp_heading(self, es, target, most)
            pitches = [most]
        if params.get('topGrasp'):
            pitches = [0.]
        grasp_wrist = None
        assembly_points = None
        if assembly_grip:
            opening = self.arm.geometry.jaw_opening
            if opening is not None and 2*min(target_box.hx,target_box.hy) >= opening:
                raise PredictionError('body middle is wider than the gripper opening')
            angle = assembly_grip['yaw'] + (math.pi/2 if target_box.hy > target_box.hx else 0.)
            def grasp_wrist(heading):
                candidates = [wrap_angle(angle-heading+k*math.pi) for k in (-1,0,1)]
                return min(candidates, key=abs), 0.
            def driver_heading(heading):
                return None if abs(grasp_wrist(heading)[0]) <= self.arm.geometry.wrist_yaw_max else 'body pinch wrist yaw unreachable'
            # A middle-quarter pinch clears an overlaid transverse component's root;
            # the short body width, not the merged wing span, goes between the jaws.
            from steward.ego.fasten_release import body_grip_points, loose_components
            assembly_points = body_grip_points(assembly_grip, loose_components(es, target.id))
            if not assembly_points:
                raise PredictionError('no exposed body pinch outside loose component volumes')
            rx,ry,_ = self._robot_xy_yaw(es)
            quarter = max(target_box.hx, target_box.hy)*.5
            assembly_points.sort(key=lambda v:(round(abs(math.hypot(*v[1])-quarter), 6),
                                               math.hypot(v[0][0]-rx,v[0][1]-ry)))
        approach, grasp_offset, grasp_z = None, None, tp[2]
        first: PredictionError | None = None
        rim_z = max(tp[2], target_top - self.config.edge_grasp_depth) if self._affords(target, "container") else tp[2]
        # Under the USD mounts a grasp near 0.60 m only reaches ~0.35 m tool-down. Drawer contents
        # close near their true height (raising the close into the elevated floor grasps air) but
        # come in over ``elevated_grasp_floor`` and dock past the tool-down envelope so they tilt.
        floor = self.config.elevated_grasp_floor
        in_drawer = support is not None and support.category.value == "drawer"
        approach_over = None
        for xy, offset in (assembly_points or self._grasp_points(es, target, tp)):
            base_z = rim_z if any(offset) else tp[2]
            if in_drawer and most > 0:
                # Prefer the object's true height so a short bottle is not grasped in air; raise only
                # when the chosen dock reach has no close IK (see post-approach bump below).
                raise_steps = (0.0, 0.03, 0.06, 0.09, 0.12, 0.16)
                close_floor = base_z
            else:
                low = base_z < floor + 0.04 and most > 0
                raise_steps = ((0.12, 0.16, 0.20, 0.24) if low
                               else ((0.0, 0.04, 0.08, 0.12) if most > 0 else (0.0,)))
                close_floor = floor if low else 0.0
            if params.get('topGrasp') or assembly_grip:
                raise_steps = (0.,)
                close_floor = 0.
            for raise_z in raise_steps:
                z = min(self.lift_max, max(base_z + raise_z, close_floor))
                if in_drawer and most > 0:
                    # Stretch/tilt at an elevated approach height: a deep dock at the raw object
                    # height (≈0.63 m) has no USD IK at 0.50 m. Lower to ``z`` after untilting.
                    high = min(self.lift_max, max(z, floor, over or 0.0, 0.70))
                    high = min(high, max(floor + 0.20, z + 0.25, 0.70))
                    carry_for = min(carry_lift, max(floor + 0.20, z + 0.20, high))
                    path_lifts = [high, max(carry_for, high)]
                    dock_h = high
                else:
                    high = min(self.lift_max, max(over or z, z))
                    carry_for = carry_lift
                    path_lifts = [high, z, max(carry_for, high)]
                    dock_h = z
                try:
                    approach = self._approach(es, anchor, xy, dock_h,
                                              {target.id}, load=target if carried_low else None,
                                              heading_issue=driver_heading or (self._edge_facing(offset) if self.config.edge_grasp_first else None),
                                              pitches=pitches, wrist=grasp_wrist,
                                              lifts=path_lifts,
                                              require_tilt=in_drawer and most > 0)
                except PredictionError as exc:
                    first = first or exc
                    continue
                if in_drawer and most > 0:
                    reach_m = float(approach["reach"])
                    # Prefer tool-down at the object's true height: raising the close into air
                    # grasps nothing, and a mild tilt at the dock reach often packs joint2.
                    # Execution creeps in after untilting when closePitch is ~0.
                    close_z = base_z
                    td_close = None
                    for r in (0.40, 0.36, 0.34, 0.32, 0.30, 0.28):
                        if r > reach_m + 1e-9:
                            continue
                        if self.arm.reachable(close_z, r, wrist_pitch=0.0):
                            td_close = r
                            break
                    if td_close is not None:
                        z = close_z
                        approach["closePitch"] = 0.0
                        # Tell execution how far tool-down can go at this close height.
                        approach["closeReach"] = float(td_close)
                        carry_lift = max(carry_for, high)
                    else:
                        close_pf = self.arm.pitch_for(z, reach_m)
                        # Raise the close until this dock reach has a fresh IK solve (pitch_for alone
                        # can land on a limit-packed posture ``reachable`` rejects).
                        while z + 0.02 <= high + 1e-9:
                            if (close_pf is not None
                                    and self.arm.reachable(z, reach_m, wrist_pitch=float(close_pf))):
                                break
                            z = min(high, z + 0.02)
                            close_pf = self.arm.pitch_for(z, reach_m)
                        if (close_pf is None
                                or not self.arm.reachable(z, reach_m, wrist_pitch=float(close_pf))):
                            first = first or PredictionError(
                                f"close cannot hold dock reach {reach_m:.2f} m up to approach {high:.2f} m")
                            approach = None
                            continue
                        approach["closePitch"] = float(close_pf)
                        approach["wristPitch"] = max(float(approach.get("wristPitch") or 0.0),
                                                     float(close_pf))
                        carry_lift = max(carry_for, high)
                # Drawer closes always keep the elevated approachLift, even when close height rises
                # near ``high``: execution needs the untilt/creep path for tool-down closes.
                if in_drawer and most > 0:
                    grasp_offset, grasp_z, approach_over = offset, z, high
                else:
                    grasp_offset, grasp_z, approach_over = offset, z, (high if high > z + 1e-9 else over)
                break
            if approach is not None:
                break
        if approach is None:
            raise first
        sigma = math.hypot(self._sigma_xy(target), self._sigma_xy(self._self(es)))
        # Default capture covers Cobot TCP settle noise. Inside a receptacle, a wide close lets
        # the pinch sit on the container wall: ideal grasp then attaches the pourable box instead
        # of the piece (flip picks spilled the whole dog puzzle onto the table).
        r = self.config.capture_radius
        if (support is not None and self._affords(support, "receptacle")
                and not self._affords(target, "pourable")):
            r = min(r, 0.05)
        p_miss = math.exp(-(r * r) / (2 * sigma * sigma)) if sigma > 0 else 0.0
        issues = []
        if p_miss > 0.05:
            issues.append(f"target position uncertainty {sigma * 100:.1f} cm gives {p_miss:.0%} miss probability; "
                          "observe the target more closely first")
        # A grasp verifies itself: closing on nothing costs one retreat, nothing is damaged, and
        # the miss is reported. So a doubtful target is only half the risk its doubt suggests.
        existence_risk = self.config.phantom_pick_cost * (1.0 - target.existence)
        if 1.0 - target.existence > 0.2:
            issues.append(f"target existence probability is only {target.existence:.2f}")
        blind_risk = 0.0
        if blind:
            blind_risk = self.config.blind_support_risk
            issues.append(f"no support surface is known under {target.id}; lowering to its position blind")
        category_risk = 0.0
        if miscategorised:
            category_risk = self.config.miscategorised_risk
            issues.append(f"{target.id} is read as {target.category.value}, which is not graspable, but is "
                          "tableware-sized and raised; trying the grasp anyway")
        nest_risk = 0.0
        if contents:
            # Something read as inside the target is, with a real segmenter, usually a neighbour
            # whose box overlaps it (three bowls "inside" one bowl is a row of bowls). Lifting a
            # bowl that really holds another spills the inner one, so each claimed content adds
            # risk in proportion to how sure the estimator is -- but it never forbids the pick.
            confidence = max(self._relation_confidence(es, c.id, target.id) for c in contents)
            nest_risk = min(self.config.nesting_risk_cap, self.config.nesting_risk_each * len(contents)) * confidence
            issues.append(f"{target.id} may hold {', '.join(e.id for e in contents)} (confidence {confidence:.2f})")
        risk = 1.0 - (1.0 - p_miss) * (1.0 - existence_risk) * (1.0 - blind_risk) * (1.0 - nest_risk) \
            * (1.0 - category_risk)
        plan = {**approach, "target": target.id, "lift": grasp_z, "targetP": [float(v) for v in tp],
                "support": "floor" if on_floor else ("unknown" if blind else support.id),
                "carryLift": carry_lift, "keepReach": carried_low,
                "captureRadius": r,
                "gentle": self._affords(target, "pourable")}
        if params.get('stayNear'):
            plan['stayNear'] = True
        if assembly_grip:
            plan.update(assemblyGrip=True, wristYaw=grasp_wrist(approach['yaw'])[0], protectGraspFrame=True,
                        graspWidthM=2*min(target_box.hx,target_box.hy), graspSource='frozen_body_middle')
        holes = self._screw_holes(target)
        if holes and all(h.get("frame") == "entity" for h in holes):
            plan["protectGraspFrame"] = True
        if approach_over is not None:
            plan["approachLift"] = approach_over
        if grasp_z > tp[2]:
            plan["graspDz"] = grasp_z - tp[2]  # an edge grasp closes on the rim, above the centre
        if grasp_offset is not None and any(grasp_offset):
            plan["graspOffset"] = list(grasp_offset)
            issues.append(f"{target.id} is taken by its edge, {math.hypot(*grasp_offset):.2f} m from its centre")
        if (params.get("flip") and self.config.flip_in_air) or params.get("prepareDriver"):
            plan["wristPitch"] = self.arm.geometry.wrist_pitch_max
        receipt = self._receipt(es, "pick", params, risk, issues, plan)
        if params.get('prepareDriver'):
            from steward.ego.fasten_plan import tool_estimate
            receipt['warnings'].extend(tool_estimate(target, es.stamp).get('warnings', []))
        return receipt

    def _predict_place(self, es: SceneState, params: dict) -> dict:
        held = self._held(es)
        if held is None:
            raise PredictionError("the gripper is not holding anything")
        support = es.find(params.get("on", "")) if "on" in params else None
        seat_entry = None
        if params.get('seatBody') and params.get('seatHole'):
            from steward.ego.fasten_plan import seat_target
            from steward.ego.fasten_release import component_support
            seat_entry = seat_target(self, es, dict(body=params['seatBody'],
                part=params['on'], hole=params['seatHole']))
            if seat_entry.get('componentFrame'):
                support = component_support(es, seat_entry['componentFrame'])
        if support is None and "on" in params:
            raise PredictionError(f"unknown support {params.get('on')!r}")
        if params.get("flip") and params.get("roll") is not None:
            raise PredictionError("place cannot flip and roll at once; flip turns it upside down, roll tips it on its side")
        assembly_placement = None
        assembly_ignore = set()
        if params.get('assemblyBody') == held.id:
            from copy import deepcopy
            from steward.ego.assembly_motion import local_bounds
            info = held.state_vars.get('orientation')
            ledger = (info.value or {}).get('stationaryAssembly', {}) if info else {}
            if ledger.get('reference'):
                low, high = local_bounds(ledger)
                assembly_placement = (low, high)
                assembly_ignore.update(ledger.get('aliases', []))
                assembly_ignore.update(p['entity'] for p in ledger.get('parts', []))
                held = deepcopy(held)
                size = 2*np.maximum(abs(low),abs(high))
                size[:2] = max(size[:2])  # covers base/wrist yaw changes while docking
                held.rigid.links[0].geometry['collision'][0]['size'] = size.tolist()
        held_box, hb, ht = entity_box(held)
        held_h = ht - hb
        held_half = max(held_box.hx, held_box.hy)
        place_roll = self._place_roll(params)
        flip_roll = math.pi
        if assembly_placement is not None and params.get('flip') and not self.config.flip_in_air:
            from steward.ego.assembly_motion import half_turn_roll
            try:
                flip_roll = half_turn_roll(self._arm_joints(es).get('wrist_roll', 0.),
                                           self.arm.geometry.wrist_roll_max)
            except ValueError as exc:
                raise PredictionError(str(exc)) from exc
        if place_roll is not None and abs(abs(place_roll) - math.pi / 2) <= 0.2:
            # A quarter turn about the arm stands the former horizontal extent up.
            held_h, held_half = 2.0 * held_half, held_h / 2.0
        if support is None and "at" in params:
            if params.get("alignHoles") is not False and (params.get("alignHoles") is True
                    or params.get("alignTo") is not None or params.get("hole") is not None):
                raise PredictionError("[RSI_ALIGN_REQUIRED] floor placement does not support hole alignment; "
                                      "specify an observed support and mate, or use alignHoles=false for ordinary placement")
            return self._predict_floor_place(es, held, params)
        if support is None:
            raise PredictionError(f"unknown support {params.get('on')!r}")
        support_orientation = getattr(support, "state_vars", {}).get("orientation")
        support_posture = ((support_orientation.value or {}).get("posture", {})
                           if support_orientation is not None else {})
        mate, aligned_offset, aligned_yaw, align_issues = self._place_hole_align(es, held, support, params)
        place_params = dict(params)
        if (aligned_offset is None and params.get("yaw") is None and self._screw_holes(held)
                and not params.get("flip") and place_roll is None):
            orientation = getattr(held, "state_vars", {}).get("orientation")
            before = (orientation.value or {}).get("graspWorldYawRad") if orientation is not None else None
            if before is not None:
                place_params["yaw"] = float(before)
                align_issues.append("[RSI_TRANSPORT_YAW] ordinary placement of an interface part restores "
                                    "its pre-transport estimated heading; this is not hole alignment or "
                                    "observed directed yaw. Explicit yaw overrides this default")
        mating_on = False
        if aligned_offset is not None:
            place_params["offset"] = list(aligned_offset)
            place_params["clearance"] = 0.0
            if aligned_yaw is not None and place_params.get("yaw") is None:
                place_params["yaw"] = aligned_yaw
        if self._affords(support, "receptacle"):
            if self._affords(support, "pullable"):
                spot, area, bottom, issues = self._drawer_spot(es, support, held, held_half)
            else:
                spot, area, bottom, issues = self._nest_spot(es, support, held, held_half)
            issues = align_issues + issues
            hold_dz = self._hold_dz(es, held)
            surface = bottom
            height = bottom + 0.012 + held_h / 2 + 0.02 + hold_dz
            rim = entity_box(support)[2]
            over = (min(self.lift_max, rim + held_h / 2 + self.config.rim_clearance + hold_dz)
                    if rim > height - hold_dz - held_h / 2 else None)
            sigma = math.hypot(self._sigma_xy(support), self._sigma_xy(self._self(es)))
            clearance = min(area.hx, area.hy) - held_half
            risk = 0.03 if clearance >= 2 * sigma + 0.003 else 0.4
            if risk > 0.1:
                issues.append(f"only {clearance * 100:.1f} cm of clearance with {sigma * 100:.1f} cm uncertainty")
        elif self._affords(support, "support-surface") or self._screw_holes(support):
            over = None
            # Screw-hole parts that are not furniture: the held piece may hang past them (wing on
            # fuselage). Mark the plan so release only checks the centre, not footprint containment.
            mating_on = bool(self._screw_holes(support) and not self._affords(support, "support-surface"))
            sbox, _, top = entity_box(support)
            assembly = (support_orientation.value or {}).get('assembly') if support_orientation else None
            if mate is support and assembly:
                ref = assembly['reference']
                top = ref.get('landingTopM', ref['position'][2] + ref['size'][2]/2)
            if params.get('seatBody') and params.get('seatHole'):
                # The mating crossbar can lie below the part's legs. Use the
                # selected interface surface, not the global top of its box.
                from steward.ego.fasten_plan import seat_target
                entry = seat_entry
                top = float(entry['point'][2])
                # Resolve a world target in THIS receipt's support frame. An
                # offset computed before a perception update can move with a mask.
                from steward.ego.fasten_simple import offset_on
                place_params['offset'] = offset_on(support, entry['point'][:2])
            final_yaw = float(place_params["yaw"]) if place_params.get("yaw") is not None else held_box.yaw
            # Mating onto a part that is not furniture: keep the load on its footprint, not inset.
            edge_keep = 0.0 if mating_on else None
            # Flip-in-air releases from higher with a drop: keep ``place_clearance`` from neighbours
            # so the piece does not land on another piece already on the table.
            flip_place = bool(params.get("flip")) and self.config.flip_in_air
            clear = float(place_params.get("clearance",
                                         self.config.place_clearance if flip_place else 0.015))
            spot, issues = self._choose_spot(es, support, sbox, held, held_box, place_params.get("offset"),
                                             final_yaw, clear, edge_keep=edge_keep,
                                             ignore_ids=assembly_ignore, flat_ok=assembly_placement is not None)
            issues = align_issues + issues
            area = sbox
            if params.get('closeRelease') and seat_entry is None:
                from steward.ego.fasten_plan import fresh_view
                from steward.ego.fasten_geometry import surface_patch
                view = fresh_view(support, es.stamp)
                patch = (surface_patch(view['points'], spot, max_tilt_deg=25) if view and
                         es.stamp-view['stamp'] <= 20 and not view['truncated'] and
                         view['invalidDepthFraction'] <= .05 else None)
                if patch:
                    # Clear the higher edge of the footprint over a sloping
                    # support; do not drive the crossbar into its high end.
                    normal = np.asarray(patch['normal'])
                    slope = -normal[:2]/normal[2]
                    c,s=math.cos(final_yaw),math.sin(final_yaw)
                    local_slope=np.array([[c,s],[-s,c]]) @ slope
                    top = patch['point'][2]+float(abs(local_slope) @ [held_box.hx,held_box.hy])
            surface = top
            release_gap = .002 if params.get('closeRelease') else .02
            height = top + held_h / 2 + release_gap + self._hold_dz(es, held)
            if assembly_placement is not None:
                low, high = assembly_placement
                bottom = high[2] if params.get('flip') else -low[2]
                hold_dz = self._hold_dz(es, held) * (-1 if params.get('flip') else 1)
                height = top + bottom + .02 + hold_dz
            ex, ey = self._turned_half(held_box, final_yaw, sbox.yaw)
            edge = self._edge_margin(sbox, spot, ex, ey)
            sigma = self._sigma_xy(support)
            risk = 0.02
            if edge < 2 * sigma + 0.03 and self._affords(support, "support-surface"):
                issues.append(f"placement is {edge * 100:.0f} cm from the edge with {sigma * 100:.1f} cm support uncertainty")
                risk = 0.4
        else:
            raise PredictionError(f"{support.id} ({support.category.value}) is neither a surface nor a receptacle")
        yaw_goal = place_params.get("yaw")
        turn = None
        if yaw_goal is not None:
            want, flip = float(yaw_goal), bool(params.get("flip"))

            def turn(dock_yaw: float) -> str | None:
                try:
                    self._wrist_for(es, held, want, dock_yaw, flip)
                except PredictionError as exc:
                    return str(exc)
                return None
        flip = bool(params.get("flip"))
        if flip and self.config.flip_in_air:
            # A piece taken with the tool tilted most needs a higher place: query short docks at the
            # tilted-carry floor instead of a fixed ~0.90 m guess.
            pitch = self.arm.geometry.wrist_pitch_max
            height = max(height, self.config.tilted_carry_floor)
            while height < self.lift_max - 1e-9 and not all(
                    self.arm.reachable(height, r, wrist_pitch=pitch) for r in (0.40, 0.43, 0.45)):
                height = min(self.lift_max, height + 0.02)

        def wrist_turn(dock_yaw: float) -> float:
            if yaw_goal is None:
                return 0.0
            return self._wrist_for(es, held, float(yaw_goal), dock_yaw, flip) - self._arm_joints(es).get("wrist_yaw", 0.0)

        tool_roll = (flip_roll if flip and not self.config.flip_in_air else place_roll)
        approach, grip = self._grip_approach(es, self._anchor(es, support), held, spot, height, wrist_turn, flip, turn,
                                             lifts=[over, height] if over is not None else None, tool_roll=tool_roll,
                                             exact_rotation=assembly_placement is not None)
        assembly_turn = None
        if assembly_placement is not None and flip and not self.config.flip_in_air:
            low, high = assembly_placement
            lx,ly,*_ = self._held_relative(es)
            joints = self._arm_joints(es)
            offset = [lx-self.tool_offset-joints.get('reach',0.), ly-self.lateral, self._hold_dz(es,held)]
            radius = float(np.linalg.norm(np.maximum(abs(low), abs(high)))+np.linalg.norm(offset))
            assembly_turn = max(height, surface+radius+abs(self._hold_dz(es,held))+.02)
            pitch = float(approach.get('wristPitch') or 0.)
            wy = approach.get('wristYaw', self._arm_joints(es).get('wrist_yaw', 0.))
            rr = self._arm_joints(es).get('wrist_roll', 0.)
            # Rigid combination rotates clear of the table before lowering.
            path = [dict(lift=assembly_turn, reach=approach['reach'], wrist_yaw=wy,
                         wrist_roll=float(r), wrist_pitch=pitch) for r in np.linspace(rr, flip_roll, 19)]
            path += [dict(lift=float(z), reach=approach['reach'], wrist_yaw=wy,
                          wrist_roll=flip_roll, wrist_pitch=pitch) for z in np.linspace(assembly_turn,height,8)]
            if not self.arm.reachable_path(path):
                raise PredictionError('assembly half-turn or lowering path physically unreachable')
        fragility = (held.rigid.links[0].material.value or {}).get("fragility", 0.0) if held.rigid.links[0].material else 0.0
        risk = 1.0 - (1.0 - risk) * (1.0 - 0.05 * fragility)
        plan = {**approach, "object": held.id, "support": support.id, "spot": list(spot), "gripSpot": list(grip),
                "lift": height, "supportBox": {"cx": area.cx, "cy": area.cy, "yaw": area.yaw, "hx": area.hx, "hy": area.hy},
                "heldHalf": held_half}
        if seat_entry is not None:
            plan['seatTarget'] = seat_entry
        if assembly_turn is not None:
            plan["assemblyTurnLift"] = assembly_turn
        if mate is not None:
            plan["alignTo"] = mate.id
            # These are actuator/estimated-geometry bounds, not a declaration that
            # the holes or the final pose have been observed to coincide.
            plan["placementToleranceM"] = 0.02
            if not mating_on:
                plan["releaseHalfXY"] = list(self._turned_half(held_box, float(aligned_yaw), area.yaw))
        if mating_on:
            plan["mate"] = True
        if over is not None:
            plan["approachLift"] = over
        if yaw_goal is not None:
            plan["wristYaw"] = self._wrist_for(es, held, float(yaw_goal), approach["yaw"], bool(params.get("flip")))
            plan["placementTarget"] = {
                "yawRad": float(yaw_goal), "sizeM": [2 * held_box.hx, 2 * held_box.hy, ht - hb],
                "support": support.id, "planar": not params.get("flip") and place_roll is None,
                "partCategory": held.category.value,
            }
        if mate is not None and 'placementTarget' in plan:
            orientation = mate.state_vars.get('orientation')
            info = orientation.value if orientation else {}
            saved = info.get('assembly', {}).get('reference')
            category = (info.get('holeRegistration') or {}).get('templateCategory', mate.category.value)
            box, bottom, top = entity_box(mate)
            plan['placementTarget']['mateReference'] = saved or {
                'entity': mate.id, 'category': category,
                'family': 'toy_airplane' if category == 'airplane_fuselage' else category,
                'position': list(mate.pose.value['p']), 'yaw': box.yaw,
                'size': [2*box.hx, 2*box.hy, top-bottom],
                'holes': self._screw_holes(mate),
                'holeSource': 'observed' if all(h.get('source') == 'observed' for h in self._screw_holes(mate)) else 'prior',
                'registration': info.get('holeRegistration'),
                'support': self._anchor(es, mate).id, 'registeredAt': mate.pose.t,
            }
            from steward.ego.fasten_release import placement_pairs
            plan['placementTarget']['holePairs'] = placement_pairs(
                self, held, mate, spot, float(yaw_goal), params)
        if mate is not None and 'placementTarget' in plan and plan['placementTarget']['planar']:
            holes = self._screw_holes(held)
            orientation = held.state_vars.get('orientation')
            info = orientation.value if orientation else {}
            if holes and all(h.get('frame') == 'entity' for h in holes):
                # Transported registered geometry, never a new visual observation.
                plan['placementTarget']['releasedFrame'] = {
                    'holes': holes, 'registration': info.get('holeRegistration'),
                    'settledCentreZM': surface + held_h/2,
                    'sourceObservedAt': (info.get('graspObservedAt') if info.get('graspObservedAt') is not None else held.pose.t),
                }
        if params.get("flip"):
            plan["wristRoll"] = flip_roll
            if assembly_placement is not None:
                plan['relativeTurn'] = dict(fromRoll=self._arm_joints(es).get('wrist_roll', 0.),
                                            toRoll=flip_roll)
            if self.config.flip_in_air:
                plan.update(self._turn_over_plan(es, held, plan, height, surface, area))
        elif place_roll is not None:
            plan["wristRoll"] = place_roll  # tipped about the arm (e.g. π/2 to set it down on its side)
        plan["gentle"] = self._affords(held, "pourable")
        plan['stayNear'] = bool(params.get('stayNear', False))
        return self._receipt(es, "place", params, risk, issues, plan)

    def _place_hole_align(self, es: SceneState, held: Entity, support: Entity,
                          params: dict) -> tuple[Entity | None, list[float] | None, float | None, list[str]]:
        """When the held part and a mate both have screw holes, the offset (and yaw) that seats them.

        ``alignTo`` names the mate; without it, ``on`` is the mate when that part itself has holes.
        ``alignHoles: false`` skips mating. ``hole`` picks a mate hole id (or a shared prefix).
        The offset is always in the ``on`` support's frame so ``_choose_spot`` can use it.

        :returns: ``(mate, offset_xy, yaw, issues)``; offset/yaw are None when not aligning.
        """
        if params.get("alignHoles") is False:
            return None, None, None, []
        requested = (params.get("alignHoles") is True or params.get("alignTo") is not None
                     or params.get("hole") is not None)
        held_holes = self._screw_holes(held)
        if not held_holes:
            if requested:
                raise PredictionError(f"[RSI_ALIGN_REQUIRED] {held.id} has no screw holes; "
                                      "requested alignment cannot be planned. Verify the held object's identity "
                                      "and features; ordinary placement requires alignHoles=false.")
            return None, None, None, []
        mate_id = params.get("alignTo")
        if mate_id is not None:
            mate = es.find(mate_id)
            if mate is None:
                raise PredictionError(f"unknown alignTo {mate_id!r}")
        elif self._screw_holes(support) and (requested or params.get("offset") is None):
            mate = support
        else:
            if requested:
                raise PredictionError(f"[RSI_ALIGN_REQUIRED] {support.id} has no screw holes to align to; "
                                      "observe the intended mate or specify alignTo. "
                                      "Ordinary placement requires alignHoles=false.")
            return None, None, None, []
        mate_holes = self._screw_holes(mate)
        if not mate_holes:
            raise PredictionError(f"{mate.id} has no screw holes to align to")
        if params.get("flip") or abs(float(params.get("roll") or 0)) > 1e-6:
            raise PredictionError("[RSI_ASSEMBLY_POSTURE] planar hole fitting cannot validate a flipped/rolled release; "
                                  "recover posture separately on a clear stable support, reobserve, then mate")
        posture_warnings = []
        for obj in (held, mate):
            orient = getattr(obj, 'state_vars', {}).get('orientation')
            posture = (orient.value or {}).get('posture', {}) if orient else {}
            if posture.get('status') == 'inconsistent' or posture.get('upright') is False:
                posture_warnings.append(f"[RSI_ASSEMBLY_POSTURE] {obj.id}: {posture}; best estimate used, verify after release")
        # Only observed features or outline-registered priors can enter the fitter.
        unregistered = [obj.id for obj, holes in ((held, held_holes), (mate, mate_holes))
                        if any(h.get("frame") != "entity" or not (
                            h.get("source") == "observed" or
                            (h.get("source") == "prior" and h.get("registration") == "outline_axes"
                             and len(h.get("positionsByRegistration", [])) == 2)) for h in holes)]
        if unregistered:
            posture_warnings.append(f"[RSI_HOLE_FRAME] assumed entity coordinates for {unregistered}; count={len(unregistered)}; verify after release")
        held_holes = self._select_holes(held_holes, params.get("heldHole"), held.id)
        mate_holes = self._select_holes(mate_holes, params.get("hole"), mate.id)
        registered = any(h.get("registration") == "outline_axes" for h in held_holes + mate_holes)

        def on_support(offset_mate):
            if mate is support:
                return np.asarray(offset_mate)
            # Express the held centre (mate frame) as an offset on the surface under it.
            mx, my, myaw = mate.pose.value["p"][0], mate.pose.value["p"][1], yaw_from_quat(mate.pose.value["q"])
            wx = mx + math.cos(myaw) * offset_mate[0] - math.sin(myaw) * offset_mate[1]
            wy = my + math.sin(myaw) * offset_mate[0] + math.cos(myaw) * offset_mate[1]
            sbox, _, _ = entity_box(support)
            dx, dy = wx - sbox.cx, wy - sbox.cy
            c, s = math.cos(-sbox.yaw), math.sin(-sbox.yaw)
            return np.array([c * dx - s * dy, s * dx + c * dy])

        use_catalog = params.get("hole") is None and params.get("heldHole") is None
        group_pairs, grouped = self._hole_group_pairs(held_holes, mate_holes, use_catalog)
        candidates = self._hole_fit_candidates(held, held_holes, mate, mate_holes, params.get("yaw"),
                                               use_catalog=use_catalog)
        candidates = [(on_support(o), a, pairs) for o, a, pairs in candidates]
        # An explicit offset selects among same-yaw alternatives; it never overrides the fit.
        if params.get("offset") is not None:
            wanted = np.asarray(params["offset"], float)
            if wanted.shape != (2,) or not np.isfinite(wanted).all():
                raise PredictionError("[RSI_HOLE_MATCH] offset must be two finite surface-frame coordinates")
            cfg = getattr(self, "config", None) or EsimConfig()
            candidates = [(o, a, pairs) for o, a, pairs in candidates
                          if np.linalg.norm(o - wanted) <= cfg.hole_fit_tolerance_m]
        issues = posture_warnings + ["[RSI_ASSEMBLY_POSTURE] planar fit does not observe uprightness. Keep the mate stationary; "
                  "check top/side features and support before release, then reobserve both parts. "
                  "Task-required pose changes must be planned separately before registering the new interface frame; "
                  "a merged box is not attachment and a planar fit does not validate a side/underside interface"]
        orientation = getattr(held, "state_vars", {}).get("orientation")
        grasp_yaw = (orientation.value or {}).get("graspWorldYawRad") if orientation is not None else None
        offset, yaw, used = self._choose_hole_fit(
            candidates, "offset", registered=registered,
            preferred_yaw=(grasp_yaw if grasp_yaw is not None else yaw_from_quat(held.pose.value["q"])),
            issues=issues)
        if grasp_yaw is not None and len(candidates) > 1:
            issues.append("[RSI_MATING_FRAME] ambiguous fits ranked against the pre-transport estimated "
                          "heading; transport must not select the opposite half-turn. Directed yaw remains unknown")
        ids = ", ".join(f"{a}->{b}" for a, b in used)
        issues.append(f"[RSI_HOLE_MATCH] aligned screw holes on {mate.id}: {ids}; "
                      "planar fit only; verify pose and seating after release")
        if grouped:
            issues.append(f"[RSI_HOLE_GROUP] catalogue selected {len(group_pairs)} compatible interface pairs; "
                          "hole/heldHole explicitly override catalogue pairing")
        if registered:
            issues.append("[RSI_HOLE_REGISTER] used outline-compatible prior hypotheses, not observed holes; "
                          "half-turn alternatives remain hypotheses; reobserve after release")
        if len(used) == 1:
            issues.append("[RSI_YAW_AMBIGUOUS] [RSI_SINGLE_HOLE_YAW] one hole leaves yaw unconstrained; "
                          + ("using the requested yaw" if params.get("yaw") is not None else
                             "restoring the pre-transport estimated heading when available, otherwise preserving the held heading")
                          + f" ({yaw:.6f} world radians). This cancels estimated transport rotation, not initial yaw uncertainty; "
                          "supply yaw for a repeatable world-frame command. A round or near-square outline cannot "
                          "verify directed orientation; inspect an asymmetric feature and reobserve after release")
        return mate, offset, yaw, issues

    @staticmethod
    def _select_holes(holes: list[dict], selector, entity_id: str) -> list[dict]:
        """Explicit identifiers or an identifier prefix; no category/family inference."""
        if selector is None:
            return holes
        if isinstance(selector, (list, tuple)):
            wanted = set(selector)
            picked = [h for h in holes if h['id'] in wanted]
            if len(picked) != len(wanted):
                raise PredictionError(f'{entity_id} is missing selected interface holes')
            return picked
        name = str(selector)
        exact = [h for h in holes if h["id"] == name]
        picked = exact or [h for h in holes if h["id"].startswith(name + "_")]
        if not picked:
            raise PredictionError(f"{entity_id} has no hole matching {name!r}")
        return picked

    def _mate_offset_yaw(self, held: Entity, held_holes: list[dict], mate: Entity,
                         mate_holes: list[dict], yaw: float | None = None
                         ) -> tuple[list[float], float, list[tuple[str, str]]]:
        """Fit selected observed or registered holes, retaining ambiguity instead of polar order.

        Distinct proper rotations/translations within the sensor-fit tolerance are alternatives,
        not evidence that the first permutation is correct. A caller's camera-grounded world
        yaw constrains the fit itself, including translation for off-centre features.
        """
        return self._choose_hole_fit(self._hole_fit_candidates(held, held_holes, mate, mate_holes, yaw),
                                     "offsetInMateM", preferred_yaw=yaw_from_quat(held.pose.value["q"]))

    @staticmethod
    def _hole_frames(holes: list[dict]) -> list[list[dict]]:
        if not any(h.get("registration") == "outline_axes" for h in holes):
            return [holes]
        # One index for the entire rigid object, never independent sign choices per hole.
        return [[{**h, "p": h["positionsByRegistration"][i]} for h in holes] for i in range(2)]

    @staticmethod
    def _hole_group_pairs(held_holes, mate_holes, use_catalog=True):
        """Read interface compatibility from descriptors, never ids or categories.

        Missing metadata keeps the unconstrained geometry path. When both sides
        declare interfaces, each group is indivisible and both declarations must
        permit the pair. Explicit feature selectors bypass catalogue preferences.
        """
        if not use_catalog or not all("group" in h and "mates" in h for h in held_holes + mate_holes):
            return [(held_holes, mate_holes)], False

        def groups(holes):
            out = {}
            for h in holes:
                out.setdefault(h["group"], []).append(h)
            return out

        pairs = [(hh, mh) for hg, hh in sorted(groups(held_holes).items())
                 for mg, mh in sorted(groups(mate_holes).items())
                 if len(hh) == len(mh) and all(mg in h["mates"] for h in hh)
                 and all(hg in h["mates"] for h in mh)]
        return pairs, True

    def _hole_fit_candidates(self, held: Entity, held_holes: list[dict], mate: Entity,
                             mate_holes: list[dict], yaw: float | None = None, *, use_catalog=True):
        cfg = getattr(self, "config", None) or EsimConfig()
        groups, grouped = self._hole_group_pairs(held_holes, mate_holes, use_catalog)
        if grouped and not groups:
            raise PredictionError("[RSI_HOLE_GROUP] no compatible catalogue interfaces; "
                                  "verify identity or explicitly select hole/heldHole")
        frames = [(self._hole_frames(sorted(hh, key=lambda h: h["id"])),
                   self._hole_frames(sorted(mh, key=lambda h: h["id"]))) for hh, mh in groups]
        assignments = sum(math.perm(len(mf[0]), len(hf[0])) * len(hf) * len(mf) for hf, mf in frames)
        if assignments > cfg.hole_assignment_limit:
            raise PredictionError("[RSI_HOLE_MATCH] too many possible assignments; select heldHole and hole")
        candidates = []
        for held_frames, mate_frames in frames:
            for hh, mh in itertools.product(held_frames, mate_frames):
                for o, a, pairs in self._fit_one_hole_frame(held, hh, mate, mh, yaw):
                    if not any(abs(_wrap(a - angle)) < 1e-7 and np.linalg.norm(o - off) < 1e-7
                               for off, angle, _ in candidates):
                        candidates.append((o, a, pairs))
        return candidates

    def _fit_one_hole_frame(self, held, held_holes, mate, mate_holes, yaw):
        cfg = getattr(self, "config", None) or EsimConfig()
        n, m = len(held_holes), len(mate_holes)
        if not n or n > m:
            raise PredictionError("[RSI_HOLE_MATCH] too few mate holes; observe all required features "
                                  "or explicitly select heldHole and hole; constraints will not be truncated")
        if math.perm(m, n) > cfg.hole_assignment_limit:
            raise PredictionError("[RSI_HOLE_MATCH] too many possible assignments; select heldHole and hole")
        hp = np.asarray([h["p"][:2] for h in held_holes], float)
        mp = np.asarray([h["p"][:2] for h in mate_holes], float)
        if not np.isfinite(hp).all() or not np.isfinite(mp).all() or (yaw is not None and not math.isfinite(float(yaw))):
            raise PredictionError("[RSI_HOLE_MATCH] non-finite feature geometry or yaw")
        myaw = yaw_from_quat(mate.pose.value["q"])
        hc = hp.mean(0)
        ha = hp - hc
        # Coincident points, like a single point, cannot constrain rotation.
        free_yaw = n == 1 or float(np.linalg.norm(ha)) < 1e-9
        if free_yaw and yaw is None:
            orientation = getattr(held, "state_vars", {}).get("orientation")
            before = (orientation.value or {}).get("graspWorldYawRad") if orientation is not None else None
            # A circular outline has an arbitrary frame, but encoders still tell
            # us how much we rotated it since grasping. Undo that transport turn
            # instead of inheriting the heading of the pick's retreat waypoint.
            yaw = float(before) if before is not None else yaw_from_quat(held.pose.value["q"])
        candidates = []
        for indices in itertools.permutations(range(m), n):
            matched = mp[list(indices)]
            mc = matched.mean(0)
            if yaw is None:
                u, _, vt = np.linalg.svd((matched - mc).T @ ha)
                r = u @ vt
                if np.linalg.det(r) < 0:
                    u[:, -1] *= -1
                    r = u @ vt
            else:
                angle = float(yaw) - myaw
                c, sn = math.cos(angle), math.sin(angle)
                r = np.array([[c, -sn], [sn, c]])
            offset = mc - r @ hc
            error = float(np.linalg.norm(hp @ r.T + offset - matched, axis=1).max())
            if error > cfg.hole_fit_tolerance_m:
                continue
            angle = _wrap(myaw + math.atan2(r[1, 0], r[0, 0]))
            # Collapse identical transforms only, not physically different near-symmetric fits.
            if any(abs(_wrap(angle - a)) < 1e-7 and np.linalg.norm(offset - o) < 1e-7
                   for o, a, _ in candidates):
                continue
            used = [(h["id"], mate_holes[j]["id"]) for h, j in zip(held_holes, indices)]
            candidates.append((offset, angle, used))
        return candidates

    @staticmethod
    def _choose_hole_fit(candidates, offset_key, *, registered=False, preferred_yaw=0., issues=None):
        prefix = "[RSI_HOLE_REGISTER] outline-compatible prior; " if registered else ""
        if not candidates:
            raise PredictionError(prefix + "[RSI_HOLE_MATCH] no rigid planar fit within sensor tolerance; "
                                  "reobserve features or correct the requested yaw/offset/selection")
        # Minimise the held object's rotation. Stable ties preserve template frame
        # order (primary before half-turn), then the sorted feature-id traversal.
        # Rounding removes numerical SVD noise from otherwise identical rotations.
        ranked = sorted(candidates, key=lambda c: round(abs(_wrap(c[1] - preferred_yaw)), 10))
        if len(ranked) > 1 and issues is not None:
            alternatives = [{"yaw": a, "yawDeg": math.degrees(a), offset_key: o.tolist(),
                             "pairs": pairs} for o, a, pairs in ranked]
            issues.append(prefix + f"[RSI_YAW_AMBIGUOUS] {len(ranked)} geometric alternatives: {alternatives}; "
                          "using the first candidate by minimum preferred-heading rotation, then stable template/feature order. "
                          "This default is not observed disambiguation. Verify after release: retract and obtain fresh views of both parts. "
                          "Compare relative centres AND visible asymmetric features with the hypotheses. "
                          "Only if independent evidence rejects this fit and supports another, "
                          "re-identify, pick again and re-place using another listed candidate's world yaw "
                          "(radians) and offset in the on-surface frame. hole/heldHole override catalogue pairing. "
                          "An undirected outer axis cannot resolve a half turn. If the features and relative centres "
                          "are identical under both hypotheses, do not blindly re-pick or alternate; directed yaw "
                          "remains unknown. Keep verified parts and their supports undisturbed; unknown evidence is not success")
        offset, angle, used = ranked[0]
        return offset.tolist(), angle, used

    def _place_roll(self, params: dict) -> float | None:
        """Wrist roll for place, or None to leave the tool upright. Mutually exclusive with ``flip``."""
        if params.get("roll") is None:
            return None
        roll = float(params["roll"])
        limit = self.arm.geometry.wrist_roll_max
        if abs(roll) < 0.05:
            raise PredictionError(f"roll {roll:.2f} rad is too small; omit roll to leave the tool upright")
        if abs(roll) > limit:
            raise PredictionError(f"roll {roll:.2f} rad is outside +-{limit:.2f}")
        return roll

    def _turn_over_plan(self, es: SceneState, held: Entity, approach: dict, height: float, surface: float,
                        area: Box2D) -> dict:
        """For ``flip_in_air``: ``turnLift``, where the held object is turned over above its spot
        from the ``approach`` dock, and ``lift``, the lowest height the arm then brings it down to,
        at most ``flip_drop_max`` above ``height`` (where it would be set down on ``surface``, the
        top of footprint ``area``)."""
        cfg = self.config
        reach = approach["reach"]
        heading = np.array([math.cos(approach["yaw"]), math.sin(approach["yaw"])])
        edge = float(min((area.corners() - np.array(approach["dock"][:2])) @ heading))
        # Prefer the plan's carry pitch, then the max: FK of a loaded USD posture can read 0 while
        # the piece was taken tilted most (``flip_in_air``), which would refuse a valid place.
        pitch = float(approach.get("wristPitch") or self._held_pitch(es)[0] or 0.0)
        if self.config.flip_in_air:
            pitch = max(pitch, self.arm.geometry.wrist_pitch_max)
        if pitch < cfg.flip_pitch_min:
            raise PredictionError(f"{held.id} was taken with the tool tilted {pitch:.2f} rad; turning it over needs it "
                                  f"taken tilted at least {cfg.flip_pitch_min:.2f} rad: put it down and pick it again "
                                  "with flip")
        turn = min(self.lift_max, height + cfg.flip_turn_above)
        yaw = approach.get("wristYaw", self._arm_joints(es).get("wrist_yaw", 0.0))
        low = self.arm.turn_over(turn, reach, pitch, height, surface, edge, wrist_yaw=yaw)
        if low is None:
            raise PredictionError(f"the arm cannot turn {held.id} over {reach:.2f} m out")
        if low - height > cfg.flip_drop_max:
            raise PredictionError(f"turned over, {held.id} comes down only to {low - height:.2f} m above where it is "
                                  f"set down; it would drop more than {cfg.flip_drop_max:.2f} m")
        # A dock that lands short (dock_tol is 6 cm) must still hold the lowered pose with the tool
        # tilted; raise the release height into the USD envelope for that short reach.
        short = max(0.10, reach - 0.06)
        while low < turn - 1e-9 and not self.arm.reachable(low, short, wrist_yaw=yaw, wrist_pitch=pitch):
            low = min(turn, low + 0.02)
        # Prefer the lowest release the short-reach envelope allows (a hard 0.92 m floor left a
        # long drop that bounced pieces onto neighbours).
        floor = self.arm.min_lift_for(short, pitch, wrist_yaw=yaw)
        if floor is not None:
            low = max(low, floor + 0.02)
        else:
            low = max(low, self.config.tilted_carry_floor)
        return {"turnLift": turn, "lift": low}

    def _grip_spot(self, es: SceneState, spot: tuple[float, float], dock_yaw: float, wrist_turn: float,
                   flip: bool) -> tuple[float, float]:
        """Where the gripper goes for the held object's centre to land on ``spot`` with the base at
        ``dock_yaw`` and the wrist turned ``wrist_turn`` further: an object taken off-centre (a lid
        by its edge) hangs off the gripper by an offset that turns with the base and the wrist, and
        turning it over (``flip``) mirrors it across the arm."""
        lx, ly, _, _, _ = self._held_relative(es)
        ox, oy = lx - (self.tool_offset + self._arm_joints(es).get("reach", 0.0)), ly - self.lateral
        if flip:
            oy = -oy
        a = dock_yaw + wrist_turn
        return spot[0] - (math.cos(a) * ox - math.sin(a) * oy), spot[1] - (math.sin(a) * ox + math.cos(a) * oy)

    def _held_pitch(self, es: SceneState) -> list[float]:
        """The tool pitch the held object was taken at, the only one it is carried and set down at."""
        return [self._arm_joints(es).get("wrist_pitch", 0.0)]

    def _grip_approach(self, es: SceneState, anchor: Entity, held: Entity, spot: tuple[float, float], height: float,
                       wrist_turn: Callable[[float], float], flip: bool,
                       heading_issue: Callable[[float], str | None] | None = None,
                       lifts: Sequence[float] | None = None,
                       tool_roll: float | None = None, exact_rotation: bool = False) -> tuple[dict, tuple[float, float]]:
        """The approach that brings the held object's centre over ``spot``, and the gripper's spot
        for it: on each side of the support, where the gripper goes for the heading docked there.

        ``tool_roll`` is the wrist roll held through the approach (``flip`` without ``flip_in_air``
        uses π; a place ``roll`` uses that angle); without it the current roll is kept.
        """
        def grip_at(heading: float) -> tuple[float, float]:
            try:
                turn = wrist_turn(heading)
            except PredictionError:
                turn = 0.0  # a side the wrist cannot turn at is refused by heading_issue
            if exact_rotation:
                lx,ly,*_ = self._held_relative(es)
                offset = np.array([lx-self.tool_offset-current.get('reach',0.), ly-self.lateral,
                                   -self._hold_dz(es,held)])
                pitch = current.get('wrist_pitch',0.)
                r0 = self.arm.tool_rotation(current.get('wrist_yaw',0.), current.get('wrist_roll',0.),pitch)
                r1 = self.arm.tool_rotation(current.get('wrist_yaw',0.)+turn,roll,pitch)
                moved = matrix_from_quat(quat_from_yaw(heading)) @ r1 @ r0.T @ offset
                return spot[0]-float(moved[0]), spot[1]-float(moved[1])
            return self._grip_spot(es, spot, heading, turn, flip)

        current = self._arm_joints(es)
        start_joints = {j.id: j.q.value[0] for j in self._self(es).rigid.joints if j.q.value}
        roll = current.get("wrist_roll", 0.0) if tool_roll is None else tool_roll

        def wrist_at(heading: float) -> tuple[float, float]:
            return (current.get("wrist_yaw", 0.0) + wrist_turn(heading), roll)

        def path(heading: float, reach: float, near: float) -> bool:
            # Driving preserves the arm's base-relative posture; these stages begin after docking.
            # The carrying posture may lie outside the planning range: the path starts at the nearest
            # posture in range, and its first waypoint is still solved continuously from the encoders.
            yaw, roll = wrist_at(heading)
            g = self.arm.geometry
            at = {"wrist_yaw": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, **current}
            at["reach"] = min(max(at["reach"], g.reach_min), g.reach_max)
            at["lift"] = min(max(at["lift"], g.lift_min), g.lift_max)
            targets = [at]
            stages = [{"reach": near}, {"wrist_yaw": yaw}, {"wrist_roll": roll},
                      {"lift": (lifts or [height])[0]}, {"reach": reach}]
            stages += [{"lift": h} for h in (lifts or [height])[1:]]
            for change in stages:
                end = {**at, **change}
                steps = max(1, *(math.ceil(abs(end[k] - at[k]) /
                                          (self.config.lift_check_step if k in ("lift", "reach")
                                           else self.config.wrist_check_step)) for k in change))
                targets.extend({k: at[k] + (end[k] - at[k]) * i / steps for k in at} for i in range(1, steps + 1))
                at = end
            return self.arm.reachable_path(targets, start_joints=start_joints)

        hold_pitch = self._held_pitch(es)
        if flip and self.config.flip_in_air:
            hold_pitch = [self.arm.geometry.wrist_pitch_max]
        # After a tilted pick the skill untilts on exit: FK then reports ~0 while a deep place/pour
        # dock still sits past the tool-down envelope. Try tool-down first, then ``pitch_for``.
        elif abs(float(hold_pitch[0])) < 1e-3:
            hold_pitch = [0.0, None]
        approach = self._approach(es, anchor, grip_at, height, {held.id}, heading_issue=heading_issue,
                                  pitches=hold_pitch, lifts=lifts, wrist=wrist_at,
                                  approach_path=None if flip and self.config.flip_in_air else path)
        return approach, grip_at(approach["yaw"])

    def _wrist_for(self, es: SceneState, held: Entity, yaw: float, dock_yaw: float, flip: bool = False) -> float:
        """Wrist yaw that sets the held object down at heading ``yaw`` with the base at ``dock_yaw``.
        Turning it over (``flip``: a half roll about the gripper axis at heading a) maps a heading h
        to 2a - h, so the object's offset from the gripper axis changes sign."""
        _, _, ryaw = self._robot_xy_yaw(es)
        wrist = self._arm_joints(es).get("wrist_yaw", 0.0)
        relative = yaw_from_quat(held.pose.value["q"]) - ryaw - wrist
        target = _wrap(yaw - dock_yaw + relative) if flip else _wrap(yaw - dock_yaw - relative)
        if abs(target) > self.arm.geometry.wrist_yaw_max:
            raise PredictionError(f"heading {math.degrees(yaw):.0f} deg needs a wrist turn of {math.degrees(target):.0f} deg, "
                                  f"beyond +-{math.degrees(self.arm.geometry.wrist_yaw_max):.0f}")
        return target

    def _predict_pour(self, es: SceneState, params: dict) -> dict:
        """Tip the held ``pourable`` container over a support surface so its contents fall out
        there, then hold it upright again (the container stays in the gripper)."""
        held = self._held(es)
        if held is None:
            raise PredictionError("the gripper is not holding anything")
        if not self._affords(held, "pourable"):
            raise PredictionError(f"{held.id} ({held.category.value}) cannot be poured")
        support = es.find(params.get("on", ""))
        if support is None or not self._affords(support, "support-surface"):
            raise PredictionError(f"unknown support surface {params.get('on')!r}")
        sbox, _, top = entity_box(support)
        held_box, hb, ht = entity_box(held)
        sides = [2 * max(b.hx, b.hy) for b in (entity_box(c)[0] for c in self._contents(es, held))]
        spill = max(held_box.hx, held_box.hy) + self.config.pour_spill_sizes * max(sides, default=0.0)
        if min(sbox.hx, sbox.hy) < spill:
            raise PredictionError(f"{support.id} is too small to pour {held.id} on: the contents could slide over "
                                  f"its edge (a spot {spill:.2f} m from every edge is needed)")
        spot, issues = self._choose_spot(es, support, sbox, held, held_box, params.get("offset"), held_box.yaw,
                                         float(params.get("clearance", 0.015)), edge_keep=spill)
        height = top + (ht - hb) / 2 + self.config.pour_clearance + self._hold_dz(es, held)
        approach, grip = self._grip_approach(es, self._anchor(es, support), held, spot, height, lambda _: 0.0, False)
        roll = float(params.get("roll", self.config.pour_roll))
        if not (0.3 <= abs(roll) <= self.arm.geometry.wrist_roll_max):
            raise PredictionError(f"roll {roll:.2f} rad is outside [0.3, {self.arm.geometry.wrist_roll_max}]")
        # The container tips about its edge across the arm: half its extent across the docked heading.
        rel = held_box.yaw - approach["yaw"]
        pivot = abs(math.sin(rel)) * held_box.hx + abs(math.cos(rel)) * held_box.hy
        rise = self.config.pour_rise
        if height + pivot + rise > self.lift_max:
            raise PredictionError(f"tipping {held.id} about its edge needs a lift of {height + pivot + rise:.2f} m, "
                                  f"above {self.lift_max} m")
        # How far the wrist can tip it depends on where the arm holds it: tip as far as the whole
        # motion (the lift rising with the tip, then by ``pour_rise``, and back upright; see the
        # skill) stays within the arm's reach. It tips over either edge alike, so unless the caller
        # chose a side the other one is tried too: from some postures one way turns the wrist
        # against its stop (joint6 on the Piper).
        # Prefer the approach pitch (``pitch_for`` after an untilted carry) over live FK: pour docks
        # past ~0.50 m must keep that tilt through the tip, not re-plan tool-down.
        pitch = float(approach.get("wristPitch") or self._held_pitch(es)[0] or 0.0)
        yaw = self._arm_joints(es).get("wrist_yaw", 0.0)

        def tips(r: float) -> bool:
            steps = max(1, math.ceil(abs(r) / self.config.pour_check_step))
            rises = max(1, math.ceil(rise / self.config.lift_check_step))

            def at(k: int, up: float) -> dict:
                return {"lift": height + pivot * abs(math.sin(r * k / steps)) + up, "reach": approach["reach"],
                        "wrist_yaw": yaw, "wrist_roll": r * k / steps, "wrist_pitch": pitch}

            return self.arm.reachable_path([at(k, 0.0) for k in range(steps + 1)]
                                           + [at(steps, rise * j / rises) for j in range(1, rises + 1)]
                                           + [at(k, rise) for k in range(steps - 1, -1, -1)])

        def farthest(r: float, least: float) -> float:
            """The largest tip, from ``r`` toward upright in 0.1 rad steps, that :func:`tips`; below ``least`` if none."""
            sign = 1.0 if r >= 0 else -1.0
            while abs(r) >= least and not tips(r):
                r -= sign * 0.1
            return r

        wanted = roll
        for side in ([roll] if "roll" in params else [roll, -roll]):
            roll = farthest(side, self.config.pour_roll_min)
            if abs(roll) >= self.config.pour_roll_min:
                break
        if abs(roll) < self.config.pour_roll_min:
            most = max((farthest(side, 0.3) for side in ([wanted] if "roll" in params else [wanted, -wanted])), key=abs)
            raise PredictionError(f"holding {held.id} {approach['reach']:.2f} m out, the wrist tips it at most "
                                  f"{max(abs(most), 0.0):.1f} rad without turning over, short of the "
                                  f"{self.config.pour_roll_min:.1f} rad its walls need to let the contents out; pour "
                                  "onto a spot nearer the edge, or take the contents out one by one")
        if abs(abs(wanted) - abs(roll)) > 1e-6:
            issues.append(f"the wrist tips {held.id} only {roll:.1f} rad here (asked {wanted:.1f})")
        plan = {**approach, "object": held.id, "support": support.id, "spot": list(spot), "gripSpot": list(grip),
                "lift": height, "roll": roll, "pivot": pivot, "pourS": self.config.pour_hold_s, "rise": rise,
                "gentle": True}
        # Deep pour docks must keep the planned tilt through tip and exit (tool-down tops out ~0.50 m).
        if abs(pitch) >= 1e-3:
            plan["wristPitch"] = float(pitch)
        return self._receipt(es, "pour", params, 0.1, issues, plan)

    def _predict_floor_place(self, es: SceneState, held: Entity, params: dict) -> dict:
        """Put the held object down on the floor with its centre at ``at``: stand behind the spot
        facing it, drive straight in from a pre-pose, lower the object to just above the floor and
        release. ``yaw`` (radians) sets the object's final heading; otherwise the robot tries
        several headings, starting with the direction it is coming from. ``flip`` requests a
        half roll before release; ``roll`` requests its explicit angle. Rotation and lowering
        must form a reachable path from the current encoders, otherwise prediction fails."""
        cfg = self.config
        try:
            tx, ty = (float(v) for v in params["at"][:2])
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"at must be [x, y] in metres: {exc}") from None
        rel = self._held_relative(es)
        carrying = rel
        flip = bool(params.get("flip"))
        if flip:
            # A half roll reflects an off-centre grasp about the current wrist's forward axis.
            # Dock for the resulting centre and heading, while navigation still carries the old pose.
            lx, ly, dyaw, hx, hy = rel
            joints = self._arm_joints(es)
            angle = joints.get("wrist_yaw", 0.0)
            tool_x = self.tool_offset + joints["reach"]
            ox, oy = lx - tool_x, ly - self.lateral
            c, s = math.cos(2 * angle), math.sin(2 * angle)
            rel = (tool_x + c * ox + s * oy, self.lateral + s * ox - c * oy,
                   2 * angle - dyaw, hx, hy)
        lx, ly, dyaw, _, _ = rel
        rx, ry, ryaw = self._robot_xy_yaw(es)
        if "yaw" in params:
            headings = [float(params["yaw"]) - dyaw]
        else:
            toward = math.atan2(ty - ry, tx - rx)
            headings = [toward + k * math.pi / 4 for k in (0, 1, -1, 2, -2, 3, -3, 4)]
        lift_now = self._arm_joints(es).get("lift")
        _, held_bottom, _ = entity_box(held)
        if lift_now is None:
            raise PredictionError("the lift position is unknown")
        place_roll = math.pi if flip else self._place_roll(params)
        if place_roll is not None and abs(abs(place_roll) - math.pi / 2) <= 0.2:
            # Sideways on the floor: the former footprint stands up, so the release lift rises.
            held_box_now, _, held_top = entity_box(held)
            side = max(held_box_now.hx, held_box_now.hy)
            lift = lift_now - (held_bottom - cfg.floor_release_gap) + (side - (held_top - held_bottom) / 2)
        else:
            lift = lift_now - (held_bottom - cfg.floor_release_gap)
        if not (self.lift_min <= lift <= self.lift_max):
            raise PredictionError(f"lowering {held.id} to the floor needs lift {lift:.2f} m, outside the lift range")
        obstacles = self._floor_obstacles(es, {held.id})
        held_box, _, _ = entity_box(held)
        here_yaw = held_box.yaw if "yaw" not in params else float(params["yaw"])
        if not flip and math.hypot(tx - held_box.cx, ty - held_box.cy) <= 0.05 and abs(_wrap(here_yaw - held_box.yaw)) <= 0.1:
            here, trapped = self._put_down_here(es, held, obstacles)
            if trapped:
                raise PredictionError(f"cannot put {held.id} down where it is: {trapped}; set it on a surface "
                                      "within reach with place on=<surface id>, or carry it elsewhere")
            plan = {"path": [[rx, ry]], "yaw": ryaw, "lift": lift, "object": held.id,
                    "floor": [held_box.cx, held_box.cy], "pathLength": 0.0, **here}
            plan.update(self._floor_rotation(es, plan, place_roll))
            return self._receipt(es, "place", params, 0.02, [], plan)
        rel, reach = self._release_reach(es, held, rel)
        lx = rel[0]
        nav_load = carrying if flip else rel
        sweep = max(self._sweep(rel), self._sweep(nav_load))
        grid = build_grid(obstacles, sweep + cfg.nav_margin, [(rx, ry), (tx, ty)])
        reasons = []
        for h in headings:
            c, s = math.cos(h), math.sin(h)
            dock = (tx - (c * lx - s * ly), ty - (s * lx + c * ly))
            blocked = self._footprint_blocked(obstacles, dock, h, rel)
            if blocked is None and flip:
                blocked = self._footprint_blocked(obstacles, dock, h, carrying)
            if blocked:
                reasons.append(f"heading {math.degrees(h):.0f} deg: {held.id} or the robot would hit {blocked}")
                continue
            trapped = self._release_blocked(obstacles, held, (tx, ty), h + dyaw, dock, h)
            if trapped:
                reasons.append(f"heading {math.degrees(h):.0f} deg: {trapped}")
                continue
            predock = self._retreat(obstacles, dock, (-c, -s), h, nav_load, cfg.predock_backoff)
            if predock is None:
                reasons.append(f"heading {math.degrees(h):.0f} deg: no straight approach with room to turn while "
                               f"carrying {held.id}")
                continue
            path = self._navigation_path(es, obstacles, predock, nav_load, grid)
            if path is None:
                reasons.append(f"heading {math.degrees(h):.0f} deg: no collision-free path while carrying {held.id}")
                continue
            plan = {"path": [list(q) for q in path], "predock": list(predock), "dock": list(dock), "yaw": h,
                    "lift": lift, "object": held.id, "floor": [tx, ty], "pathLength": path_length(path)}
            if reach is not None:
                plan["reach"] = reach
            plan.update(self._floor_rotation(es, plan, place_roll))
            return self._receipt(es, "place", params, 0.03, [], plan)
        hint = self._boxed_in(es, obstacles, rel)
        raise PredictionError(f"cannot put {held.id} down at ({tx:.2f}, {ty:.2f}): " + "; ".join(reasons[:4])
                              + (f"; {hint}" if hint else ""))

    def _floor_rotation(self, es: SceneState, plan: dict, roll: float | None) -> dict:
        """Validate floor placement's extension, rotation and lowering in execution order.

        No requested rotation returns no extra fields. A requested rotation returns ``wristRoll``
        only after the sampled path is reachable from measured joints, or raises ``PredictionError``.
        """
        if roll is None:
            return {}
        current = self._arm_joints(es)
        at = {"wrist_yaw": 0.0, "wrist_roll": 0.0, "wrist_pitch": 0.0, **current}
        targets = [at]
        for change in ({"reach": plan.get("reach", at["reach"])}, {"wrist_roll": roll}, {"lift": plan["lift"]}):
            end = {**at, **change}
            steps = max(1, *(math.ceil(abs(end[k] - at[k]) /
                                      (self.config.lift_check_step if k in ("reach", "lift")
                                       else self.config.wrist_check_step)) for k in change))
            targets.extend({k: at[k] + (end[k] - at[k]) * i / steps for k in at} for i in range(1, steps + 1))
            at = end
        measured = {j.id: j.q.value[0] for j in self._self(es).rigid.joints if j.q.value}
        if not self.arm.reachable_path(targets, start_joints=measured):
            raise PredictionError(f"cannot rotate the held object to roll {roll:.2f} rad and lower it to the floor "
                                  "from the current arm posture; place it on a reachable support instead")
        return {"wristRoll": roll}

    def _nest_spot(self, es: SceneState, bowl: Entity, held: Entity, held_half: float):
        """Nest a stackable object in the centre of a larger receptacle that holds nothing yet."""
        if not self._affords(held, "stackable"):
            raise PredictionError(f"{held.id} ({held.category.value}) does not nest inside {bowl.id}")
        box, bottom, _ = entity_box(bowl)
        if held_half >= self.config.nest_fit * min(box.hx, box.hy):
            raise PredictionError(f"{held.id} is too large to nest inside {bowl.id}")
        inside = self._contents(es, bowl, exclude=held.id)
        if inside:
            raise PredictionError(f"{bowl.id} already holds {', '.join(e.id for e in inside)}; nest into the innermost one")
        return (box.cx, box.cy), box, bottom, []

    def _drawer_spot(self, es: SceneState, drawer: Entity, held: Entity, held_half: float):
        """A free spot on the drawer floor within the part that sticks out of the furniture."""
        _, bottom, _ = entity_box(drawer)
        box = self._front(es, drawer)
        opening = self._opening(es, drawer)
        front_inner = box.hx - 0.05
        lo = max(-box.hx + 0.02, front_inner - opening + 0.02) + held_half
        hi = front_inner - held_half
        if hi < lo:
            raise PredictionError(f"{drawer.id} is open {opening * 100:.0f} cm, too little for {held.id}; open it first")
        y_lim = box.hy - 0.02 - held_half
        if y_lim < 0:
            raise PredictionError(f"{held.id} does not fit in {drawer.id}")
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        others = [entity_box(e)[0] for e in self._contents(es, drawer, exclude=held.id)]
        lx = (lo + hi) / 2
        for ly in sorted(np.arange(-y_lim, y_lim + 1e-9, 0.02), key=abs):
            spot = (box.cx + c * lx - s * ly, box.cy + s * lx + c * ly)
            if all(math.hypot(spot[0] - o.cx, spot[1] - o.cy) >= held_half + max(o.hx, o.hy) + 0.015 for o in others):
                cx_area = (lo + hi) / 2
                area = Box2D(drawer.id, box.cx + c * cx_area, box.cy + s * cx_area, box.yaw,
                             (hi - lo) / 2 + held_half, box.hy - 0.02)
                return spot, area, bottom, []
        raise PredictionError(f"no free spot left in the open part of {drawer.id}")

    def _predict_drawer(self, es: SceneState, skill: str, params: dict) -> dict:
        """Open or close a drawer by its handle: dock facing the handle, grasp, move the base along the
        drawer axis, release, then back away so the inside is in view."""
        cfg = self.config
        drawer = es.find(params.get("target", ""))
        if drawer is None:
            raise PredictionError(f"unknown drawer {params.get('target')!r}")
        if not self._affords(drawer, "pullable"):
            raise PredictionError(f"{drawer.id} ({drawer.category.value}) is not a drawer")
        if self._held(es) is not None:
            raise PredictionError("the gripper must be empty to use a handle")
        _, bottom, top = entity_box(drawer)
        box = self._front(es, drawer)
        n = tuple(params["axis"]) if params.get("axis") else (math.cos(box.yaw), math.sin(box.yaw))
        # A look along the open drawer can shift the estimated box. The handle just pulled, moved
        # by that pull, is the one to push back.
        if params.get("handle"):
            handle = [float(v) for v in params["handle"]]
        else:
            handle = [box.cx + n[0] * (box.hx - 0.015), box.cy + n[1] * (box.hx - 0.015), (bottom + top) / 2]
            # The drawer box wanders along the furniture. The handle is centred on the piece, and
            # it is not proud of that piece's front: a swollen drawer puts the hand on empty air.
            host = self._drawer_furniture(es, drawer)
            if host is not None:
                hbox = self._front(es, host)
                hn = (math.cos(hbox.yaw), math.sin(hbox.yaw))
                if n[0] * hn[0] + n[1] * hn[1] > 0.5:
                    lateral = (-n[1], n[0])
                    side = hbox.cx * lateral[0] + hbox.cy * lateral[1]
                    along = handle[0] * n[0] + handle[1] * n[1]
                    face = hbox.cx * hn[0] + hbox.cy * hn[1] + hbox.hx + 0.02
                    if along > face + 0.03:
                        along = face
                    handle[0] = n[0] * along + lateral[0] * side
                    handle[1] = n[1] * along + lateral[1] * side
        opening = self._opening(es, drawer)
        if skill == "open":
            if opening > 0.2:
                raise PredictionError(f"{drawer.id} is already open ({opening * 100:.0f} cm)")
            distance = min(max(float(params.get("distance", cfg.drawer_pull)), 0.05), cfg.drawer_pull_max)
            sign = 1.0
        else:
            # The look into an open drawer rebuilds the box, and the opening left on the track can
            # shrink to a few centimetres while the drawer is still out. The distance just pulled
            # is the one that puts it back.
            requested = params.get("distance")
            if opening < 0.05 and requested is None:
                raise PredictionError(f"{drawer.id} is already closed")
            if requested is None:
                distance = min(opening + 0.02, cfg.drawer_pull_max + 0.04)
            else:
                distance = min(max(float(requested), 0.05), cfg.drawer_pull_max + 0.04)
            sign = -1.0
        standoff = self.tool_offset + cfg.handle_reach
        yaw = math.atan2(-n[1], -n[0])
        right = (math.sin(yaw) * self.lateral, -math.cos(yaw) * self.lateral)  # the arm's own line on the handle
        dock = (handle[0] + n[0] * standoff + right[0], handle[1] + n[1] * standoff + right[1])
        end = (dock[0] + sign * n[0] * distance, dock[1] + sign * n[1] * distance)
        clear = (end[0] + n[0] * cfg.drawer_viewing_backoff, end[1] + n[1] * cfg.drawer_viewing_backoff)
        predock = (dock[0] + n[0] * cfg.predock_backoff, dock[1] + n[1] * cfg.predock_backoff)
        if not (self.lift_min <= handle[2] <= self.lift_max):
            raise PredictionError(f"handle height {handle[2]:.2f} m is outside the lift range")
        # Contents ride the drawer. A bottle at the open mouth is where the robot stands
        # at the end of the close, and is not a fixed obstacle.
        exclude = {drawer.id}
        dp = drawer.pose.value["p"]
        for ent in es.entities:
            if ent.id in exclude or ent.role in ("environment", "robot"):
                continue
            if ent.category.value in ("cabinet", "sideboard", "counter", "table", "wall", "drawer"):
                continue
            box, bottom, top = entity_box(ent)
            if 2 * max(box.hx, box.hy) > 0.35:
                continue
            if abs((bottom + top) / 2 - dp[2]) > 0.25:
                continue
            if math.hypot(box.cx - dp[0], box.cy - dp[1]) <= 0.75:
                exclude.add(ent.id)
        obstacles = self._floor_obstacles(es, exclude)
        # A wall boxed thicker than the panel fills the floor in front of the cabinet, and the
        # close is then refused while the base is still in open floor. The path keeps the wall.
        wall_ids = {ent.id for ent in es.entities if ent.category.value == "wall"}
        stance = [box for box in obstacles if box.id not in wall_ids]
        for name, xy in (("dock", dock), ("end", end), ("viewing", clear)):
            blocked = self._footprint_blocked(stance, xy, yaw)
            if blocked:
                raise PredictionError(f"{name} pose in front of {drawer.id} collides with {blocked}")
        blocked = self._turn_blocked(obstacles, predock)
        if blocked:
            raise PredictionError(f"no room to turn in front of {drawer.id} ({blocked})")
        rx, ry, _ = self._robot_xy_yaw(es)
        grid = build_grid(obstacles, self.radius + cfg.nav_margin, [(rx, ry), predock])
        path = self._navigation_path(es, obstacles, predock, grid=grid)
        if path is None:
            raise PredictionError(f"no collision-free path to the front of {drawer.id}")
        sigma = math.hypot(self._sigma_xy(drawer), self._sigma_xy(self._self(es)))
        r = cfg.capture_radius
        p_miss = self._handle_miss(sigma)
        issues = [f"handle position uncertainty {sigma * 100:.1f} cm gives {p_miss:.0%} miss probability"] if p_miss > 0.05 else []
        plan = {"path": [list(q) for q in path], "predock": list(predock), "dock": list(dock), "yaw": yaw,
                "end": list(end), "viewing": list(clear), "handle": handle, "distance": distance, "direction": sign,
                "axis": list(n), "drawer": drawer.id, "captureRadius": r, "pathLength": path_length(path),
                "handleProbesAlong": list(cfg.handle_probes_along_m), "handleProbesSide": list(cfg.handle_probes_side_m)}
        return self._receipt(es, skill, params, max(0.03, p_miss), issues, plan)

    def _handle_miss(self, sigma: float) -> float:
        """Probability that a handle whose position has standard deviation ``sigma`` (m, each
        horizontal axis) lies outside what the drawer skill reaches for.

        The skill closes the gripper at the planned handle and, when that holds nothing, again at
        each ``handle_probes_along_m`` step along the arm and each ``handle_probes_side_m`` step
        of the base across it; every closing takes a handle within ``capture_radius``."""
        cfg = self.config
        if sigma <= 0:
            return 0.0

        def within(lo: float, hi: float) -> float:
            return 0.5 * (math.erf(hi / (sigma * math.sqrt(2))) - math.erf(lo / (sigma * math.sqrt(2))))

        r = cfg.capture_radius
        along, across = (0.0, *cfg.handle_probes_along_m), (0.0, *cfg.handle_probes_side_m)
        return 1.0 - within(min(along) - r, max(along) + r) * within(min(across) - r, max(across) + r)

    @staticmethod
    def _turned_half(held_box: Box2D, yaw: float, axis_yaw: float) -> tuple[float, float]:
        """Half extents of ``held_box`` turned to ``yaw``, along the axes of a frame at ``axis_yaw``."""
        turn = yaw - axis_yaw
        c, s = abs(math.cos(turn)), abs(math.sin(turn))
        return c * held_box.hx + s * held_box.hy, s * held_box.hx + c * held_box.hy

    @staticmethod
    def _edge_margin(sbox: Box2D, xy: tuple[float, float], ex: float, ey: float) -> float:
        """Smallest gap between a footprint of half extents (ex, ey) on the support's axes, centred
        at ``xy``, and the support's edges (negative when it overhangs)."""
        dx, dy = xy[0] - sbox.cx, xy[1] - sbox.cy
        c, s = math.cos(-sbox.yaw), math.sin(-sbox.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return min(sbox.hx - abs(lx) - ex, sbox.hy - abs(ly) - ey)

    @staticmethod
    def _edge_distance(sbox: Box2D, xy: tuple[float, float]) -> float:
        dx, dy = xy[0] - sbox.cx, xy[1] - sbox.cy
        c, s = math.cos(-sbox.yaw), math.sin(-sbox.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return min(sbox.hx - abs(lx), sbox.hy - abs(ly))

    def _choose_spot(self, es: SceneState, support: Entity, sbox: Box2D, held: Entity, held_box: Box2D,
                     offset: list | None, yaw: float | None = None,
                     clearance: float = 0.015, edge_keep: float | None = None,
                     ignore_ids=(), flat_ok=False) -> tuple[tuple[float, float], list[str]]:
        """A spot for ``held`` on ``support``: ``offset`` in the support frame if given, else the free
        spot nearest the robot. A requested spot where the held footprint, turned to ``yaw``, would
        come within ``clearance`` of another object moves to the free spot nearest it, at most
        ``spot_shift_max`` away, and an issue says where; none that near is an error. Every spot
        keeps ``edge_keep`` (default: the held object's half extent) from the support's edges."""
        cfg = self.config
        # The held footprint, turned to the heading it is put down with, projected on the support's
        # own axes: max(hx, hy) is not its half width once turned (at 45 degrees a 0.20 m half-wide
        # load passed a 0.28 m half-wide support and overhung it by up to 8 cm).
        ex, ey = self._turned_half(held_box, held_box.yaw if yaw is None else yaw, sbox.yaw)
        keep_x, keep_y = (ex, ey) if edge_keep is None else (edge_keep, edge_keep)
        keep = max(keep_x, keep_y)

        def inside(xy: tuple[float, float]) -> bool:
            dx, dy = xy[0] - sbox.cx, xy[1] - sbox.cy
            lx = math.cos(sbox.yaw) * dx + math.sin(sbox.yaw) * dy
            ly = -math.sin(sbox.yaw) * dx + math.cos(sbox.yaw) * dy
            return sbox.hx - abs(lx) >= keep_x - 1e-9 and sbox.hy - abs(ly) >= keep_y - 1e-9

        others = []
        on_support = set()
        def occupies(ent):
            if ent is None or ent.id in ignore_ids:
                return False
            _, bottom, top = entity_box(ent)
            return not (flat_ok and top-bottom <= .004 and top <= entity_box(support)[2]+.008)
        for rel in es.relations:
            if rel.type == "supported-by" and rel.b.get("entity") == support.id and rel.a["entity"] != held.id:
                ent = es.find(rel.a["entity"])
                if occupies(ent):
                    others.append(entity_box(ent)[0])
                    on_support.add(ent.id)
        # Contents of a receptacle already on ``support`` (puzzle pieces in a box on the table)
        # also block a flip-in-air drop: the XY spot can sit clear of the box rim and still fall in.
        for rel in es.relations:
            if rel.type == "supported-by" and rel.b.get("entity") in on_support and rel.a["entity"] != held.id:
                ent = es.find(rel.a["entity"])
                if occupies(ent):
                    others.append(entity_box(ent)[0])
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)

        def world(lx: float, ly: float) -> tuple[float, float]:
            return sbox.cx + c * lx - s * ly, sbox.cy + s * lx + c * ly

        if offset is not None:
            spot = world(float(offset[0]), float(offset[1]))
            if not inside(spot):
                raise PredictionError(f"requested offset is within {keep:.2f} m of the edge of {support.id}")
            if clearance <= 0.0:
                # The caller allows contact or overlap (interlocking pieces, stacking): the spot stands.
                return spot, []
            sigma = self._sigma_xy(held)
            hx = max(held_box.hx - 2 * sigma, 0.01) + clearance / 2
            hy = max(held_box.hy - 2 * sigma, 0.01) + clearance / 2
            heading = held_box.yaw if yaw is None else yaw
            grown = []
            for o in others:
                pad = clearance / 2
                ent = es.find(o.id)
                # Flip-in-air drops from turnLift: XY that only grazes a neighbour still lands on it
                # after the fall. Keep a full clearance from every blocker (pieces in a box on the
                # table, the box rim, and anything already on the support), not only receptacles.
                if clearance >= cfg.place_clearance - 1e-9:
                    pad = max(pad, clearance)
                    if (ent is not None
                            and (self._affords(ent, "receptacle") or self._affords(ent, "container"))):
                        pad = max(pad, clearance + 0.06)
                grown.append(Box2D(o.id, o.cx, o.cy, o.yaw, o.hx + pad, o.hy + pad))

            def blocker(xy: tuple[float, float]) -> str | None:
                placed = Box2D(held.id, xy[0], xy[1], heading, hx, hy)
                return next((o.id for o in grown if rects_overlap(placed, o)), None)

            occupant = blocker(spot)
            if occupant is None:
                return spot, []
            step, reach = cfg.spot_shift_step, max(cfg.spot_shift_max, clearance * 2)
            ticks = np.arange(-reach, reach + 1e-9, step)
            shifts = sorted(((float(dx), float(dy)) for dx in ticks for dy in ticks if math.hypot(dx, dy) <= reach),
                            key=lambda d: math.hypot(*d))
            for dx, dy in shifts:
                moved = (spot[0] + dx, spot[1] + dy)
                if not (inside(moved) and blocker(moved) is None):
                    continue
                # Flip-in-air clearance shifts that hug the rim pass feasible but fail the risk
                # gate ("0 cm from the edge"). Skip rim spots and keep searching, or fall through.
                if (clearance >= cfg.place_clearance - 1e-9
                        and self._edge_margin(sbox, moved, ex, ey) < cfg.place_edge_margin):
                    continue
                return moved, [f"requested spot is occupied by {occupant}; using the free spot nearest it, "
                               f"{math.hypot(dx, dy) * 100:.0f} cm away at ({moved[0]:.3f}, {moved[1]:.3f})"]
            # Crowded tables (puzzle pieces still in a box): fall through to the free-spot search
            # instead of refusing the place — the requested offset becomes a soft preference only.
            offset_fallback = (
                f"requested spot is occupied by {occupant} and no free spot lies within {reach:.2f} m; "
                f"using a free spot on {support.id}")
            # continue into the automatic free-spot search below
        else:
            offset_fallback = None
        rx, ry, _ = self._robot_xy_yaw(es)
        # A support narrower than the margin on a side (a bowl onto a plate, a cup onto a saucer)
        # still takes the load wholly on it: there the spot only keeps the load's own projection.
        want_x, want_y = max(cfg.place_edge_margin + ex, keep_x), max(cfg.place_edge_margin + ey, keep_y)
        margin_x = want_x if sbox.hx >= want_x else keep_x
        margin_y = want_y if sbox.hy >= want_y else keep_y
        # How deep into a surface the held object's centre can go from a docked base: the arm's
        # longest reach past the dock, plus how far the object hangs ahead of the gripper.
        lx0, ly0, _, _, _ = self._held_relative(es) or (0.0, 0.0, 0.0, 0.0, 0.0)
        hang = math.hypot(lx0 - self.tool_offset - self._arm_joints(es).get("reach", 0.0), ly0 - self.lateral)
        deepest = self.reach_max + self.tool_offset - self.half_x - cfg.dock_gap + hang
        # Match the flip-in-air grown pads when searching free spots so a fallback after a crowded
        # offset does not land inside the box mouth or on a neighbour the offset path already refused.
        free_pad = clearance if clearance >= cfg.place_clearance - 1e-9 else max(cfg.place_clearance, 0.02)
        best, best_cost = None, math.inf
        for lx in _spots(sbox.hx, margin_x):
            for ly in _spots(sbox.hy, margin_y):
                spot = world(float(lx), float(ly))
                blocked = False
                for o in others:
                    pad = free_pad
                    ent = es.find(o.id)
                    if (clearance >= cfg.place_clearance - 1e-9 and ent is not None
                            and (self._affords(ent, "receptacle") or self._affords(ent, "container"))):
                        pad = max(pad, free_pad + 0.06)
                    if math.hypot(spot[0] - o.cx, spot[1] - o.cy) < (
                            max(held_box.hx, held_box.hy) + max(o.hx, o.hy) + pad):
                        blocked = True
                        break
                if blocked:
                    continue
                edge = min(sbox.hx - abs(lx), sbox.hy - abs(ly))
                # Far from the edges is safer, but only as deep as the arm reaches.
                cost = 0.3 * math.hypot(spot[0] - rx, spot[1] - ry) - 2.0 * min(edge, deepest) \
                    + (10.0 if edge > deepest else 0.0)
                if cost < best_cost:
                    best, best_cost = spot, cost
        if best is None:
            raise PredictionError(offset_fallback or f"no free spot on {support.id}")
        return best, ([offset_fallback] if offset_fallback else [])


def _spots(half: float, margin: float, step: float = 0.05) -> np.ndarray:
    """Candidate offsets along one support axis of half extent ``half``, at least ``margin`` from
    either edge: every ``step``, or the centre alone when the span is shorter than a step."""
    if half - margin < -1e-9:
        return np.zeros(0)
    ticks = np.arange(-half + margin, half - margin + 1e-9, step)
    return ticks if len(ticks) > 1 else np.zeros(1)
