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
from dataclasses import dataclass

import numpy as np

from steward.ego.planner import (
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
)
from steward.math3d import yaw_from_quat
from steward.scene.schema import Entity, SceneState


@dataclass
class EsimConfig:
    capture_radius: float = 0.05
    receipt_ttl: float = 60.0
    dock_gap: float = 0.07
    predock_backoff: float = 0.35
    reach_min: float = 0.08
    reach_max: float = 0.65
    lift_min: float = 0.32
    lift_max: float = 1.28
    place_edge_margin: float = 0.10
    place_clearance: float = 0.14
    spot_shift_max: float = 0.3  # an occupied requested spot moves to the nearest free one at most this far
    spot_shift_step: float = 0.01
    nav_margin: float = 0.03
    floor_furniture_height: float = 0.35
    handle_reach: float = 0.18
    drawer_pull: float = 0.30
    drawer_pull_max: float = 0.33
    drawer_viewing_backoff: float = 0.25
    nest_fit: float = 0.92
    floor_carry_lift: float = 0.05
    floor_release_gap: float = 0.015
    carry_retreat_max: float = 1.2
    park_keep_clear: float = 0.9
    park_spacing: float = 0.5
    wrist_yaw_max: float = 3.1
    wrist_roll_max: float = 2.5
    pour_roll: float = 1.9
    pour_clearance: float = 0.02
    pour_spill: float = 0.25  # how far beyond the container poured contents may slide; the spot keeps that from every edge
    pour_hold_s: float = 1.5


class PredictionError(ValueError):
    pass


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Esim:
    def __init__(self, robot_model: dict, config: EsimConfig | None = None):
        self.config = config or EsimConfig()
        tree = robot_model["body"]
        base = next(link for link in tree["links"] if link["id"] == "base")
        bx, by, _ = base["geometry"]["collision"][0]["size"]
        self.half_x, self.half_y = bx / 2, by / 2
        self.radius = math.hypot(self.half_x, self.half_y)
        lift = next(j for j in tree["joints"] if j["id"] == "lift")
        grip = next(j for j in tree["joints"] if j["id"] == "gripper_mount")
        self.tool_offset = lift["origin"]["value"]["p"][0] + grip["origin"]["value"]["p"][0]
        self._ids = itertools.count(1)
        self._blockers: set[str] = set()

    # ---------------------------------------------------------------- helpers
    def _self(self, es: SceneState) -> Entity:
        return es.entity(es.self_id)

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
            "plan": plan,
        }

    # -------------------------------------------------------------- approach
    def _approach(self, es: SceneState, support: Entity, target_xy: tuple[float, float], height: float,
                  exclude: set[str], load: Entity | None = None) -> dict:
        """Choose the support side to dock at; return path, dock pose and reach, or raise.

        With ``load`` (a floor-standing object about to be picked) a side is only accepted when the
        robot can still turn at the pre-dock pose while carrying the load in front of it, so a
        pick never leaves the robot boxed in."""
        sbox, _, _ = entity_box(support)
        rx, ry, _ = self._robot_xy_yaw(es)
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)
        dx, dy = target_xy[0] - sbox.cx, target_xy[1] - sbox.cy
        tx, ty = c * dx + s * dy, -s * dx + c * dy
        candidates = [((sbox.hx, ty), (1.0, 0.0), sbox.hx - tx), ((-sbox.hx, ty), (-1.0, 0.0), sbox.hx + tx),
                      ((tx, sbox.hy), (0.0, 1.0), sbox.hy - ty), ((tx, -sbox.hy), (0.0, -1.0), sbox.hy + ty)]
        cfg = self.config
        obstacles_all = floor_obstacles(es, exclude)
        # A floor-standing load stays an obstacle for the path to it (only its own back-out ignores it).
        loaded = floor_obstacles(es, exclude - {load.id}) if load is not None else None
        if loaded is not None:
            obstacles_all, load_free = loaded, obstacles_all
        carry = self._carry(es)
        grid = build_grid(obstacles_all, self._sweep(carry) + cfg.nav_margin, [(rx, ry), target_xy])
        best, reasons = None, []
        if not (cfg.lift_min <= height <= cfg.lift_max):
            raise PredictionError(f"height {height:.2f} m is outside the lift range [{cfg.lift_min}, {cfg.lift_max}]")
        for edge, normal, depth in candidates:
            standoff = self.half_x + cfg.dock_gap
            reach = standoff + depth - self.tool_offset
            if not (cfg.reach_min <= reach <= cfg.reach_max):
                reasons.append(f"side {normal}: reach {reach:.2f} m out of range")
                continue
            nx, ny = c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]
            ex, ey = sbox.cx + c * edge[0] - s * edge[1], sbox.cy + s * edge[0] + c * edge[1]
            dock = (ex + nx * standoff, ey + ny * standoff)
            yaw = math.atan2(-ny, -nx)
            predock = (dock[0] + nx * cfg.predock_backoff, dock[1] + ny * cfg.predock_backoff)
            blocked = self._footprint_blocked(obstacles_all, dock, yaw, carry)
            if blocked:
                reasons.append(f"side {normal}: dock pose collides with {blocked}")
                continue
            blocked = self._turn_blocked(obstacles_all, predock, self._sweep(carry))
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
                slide = (self.tool_offset + reach, 0.0, lbox.yaw - yaw, max(lbox.hx - sigma - 0.02, 0.05),
                         max(lbox.hy - sigma - 0.02, 0.05))
                turn = (self.tool_offset + reach, 0.0, lbox.yaw - yaw, max(lbox.hx - sigma, 0.05),
                        max(lbox.hy - sigma, 0.05))
                retreat = self._retreat(load_free, dock, (nx, ny), yaw, slide, cfg.predock_backoff, turn)
                if retreat is None:
                    reasons.append(f"side {normal}: carrying {load.id} from here the robot could not back out to "
                                   "a spot with room to turn")
                    continue
            path = astar(grid, (rx, ry), predock)
            if path is None:
                reasons.append(f"side {normal}: no collision-free path to the pre-dock pose")
                continue
            length = path_length(path)
            if best is None or length < best["pathLength"]:
                best = {"path": [list(p) for p in path], "predock": list(predock), "dock": list(dock), "yaw": yaw,
                        "reach": reach, "pathLength": length}
                if len(path) > 1:
                    # Standing at the pre-dock pose already, there is no path to keep clear.
                    best["clearance"] = path_clearance(path, obstacles_all)
                if retreat is not None:
                    best["retreat"] = list(retreat)
        if best is None:
            hint = self._boxed_in(es, obstacles_all, carry)
            raise PredictionError("no feasible approach: " + "; ".join(reasons) + (f"; {hint}" if hint else ""))
        return best

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
        movable and come with parking spots out of everyone's way; tableware lying on the floor is
        not, since the lift cannot reach down to it."""
        held = self._held(es)
        out = []
        for eid in sorted(self._blockers):
            ent = es.find(eid)
            if ent is None or ent.role == "environment" or ent.id == es.self_id:
                continue
            _, bottom, _ = entity_box(ent)
            movable = (self._affords(ent, "graspable") and self._affords(ent, "floor-standing") and bottom < 0.05
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

    def _carry(self, es: SceneState) -> tuple | None:
        """The held object's footprint when it is carried low enough to hit floor obstacles (a chair
        lifted a few centimetres); objects carried above furniture height are ignored."""
        held = self._held(es)
        if held is None or entity_box(held)[1] >= self.config.floor_furniture_height:
            return None
        return self._held_relative(es)

    def _retreat(self, obstacles: list[Box2D], start: tuple[float, float], direction: tuple[float, float],
                 yaw: float, carry: tuple, min_d: float, turn_carry: tuple | None = None) -> tuple[float, float] | None:
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

    def _boxed_in(self, es: SceneState, obstacles: list[Box2D], carry: tuple | None) -> str | None:
        """When a carried load leaves no room to turn where the robot stands, say how to get out:
        put the load down where it is."""
        if carry is None:
            return None
        rx, ry, ryaw = self._robot_xy_yaw(es)
        blocked = self._turn_blocked(obstacles, (rx, ry), self._sweep(carry))
        if blocked is None:
            return None
        held = self._held(es)
        box, _, _ = entity_box(held)
        return (f"carrying {held.id} the robot has no room to turn here ({blocked}); put it down where it is with "
                f"place at=[{box.cx:.2f}, {box.cy:.2f}] yaw={box.yaw:.2f}, then move without it")

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
            raise PredictionError(f"unknown skill {skill!r}; known: navigate, pick, place, pour, open, close")
        except PredictionError as exc:
            blockers = self._describe_blockers(es)
            issues = [str(exc)] + [
                f"{b['id']} ({b['category']}) is in the way and can be moved: pick it, put it down out of the way "
                f"(for example place at={b['parkAt'][0]}), then predict again"
                for b in blockers if b["movable"] and b["parkAt"]]
            receipt = self._receipt(es, skill, params, 1.0, issues, {}, feasible=False)
            receipt["blockers"] = blockers
            return receipt

    def _predict_navigate(self, es: SceneState, params: dict) -> dict:
        rx, ry, ryaw = self._robot_xy_yaw(es)
        goal = (float(params["x"]), float(params["y"]))
        yaw = float(params.get("yaw", ryaw))
        held = self._held(es)
        obstacles = floor_obstacles(es, {held.id} if held is not None else set())
        carry = self._carry(es)
        grid = build_grid(obstacles, self._sweep(carry) + self.config.nav_margin, [(rx, ry), goal])
        blocked = self._footprint_blocked(obstacles, goal, yaw, carry)
        if blocked:
            raise PredictionError(f"goal pose collides with {blocked}")
        path = astar(grid, (rx, ry), goal)
        if path is None:
            hint = self._boxed_in(es, obstacles, carry)
            raise PredictionError("no collision-free path to the goal" + (f"; {hint}" if hint else ""))
        clearance = path_clearance(path, obstacles) if len(path) > 1 else math.inf
        issues, risk = [], 0.02
        if clearance < self._sweep(carry) + 0.05:
            issues.append(f"path passes within {clearance:.2f} m of an obstacle")
            risk = 0.2
        plan = {"path": [list(p) for p in path], "yaw": yaw, "pathLength": path_length(path)}
        return self._receipt(es, "navigate", params, risk, issues, plan)

    def _predict_pick(self, es: SceneState, params: dict) -> dict:
        target = es.find(params.get("target", ""))
        if target is None:
            raise PredictionError(f"unknown target {params.get('target')!r}")
        if self._held(es) is not None:
            raise PredictionError("the gripper is already holding an object")
        if not self._affords(target, "graspable"):
            raise PredictionError(f"{target.id} ({target.category.value}) is not graspable")
        support = self._support_of(es, target)
        _, target_bottom, _ = entity_box(target)
        on_floor = support is None and target_bottom < 0.05
        if support is None and not on_floor:
            raise PredictionError(f"no support surface is known under {target.id}; observe it first")
        tp = target.pose.value["p"]
        contents = self._contents(es, target)
        if self._affords(target, "pourable"):
            covering = self._covering(es, target)
            if covering:
                raise PredictionError(f"{', '.join(covering)} lies on top of {target.id}; take it off first")
        elif contents:
            raise PredictionError(f"{target.id} holds {', '.join(e.id for e in contents)}; take those out first")
        anchor = target if on_floor else self._anchor(es, support)
        approach = self._approach(es, anchor, (tp[0], tp[1]), tp[2], {target.id}, load=target if on_floor else None)
        sigma = math.hypot(self._sigma_xy(target), self._sigma_xy(self._self(es)))
        r = self.config.capture_radius
        p_miss = math.exp(-(r * r) / (2 * sigma * sigma)) if sigma > 0 else 0.0
        issues = []
        if p_miss > 0.05:
            issues.append(f"target position uncertainty {sigma * 100:.1f} cm gives {p_miss:.0%} miss probability; "
                          "observe the target more closely first")
        existence_risk = 1.0 - target.existence
        if existence_risk > 0.2:
            issues.append(f"target existence probability is only {target.existence:.2f}")
        risk = 1.0 - (1.0 - p_miss) * (1.0 - existence_risk)
        carry_lift = tp[2] + (self.config.floor_carry_lift if on_floor else 0.10)
        plan = {**approach, "target": target.id, "support": "floor" if on_floor else support.id, "lift": tp[2],
                "carryLift": min(self.config.lift_max, carry_lift), "captureRadius": self.config.capture_radius,
                "gentle": self._affords(target, "pourable")}
        return self._receipt(es, "pick", params, risk, issues, plan)

    def _predict_place(self, es: SceneState, params: dict) -> dict:
        held = self._held(es)
        if held is None:
            raise PredictionError("the gripper is not holding anything")
        support = es.find(params.get("on", "")) if "on" in params else None
        if support is None and "on" in params:
            raise PredictionError(f"unknown support {params.get('on')!r}")
        held_box, hb, ht = entity_box(held)
        held_h = ht - hb
        held_half = max(held_box.hx, held_box.hy)
        if support is None and "at" in params:
            return self._predict_floor_place(es, held, params)
        if support is None:
            raise PredictionError(f"unknown support {params.get('on')!r}")
        if self._affords(support, "receptacle"):
            if self._affords(support, "pullable"):
                spot, area, bottom, issues = self._drawer_spot(es, support, held, held_half)
            else:
                spot, area, bottom, issues = self._nest_spot(es, support, held, held_half)
            height = bottom + 0.012 + held_h / 2 + 0.02
            sigma = math.hypot(self._sigma_xy(support), self._sigma_xy(self._self(es)))
            clearance = min(area.hx, area.hy) - held_half
            risk = 0.03 if clearance >= 2 * sigma + 0.003 else 0.4
            if risk > 0.1:
                issues.append(f"only {clearance * 100:.1f} cm of clearance with {sigma * 100:.1f} cm uncertainty")
        elif self._affords(support, "support-surface"):
            sbox, _, top = entity_box(support)
            final_yaw = float(params["yaw"]) if params.get("yaw") is not None else held_box.yaw
            spot, issues = self._choose_spot(es, support, sbox, held, held_box, params.get("offset"),
                                             final_yaw, float(params.get("clearance", 0.015)))
            area = sbox
            height = top + held_h / 2 + 0.02
            edge = self._edge_distance(sbox, spot) - held_half
            sigma = self._sigma_xy(support)
            risk = 0.02
            if edge < 2 * sigma + 0.03:
                issues.append(f"placement is {edge * 100:.0f} cm from the edge with {sigma * 100:.1f} cm support uncertainty")
                risk = 0.4
        else:
            raise PredictionError(f"{support.id} ({support.category.value}) is neither a surface nor a receptacle")
        approach = self._approach(es, self._anchor(es, support), spot, height, {held.id})
        fragility = (held.rigid.links[0].material.value or {}).get("fragility", 0.0) if held.rigid.links[0].material else 0.0
        risk = 1.0 - (1.0 - risk) * (1.0 - 0.05 * fragility)
        plan = {**approach, "object": held.id, "support": support.id, "spot": list(spot), "lift": height,
                "supportBox": {"cx": area.cx, "cy": area.cy, "yaw": area.yaw, "hx": area.hx, "hy": area.hy},
                "heldHalf": held_half}
        if params.get("yaw") is not None:
            plan["wristYaw"] = self._wrist_for(es, held, float(params["yaw"]), approach["yaw"], bool(params.get("flip")))
        if params.get("flip"):
            plan["wristRoll"] = math.pi  # turned over about the arm axis before it is set down
        plan["gentle"] = self._affords(held, "pourable")
        return self._receipt(es, "place", params, risk, issues, plan)

    def _wrist_for(self, es: SceneState, held: Entity, yaw: float, dock_yaw: float, flip: bool = False) -> float:
        """Wrist yaw that sets the held object down at heading ``yaw`` with the base at ``dock_yaw``.
        Turning it over (``flip``: a half roll about the gripper axis at heading a) maps a heading h
        to 2a - h, so the object's offset from the gripper axis changes sign."""
        _, _, ryaw = self._robot_xy_yaw(es)
        wrist = next((j.q.value[0] for j in self._self(es).rigid.joints if j.id == "wrist_yaw" and j.q.value), 0.0)
        relative = yaw_from_quat(held.pose.value["q"]) - ryaw - wrist
        target = _wrap(yaw - dock_yaw + relative) if flip else _wrap(yaw - dock_yaw - relative)
        if abs(target) > self.config.wrist_yaw_max:
            raise PredictionError(f"heading {math.degrees(yaw):.0f} deg needs a wrist turn of {math.degrees(target):.0f} deg, "
                                  f"beyond +-{math.degrees(self.config.wrist_yaw_max):.0f}")
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
        # Contents slide out up to pour_spill beyond the container: the spot keeps them on the surface.
        spill = max(held_box.hx, held_box.hy) + self.config.pour_spill
        if min(sbox.hx, sbox.hy) < spill:
            raise PredictionError(f"{support.id} is too small to pour {held.id} on: the contents could slide over "
                                  f"its edge (a spot {spill:.2f} m from every edge is needed)")
        spot, issues = self._choose_spot(es, support, sbox, held, held_box, params.get("offset"), held_box.yaw,
                                         float(params.get("clearance", 0.015)), edge_keep=spill)
        height = top + (ht - hb) / 2 + self.config.pour_clearance
        approach = self._approach(es, self._anchor(es, support), spot, height, {held.id})
        roll = float(params.get("roll", self.config.pour_roll))
        if not (0.3 <= abs(roll) <= self.config.wrist_roll_max):
            raise PredictionError(f"roll {roll:.2f} rad is outside [0.3, {self.config.wrist_roll_max}]")
        # The container tips about its edge across the arm: half its extent across the docked heading.
        rel = held_box.yaw - approach["yaw"]
        pivot = abs(math.sin(rel)) * held_box.hx + abs(math.cos(rel)) * held_box.hy
        if height + pivot > self.config.lift_max:
            raise PredictionError(f"tipping {held.id} about its edge needs a lift of {height + pivot:.2f} m, "
                                  f"above {self.config.lift_max} m")
        plan = {**approach, "object": held.id, "support": support.id, "spot": list(spot), "lift": height,
                "roll": roll, "pivot": pivot, "pourS": self.config.pour_hold_s, "gentle": True}
        return self._receipt(es, "pour", params, 0.1, issues, plan)

    def _predict_floor_place(self, es: SceneState, held: Entity, params: dict) -> dict:
        """Put the held object down on the floor with its centre at ``at``: stand behind the spot
        facing it, drive straight in from a pre-pose, lower the object to just above the floor and
        release. ``yaw`` (radians) sets the object's final heading; otherwise the robot tries
        several headings, starting with the direction it is coming from."""
        cfg = self.config
        try:
            tx, ty = (float(v) for v in params["at"][:2])
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"at must be [x, y] in metres: {exc}") from None
        rel = self._held_relative(es)
        lx, ly, dyaw, _, _ = rel
        rx, ry, ryaw = self._robot_xy_yaw(es)
        if "yaw" in params:
            headings = [float(params["yaw"]) - dyaw]
        else:
            toward = math.atan2(ty - ry, tx - rx)
            headings = [toward + k * math.pi / 4 for k in (0, 1, -1, 2, -2, 3, -3, 4)]
        me = self._self(es)
        lift_now = next((j.q.value[0] for j in me.rigid.joints if j.id == "lift" and j.q.value), None)
        _, held_bottom, _ = entity_box(held)
        if lift_now is None:
            raise PredictionError("the lift position is unknown")
        lift = lift_now - (held_bottom - cfg.floor_release_gap)
        if not (cfg.lift_min <= lift <= cfg.lift_max):
            raise PredictionError(f"lowering {held.id} to the floor needs lift {lift:.2f} m, outside the lift range")
        obstacles = floor_obstacles(es, {held.id})
        held_box, _, _ = entity_box(held)
        here_yaw = held_box.yaw if "yaw" not in params else float(params["yaw"])
        if math.hypot(tx - held_box.cx, ty - held_box.cy) <= 0.05 and abs(_wrap(here_yaw - held_box.yaw)) <= 0.1:
            # Put it down where it is: no motion, so no path or turning room is needed.
            plan = {"path": [[rx, ry]], "predock": [rx, ry], "dock": [rx, ry], "yaw": ryaw, "lift": lift,
                    "object": held.id, "floor": [held_box.cx, held_box.cy], "pathLength": 0.0}
            return self._receipt(es, "place", params, 0.02, [], plan)
        sweep = self._sweep(rel)
        grid = build_grid(obstacles, sweep + cfg.nav_margin, [(rx, ry), (tx, ty)])
        reasons = []
        for h in headings:
            c, s = math.cos(h), math.sin(h)
            dock = (tx - (c * lx - s * ly), ty - (s * lx + c * ly))
            blocked = self._footprint_blocked(obstacles, dock, h, rel)
            if blocked:
                reasons.append(f"heading {math.degrees(h):.0f} deg: {held.id} or the robot would hit {blocked}")
                continue
            predock = self._retreat(obstacles, dock, (-c, -s), h, rel, cfg.predock_backoff)
            if predock is None:
                reasons.append(f"heading {math.degrees(h):.0f} deg: no straight approach with room to turn while "
                               f"carrying {held.id}")
                continue
            path = astar(grid, (rx, ry), predock)
            if path is None:
                reasons.append(f"heading {math.degrees(h):.0f} deg: no collision-free path while carrying {held.id}")
                continue
            plan = {"path": [list(q) for q in path], "predock": list(predock), "dock": list(dock), "yaw": h,
                    "lift": lift, "object": held.id, "floor": [tx, ty], "pathLength": path_length(path)}
            return self._receipt(es, "place", params, 0.03, [], plan)
        hint = self._boxed_in(es, obstacles, rel)
        raise PredictionError(f"cannot put {held.id} down at ({tx:.2f}, {ty:.2f}): " + "; ".join(reasons[:4])
                              + (f"; {hint}" if hint else ""))

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
        n = (math.cos(box.yaw), math.sin(box.yaw))
        handle = [box.cx + n[0] * (box.hx - 0.015), box.cy + n[1] * (box.hx - 0.015), (bottom + top) / 2]
        opening = self._opening(es, drawer)
        if skill == "open":
            if opening > 0.2:
                raise PredictionError(f"{drawer.id} is already open ({opening * 100:.0f} cm)")
            distance = min(max(float(params.get("distance", cfg.drawer_pull)), 0.05), cfg.drawer_pull_max)
            sign = 1.0
        else:
            if opening < 0.05:
                raise PredictionError(f"{drawer.id} is already closed")
            distance = min(opening + 0.02, cfg.drawer_pull_max + 0.04)
            sign = -1.0
        standoff = self.tool_offset + cfg.handle_reach
        dock = (handle[0] + n[0] * standoff, handle[1] + n[1] * standoff)
        end = (dock[0] + sign * n[0] * distance, dock[1] + sign * n[1] * distance)
        clear = (end[0] + n[0] * cfg.drawer_viewing_backoff, end[1] + n[1] * cfg.drawer_viewing_backoff)
        predock = (dock[0] + n[0] * cfg.predock_backoff, dock[1] + n[1] * cfg.predock_backoff)
        yaw = math.atan2(-n[1], -n[0])
        if not (cfg.lift_min <= handle[2] <= cfg.lift_max):
            raise PredictionError(f"handle height {handle[2]:.2f} m is outside the lift range")
        obstacles = floor_obstacles(es, set())
        for name, xy in (("dock", dock), ("end", end), ("viewing", clear)):
            blocked = self._footprint_blocked(obstacles, xy, yaw)
            if blocked:
                raise PredictionError(f"{name} pose in front of {drawer.id} collides with {blocked}")
        blocked = self._turn_blocked(obstacles, predock)
        if blocked:
            raise PredictionError(f"no room to turn in front of {drawer.id} ({blocked})")
        rx, ry, _ = self._robot_xy_yaw(es)
        grid = build_grid(obstacles, self.radius + cfg.nav_margin, [(rx, ry), predock])
        path = astar(grid, (rx, ry), predock)
        if path is None:
            raise PredictionError(f"no collision-free path to the front of {drawer.id}")
        sigma = math.hypot(self._sigma_xy(drawer), self._sigma_xy(self._self(es)))
        r = cfg.capture_radius
        p_miss = math.exp(-(r * r) / (2 * sigma * sigma)) if sigma > 0 else 0.0
        issues = [f"handle position uncertainty {sigma * 100:.1f} cm gives {p_miss:.0%} miss probability"] if p_miss > 0.05 else []
        plan = {"path": [list(q) for q in path], "predock": list(predock), "dock": list(dock), "yaw": yaw,
                "end": list(end), "viewing": list(clear), "handle": handle, "distance": distance, "direction": sign,
                "axis": list(n), "drawer": drawer.id, "captureRadius": r, "pathLength": path_length(path)}
        return self._receipt(es, skill, params, max(0.03, p_miss), issues, plan)

    @staticmethod
    def _edge_distance(sbox: Box2D, xy: tuple[float, float]) -> float:
        dx, dy = xy[0] - sbox.cx, xy[1] - sbox.cy
        c, s = math.cos(-sbox.yaw), math.sin(-sbox.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return min(sbox.hx - abs(lx), sbox.hy - abs(ly))

    def _choose_spot(self, es: SceneState, support: Entity, sbox: Box2D, held: Entity, held_box: Box2D,
                     offset: list | None, yaw: float | None = None,
                     clearance: float = 0.015, edge_keep: float | None = None) -> tuple[tuple[float, float], list[str]]:
        """A spot for ``held`` on ``support``: ``offset`` in the support frame if given, else the free
        spot nearest the robot. A requested spot where the held footprint, turned to ``yaw``, would
        come within ``clearance`` of another object moves to the free spot nearest it, at most
        ``spot_shift_max`` away, and an issue says where; none that near is an error. Every spot
        keeps ``edge_keep`` (default: the held object's half extent) from the support's edges."""
        cfg = self.config
        keep = max(held_box.hx, held_box.hy) if edge_keep is None else edge_keep
        others = []
        for rel in es.relations:
            if rel.type == "supported-by" and rel.b.get("entity") == support.id and rel.a["entity"] != held.id:
                ent = es.find(rel.a["entity"])
                if ent is not None:
                    others.append(entity_box(ent)[0])
        c, s = math.cos(sbox.yaw), math.sin(sbox.yaw)

        def world(lx: float, ly: float) -> tuple[float, float]:
            return sbox.cx + c * lx - s * ly, sbox.cy + s * lx + c * ly

        if offset is not None:
            spot = world(float(offset[0]), float(offset[1]))
            if self._edge_distance(sbox, spot) < keep:
                raise PredictionError(f"requested offset is within {keep:.2f} m of the edge of {support.id}")
            if clearance <= 0.0:
                # The caller allows contact or overlap (interlocking pieces, stacking): the spot stands.
                return spot, []
            sigma = self._sigma_xy(held)
            hx = max(held_box.hx - 2 * sigma, 0.01) + clearance / 2
            hy = max(held_box.hy - 2 * sigma, 0.01) + clearance / 2
            heading = held_box.yaw if yaw is None else yaw
            grown = [Box2D(o.id, o.cx, o.cy, o.yaw, o.hx + clearance / 2, o.hy + clearance / 2) for o in others]

            def blocker(xy: tuple[float, float]) -> str | None:
                placed = Box2D(held.id, xy[0], xy[1], heading, hx, hy)
                return next((o.id for o in grown if rects_overlap(placed, o)), None)

            occupant = blocker(spot)
            if occupant is None:
                return spot, []
            step, reach = cfg.spot_shift_step, cfg.spot_shift_max
            ticks = np.arange(-reach, reach + 1e-9, step)
            shifts = sorted(((float(dx), float(dy)) for dx in ticks for dy in ticks if math.hypot(dx, dy) <= reach),
                            key=lambda d: math.hypot(*d))
            for dx, dy in shifts:
                moved = (spot[0] + dx, spot[1] + dy)
                if self._edge_distance(sbox, moved) >= keep and blocker(moved) is None:
                    return moved, [f"requested spot is occupied by {occupant}; using the free spot nearest it, "
                                   f"{math.hypot(dx, dy) * 100:.0f} cm away at ({moved[0]:.3f}, {moved[1]:.3f})"]
            raise PredictionError(f"requested spot is occupied by {occupant} and no free spot lies within "
                                  f"{reach:.2f} m of it")
        rx, ry, _ = self._robot_xy_yaw(es)
        margin = max(cfg.place_edge_margin + max(held_box.hx, held_box.hy), keep)
        best, best_cost = None, math.inf
        for lx in np.arange(-sbox.hx + margin, sbox.hx - margin + 1e-9, 0.05):
            for ly in np.arange(-sbox.hy + margin, sbox.hy - margin + 1e-9, 0.05):
                spot = world(float(lx), float(ly))
                if any(math.hypot(spot[0] - o.cx, spot[1] - o.cy)
                       < max(cfg.place_clearance, max(held_box.hx, held_box.hy) + max(o.hx, o.hy) + 0.02) for o in others):
                    continue
                edge = min(sbox.hx - abs(lx), sbox.hy - abs(ly))
                cost = 0.3 * math.hypot(spot[0] - rx, spot[1] - ry) - 2.0 * edge
                if cost < best_cost:
                    best, best_cost = spot, cost
        if best is None:
            raise PredictionError(f"no free spot on {support.id}")
        return best, []
