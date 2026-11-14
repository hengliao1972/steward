"""World-only adapter for fastener detection and activation of preallocated welds."""
from __future__ import annotations

import mujoco
import numpy as np

from steward.math3d import pose_compose, pose_inverse
from steward.scene.fasteners import (STATE_KEY, Dwell, active_welds, component, driving, features,
                                     mating, occupied, records, seated_pose, stationary)
from steward.scene.schema import exact


class Fastening:
    def __init__(self, sim):
        self.sim = sim
        self.inventory = features(sim.template)
        self.dwell = Dwell()
        # Removal of either part invalidates that screw's whole fastener record.
        valid = records(sim.template)
        for ent in sim.template.entities:
            if ent.id not in valid:
                ent.state_vars.pop(STATE_KEY, None)
        self.welds = active_welds(sim.template) if sim.fasten.mode != "off" else {}
        self.slots = {pair: mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
                      for pair, name in sim.build.fastener_welds.items()}

    def component(self, entity):
        return component(entity, self.welds)

    def targets(self, entity, target):
        """Ideal-grasp poses from stored transforms, independent of solver drift."""
        poses = {entity: target}
        pending = [entity]
        while pending:
            current = pending.pop()
            for (a, b), rel in self.welds.items():
                if current == b:
                    a, b, rel = b, a, pose_inverse(rel)
                if current == a and b not in poses:
                    poses[b] = pose_compose(poses[a], rel)
                    pending.append(b)
        return poses

    def pose(self, feature):
        sim = self.sim
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY,
                               sim.build.names[feature.entity].bodies[feature.link])
        return {"p": sim.data.xpos[bid].tolist(), "q": sim.data.xquat[bid].tolist()}

    def is_stationary(self, entity):
        bid = next(iter(self.sim.build.names[entity].bodies.values()))
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, bid)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY,
                                bid, velocity, 0)
        return stationary(velocity, self.sim.fasten)

    def update(self):
        sim, cfg = self.sim, self.sim.fasten
        if cfg.mode == "off" or not self.slots:
            return
        held = {g.attached for rs in sim.robots.values() for g in rs.grippers.values() if g.attached and g.closed}
        drivers = [f.at(self.pose(f)) for f in self.inventory["powered-driver"] if f.entity in held]
        if cfg.mode == "seat":
            drivers = [None]
        if not drivers:
            self.dwell.update([], sim.time, cfg.dwell)
            return
        used = occupied(sim.template)
        holes = [f.at(self.pose(f)) for f in self.inventory["screw-hole"] if f.key not in used]
        candidates = {}
        for feature in self.inventory["screw"]:
            if feature.entity in held or STATE_KEY in sim.template.entity(feature.entity).state_vars:
                continue
            screw = feature.at(self.pose(feature))
            for driver in drivers:
                if driver is not None and not driving(driver, screw, cfg):
                    continue
                for upper in holes:
                    for lower in holes:
                        if not mating(upper, lower, screw, cfg):
                            continue
                        # Never join separately held components: two ideal grasps would fight.
                        joined = self.component(upper["entity"]) | self.component(lower["entity"])
                        if len(held & joined) > 1:
                            continue
                        if cfg.mode == "seat" and not all(self.is_stationary(eid) for eid in
                                (feature.entity, upper["entity"], lower["entity"])):
                            continue
                        key = (driver["entity"] if driver else None, feature.entity, upper["key"], lower["key"])
                        candidates[key] = (feature, screw, upper, lower)
        ready = self.dwell.update(candidates, sim.time, cfg.dwell)
        # With centimetre tolerances adjacent bores can both qualify. Prefer the
        # closest screw-to-upper and upper-to-lower pair, independent of ID order.
        def distance(key):
            _, screw, upper, lower = candidates[key]
            return (np.linalg.norm(screw["tip"] - upper["entry"])
                    + np.linalg.norm(upper["entry"] - lower["entry"]
                                     - upper["depth"] * upper["axis"]))
        for key in sorted(ready, key=distance):
            feature, screw, upper, lower = candidates[key]
            if STATE_KEY in sim.template.entity(feature.entity).state_vars:
                continue
            used = occupied(sim.template)
            if upper["key"] in used or lower["key"] in used:
                continue
            self.attach(key[0], feature, screw, upper, lower)

    def attach(self, driver, feature, screw, upper, lower):
        sim = self.sim
        link_target = seated_pose(feature, upper, screw)
        local_link = pose_compose(pose_inverse(sim._entity_pose(feature.entity)), screw["pose"])
        sim._set_root_pose(sim.template.entity(feature.entity), pose_compose(link_target, pose_inverse(local_link)))
        # Disable the seated screw before forwarding: it is embedded in solid markers.
        for gid in range(sim.model.ngeom):
            if sim._body_entity.get(int(sim.model.geom_bodyid[gid])) == feature.entity:
                sim.model.geom_contype[gid] = sim.model.geom_conaffinity[gid] = 0
        mujoco.mj_forward(sim.model, sim.data)
        welds = []
        for pair in (tuple(sorted((upper["entity"], lower["entity"]))),
                     tuple(sorted((upper["entity"], feature.entity)))):
            a, b = pair
            rel = self.welds.get(pair)
            if rel is None:
                rel = pose_compose(pose_inverse(sim._entity_pose(a)), sim._entity_pose(b))
                self.welds[pair] = rel
                eq = self.slots[pair]
                # MuJoCo weld eq_data: anchor(3), relative position(3), quaternion(4), torque scale.
                sim.model.eq_data[eq, :3] = 0
                sim.model.eq_data[eq, 3:10] = [*rel["p"], *rel["q"]]
                sim.data.eq_active[eq] = True
            welds.append({"a": a, "b": b, "rel": rel})
        holes = [{"entity": h["entity"], "link": h["link"], "hole": h["key"][2]} for h in (upper, lower)]
        sim.template.entity(feature.entity).state_vars[STATE_KEY] = exact({"holes": holes, "welds": welds})
        for rs in sim.robots.values():
            for g in rs.grippers.values():
                if g.attached:
                    sim._suspend_contacts(rs, g)
        sim.events.append({"t": sim.time, "type": "fasten", "screw": feature.entity,
                           "tool": driver, "holes": holes, "mode": sim.fasten.mode})
        mujoco.mj_forward(sim.model, sim.data)
