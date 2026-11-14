#!/usr/bin/env python3
"""Headless, no robot controller: PYTHONPATH=src python scripts/fasten_demo.py.

Uses the upstream toy-assembly specification and fresh generator assets, with an
invisible, fixed grip fixture in place of a robot. Directly setting its pose and
_GripState simulates a held electric screwdriver; no fasten action is injected.
The lift/flip test clamps ONLY the fuselage by pose, so the wing must follow via
MuJoCo weld forces (not the ideal-grasp component-follow code). No rendering.
"""
from __future__ import annotations

import math
import tempfile

import mujoco
import numpy as np

from steward.assets import generators
from steward.assets.library import instantiate
from steward.authoring.service import (ASSEMBLY_SHEET_M, AuthoringService, TOY_ASSEMBLY_ASSETS,
                                       airplane_manual_texture, airplane_finished_texture,
                                       car_manual_texture, car_finished_texture, toy_assembly_spec)
from steward.math3d import matrix_from_quat, pose_compose, pose_inverse, quat_from_axis_angle
from steward.scene.fasteners import STATE_KEY, FastenConfig
from steward.scene.schema import Entity, Link, RigidTree, SceneState, exact, pose
from steward.world.sim import WorldSim

BASE, WING, TOOL = 'w:plane-fuselage', 'w:plane-wing', 'w:screwdriver'
SCREWS = ['w:plane-screw-0', 'w:plane-screw-1']
FIXTURE = 'w:grip-fixture'


def toy_scene():
    """Compose the actual workshop without publish_asset's thumbnail renderer."""
    sheets = {'airplane-manual': ('manual', airplane_manual_texture),
              'airplane-finished': ('picture', airplane_finished_texture),
              'car-manual': ('manual', car_manual_texture),
              'car-finished': ('picture', car_finished_texture)}
    scene = SceneState(mode='world')
    with tempfile.TemporaryDirectory(prefix='fasten-demo-') as root:
        author = AuthoringService(root)
        for placement in toy_assembly_spec()['entities']:
            asset = placement['asset']
            if asset == 'steward-bot':
                continue
            if asset in sheets:
                category, texture = sheets[asset]
                template = generators.printed_board(width=ASSEMBLY_SHEET_M[0], depth=ASSEMBLY_SHEET_M[1],
                    thickness=.004, texture=texture(), mass=.04, category=category)
            else:
                generator, params = TOY_ASSEMBLY_ASSETS[asset]
                template = getattr(generators, generator)(**params)
            scene.entities.append(instantiate(template.to_dict(), placement['id'],
                                               author._placement_pose(scene, placement)))
    scene.entities.append(Entity(id=FIXTURE, role='robot', category=exact('test-fixture'), pose=exact(pose()),
        body=RigidTree(root=exact('fixed'), links=[Link(id='grip', name='invisible grip fixture',
                                                       affordances=[exact('grasp-point')])])))
    return scene.to_dict()


def set_pose(sim, eid, target):
    sim._set_root_pose(sim.template.entity(eid), target)
    mujoco.mj_forward(sim.model, sim.data)


def grip(sim):
    rs = sim.robots[FIXTURE]
    return rs, rs.grippers['grip']


def release(sim):
    rs, g = grip(sim)
    sim._release(rs, g)
    g.closed = False


def hold(sim, eid, target):
    """Internal ideal-grip fixture; not a robot sensor/API or a fasten shortcut."""
    rs, g = grip(sim)
    if g.attached != eid:
        release(sim)
        g.attached, g.rel, g.closed = eid, pose(), True
        sim._suspend_contacts(rs, g)
    set_pose(sim, FIXTURE, target)
    sim._follow_attached()
    mujoco.mj_forward(sim.model, sim.data)


def feature_world(sim, kind, eid, hole=None):
    feature = next(f for f in sim._fastening.inventory[kind]
                   if f.entity == eid and (hole is None or f.spec['id'] == hole))
    return feature.at(sim._fastening.pose(feature))


def prepare(sim):
    sim.step_seconds(.5)
    base = sim._entity_pose(BASE)
    # Match the two wing holes. The marker protrudes 0.5 mm above the fuselage box.
    set_pose(sim, WING, pose_compose(base, pose([.02, 0, .041])))
    sim.step_seconds(.5)
    for i, eid in enumerate(SCREWS):
        hole = feature_world(sim, 'screw-hole', WING, f'mount_{i}')
        set_pose(sim, eid, pose(hole['entry'], sim._entity_pose(WING)['q']))
    sim.step_seconds(.2)


def tool_pose(sim, screw):
    sc = feature_world(sim, 'screw', screw)
    # Local +x points down the screw, local +z follows screw-local +x.
    q = pose_compose(pose(q=sc['pose']['q']), pose(q=quat_from_axis_angle([0, 1, 0], math.pi/2)))['q']
    bit = next(f.spec['tip'] for f in sim._fastening.inventory['powered-driver'] if f.entity == TOOL)
    return pose(sc['head'] - matrix_from_quat(q) @ np.asarray(bit), q)


def tighten(sim, screw, *, held=True, seconds=.7):
    # Following the observed-in-World head is a deterministic test fixture only.
    # Production detection still checks real world poses and real grip state.
    for _ in range(math.ceil(seconds / sim.timestep)):
        target = tool_pose(sim, screw)
        if held:
            hold(sim, TOOL, target)
        else:
            release(sim)
            set_pose(sim, TOOL, target)
        sim.step()
        if STATE_KEY in sim.template.entity(screw).state_vars:
            return True
    return False


def relative(sim):
    return pose_compose(pose_inverse(sim._entity_pose(BASE)), sim._entity_pose(WING))


def flip_and_measure(sim):
    release(sim)
    set_pose(sim, TOOL, pose([0, 0, 1]))  # keep the tool away from the experiment
    initial = relative(sim)
    base = sim._entity_pose(BASE)
    lifted = dict(base, p=[base['p'][0], base['p'][1], base['p'][2] + .55])
    # Only the base is externally manipulated. The wing and screws remain dynamic.
    for n in range(1, math.ceil(1.5 / sim.timestep) + 1):
        t = min(n * sim.timestep, 1.5)
        lift = min(t / .5, 1.)
        angle = math.pi * max(0., t - .5)
        q = pose_compose(pose(q=base['q']), pose(q=quat_from_axis_angle([0, 1, 0], angle)))['q']
        target = pose([base['p'][0], base['p'][1], base['p'][2] + .55 * lift], q)
        set_pose(sim, BASE, target)
        sim.step()
    target = pose(lifted['p'], pose_compose(pose(q=base['q']), pose(q=[0, 0, 1, 0]))['q'])
    # Measure worst drift throughout the requested three-second hold, not just its last frame.
    drift, angular = 0., 0.
    for _ in range(math.ceil(3.0 / sim.timestep)):
        set_pose(sim, BASE, target)
        sim.step()
        set_pose(sim, BASE, target)
        after = relative(sim)
        drift = max(drift, float(np.linalg.norm(np.asarray(after['p']) - initial['p'])))
        angular = max(angular, 2 * math.acos(min(1., abs(float(np.dot(after['q'], initial['q']))))))
    drop = sim._entity_pose(BASE)['p'][2] - sim._entity_pose(WING)['p'][2]
    return drift, angular, drop


def run_case(scene, fastened):
    sim = WorldSim(scene, fasten=FastenConfig(mode="driver" if fastened else "off"))
    try:
        prepare(sim)
        if fastened:
            for screw in SCREWS:
                assert tighten(sim, screw), f'no fasten event for {screw}'
            assert WING in sim._fastening.component(BASE)
            assert all(s in sim._fastening.component(BASE) for s in SCREWS)
            assert sum(e['type'] == 'fasten' for e in sim.events) == 2
        else:
            sim.step_seconds(1.4)  # same waiting interval without a held driver
            assert not sim._fastening.welds
        drift, angular, drop = flip_and_measure(sim)
        ok = drift < .005 if fastened else drift > .05 and drop > .1
        label = 'fastened' if fastened else 'unfastened control'
        print(f'{"PASS" if ok else "FAIL"} {label}: max relative drift={drift*1000:.3f} mm, '
              f'rotation drift={math.degrees(angular):.3f} deg, wing below base={drop*1000:.3f} mm', flush=True)
        assert ok, label
        return drift
    finally:
        sim.stop()


def main():
    try:
        scene = toy_scene()
        # Always run both cases, even if one fails.
        failed = False
        for fastened in (True, False):
            try:
                run_case(scene, fastened)
            except Exception as exc:
                failed = True
                print(f'FAIL {"fastened" if fastened else "control"}: {type(exc).__name__}: {exc}', flush=True)
        return int(failed)
    except Exception as exc:
        print(f'FAIL setup: {type(exc).__name__}: {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
