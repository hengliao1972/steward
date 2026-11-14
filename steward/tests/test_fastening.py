"""Headless MuJoCo integration tests for screw fastening."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from steward.assets.generators import (airplane_fuselage, airplane_wing, building, dining_table,
                                       electric_screwdriver, toy_screw)
from steward.math3d import pose_compose, pose_inverse
from steward.scene.fasteners import STATE_KEY, FastenConfig, records
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, Link, RigidTree, SceneState, exact, pose, primitive
from steward.world.sim import EditError, WorldSim

_spec = importlib.util.spec_from_file_location('fasten_demo', Path(__file__).parents[1] / 'scripts' / 'fasten_demo.py')
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def scene_dict():
    scene = SceneState(mode='world')
    for factory, eid, at in [
        (building, 'w:room', [0, 0, 0]), (dining_table, 'w:table', [0, 0, 0]),
        (airplane_fuselage, demo.BASE, [0, 0, .751]),
        (airplane_wing, demo.WING, [.2, .1, .751]),
        (electric_screwdriver, demo.TOOL, [-.4, 0, .751]),
        (toy_screw, demo.SCREWS[0], [-.2, .2, .751]),
        (toy_screw, demo.SCREWS[1], [-.1, .2, .751]),
    ]:
        ent = factory(); ent.id = eid; ent.pose = exact(pose(at))
        scene.entities.append(ent)
    scene.entities.append(Entity(demo.FIXTURE, 'robot', exact('fixture'), exact(pose()),
        RigidTree(exact('fixed'), [Link('grip', 'grip', affordances=[exact('grasp-point')])])))
    return scene.to_dict()


@pytest.fixture
def sim():
    world = WorldSim(scene_dict(), fasten=FastenConfig())
    demo.prepare(world)
    try:
        yield world
    finally:
        world.stop()


def secured(sim, both=True):
    for screw in demo.SCREWS if both else demo.SCREWS[:1]:
        assert demo.tighten(sim, screw)
    demo.release(sim)
    demo.set_pose(sim, demo.TOOL, pose([1, 0, 1]))


def test_tighten_seats_two_screws_and_publishes_only_world_truth(sim):
    before = sim._entity_pose(demo.SCREWS[0])['p'][2]
    secured(sim)
    assert len(records(sim.template)) == 2
    assert len(sim._fastening.welds) == 3  # one part weld, two screw welds
    assert sum(sim.data.eq_active) == 3
    assert sim._entity_pose(demo.SCREWS[0])['p'][2] < before - .01
    assert sum(e['type'] == 'fasten' for e in sim.events) == 2
    assert len([r for r in sim._relations() if r.type == 'attached']) == 3
    for screw in demo.SCREWS:
        f = demo.feature_world(sim, 'screw', screw)
        idx = demo.SCREWS.index(screw)
        h = demo.feature_world(sim, 'screw-hole', demo.WING, f'mount_{idx}')
        assert np.linalg.norm(f['seat'] - h['entry']) < .001
    # Existing perception payload stays just the old id/p markers.
    assert all(set(h) == {'id', 'p'} for h in sim._screw_holes(sim.template.entity(demo.WING), None))
    world = SceneState.from_dict(sim.world_state())
    world.mode = 'ego'
    assert not build_mjcf(world, fasteners=True).fastener_welds


def test_dwell_resets_on_contact_break_and_uses_simulation_time(sim):
    screw = demo.SCREWS[0]
    assert not demo.tighten(sim, screw, seconds=.1)
    demo.hold(sim, demo.TOOL, pose([1, 0, 1]))
    sim.step()
    assert not demo.tighten(sim, screw, seconds=.1)
    assert demo.tighten(sim, screw, seconds=.15)


@pytest.mark.parametrize('fault', ['unheld', 'wrong-axis', 'tip-far', 'hole-offset', 'no-screw'])
def test_no_false_activation(sim, fault):
    screw = demo.SCREWS[0]
    if fault == 'hole-offset':
        at = sim._entity_pose(demo.WING)
        at['p'][0] += .04
        demo.set_pose(sim, demo.WING, at)
    if fault == 'no-screw':
        sim.edit_remove(screw)
        target = pose([0, .012, .9])
    else:
        target = demo.tool_pose(sim, screw)
    if fault == 'wrong-axis':
        target['q'] = [1, 0, 0, 0]
    if fault == 'tip-far':
        target['p'][0] += .03
    for _ in range(400):
        if fault == 'unheld':
            demo.set_pose(sim, demo.TOOL, target)
        else:
            demo.hold(sim, demo.TOOL, target)
        sim.step()
    assert not records(sim.template)
    assert not any(e['type'] == 'fasten' for e in sim.events)


def test_rebuild_snapshot_restore_and_reset(sim):
    secured(sim)
    before = copy.deepcopy(records(sim.template))
    time = sim.time
    sim.edit_spawn('w:extra', toy_screw().to_dict(), {'at': {'x': 1.2, 'y': 0, 'z': .1}})
    assert sim.time == time
    assert records(sim.template) == before
    assert sum(sim.data.eq_active) == 3
    sim.edit_remove('w:extra')
    restored = WorldSim(sim.world_state(), fasten=FastenConfig())
    try:
        assert records(restored.template) == before
        assert sum(restored.data.eq_active) == 3
        drift, _, _ = demo.flip_and_measure(restored)
        assert drift < .005
    finally:
        restored.stop()
    sim.reset()
    assert not records(sim.template)
    assert not any(sim.data.eq_active)


def test_remove_one_screw_preserves_other_and_last_removal_detaches(sim):
    secured(sim)
    sim.edit_remove(demo.SCREWS[0])
    assert demo.WING in sim._fastening.component(demo.BASE)
    assert sum(sim.data.eq_active) == 2
    sim.edit_remove(demo.SCREWS[1])
    assert sim._fastening.component(demo.BASE) == {demo.BASE}
    assert not any(sim.data.eq_active)


def test_removing_part_cleans_records_and_reenables_screw_collision(sim):
    secured(sim)
    sim.edit_remove(demo.WING)
    assert not records(sim.template)
    assert not any(sim.data.eq_active)
    for screw in demo.SCREWS:
        assert STATE_KEY not in sim.template.entity(screw).state_vars
        gids = [i for i in range(sim.model.ngeom)
                if sim._body_entity.get(int(sim.model.geom_bodyid[i])) == screw]
        assert any(sim.model.geom_contype[g] for g in gids)


def test_component_grasp_release_edit_and_rebuild_while_held(sim):
    secured(sim)
    initial = demo.relative(sim)
    # Invoke normal capture logic; place fixture below the base, far from its wing.
    target = pose_compose(sim._entity_pose(demo.BASE), pose([-.06, 0, .02]))
    demo.set_pose(sim, demo.FIXTURE, target)
    rs, g = demo.grip(sim)
    g.closed = True
    sim._try_attach(rs, g, 'grip')
    assert g.attached == demo.BASE
    with pytest.raises(EditError):
        sim.edit_move(demo.WING, {'at': {'x': 1, 'y': 1, 'z': 1}})
    # Changing the grip fixture exercises the production ideal-grasp follow path.
    demo.set_pose(sim, demo.FIXTURE, pose([0, 0, 1.3], [0, 1, 0, 0]))
    sim.step(2)
    np.testing.assert_allclose(demo.relative(sim)['p'], initial['p'], atol=.001)
    sim._rebuild(SceneState.from_dict(sim.world_state()))
    sim.step(2)
    assert g.attached == demo.BASE
    assert sum(sim.data.eq_active) == 3
    demo.release(sim)
    for eid in (demo.BASE, demo.WING):
        gids = [i for i in range(sim.model.ngeom) if sim._body_entity.get(int(sim.model.geom_bodyid[i])) == eid]
        assert any(sim.model.geom_contype[g] for g in gids)
    assert sim.edit_move(demo.BASE, {'at': {'x': 0, 'y': 0, 'z': 1.5}})['carried']
    np.testing.assert_allclose(demo.relative(sim)['p'], initial['p'], atol=.001)


def test_occupied_holes_cannot_take_another_screw(sim):
    secured(sim, both=False)
    original = len(sim.events)
    hole = demo.feature_world(sim, 'screw-hole', demo.WING, 'mount_0')
    demo.set_pose(sim, demo.SCREWS[1], pose(hole['entry']))
    assert not demo.tighten(sim, demo.SCREWS[1])
    assert not any(e['type'] == 'fasten' for e in sim.events[original:])


def test_one_base_can_join_two_distinct_parts_without_category_matching():
    scene = SceneState.from_dict(scene_dict())
    scene.entities = [e for e in scene.entities if e.id != demo.WING]
    scene.entity(demo.BASE).category = exact('arbitrary-block')
    for i, y in enumerate((.012, -.012)):
        plate = Entity(f'w:tab-{i}', 'object', exact('arbitrary-tab'), exact(pose([.02, y, .792])),
            RigidTree(exact('free'), [Link('body', 'tab',
                geometry={'collision': [primitive('box', [.02, .02, .01], pose([0, 0, .005]))]},
                inertial=exact({'mass': .02}), affordances=[exact({'screw-hole': {
                    'id': 'mount', 'p': [0, 0, .01], 'axis': [0, 0, 1], 'entry': [0, 0, .01], 'depth': .01}})])]))
        scene.entities.append(plate)
    sim = WorldSim(scene.to_dict(), fasten=FastenConfig())
    try:
        sim.step_seconds(.5)
        for i, screw in enumerate(demo.SCREWS):
            hole = demo.feature_world(sim, 'screw-hole', f'w:tab-{i}', 'mount')
            demo.set_pose(sim, screw, pose(hole['entry']))
            assert demo.tighten(sim, screw)
        assert {'w:tab-0', 'w:tab-1', *demo.SCREWS} <= sim._fastening.component(demo.BASE)
        assert sum(sim.data.eq_active) == 4
        demo.release(sim)
        sim.edit_remove(demo.SCREWS[0])
        assert 'w:tab-0' not in sim._fastening.component(demo.BASE)
        assert 'w:tab-1' in sim._fastening.component(demo.BASE)
    finally:
        sim.stop()


@pytest.mark.parametrize('fastened', [True, False])
def test_three_second_flip_with_unfastened_control(fastened):
    demo.run_case(scene_dict(), fastened)


def test_nonfastening_scene_has_no_constraints_or_physics_hooks():
    ent = dining_table()
    sim = WorldSim(SceneState(mode='world', entities=[ent]).to_dict())
    try:
        assert not sim._fastening.slots
        assert sim.model.neq == 0
        assert sim._fastening.component(ent.id) == {ent.id}
    finally:
        sim.stop()


@pytest.mark.parametrize('mode', ['driver', 'seat', 'off'])
def test_three_modes_and_off_snapshot_restore(mode):
    world = WorldSim(scene_dict(), fasten=FastenConfig(mode=mode))
    try:
        demo.prepare(world)
        if mode == 'driver':
            world.step_seconds(.4)
            assert not records(world.template)
            secured(world)
        elif mode == 'seat':
            world.step_seconds(2.)
            assert len(records(world.template)) == 2
            assert all(e['tool'] is None for e in world.events if e['type'] == 'fasten')
        else:
            assert not demo.tighten(world, demo.SCREWS[0])
            assert world.model.neq == 0
            assert not records(world.template)
            return
        snapshot = world.world_state()
        disabled = WorldSim(snapshot, fasten=FastenConfig(mode='off'))
        try:
            assert disabled.model.neq == 0
            assert not disabled._fastening.welds
            assert disabled._fastening.component(demo.BASE) == {demo.BASE}
            for screw in demo.SCREWS:
                gids = [g for g in range(disabled.model.ngeom)
                        if disabled._body_entity.get(int(disabled.model.geom_bodyid[g])) == screw]
                assert any(disabled.model.geom_contype[g] for g in gids)
            disabled._rebuild(SceneState.from_dict(snapshot))
            assert disabled.model.neq == 0
            disabled.reset()
            assert disabled.model.neq == 0
        finally:
            disabled.stop()
    finally:
        world.stop()
