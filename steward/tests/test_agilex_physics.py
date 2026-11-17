"""Actual imported meshes: native home, raw lift, cameras, fingers and self-masking."""
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from steward.assets.agilex import agilex_usd
from steward.assets.generators import building
from steward.ego.bodies import prepare_body
from steward.ego.estimator import EgoEstimator
from steward.ego.kinematics import _link_pose
from steward.ego.selfmask import self_pixels
from steward.ego.skills import Job, SkillExecutor
from steward.world.sim import PerceptionConfig, WorldSim
from test_cobot_skills import _scene
from test_cobot_perception import _sync
from test_selfmask import _robot_pixels
from test_flip_physics import SteppedClock

pytestmark = pytest.mark.skipif(os.environ.get('STEWARD_TEST_AGILEX_USD') != '1',
                               reason='requires the actual imported robot.usd meshes')


@pytest.fixture
def rig():
    scene = _scene()
    original = next(e for e in scene['entities'] if e['id'] == 'w:cobot')
    robot = agilex_usd()
    assert any(g['kind'] == 'mesh' for link in robot['body']['links']
               for g in link.get('geometry', {}).get('visual', [])), 'actual mesh import required'
    robot.update(id='w:cobot', pose=copy.deepcopy(original['pose']))
    scene['entities'] = [robot if e['id'] == robot['id'] else e for e in scene['entities']]
    sim = WorldSim(scene)
    try:
        yield sim, robot
    finally:
        sim.stop()


def test_native_home_keeps_usd_precision_and_physical_shoulder_height(rig):
    sim, robot = rig
    raw = sim.sensors(robot['id'])['joints']
    assert raw['lift'] == pytest.approx(-.45, abs=1e-6)
    lift = next(j for j in robot['body']['joints'] if j['id'] == 'lift')
    assert lift['limits']['value'] == pytest.approx({'lo': -.6, 'hi': 0}, abs=1e-6)
    assert _link_pose(robot, raw, 'L_arm_base')['p'][2] == pytest.approx(1.1268813256, abs=1e-6)
    assert _link_pose(robot, dict(raw, lift=-.6), 'L_arm_base')['p'][2] == pytest.approx(.9768813256, abs=1e-6)


@pytest.mark.parametrize('height', [-.55, -.3, -.05])
def test_raw_lift_servo_follows_the_negative_usd_range(rig, height):
    sim, robot = rig
    sim.command(robot['id'], {'joints': {'lift': height}})
    sim.step(1500)
    assert sim.sensors(robot['id'])['joints']['lift'] == pytest.approx(height, abs=.02)
    assert not any(w.number for w in sim.data.warning)


def test_head_cameras_follow_lift_and_match_compiled_extrinsics(rig):
    sim, robot = rig
    estimator = EgoEstimator(robot_model=robot)
    _sync(estimator, sim)
    before = {name: np.array(pose['p']) for name, pose in estimator.camera_poses().items()}
    assert set(before) == {'head_camera', 'wide_camera'}
    sim.command(robot['id'], {'joints': {'lift': -.25}})
    sim.step(1200)
    _sync(estimator, sim)
    for name, pose in estimator.camera_poses().items():
        actual, _ = sim.camera_pose(robot['id'], name)
        assert pose['p'] == pytest.approx(actual['p'], abs=.001)
        assert abs(np.dot(pose['q'], actual['q'])) > .99999
        assert pose['p'][2] - before[name][2] > .15


@pytest.mark.parametrize('side', ['L', 'R'])
def test_both_fingers_mimic_opening_and_closing_from_zero_home(rig, side):
    sim, robot = rig
    names = sim.build.names[robot['id']]
    raw = sim.sensors(robot['id'])['joints']
    assert [raw[f'{side}_joint{n}'] for n in range(1, 7)] == [0.] * 6
    for travel in [.04, .02, 0.]:
        sim.command(robot['id'], {'joints': {f'{side}_joint7': travel}})
        sim.step(1000)
        fingers = [float(sim.data.joint(names.joints[f'{side}_joint{n}']).qpos[0]) for n in (7, 8)]
        assert fingers[0] == pytest.approx(travel, abs=.005)
        assert abs(sum(fingers)) < .001


@pytest.mark.parametrize('posture', ['home', 'raised', 'arm'])
def test_mesh_mask_covers_visible_body_without_removing_tableware(rig, posture):
    sim, robot = rig
    if posture != 'home':
        joints = {'lift': -.25} if posture == 'raised' else {'L_joint2': .3, 'L_joint3': -2.4}
        sim.command(robot['id'], {'joints': joints})
        sim.step(1200)
    estimator = EgoEstimator(robot_model=robot)
    _sync(estimator, sim)
    pose, sensor = sim.camera_pose(robot['id'], 'wide_camera')
    predicted = self_pixels(robot, estimator.link_pose, pose, sensor['intrinsics'])
    visible = _robot_pixels(sim, robot['id'], 'wide_camera')
    assert visible.sum() > 0, 'this test posture must expose the robot to its camera'
    assert (predicted & visible).sum() / visible.sum() > .95
    for entity in ('w:bowl-1', 'w:plate-1'):
        pixels = _robot_pixels(sim, robot['id'], 'wide_camera', entity_id=entity)
        assert pixels.sum() > 0, f'{entity} must be visible for the exclusion check'
        assert (predicted & pixels).sum() / pixels.sum() < .02, f'mask removed visible {entity}'


@contextmanager
def _recorded_motion(name):
    fixture = json.loads((Path(__file__).parent / 'fixtures' / name).read_text())
    setup = prepare_body('agilex-usd')
    robot, arm = copy.deepcopy(setup.model), setup.arm
    robot.update(id=fixture['robot']['id'], pose=fixture['robot']['pose'])
    for joint in robot['body']['joints']:
        if joint['id'] in fixture['robot']['joints']:
            joint['q']['value'] = fixture['robot']['joints'][joint['id']]
    room = building().to_dict()
    room['id'] = 'w:room'
    sim = WorldSim({'mode': 'world', 'entities': [room, robot, *fixture['entities']]},
                   perception=PerceptionConfig(odom_pos_sigma=0, odom_yaw_sigma=0))
    try:
        def state():
            return arm.virtualize({**sim.proprioception(robot['id']), 'odometry': sim.odometry(robot['id'])})

        measured = state()
        controller = fixture.get('controller')
        arm._target = dict(measured['joints']) if controller is None else dict(controller['target'])
        if controller is not None:
            arm._solved = controller['solution']
        executor = SkillExecutor(lambda command: sim.command(robot['id'], arm.command(command)),
                                 state, lambda eid: eid, config=setup.config.motion,
                                 clock=SteppedClock(sim), motion_policy=arm.motion_policy)
        executor._setpoints = dict(measured['joints'] if controller is None else controller['setpoints'])
        yield sim, executor, fixture, state
    finally:
        sim.stop()


@pytest.mark.parametrize('dx,dz', [(0., 0.), (.005, -.005), (-.005, .005)])
def test_drawer_pick_enters_below_the_top_and_withdraws_before_lifting(monkeypatch, dx, dz):
    """A docked real-mesh posture retains the cabinet top above the deep drawer's bottle."""
    with _recorded_motion('agilex-drawer-pick.json') as (sim, executor, fixture, state):
        plan = fixture['plan']
        fixture['target']['p'][0] += dx
        fixture['target']['p'][2] += dz
        executor._state = lambda: {**state(), 'entities': {plan['target']: fixture['target']}}
        # The recording starts at the dock; keep live alignment during the horizontal entry.
        monkeypatch.setattr(executor, '_dock', lambda *args: None)
        job = Job('drawer-pick', 'pick', 'r')
        executor._pick_target(job, plan)
        sim.step_seconds(.5)
        grip = sim.robots[fixture['robot']['id']].grippers['L_link6']
        assert grip.attached == 'w:pill-bottle', 'the bottle must leave the drawer in the working gripper'
        assert sim._entity_pose('w:pill-bottle')['p'][2] > .8, 'the grasp must clear the cabinet top'
        supports = {r['a']['entity']: r['b']['entity'] for r in sim.world_state()['relations']
                    if r['type'] == 'supported-by'}
        assert supports['w:perfume-bottle'] == 'w:cabinet-1', 'the neighbouring bottle must stay in the drawer'
        assert state()['odometry']['x'] == pytest.approx(plan['predock'][0], abs=.04)


def test_retreat_preserves_the_released_piece_on_the_table(monkeypatch):
    """A recorded post-release posture isolates real-mesh contact during arm/base withdrawal."""
    with _recorded_motion('agilex-release.json') as (sim, executor, fixture, state):
        measured = state()
        # The fixture starts after release; exercise the actual retreat commands and physics.
        for name in ('_dock', '_restore_planned_reach', '_align_lateral', '_reach_over', '_report_placement'):
            monkeypatch.setattr(executor, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(executor, '_over_support', lambda plan: True)
        x, y, yaw = (measured['odometry'][key] for key in ('x', 'y', 'yaw'))
        before = sim._entity_pose('w:piece-21')['p']
        executor._place(Job('retreat', 'place', 'r'),
                        {'spot': before, 'lift': measured['joints']['lift'], 'object': 'piece', 'support': 'table',
                         'predock': [x - .2 * math.cos(yaw), y - .2 * math.sin(yaw)], 'yaw': yaw})
        sim.step_seconds(1)
        after = sim._entity_pose('w:piece-21')['p']
        assert np.linalg.norm(np.array(after[:2]) - before[:2]) < .02
        supports = {r['a']['entity']: r['b']['entity'] for r in sim.world_state()['relations']
                    if r['type'] == 'supported-by'}
        assert supports['w:piece-21'] == 'w:table-1'



def test_handle_lowering_clears_the_cabinet_and_opens_the_drawer(monkeypatch):
    """At a short handle dock, descend outside the cabinet instead of onto its countertop."""
    with _recorded_motion('agilex-handle.json') as (sim, executor, fixture, state):
        job = Job('handle', 'open', 'r')
        monkeypatch.setattr(executor, '_dock', lambda *args: None)  # Recorded after docking.
        moved = executor._drawer(job, fixture['plan'])
        assert moved == pytest.approx(fixture['plan']['distance'], abs=.03)
        drawer = sim.template.entity('w:cabinet-1')
        _, joint = next(iter(sim._handle_links(drawer)))
        actual = sim.data.joint(sim.build.names[drawer.id].joints[joint.id]).qpos[0]
        assert actual > .2, job.log
        assert not any('handle lower failed' in message for message in job.log), job.log
