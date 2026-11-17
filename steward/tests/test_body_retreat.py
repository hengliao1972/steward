"""Physical release and retreat through each body's formal arm and motion policy."""
import copy
import os
from contextlib import contextmanager

import numpy as np
import pytest

from steward.assets.generators import building, cabinet_with_drawer, dining_table, printed_board
from steward.ego.bodies import prepare_body
from steward.ego.esim import Esim
from steward.ego.kinematics import _link_pose
from steward.ego.skills import Job, SkillExecutor, SkillFailed
from steward.math3d import pose_compose
from steward.world.sim import PerceptionConfig, WorldSim
from test_flip_physics import SteppedClock

BODIES = [
    'steward-bot', 'cobot-magic',
    pytest.param('agilex-usd', marks=pytest.mark.skipif(
        os.environ.get('STEWARD_TEST_AGILEX_USD') != '1',
        reason='requires the actual imported robot.usd meshes')),
]


@contextmanager
def rig(body, extras, base_xy=(0., 0.), target=None):
    setup = prepare_body(body)
    robot = copy.deepcopy(setup.model)
    robot['id'] = 'w:robot'
    arm = setup.arm or Esim(robot, setup.config.esim).arm
    raw = {j['id']: j['q']['value'][0] for j in robot['body']['joints'] if j['q']['value']}
    arm.virtualize({'joints': raw, 'odometry': {'x': 0., 'y': 0., 'yaw': 0.}})
    target = target or dict(lift=.85, reach=.4, wrist_pitch=0., wrist_roll=0., wrist_yaw=0.)
    command = arm.command(target)
    physical = command.get('joints', command)
    for j in robot['body']['joints']:
        if j['id'] in physical:
            j['q']['value'] = [physical[j['id']]]
    x, y = base_xy(arm.geometry) if callable(base_xy) else base_xy
    robot['pose']['value'] = {'p': [x, y, 0.], 'q': [1., 0., 0., 0.]}
    room = building().to_dict()
    room['id'] = 'w:room'
    objects = extras(arm.geometry) if callable(extras) else extras
    sim = WorldSim({'mode': 'world', 'entities': [room, robot, *objects]},
                   perception=PerceptionConfig(odom_pos_sigma=0., odom_yaw_sigma=0.))
    try:
        def state():
            return arm.virtualize({**sim.proprioception(robot['id']), 'odometry': sim.odometry(robot['id'])})
        state()
        executor = SkillExecutor(lambda c: sim.command(robot['id'], arm.command(c)), state, lambda e: e,
                                 config=setup.config.motion, clock=SteppedClock(sim), motion_policy=arm.motion_policy)
        executor._setpoints = dict(target)
        executor._dock = lambda *a, **k: None  # Each fixture starts already docked, with native IK.
        yield sim, executor, state, arm
    finally:
        sim.stop()


@pytest.mark.parametrize('body', BODIES)
def test_ordinary_surface_place_and_retreat(body):
    def objects(g):
        table = dining_table(width=1.2, depth=.9).to_dict()
        table['id'] = 'w:table'
        table['pose']['value']['p'] = [.95, g.lateral, 0.]
        piece = printed_board().to_dict()
        piece['id'] = 'w:piece'
        piece['pose']['value']['p'] = [g.tool_offset+.4, g.lateral, .847]
        return [table, piece]
    with rig(body, objects) as (sim, ex, state, arm):
        ex._command({'gripper': 'close'})
        sim.step_seconds(.15)
        assert state()['gripper']['holding'], 'fixture must begin holding the piece'
        spot = [arm.geometry.tool_offset+.4, arm.geometry.lateral]
        plan = dict(spot=spot, gripSpot=spot, lift=.77, reach=.4, object='piece', support='table',
                    predock=[-.2, 0.], yaw=0., supportBox=dict(cx=.95,cy=spot[1],hx=.6,hy=.45,yaw=0.))
        job = Job('place', 'place', 'r')
        ex._place(job, plan)
        sim.step_seconds(1.)
        p = sim._entity_pose('w:piece')['p']
        supports = {r['a']['entity']: r['b']['entity'] for r in sim.world_state()['relations']
                    if r['type'] == 'supported-by'}
        assert not state()['gripper']['holding']
        assert supports.get('w:piece') == 'w:table', (p, supports)
        assert np.linalg.norm(np.array(p[:2])-spot[:2]) < .04, (p, spot)
        assert not any('exit after place' in line for line in job.log), job.log


def _check_drawer(body, direction, handle_error=(0., 0., 0.)):
    cabinet = cabinet_with_drawer().to_dict()
    cabinet['id'] = 'w:cabinet'
    cabinet['pose']['value'] = {'p':[2.7,.6,0.], 'q':[0.,0.,0.,1.]}
    opening = 0. if direction > 0 else .3
    next(j for j in cabinet['body']['joints'] if j['id']=='drawer_slide')['q']['value']=[opening]
    drawer = next(link for link in cabinet['body']['links'] if link['id'] == 'drawer')
    handle_geom = max(drawer['geometry']['collision'], key=lambda geom: geom['pose']['p'][0])
    drawer_pose = pose_compose(cabinet['pose']['value'], _link_pose(cabinet, {'drawer_slide': opening}, 'drawer'))
    handle = (np.asarray(pose_compose(drawer_pose, handle_geom['pose'])['p']) + handle_error).tolist()

    def dock(g):
        return handle[0] - g.tool_offset - .25, handle[1] - g.lateral

    with rig(body,[cabinet],dock) as (sim, ex, state, arm):
        x,y = dock(arm.geometry)
        plan = dict(handle=handle,drawer='drawer',axis=[-1.,0.],distance=.3,direction=direction,yaw=0.,
                    dock=[x,y],end=[x-direction*.3,y],predock=[x-.35,y],viewing=[x-direction*.3-.55,y],
                    captureRadius=.12,handleProbesAlong=[.04,.08,.12,-.05,-.09,.16],
                    handleProbesSide=[.06,-.06,.10,-.10,.14,-.14])
        job=Job('drawer','open' if direction>0 else 'close','r')
        moved = ex._drawer(job, plan)
        sim.step_seconds(.5)
        actual = float(sim.data.joint(sim.build.names['w:cabinet'].joints['drawer_slide']).qpos[0])
        assert moved == pytest.approx(direction*.3,abs=.035)
        assert actual > .2 if direction > 0 else actual < .05, (actual,job.log)
        assert not state()['gripper']['holdingHandle']
        assert any(e['type']=='grasp/handle-release' for e in sim.events)
        joints = state()['joints']
        assert joints['reach'] == pytest.approx(ex.config.retract_reach, abs=.02)
        assert joints['lift'] == pytest.approx(ex.config.stow_lift, abs=.02)
        odom = state()['odometry']
        assert [odom['x'], odom['y']] == pytest.approx(plan['viewing'], abs=.03)


@pytest.mark.parametrize('body', BODIES)
@pytest.mark.parametrize('direction', [1., -1.], ids=['open', 'close'])
def test_drawer_actual_travel_and_release(body, direction):
    _check_drawer(body, direction)


@pytest.mark.parametrize('direction', [1., -1.], ids=['open', 'close'])
def test_cobot_withdraws_after_an_offset_handle_approach(direction):
    """The offset grasp takes the simplified arm into the branch that cannot retract at stow height."""
    _check_drawer('cobot-magic', direction, handle_error=(.015, 0., -.016))


@pytest.mark.parametrize('body', BODIES)
@pytest.mark.parametrize('feedback', ['settled','lagging','missing','missing_target'])
def test_release_guard_with_body_configuration(body, feedback, monkeypatch):
    cfg = prepare_body(body).config.motion
    state = {'stamp': 0., 'joints': {'lift': .85, 'reach': .3}}
    ex = SkillExecutor(lambda c: None, lambda: state, lambda e: e, config=cfg)
    ex._setpoints = dict(state['joints'])
    if feedback == 'missing_target':
        ex._setpoints.pop('reach')
    if feedback == 'missing':
        state['joints'].pop('reach')
    if feedback == 'lagging':
        state['joints']['lift'] -= .02

    def tick():
        state['stamp'] += cfg.tick_s
        return state

    monkeypatch.setattr(ex, '_tick', tick)
    if feedback == 'settled':
        ex._settle_for_release()
    else:
        with pytest.raises(SkillFailed, match='release'):
            ex._settle_for_release()
