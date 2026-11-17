"""Measured TCP feedback and the release gate for both Piper embodiments."""
import math

import mujoco
import numpy as np
import pytest

from test_agilex import imported_profile  # noqa: F401 - shared pytest fixture
from steward.assets.agilex import agilex_usd
from steward.assets.cobot import cobot_magic
from steward.ego.piper_arm import PiperArm
from steward.ego.skills import Job, MotionConfig, SkillExecutor, SkillFailed
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, SceneState
from steward.math3d import matrix_from_quat, quat_from_axis_angle, quat_mul
from steward.ego.piper_arm import TOOL_DOWN
from steward.ego.esim import Esim
from steward.scene.schema import exact


@pytest.fixture(params=['cobot-magic', 'agilex-usd'])
def arm(request, imported_profile):
    model = cobot_magic().to_dict() if request.param == 'cobot-magic' else agilex_usd(imported_profile)
    return PiperArm(model)


@pytest.mark.parametrize('lag', ['lift', 'arm', 'limit_overshoot'])
def test_tcp_feedback_retains_servo_error_near_the_command(arm, lag):
    arm.virtualize({'joints': arm.initial})
    raw = arm.command({'lift': .85, 'reach': .35, 'wrist_yaw': 0, 'wrist_roll': 0})['joints']
    if lag == 'lift':
        raw['lift'] += .05
    elif lag == 'limit_overshoot':
        lift = next(j for j in arm.ik.robot['body']['joints'] if j['id'] == 'lift')
        raw['lift'] = lift['limits']['value']['lo'] - .0002
    else:
        raw[arm.names[0]] += .03
    built = build_mjcf(SceneState(mode='world', entities=[Entity.from_dict(arm.ik.robot)]))
    model = built.compile()
    data = mujoco.MjData(model)
    built.apply_initial(model, data)
    names = built.names[arm.ik.robot['id']]
    for name, value in raw.items():
        data.joint(names.joints[name]).qpos[0] = value
    mujoco.mj_forward(model, data)
    tip = data.body(names.bodies[arm.geometry.gripper])
    expected = tip.xpos + tip.xmat.reshape(3, 3) @ arm.ik.tcp_offsets[arm.config.arm]
    state = arm.virtualize({'joints': raw})
    reported = [state['toolOffset'] + state['joints']['reach'], state['armLateral'], state['joints']['lift']]
    assert reported == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('pitch,roll', [(0., 0.), (.2, 1.4), (.6, -1.5), (1.05, math.pi), (1.05, -math.pi)])
def test_measured_wrist_angles_preserve_a_tilted_half_turn(arm, pitch, roll):
    yaw = .4
    arm._target = dict(lift=.85, reach=.35, wrist_yaw=yaw + .01, wrist_pitch=pitch, wrist_roll=roll)
    pitched = quat_mul(quat_from_axis_angle([0, 1, 0], -pitch), TOOL_DOWN)
    axis, angle = ([1, 0, 0], -roll) if arm._roll_about_tool_x(pitch, roll) else ([0, 0, 1], roll)
    rotation = quat_mul(quat_from_axis_angle([0, 0, 1], yaw),
                        quat_mul(pitched, quat_from_axis_angle(axis, angle)))
    np.testing.assert_allclose(arm.tool_rotation(yaw, roll, pitch), matrix_from_quat(rotation), atol=1e-8)
    measured = arm._virtual_from_tcp({'p': [.7, .3, .85], 'q': rotation})
    assert [measured[k] for k in ('wrist_yaw', 'wrist_pitch', 'wrist_roll')] == pytest.approx([yaw, pitch, roll], abs=1e-8)


@pytest.mark.parametrize('roll,feasible', [(1.9, False), (2.3, True), (-2.3, True)])
def test_pour_checks_container_tilt_instead_of_wrist_roll(arm, monkeypatch, roll, feasible):
    from test_esim_wrist import _scene

    model, scene = _scene()
    scene.entity('e:3').rigid.links[0].affordances.append(exact('pourable'))
    planner = Esim(model, arm=arm)
    pitch = 1.05
    # Isolate orientation from docking and IK; an admissible path alone cannot empty a box.
    monkeypatch.setattr(planner, '_arm_joints', lambda es: {
        'lift': .85, 'reach': .3, 'wrist_yaw': 0., 'wrist_roll': 0., 'wrist_pitch': pitch})
    monkeypatch.setattr(planner, '_grip_approach', lambda *args, **kwargs:
                        ({'reach': .3, 'yaw': 0., 'wristPitch': pitch}, (0., 0.)))
    monkeypatch.setattr(arm, 'reachable_path', lambda *args, **kwargs: True)
    receipt = planner.predict(scene, 'pour', {'on': 'e:4', 'roll': roll})
    assert receipt['feasible'] is feasible, receipt['issues']
    tilt = math.acos(math.cos(pitch)**2 + math.sin(pitch)**2 * math.cos(roll))
    if feasible:
        assert receipt['plan']['containerTilt'] == pytest.approx(tilt)
        assert tilt >= planner.config.pour_roll_min
    else:
        assert tilt < planner.config.pour_roll_min < abs(roll)
        assert 'tilts the container' in '; '.join(receipt['issues'])


@pytest.mark.parametrize('feedback', ['lagging', 'missing', 'missing_target', 'settled', 'loaded'])
def test_place_keeps_its_load_unless_release_position_is_measured(monkeypatch, feedback):
    state = {'stamp': 0., 'joints': {'lift': .80, 'reach': .3}}
    commands = []
    executor = SkillExecutor(commands.append, lambda: state, lambda eid: eid,
                             MotionConfig(release_settle_s=.12))
    executor._setpoints = {'lift': .85, 'reach': .3}
    if feedback == 'missing_target':
        executor._setpoints.pop('lift')

    def tick():
        state['stamp'] += .04
        if feedback == 'missing':
            state['joints'].pop('lift', None)
        elif feedback == 'settled':
            state['joints']['lift'] = .85
        elif feedback == 'loaded':
            state['joints']['lift'] = .8431
        return state

    monkeypatch.setattr(executor, '_tick', tick)
    for name in ('_dock', '_restore_planned_reach', '_align_lateral', '_reach_over',
                 '_wait', '_exit_long_reach', '_report_placement'):
        monkeypatch.setattr(executor, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, '_over_support', lambda plan: True)
    plan = {'spot': [.6, 0, .85], 'lift': .85, 'object': 'piece', 'support': 'table', 'stayNear': True}
    if feedback in ('settled', 'loaded'):
        executor._place(Job('j', 'place', 'r'), plan)
        assert {'gripper': 'open'} in commands
    else:
        with pytest.raises(SkillFailed, match='release'):
            executor._place(Job('j', 'place', 'r'), plan)
        assert {'gripper': 'open'} not in commands
