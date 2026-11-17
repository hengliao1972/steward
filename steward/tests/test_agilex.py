"""The imported dual Piper is a selectable, camera-calibrated task embodiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import pytest

from steward.assets.agilex import agilex_usd
from steward.ego.bodies import body_arm, prepare_body, robot_model
from steward.ego.estimator import _apply_joint
from steward.ego.kinematics import _link_pose
from steward.embodiments import embodiment_of, with_body
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, SceneState

FIXTURES = Path(__file__).parent / 'fixtures'
PROFILE = Path(__file__).parents[1] / 'config/agilex-usd.toml'


@pytest.fixture
def imported_profile(tmp_path):
    template = tmp_path / 'template.json'
    template.write_bytes((FIXTURES / 'agilex-kinematics.json').read_bytes())
    config = tmp_path / 'body.toml'
    config.write_text(PROFILE.read_text().replace('../assets/agilex/runtime/template.json', 'template.json'))
    return config


def test_missing_or_wrong_template_never_substitutes_another_robot(imported_profile):
    template = imported_profile.parent / 'template.json'
    template.unlink()
    with pytest.raises(ValueError, match='template missing'):
        agilex_usd(imported_profile)
    template.write_text(json.dumps({'body': {'links': [], 'joints': []}}))
    with pytest.raises(ValueError, match='expected dual-Piper'):
        agilex_usd(imported_profile)


def test_registered_body_keeps_usd_kinematics_and_supports_scenario_replacement(imported_profile, monkeypatch):
    monkeypatch.setenv('STEWARD_AGILEX_USD_CONFIG', str(imported_profile))
    model = robot_model('agilex-usd')
    assert embodiment_of(model) == 'agilex-usd'
    assert _body_summary(model) == json.loads((FIXTURES.parent / 'expected/agilex-body.json').read_text())
    original = robot_model('steward-bot')
    scenario = {'robots': [original['id']], 'sceneState': {'entities': [original]}}
    swapped = with_body(scenario, 'agilex-usd')['sceneState']['entities'][0]
    assert swapped['pose'] == original['pose'] and swapped['id'] == original['id']
    assert scenario['sceneState']['entities'][0] == original
    joints = {j['id']: j for j in model['body']['joints']}
    assert joints['lift']['limits']['value']['lo'] == pytest.approx(-.6)
    assert joints['L_joint3']['origin']['value']['p'][:2] == pytest.approx([.28358, .028726])
    assert joints['L_joint8'].get('actuator') is None
    assert model['body']['couplings'][1] == {'joint1': 'L_joint8', 'joint2': 'L_joint7', 'offset': 0, 'multiplier': -1}
    arm = body_arm('agilex-usd', model)
    assert arm.geometry.tool_offset == pytest.approx(.36848)
    assert arm.geometry.lateral == pytest.approx(.306)
    assert arm.shoulder_z == pytest.approx(1.5768813256)
    arm.virtualize({'joints': arm.initial, 'odometry': {'x': 0, 'y': 0, 'yaw': 0}})
    solved = arm.command({'lift': .85, 'reach': .3, 'wrist_yaw': 0, 'wrist_roll': 0, 'wrist_pitch': 0})['joints']
    measured = arm._fk(solved['lift'], [solved[n] for n in arm.names])
    assert measured['lift'] == pytest.approx(.85, abs=.006)
    assert measured['reach'] == pytest.approx(.3, abs=.006)


def test_encoder_fk_includes_off_center_hinge_pivots(imported_profile):
    model = agilex_usd(imported_profile)
    joint = next(j for j in model['body']['joints'] if j['id'] == 'L_joint1')
    joint['anchor'] = {'value': [.06, .02, 0]}
    joint['q']['value'] = [.7]
    encoders = {j['id']: j['q']['value'][0] for j in model['body']['joints'] if j['q']['value']}
    built = build_mjcf(SceneState(mode='world', entities=[Entity.from_dict(model)]))
    m = built.compile()
    d = mujoco.MjData(m)
    built.apply_initial(m, d)
    expected = d.body(built.names[model['id']].bodies['L_link1']).xpos
    assert _link_pose(model, encoders, 'L_link1')['p'] == pytest.approx(expected, abs=1e-6)
    parent = _link_pose(model, encoders, 'L_arm_base')
    assert _apply_joint(parent, joint, .7)['p'] == pytest.approx(expected, abs=1e-6)
    lower = dict(encoders, lift=encoders['lift'] - .1)
    before = _link_pose(model, encoders, 'top_camera_link')['p']
    after = _link_pose(model, lower, 'top_camera_link')['p']
    assert np.subtract(after, before) == pytest.approx([0, 0, -.1])


@pytest.mark.skipif(os.environ.get('STEWARD_TEST_AGILEX_USD') != '1',
                    reason='long-running real-mesh task; set STEWARD_TEST_AGILEX_USD=1')
def test_usd_robot_navigates_picks_and_places_through_ego_receipts(tmp_path):
    from steward.authoring.service import AuthoringService, build_cobot_workstation
    from steward.ego.runtime import EgoRuntime, RobotLink
    from steward.ego.server import ego_api
    from steward.net import call
    from steward.world.server import StewardHost

    build_cobot_workstation(AuthoringService(tmp_path))
    host = StewardHost(tmp_path, scenario='cobot-table', body='agilex-usd', phase='phase0',
                       speed=math.inf, seed=3, run_dir=tmp_path / 'run').start()
    runtime = api = None
    errors = []
    try:
        setup = prepare_body('agilex-usd')
        link = RobotLink(host.robot_servers['w:cobot-1'].url)
        link.require_body('agilex-usd')
        runtime = EgoRuntime(link, setup.model, setup.config, arm=setup.arm)
        runtime.start()
        api = ego_api(runtime).start()

        def task():
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    objects = call('GET', api.url + '/ego/summary')['objects']
                    bowls = [o for o in objects if o['category'] == 'bowl']
                    tables = [o for o in objects if o['category'] == 'table']
                    if bowls and tables:
                        break
                    time.sleep(.1)
                assert bowls and tables, objects
                for skill, params in [('pick', {'target': bowls[0]['id']}),
                                      ('place', {'on': tables[0]['id'], 'offset': [-.3, -.15]})]:
                    receipt = call('POST', api.url + '/ego/predict', {'skill': skill, 'params': params}, timeout=120)
                    assert receipt['feasible'] and receipt['withinRiskLimit'], receipt
                    job = call('POST', api.url + '/skill/run', {'receipt': receipt['id']})
                    done = call('POST', api.url + '/skill/wait', {'job': job['id'], 'timeoutS': 150}, timeout=160)
                    assert done['state'] == 'succeeded', done
                position = host.sim._entity_pose('w:bowl-1')['p']
                table = host.sim._entity_pose('w:table-1')['p']
                assert position[2] > .74
                assert np.linalg.norm(np.subtract(position[:2], [table[0] - .3, table[1] - .15])) < .06
                assert not any(w.number for w in host.sim.data.warning)
            except BaseException as exc:  # Report worker assertions after the rendering owner rejoins it.
                errors.append(exc)

        worker = threading.Thread(target=task)
        worker.start()
        while worker.is_alive():
            host.pump(.02)
        worker.join()
        if errors:
            raise errors[0]
    finally:
        if api is not None:
            api.stop()
        if runtime is not None:
            runtime.stop()
        host.stop()


def test_invalid_calibration_rejects_unreachable_lift_home(imported_profile):
    imported_profile.write_text(imported_profile.read_text().replace('lift_home = -0.45', 'lift_home = 0.2'))
    with pytest.raises(ValueError, match='outside the USD limits'):
        agilex_usd(imported_profile)
    imported_profile.write_text(imported_profile.read_text().replace('arm_kp = 1500.0', 'arm_kp = -1.0'))
    with pytest.raises(ValueError, match='Invalid arm_kp'):
        agilex_usd(imported_profile)


def test_mesh_self_filter_uses_asset_coordinates(tmp_path):
    from test_usd import _mesh
    from steward.ego.selfmask import _surface_points
    from steward.ego.estimator import EgoEstimator
    from steward.assets.generators import steward_bot

    geom = _mesh(tmp_path / 'part.msh')
    geom['pose']['p'] = [.5, 0, 0]
    points = _surface_points(geom)
    assert points.min(axis=0) == pytest.approx([.5, 0, 0], abs=1e-6)
    assert points.max(axis=0) == pytest.approx([.7, .3, .4], abs=1e-6)
    model = steward_bot().to_dict()
    base = next(link for link in model['body']['links'] if link['id'] == 'base')
    base['geometry']['visual'] = [geom]
    boxes = EgoEstimator(model).body_boxes()
    assert any(np.allclose(centre, [.6, .15, .2]) and np.allclose(half, [.1, .15, .2])
               for centre, half, _ in boxes)


def _body_summary(model):
    from steward.embodiments import describe

    return {"embodiment": describe(embodiment_of(model)), "root": model['body']['root']['value'],
            "actuators": sorted(j['id'] for j in model['body']['joints'] if j.get('actuator')),
            "cameras": [{"link": link['id'], **sensor} for link in model['body']['links'] for sensor in link.get('sensors', [])],
            "grippers": [link['id'] for link in model['body']['links'] if {'value': 'grasp-point'} in link.get('affordances', [])],
            "couplings": model['body']['couplings']}
