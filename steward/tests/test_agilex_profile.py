"""USD calibration changes reach the model and shared runtime without changing Cobot Magic."""
from dataclasses import asdict, replace

import mujoco
import pytest

from test_agilex import imported_profile  # noqa: F401 - shared pytest fixture
from steward.assets.agilex import agilex_usd
from steward.assets.agilex_profile import PROFILE, load_agilex_profile
from steward.assets.cobot import cobot_magic
from steward.ego.bodies import body_arm, runtime_config
from steward.ego.runtime import RuntimeConfig
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import Entity, SceneState


def test_partial_profile_resolves_paths_and_retains_usd_defaults(tmp_path):
    profile = tmp_path / 'body.toml'
    profile.write_text('template = "local/template.json"\n[motion]\nlinear_speed = 0.2\n')
    cfg = load_agilex_profile(profile)
    assert cfg['template'] == str(tmp_path / 'local/template.json')
    assert cfg['motion']['linear_speed'] == .2
    assert cfg['motion']['angular_speed'] == .9
    profile.write_text('[motion]\nlinear_speed = 0.2\n')
    assert load_agilex_profile(profile)['template'] == load_agilex_profile(PROFILE)['template']
    cfg['base']['kp'][0] = 1
    assert load_agilex_profile(profile)['base']['kp'][0] == 10000
    profile.write_text('[planning]\nhandle_probes_side_m = [0.03, -0.03]\n')
    assert load_agilex_profile(profile)['planning']['handle_probes_side_m'] == [.03, -.03]


def test_legacy_flat_profile_keeps_its_model(imported_profile):
    complete = agilex_usd(imported_profile)
    imported_profile.write_text(imported_profile.read_text().split('[base]')[0].replace('finger_effort = 100.0\n', ''))
    assert agilex_usd(imported_profile) == complete
    imported_profile.write_text(imported_profile.read_text().replace('arm_effort = 100.0', 'arm_effort = 75.0'))
    legacy = load_agilex_profile(imported_profile)
    assert legacy['arm_effort'] == legacy['finger_effort'] == 75
    imported_profile.write_text(imported_profile.read_text() + '\nfinger_effort = 40.0\n')
    assert load_agilex_profile(imported_profile)['finger_effort'] == 40


def test_profile_drives_compiled_actuators_initial_lift_and_cameras(imported_profile):
    text = imported_profile.read_text()
    for old, new in [
        ('lift_home = -0.45', 'lift_home = -0.3'),
        ('finger_effort = 100.0', 'finger_effort = 40.0'),
        ('kp = [10000.0, 10000.0, 3500.0]', 'kp = [9000.0, 8000.0, 3000.0]'),
        ('size = [320, 240]', 'size = [640, 480]'),
        ('fovy_deg = 60.0', 'fovy_deg = 55.0'),
        ('position = [0.05, 0.0, 0.0]', 'position = [0.08, 0.0, 0.02]'),
        ('rate_hz = 10.0', 'rate_hz = 8.0'),
    ]:
        text = text.replace(old, new)
    imported_profile.write_text(text)
    model = agilex_usd(imported_profile)
    built = build_mjcf(SceneState(mode='world', entities=[Entity.from_dict(model)]))
    compiled = built.compile()
    data = mujoco.MjData(compiled)
    built.apply_initial(compiled, data)
    names = built.names[model['id']]
    assert data.joint(names.joints['lift']).qpos[0] == pytest.approx(-.3)
    assert compiled.actuator(names.actuators['root_x']).gainprm[0] == 9000
    assert compiled.actuator(names.actuators['L_joint7']).forcerange == pytest.approx([-40, 40])
    assert compiled.camera(model['id'] + '::head_camera').fovy == pytest.approx(55)
    sensor = next(l for l in model['body']['links'] if l['id'] == 'top_camera_link')['sensors'][0]
    assert sensor['intrinsics']['width'] == 640 and sensor['intrinsics']['height'] == 480
    assert sensor['mount']['p'] == [.08, 0, .02] and sensor['rateHz'] == 8
    for _ in range(100):
        mujoco.mj_step(compiled, data)
    assert not any(w.number for w in data.warning)


def test_usd_runtime_uses_its_profile_and_preserves_unrelated_base_settings(imported_profile, monkeypatch):
    simple = cobot_magic().to_dict()
    simple_runtime = asdict(runtime_config('cobot-magic'))
    text = imported_profile.read_text()
    for old, new in [
        ('linear_speed = 0.45', 'linear_speed = 0.2'),
        ('receipt_ttl = 300.0', 'receipt_ttl = 450.0'),
        ('pour_roll = 2.3', 'pour_roll = 2.5'),
        ('pour_roll_min = 1.7', 'pour_roll_min = 1.8'),
        ('home_speed_rad_s = 0.6', 'home_speed_rad_s = 0.4'),
        ('release_joint_tol = 0.008', 'release_joint_tol = 0.006'),
        ('release_settle_s = 0.6', 'release_settle_s = 1.0'),
        ('deploy_joints = [0.0, 1.57, -1.35, 0.0, 0.0, 0.0]',
         'deploy_joints = [0.0, 1.5, -1.3, 0.0, 0.0, 0.0]'),
    ]:
        text = text.replace(old, new)
    imported_profile.write_text(text)
    monkeypatch.setenv('STEWARD_AGILEX_USD_CONFIG', str(imported_profile))
    model = agilex_usd()
    arm = body_arm('agilex-usd', model)
    assert arm.config.home_speed_rad_s == .4
    assert arm.config.deploy_joints == (0, 1.5, -1.3, 0, 0, 0)
    assert arm.ik.working_joints == arm.config.deploy_joints
    assert arm._working_seeds({'lift': .9, 'reach': .4, 'wrist_pitch': 0})[0][1] == list(arm.config.deploy_joints)
    base = RuntimeConfig()
    base = replace(base, motion=replace(base.motion, position_tol=.0123))
    cfg = runtime_config('agilex-usd', base)
    assert cfg.motion.linear_speed == .2 and cfg.esim.receipt_ttl == 450
    assert cfg.esim.pour_roll == 2.5 and cfg.esim.pour_roll_min == 1.8
    assert cfg.motion.position_tol == .0123
    assert cfg.motion.release_joint_tol == .006 and cfg.motion.release_settle_s == 1.0
    assert cfg.motion.working_lift == arm.config.working_lift
    assert cfg.esim.drawer_entry_lift == cfg.motion.working_lift
    assert cfg.motion.working_reach == arm.config.working_reach
    assert cfg.motion.wrist_pitch_max == arm.geometry.wrist_pitch_max
    assert base.motion.linear_speed != .2
    assert cobot_magic().to_dict() == simple
    assert asdict(runtime_config('cobot-magic')) == simple_runtime
    # A broken USD profile must not be consulted by the simplified body.
    imported_profile.write_text('not valid TOML')
    assert asdict(runtime_config('cobot-magic')) == simple_runtime


@pytest.mark.parametrize('text, field', [
    ('[broken', 'TOML'),
    ('arm_kpp = 1000.0', 'arm_kpp'),
    ('[motion]\nlinear_speeed = 0.3', 'motion.linear_speeed'),
    ('base = 2', 'base'),
    ('template = ""', 'template'),
    ('lift_home = nan', 'lift_home'),
    ('arm_kp = -1.0', 'arm_kp'),
    ('finger_effort = 0.0', 'finger_effort'),
    ('work_pitch_deg = 91.0', 'work_pitch_deg'),
    ('[base]\nkp = [100.0, 100.0]', 'base.kp'),
    ('[base]\nvelocity = [0.0, 0.5, 1.0]', 'base.velocity'),
    ('[cameras.head]\nsize = [320.5, 240]', 'cameras.head.size'),
    ('[cameras.head]\nsize = [0, 240]', 'cameras.head.size'),
    ('[cameras.head]\nfovy_deg = 180.0', 'cameras.head.fovy_deg'),
    ('[cameras.head]\nposition = [0.0, nan, 0.0]', 'cameras.head.position'),
    ('[cameras.wide]\nrate_hz = -1.0', 'cameras.wide.rate_hz'),
    ('[arm]\narm = "X"', 'arm.arm'),
    ('[arm]\ncontinuation_steps = true', 'arm.continuation_steps'),
    ('[arm]\ncontinuation_steps = 0', 'arm.continuation_steps'),
    ('[arm]\nreach = [0.6, 0.1]', 'arm.reach'),
    ('[arm]\nsmall_step = [0.0, 0.2]', 'arm.small_step'),
    ('[arm]\npitch_unseen_cos = 1.5', 'arm.pitch_unseen_cos'),
    ('[motion]\nlinear_speed = 0.8', 'motion.linear_speed'),
    ('[motion]\nangular_speed = 2.0', 'motion.angular_speed'),
    ('[motion]\nretract_reach = 0.9', 'motion.retract_reach'),
    ('[motion]\nrelease_joint_tol = 0.0', 'motion.release_joint_tol'),
    ('[motion]\nrelease_settle_s = -1.0', 'motion.release_settle_s'),
    ('[planning]\nflip_in_air = 1', 'planning.flip_in_air'),
    ('[planning]\nreceipt_ttl = -1.0', 'planning.receipt_ttl'),
    ('[planning]\ndrawer_entry_pitch = 1.1', 'planning.drawer_entry_pitch'),
    ('[planning]\ndrawer_entry_backoff = 0.0', 'planning.drawer_entry_backoff'),
    ('[planning]\npour_roll = 0.2', 'planning.pour_roll'),
    ('[planning]\npour_roll = 4.0', 'planning.pour_roll'),
    ('[planning]\npour_roll_min = 0.0', 'planning.pour_roll_min'),
    ('[planning]\npour_roll_min = 3.2', 'planning.pour_roll_min'),
    ('[planning]\nhandle_probes_side_m = []', 'planning.handle_probes_side_m'),
])
def test_invalid_profile_fails_with_the_field_and_path(tmp_path, text, field):
    profile = tmp_path / 'bad.toml'
    profile.write_text(text)
    with pytest.raises(ValueError) as error:
        load_agilex_profile(profile)
    assert field in str(error.value) and str(profile) in str(error.value)


def test_deployment_seed_is_checked_against_the_imported_limits(imported_profile):
    text = imported_profile.read_text().replace('deploy_joints = [0.0, 1.57', 'deploy_joints = [9.0, 1.57')
    imported_profile.write_text(text)
    with pytest.raises(ValueError, match='arm.deploy_joints.*L_joint1.*outside the USD limits'):
        agilex_usd(imported_profile)


@pytest.mark.parametrize('before, after, field', [
    ('arm_kp = 1500.0', 'arm_kp = true', 'arm_kp'),
    ('kp = [10000.0, 10000.0, 3500.0]', 'kp = [10000.0, 3500.0]', 'base.kp'),
    ('arm_kp = 1500.0', 'arm_kpp = 1500.0', 'arm_kpp'),
    ('arm_kp = 1500.0\n', '', 'arm_kp'),
])
def test_editing_shipped_defaults_cannot_change_the_schema(tmp_path, monkeypatch, before, after, field):
    import steward.assets.agilex_profile as profiles

    edited = tmp_path / 'defaults.toml'
    edited.write_text(PROFILE.read_text().replace(before, after))
    monkeypatch.setattr(profiles, 'PROFILE', edited)
    with pytest.raises(ValueError) as error:
        load_agilex_profile(edited)
    assert field in str(error.value) and str(edited) in str(error.value)
