"""Body composition, policy substitution and rejection of mismatched legacy controllers."""
import numpy as np
import pytest

from test_agilex import imported_profile  # noqa: F401 - shared pytest fixture
from steward.assets.agilex import agilex_usd
from steward.assets.agilex_profile import load_agilex_profile
from steward.assets.cobot import cobot_magic
from steward.ego.bodies import prepare_body
from steward.ego.cobot_runtime import CobotEgoRuntime
from steward.ego.cobot_skills import CobotSkills
from steward.ego.motion_policy import MotionPolicyConfig, PiperMotionPolicy, PlanarMotionPolicy
from steward.ego.piper_ik import IkError, PiperIK
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.skills import Job, MotionConfig, OutOfReach, SkillExecutor, SkillFailed, SkillStopped
from steward.math3d import pose_compose


def test_usd_composition_reads_one_profile_for_model_arm_and_policy(imported_profile, monkeypatch):
    import steward.ego.agilex as assembly

    imported_profile.write_text(imported_profile.read_text().replace('approach_floor = 0.85', 'approach_floor = 0.93')
                               .replace('handle_lower_clearance = 0.10', 'handle_lower_clearance = 0.07'))
    monkeypatch.setenv('STEWARD_AGILEX_USD_CONFIG', str(imported_profile))
    loads = []

    def read():
        loads.append(True)
        return load_agilex_profile()

    monkeypatch.setattr(assembly, 'load_agilex_profile', read)
    setup = prepare_body('agilex-usd')
    assert loads == [True]
    assert setup.arm.ik.robot is setup.model
    assert isinstance(setup.arm.motion_policy, PiperMotionPolicy)
    assert setup.arm.motion_policy.settings.approach_floor == .93
    assert setup.arm.motion_policy.settings.handle_lower_clearance == .07
    assert setup.config.motion.working_lift == setup.arm.config.working_lift
    runtime = EgoRuntime(RobotLink('http://unused.invalid'), setup.model, setup.config, arm=setup.arm)
    assert runtime.executor.motion_policy is setup.arm.motion_policy
    assert runtime.esim.arm is setup.arm
    assert set(runtime.estimator.cameras) == {'head_camera', 'wide_camera'}
    # The next body's policy is independent, even with an invalid USD profile.
    imported_profile.write_text('invalid TOML')
    simplified = prepare_body('cobot-magic').arm.motion_policy
    assert isinstance(simplified, PiperMotionPolicy)
    assert simplified.settings.approach_floor == .85
    assert simplified is not setup.arm.motion_policy


def test_custom_policy_drives_job_approach_withdrawal_and_recovery(monkeypatch):
    events = []

    class Policy(PlanarMotionPolicy):
        def approach(self, driver, plan, xy, lift):
            events.append(('approach', xy, lift))
            driver._command({'lift': lift})

        def withdraw(self, driver, plan):
            events.append(('withdraw',))
            raise OutOfReach('withdrawal blocked')

        def recover(self, driver):
            events.append(('recover',))
            driver._command({'reach': .22})

    commands = []
    executor = SkillExecutor(commands.append, lambda: {'stamp': 1}, lambda eid: eid, motion_policy=Policy())

    def pick(job, plan):
        executor._arm_engaged = True
        executor._reach_over(plan, [.5, .2], .9)
        executor._withdraw(plan)

    monkeypatch.setattr(executor, '_pick', pick)
    job = Job('job-1', 'pick', 'r-1')
    executor._run(job, 'pick', {'plan': {}})
    assert job.state == 'failed' and job.message == 'withdrawal blocked'
    assert events == [('approach', [.5, .2], .9), ('withdraw',), ('recover',)]
    assert commands == [{'lift': .9}, {'reach': .22}]


def test_drawer_lowering_failure_recovers_and_logs_on_the_active_job(monkeypatch):
    state = {'stamp': 0., 'joints': {'lift': .85, 'reach': .4}, 'odometry': {'x': 0., 'y': 0., 'yaw': 0.}}
    lowered = []

    def command(values):
        if values.get('gripper') == 'close':
            raise SkillStopped()  # Stop at the grasp after the real handle-approach policy.
        if 'lift' in values and values['lift'] < .85:
            lowered.append(values['lift'])
            if len(lowered) == 1:
                raise OutOfReach('lowering posture blocked')
        state['joints'].update(values)

    executor = SkillExecutor(command, lambda: state, lambda eid: eid)
    for name in ('_dock', '_align_lateral', '_wait', '_translate_to'):
        monkeypatch.setattr(executor, name, lambda *args: None)
    monkeypatch.setattr(executor, '_reach_to', lambda xy: .4)
    monkeypatch.setattr(executor, '_gripper_xy', lambda: np.array([.7, 0.]))
    job = Job('drawer-recovery', 'open', 'receipt')
    with pytest.raises(SkillStopped):
        executor._drawer(job, {'handle': [.7, 0., .65]})
    assert lowered[:2] == pytest.approx([.8, .8])
    assert state['joints']['lift'] == pytest.approx(.65)
    assert job.log == ['handle lower failed (lowering posture blocked); reseed and retry']


def test_policy_recovery_budget_controls_base_retries(monkeypatch):
    policy = PlanarMotionPolicy(MotionPolicyConfig(reach_attempts=3, base_adjustment=.025))
    state = {'joints': {'wrist_pitch': 1.05, 'reach': .2}, 'odometry': {'x': 0, 'y': 0, 'yaw': 0}}
    executor = SkillExecutor(lambda cmd: None, lambda: state, lambda eid: eid,
                             MotionConfig(wrist_pitch_max=1.05), motion_policy=policy)
    steps = []
    attempts = []

    def unreachable(name, value):
        attempts.append((name, value))
        raise OutOfReach('blocked')

    monkeypatch.setattr(executor, '_reach_to', lambda xy: .6)
    monkeypatch.setattr(executor, '_set_joint', unreachable)
    monkeypatch.setattr(executor, '_translate_to', lambda x, y, yaw: steps.append([x, y, yaw]))
    with pytest.raises(OutOfReach, match='blocked'):
        executor._reach_out([1, 0], 1.05)
    assert attempts == [('reach', .6)] * 3
    assert steps == [[.025, 0, 0]] * 2


@pytest.mark.parametrize('direction', [-1., 1.])
def test_drawer_travel_starts_from_the_actual_handle_grasp(monkeypatch, direction):
    state = {'stamp': 0., 'odometry': {'x': 1.5, 'y': .2, 'yaw': 0.},
             'gripper': {'holdingHandle': True}, 'joints': {'lift': .65, 'reach': .4}}
    destinations = []
    executor = SkillExecutor(lambda command: None, lambda: state, lambda eid: eid)
    for name in ('_dock', '_wait', '_set_joint'):
        monkeypatch.setattr(executor, name, lambda *args: None)
    monkeypatch.setattr(executor.motion_policy, 'approach_handle', lambda *args: (.65, .44, .4))

    def translate(x, y, yaw, *args):
        destinations.append([x, y])
        state['odometry'] = {'x': x, 'y': y, 'yaw': yaw}

    monkeypatch.setattr(executor, '_translate_to', translate)
    moved = executor._drawer(Job('handle', 'open' if direction > 0 else 'close', 'r'),
                             {'handle': [2., .2, .65], 'drawer': 'd', 'distance': .3, 'direction': direction,
                              'axis': [-1., 0.], 'yaw': 0., 'dock': [1.8, .2], 'end': [1.5, .2],
                              'viewing': [1., .2]})
    assert destinations[0] == pytest.approx([1.5 - direction * .3, .2])
    assert moved == pytest.approx(direction * .3)


def test_stop_during_failure_recovery_sends_no_later_commands(monkeypatch):
    executor = SkillExecutor(lambda cmd: None, lambda: {'stamp': 1, 'joints': {'lift': .9}}, lambda eid: eid)
    attempted = []

    def stop(name, value):
        attempted.append(name)
        raise SkillStopped()

    def pick(job, plan):
        executor._arm_engaged = True
        raise SkillFailed('missed grasp')

    monkeypatch.setattr(executor, '_set_joint', stop)
    monkeypatch.setattr(executor, '_ramp_joint', lambda *args: pytest.fail('recovery continued after stop'))
    monkeypatch.setattr(executor, '_pick', pick)
    job = Job('job-1', 'pick', 'r-1')
    executor._run(job, 'pick', {'plan': {}})
    assert attempted == ['lift']
    assert job.state == 'failed' and job.message == 'missed grasp'


@pytest.mark.parametrize('fragment', [
    'reach_attempts = 0', 'reach_attempts = 1.5', 'base_adjustment = nan',
    'intermediate_reaches = [0.44, 0.4]', 'intermediate_reaches = [0.9]',
    'recovery_lift_max = 0.5', 'handle_lower_reaches = [0.42, 0.36]', 'handle_reach_cap = 0.3',
    'handle_lower_clearance = 0.0',
])
def test_invalid_body_trajectory_fails_at_profile_load(tmp_path, fragment):
    path = tmp_path / 'body.toml'
    path.write_text('[trajectory]\n' + fragment + '\n')
    with pytest.raises(ValueError, match='trajectory'):
        load_agilex_profile(path)


def test_legacy_skill_refuses_the_real_model(imported_profile):
    ik = PiperIK(agilex_usd(imported_profile))
    with pytest.raises(ValueError, match='only cobot-magic'):
        CobotSkills(lambda cmd: pytest.fail('sent command'), lambda: {}, ik=ik)


def test_identity_is_served_on_the_robot_port_and_legacy_runtime_rejects_it(tmp_path, imported_profile, monkeypatch):
    from steward.authoring.service import AuthoringService, build_cobot_workstation
    from steward.world.server import StewardHost

    monkeypatch.setenv('STEWARD_AGILEX_USD_CONFIG', str(imported_profile))
    build_cobot_workstation(AuthoringService(tmp_path))
    host = StewardHost(tmp_path, scenario='cobot-table', body='agilex-usd', phase='phase0',
                       speed=0, run_dir=tmp_path / 'run').start()
    try:
        link = RobotLink(host.robot_servers['w:cobot-1'].url)
        assert link.identity() == {'embodiment': 'agilex-usd'}
        link.require_body('agilex-usd')
        with pytest.raises(ValueError, match='cannot drive'):
            CobotEgoRuntime(link)
    finally:
        host.stop()


def test_relaxed_ik_keeps_the_declared_tcp_fixed(monkeypatch):
    model = cobot_magic().to_dict()
    for link in model['body']['links']:
        if link['id'] == 'L_link6':
            for affordance in link['affordances']:
                if isinstance(affordance['value'], dict) and 'tcp-offset' in affordance['value']:
                    affordance['value']['tcp-offset']['p'] = [.03, .02, .22]
    ik = PiperIK(model)
    requested = {'p': [.4, .2, .8], 'q': [1, 0, 0, 0]}
    accepted = []

    def solve(arm, position, orientation, **kwargs):
        if 'orientation_tolerance' not in kwargs:
            raise IkError('strict pose unavailable')
        accepted.append({'p': position, 'q': orientation})
        return [0] * 6

    monkeypatch.setattr(ik, 'solve', solve)
    monkeypatch.setattr(ik, 'fk', lambda arm, joints: accepted[-1])
    ik.solve_tool('L', requested['p'], requested['q'])
    offset = {'p': [.03, .02, .22], 'q': [1, 0, 0, 0]}
    assert pose_compose(accepted[-1], offset)['p'] == pytest.approx(pose_compose(requested, offset)['p'])
    assert not np.allclose(accepted[-1]['p'], requested['p'])


def test_old_arm_import_keeps_the_shared_controller_identity():
    from steward.ego.cobot_arm import PiperArm as OldArm
    from steward.ego.piper_arm import PiperArm
    from steward.ego.ik import CobotIK

    assert OldArm is PiperArm
    assert isinstance(CobotIK(), PiperIK)


def test_collision_policy_is_injected_into_both_arm_constraints():
    from steward.ego.collision_policy import PiperSelfCollisionPolicy
    from steward.ego.piper_arm import PiperArm

    selected = []

    class FrameOnly:
        def pairs(self, robot, arm):
            selected.append(arm)
            return PiperSelfCollisionPolicy(body_links=('frame',)).pairs(robot, arm)

    policy = FrameOnly()
    arm = PiperArm(cobot_magic().to_dict(), collision_policy=policy)
    assert arm.ik.collision_policy is policy and selected == ['L', 'R']


def test_collision_policy_rejects_a_missing_body_link():
    from steward.ego.collision_policy import PiperSelfCollisionPolicy

    with pytest.raises(ValueError, match='missing links'):
        PiperIK(cobot_magic().to_dict(), collision_policy=PiperSelfCollisionPolicy(body_links=('absent',)))
