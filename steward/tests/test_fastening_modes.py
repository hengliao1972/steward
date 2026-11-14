"""Offline adapter/config/story checks: no MuJoCo model, step or GL."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np

from test_fasteners_math import fixture
from steward.math3d import quat_from_axis_angle, pose_compose, quat_rotate
from steward.scene.fasteners import FastenConfig, Dwell, features, mating, driving, stationary
from steward.scene.schema import pose


class FasteningModeTests(unittest.TestCase):
    def test_config_environment_and_validation(self):
        self.assertEqual(FastenConfig.from_env({}), FastenConfig(mode="off"))
        for mode in ('driver', 'seat', 'off'):
            cfg = FastenConfig.from_env({'STEWARD_FASTEN_MODE': mode,
                'STEWARD_FASTEN_RADIAL': '.012', 'STEWARD_FASTEN_DWELL': '.3',
                'STEWARD_FASTEN_DRIVER_ANGLE_DEG': '30'})
            self.assertEqual(cfg.mode, mode)
            self.assertEqual(cfg.radial, .012)
            self.assertEqual(cfg.dwell, .3)
            self.assertAlmostEqual(cfg.driver_angle, math.pi / 6)
        for kwargs in ({'mode': 'typo'}, {'radial': -1}, {'dwell': 0},
                       {'tip_distance': float('nan')}, {'angle': math.pi},
                       {'driver_angle': float('inf')}):
            with self.assertRaises(ValueError):
                FastenConfig(**kwargs)

    def test_relaxed_boundaries_and_stationary_gate(self):
        _, _, up, low, _, sc = fixture()
        cfg = FastenConfig()
        for distance, allowed in ((.0199, True), (.0201, False)):
            shifted = dict(low, entry=low['entry'] + [distance, 0, 0])
            self.assertEqual(mating(up, shifted, sc, cfg), allowed)
        for degrees, allowed in ((24.9, True), (25.1, False)):
            a = math.radians(degrees)
            driver = {'axis': np.array([math.sin(a), 0, -math.cos(a)]),
                      'tip': sc['head'] + [.0149, 0, 0]}
            self.assertEqual(driving(driver, sc, cfg), allowed)
        self.assertTrue(stationary([0, 0, .099, .0099, 0, 0], cfg))
        self.assertFalse(stationary([0, 0, .101, 0, 0, 0], cfg))
        self.assertFalse(stationary([0, 0, 0, .0101, 0, 0], cfg))
        self.assertFalse(stationary([0, 0, 0, float('nan'), 0, 0], cfg))

    def test_world_adapter_modes_without_physics(self):
        # Stub only name lookup; production candidate/dwell gates execute normally.
        stub = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_EQUALITY=0), mj_name2id=lambda *a: 0)
        path = Path(__file__).parents[1] / 'src/steward/world/fastening.py'
        spec = importlib.util.spec_from_file_location('_offline_fastening', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'mujoco': stub}):
            spec.loader.exec_module(module)
        for mode in ('driver', 'seat', 'off'):
            with self.subTest(mode=mode):
                scene, _, up, _, _, sc = fixture()
                held = SimpleNamespace(attached='w:tool', closed=True)
                sim = SimpleNamespace(template=scene, fasten=FastenConfig(mode=mode), model=None,
                    build=SimpleNamespace(fastener_welds={('w:base', 'w:plate'): 'slot'}), time=0.,
                    robots={'fixture': SimpleNamespace(grippers={'g': held})})
                adapter = module.Fastening(sim)
                q = quat_from_axis_angle([0, 1, 0], math.pi / 2)
                tool = features(scene)['powered-driver'][0]
                poses = {'w:base': pose(), 'w:plate': pose([.02, 0, .0405]),
                         'w:bolt': pose(up['entry']),
                         'w:tool': pose(sc['head'] - tool.at(pose(q=q))['tip'], q)}
                adapter.pose = lambda f: poses[f.entity]
                adapter.is_stationary = lambda eid: True
                adapter.attach = Mock()
                held.closed = False
                adapter.update()
                sim.time = .2
                adapter.update()
                self.assertEqual(adapter.attach.called, mode == 'seat')
                adapter.attach.reset_mock()
                adapter.dwell = Dwell()
                held.closed = True
                sim.time = 1.
                adapter.update()
                sim.time = 1.199
                adapter.update()
                adapter.attach.assert_not_called()
                sim.time = 1.2
                adapter.update()
                self.assertEqual(adapter.attach.called, mode != 'off')
                if mode == 'seat':
                    self.assertIsNone(adapter.attach.call_args.args[0])
                    adapter.attach.reset_mock()
                    adapter.is_stationary = lambda eid: False
                    sim.time = 2.
                    adapter.update()
                    sim.time = 3.
                    adapter.update()
                    adapter.attach.assert_not_called()
                    adapter.is_stationary = lambda eid: True
                    adapter.update()
                    sim.time = 3.1
                    adapter.update()
                    adapter.attach.assert_not_called()
                adapter.attach.reset_mock()
                adapter.dwell = Dwell()
                held.attached = 'w:bolt'
                sim.time = 4.
                adapter.update()
                sim.time = 5.
                adapter.update()
                adapter.attach.assert_not_called()
        if __name__ == '__main__':
            self.assertNotIn('mujoco', sys.modules)

    def test_story_import_needs_no_gl_and_manual_conflicts_with_upright(self):
        path = Path(__file__).parents[1] / 'scripts/fasten_story.py'
        spec = importlib.util.spec_from_file_location('fasten_story', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(len(module.LABELS), 7)
        if __name__ == '__main__':
            self.assertNotIn('mujoco', sys.modules)
        # Independent analytic manual poses; no authoring/PIL/renderer dependency.
        from steward.world.goals import evaluate
        from steward.assets.generators import airplane_fuselage, airplane_landing_gear, toy_screw
        from steward.scene.schema import SceneState
        base, gear, screw = airplane_fuselage(), airplane_landing_gear(), toy_screw()
        base.id, gear.id, screw.id = 'base', 'gear', 'screw'
        inv = features(SceneState(mode='world', entities=[base, gear, screw]))
        flipped = pose(q=[0, 1, 0, 0])
        low = next(f for f in inv['screw-hole'] if f.spec['id'] == 'gear_1').at(flipped)
        up = next(f for f in inv['screw-hole'] if f.entity == 'gear').at(pose([0, 0, .0005]))
        sc = inv['screw'][0].at(pose(up['entry']))
        self.assertTrue(mating(up, low, sc, FastenConfig()))
        final_gear = pose_compose(flipped, pose([0, 0, .0005]))
        self.assertLess(quat_rotate(final_gear['q'], [0, 0, 1])[2], 0)
        poses = {'base': pose(), 'gear': final_gear}
        goal = {'kind': 'relative-poses', 'upright': True, 'entities': ['base', 'gear'],
                'constraints': [{'a': 'base', 'b': 'gear', 'offset': [0, 0]}]}
        report = evaluate(goal, poses.__getitem__)
        self.assertFalse(report['errors'][0]['satisfied'])
        self.assertEqual(report['errors'][0]['positionErrM'], 0)
        self.assertEqual(report['errors'][0]['yawErrDeg'], 0)


if __name__ == '__main__':
    unittest.main()
