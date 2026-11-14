"""Offline: PYTHONPATH=src python tests/test_fasteners_math.py (no pytest/MuJoCo)."""
import math
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from steward.assets.generators import (airplane_fuselage, airplane_wing, airplane_landing_gear,
                                       car_body, car_wheel, electric_screwdriver, toy_screw,
                                       printed_board, open_box, cup, bowl, dining_table, steward_bot,
                                       cabinet_with_drawer, sideboard)
from steward.assets.cobot import cobot_magic
from steward.math3d import pose_compose, quat_from_axis_angle
from steward.scene.fasteners import (STATE_KEY, Dwell, FastenConfig, active_welds, component, driving,
                                     features, mating, occupied, records, seated_pose, weld_pairs)
from steward.scene.schema import SceneState, exact, pose
from steward.scene.mjcf import build_mjcf
from steward.scene.validate import validate_world


def fixture():
    parts = [airplane_fuselage(), airplane_wing(), toy_screw(), electric_screwdriver()]
    for ent, eid in zip(parts, ('base', 'plate', 'bolt', 'tool')):
        ent.id = 'w:' + eid
    scene = SceneState(mode='world', entities=parts)
    inv = features(scene)
    upper = next(f for f in inv['screw-hole'] if f.entity == 'w:plate')
    lower = inv['screw-hole'][0]
    screw = inv['screw'][0]
    up = upper.at(pose([0.02, 0, 0.0405]))
    low = lower.at(pose())
    sc = screw.at(pose(up['entry']))
    return scene, upper, up, low, screw, sc


class FastenerMathTests(unittest.TestCase):
    def test_import_does_not_load_mujoco(self):
        # Only assert in the standalone offline invocation; pytest may import it elsewhere.
        if __name__ == '__main__':
            self.assertNotIn('mujoco', sys.modules)

    def test_mating_and_head_seat(self):
        _, upper, up, low, screw, sc = fixture()
        self.assertTrue(mating(up, low, sc, FastenConfig()))
        seated = screw.at(seated_pose(screw, up, sc))
        np.testing.assert_allclose(seated['seat'], up['entry'], atol=1e-12)
        self.assertGreater(float((up['entry'] - seated['tip']) @ up['axis']), 0.017)

    def test_screw_line_must_pass_both_holes(self):
        _, _, up, low, _, sc = fixture()
        low = dict(low, entry=low['entry'] + [-.015, 0, 0])
        sc = dict(sc, tip=sc['tip'] + [.015, 0, 0], seat=sc['seat'] + [.015, 0, 0])
        self.assertFalse(mating(up, low, sc, FastenConfig()))

    def test_near_parallel_screw_snaps_exactly_to_hole_axis(self):
        _, _, up, _, screw, _ = fixture()
        sc = screw.at(pose(up['entry'], quat_from_axis_angle([1, 0, 0], .1)))
        seated = screw.at(seated_pose(screw, up, sc))
        np.testing.assert_allclose(seated['axis'], up['axis'], atol=1e-12)
        np.testing.assert_allclose(seated['seat'], up['entry'], atol=1e-12)

    def test_rejects_lateral_gap_angle_short_screw_and_same_entity(self):
        _, _, up, low, _, sc = fixture()
        for field, value in [('entry', up['entry'] + [0.0201, 0, 0]),
                             ('entry', up['entry'] + [0, 0, 0.0031]),
                             ('axis', np.array([1, 0, 0])), ('entity', low['entity'])]:
            with self.subTest(field=field, value=value):
                self.assertFalse(mating(dict(up, **{field: value}), low, sc, FastenConfig()))
        self.assertFalse(mating(up, low, dict(sc, seat=sc['tip'] + [0, 0, 0.005]), FastenConfig()))
        self.assertFalse(mating(up, low, dict(sc, axis=-sc['axis']), FastenConfig()))
        self.assertFalse(mating(up, low, dict(sc, tip=sc['tip'] + [0, 0, .01]), FastenConfig()))

    def test_arbitrary_world_rotation_and_translation(self):
        scene, upper, _, _, screw, _ = fixture()
        lower = features(scene)['screw-hole'][0]
        transform = pose([.4, -.9, 1.2], quat_from_axis_angle([1, 2, 3], 1.9))
        up = upper.at(pose_compose(transform, pose([.02, 0, .0405])))
        low = lower.at(transform)
        sc = screw.at(pose(up['entry'], transform['q']))
        self.assertTrue(mating(up, low, sc, FastenConfig()))
        np.testing.assert_allclose(screw.at(seated_pose(screw, up, sc))['seat'], up['entry'], atol=1e-12)

    def test_driver_tip_and_direction(self):
        scene, _, _, _, _, sc = fixture()
        f = features(scene)['powered-driver'][0]
        q = quat_from_axis_angle([0, 1, 0], math.pi / 2)
        at_origin = f.at(pose(q=q))
        driver = f.at(pose(sc['head'] - at_origin['tip'], q))
        self.assertTrue(driving(driver, sc, FastenConfig()))
        self.assertFalse(driving(dict(driver, tip=driver['tip'] + [.0151, 0, 0]), sc, FastenConfig()))
        self.assertFalse(driving(dict(driver, axis=-driver['axis']), sc, FastenConfig()))

    def test_continuous_sim_time_and_key_identity(self):
        dwell = Dwell()
        key = ('tool', 'bolt', 'hole1', 'hole2')
        self.assertEqual(dwell.update([key], 5, .5), [])
        self.assertEqual(dwell.update([key], 5.499, .5), [])
        self.assertEqual(dwell.update([key], 5.5, .5), [key])
        dwell.update([], 5.6, .5)
        self.assertEqual(dwell.update([key], 5.7, .5), [])
        other = ('another tool', *key[1:])
        self.assertEqual(dwell.update([other], 6.0, .5), [])
        self.assertEqual(dwell.update([other], 6.5, .5), [other])

    def test_slots_are_per_entity_pair_and_opt_in(self):
        scene, *_ = fixture()
        inv = features(scene)
        pairs = weld_pairs(inv)
        self.assertEqual(len(pairs), 3)  # two parts + one screw
        self.assertEqual(len(set(pairs)), len(pairs))
        inv['screw'] = []
        self.assertEqual(weld_pairs(inv), [])
        for ent in scene.entities:
            ent.rigid.links[0].affordances = [exact('graspable')]
        self.assertEqual(weld_pairs(features(scene)), [])

    def test_mjcf_slots_are_inactive_and_never_implicitly_added_to_ego(self):
        scene, *_ = fixture()
        self.assertEqual(validate_world(scene.to_dict()), [])
        self.assertEqual(build_mjcf(scene).fastener_welds, {})
        build = build_mjcf(scene, fasteners=True)
        xml = ET.fromstring(build.xml)
        rows = xml.findall('equality/weld')
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.get('active') == 'false' for row in rows))
        self.assertTrue(all(row.get('relpose') == '0 0 0 1 0 0 0' for row in rows))
        scene.mode = 'ego'
        self.assertEqual(build_mjcf(scene, fasteners=True).fastener_welds, {})

    def test_other_task_assets_emit_identical_xml_with_fastening_enabled(self):
        for task, factories in [
            ('puzzle', [printed_board, open_box, dining_table, steward_bot]),
            ('chores', [cup, bowl, cabinet_with_drawer, sideboard, steward_bot]),
            ('cobot', [cobot_magic, bowl, dining_table]),
            ('shuttle', [cup, bowl, dining_table, cabinet_with_drawer, sideboard, steward_bot])]:
            with self.subTest(task=task):
                scene = SceneState(mode='world', entities=[factory() for factory in factories])
                for i, ent in enumerate(scene.entities):
                    ent.id = f'w:part-{i}'
                ordinary = build_mjcf(scene)
                enabled = build_mjcf(scene, fasteners=True)
                self.assertEqual(ordinary.xml, enabled.xml)
                self.assertEqual(enabled.fastener_welds, {})

    def test_mjcf_restores_explicit_relative_pose_and_embedded_screw_contacts(self):
        scene, *_ = fixture()
        scene.entity('w:bolt').state_vars[STATE_KEY] = exact({'holes': [
            {'entity': 'w:base', 'link': 'body', 'hole': 'wing_0'},
            {'entity': 'w:plate', 'link': 'body', 'hole': 'mount_0'}], 'welds': [
                {'a': 'w:base', 'b': 'w:plate', 'rel': pose([.02, 0, .0405])},
                {'a': 'w:bolt', 'b': 'w:plate', 'rel': pose([0, 0, .008])}]})
        xml = ET.fromstring(build_mjcf(scene, fasteners=True).xml)
        rows = [row for row in xml.findall('equality/weld') if row.get('active') == 'true']
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].get('relpose'), '0.02 0 0.0405 1 0 0 0')
        screw_body = next(b for b in xml.iter('body') if b.get('name') == 'w:bolt/body')
        self.assertTrue(all(g.get('contype') == '0' and g.get('conaffinity') == '0'
                            for g in screw_body.findall('geom')))
        scene.entities = [e for e in scene.entities if e.id != 'w:base']
        xml = ET.fromstring(build_mjcf(scene, fasteners=True).xml)
        self.assertFalse(any(row.get('active') == 'true' for row in xml.findall('equality/weld')))

    def test_records_roundtrip_deletion_and_ego_isolation(self):
        scene, *_ = fixture()
        bolt = scene.entity('w:bolt')
        bolt.state_vars[STATE_KEY] = exact({'holes': [
            {'entity': 'w:base', 'link': 'body', 'hole': 'wing_0'},
            {'entity': 'w:plate', 'link': 'body', 'hole': 'mount_0'}], 'welds': [
                {'a': 'w:base', 'b': 'w:plate', 'rel': pose([.02, 0, .0405])},
                {'a': 'w:bolt', 'b': 'w:plate', 'rel': pose()}]})
        scene = SceneState.from_dict(scene.to_dict())
        self.assertEqual(len(records(scene)), 1)
        self.assertEqual(len(occupied(scene)), 2)
        self.assertEqual(component('w:base', active_welds(scene)), {'w:base', 'w:plate', 'w:bolt'})
        scene.mode = 'ego'
        self.assertEqual(records(scene), {})
        scene.mode = 'world'
        scene.entities = [e for e in scene.entities if e.id != 'w:plate']
        self.assertEqual(records(scene), {})

    def test_side_and_underside_holes_have_physical_axes(self):
        scene = SceneState(mode='world', entities=[car_body(), airplane_fuselage(), airplane_landing_gear()])
        for i, e in enumerate(scene.entities):
            e.id = str(i)
        holes = features(scene)['screw-hole']
        side = [f for f in holes if f.entity == '0']
        for f in side:
            self.assertEqual(f.spec['axis'], [0, math.copysign(1, f.spec['p'][1]), 0])
        underside = [f for f in holes if f.spec['id'].startswith('gear')]
        self.assertTrue(all(f.spec['axis'] == [0, 0, -1] for f in underside))
        gear = [f for f in holes if f.entity == '2']
        self.assertTrue(all(abs(f.spec['entry'][2] - .0085) < 1e-12 for f in gear))

    def test_wheel_fits_side_mount_and_gear_fits_inverted_base(self):
        # The same predicate works outside the upright wing example.
        for base, top, hole_id, base_q, top_p in [
            (car_body(), car_wheel(), 'wheel_0', quat_from_axis_angle([1, 0, 0], math.pi/2), [.05, -.01575, .0509]),
            (airplane_fuselage(), airplane_landing_gear(), 'gear_0', [0, 1, 0, 0], [0, 0, .0005])]:
            base.id, top.id = 'base', 'top'
            screw = toy_screw(); screw.id = 'screw'
            inv = features(SceneState(mode='world', entities=[base, top, screw]))
            low = next(f for f in inv['screw-hole'] if f.entity == 'base' and f.spec['id'] == hole_id).at(pose(q=base_q))
            high = next(f for f in inv['screw-hole'] if f.entity == 'top').at(pose(top_p))
            # For the upside-down base, the two y-symmetric holes exchange indices.
            if hole_id == 'gear_0':
                low = next(f for f in inv['screw-hole'] if f.spec['id'] == 'gear_1').at(pose(q=base_q))
            sc = inv['screw'][0].at(pose(high['entry']))
            self.assertTrue(mating(high, low, sc, FastenConfig()), (high, low))


if __name__ == '__main__':
    unittest.main()
