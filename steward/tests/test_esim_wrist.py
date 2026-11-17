"""Held-object planning preserves orientation and checks the route from physical encoders."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.arm import Arm
from steward.ego.esim import Esim, EsimConfig
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.planner import entity_box
from steward.scene.schema import exact


def _scene(yaw=0.0, roll=0.0):
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": yaw, "wrist_roll": roll}
    for eid, category, pos, size in [("e:4", "table", [0.0, 0.0, 0.375], [0.6, 0.6, 0.75]),
                                     ("e:3", "puzzle_piece", [0.0, 0.6, 0.85], [0.115, 0.115, 0.006])]:
        est.tracks[eid] = Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos),
                               var=np.full(3, 1e-4), yaw=-math.pi / 2, yaw_var=1e-3, size=np.array(size),
                               observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)
    est.held_id, est.tracks["e:3"].held = "e:3", True
    return model, est.to_scene()


class OrientationArm(Arm):
    def __init__(self, geometry, accepts=lambda yaw, roll: True):
        super().__init__(geometry)
        self.accepts = accepts
        self.paths = []

    def reachable(self, lift, reach, wrist_yaw=0.0, wrist_roll=0.0, wrist_pitch=0.0):
        return (super().reachable(lift, reach, wrist_yaw, wrist_roll, wrist_pitch)
                and self.accepts(wrist_yaw, wrist_roll))

    def reachable_path(self, targets, *, start_joints=None):
        self.paths.append((targets, start_joints))
        return super().reachable_path(targets, start_joints=start_joints)


@pytest.mark.parametrize('x, enabled', [(2.485, True), (2.50, True), (2.3, True), (2.485, False)])
def test_drawer_entry_uses_a_tilted_front_grasp_only_under_the_cabinet_top(monkeypatch, x, enabled):
    covered = x >= 2.45
    scene, esim = _drawer_entry_scene(monkeypatch, covered, enabled)
    scene.entity('e:bottle').pose.value['p'][0] = x
    receipt = esim.predict(scene, 'pick', {'target': 'e:bottle'})
    assert receipt['feasible'], receipt['issues']
    plan = receipt['plan']
    if covered and enabled:
        assert plan['closePitch'] == .5
        assert plan['lift'] == pytest.approx(.63)
        face = min(entity_box(scene.entity('e:cabinet'))[0].corners()[:, 0])
        grasp = x + plan['graspOffset'][0]
        assert grasp <= face
        assert grasp - plan['drawerEntry'] <= face - esim.config.drawer_entry_backoff + 1e-9
        assert plan['drawerEntrySide'] > 0
        assert plan['drawerEntry'] > 0 and plan['drawerExit'] > plan['drawerEntry']
    else:
        assert 'drawerEntry' not in plan


def test_drawer_entry_refuses_a_close_pose_outside_the_arm_workspace(monkeypatch):
    scene, esim = _drawer_entry_scene(monkeypatch, True, True)
    reachable = esim.arm.reachable
    monkeypatch.setattr(esim.arm, 'reachable', lambda *args, **kwargs:
                        kwargs.get('wrist_pitch') != .5 and reachable(*args, **kwargs))
    receipt = esim.predict(scene, 'pick', {'target': 'e:bottle'})
    assert not receipt['feasible']
    assert 'drawer entry is outside the arm workspace' in '; '.join(receipt['issues'])


@pytest.mark.parametrize('robot_xy', [(1., .3), (2.7, -.3)])
def test_covered_drawer_approach_uses_the_open_front_and_working_height(monkeypatch, robot_xy):
    scene, esim = _drawer_entry_scene(monkeypatch, True, True)
    scene.entity(scene.self_id).pose.value['p'][:2] = robot_xy
    monkeypatch.delattr(esim, '_approach')
    receipt = esim.predict(scene, 'pick', {'target': 'e:bottle'})
    assert receipt['feasible'], receipt['issues']
    plan = receipt['plan']
    front = esim._front(scene, scene.entity('e:cabinet')).yaw + math.pi
    assert math.cos(plan['yaw'] - front) >= math.sqrt(.5)
    assert plan['approachLift'] >= esim.config.drawer_entry_lift
    assert plan['wristPitch'] == esim.config.drawer_entry_pitch


def test_minimum_lift_search_stays_inside_the_requested_grasp_band():
    model, _ = _scene()
    arm = Esim(model).arm
    assert arm.min_lift_for(.3, lo=.6, hi=.7) == .6
    assert arm.min_lift_for(.3, lo=.8, hi=.7) is None


def _drawer_entry_scene(monkeypatch, covered, enabled):
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {'x': 1., 'y': .3, 'yaw': 0., 'sigma': {'pos': .003, 'yaw': .003}}
    est.joints = {'lift': .85, 'reach': .4, 'wrist_yaw': 0., 'wrist_roll': 0., 'wrist_pitch': 0.}
    for eid, category, pos, size in [
        ('e:cabinet', 'cabinet', [2.7, .6, .4], [.5, .5, .8]),
        ('e:drawer', 'drawer', [2.4, .6, .65], [.5, .4, .22]),
        ('e:bottle', 'pill_bottle', [2.485 if covered else 2.3, .46, .595], [.05, .05, .08]),
    ]:
        est.tracks[eid] = Track(id=eid, category_counts={category: 5.}, pos=np.array(pos),
                               var=np.full(3, 1e-4), yaw=0., yaw_var=1e-3, size=np.array(size),
                               observations=20, first_seen=0., last_seen=0., existence=1.)
    esim = Esim(model, EsimConfig(drawer_entry_pitch=.5 if enabled else 0.))
    esim.arm = Arm(dataclasses.replace(esim.arm.geometry, wrist_pitch_max=1.05))
    # Isolate the close path after navigation has supplied a reachable dock.
    monkeypatch.setattr(esim, '_approach', lambda *args, **kwargs:
                        {'path': [[1., .3], [1.3, .15]], 'predock': [1.3, .15], 'dock': [1.65, .15],
                         'yaw': 0., 'reach': .45, 'wristPitch': .25})
    return est.to_scene(), esim


@pytest.mark.parametrize("case,yaw,roll", [("yaw", 1.1, 0.0), ("roll", 0.0, 0.4)])
def test_held_orientation_can_make_a_zero_orientation_endpoint_unreachable(case, yaw, roll):
    model, scene = _scene(yaw, roll)
    esim = Esim(model)
    esim.arm = OrientationArm(esim.arm.geometry, lambda y, r: abs(y) < 0.1 and abs(r) < 0.1)
    assert esim.arm.reachable(0.8, 0.3)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1]})
    expected = json.loads((Path(__file__).parent / "expected/wrist-placement.json").read_text())
    assert {"feasible": receipt["feasible"], "issues": receipt["issues"]} == expected[case]


def test_dock_selection_uses_required_tool_yaw_not_only_its_scalar_limit():
    model, scene = _scene()
    esim = Esim(model)
    esim.arm = OrientationArm(esim.arm.geometry, lambda yaw, roll: abs(yaw) < 0.6)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.0})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["wristYaw"] == pytest.approx(0.0)
    assert receipt["plan"]["yaw"] == pytest.approx(0.0)


def test_placement_checks_rotation_before_lowering_from_current_encoders():
    model, scene = _scene(0.3, 0.2)
    esim = Esim(model)
    arm = esim.arm = OrientationArm(esim.arm.geometry)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.2})
    assert receipt["feasible"], receipt["issues"]
    transitions = [(path, start) for path, start in arm.paths if start is not None]
    assert transitions
    for path, start in transitions:
        assert path[0]["wrist_yaw"] == start["wrist_yaw"] == 0.3
        assert all(p["wrist_roll"] == 0.2 for p in path)
        lowered = next(i for i, p in enumerate(path) if p["lift"] < 0.85)
        assert all(p["wrist_yaw"] == pytest.approx(path[-1]["wrist_yaw"]) for p in path[lowered:])
        assert all(abs(b["wrist_yaw"] - a["wrist_yaw"]) <= esim.config.wrist_check_step + 1e-9
                   for a, b in zip(path, path[1:]))


def test_endpoint_reachability_does_not_override_a_blocked_rotation_path():
    model, scene = _scene()
    esim = Esim(model)

    class BlockedPath(OrientationArm):
        def reachable_path(self, targets, *, start_joints=None):
            return start_joints is None and super().reachable_path(targets)

    esim.arm = BlockedPath(esim.arm.geometry)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": 0.2})
    assert not receipt["feasible"]
    assert "no continuous arm approach" in "; ".join(receipt["issues"])


def test_placement_starts_from_a_carrying_reach_short_of_the_planning_range():
    model, scene = _scene()
    esim = Esim(model)
    # Raise the planning minimum above the carrying reach so the approach must clamp first.
    geometry = dataclasses.replace(esim.arm.geometry, reach_min=0.2)
    arm = esim.arm = OrientationArm(geometry)
    assert esim._arm_joints(scene)["reach"] < geometry.reach_min
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1]})
    assert receipt["feasible"], receipt["issues"]
    transitions = [path for path, start in arm.paths if start is not None]
    assert transitions and all(path[0]["reach"] == geometry.reach_min for path in transitions)


def test_place_with_a_quarter_turn_roll_tips_the_tool_onto_its_side():
    model, scene = _scene()
    esim = Esim(model)
    arm = esim.arm = OrientationArm(esim.arm.geometry)
    receipt = esim.predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "roll": math.pi / 2})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["wristRoll"] == pytest.approx(math.pi / 2)
    rolled = [path for path, _ in arm.paths if any(abs(t["wrist_roll"] - math.pi / 2) < 1e-6 for t in path)]
    assert rolled


def test_place_rejects_flip_together_with_roll():
    model, scene = _scene()
    receipt = Esim(model).predict(scene, "place", {"on": "e:4", "flip": True, "roll": math.pi / 2})
    assert not receipt["feasible"]
    assert "cannot flip and roll" in "; ".join(receipt["issues"])


def test_tabletop_flip_fits_the_slide_robots_declared_wrist_limits():
    model, scene = _scene()
    receipt = Esim(model).predict(scene, "place", {"on": "e:4", "offset": [0.0, 0.1], "flip": True})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["wristRoll"] == pytest.approx(math.pi)


@pytest.mark.parametrize("at", [[0.0, 0.6], [0.6, 1.0]])
def test_floor_flip_plans_a_half_turn_before_release(at):
    model, scene = _scene()
    receipt = Esim(model).predict(scene, "place", {"at": at, "flip": True})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["wristRoll"] == pytest.approx(math.pi)


def test_floor_flip_respects_an_explicitly_narrower_wrist_limit():
    from steward.ego.esim import EsimConfig

    model, scene = _scene()
    receipt = Esim(model, EsimConfig(wrist_roll_max=2.5)).predict(scene, "place", {"at": [0.6, 1.0], "flip": True})
    assert not receipt["feasible"]


def test_slide_arm_does_not_plan_roll_beyond_the_model_joint_stop():
    from steward.ego.esim import EsimConfig

    model, scene = _scene()
    joint = next(j for j in model["body"]["joints"] if j["id"] == "wrist_roll")
    joint["limits"]["value"] = {"lo": -2.0, "hi": 2.0}
    esim = Esim(model, EsimConfig(wrist_roll_max=math.pi))
    assert esim.arm.geometry.wrist_roll_max == 2.0
    receipt = esim.predict(scene, "place", {"at": [0.6, 1.0], "flip": True})
    assert not receipt["feasible"]


def test_floor_flip_rejects_a_blocked_path_even_if_endpoints_are_reachable():
    model, scene = _scene()
    esim = Esim(model)

    class BlockedPath(OrientationArm):
        def reachable_path(self, targets, *, start_joints=None):
            return False

    esim.arm = BlockedPath(esim.arm.geometry)
    assert esim.arm.reachable(0.01, 0.1, wrist_roll=math.pi)
    receipt = esim.predict(scene, "place", {"at": [0.6, 1.0], "flip": True})
    assert not receipt["feasible"]
    assert "current arm posture" in "; ".join(receipt["issues"])


def test_pouring_preserves_held_yaw_while_sampling_roll():
    model, scene = _scene(0.3)
    scene.entity("e:3").rigid.links[0].affordances.append(exact("pourable"))
    esim = Esim(model)
    arm = esim.arm = OrientationArm(esim.arm.geometry)
    receipt = esim.predict(scene, "pour", {"on": "e:4"})
    assert receipt["feasible"], receipt["issues"]
    tips = [path for path, _ in arm.paths if any(abs(t["wrist_roll"]) > 1.0 for t in path)]
    assert tips
    assert all(t["wrist_yaw"] == pytest.approx(0.3) for path in tips for t in path)


@pytest.mark.parametrize("planned_yaw", [None, 1.1])
def test_turnover_checks_the_planned_or_held_yaw(monkeypatch, planned_yaw):
    model, scene = _scene(0.3)
    esim = Esim(model)
    esim.config.flip_pitch_min = 0.0
    checked = []

    def turn_over(lift, reach, pitch, floor, surface, edge, wrist_yaw=0.0):
        checked.append(wrist_yaw)
        return floor

    monkeypatch.setattr(esim.arm, "turn_over", turn_over)
    plan = {"reach": 0.3, "dock": [0.0, 1.0], "yaw": -math.pi / 2}
    if planned_yaw is not None:
        plan["wristYaw"] = planned_yaw
    box, _, top = entity_box(scene.entity("e:4"))
    assert esim._turn_over_plan(scene, scene.entity("e:3"), plan, 0.8, top, box)["lift"] == 0.8
    assert checked == [0.3 if planned_yaw is None else planned_yaw]
