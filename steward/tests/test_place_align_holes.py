"""Place seats mating screw holes instead of a free-hand offset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.assets.generators import steward_bot
from steward.ego.arm import Arm
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.priors import screw_holes
from steward.math3d import yaw_from_quat


def _track(eid: str, category: str, pos: list[float], size: list[float], yaw: float = 0.0) -> Track:
    return Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos, dtype=float),
                 var=np.full(3, 1e-4), yaw=yaw, yaw_var=1e-3, size=np.array(size, dtype=float),
                 observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)


def _assembly_scene(held: str = "e:wing", tips: dict[str, float] | None = None):
    """Robot south of a table, same docking geometry as the wrist placement fixtures."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:fuse"] = _track("e:fuse", "airplane_fuselage", [0.0, 0.1, 0.77], [0.18, 0.045, 0.04])
    est.tracks["e:wing"] = _track("e:wing", "airplane_wing", [0.15, 0.35, 0.755], [0.055, 0.20, 0.01], yaw=0.3)
    est.tracks["e:gear"] = _track("e:gear", "airplane_landing_gear", [-0.15, 0.35, 0.762], [0.03, 0.08, 0.025])
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    for eid, tip in (tips or {}).items():
        est.tracks[eid].tip_roll = tip
        est.tracks[eid].tip_roll_source = "observed"
    est.held_id = held
    est.tracks[held].held = True
    return model, est.to_scene()


def _esim(model) -> Esim:
    esim = Esim(model)
    esim.arm = Arm(esim.arm.geometry)
    return esim


def test_category_priors_list_the_toy_screw_holes():
    assert {h["id"] for h in screw_holes("airplane_fuselage")} == {"wing_0", "wing_1", "gear_0", "gear_1"}
    assert {h["id"] for h in screw_holes("airplane_wing")} == {"mount_0", "mount_1"}
    assert len(screw_holes("car_body")) == 4
    assert screw_holes("car_wheel")[0]["id"] == "hub"


def test_ego_state_carries_screw_hole_affordances_from_priors():
    _, scene = _assembly_scene()
    holes = {a.value["screw-hole"]["id"]
             for a in scene.entity("e:fuse").rigid.links[0].affordances
             if isinstance(a.value, dict) and "screw-hole" in a.value}
    assert holes == {"wing_0", "wing_1", "gear_0", "gear_1"}


def test_place_wing_on_fuselage_seats_the_wing_mount_holes():
    model, scene = _assembly_scene("e:wing")
    receipt = _esim(model).predict(scene, "place", {"on": "e:fuse"})
    assert receipt["feasible"], receipt["issues"]
    assert receipt["plan"]["alignTo"] == "e:fuse"
    assert receipt["plan"]["mate"] is True
    assert any("aligned screw holes" in i for i in receipt["issues"])
    fuse = scene.entity("e:fuse")
    fx, fy = fuse.pose.value["p"][0], fuse.pose.value["p"][1]
    fyaw = yaw_from_quat(fuse.pose.value["q"])
    spot = receipt["plan"]["spot"]
    # Wing centre should sit 2 cm forward of the fuselage centre (wing holes at x=0.02).
    lx = math.cos(-fyaw) * (spot[0] - fx) - math.sin(-fyaw) * (spot[1] - fy)
    ly = math.sin(-fyaw) * (spot[0] - fx) + math.cos(-fyaw) * (spot[1] - fy)
    assert lx == pytest.approx(0.02, abs=1e-3)
    assert ly == pytest.approx(0.0, abs=1e-3)


def test_over_support_allows_a_wide_mate_whose_footprint_hangs_past_the_part():
    """A wing on a fuselage fails footprint containment; mate mode only needs the centre on it."""
    from steward.ego.skills import SkillExecutor

    class _Fake:
        def _gripper_xy(self):
            return np.array([0.02, 0.0])

    plan = {
        "spot": [0.02, 0.0], "gripSpot": [0.02, 0.0], "heldHalf": 0.10,
        "supportBox": {"cx": 0.0, "cy": 0.0, "yaw": 0.0, "hx": 0.09, "hy": 0.0225},
    }
    assert not SkillExecutor._over_support(_Fake(), plan)
    plan["mate"] = True
    assert SkillExecutor._over_support(_Fake(), plan)


def test_place_gear_on_table_aligns_to_fuselage_gear_holes():
    # Gear mounts are on the underside: tip the fuselage over so those holes face up first.
    model, scene = _assembly_scene("e:gear", tips={"e:fuse": math.pi})
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:fuse", "hole": "gear"})
    assert receipt["feasible"], receipt["issues"]
    fuse = scene.entity("e:fuse")
    spot = receipt["plan"]["spot"]
    assert spot[0] == pytest.approx(fuse.pose.value["p"][0], abs=1e-3)
    assert spot[1] == pytest.approx(fuse.pose.value["p"][1], abs=1e-3)


def test_place_wheel_aligns_hub_to_named_car_mount():
    # Wheel mounts are on the car sides: tip so the -y face (wheel_1) points up.
    model, scene = _assembly_scene("e:wheel", tips={"e:body": -math.pi / 2})
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:body", "hole": "wheel_1"})
    assert receipt["feasible"], receipt["issues"]
    body = scene.entity("e:body")
    from steward.ego.esim import Esim
    planar = Esim._hole_planar(body, next(h for h in
        [{"id": a.value["screw-hole"]["id"], **a.value["screw-hole"]}
         for a in body.rigid.links[0].affordances
         if isinstance(a.value, dict) and "screw-hole" in a.value] if h["id"] == "wheel_1"))
    byaw = yaw_from_quat(body.pose.value["q"])
    bx, by = body.pose.value["p"][0], body.pose.value["p"][1]
    # Planar seat uses the tipped projection, not raw body xy (that was the 6–10 cm miss).
    want = (bx + math.cos(byaw) * planar[0] - math.sin(byaw) * planar[1],
            by + math.sin(byaw) * planar[0] + math.cos(byaw) * planar[1])
    spot = receipt["plan"]["spot"]
    assert spot[0] == pytest.approx(want[0], abs=1e-3)
    assert spot[1] == pytest.approx(want[1], abs=1e-3)


def test_place_wheel_on_upright_car_is_refused_until_holes_face_up():
    model, scene = _assembly_scene("e:wheel")
    receipt = _esim(model).predict(scene, "place", {"on": "e:body", "hole": "wheel_0"})
    assert not receipt["feasible"]
    assert any("do not face up" in i for i in receipt["issues"])


def test_place_screw_into_wheel_requires_tip_down_vertical():
    """flip is refused; roll=±π/2 stands tip-down; omit roll and place also seats tip-down."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.20, 0.35, 0.76], [0.048, 0.048, 0.066])
    est.held_id = "e:screw"
    est.tracks["e:screw"].held = True
    scene = est.to_scene()
    esim = _esim(model)
    # Already upright: omit roll. flip is refused. roll=±π/2 needs a tipable sideways grasp.
    flipped = esim.predict(scene, "place", {"on": "e:wheel", "flip": True})
    assert not flipped["feasible"]
    assert any("tip-down" in i for i in flipped["issues"])
    ok = esim.predict(scene, "place", {"on": "e:wheel"})
    assert ok["feasible"], ok["issues"]
    # With the tip marker, place seats the screw onto the hub.
    assert ok["plan"].get("mate") is True or ok["plan"].get("alignTo") == "e:wheel"


def test_place_sideways_screw_onto_board_auto_stands_tip_down():
    """Sideways screw with shaft across the gripper: place onto the board auto-rolls tip-down."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    grip = est.gripper_position()
    est.tracks["e:screw"] = _track("e:screw", "toy_screw",
                                   [float(grip[0]), float(grip[1]), float(grip[2])],
                                   [0.048, 0.048, 0.066], yaw=-math.pi / 2)
    est.tracks["e:screw"].tip_roll = math.pi / 2
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.mark_held("e:screw")
    receipt = _esim(model).predict(est.to_scene(), "place", {"on": "e:board", "hole": "hole_0"})
    assert receipt["feasible"], receipt["issues"]
    assert abs(abs(receipt["plan"]["wristRoll"]) - math.pi / 2) <= 0.2
    assert any("auto roll" in i and "tip-down" in i for i in receipt["issues"])


def test_wheel_hub_is_through_so_flat_wheel_accepts_screw_even_if_normal_reads_down():
    """Signed normal ``down`` on a through hub must not force a flip before inserting."""
    from steward.ego.priors import screw_holes as prior_holes

    assert prior_holes("car_wheel")[0].get("through") is True
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    # Observed hole reports the opposite normal (as in the old generator); through still applies.
    est.tracks["e:wheel"].screw_holes = [
        {"id": "hub", "p": [0.0, 0.0, 0.008], "n": [0.0, 0.0, -1.0]}]
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.20, 0.35, 0.76], [0.048, 0.048, 0.066])
    est.held_id = "e:screw"
    est.tracks["e:screw"].held = True
    scene = est.to_scene()
    hub = next(a.value["screw-hole"] for a in scene.entity("e:wheel").rigid.links[0].affordances
               if isinstance(a.value, dict) and "screw-hole" in a.value)
    assert hub.get("through") is True
    assert hub["n"][2] == pytest.approx(-1.0)
    assert Esim._hole_up_dot(scene.entity("e:wheel"), hub) < -0.5
    assert Esim._hole_access_up_dot(scene.entity("e:wheel"), hub) > 0.5
    ok = _esim(model).predict(scene, "place", {"on": "e:wheel"})
    assert ok["feasible"], ok["issues"]


def test_ego_screw_holes_through_hub_faces_up_when_axis_vertical():
    import threading

    from steward.ego.runtime import EgoRuntime

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    est.tracks["e:wheel"].tip_roll_source = "observed"
    est.tracks["e:wheel"].screw_holes = [
        {"id": "hub", "p": [0.0, 0.0, 0.008], "n": [0.0, 0.0, -1.0]}]
    rt = EgoRuntime.__new__(EgoRuntime)
    rt.estimator = est
    rt._lock = threading.RLock()
    rt._generation = 0
    rt._episode = None

    class _Offline:
        def sensors(self):
            raise RuntimeError("offline")

    rt.link = _Offline()
    out = rt.screw_holes("e:wheel")
    assert out["holesFaceUp"] is True
    assert out["holes"][0]["through"] is True
    assert out["holes"][0]["facing"] == "up"
    assert out["holes"][0]["upDot"] == pytest.approx(-1.0, abs=0.05)


def test_place_car_body_auto_tips_so_side_holes_face_up():
    model, scene = _assembly_scene("e:body")
    receipt = _esim(model).predict(scene, "place", {"on": "e:table"})
    assert receipt["feasible"], receipt["issues"]
    assert abs(abs(receipt["plan"]["wristRoll"]) - math.pi / 2) <= 0.2
    assert any("auto roll" in i for i in receipt["issues"])


def test_pick_wheel_refused_while_car_holes_face_sideways():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    scene = est.to_scene()
    receipt = _esim(model).predict(scene, "pick", {"target": "e:wheel"})
    assert not receipt["feasible"]
    assert any("do not pick" in i and "facing up" in i for i in receipt["issues"])


def test_pick_wheel_allowed_when_a_car_side_faces_up():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    est.tracks["e:body"].tip_roll = math.pi / 2
    est.tracks["e:body"].tip_roll_source = "observed"
    est.tracks["e:wheel"] = _track("e:wheel", "car_wheel", [0.35, 0.3, 0.758], [0.05, 0.05, 0.016])
    scene = est.to_scene()
    receipt = _esim(model).predict(scene, "pick", {"target": "e:wheel"})
    assert receipt["feasible"], receipt["issues"]


def test_place_wrist_roll_alone_does_not_claim_holes_face_up():
    """Recording wristRoll must not mark tip observed — that was the false holesFaceUp bug."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    est.tracks["e:body"].held = True
    est.held_id = "e:body"
    assert est.release_at("e:body", [0.25, 0.1], tip_roll=math.pi / 2)
    assert est.tracks["e:body"].tip_roll_source == "unknown"
    scene = est.to_scene()
    holes = {a.value["screw-hole"]["id"]: a.value["screw-hole"]
             for a in scene.entity("e:body").rigid.links[0].affordances
             if isinstance(a.value, dict) and "screw-hole" in a.value}
    # Pose stays upright until a detection observes tip, so side mounts still face sideways.
    from steward.math3d import quat_rotate
    n0 = quat_rotate(scene.entity("e:body").pose.value["q"], holes["wheel_0"]["n"])
    assert abs(n0[2]) < 0.5


def test_ego_screw_holes_reports_side_then_up_after_tip():
    import threading

    from steward.ego.runtime import EgoRuntime

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks["e:body"] = _track("e:body", "car_body", [0.25, 0.1, 0.772], [0.16, 0.10, 0.045])
    rt = EgoRuntime.__new__(EgoRuntime)
    rt.estimator = est
    rt._lock = threading.RLock()
    rt._generation = 0
    rt._episode = None

    class _Offline:
        def sensors(self):
            raise RuntimeError("offline")

    rt.link = _Offline()
    flat = rt.screw_holes("e:body")
    assert flat["holesFaceUp"] is False
    assert flat["tipSource"] == "unknown"
    assert flat["hint"]
    est.tracks["e:body"].tip_roll = math.pi / 2
    est.tracks["e:body"].tip_roll_source = "observed"
    tipped = rt.screw_holes("e:body")
    assert tipped["holesFaceUp"] is True
    assert tipped["tipSource"] == "observed"
    assert any(h["facing"] == "up" for h in tipped["holes"])
    assert tipped["tipRollDeg"] == pytest.approx(90.0, abs=0.1)


def test_align_holes_false_keeps_a_manual_offset_on_a_table():
    model, scene = _assembly_scene("e:wing")
    receipt = _esim(model).predict(scene, "place", {"on": "e:table", "alignTo": "e:fuse", "alignHoles": False,
                                                    "offset": [-0.28, -0.28], "clearance": 0.0})
    assert receipt["feasible"], receipt["issues"]
    assert "alignTo" not in receipt["plan"]
    table = scene.entity("e:table")
    tyaw = yaw_from_quat(table.pose.value["q"])
    tx, ty = table.pose.value["p"][0], table.pose.value["p"][1]
    spot = receipt["plan"]["spot"]
    lx = math.cos(-tyaw) * (spot[0] - tx) - math.sin(-tyaw) * (spot[1] - ty)
    ly = math.sin(-tyaw) * (spot[0] - tx) + math.cos(-tyaw) * (spot[1] - ty)
    assert lx == pytest.approx(-0.28, abs=1e-3)
    assert ly == pytest.approx(-0.28, abs=1e-3)
