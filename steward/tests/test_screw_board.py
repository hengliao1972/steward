"""Screw-board scenario: a four-hole board, four screws, and an upright-driven screwdriver."""

from __future__ import annotations

import math

import numpy as np
import pytest

from steward.assets.generators import electric_screwdriver, screw_board, steward_bot, toy_screw
from steward.authoring.service import AuthoringService, build_screw_board, screw_board_goal
from steward.ego.arm import Arm
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.priors import screw_holes


def _track(eid: str, category: str, pos: list[float], size: list[float], yaw: float = 0.0) -> Track:
    return Track(id=eid, category_counts={category: 5.0}, pos=np.array(pos, dtype=float),
                 var=np.full(3, 1e-4), yaw=yaw, yaw_var=1e-3, size=np.array(size, dtype=float),
                 observations=10, first_seen=0.0, last_seen=0.0, existence=1.0)


def _esim(model) -> Esim:
    esim = Esim(model)
    esim.arm = Arm(esim.arm.geometry)
    return esim


def _hole_ids(entity) -> set[str]:
    out = set()
    for link in entity.rigid.links:
        for aff in link.affordances:
            value = aff.value
            if isinstance(value, dict) and "screw-hole" in value:
                out.add(value["screw-hole"]["id"])
    return out


def test_screw_board_and_screw_carry_the_holes_the_task_describes():
    assert _hole_ids(screw_board()) == {"hole_0", "hole_1", "hole_2", "hole_3"}
    assert _hole_ids(toy_screw()) == {"tip"}
    assert {h["id"] for h in screw_holes("screw_board")} == {"hole_0", "hole_1", "hole_2", "hole_3"}
    assert screw_holes("toy_screw")[0]["id"] == "tip"
    assert any(a.value == "screwdriver" for a in electric_screwdriver().rigid.links[0].affordances)
    # Easy-mode geometry: ~3× shaft/head and board hole radius vs the original tiny parts.
    screw_geoms = [g for g in toy_screw().rigid.links[0].geometry["collision"] if g.get("shape") == "cylinder"]
    radii = sorted(g["size"][0] for g in screw_geoms)
    assert abs(radii[0] - 0.012) < 1e-6 and abs(radii[1] - 0.024) < 1e-6
    hole = next(g for g in screw_board().rigid.links[0].geometry["collision"] if g.get("shape") == "cylinder")
    assert abs(hole["size"][0] - 0.021) < 1e-6


def test_build_screw_board_publishes_a_settled_scene(tmp_path):
    out = build_screw_board(AuthoringService(tmp_path))
    assert out["scene"]["report"]["ok"]
    assert out["scene"]["entities"] == 9
    scenario = AuthoringService(tmp_path).load_scenario("screw-board")
    assert scenario["task"].startswith("桌子上放着一块木板")
    assert "尖端朝下" in scenario["task"] and "drive_screw" in scenario["task"]
    assert "insert_screw_into_hole" in scenario["task"] and "tipSource=observed" in scenario["task"]
    assert "一次只拧一颗" in scenario["task"] and "刀身" in scenario["task"]
    assert scenario["goal"] == screw_board_goal()
    ids = {e["id"] for e in scenario["sceneState"]["entities"]}
    assert {"w:table", "w:board", "w:screwdriver", "w:screw-0", "w:screw-3", "w:robot-1"} <= ids
    by_id = {e["id"]: e for e in scenario["sceneState"]["entities"]}
    assert by_id["w:board"]["category"]["value"] == "screw_board"
    affs = by_id["w:board"]["body"]["links"][0]["affordances"]
    hole_ids = {a["value"]["screw-hole"]["id"] for a in affs
                if isinstance(a.get("value"), dict) and "screw-hole" in a["value"]}
    assert hole_ids == {"hole_0", "hole_1", "hole_2", "hole_3"}


def test_insert_screw_into_hole_predicts_tip_down_place_with_retries():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.25, 0.0, 0.79], [0.048, 0.048, 0.066])
    est.held_id = "e:screw"
    est.tracks["e:screw"].held = True
    esim = _esim(model)
    # Upright held screw (tip observed): insert plans tip-down without a stand roll.
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.tracks["e:screw"].tip_roll = 0.0
    ok = esim.predict(est.to_scene(), "insert_screw_into_hole",
                      {"on": "e:board", "hole": "hole_0", "maxAttempts": 2})
    assert ok["feasible"], ok["issues"]
    assert ok["skill"] == "insert_screw_into_hole"
    assert ok["plan"]["insert"] is True
    assert ok["plan"]["insertRelease"] is True
    assert ok["plan"]["gentle"] is True
    assert ok["plan"]["maxAttempts"] == 2
    assert ok["plan"]["insertParams"]["object"] == "e:screw"
    assert ok["plan"]["insertParams"]["on"] == "e:board"
    assert ok["plan"]["support"] == "e:board"
    assert any("retry" in i for i in ok["issues"])
    assert any("tipSource" in i for i in ok["issues"])
    # Must be holding a screw; board alone is refused.
    est.held_id = None
    est.tracks["e:screw"].held = False
    empty = esim.predict(est.to_scene(), "insert_screw_into_hole", {"on": "e:board"})
    assert not empty["feasible"]
    assert any("holding" in i for i in empty["issues"])


def test_insert_screw_runtime_requires_observed_tip_on_held_screw():
    import threading

    from steward.ego.runtime import EgoRuntime

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.15, 0.35, 0.92], [0.048, 0.048, 0.066])
    est.held_id = "e:screw"
    est.tracks["e:screw"].held = True
    rt = EgoRuntime.__new__(EgoRuntime)
    rt.estimator = est
    rt.esim = _esim(model)
    rt.executor = type("E", (), {"pull_of": staticmethod(lambda _t: {})})()
    rt.config = type("C", (), {"max_risk": 0.35, "max_pick_risk": 0.35})()
    rt.receipts = {}
    rt._receipt_generation = {}
    rt._generation = 0
    rt._lock = threading.RLock()
    rt._episode = None

    class _Offline:
        def sensors(self):
            raise RuntimeError("offline")

    rt.link = _Offline()
    unknown = EgoRuntime.predict(rt, "insert_screw_into_hole", {"on": "e:board", "hole": "hole_0"})
    assert not unknown["feasible"]
    assert any("tipSource" in i or "not yet confirmed" in i for i in unknown["issues"])
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.tracks["e:screw"].tip_roll = 0.0
    ok = EgoRuntime.predict(rt, "insert_screw_into_hole", {"on": "e:board", "hole": "hole_0"})
    assert ok["feasible"], ok["issues"]
    assert ok["plan"].get("insertRelease") is True


def test_insert_screw_stands_sideways_and_inverts_head_down_onto_hole():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    grip = est.gripper_position()
    # Sideways with shaft across the gripper (yaw=-π/2): wrist roll can stand tip-down.
    est.tracks["e:screw"] = _track("e:screw", "toy_screw",
                                   [float(grip[0]), float(grip[1]), float(grip[2])],
                                   [0.048, 0.048, 0.066], yaw=-math.pi / 2)
    est.tracks["e:screw"].tip_roll = math.pi / 2
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.mark_held("e:screw")
    esim = _esim(model)
    side = esim.predict(est.to_scene(), "insert_screw_into_hole", {"on": "e:board", "hole": "hole_0"})
    assert side["feasible"], side["issues"]
    assert abs(abs(side["plan"]["wristRoll"]) - math.pi / 2) <= 0.2
    assert math.hypot(side["plan"]["gripSpot"][0] - side["plan"]["spot"][0],
                      side["plan"]["gripSpot"][1] - side["plan"]["spot"][1]) < 0.03
    assert any("tip-down" in i or "points up" in i for i in side["issues"])
    est.joints["wrist_roll"] = float(side["plan"]["wristRoll"])
    est._follow_held()
    stood = est.to_scene(self_frame={"joints": est.joints, "odometry": est.odometry})
    assert Esim._screw_head_up_dot(stood.find("e:screw")) > 0.7
    # Head-down must invert with π, not a quarter-turn that leaves it sideways.
    est.joints["wrist_roll"] = 0.0
    est.tracks["e:screw"].tip_roll = math.pi
    est.mark_held("e:screw")
    down = esim.predict(est.to_scene(), "insert_screw_into_hole", {"on": "e:board", "hole": "hole_0"})
    assert down["feasible"], down["issues"]
    assert abs(abs(down["plan"]["wristRoll"]) - math.pi) <= 0.2
    assert any("invert" in i or "head-down" in i for i in down["issues"])
    assert math.hypot(down["plan"]["gripSpot"][0] - down["plan"]["spot"][0],
                      down["plan"]["gripSpot"][1] - down["plan"]["spot"][1]) < 0.03


def test_insert_screw_refuses_sideways_shaft_along_roll_axis():
    """Knocked-over screw grasped end-on: auto roll must refuse, not claim tip-down then leave it flat."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    grip = est.gripper_position()
    est.tracks["e:screw"] = _track("e:screw", "toy_screw",
                                   [float(grip[0]), float(grip[1]), float(grip[2])],
                                   [0.048, 0.048, 0.066], yaw=0.0)
    est.tracks["e:screw"].tip_roll = math.pi / 2
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.mark_held("e:screw")
    bad = _esim(model).predict(est.to_scene(), "insert_screw_into_hole", {"on": "e:board", "hole": "hole_3"})
    assert not bad["feasible"]
    assert any("gripper reach" in i or "crosses the fingers" in i or "headGrip" in i for i in bad["issues"])


def test_insert_screw_into_hole_retries_when_verify_fails():
    from steward.ego.skills import Job, SkillExecutor, SkillFailed

    plans = {
        "place": {
            "object": "e:screw", "support": "e:board", "spot": [0.07, 0.15],
            "yaw": 0.0, "lift": 0.9, "dock": [0.0, 0.5], "path": [],
            "insert": True, "maxAttempts": 2,
            "insertParams": {"object": "e:screw", "on": "e:board", "hole": "hole_0", "maxAttempts": 2},
        },
        "pick": {
            "object": "e:screw", "spot": [0.07, 0.15], "yaw": 0.0, "lift": 0.9,
            "dock": [0.0, 0.5], "path": [],
        },
    }
    calls = {"place": 0, "pick": 0, "verify": 0}

    class FakeExec(SkillExecutor):
        def _place(self, job, plan):  # noqa: ANN001
            calls["place"] += 1
            job.log.append("placed")

        def _pick(self, job, plan):  # noqa: ANN001
            calls["pick"] += 1
            job.log.append("picked")

    def verify(_sid: str) -> tuple[bool, str]:
        calls["verify"] += 1
        if calls["verify"] < 2:
            return False, "not in hole"
        return True, "seated tip-down"

    def replan_pick(_sid: str) -> dict:
        return plans["pick"]

    def replan_place(_insert: dict) -> dict:
        return plans["place"]

    holding = {"v": True}

    def state() -> dict:
        return {"gripper": {"holding": holding["v"]}}

    def after_place():
        holding["v"] = False

    ex = FakeExec(lambda _c: None, state, lambda *_a, **_k: None,
                  verify_screw_seat=verify, replan_pick=replan_pick,
                  replan_insert_place=replan_place)
    # First place leaves gripper empty so the second attempt re-picks.
    orig_place = ex._place

    def place_and_release(job, plan):  # noqa: ANN001
        orig_place(job, plan)
        after_place()

    ex._place = place_and_release  # type: ignore[method-assign]
    job = Job(id="j1", skill="insert_screw_into_hole", receipt="r1")
    ex._insert_screw_into_hole(job, plans["place"])
    assert calls["place"] == 2
    assert calls["pick"] == 1
    assert calls["verify"] == 2
    assert "inserted e:screw" in job.message

    # Exhausted attempts raise.
    calls["place"] = calls["pick"] = calls["verify"] = 0
    holding["v"] = True

    def always_fail(_sid: str) -> tuple[bool, str]:
        calls["verify"] += 1
        return False, "still tipped"

    ex._verify_screw_seat = always_fail
    job2 = Job(id="j2", skill="insert_screw_into_hole", receipt="r2")
    with pytest.raises(SkillFailed, match="after 2 attempts"):
        ex._insert_screw_into_hole(job2, plans["place"])
    assert calls["place"] == 2
    assert calls["verify"] == 2


def test_held_screwdriver_tip_follows_wrist_roll_in_ego_state():
    """ES must tip the held driver when the wrist rolls — otherwise drive_screw plans from a flat belief."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    grip = est.gripper_position()
    # Shaft along +X (arm lateral when facing −Y): flat on the table, tip_roll 0.
    est.tracks["e:driver"] = _track("e:driver", "electric_screwdriver",
                                    [float(grip[0]), float(grip[1]), float(grip[2])],
                                    [0.16, 0.035, 0.04], yaw=0.0)
    est.tracks["e:driver"].tip_roll = 0.0
    est.tracks["e:driver"].tip_roll_source = "observed"
    est.mark_held("e:driver")
    flat = est.to_scene()
    assert abs(Esim._screwdriver_bit_down_dot(flat.find("e:driver"))) < 0.35
    # Stand bit-down: yaw shaft already lateral; roll −π/2 about reach.
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": -math.pi / 2}
    est._follow_held()
    stood = est.to_scene(self_frame={"joints": est.joints, "odometry": est.odometry})
    bit = Esim._screwdriver_bit_down_dot(stood.find("e:driver"))
    assert bit < -0.7, f"ES bit should point down after wrist_roll, got bitDownDot={bit:.2f}"


def test_drive_screw_requires_screwdriver_and_seated_screw():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    # Screw at board hole_0 (body [0.07, 0.05] → world [0.07, 0.15]), upright head-up.
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.07, 0.15, 0.802], [0.048, 0.048, 0.066])
    grip = est.gripper_position()
    # Lateral shaft (yaw=0): bit along gripper ±y so wrist roll stands it tip-down.
    est.tracks["e:driver"] = _track("e:driver", "electric_screwdriver",
                                    [float(grip[0]), float(grip[1]), float(grip[2])],
                                    [0.16, 0.035, 0.04], yaw=0.0)
    est.mark_held("e:driver")
    esim = _esim(model)
    ok = esim.predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert ok["feasible"], ok["issues"]
    assert ok["plan"]["target"] == "e:screw"
    assert ok["plan"].get("bitDown") is True
    assert "wristYaw" not in ok["plan"]
    assert abs(abs(ok["plan"]["wristRoll"]) - math.pi / 2) <= 0.2
    assert ok["plan"]["pressLift"] < ok["plan"]["lift"]
    # Centred grasp: after stand the bit hangs under the body centre ≡ pinch, so grip≈spot.
    assert math.hypot(ok["plan"]["gripSpot"][0] - ok["plan"]["spot"][0],
                      ok["plan"]["gripSpot"][1] - ok["plan"]["spot"][1]) < 0.02
    assert ok["plan"]["bitHang"] >= 0.05
    assert any("seated" in i and "head-up" in i for i in ok["issues"])
    assert any("points down" in i for i in ok["issues"])
    assert any("bit aimed at screw" in i for i in ok["issues"])
    # Apply the planned roll: ES bit must point down.
    est.joints["wrist_roll"] = float(ok["plan"]["wristRoll"])
    est._follow_held()
    stood = est.to_scene(self_frame={"joints": est.joints, "odometry": est.odometry})
    assert Esim._screwdriver_bit_down_dot(stood.find("e:driver")) < -0.7
    # Holding a screw instead of the driver is refused.
    est.held_id = "e:screw"
    est.tracks["e:screw"].held = True
    est.tracks["e:driver"].held = False
    est.tracks["e:driver"].held_q = None
    bad = esim.predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert not bad["feasible"]
    assert any("screwdriver" in i for i in bad["issues"])


def test_pick_screwdriver_prefers_broadside_over_shorter_end_on():
    """Shaft along +X: south dock is end-on and shorter; pick must still take a broadside table edge."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": -1.85, "yaw": math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, -0.7, 0.375], [0.8, 0.8, 0.75])
    shaft_yaw = 0.0
    est.tracks["e:driver"] = _track(
        "e:driver", "electric_screwdriver", [0.25, -0.45, 0.78], [0.16, 0.035, 0.04], yaw=shaft_yaw)
    ok = _esim(model).predict(est.to_scene(), "pick", {"target": "e:driver"})
    assert ok["feasible"], ok["issues"]
    # Broadside: dock heading ⟂ shaft (|cos| small). End-on from the south is |cos|≈1.
    assert abs(math.cos(ok["plan"]["yaw"] - shaft_yaw)) <= 0.71


def test_drive_screw_shifts_grip_for_reach_axis_hold_offset():
    """Off-centre along reach: grip must leave the screw so the bit (under the body) lands on it."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.07, 0.15, 0.802], [0.048, 0.048, 0.066])
    grip = est.gripper_position()
    # Body centre 5 cm closer to the table than the pinch (gripper +x = world −Y when facing −Y).
    est.tracks["e:driver"] = _track(
        "e:driver", "electric_screwdriver",
        [float(grip[0]), float(grip[1]) - 0.05, float(grip[2])],
        [0.16, 0.035, 0.04], yaw=0.0)
    est.mark_held("e:driver")
    assert abs(float(est.tracks["e:driver"].held_offset[0]) - 0.05) < 1e-6
    ok = _esim(model).predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert ok["feasible"], ok["issues"]
    gx, gy = ok["plan"]["gripSpot"]
    sx, sy = ok["plan"]["spot"]
    # Reach hold survives ±π/2 stand: pinch sits ~5 cm farther from the table than the screw.
    assert abs(gx - sx) < 0.02
    assert abs(gy - (sy + 0.05)) < 0.02
    assert math.hypot(gx - sx, gy - sy) > 0.03


def test_drive_screw_refuses_end_on_grasp_along_roll_axis():
    """Shaft along gripper reach: yaw+roll cannot tip the bit — predict must refuse, not fake bitDown."""
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.95, "reach": 0.25, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.07, 0.15, 0.802], [0.048, 0.048, 0.066])
    grip = est.gripper_position()
    est.tracks["e:driver"] = _track("e:driver", "electric_screwdriver",
                                    [float(grip[0]), float(grip[1]), float(grip[2])],
                                    [0.16, 0.035, 0.04], yaw=-math.pi / 2)
    est.mark_held("e:driver")
    bad = _esim(model).predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert not bad["feasible"]
    assert any("wrist-roll" in i or "crosses the fingers" in i or "bitGrip" in i for i in bad["issues"])


def test_drive_screw_refuses_screw_not_in_hole_or_not_head_up():
    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:driver"] = _track("e:driver", "electric_screwdriver", [0.25, 0.3, 0.78], [0.16, 0.035, 0.04])
    est.held_id = "e:driver"
    est.tracks["e:driver"].held = True
    esim = _esim(model)
    # Far from every hole.
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.20, 0.25, 0.802], [0.048, 0.048, 0.066])
    far = esim.predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert not far["feasible"]
    assert any("not in a screw hole" in i for i in far["issues"])
    # In a hole but tipped on its side (observed tip).
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.07, 0.15, 0.802], [0.048, 0.048, 0.066])
    est.tracks["e:screw"].tip_roll = math.pi / 2
    est.tracks["e:screw"].tip_roll_source = "observed"
    tipped = esim.predict(est.to_scene(), "drive_screw", {"target": "e:screw"})
    assert not tipped["feasible"]
    assert any("head up" in i for i in tipped["issues"])


def test_drive_screw_runtime_requires_observed_upright_tip():
    import threading

    from steward.ego.runtime import EgoRuntime

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0, "wrist_roll": 0.0}
    est.tracks["e:table"] = _track("e:table", "table", [0.0, 0.0, 0.375], [0.8, 0.8, 0.75])
    est.tracks["e:board"] = _track("e:board", "screw_board", [0.0, 0.1, 0.76], [0.28, 0.20, 0.018])
    est.tracks["e:screw"] = _track("e:screw", "toy_screw", [0.07, 0.15, 0.802], [0.048, 0.048, 0.066])
    est.tracks["e:driver"] = _track("e:driver", "electric_screwdriver", [0.25, 0.3, 0.78], [0.16, 0.035, 0.04])
    est.held_id = "e:driver"
    est.tracks["e:driver"].held = True
    rt = EgoRuntime.__new__(EgoRuntime)
    rt.estimator = est
    rt.esim = _esim(model)
    rt.executor = type("E", (), {"pull_of": staticmethod(lambda _t: {})})()
    rt.config = type("C", (), {"max_risk": 0.35, "max_pick_risk": 0.35})()
    rt.receipts = {}
    rt._receipt_generation = {}
    rt._generation = 0
    rt._lock = threading.RLock()
    rt._episode = None

    class _Offline:
        def sensors(self):
            raise RuntimeError("offline")

    rt.link = _Offline()
    unknown = EgoRuntime.predict(rt, "drive_screw", {"target": "e:screw"})
    assert not unknown["feasible"]
    assert any("tipSource" in i or "not yet confirmed" in i for i in unknown["issues"])
    est.tracks["e:screw"].tip_roll_source = "observed"
    est.tracks["e:screw"].tip_roll = 0.0
    ok = EgoRuntime.predict(rt, "drive_screw", {"target": "e:screw"})
    assert ok["feasible"], ok["issues"]
