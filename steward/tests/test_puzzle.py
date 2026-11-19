"""Picture-puzzle scenarios and the generic capabilities they rely on: printed faces, pourable
containers, the wrist, close inspection, annotations and arrangement goals."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import pathlib
import sys
import threading
import time
import traceback

import mujoco
import numpy as np
import pytest
from PIL import Image

from steward.assets.generators import steward_bot
from steward.authoring.service import PUZZLE_PIECE_M, AuthoringService, puzzle_box_m
from steward.ego.estimator import Appearance, EgoEstimator, Track
from steward.ego.perception import face_colour_stats, top_face
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.math3d import quat_rotate, wrap_angle
from steward.mission import _approach_to_observe, _explore
from steward.net import call
from steward.scene.schema import SceneState
from steward.world.goals import evaluate
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def _scenario(kitchen_root, name="puzzle-cat") -> dict:
    return AuthoringService(kitchen_root).load_scenario(name)


def test_every_puzzle_is_a_scenario_at_phase0_with_an_arrangement_goal(kitchen_root):
    for name, pieces in (("puzzle-cat", 9), ("puzzle-greatwall", 9), ("puzzle-dog", 16)):
        scenario = _scenario(kitchen_root, name)
        assert scenario["phase"] == "phase0" and scenario["goal"]["kind"] == "relative-poses"
        ids = {e["id"] for e in scenario["sceneState"]["entities"]}
        assert {"w:box", "w:lid", "w:table-1"} <= ids and sum(i.startswith("w:piece-") for i in ids) == pieces
        side = math.isqrt(pieces)
        assert len(scenario["goal"]["constraints"]) == 2 * side * (side - 1)
        assert f"共 {pieces} 片" in scenario["task"]


def test_the_four_by_four_pieces_lie_inside_their_box_and_the_pour_spot_fits_the_table(kitchen_root):
    from steward.ego.esim import EsimConfig

    sim = WorldSim(_scenario(kitchen_root, "puzzle-dog")["sceneState"])
    sim.step_seconds(0.5)
    box = sim._entity_pose("w:box")["p"]
    half_box = puzzle_box_m(4) / 2
    for r in range(4):
        for c in range(4):
            p = sim._entity_pose(f"w:piece-{r}{c}")["p"]
            assert abs(p[0] - box[0]) < half_box and abs(p[1] - box[1]) < half_box
    table = next(e for e in _scenario(kitchen_root, "puzzle-dog")["sceneState"]["entities"] if e["id"] == "w:table-1")
    assert table["asset"]["id"] == "puzzle-table-180x57"
    assert 0.575 / 2 > half_box + EsimConfig().pour_spill_sizes * PUZZLE_PIECE_M


def test_a_printed_board_shows_its_picture_on_top_with_image_up_towards_plus_y(kitchen_root):
    scene = _scenario(kitchen_root)["sceneState"]
    lid = next(e for e in scene["entities"] if e["id"] == "w:lid")
    png = base64.b64decode(lid["body"]["links"][0]["geometry"]["collision"][0]["texture"])
    picture = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), float)
    sim = WorldSim(scene)
    sim.edit_move("w:lid", {"at": {"x": 2.0, "y": 1.5}, "yaw": 0.0})
    sim.step_seconds(0.3)
    p = sim._entity_pose("w:lid")["p"]
    top = sim.render({"lookat": [p[0], p[1], p[2]], "distance": 0.6, "azimuth": 90.0, "elevation": -90.0}, 64, 64)
    seen = np.asarray(Image.open(io.BytesIO(top)).convert("RGB"), float)
    expected = np.asarray(Image.fromarray(picture.astype(np.uint8)).resize((16, 16)), float)
    sample = np.asarray(Image.fromarray(seen[24:40, 24:40].astype(np.uint8)).resize((16, 16)), float)
    upright = np.abs(sample - np.asarray(Image.fromarray(expected[4:12, 4:12].astype(np.uint8)).resize((16, 16)), float)).mean()
    turned = np.abs(sample - np.rot90(np.asarray(Image.fromarray(expected[4:12, 4:12].astype(np.uint8)).resize((16, 16)), float), 2)).mean()
    assert upright < turned, "the picture is not mirrored or turned on the lid"


def test_the_goal_counts_neighbours_in_place_wherever_and_however_turned_the_picture_lies(kitchen_root):
    scenario = _scenario(kitchen_root)
    sim = WorldSim(scenario["sceneState"])
    assert evaluate(scenario["goal"], sim._entity_pose)["satisfied"] == 0
    sim.edit_move("w:lid", {"at": {"x": 2.0, "y": 1.5}})
    step = PUZZLE_PIECE_M + 0.004
    for r in range(3):
        for c in range(3):
            if (r, c) == (2, 2):
                continue  # one piece missing
            sim.edit_move(f"w:piece-{r}{c}", {"on": "w:table-1", "offset": [0.3 + c * step, 0.12 - r * step], "yaw": 0.0})
    sim.step_seconds(0.3)
    report = evaluate(scenario["goal"], sim._entity_pose)
    assert report["largestGroup"] == 8 and report["satisfied"] == 10 and not report["complete"]
    # each constraint carries its own error, so a caller sees which seam is off by how much
    by_pair = {(e["a"], e["b"]): e for e in report["errors"]}
    assert len(report["errors"]) == len(scenario["goal"]["constraints"])
    assert all(e["satisfied"] for pair, e in by_pair.items() if "w:piece-22" not in pair)
    assert all(0.0 <= e["positionErrM"] <= 0.03 and e["yawErrDeg"] <= 15.0
               for e in by_pair.values() if e["satisfied"])
    missing = [e for pair, e in by_pair.items() if "w:piece-22" in pair]
    assert len(missing) == 2 and not any(e["satisfied"] for e in missing)
    assert all(e["positionErrM"] > 0.03 for e in missing)


def test_a_pourable_box_is_grasped_with_its_contents_only_without_its_lid_and_carries_them(kitchen_root):
    sim = WorldSim(_scenario(kitchen_root)["sceneState"])
    box = sim._entity_pose("w:box")["p"]
    reach = 0.45
    sim.command("w:robot-1", {"base": [box[0] - (0.26 + reach), box[1], 0.0], "lift": box[2] + 0.03, "reach": reach})
    sim.step_seconds(4.0)
    sim.command("w:robot-1", {"gripper": "close"})
    sim.step_seconds(0.2)
    assert sim.robots["w:robot-1"].attached == "w:lid", "closing from above, the gripper meets the lid first"
    sim.command("w:robot-1", {"gripper": "open"})
    sim.edit_move("w:lid", {"at": {"x": 2.0, "y": 1.5}})
    sim.step_seconds(0.5)
    sim.command("w:robot-1", {"lift": box[2] + 0.025})
    sim.step_seconds(1.0)
    sim.command("w:robot-1", {"gripper": "close"})
    sim.step_seconds(0.2)
    assert sim.robots["w:robot-1"].attached == "w:box", "a box is taken with what lies in it"
    lift0 = sim.sensors("w:robot-1")["joints"]["lift"]
    for k in range(1, 30):
        sim.command("w:robot-1", {"lift": lift0 + 0.1 * k / 29})
        sim.step_seconds(0.04)
    sim.step_seconds(0.5)
    rise = sim._entity_pose("w:box")["p"][2] - box[2]
    assert rise > 0.08
    assert all(sim._entity_pose(f"w:piece-{r}{c}")["p"][2] - box[2] > 0.08 for r in range(3) for c in range(3)), \
        "the pieces rise with the box"


def test_fingers_take_a_piece_beside_the_box_wall_the_box_by_its_rim_and_nothing_through_a_lid(kitchen_root):
    sim = WorldSim(_scenario(kitchen_root)["sceneState"])
    sim.step_seconds(0.5)  # settled: what rests on what comes from the contacts
    rs = sim.robots["w:robot-1"]
    link, g = next(iter(rs.grippers.items()))

    def close_at(point) -> str | None:
        sim.grasp_point = lambda robot, gripper=None: {"p": list(point), "q": [1.0, 0.0, 0.0, 0.0]}
        g.attached = g.handle = None
        sim._try_attach(rs, g, link)
        return g.attached

    box = sim.entity_obb("w:box")
    corner = min((f"w:piece-{r}{c}" for r in range(3) for c in range(3)),
                 key=lambda p: sim._entity_pose(p)["p"][0])  # one of the pieces nearest the box's -x wall
    piece = sim._entity_pose(corner)["p"]
    assert close_at(piece) is None, "under the lid nothing in the box can be taken"
    sim.edit_move("w:lid", {"at": {"x": 2.0, "y": 1.5}})
    sim.step_seconds(0.5)
    assert close_at(sim._entity_pose(corner)["p"]) == corner, "on a piece's middle by the wall: the piece, not the box"
    from steward.ego.esim import EsimConfig

    rim = [box.center[0] - box.size[0] / 2 + EsimConfig().edge_grasp_inset, box.center[1], box.center[2]]
    assert close_at(rim) == "w:box", "on the rim, the wall between the fingers: the box"


def _track(oid, category, pos, size, observations=10, yaw=0.0):
    return Track(id=oid, category_counts={category: 5.0}, pos=np.array(pos, float), var=np.full(3, 1e-4), yaw=yaw,
                 yaw_var=1e-3, size=np.array(size, float), observations=observations, first_seen=0.0, last_seen=0.0,
                 existence=1.0)


def test_an_annotation_names_an_object_and_its_affordances_for_planning():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:3"] = _track("e:3", "unknown", [0.5, 0.0, 0.8], [0.4, 0.4, 0.05])
    est.annotate("e:3", "box", ["graspable", "container", "pourable"])
    ent = next(e for e in est.to_scene().entities if e.id == "e:3")
    assert ent.category.value == "box"
    assert [a.value for a in ent.rigid.links[0].affordances] == ["graspable", "container", "pourable"]
    with pytest.raises(KeyError):
        est.annotate("e:99", "box")


def test_a_lid_lying_on_a_box_is_not_folded_into_it():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    box = _track("e:1", "unknown", [0.5, 0.0, 0.775], [0.41, 0.41, 0.05])
    lid = _track("e:2", "unknown", [0.5, 0.0, 0.804], [0.43, 0.43, 0.008], observations=3)
    box.bottom_seen = lid.bottom_seen = True
    est.tracks = {"e:1": box, "e:2": lid}
    est._merge_parts(np.array([-0.5, 0.0, 1.5]))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_what_lies_in_a_held_box_moves_with_it_until_it_is_poured_out():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.003, "yaw": 0.003}}
    est.joints = {"lift": 0.75, "reach": 0.3}
    grip = est.gripper_position()
    box = est.tracks["e:2"] = _track("e:2", "box", [grip[0], grip[1], 0.775], [0.54, 0.54, 0.05])
    box.annotated_affordances = ["graspable", "pourable"]
    piece = est.tracks["e:7"] = _track("e:7", "puzzle_piece", [grip[0] + 0.19, grip[1] - 0.06, 0.76],
                                       [0.115, 0.115, 0.006])
    other = est.tracks["e:8"] = _track("e:8", "puzzle_piece", [grip[0] + 0.7, grip[1], 0.753], [0.115, 0.115, 0.006])
    assert est.support_of(piece) is box
    est.mark_held("e:2")
    est.joints = {"lift": 0.95, "reach": 0.3}
    est._follow_held()
    assert piece.carried and piece.pos[2] == pytest.approx(0.96) and est.support_of(piece) is box
    assert not other.carried and other.pos[2] == pytest.approx(0.753), "only what lies in the box moves"
    est._clear_held()
    assert not piece.carried and est.support_of(piece) is box, "put down without pouring, it stays in the box"

    est.mark_held("e:2")
    est.joints = {"lift": 0.95, "reach": 0.3, "wrist_roll": est.config.poured_roll + 0.1}
    est._follow_held()
    assert "e:7" not in est.tracks and "e:8" in est.tracks
    assert est.dropped[-1]["id"] == "e:7" and est.dropped[-1]["reason"].startswith("poured out of e:2")


def test_a_face_the_camera_cannot_see_is_explained_from_the_camera_geometry():
    from steward.ego.runtime import view_hint

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.003, "yaw": 0.003}}
    cam = est.camera_pose()
    intrinsics = est.camera_intrinsics()
    lowest = math.radians(30.0 + intrinsics["fovy"] / 2)  # the head camera looks 30° down

    def hint(dx, dy):
        return view_hint(cam, intrinsics, np.array([cam["p"][0] + dx, cam["p"][1] + dy, 0.753]), 0.756, 1.5)

    near = hint(0.2, 0.0)
    assert near.startswith("too close") and f"{(cam['p'][2] - 0.756) / math.tan(lowest):.2f} m ahead" in near
    assert "to the left" in hint(0.5, 1.0) and hint(-0.5, 0.0).startswith("it is behind")
    assert hint(0.8, 0.0).startswith("in view but hidden")
    assert "farther than 1.50 m" in hint(1.8, 0.0)


def test_a_settled_heading_turns_only_after_several_views_in_a_row_show_the_turn():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.stamp = 1.0
    piece = est.tracks["e:5"] = _track("e:5", "puzzle_piece", [0.6, 0.0, 0.753], [0.115, 0.115, 0.006], yaw=math.pi)
    det = {"category": "puzzle_piece", "size": [0.115, 0.115, 0.006]}
    turned = -math.pi / 2

    def view(yaw):
        est._fuse(piece, det, piece.pos.copy(), yaw, 0.02)

    for yaw in (turned, math.pi, turned, turned, math.pi):
        view(yaw)
        assert abs(wrap_angle(piece.yaw - math.pi)) < 0.05, "a neighbour's view now and then leaves the heading"
    for _ in range(est.config.yaw_confirm_views):
        view(turned)
    assert abs(wrap_angle(piece.yaw - turned)) < 0.05, "a turn seen in every view is followed"

    piece.yaw_var = 0.3**2  # just put down by the robot: its heading follows the first view
    view(0.0)
    assert abs(piece.yaw) < 0.1


def test_a_detection_of_a_neighbour_does_not_take_the_track_of_the_piece_beside_it():
    from steward.math3d import pose_compose, pose_inverse

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:1"] = _track("e:1", "table", [0.6, 0.0, 0.375], [1.0, 0.6, 0.75])
    est.tracks["e:50"] = _track("e:50", "puzzle_piece", [0.627, 0.0, 0.76], [0.115, 0.115, 0.006])
    odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.003, "yaw": 0.003}}
    est.odometry = odometry
    cam = est.camera_pose()

    def det(x: float, sigma_range: float) -> dict:
        local = pose_compose(pose_inverse(cam), {"p": [x, 0.0, 0.76], "q": [1.0, 0.0, 0.0, 0.0]})
        return {"category": "puzzle_piece", "pose": local, "size": [0.115, 0.115, 0.006], "range": sigma_range}

    # The neighbour's detection is the more certain one and is fused first, yet the track stays with
    # the piece it lies on.
    est.update({"stamp": 1.0, "odometry": odometry, "joints": {}, "gripper": {"state": "open", "holding": False},
                "detections": [det(0.5, 0.2), det(0.627, 0.9)]})
    tracks = sorted((t for t in est.tracks.values() if t.category == "puzzle_piece"), key=lambda t: t.pos[0])
    assert len(tracks) == 2 and tracks[1].id == "e:50"
    assert tracks[0].pos[0] == pytest.approx(0.5, abs=0.01) and tracks[1].pos[0] == pytest.approx(0.627, abs=0.01)


def test_top_face_reads_a_face_in_the_object_frame_and_hides_what_is_occluded():
    h, w, fovy = 120, 160, 60.0
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    cam = {"p": [0.0, 0.0, 1.0], "q": [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]}  # looking straight down
    depth = np.full((h, w), 1.0, np.float32)  # a face at the floor, 1 m below
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:, : w // 2] = [255, 0, 0]  # camera left half red: for this camera image left is world +y
    frame = {"depth": depth, "rgb": rgb, "fovy": fovy}
    rows = top_face(frame, cam, np.array([0.0, 0.0, -0.003]), np.array([0.2, 0.2, 0.006]), 0.0, 0.0, 8)
    assert rows[0][4] == [255, 0, 0] and rows[7][4] == [0, 0, 0], "row 0 is the object's +y edge"
    depth[:, :] = 0.5  # something nearer covers the face
    assert all(cell is None for row in top_face(frame, cam, np.array([0.0, 0.0, -0.003]), np.array([0.2, 0.2, 0.006]),
                                                  0.0, 0.0, 8) for cell in row)
    assert f > 0


@pytest.mark.parametrize(("scenario", "side", "lid_at", "pour_at", "min_out"), [
    ("puzzle-cat", 3, [0.34, 0.0], [0.1, 0.0], 7),
    ("puzzle-dog", 4, [0.40, 0.05], [0.0, 0.0], 12),
])
def test_the_robot_takes_off_the_lid_and_pours_the_pieces_out_onto_the_table(kitchen_root, tmp_path, scenario, side,
                                                                              lid_at, pour_at, min_out):
    demo_dir = os.environ.get("STEWARD_LOCALIZATION_LIVE_DEMO")
    demo = pathlib.Path(demo_dir) if demo_dir else None
    host = StewardHost(kitchen_root, scenario=scenario, speed=1.0 if demo else math.inf,
                       seed=3, run_dir=pathlib.Path(tmp_path)).start()
    assert host.phase == "phase0", "the scenario declares its level"
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    api = ego_api(runtime).start()
    done, out = threading.Event(), {}

    def track_of(world_id, min_size):
        p = np.array(host.sim._entity_pose(world_id)["p"])
        objs = [o for o in call("GET", f"{api.url}/ego/summary")["objects"] if max(o["sizeM"][:2]) >= min_size]
        return min(objs, key=lambda o: np.linalg.norm(np.array(o["position"]) - p))

    def run(skill, params):
        receipt = call("POST", f"{api.url}/ego/predict", {"skill": skill, "params": params})
        objects = {o["id"]: (o["category"], o["position"], o["sizeM"]) for o in call("GET", f"{api.url}/ego/summary")["objects"]}
        assert receipt["withinRiskLimit"], (skill, receipt["issues"], objects)
        job = call("POST", f"{api.url}/skill/run", {"receipt": receipt["id"]})
        return call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)["state"]

    def flow():
        try:
            runtime.start()
            _explore(api.url, 120)
            table, lid = track_of("w:table-1", 0.5), track_of("w:lid", 0.15)
            _approach_to_observe(api.url, lid, 1.0)
            lid = track_of("w:lid", 0.15)
            assert call("POST", f"{api.url}/ego/inspect", {"entity": lid["id"], "px": 16})["coverage"] > 0.8
            call("POST", f"{api.url}/ego/annotate", {"entity": lid["id"], "affordances": ["graspable", "flat"]})
            out["lid"] = [run("pick", {"target": lid["id"]}), run("place", {"on": table["id"], "offset": lid_at})]
            box = track_of("w:box", 0.15)
            call("POST", f"{api.url}/ego/annotate", {"entity": box["id"], "label": "box",
                                                       "affordances": ["graspable", "container", "receptacle", "pourable"]})
            out["box"] = [run("pick", {"target": box["id"]}), run("pour", {"on": table["id"], "offset": pour_at}),
                          run("place", {"at": [-0.6, -0.6]})]
        except Exception:  # noqa: BLE001 - reported by the assertions below
            out["error"] = traceback.format_exc()
        finally:
            done.set()

    try:
        if demo is not None:
            sim = host.sim
            episode = sim.episode
            sim.paused = True
            host.attach_ego_api("w:robot-1", api.url)
            url = f"{host.wca.url}/console?view=localization"
            (demo / "console.json").write_text(json.dumps({"url": url}) + "\n", encoding="utf-8")
            print(url, flush=True)
            while sim.paused and not (demo / "stop").exists() and host.sim is sim and sim.episode == episode:
                host.pump(0.05)
            if (demo / "stop").exists() or host.sim is not sim or sim.episode != episode:
                return
        threading.Thread(target=flow, daemon=True).start()
        deadline = time.time() + 300
        while not done.is_set() and time.time() < deadline:
            if demo is not None and ((demo / "stop").exists() or host.sim is not sim or sim.episode != episode):
                return
            host.pump(0.05)
        if "error" in out:
            print(out["error"])
        assert "error" not in out, out.get("error", "")[-300:]
        assert out["lid"] == ["succeeded", "succeeded"] and out["box"] == ["succeeded"] * 3, out
        pieces = [f"w:piece-{r}{c}" for r in range(side) for c in range(side)]
        fallen = [w for w in pieces if host.sim._entity_pose(w)["p"][2] < 0.7]
        assert not fallen, f"no piece slides off the table: {fallen}"
        box_xy = np.array(host.sim._entity_pose("w:box")["p"][:2])
        beyond_box = puzzle_box_m(side) / 2 + 0.04
        out_of_box = [w for w in pieces
                      if np.linalg.norm(np.array(host.sim._entity_pose(w)["p"][:2]) - box_xy) > beyond_box
                      and host.sim._entity_pose(w)["p"][2] > 0.7]
        assert len(out_of_box) >= min_out, f"most pieces lie on the table outside the box: {out_of_box}"
        upright = sum(quat_rotate(host.sim._entity_pose(w)["q"], [0, 0, 1])[2] > 0.9 for w in out_of_box)
        assert upright + sum(quat_rotate(host.sim._entity_pose(w)["q"], [0, 0, 1])[2] < -0.9 for w in out_of_box) \
            == len(out_of_box), "every poured piece lies flat, face up or down"
    finally:
        if demo is not None and host.sim is sim and sim.episode == episode and not (demo / "stop").exists():
            sim.paused = True
            status = "failed" if sys.exc_info()[0] is not None else "finished"
            (demo / "result.json").write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")
            while not (demo / "stop").exists() and host.sim is sim and sim.episode == episode:
                host.pump(0.05)
        api.stop()
        runtime.stop()
        host.stop()


def test_pieces_dropped_behind_and_under_the_table_are_picked_off_the_floor_and_put_back(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="puzzle-dog", speed=math.inf, seed=3, run_dir=pathlib.Path(tmp_path)).start()
    drops = {"w:piece-13": (-0.064, 0.024), "w:piece-02": (-0.013, -0.093)}  # the far table edge is at y -0.025
    for wid, (x, y) in drops.items():
        host.sim.edit_move(wid, {"at": {"x": x, "y": y}, "yaw": 0.3})
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    api = ego_api(runtime).start()
    done, out = threading.Event(), {}

    def run(skill, params):
        receipt = call("POST", f"{api.url}/ego/predict", {"skill": skill, "params": params})
        assert receipt["withinRiskLimit"], (skill, receipt["issues"])
        job = call("POST", f"{api.url}/skill/run", {"receipt": receipt["id"]})
        return call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)["state"]

    def flow():
        try:
            runtime.start()
            _explore(api.url, 120)
            assert run("navigate", {"x": 0.0, "y": 0.9, "yaw": -math.pi / 2}) == "succeeded"
            t0 = runtime.estimator.stamp
            while runtime.estimator.stamp < t0 + 4.0:
                time.sleep(0.2)
            objects = call("GET", f"{api.url}/ego/summary")["objects"]
            table = next(o for o in objects if o["category"] == "table")
            def on_floor_near(xy):
                small = [o for o in call("GET", f"{api.url}/ego/summary")["objects"]
                         if max(o["sizeM"][:2]) < 0.3 and o["position"][2] < 0.1]
                return min(small, key=lambda o: math.dist(o["position"][:2], xy))

            for wid, xy in drops.items():
                # Docked beside it the camera no longer sees the floor there: look from where it is in view.
                _approach_to_observe(api.url, on_floor_near(xy), 1.3)
                t1 = runtime.estimator.stamp
                while runtime.estimator.stamp < t1 + 3.0:
                    time.sleep(0.2)
                piece = on_floor_near(xy)
                call("POST", f"{api.url}/ego/annotate", {"entity": piece["id"], "affordances": ["graspable", "flat"]})
                out[wid] = [run("pick", {"target": piece["id"]}), run("place", {"on": table["id"]})]
        except Exception:  # noqa: BLE001 - reported by the assertions below
            out["error"] = traceback.format_exc()
        finally:
            done.set()

    threading.Thread(target=flow, daemon=True).start()
    try:
        deadline = time.time() + 240
        while not done.is_set() and time.time() < deadline:
            host.pump(0.05)
        assert "error" not in out, out.get("error", "")[-400:]
        assert all(out[wid] == ["succeeded", "succeeded"] for wid in drops), out
        assert all(host.sim._entity_pose(wid)["p"][2] > 0.7 for wid in drops), "both pieces lie on the table again"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def test_textured_boxes_compile_from_in_memory_assets(kitchen_root):
    build_scene = SceneState.from_dict(_scenario(kitchen_root)["sceneState"])
    from steward.scene.mjcf import build_mjcf
    build = build_mjcf(build_scene)
    assert any(name.startswith("tex-") for name in build.assets)
    model = build.compile()
    assert model.ntex >= 10 and model.nmat >= 10
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "m0") >= 0


def test_a_labelled_object_still_takes_the_detections_of_the_perceived_category():
    from steward.math3d import pose_compose, pose_inverse

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:50"] = _track("e:50", "lid", [0.8, 0.0, 0.9], [0.44, 0.44, 0.008])
    est.annotate("e:50", "puzzle_box_lid", ["graspable", "flat"])
    odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.003, "yaw": 0.003}}
    est.odometry = odometry
    local = pose_compose(pose_inverse(est.camera_pose()), {"p": [0.81, 0.0, 0.9], "q": [1.0, 0.0, 0.0, 0.0]})
    est.update({"stamp": 1.0, "odometry": odometry, "joints": {}, "gripper": {"state": "open", "holding": False},
                "detections": [{"category": "lid", "pose": local, "size": [0.44, 0.44, 0.008], "range": 0.8}]})
    assert list(est.tracks) == ["e:50"] and est.tracks["e:50"].category == "puzzle_box_lid"


def test_a_square_object_keeps_its_frame_when_a_view_fits_its_box_a_quarter_turn_round():
    from steward.ego.estimator import _continuous_heading

    square = np.array([0.115, 0.112, 0.006])
    yaw, size = _continuous_heading(math.radians(-80.0), square, math.radians(12.0))
    assert math.degrees(yaw) == pytest.approx(10.0) and list(size) == [0.112, 0.115, 0.006]
    oblong = np.array([0.3, 0.15, 0.1])
    yaw, size = _continuous_heading(math.radians(-60.0), oblong, math.radians(12.0))
    assert math.degrees(yaw) == pytest.approx(-60.0) and list(size) == list(oblong)
    yaw, _ = _continuous_heading(math.radians(170.0), oblong, math.radians(-5.0))
    assert math.degrees(yaw) == pytest.approx(-10.0)


def test_world_rows_turn_a_face_by_its_heading_so_row_0_faces_the_map_plus_y():
    from steward.ego.perception import world_up_face

    # A mark on the middle of the object's +x edge (right column).
    rows = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    assert world_up_face(rows, math.radians(5.0)) == (rows, 0)
    turned, turn = world_up_face(rows, math.radians(92.0))
    assert turn == 90 and turned == [[0, 1, 0], [0, 0, 0], [0, 0, 0]], "local +x points to the map's +y"
    turned, turn = world_up_face(rows, math.radians(-88.0))
    assert turn == 270 and turned == [[0, 0, 0], [0, 0, 0], [0, 1, 0]], "local +x points to the map's -y"
    turned, turn = world_up_face(rows, math.radians(179.0))
    assert turn == 180 and turned == [[0, 0, 0], [1, 0, 0], [0, 0, 0]]


def test_an_occupied_requested_spot_moves_to_the_free_spot_nearest_it():
    from steward.ego.esim import Esim, entity_box

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks = {"e:4": _track("e:4", "table", [0.5, 0.0, 0.375], [1.0, 1.0, 0.75]),
                  "e:1": _track("e:1", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006]),
                  "e:2": _track("e:2", "puzzle_piece", [0.2, -0.3, 0.753], [0.115, 0.115, 0.006])}
    es = est.to_scene()
    table, piece = es.find("e:4"), es.find("e:2")
    esim = Esim(model)

    def spot(offset):
        return esim._choose_spot(es, table, entity_box(table)[0], piece, entity_box(piece)[0], offset, 0.0, 0.015)

    free, issues = spot([0.25, 0.25])
    assert free == pytest.approx((0.75, 0.25)) and issues == []
    moved, issues = spot([0.0, 0.0])
    shift = math.dist(moved, (0.5, 0.0))
    assert 0.115 < shift < 0.16 and "occupied by e:1" in issues[0], "it lies edge to edge beside the occupant"


def test_a_face_is_found_on_its_reference_with_the_quarter_turns_that_line_it_up():
    from steward.ego.perception import match_face

    rng = np.random.default_rng(7)
    reference = rng.integers(0, 256, (24, 24, 3))
    patch = reference[4:12, 10:18]
    for k in range(4):
        # An object lying turned k quarter turns clockwise shows the patch turned back by k.
        face = np.rot90(patch, -k).tolist()
        best = match_face(face, reference.tolist())["matches"][0]
        assert (best["row"], best["col"], best["turns"]) == (8.0, 14.0, k) and best["score"] > 0.99
    plain = match_face(np.full((8, 8, 3), 200).tolist(), reference.tolist())
    assert plain["spread"] == 0.0


def test_the_lid_shows_the_picture_as_large_as_the_finished_puzzle_on_a_plain_border(kitchen_root):
    from steward.authoring.service import PUZZLE_LID_BORDER_RGB, PUZZLES

    scenario = _scenario(kitchen_root)
    lid = next(e for e in scenario["sceneState"]["entities"] if e["id"] == "w:lid")
    png = base64.b64decode(lid["body"]["links"][0]["geometry"]["collision"][0]["texture"])
    print_rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), int)
    border = np.all(np.abs(print_rgb - PUZZLE_LID_BORDER_RGB) <= 2, axis=2)
    row = print_rgb.shape[0] // 2
    picture_share = (~border[row]).sum() / print_rgb.shape[1]
    lid_m = lid["body"]["links"][0]["geometry"]["collision"][0]["size"][0]
    assert picture_share == pytest.approx(PUZZLES["cat"].grid * PUZZLE_PIECE_M / lid_m, abs=0.02)
    assert border[0, 0] and border[-1, -1]
    assert "共 9 片" in scenario["task"]


def test_a_pour_spot_keeps_the_contents_slide_from_every_edge_of_the_surface():
    from steward.ego.esim import Esim, PredictionError, entity_box

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks = {"e:4": _track("e:4", "table", [0.5, 0.0, 0.375], [2.0, 1.0, 0.75]),
                  "e:2": _track("e:2", "box", [-0.2, -0.3, 0.775], [0.41, 0.41, 0.05])}
    es = est.to_scene()
    table, box = es.find("e:4"), es.find("e:2")
    esim = Esim(model)
    keep = 0.205 + 0.25
    spot, _ = esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], None, 0.0, 0.015, edge_keep=keep)
    assert esim._edge_distance(entity_box(table)[0], spot) >= keep - 1e-9
    with pytest.raises(PredictionError, match="of the edge"):
        esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], [0.0, -0.3], 0.0, 0.015, edge_keep=keep)


def test_a_held_box_overhanging_the_base_is_not_put_down_on_the_floor_under_it():
    from steward.ego.esim import Esim

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks = {"e:2": _track("e:2", "box", [0.35, 0.0, 0.775], [0.54, 0.54, 0.05]),
                  "e:3": _track("e:3", "puzzle_piece", [0.35, 0.0, 0.003], [0.115, 0.115, 0.006])}
    es = est.to_scene()
    box, piece = es.find("e:2"), es.find("e:3")
    esim = Esim(model)
    assert "under the robot's base" in esim._release_blocked([], box, (0.35, 0.0), 0.0, (0.0, 0.0), 0.0)
    assert esim._release_blocked([], box, (0.75, 0.0), 0.0, (0.0, 0.0), 0.0) is None
    assert esim._release_blocked([], piece, (0.0, 0.0), 0.0, (0.0, 0.0), 0.0) is None, "the base drives over it"


def test_a_held_box_overhanging_the_base_is_set_down_on_the_floor_with_the_arm_extended():
    from steward.ego.esim import Esim

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.8, "reach": 0.1}
    est.tracks = {"e:2": _track("e:2", "box", [0.35, 0.0, 0.775], [0.54, 0.54, 0.05])}
    est.held_id, est.tracks["e:2"].held = "e:2", True
    esim = Esim(model)
    receipt = esim.predict(est.to_scene(), "place", {"at": [0.35, 0.0]})
    assert receipt["feasible"], receipt["issues"]
    plan = receipt["plan"]
    shift = plan["reach"] - 0.1
    assert shift > 0 and plan["dock"] == pytest.approx([-shift, 0.0]), "it backs off as far as the arm extends"
    assert plan["predock"][0] < plan["dock"][0], "then backs away from the box it set down"
    far = esim.predict(est.to_scene(), "place", {"at": [2.0, 0.0], "yaw": 0.0})
    assert far["feasible"], far["issues"]
    assert far["plan"]["reach"] == pytest.approx(plan["reach"])
    assert far["plan"]["dock"][0] == pytest.approx(2.0 - (0.35 + shift), abs=1e-6)


def test_a_heading_the_wrist_cannot_turn_to_from_the_nearest_side_is_placed_from_another_side():
    from steward.ego.esim import Esim

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.odometry = {"x": 0.0, "y": 1.0, "yaw": -math.pi / 2, "sigma": {"pos": 0.0, "yaw": 0.0}}
    est.joints = {"lift": 0.85, "reach": 0.1, "wrist_yaw": 0.0}
    est.tracks = {"e:4": _track("e:4", "table", [0.0, 0.0, 0.375], [0.6, 0.6, 0.75]),
                  "e:3": _track("e:3", "puzzle_piece", [0.0, 0.6, 0.85], [0.115, 0.115, 0.006], yaw=-math.pi / 2)}
    est.held_id, est.tracks["e:3"].held = "e:3", True
    receipt = Esim(model).predict(est.to_scene(), "place", {"on": "e:4", "offset": [0.0, 0.1], "yaw": math.pi / 2})
    assert receipt["feasible"], receipt["issues"]
    assert abs(math.cos(receipt["plan"]["yaw"])) == pytest.approx(1.0, abs=1e-6), "docked at a side face, not the near one"
    assert abs(receipt["plan"]["wristYaw"]) == pytest.approx(math.pi / 2, abs=1e-6)


def test_a_remembered_face_gathers_views_and_is_replaced_after_the_object_is_put_down():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:3"] = _track("e:3", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006])
    track = est.tracks["e:3"]
    cols, rows = est.face_grid(track, 0.005)
    assert (cols, rows) == (23, 23)
    assert est.face_grid(track, 0.002) == (58, 58), "the default 2 mm cell resolves a piece into dozens of cells"
    left, right = np.full((rows, cols, 3), np.nan), np.full((rows, cols, 3), np.nan)
    left[:, :12], right[:, 12:] = 100.0, 200.0
    est.remember_face("e:3", left, 1.0)
    est.remember_face("e:3", right, 2.0)
    assert track.appearance.coverage == 1.0 and track.appearance.stamp == 2.0, "two half views make the whole face"
    track.size = np.array([0.125, 0.12, 0.006])
    assert est.face_grid(track, 0.005) == (23, 23), "the grid stays while the extent estimate wobbles"
    est.held_id, track.held = "e:3", True
    est._clear_held()
    assert track.appearance.stale, "put down, it may lie turned over"
    est.remember_face("e:3", left, 3.0)
    assert track.appearance.coverage == pytest.approx(12 / 23) and not track.appearance.stale


def test_interior_face_stats_ignore_a_neighbour_tinting_the_edge():
    plain = [[[200, 200, 200]] * 8 for _ in range(8)]
    assert face_colour_stats(plain)["interiorSpread"] == 0.0 and face_colour_stats(plain)["plain"]
    edged = [list(row) for row in plain]
    for i in range(8):
        edged[0][i] = edged[i][0] = [40, 180, 40]  # a neighbour's green along two edges
    stats = face_colour_stats(edged)
    assert stats["spread"] > 20 and stats["interiorSpread"] == 0.0 and stats["plain"]
    rgb = np.full((8, 8, 3), 200.0)
    rgb[0, :], rgb[:, 0] = 40.0, 40.0
    track = _track("e:3", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006])
    track.appearance = Appearance(rgb, (0.115, 0.115), 1.0)
    assert track.appearance.spread > 20 and track.appearance.interior_spread == 0.0
    assert track.appearance.plain and track.appearance.interior_colors == 1


def test_a_lit_cardboard_back_is_plain_while_a_printed_face_is_not():
    """Lighting gradients inflate RGB std on a one-colour back; colour-bucket count still tells them apart."""
    back = []
    for i in range(12):
        shade = 140 + i * 8  # bright-to-dark gradient like a table lamp
        back.append([[shade, int(shade * 0.85), int(shade * 0.62)]] * 12)
    back_stats = face_colour_stats(back)
    assert back_stats["interiorSpread"] > 20 and back_stats["plain"] and back_stats["interiorColors"] <= 12
    rng = np.random.default_rng(0)
    print_face = rng.integers(0, 256, (12, 12, 3)).tolist()
    print_stats = face_colour_stats(print_face)
    assert not print_stats["plain"] and print_stats["interiorColors"] > 20


def test_a_fine_face_averages_into_coarser_cells_over_the_same_extent():
    from steward.ego.perception import resample_face

    fine = np.zeros((6, 6, 3))
    fine[:3] = 90.0
    fine[0, 0] = np.nan
    coarse = resample_face(fine, 2, 3)
    assert coarse.shape == (3, 2, 3)
    assert coarse[0, 0, 0] == 90.0 and coarse[1, 1, 0] == pytest.approx(45.0) and coarse[2, 0, 0] == 0.0


def test_a_zero_clearance_offset_is_used_exactly_even_where_another_object_lies():
    from steward.ego.esim import Esim, entity_box

    model = steward_bot().to_dict()
    est = EgoEstimator(robot_model=model)
    est.tracks = {"e:4": _track("e:4", "table", [0.5, 0.0, 0.375], [1.0, 1.0, 0.75]),
                  "e:1": _track("e:1", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006]),
                  "e:2": _track("e:2", "puzzle_piece", [0.2, -0.3, 0.753], [0.115, 0.115, 0.006])}
    es = est.to_scene()
    table, piece = es.find("e:4"), es.find("e:2")
    esim = Esim(model)
    spot, issues = esim._choose_spot(es, table, entity_box(table)[0], piece, entity_box(piece)[0], [0.0, 0.0], 0.0, 0.0)
    assert spot == (0.5, 0.0) and issues == [], "interlocking pieces are placed where asked"


def test_two_pieces_edge_to_edge_are_not_merged_into_one_track():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {"e:1": _track("e:1", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006]),
                  "e:2": _track("e:2", "puzzle_piece", [0.615, 0.0, 0.753], [0.115, 0.115, 0.006])}
    est._merge_duplicates()
    assert set(est.tracks) == {"e:1", "e:2"}, "edge to edge they are two pieces"
    est.tracks["e:2"].pos = np.array([0.53, 0.0, 0.753])  # centres overlap: the same piece seen twice
    est._merge_duplicates()
    assert len(est.tracks) == 1
