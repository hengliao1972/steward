"""Picture-puzzle scenarios and the generic capabilities they rely on: printed faces, pourable
containers, the wrist, close inspection, annotations and arrangement goals."""

from __future__ import annotations

import base64
import io
import math
import pathlib
import threading
import time
import traceback

import mujoco
import numpy as np
import pytest
from PIL import Image

from steward.assets.generators import steward_bot
from steward.authoring.service import PUZZLE_PIECE_M, AuthoringService
from steward.ego.estimator import EgoEstimator, Track
from steward.ego.perception import top_face
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.math3d import quat_rotate
from steward.mission import _approach_to_observe, _explore
from steward.net import call
from steward.scene.schema import SceneState
from steward.world.goals import evaluate
from steward.world.server import StewardHost
from steward.world.sim import WorldSim


def _scenario(kitchen_root, name="puzzle-cat") -> dict:
    return AuthoringService(kitchen_root).load_scenario(name)


def test_both_puzzles_are_scenarios_at_phase0_with_an_arrangement_goal(kitchen_root):
    for name in ("puzzle-cat", "puzzle-greatwall"):
        scenario = _scenario(kitchen_root, name)
        assert scenario["phase"] == "phase0" and scenario["goal"]["kind"] == "relative-poses"
        ids = {e["id"] for e in scenario["sceneState"]["entities"]}
        assert {"w:box", "w:lid", "w:table-1"} <= ids and sum(i.startswith("w:piece-") for i in ids) == 9


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


def test_a_detection_of_a_neighbour_does_not_take_the_track_of_the_piece_beside_it():
    from steward.math3d import pose_compose, pose_inverse

    est = EgoEstimator(robot_model=steward_bot().to_dict())
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
    tracks = sorted(est.tracks.values(), key=lambda t: t.pos[0])
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


def test_the_robot_takes_off_the_lid_and_pours_the_pieces_out_onto_the_table(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="puzzle-cat", speed=math.inf, seed=3, run_dir=pathlib.Path(tmp_path)).start()
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
            table, lid = track_of("w:table-1", 1.0), track_of("w:lid", 0.3)
            _approach_to_observe(api.url, lid, 1.0)
            lid = track_of("w:lid", 0.3)
            assert call("POST", f"{api.url}/ego/inspect", {"entity": lid["id"], "px": 16})["coverage"] > 0.8
            call("POST", f"{api.url}/ego/annotate", {"entity": lid["id"], "affordances": ["graspable", "flat"]})
            out["lid"] = [run("pick", {"target": lid["id"]}), run("place", {"on": table["id"], "offset": [0.72, 0.22]})]
            box = track_of("w:box", 0.3)
            call("POST", f"{api.url}/ego/annotate", {"entity": box["id"], "label": "box",
                                                       "affordances": ["graspable", "container", "receptacle", "pourable"]})
            out["box"] = [run("pick", {"target": box["id"]}), run("pour", {"on": table["id"], "offset": [0.2, 0.0]}),
                          run("place", {"on": table["id"]})]
        except Exception:  # noqa: BLE001 - reported by the assertions below
            out["error"] = traceback.format_exc()
        finally:
            done.set()

    threading.Thread(target=flow, daemon=True).start()
    try:
        deadline = time.time() + 300
        while not done.is_set() and time.time() < deadline:
            host.pump(0.05)
        if "error" in out:
            print(out["error"])
        assert "error" not in out, out.get("error", "")[-300:]
        assert out["lid"] == ["succeeded", "succeeded"] and out["box"] == ["succeeded"] * 3, out
        pieces = [f"w:piece-{r}{c}" for r in range(3) for c in range(3)]
        fallen = [w for w in pieces if host.sim._entity_pose(w)["p"][2] < 0.7]
        assert not fallen, f"no piece slides off the table: {fallen}"
        box_xy = np.array(host.sim._entity_pose("w:box")["p"][:2])
        out_of_box = [w for w in (f"w:piece-{r}{c}" for r in range(3) for c in range(3))
                      if np.linalg.norm(np.array(host.sim._entity_pose(w)["p"][:2]) - box_xy) > 0.25
                      and host.sim._entity_pose(w)["p"][2] > 0.7]
        assert len(out_of_box) >= 7, f"most pieces lie on the table outside the box: {out_of_box}"
        upright = sum(quat_rotate(host.sim._entity_pose(w)["q"], [0, 0, 1])[2] > 0.9 for w in out_of_box)
        assert upright + sum(quat_rotate(host.sim._entity_pose(w)["q"], [0, 0, 1])[2] < -0.9 for w in out_of_box) \
            == len(out_of_box), "every poured piece lies flat, face up or down"
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
    from steward.authoring.service import PUZZLE_GRID, PUZZLE_LID_BORDER_RGB

    scenario = _scenario(kitchen_root)
    lid = next(e for e in scenario["sceneState"]["entities"] if e["id"] == "w:lid")
    png = base64.b64decode(lid["body"]["links"][0]["geometry"]["collision"][0]["texture"])
    print_rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), int)
    border = np.all(np.abs(print_rgb - PUZZLE_LID_BORDER_RGB) <= 2, axis=2)
    row = print_rgb.shape[0] // 2
    picture_share = (~border[row]).sum() / print_rgb.shape[1]
    lid_m = lid["body"]["links"][0]["geometry"]["collision"][0]["size"][0]
    assert picture_share == pytest.approx(PUZZLE_GRID * PUZZLE_PIECE_M / lid_m, abs=0.02)
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
    keep = 0.205 + esim.config.pour_spill
    spot, _ = esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], None, 0.0, 0.015, edge_keep=keep)
    assert esim._edge_distance(entity_box(table)[0], spot) >= keep - 1e-9
    with pytest.raises(PredictionError, match="of the edge"):
        esim._choose_spot(es, table, entity_box(table)[0], box, entity_box(box)[0], [0.0, -0.3], 0.0, 0.015, edge_keep=keep)


def test_a_remembered_face_gathers_views_and_is_replaced_after_the_object_is_put_down():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:3"] = _track("e:3", "puzzle_piece", [0.5, 0.0, 0.753], [0.115, 0.115, 0.006])
    track = est.tracks["e:3"]
    cols, rows = est.face_grid(track, 0.005)
    assert (cols, rows) == (23, 23)
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
