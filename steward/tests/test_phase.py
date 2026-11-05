"""Phase levels and the Phase 1 perception pipeline (render → segment → estimate)."""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

import mujoco

from steward.assets.generators import steward_bot
from steward.authoring.service import AuthoringService
from steward.ego.classify import classify
from steward.ego.estimator import EgoEstimator
from steward.ego.perception import box_of, decode_rgbd
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.server import ego_api
from steward.net import call
from steward.phases import PHASES, phase_modules
from steward.world.console import console_state
from steward.world.server import StewardHost
from steward.world.sim import PerceptionConfig, WorldSim


def _pump_until(host: StewardHost, done, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not done() and time.time() < deadline:
        host.pump(0.05)
    assert done(), "timed out"


def test_box_fit_recovers_the_extent_and_heading_of_a_rotated_box():
    rng = np.random.default_rng(0)
    local = rng.uniform([-0.3, -0.1, 0.0], [0.3, 0.1, 0.2], (800, 3))
    yaw = 0.6
    c, s = math.cos(yaw), math.sin(yaw)
    points = np.stack([local[:, 0] * c - local[:, 1] * s + 1.0, local[:, 0] * s + local[:, 1] * c - 2.0, local[:, 2]], 1)
    centre, size, fitted = box_of(points)
    assert centre[:2] == pytest.approx([1.0, -2.0], abs=0.01)
    assert size == pytest.approx([0.6, 0.2, 0.2], abs=0.02)
    assert abs(math.sin(fitted - yaw)) < 0.05


def test_classifier_tells_tableware_from_furniture_by_size_colour_and_footing():
    def best(size, bottom, color, invalid=0.0):
        probs = classify(np.array(size), bottom, np.array(color), invalid)
        return max(probs, key=probs.get)

    assert best([0.22, 0.22, 0.07], 0.75, [0.8, 0.15, 0.15]) == "bowl"
    assert best([0.11, 0.085, 0.095], 0.75, [0.05, 0.05, 0.06]) == "mug"
    assert best([0.08, 0.08, 0.19], 0.75, [0.8, 0.85, 0.9], invalid=0.3) == "wine_glass"
    assert best([1.2, 0.8, 0.75], 0.0, [0.8, 0.67, 0.5]) == "table"
    assert best([0.56, 0.44, 0.22], 0.55, [0.8, 0.67, 0.5]) == "drawer"
    assert best([4.5, 0.1, 1.45], 0.0, [0.9, 0.9, 0.88]) == "wall"


def test_every_phase_level_names_its_module_choices():
    assert set(PHASES) >= {"phase0", "phase1"}
    assert phase_modules("phase0")["perception"] == "ideal"
    assert phase_modules("phase1")["perception"] == "rgbd"
    with pytest.raises(ValueError):
        phase_modules("phase9")


def test_switching_the_phase_level_changes_the_sensors_and_the_robot_follows(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path).start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    try:
        assert "detections" in host.sim.sensors("w:robot-1")
        _pump_until(host, lambda: runtime.perception_mode == "detections" and len(runtime.estimator.tracks) > 3)
        assert console_state(host)["phase"]["level"] == "phase0"
        with pytest.raises(ValueError):
            host.set_phase("phase9")
        state = host.set_phase("phase1")
        assert state["level"] == "phase1" and host.sim.events[-1]["type"] == "phase/switch"
        assert host.sim.sensors("w:robot-1").get("imaging") == "rgbd"
        _pump_until(host, lambda: runtime.perception_mode == "rgbd")
        _pump_until(host, lambda: len(runtime.estimator.tracks) > 3)
        assert all(t.voxels is not None for t in runtime.estimator.tracks.values()), \
            "after the switch every object comes from the RGB-D pipeline"
        host.set_phase("phase0")
        _pump_until(host, lambda: runtime.perception_mode == "detections")
    finally:
        runtime.stop()
        host.stop()


def test_a_named_level_outranks_the_scenarios_declared_one_for_every_later_episode(kitchen_root, tmp_path):
    declared = StewardHost(kitchen_root, scenario="puzzle-dog", run_dir=tmp_path)
    assert declared.phase == "phase0", "without a named level the scenario's declared one applies"
    host = StewardHost(kitchen_root, scenario="puzzle-dog", run_dir=tmp_path, phase="phase1")
    assert host.phase == "phase1" and host.sim.perception.mode == "rgbd"
    host.set_phase("phase1-sam")
    host.load_episode("puzzle-dog")
    assert host.phase == "phase1-sam" and host.sim.perception.masks is False, "a console switch outlives the episode"
    for h in (declared, host):
        h.stop()


def test_phase1_exploration_builds_an_accurate_ego_state_from_rgbd(kitchen_root, tmp_path):
    host = StewardHost(kitchen_root, scenario="phase0-chores", speed=4.0, seed=3, run_dir=tmp_path,
                       phase="phase1").start()
    runtime = EgoRuntime(RobotLink(host.robot_servers["w:robot-1"].url), steward_bot().to_dict())
    runtime.start()
    api = ego_api(runtime).start()
    finished = threading.Event()

    def explore() -> None:
        job = call("POST", f"{api.url}/ego/explore")
        call("POST", f"{api.url}/skill/wait", {"job": job["id"], "timeoutS": 120}, timeout=130)
        time.sleep(1.5)
        finished.set()

    try:
        threading.Thread(target=explore, daemon=True).start()
        _pump_until(host, finished.is_set)
        record = host.observer.latest["w:robot-1"]
        assert record["phase"] == "phase1"
        relations = host.sim.world_state()["relations"]
        in_drawers = sum(1 for r in relations if r["type"] == "supported-by" and r["b"].get("link") == "drawer")
        assert record["matchedObjects"] >= record["gtObjects"] - in_drawers - 1, \
            f"everything outside the closed drawers is found ({in_drawers} lie in them): {record}"
        assert record["meanPositionErrorM"] < 0.03, record
        assert record["meanSizeErrorM"] < 0.04, record
        assert record["falsePositives"] <= 1, record
        assert "leak" not in record
        assert record["categoryAccuracy"] >= 0.8, record
        objects = call("GET", f"{api.url}/ego/summary")["objects"]
        assert {"bowl", "drawer"} <= {o["category"] for o in objects}
        walls = [o for o in objects if o["category"] == "wall"]
        assert len(walls) >= 3, "near walls fill the image edge to edge and still become tracks"

        def plane(wall: dict) -> tuple[str, float]:
            yaw = math.radians(wall["yawDeg"])
            if abs(math.cos(yaw)) > abs(math.sin(yaw)):
                return "y", round(wall["position"][1], 1)
            return "x", round(wall["position"][0], 1)

        # One wall may stay in two pieces where furniture in front of it hides its middle.
        assert len({plane(w) for w in walls}) >= len(walls) - 1, f"views of one wall merge into one track: {walls}"
    finally:
        api.stop()
        runtime.stop()
        host.stop()


def _frame(depth: np.ndarray, mask: np.ndarray) -> dict:
    return {"depth": depth, "mask": mask, "rgb": np.full((*depth.shape, 3), 180, np.uint8), "fovy": 58.0}


CAM = {"p": [0.0, 0.0, 3.0], "q": [1.0, 0.0, 0.0, 0.0]}


def test_a_region_whose_base_is_cut_hidden_or_overhanging_has_no_seen_bottom():
    from steward.ego.perception import segments

    h, w = 60, 80
    depth = np.full((h, w), 4.0, np.float32)
    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 5:25] = 1  # standing free: the floor right below is about as far
    depth[10:30, 5:25] = 3.0
    depth[30:, 5:25] = 2.95
    mask[5:25, 40:70] = 2  # a wall with a counter right below it, much nearer
    depth[5:25, 40:70] = 3.4
    mask[25:45, 40:70] = 3  # the counter, standing on the floor right below it
    depth[25:45, 40:70] = 2.0
    depth[45:, 40:70] = 1.95
    mask[45:60, 30:38] = 4  # reaches the lower image edge
    depth[45:60, 30:38] = 2.5
    mask[35:45, 72:78] = 5  # a table top segmented without its legs: the room below it is farther
    depth[35:45, 72:78] = 2.7
    mask[2:8, 5:15] = 6  # a bowl whose rim reaches past the table edge: half of it rests on the table
    depth[2:8, 5:15] = 3.55
    depth[8:10, 5:10] = 3.6
    seen = {round(seg.range, 1): seg.bottom_seen for seg in segments(_frame(depth, mask), CAM)}
    assert seen == {3.0: True, 3.4: False, 2.0: True, 2.5: False, 2.7: False, 3.5: True}


def test_a_partial_view_without_its_base_is_classified_by_its_visible_top():
    tabletop = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5], bottom_seen=False)
    assert max(tabletop, key=tabletop.get) == "table"
    hidden_wall = classify([3.5, 0.04, 1.03], 0.47, [0.9, 0.9, 0.9], bottom_seen=False)
    assert max(hidden_wall, key=hidden_wall.get) == "wall"
    floating = classify([1.2, 0.8, 0.30], 0.45, [0.8, 0.7, 0.5])
    assert floating.get("table", 0.0) < 0.2, "a base in view that is off the floor is not a table"


def test_achromatic_colours_are_named_by_brightness_and_hues_by_chromaticity():
    from steward.ego.estimator import color_name

    names = [color_name(np.array(rgb)) for rgb in ([0.06, 0.05, 0.06], [0.45, 0.46, 0.47], [0.86, 0.86, 0.87],
                                                    [0.7, 0.15, 0.15], [0.12, 0.25, 0.7])]
    assert names == ["black", "gray", "white", "red", "blue"]


def test_a_small_object_keeps_its_single_view_size_when_views_of_it_disagree():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    mug = Track(id="e:1", category_counts={"mug": 5.0}, pos=np.array([1.0, 0.0, 0.8]), var=np.full(3, 1e-4), yaw=0.0,
                yaw_var=1e-3, size=np.array([0.1, 0.08, 0.1]), observations=6, first_seen=0.0, last_seen=1.0)
    rng = np.random.default_rng(0)
    clouds = [rng.uniform([-0.05, -0.04, 0.75], [0.05, 0.04, 0.85], (200, 3)) + [1.0 + dx, 0.0, 0.0] for dx in (0, 0, 0, 0.12)]
    for cloud in clouds:
        mug.voxels = est._union(mug.voxels, np.unique(np.floor(cloud / 0.01).astype(np.int32), axis=0))
        mug.recent.append((cloud.mean(axis=0), np.array([0.1, 0.08, 0.1])))
    est.tracks[mug.id] = mug
    est._rebuild_geometry(mug, np.array([0.0, 0.0, 1.2]), 1.0)
    assert mug.size[0] == pytest.approx(0.1, abs=0.01), "a view 12 cm off (an object pushed aside) does not swell it"
    assert mug.pos[0] == pytest.approx(1.0, abs=0.02)


def test_an_established_piece_of_furniture_is_not_folded_into_a_larger_track_of_another_kind():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())

    def track(oid, counts, pos, size, observations):
        return Track(id=oid, category_counts=counts, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3,
                     size=np.array(size), observations=observations, first_seen=0.0, last_seen=0.0, existence=1.0)

    est.tracks["e:11"] = track("e:11", {"counter": 4.0, "unknown": 1.0}, [0.0, 0.6, 0.74], [2.6, 0.6, 1.48], 12)
    est.tracks["e:16"] = track("e:16", {"sideboard": 4.0, "counter": 3.0}, [0.0, 0.6, 0.43], [1.4, 0.47, 0.84], 20)
    est.tracks["e:20"] = track("e:20", {"unknown": 1.0}, [0.5, 0.6, 0.3], [0.35, 0.3, 0.6], 2)
    est._merge_parts(np.array([0.0, -2.0, 1.2]))
    assert "e:16" in est.tracks, "the sideboard stays its own object"
    assert "e:20" not in est.tracks, "a fragment seen twice is still folded in"


def test_tracks_the_robot_stands_in_or_wall_tall_non_walls_are_dropped():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.5, "y": -1.24, "yaw": 0.0}

    def track(oid, cat, pos, size):
        return Track(id=oid, category_counts={cat: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3,
                     size=np.array(size), observations=8, first_seen=0.0, last_seen=0.0, existence=1.0)

    for t in (track("e:398", "unknown", [-2.11, -1.83, 0.74], [1.9, 1.48, 1.48]),   # wall corner and chair, around the robot
              track("e:2126", "unknown", [-2.75, 1.56, 0.74], [2.76, 1.33, 1.48]),  # wall pieces in front of the sideboard
              track("e:6", "table", [0.6, -0.9, 0.38], [1.19, 0.8, 0.76]),
              track("e:12", "wall", [-0.1, 2.5, 0.74], [5.7, 0.07, 1.48])):
        est.tracks[t.id] = t
    est._drop_impossible(np.array([0.0, 0.0, 1.4]))
    assert sorted(est.tracks) == ["e:12", "e:6"]


def test_a_room_corner_seen_from_inside_is_concave_and_a_cabinet_front_is_not():
    from steward.ego.estimator import _concave
    from steward.ego.perception import box_of

    cam = np.array([0.0, 0.0, 1.4])
    rng = np.random.default_rng(1)
    z = rng.uniform(0.0, 1.48, 400)
    # Two walls meeting at (-3, -2.5), seen from the room centre: an L of points around the corner.
    left = np.stack([np.full(200, -3.0), rng.uniform(-2.5, -1.0, 200), z[:200]], axis=1)
    back = np.stack([rng.uniform(-3.0, -1.5, 200), np.full(200, -2.5), z[200:]], axis=1)
    corner = np.vstack([left, back])
    centre, size, _ = box_of(corner)
    assert _concave(corner, centre, float(max(size[:2])), cam)
    # The front and top of a cabinet facing the camera: the surfaces lie in front of its centre.
    front = np.stack([np.full(200, 2.4), rng.uniform(0.3, 0.9, 200), rng.uniform(0.0, 0.8, 200)], axis=1)
    top = np.stack([rng.uniform(2.4, 2.95, 200), rng.uniform(0.3, 0.9, 200), np.full(200, 0.8)], axis=1)
    cabinet = np.vstack([front, top])
    centre, size, _ = box_of(cabinet)
    assert not _concave(cabinet, centre, float(max(size[:2])), cam)

    # A table seen from its long side: the top and the legs spread around the centre, not behind it.
    top_t = np.stack([rng.uniform(0.0, 1.2, 300), rng.uniform(-1.3, -0.5, 300), np.full(300, 0.75)], axis=1)
    legs = np.vstack([np.stack([np.full(40, x), np.full(40, y), rng.uniform(0.0, 0.73, 40)], axis=1)
                      for x in (0.03, 1.17) for y in (-1.27, -0.53)])
    table = np.vstack([top_t, legs])
    centre, size, _ = box_of(table)
    assert not _concave(table, centre, float(max(size[:2])), cam)


def test_what_the_robot_holds_keeps_its_identity_while_carried():
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    mug = Track(id="e:8", category_counts={"mug": 6.0, "cup": 2.0}, pos=np.array([-1.4, 0.3, 0.9]), var=np.full(3, 1e-4),
                yaw=0.0, yaw_var=1e-3, size=np.array([0.1, 0.08, 0.1]), observations=8, first_seen=0.0, last_seen=0.0,
                existence=1.0, held=True)
    est.tracks[mug.id] = mug
    table_view = Segment(points=np.zeros((40, 3)), color=np.array([0.87, 0.76, 0.57]), pixels=900, invalid_frac=0.0,
                         border=True, range=1.0)
    est._fuse_cloud(mug, {"seg": table_view, "centre": np.array([-1.4, 0.3, 0.6]), "size": np.array([1.2, 0.8, 0.3]),
                          "yaw": 0.0, "probs": {"table": 1.0}, "footprint": 1.2}, np.array([0.0, 0.0, 1.4]))
    assert mug.category == "mug" and "table" not in mug.category_counts


def _remembered(oid: str, category: str, pos: list[float], size: list[float]):
    from steward.ego.estimator import Track

    return Track(id=oid, category_counts={category: 5.0}, pos=np.array(pos), var=np.full(3, 1e-4), yaw=0.0,
                 yaw_var=1e-3, size=np.array(size), observations=20, first_seen=0.0, last_seen=0.0, existence=1.0)


LOOKING_ALONG_X = {"p": [0.0, 0.0, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}


def _views(est, cam: dict, frame: dict, seconds: float, dt: float = 0.1) -> None:
    for _ in range(int(round(seconds / dt))):
        est.stamp += dt
        est._decay_unseen_rgbd(cam, set(), frame)


def test_an_established_object_is_dropped_only_after_free_views_from_two_places():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:5"] = _remembered("e:5", "plant", [2.0, 0.0, 0.2], [0.24, 0.2, 0.4])
    frame = {"depth": np.full((240, 320), 4.0, np.float32), "fovy": 60.0}  # the far wall, nothing in front of it
    _views(est, LOOKING_ALONG_X, frame, 10.0)
    assert "e:5" in est.tracks, "however long it looks from one place, one viewpoint does not delete a known object"
    assert est.tracks["e:5"].existence >= 0.5
    _views(est, {"p": [0.0, 0.5, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 0.6)
    assert "e:5" not in est.tracks
    assert est.dropped[-1]["id"] == "e:5" and est.dropped[-1]["reason"].startswith("seen through")


def test_furniture_whose_top_is_still_in_view_gathers_no_evidence_of_absence():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:7"] = _remembered("e:7", "table", [2.0, 0.0, 0.375], [1.2, 0.8, 0.75])
    depth = np.full((240, 320), 4.0, np.float32)  # the floor and wall between the legs
    depth[140:195, :] = 2.0  # the tabletop, at the table's distance
    frame = {"depth": depth, "fovy": 60.0}
    for cam_y in (0.0, 0.5, -0.5):
        _views(est, {"p": [0.0, cam_y, 1.2], "q": [1.0, 0.0, 0.0, 0.0]}, frame, 10.0)
    table = est.tracks["e:7"]
    assert table.existence == 1.0 and not table.absent_views, "regions that went elsewhere are no evidence it is gone"


def test_a_young_fragment_where_a_surface_is_seen_fades_away():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:9"] = _remembered("e:9", "unknown", [2.0, 0.0, 0.2], [0.24, 0.2, 0.4])
    est.tracks["e:9"].observations = 2
    frame = {"depth": np.full((240, 320), 2.0, np.float32), "fovy": 60.0}
    _views(est, LOOKING_ALONG_X, frame, 12.0)
    assert "e:9" not in est.tracks and est.dropped[-1]["reason"].startswith("faded")


def test_a_potted_plant_is_recognised_from_its_size_and_green_foliage():
    measured = classify(np.array([0.22, 0.19, 0.40]), 0.01, np.array([0.305, 0.494, 0.231]))
    assert max(measured, key=measured.get) == "plant", measured
    big = classify(np.array([0.5, 0.45, 0.9]), 0.0, np.array([0.30, 0.50, 0.22]))
    assert max(big, key=big.get) == "plant", f"a larger plant is still a plant: {big}"


def test_a_well_known_object_folded_into_a_newer_larger_track_keeps_its_id():
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks["e:3"] = _remembered("e:3", "table", [0.6, -0.9, 0.375], [1.2, 0.8, 0.75])
    est.tracks["e:3"].observations = 40
    est.tracks["e:90"] = _remembered("e:90", "table", [0.65, -0.9, 0.375], [1.4, 0.9, 0.75])
    est.tracks["e:90"].observations = 3
    est._merge_parts(np.array([-1.5, -1.2, 1.2]))
    assert list(est.tracks) == ["e:3"] and est.tracks["e:3"].observations == 43
    record = est.dropped[-1]
    assert {k: record[k] for k in ("id", "observations", "reason", "into")} == \
        {"id": "e:90", "observations": 3, "reason": "merged: part of a larger object", "into": "e:3"}


def test_a_well_known_table_that_looks_hollow_from_afar_is_kept_while_an_unknown_box_is_not():
    from steward.ego.estimator import EgoEstimator, _concave, _points

    rng = np.random.default_rng(2)
    # Only the far half of the top and the far legs: seen from (-1.4, -1.2) this reads as surfaces
    # behind the box centre, as a room corner does.
    top = np.stack([rng.uniform(0.6, 1.2, 300), rng.uniform(-0.9, -0.5, 300), np.full(300, 0.74)], axis=1)
    legs = np.vstack([np.stack([np.full(40, 1.17), np.full(40, y), rng.uniform(0.0, 0.73, 40)], axis=1)
                      for y in (-1.27, -0.53)])
    voxels = np.unique(np.floor(np.vstack([top, legs]) / 0.01).astype(np.int32), axis=0)
    cam = np.array([-1.4, -1.2, 1.5])
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": -1.44, "y": -1.18, "yaw": 0.0}
    table = _remembered("e:29", "table", [0.6, -0.9, 0.385], [1.19, 0.79, 0.74])
    table.observations, table.voxels = 470, voxels
    assert _concave(_points(voxels), table.pos, table.footprint, cam), "the case: it looks hollow from here"
    unknown = _remembered("e:30", "unknown", [0.6, -0.9, 0.385], [1.19, 0.79, 0.74])
    unknown.observations, unknown.voxels = 2, voxels.copy()
    est.tracks = {"e:29": table, "e:30": unknown}
    est._drop_impossible(cam)
    assert list(est.tracks) == ["e:29"]


def test_phase1_keeps_the_far_edge_of_a_counter(kitchen_root):
    """The far edge of a counter top is a few percent of its pixels. A random sample of the cloud
    publishes about half the depth; the box of every valid pixel keeps it."""
    scene = AuthoringService(kitchen_root).load_scenario("phase0-chores")["sceneState"]
    sim = WorldSim(scene, seed=1, perception=PerceptionConfig(mode="rgbd", masks=True))
    names = sim.build.names["w:robot-1"]
    for jname, val in zip(names.root_joints, (0.6, -2.2, 1.57), strict=True):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        sim.data.qpos[sim.model.jnt_qposadr[jid]] = val
    sim.command("w:robot-1", {"base": [0.6, -2.2, 1.57]})
    mujoco.mj_forward(sim.model, sim.data)
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update_rgbd(decode_rgbd(sim.rgbd("w:robot-1")))
    gt = sim.entity_obb("w:counter-1")
    counter = min((t for t in est.tracks.values() if t.category == "counter"),
                  key=lambda t: float(np.linalg.norm(t.pos - gt.center)))
    assert float(np.linalg.norm(counter.pos - gt.center)) < 0.12
    assert float(np.mean(np.abs(np.sort(counter.size) - np.sort(gt.size)))) < 0.05


def _paint(depth, mask, cam, fovy, points, instance: int) -> None:
    """Write ``points`` (map frame) into a depth image through ``cam``."""
    from steward.math3d import pose_compose, pose_inverse

    h, w = depth.shape
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    inv = pose_inverse(cam)
    for p in points:
        x, y, z = pose_compose(inv, {"p": p.tolist(), "q": [1.0, 0.0, 0.0, 0.0]})["p"]
        if x < 0.3:
            continue
        u = int(round((w - 1) / 2 - y * f / x))
        v = int(round((h - 1) / 2 - z * f / x))
        if 0 <= u < w and 0 <= v < h:
            depth[v, u] = x
            mask[v, u] = instance


def test_two_raised_objects_in_one_segmenter_mask_become_two_tracks():
    """Two raised objects painted as one region are wider than a bowl, so that mask is dropped.
    With segmenter labels the gap splits them, and each keeps its own size. The second object is
    wide enough that a cabinet lip, which is thinner than a bowl, is not what this test keeps."""
    from steward.ego.perception import segments
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 90, 120, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(0)

    def blob(local_centre, size, n=90):
        centre = origin + rot @ np.asarray(local_centre, float)
        return rng.uniform(centre - np.asarray(size) / 2, centre + np.asarray(size) / 2, (n, 3))

    _paint(depth, mask, cam, fovy, blob([1.15, 0.16, -0.05], [0.14, 0.14, 0.05]), 1)
    _paint(depth, mask, cam, fovy, blob([1.15, -0.16, -0.02], [0.14, 0.12, 0.08]), 1)
    bare = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 160, np.uint8), "fovy": fovy,
            "header": header, "camera": "head_camera"}
    assert len(segments(bare, cam)) == 1, "an anonymous mask stays one region"
    named = dict(bare)
    named["labels"] = {1: {"bowl": 0.5}}
    assert len(segments(named, cam)) == 2
    before, after = EgoEstimator(robot_model=steward_bot().to_dict()), EgoEstimator(robot_model=steward_bot().to_dict())
    before.update_rgbd(bare)
    after.update_rgbd(named)
    assert before.tracks == {}
    assert any(d["reason"] == "bowl longer than the bowl prior allows" for d in before.dropped)
    small = [t for t in after.tracks.values() if t.footprint < 0.25]
    assert len(after.tracks) == 2 and len(small) == 2
    assert not any(t.category == "drawer" for t in after.tracks.values())


def test_two_objects_on_a_wide_tabletop_mask_become_two_tracks():
    """A bowl and a glass painted onto one table-top region stay one region without labels. With
    labels, the two raised clusters become two tracks and the slab does not become a third."""
    from steward.ego.perception import segments

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    h, w, fovy = 120, 160, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(2)
    slab = np.column_stack([rng.uniform(1.25, 1.35, 600), rng.uniform(-0.42, 0.42, 600),
                            rng.uniform(0.74, 0.76, 600)])
    bowl = np.column_stack([rng.uniform(1.22, 1.32, 90), rng.uniform(-0.36, -0.22, 90),
                            rng.uniform(0.80, 0.86, 90)])
    glass = np.column_stack([rng.uniform(1.22, 1.30, 90), rng.uniform(0.24, 0.34, 90),
                             rng.uniform(0.80, 0.96, 90)])
    _paint(depth, mask, cam, fovy, np.vstack([slab, bowl, glass]), 1)
    bare = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 150, np.uint8), "fovy": fovy,
            "header": header, "camera": "head_camera"}
    assert len(segments(bare, cam)) == 1
    named = dict(bare)
    named["labels"] = {1: {"bowl": 0.4}}
    assert len(segments(named, cam)) == 2


def test_a_bowl_measured_with_the_table_shrinks_to_the_bowl_and_is_not_completed_as_a_drawer():
    """Table points around a bowl make a wide shallow cloud. The track keeps the bowl, so the
    drawer prior does not stretch it to a drawer's height."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(5)
    slab = np.column_stack([rng.uniform(0.40, 0.95, 180), rng.uniform(-1.18, -0.98, 180),
                            rng.uniform(0.74, 0.76, 180)])
    bowl = np.column_stack([rng.uniform(0.62, 0.80, 70), rng.uniform(-1.16, -1.00, 70),
                            rng.uniform(0.79, 0.85, 70)])
    track = _remembered("e:1", "drawer", [0.67, -1.08, 0.80], [0.55, 0.22, 0.22])
    track.voxels = _voxelize(np.vstack([slab, bowl]))
    track.mean_rgb = np.array([0.2, 0.55, 0.2])
    track.bottom_seen = True
    est.tracks = {"e:1": track}
    est._rebuild_geometry(track, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert track.category == "bowl"
    assert max(track.size[0], track.size[1]) < 0.34
    assert track.size[2] < 0.15


def test_a_rim_filled_down_to_the_table_becomes_a_wine_glass_and_objects_already_on_it_keep_their_names():
    """The head camera sees the rim and calls it a bowl. Filling the stem down to the table makes
    the box too tall for a bowl or a cup, so the name becomes a wine glass. A bowl and a cup whose
    points already reach the surface stay what they are."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    table = _remembered("e:table", "table", [0.6, -0.9, 0.378], [1.16, 0.79, 0.755])
    counter = _remembered("e:counter", "counter", [0.0, 2.18, 0.46], [1.8, 0.6, 0.9])
    rim = _remembered("e:1", "bowl", [0.50, -1.12, 0.89], [0.08, 0.08, 0.06])
    rng = np.random.default_rng(6)
    rim.voxels = _voxelize(np.column_stack([rng.uniform(0.46, 0.54, 40), rng.uniform(-1.16, -1.08, 40),
                                            rng.uniform(0.86, 0.92, 40)]))
    rim.mean_rgb = np.array([0.78, 0.80, 0.84])
    rim.bottom_seen = True
    rim.semantic_counts = {"bowl": 3.6, "cup": 2.8}
    rim.semantic_views = 4
    rim.recent.extend((np.array([0.50, -1.12, 0.89]), np.array([0.08, 0.08, 0.06])) for _ in range(3))
    bowl = _remembered("e:2", "bowl", [0.72, -1.07, 0.79], [0.20, 0.20, 0.06])
    bowl.voxels = _voxelize(np.column_stack([rng.uniform(0.62, 0.82, 40), rng.uniform(-1.17, -0.97, 40),
                                             rng.uniform(0.76, 0.82, 40)]))
    bowl.mean_rgb = np.array([0.2, 0.55, 0.2])
    bowl.bottom_seen = True
    bowl.semantic_counts = {"bowl": 3.0}
    bowl.semantic_views = 4
    bowl.recent.extend((np.array([0.72, -1.07, 0.79]), np.array([0.20, 0.20, 0.06])) for _ in range(3))
    cup = _remembered("e:3", "cup", [-0.50, 2.16, 0.96], [0.08, 0.08, 0.10])
    cup.voxels = _voxelize(np.column_stack([rng.uniform(-0.54, -0.46, 40), rng.uniform(2.12, 2.20, 40),
                                            rng.uniform(0.91, 1.01, 40)]))
    cup.mean_rgb = np.array([0.9, 0.9, 0.92])
    cup.bottom_seen = True
    cup.semantic_counts = {"cup": 3.2}
    cup.semantic_views = 4
    cup.recent.extend((np.array([-0.50, 2.16, 0.96]), np.array([0.08, 0.08, 0.10])) for _ in range(3))
    est.tracks = {"e:table": table, "e:counter": counter, "e:1": rim, "e:2": bowl, "e:3": cup}
    for track in (rim, bowl, cup):
        est._rebuild_geometry(track, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert rim.category == "wine_glass"
    assert rim.size[2] > 0.14
    assert bowl.category == "bowl"
    assert cup.category == "cup"


def test_a_narrow_cabinet_slice_keeps_its_drawer_name():
    """A cabinet face taller than a drawer prior keeps the drawer name.

    Dropping that name lets the slice be folded into the cabinet and the cabinet grows. Bowl and
    cup names on a wine-glass-height box are the ones left out."""
    from steward.ego.estimator import _labels_within_height

    assert "drawer" in _labels_within_height({"drawer": 0.85, "box": 0.4}, 0.40)
    assert _labels_within_height({"bowl": 0.85, "cup": 0.55}, 0.174) == {}


def test_two_names_for_one_glass_merge_and_two_bowls_stay_apart():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "cup", [0.47, -1.12, 0.84], [0.08, 0.07, 0.17]),
        "e:2": _remembered("e:2", "bowl", [0.48, -1.12, 0.84], [0.12, 0.08, 0.17]),
        "e:3": _remembered("e:3", "bowl", [0.24, -1.07, 0.78], [0.31, 0.31, 0.08]),
        "e:4": _remembered("e:4", "bowl", [0.71, -1.07, 0.79], [0.27, 0.20, 0.08]),
    }
    est._merge_same_place(np.zeros(3))
    assert "e:2" not in est.tracks and "e:1" in est.tracks
    assert {"e:3", "e:4"} <= set(est.tracks)


def test_a_shallow_bowl_beside_a_glass_splits_out_of_the_table_mask():
    """The green bowl is a short bump on the same mask as the glass, with table points on its side
    of the gap. The gap split keeps the bump and the glass, not the table."""
    from steward.ego.perception import _raised_cluster_indices

    rng = np.random.default_rng(4)
    glass = np.column_stack([rng.uniform(0.40, 0.48, 90), rng.uniform(-1.16, -1.08, 90),
                             rng.uniform(0.78, 0.92, 90)])
    bowl = np.column_stack([rng.uniform(0.62, 0.82, 80), rng.uniform(-1.16, -1.00, 80),
                            rng.uniform(0.78, 0.84, 80)])
    table = np.column_stack([rng.uniform(0.78, 0.98, 50), rng.uniform(-1.20, -0.98, 50),
                             rng.uniform(0.74, 0.76, 50)])
    split = _raised_cluster_indices(np.vstack([table, bowl, glass]), 20)
    assert split is not None and len(split) == 2
    widths = sorted(float(np.max(pts[:, 0]) - np.min(pts[:, 0])) for pts in
                    (np.vstack([table, bowl, glass])[idx] for idx in split))
    assert widths[0] < 0.15 and widths[1] < 0.32


def test_a_better_known_drawer_keeps_its_name_beside_the_sideboard():
    """The drawer shares the sideboard's base and footprint. Folding it in renames the sideboard
    to the drawer's id, and the open then fails because that id is no longer a drawer."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    side = _remembered("e:side", "sideboard", [-2.74, 0.6, 0.42], [1.4, 0.47, 0.84])
    side.observations = 4
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.45], [0.56, 0.44, 0.90])
    drawer.observations = 12
    est.tracks = {"e:side": side, "e:14": drawer}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:14"].category == "drawer"
    assert est.tracks["e:side"].category == "sideboard"
    young = _remembered("e:15", "drawer", [-2.74, 0.6, 0.45], [0.56, 0.44, 0.90])
    young.observations = 3
    est.tracks = {"e:side": side, "e:15": young}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:15"].category == "drawer" and "e:side" in est.tracks
    tall = _remembered("e:tall", "drawer", [-2.74, 0.6, 0.42], [1.0, 0.5, 0.75])
    short = _remembered("e:short", "drawer", [-2.74, 0.6, 0.16], [0.56, 0.4, 0.22])
    est.tracks = {"e:tall": tall, "e:short": short}
    est._merge_parts(np.zeros(3))
    assert est.tracks["e:short"].category == "drawer" and est.tracks["e:short"].size[2] < 0.4


def test_a_drawer_does_not_absorb_the_sideboard_face():
    from steward.ego.estimator import _voxelize
    from steward.ego.perception import Segment

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.55], [0.56, 0.44, 0.22])
    drawer.observations = 4
    rng = np.random.default_rng(0)
    drawer.voxels = _voxelize(np.column_stack([rng.uniform(-3.02, -2.46, 80), rng.uniform(0.38, 0.82, 80),
                                               rng.uniform(0.44, 0.66, 80)]))
    drawer.mean_rgb = np.array([0.6, 0.5, 0.4])
    face = np.column_stack([rng.uniform(-3.44, -2.04, 200), rng.uniform(0.36, 0.84, 200),
                            rng.uniform(0.02, 0.82, 200)])
    seg = Segment(points=face, color=np.array([0.55, 0.45, 0.35]), pixels=4000, invalid_frac=0.0,
                  border=False, range=1.2, semantics={"sideboard": 0.9})
    est._fuse_cloud(drawer, {"seg": seg, "centre": np.array([-2.74, 0.6, 0.42]),
                             "size": np.array([1.4, 0.48, 0.8]), "yaw": 0.0,
                             "probs": {"sideboard": 0.9}, "footprint": 1.4}, np.zeros(3))
    assert drawer.category == "drawer" and drawer.footprint < 0.7
    drawer.pinned = True
    est._fuse_cloud(drawer, {"seg": seg, "centre": np.array([-2.74, 0.6, 0.42]),
                             "size": np.array([1.4, 0.48, 0.8]), "yaw": 0.0,
                             "probs": {"sideboard": 0.9}, "footprint": 1.4}, np.zeros(3))
    assert drawer.pinned and drawer.footprint < 0.7
    cabinet = _remembered("e:4", "cabinet", [2.67, 0.6, 0.4], [0.6, 0.5, 0.8])
    cabinet.observations = 8
    cabinet.voxels = _voxelize(np.column_stack([rng.uniform(2.37, 2.97, 80), rng.uniform(0.35, 0.85, 80),
                                               rng.uniform(0.0, 0.8, 80)]))
    pulled = np.column_stack([rng.uniform(2.2, 2.97, 120), rng.uniform(0.35, 0.85, 120),
                             rng.uniform(0.4, 0.8, 120)])
    open_seg = Segment(points=pulled, color=np.array([0.5, 0.4, 0.3]), pixels=2000, invalid_frac=0.0,
                       border=False, range=1.0, semantics={"drawer": 0.8})
    est._fuse_cloud(cabinet, {"seg": open_seg, "centre": np.array([2.55, 0.6, 0.6]),
                              "size": np.array([0.9, 0.5, 0.4]), "yaw": 0.0,
                              "probs": {"drawer": 0.8}, "footprint": 0.9}, np.zeros(3))
    assert cabinet.footprint < 0.7
    # Already a little past the prior: one more step under 12 cm must still be refused.
    swollen = _remembered("e:sw", "cabinet", [2.55, 0.6, 0.4], [0.68, 0.55, 0.8])
    swollen.observations = 8
    swollen.voxels = _voxelize(np.column_stack([rng.uniform(2.21, 2.89, 80), rng.uniform(0.32, 0.88, 80),
                                               rng.uniform(0.0, 0.8, 80)]))
    further = np.column_stack([rng.uniform(2.10, 2.89, 40), rng.uniform(0.32, 0.88, 40),
                              rng.uniform(0.4, 0.8, 40)])
    more = Segment(points=further, color=np.array([0.5, 0.4, 0.3]), pixels=800, invalid_frac=0.0,
                   border=False, range=1.0, semantics={"drawer": 0.5})
    before = float(swollen.footprint)
    est._fuse_cloud(swollen, {"seg": more, "centre": np.array([2.5, 0.6, 0.6]),
                              "size": np.array([0.79, 0.56, 0.4]), "yaw": 0.0,
                              "probs": {"drawer": 0.5}, "footprint": 0.79}, np.zeros(3))
    assert swollen.footprint == pytest.approx(before)
    short = _remembered("e:15", "drawer", [-2.74, 0.6, 0.71], [0.56, 0.44, 0.22])
    short.observations = 4
    short.voxels = _voxelize(np.column_stack([rng.uniform(-3.02, -2.46, 60), rng.uniform(0.4, 0.8, 60),
                                              rng.uniform(0.60, 0.82, 60)]))
    tall_face = np.column_stack([rng.uniform(-3.2, -2.2, 120), rng.uniform(0.36, 0.84, 120),
                                rng.uniform(0.05, 0.82, 120)])
    tall_seg = Segment(points=tall_face, color=np.array([0.5, 0.4, 0.3]), pixels=2000, invalid_frac=0.0,
                       border=False, range=1.2, semantics={"drawer": 0.7})
    est._fuse_cloud(short, {"seg": tall_seg, "centre": np.array([-2.7, 0.6, 0.42]),
                            "size": np.array([1.0, 0.48, 0.77]), "yaw": 0.0,
                            "probs": {"drawer": 0.7}, "footprint": 1.0}, np.zeros(3))
    assert short.category == "drawer" and float(short.size[2]) < 0.4


def test_a_drawer_sized_track_keeps_its_name_when_the_shape_says_sideboard():
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.42], [1.0, 0.47, 0.84])
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-3.24, -2.24, 150), rng.uniform(0.36, 0.84, 150),
                           rng.uniform(0.0, 0.84, 150)])
    drawer.voxels = _voxelize(pts)
    drawer.mean_rgb = np.array([0.40, 0.26, 0.14])
    drawer.semantic_counts = {"sideboard": 4.0}
    drawer.semantic_views = 4
    drawer.bottom_seen = True
    centre, size, _ = box_of(pts, trim=0.01)
    probs = est.category_probs(size, float(centre[2] - size[2] / 2), drawer.mean_rgb, 0.0, True,
                               {"sideboard": 1.0})
    assert max(probs, key=probs.get) == "sideboard"
    est._rebuild_geometry(drawer, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert drawer.category == "drawer" and float(drawer.size[2]) <= 0.40
    assert float(drawer.pos[2]) > 0.55
    wide = _remembered("e:16", "drawer", [-2.74, 0.6, 0.42], [1.4, 0.47, 0.84])
    wide.voxels = _voxelize(np.column_stack([rng.uniform(-3.44, -2.04, 150), rng.uniform(0.36, 0.84, 150),
                                             rng.uniform(0.0, 0.84, 150)]))
    wide.mean_rgb = drawer.mean_rgb.copy()
    wide.semantic_counts = {"sideboard": 4.0}
    wide.semantic_views = 4
    wide.bottom_seen = True
    est._rebuild_geometry(wide, np.array([-1.6, -1.3, 1.2]), 2.0)
    assert wide.category == "drawer" and float(wide.size[2]) <= 0.40
    assert float(wide.pos[2]) > 0.55


def test_a_sideboard_face_labeled_as_a_drawer_keeps_its_top_band():
    """The handle is in the top band. A box of the whole face puts the hand on the plinth."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(2)
    face = np.column_stack([rng.uniform(-2.96, -2.52, 240), rng.uniform(0.15, 1.05, 240),
                            rng.uniform(0.04, 0.84, 240)])
    drawer = _remembered("e:14", "drawer", [-2.74, 0.6, 0.44], [0.9, 0.44, 0.8])
    drawer.voxels = _voxelize(face)
    drawer.mean_rgb = np.array([0.45, 0.32, 0.22])
    drawer.bottom_seen = True
    drawer.semantic_counts = {"drawer": 3.0}
    drawer.semantic_views = 3
    est._rebuild_geometry(drawer, np.array([-1.6, 0.6, 1.2]), 1.5)
    assert float(drawer.size[2]) <= 0.40
    assert float(drawer.pos[2]) > 0.55
    assert drawer.category == "drawer"


def test_a_grazing_edge_is_dropped_and_the_sideboard_stays():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    est.tracks = {
        "e:1": _remembered("e:1", "unknown", [1.0, -1.6, 0.4], [2.3, 0.29, 0.5]),
        "e:2": _remembered("e:2", "sideboard", [-2.7, 0.6, 0.42], [1.4, 0.47, 0.84]),
    }
    est._drop_impossible(np.zeros(3))
    assert "e:1" not in est.tracks and est.tracks["e:2"].category == "sideboard"


def test_a_shut_drawer_drops_the_length_it_had_while_open():
    """Pushing a drawer shut keeps the front and one prior-length behind it.
    The points that spanned the open travel were the cabinet, and they go.
    Sliding it when nothing shows it as open leaves that length alone."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 0.86, 40)
    ys = np.linspace(-0.2, 0.2, 6)
    zs = np.linspace(0.55, 0.75, 4)
    cloud = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    drawer = _remembered("e:5", "drawer", [0.43, 0.0, 0.65], [0.86, 0.40, 0.20])
    drawer.voxels = cloud
    est.tracks = {"e:5": drawer}
    est._trim_shut_drawer(drawer, np.array([0.26, 0.0, 0.0]))
    assert float(max(drawer.size[0], drawer.size[1])) == pytest.approx(0.60, abs=0.05)
    opened = _remembered("e:6", "drawer", [0.43, 0.0, 0.65], [0.86, 0.40, 0.20])
    opened.voxels = cloud.copy()
    est.tracks = {"e:6": opened}
    est.shift("e:6", np.array([0.26, 0.0, 0.0]))
    assert float(max(opened.size[0], opened.size[1])) == pytest.approx(0.86, abs=0.05)


def test_a_flat_counter_sliver_and_an_unnamed_scissors_are_dropped():
    """A paper-thin unknown wider than a plate is the counter top. Scissors the segmenter
    never named are a cabinet edge completed to that prior. A plate, and scissors the
    segmenter did name, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    named = _remembered("e:9", "scissors", [0.2, -0.4, 0.76], [0.16, 0.07, 0.012])
    named.semantic_counts = {"scissors": 2.0}
    est.tracks = {
        "e:21": _remembered("e:21", "unknown", [0.63, 1.99, 0.90], [0.56, 0.27, 0.02]),
        "e:27": _remembered("e:27", "scissors", [2.45, 0.88, 0.80], [0.16, 0.07, 0.012]),
        "e:6": _remembered("e:6", "plate", [0.30, -0.80, 0.76], [0.24, 0.23, 0.02]),
        "e:9": named,
    }
    est._drop_impossible(np.zeros(3))
    assert "e:21" not in est.tracks and "e:27" not in est.tracks
    assert set(est.tracks) == {"e:6", "e:9"}


def test_table_edge_strips_and_a_tall_sliver_are_dropped_and_the_table_stays():
    """Strips south of the table, and a tall sliver a few centimetres thick, are not objects.
    The table, the pistol and the pill bottle are thinner than furniture and stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.odometry = {"x": 1.6, "y": 0.6, "yaw": 0.0}
    labeled = _remembered("e:6", "box", [2.76, 0.62, 0.66], [0.85, 0.65, 0.28])
    labeled.label = "drawer"
    est.tracks = {
        "e:6": labeled,
        "e:3": _remembered("e:3", "table", [0.59, -0.89, 0.38], [1.19, 0.79, 0.75]),
        "e:116": _remembered("e:116", "unknown", [-0.14, -1.68, 0.56], [1.73, 0.28, 0.54]),
        "e:119": _remembered("e:119", "unknown", [0.73, -1.81, 0.55], [1.49, 0.10, 0.60]),
        "e:120": _remembered("e:120", "unknown", [0.52, -1.75, 0.54], [1.61, 0.20, 0.56]),
        "e:106": _remembered("e:106", "unknown", [3.01, -2.30, 0.78], [0.40, 0.03, 1.42]),
        "e:45": _remembered("e:45", "chair", [1.82, -3.49, 0.62], [0.40, 0.07, 1.21]),
        "e:46": _remembered("e:46", "pistol", [-2.74, 0.69, 0.63], [0.18, 0.11, 0.03]),
        "e:82": _remembered("e:82", "pill_bottle", [2.71, 0.46, 0.59], [0.05, 0.04, 0.11]),
    }
    est._drop_impossible(np.zeros(3))
    assert set(est.tracks) == {"e:6", "e:3", "e:46", "e:82"}


def test_a_stretched_bowl_and_a_drawer_only_plate_are_dropped():
    """A bowl longer than the bowl prior is not a bowl. A plate the segmenter only called
    a drawer, sitting on that drawer, is the drawer. The chores bowl (under the prior) and
    the plate the segmenter named stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    plate = _remembered("e:31", "plate", [2.81, 0.82, 0.80], [0.24, 0.24, 0.02])
    plate.semantic_counts = {"drawer": 0.96}
    plate.semantic_views = 1
    named = _remembered("e:6", "plate", [0.30, -0.80, 0.76], [0.24, 0.23, 0.03])
    named.semantic_counts = {"plate": 13.7}
    named.semantic_views = 14
    bowl = _remembered("e:18", "bowl", [0.41, 2.17, 0.94], [0.17, 0.12, 0.06])
    est.tracks = {
        "e:9": _remembered("e:9", "bowl", [0.53, 2.31, 0.93], [0.48, 0.15, 0.07]),
        "e:18": bowl,
        "e:30": _remembered("e:30", "bowl", [0.24, -1.07, 0.79], [0.32, 0.31, 0.08]),
        "e:5": _remembered("e:5", "drawer", [2.62, 0.61, 0.69], [0.63, 0.65, 0.24]),
        "e:31": plate,
        "e:6": named,
    }
    est._drop_false_tableware()
    assert set(est.tracks) == {"e:18", "e:30", "e:5", "e:6"}


def test_a_long_bowl_drops_a_detached_counter_strip_and_a_cup_keeps_one():
    """Support voxels that do not touch a bowl stretch its box. The bowl keeps the compact cluster.

    A cup with the same kind of tail is left whole: only a track already named bowl is cut, and only
    when it is longer than a bowl."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    rng = np.random.default_rng(7)
    bowl_pts = np.column_stack([rng.uniform(0.36, 0.52, 45), rng.uniform(2.11, 2.24, 45),
                                rng.uniform(0.90, 0.96, 45)])
    strip = np.array([[0.60, 2.465, 0.925], [0.68, 2.50, 0.92], [0.70, 2.51, 0.92],
                      [0.72, 2.52, 0.925], [0.74, 2.53, 0.92]])
    bowl = _remembered("e:1", "bowl", [0.53, 2.31, 0.93], [0.48, 0.15, 0.07])
    bowl.voxels = _voxelize(np.vstack([bowl_pts, strip]))
    bowl.mean_rgb = np.array([0.93, 0.93, 0.95])
    bowl.bottom_seen = True
    bowl.semantic_counts = {"bowl": 4.0}
    bowl.semantic_views = 4
    swollen = (np.array([0.53, 2.31, 0.93]), np.array([0.50, 0.15, 0.06]))
    bowl.recent.extend(swollen for _ in range(3))
    cup_pts = np.column_stack([rng.uniform(-0.54, -0.46, 40), rng.uniform(2.12, 2.20, 40),
                               rng.uniform(0.90, 1.00, 40)])
    cup_tail = np.column_stack([rng.uniform(-0.10, -0.02, 8), rng.uniform(2.48, 2.54, 8),
                                rng.uniform(0.94, 0.96, 8)])
    cup = _remembered("e:2", "cup", [-0.30, 2.30, 0.95], [0.50, 0.16, 0.10])
    cup.voxels = _voxelize(np.vstack([cup_pts, cup_tail]))
    cup.bottom_seen = True
    est.tracks = {"e:1": bowl, "e:2": cup}
    est._rebuild_geometry(bowl, np.array([-1.8, -1.2, 1.2]), 2.0)
    est._rebuild_geometry(cup, np.array([-1.8, -1.2, 1.2]), 2.0)
    assert bowl.category == "bowl"
    assert max(bowl.size[0], bowl.size[1]) < 0.25
    assert abs(float(bowl.pos[0]) - 0.44) < 0.06 and abs(float(bowl.pos[1]) - 2.175) < 0.06
    assert cup.category == "cup"
    assert max(cup.size[0], cup.size[1]) > 0.40
    assert cup.size[2] > 0.08


def test_a_cabinet_edge_called_a_bowl_is_dropped_and_a_real_bowl_stays():
    """A lip a few centimetres thick is not a bowl. The 12 cm bowl, thin on its short side
    of the prior but round in plan, stays."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "bowl", [1.36, 2.92, 0.93], [0.19, 0.052, 0.05]),
        "e:2": _remembered("e:2", "bowl", [0.70, -2.09, 0.75], [0.137, 0.110, 0.045]),
    }
    est._drop_false_tableware()
    assert set(est.tracks) == {"e:2"}


def test_a_shorter_counter_seen_broadside_keeps_its_length():
    """A 1.4 m counter is shorter than the 1.85 m prior. The cloud already spans that
    length, so neither a broadside view nor a view looking along it stretches the box.
    A later end-on view of the same cloud leaves the length alone."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 1.4, 30)
    ys = np.linspace(-0.25, 0.25, 8)
    zs = np.linspace(0.05, 0.85, 6)
    cloud = np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3)

    def counter(oid: str):
        track = _remembered(oid, "counter", [0.7, 0.0, 0.425], [1.4, 0.5, 0.85])
        track.voxels = _voxelize(cloud)
        track.mean_rgb = np.array([0.92, 0.92, 0.93])
        track.bottom_seen = True
        return track

    seen = counter("e:1")
    est.tracks = {"e:1": seen}
    est._rebuild_geometry(seen, np.array([-0.23, -2.0, 1.2]), 2.0)
    assert seen.category == "counter"
    assert float(seen.size[0]) == pytest.approx(1.4, abs=0.05)
    est._rebuild_geometry(seen, np.array([-2.0, 0.0, 1.2]), 2.0)
    assert float(seen.size[0]) == pytest.approx(1.4, abs=0.05)
    end_on = counter("e:2")
    est._rebuild_geometry(end_on, np.array([-2.0, 0.0, 1.2]), 2.0)
    assert end_on.category == "counter"
    assert float(max(end_on.size[0], end_on.size[1])) == pytest.approx(1.4, abs=0.05)


def test_a_square_table_shorter_than_the_prior_keeps_both_sides():
    """A 0.9 m table seen along one side is still 0.9 m. That side is most of the 1.2 m
    prior, so it is not filled the way an 8 cm drawer face is."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(-0.45, 0.45, 20)
    ys = np.linspace(-0.45, 0.45, 20)
    zs = np.linspace(0.02, 0.68, 5)
    table = _remembered("e:1", "table", [0.0, 0.0, 0.34], [0.9, 0.9, 0.68])
    table.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    table.bottom_seen = True
    est.tracks = {"e:1": table}
    est._rebuild_geometry(table, np.array([-2.0, -0.4, 1.2]), 2.0)
    assert table.category == "table"
    assert float(max(table.size[0], table.size[1])) == pytest.approx(0.9, abs=0.05)


def test_a_partly_seen_drawer_depth_still_takes_the_prior():
    """Thirty centimetres facing the camera is still the hidden side of a drawer.
    A facing side is left as measured only once it is already half a metre."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 0.56, 16)
    ys = np.linspace(-0.15, 0.15, 8)
    zs = np.linspace(0.49, 0.71, 5)
    drawer = _remembered("e:1", "drawer", [0.28, 0.0, 0.60], [0.56, 0.30, 0.22])
    drawer.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    drawer.bottom_seen = True
    est.tracks = {"e:1": drawer}
    est._rebuild_geometry(drawer, np.array([0.28, -2.0, 1.2]), 2.0)
    assert drawer.category == "drawer"
    assert float(min(drawer.size[0], drawer.size[1])) == pytest.approx(0.44, abs=0.05)


def test_a_thin_drawer_face_still_takes_the_prior_depth():
    """Eight centimetres of front is not the drawer. A view that meets that axis at 45 degrees
    still fills the prior depth; only a side that is already most of the prior is remembered."""
    from steward.ego.estimator import _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    xs = np.linspace(0.0, 1.4, 24)
    ys = np.linspace(-0.04, 0.04, 4)
    zs = np.linspace(0.05, 0.85, 6)
    face = _remembered("e:1", "counter", [0.7, 0.0, 0.45], [1.4, 0.08, 0.85])
    face.voxels = _voxelize(np.stack(np.meshgrid(xs, ys, zs), axis=-1).reshape(-1, 3))
    face.mean_rgb = np.array([0.92, 0.92, 0.93])
    face.bottom_seen = True
    est.tracks = {"e:1": face}
    est._rebuild_geometry(face, np.array([-1.0, -1.0, 1.2]), 2.0)
    assert face.category == "counter"
    assert float(min(face.size[0], face.size[1])) == pytest.approx(0.62, abs=0.05)


def test_a_drawer_completed_past_the_sideboard_is_dropped_and_the_real_one_stays():
    """The top band of the sideboard stays until the real drawer is pulled, then it is the one that did not move.
    The drawer in the sideboard, and one pulled out of the cabinet at drawer height, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    sideboard = _remembered("e:18", "sideboard", [-2.71, 0.59, 0.44], [1.38, 0.52, 0.84])
    sideboard.yaw = math.pi / 2  # the long side runs north; the extra drawer still lies in that box
    est.tracks = {
        "e:18": sideboard,
        "e:4": _remembered("e:4", "cabinet", [2.67, 0.60, 0.40], [0.62, 0.54, 0.80]),
        "e:52": _remembered("e:52", "drawer", [-2.74, 1.04, 0.81], [0.53, 0.44, 0.22]),
        "e:15": _remembered("e:15", "drawer", [-2.69, 0.60, 0.71], [0.56, 0.43, 0.22]),
        "e:6": _remembered("e:6", "drawer", [2.20, 0.60, 0.66], [0.56, 0.44, 0.22]),
    }
    est._drop_support_top_drawers()
    assert "e:52" in est.tracks, "a second drawer stays until one of them is pulled"
    est.shift("e:15", np.array([0.28, 0.0, 0.0]))
    assert "e:52" not in est.tracks
    assert set(est.tracks) == {"e:18", "e:4", "e:15", "e:6"}


def test_a_wide_slab_on_a_table_is_dropped_and_an_open_cabinet_drawer_stays():
    """A tabletop stretched into a drawer is dropped. An open cabinet drawer is just as long,
    and it stays: its base is down in the cabinet, not on the table."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:table": _remembered("e:table", "table", [0.6, -0.9, 0.38], [1.16, 0.79, 0.75]),
        "e:1": _remembered("e:1", "drawer", [0.6, -0.9, 0.86], [0.92, 0.29, 0.22]),
        "e:2": _remembered("e:2", "drawer", [2.67, 0.60, 0.66], [0.92, 0.50, 0.28]),
    }
    est._drop_support_top_drawers()
    assert "e:1" not in est.tracks
    assert "e:2" in est.tracks


def test_a_drawer_on_a_counter_top_is_dropped_and_one_inside_a_cabinet_stays():
    """A short slab beside the cup is the counter top. A taller box in the counter beside that
    cup is the counter face. The cabinet's own drawer, and a taller drawer-shaped track on a
    table, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "counter", [0.0, 0.0, 0.45], [1.8, 0.6, 0.9]),
        "e:2": _remembered("e:2", "drawer", [-0.4, 0.1, 0.95], [0.46, 0.22, 0.11]),
        "e:5": _remembered("e:5", "cup", [-0.45, 0.0, 0.95], [0.08, 0.08, 0.10]),
        "e:3": _remembered("e:3", "cabinet", [2.0, 0.0, 0.4], [0.62, 0.52, 0.8]),
        "e:4": _remembered("e:4", "drawer", [2.0, 0.0, 0.65], [0.56, 0.44, 0.22]),
        "e:6": _remembered("e:6", "table", [0.6, -0.9, 0.38], [1.16, 0.79, 0.75]),
        "e:7": _remembered("e:7", "drawer", [0.72, -1.07, 0.86], [0.58, 0.25, 0.22]),
        "e:8": _remembered("e:8", "drawer", [-0.4, 0.1, 0.70], [0.55, 0.39, 0.33]),
    }
    est._drop_support_top_drawers()
    assert "e:2" not in est.tracks
    assert "e:8" not in est.tracks
    assert set(est.tracks) == {"e:1", "e:5", "e:3", "e:4", "e:6", "e:7"}


def test_a_bear_in_front_of_an_open_drawer_moves_back_when_it_closes():
    """The bear was seen in the open drawer, just past the estimated box. Closing carries it back.
    A bowl up on the furniture does not come along."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:14": _remembered("e:14", "drawer", [-2.7, 0.6, 0.66], [0.56, 0.44, 0.22]),
        "e:50": _remembered("e:50", "teddy_bear", [-2.30, 0.6, 0.66], [0.10, 0.10, 0.18]),
        "e:51": _remembered("e:51", "pistol", [-3.05, 0.6, 0.64], [0.18, 0.11, 0.03]),
        "e:20": _remembered("e:20", "bowl", [-2.30, 0.6, 0.95], [0.16, 0.16, 0.06]),
        "e:3": _remembered("e:3", "table", [0.6, -0.9, 0.38], [1.2, 0.8, 0.75]),
    }
    moved = est.shift("e:14", np.array([-0.26, 0.0, 0.0]))
    assert "e:50" in moved and "e:51" in moved
    assert est.tracks["e:50"].pos[0] == pytest.approx(-2.56, abs=0.01)
    assert est.tracks["e:51"].pos[0] == pytest.approx(-3.31, abs=0.01)
    assert est.tracks["e:20"].pos[0] == pytest.approx(-2.30)
    assert est.tracks["e:14"].pos[0] == pytest.approx(-2.96)
    # A pistol that shows up only after the close is still out in front, and is brought back.
    # The bear already rode the close, so it is not moved a second time.
    before = {"e:50": [-2.30, 0.6, 0.66]}
    late = _remembered("e:46", "pistol", [-2.40, 0.6, 0.64], [0.18, 0.11, 0.03])
    est.tracks["e:46"] = late
    brought = est.resettle("e:14", np.array([-0.26, 0.0, 0.0]), before)
    assert "e:46" in brought and "e:50" not in brought
    assert est.tracks["e:46"].pos[0] == pytest.approx(-2.66, abs=0.02)
    assert est.tracks["e:50"].pos[0] == pytest.approx(-2.56, abs=0.01)


def test_an_open_drawer_is_trimmed_off_the_cabinet():
    """Points of the open drawer sit on the camera side of the cabinet. The far face stays,
    and the box comes back inside the cabinet prior."""
    from steward.ego.estimator import _trim_to_furniture_prior, _voxelize

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    cabinet = _remembered("e:4", "cabinet", [2.5, 0.6, 0.4], [0.77, 0.61, 0.8])
    rng = np.random.default_rng(3)
    # Closed cabinet occupies x 2.37..2.97. The open drawer adds x down to 2.16, toward the camera.
    cloud = np.column_stack([rng.uniform(2.16, 2.93, 200), rng.uniform(0.30, 0.91, 200),
                            rng.uniform(0.02, 0.78, 200)])
    cabinet.voxels = _voxelize(cloud)
    kept = _trim_to_furniture_prior(cabinet, cloud, np.zeros(3))
    assert len(kept) < len(cloud)
    centre, size, _yaw = box_of(kept, trim=0.0)
    assert max(float(size[0]), float(size[1])) <= 0.71
    assert float(centre[0]) > 2.5
    est.tracks = {"e:4": cabinet}
    est._rebuild_geometry(cabinet, np.zeros(3), 1.5)
    assert cabinet.footprint <= 0.71


def test_a_longer_sideboard_seen_from_the_front_keeps_its_length():
    """The extra length is across the view, so it is the piece, not a drawer pulled toward the camera."""
    from steward.ego.estimator import _trim_to_furniture_prior, _voxelize

    side = _remembered("e:1", "sideboard", [2.0, 0.0, 0.4], [0.42, 1.6, 0.78])
    rng = np.random.default_rng(4)
    cloud = np.column_stack([rng.uniform(1.8, 2.22, 180), rng.uniform(-0.8, 0.8, 180),
                            rng.uniform(0.02, 0.78, 180)])
    kept = _trim_to_furniture_prior(side, cloud, np.zeros(3))
    _centre, size, _yaw = box_of(kept, trim=0.0)
    assert max(float(size[0]), float(size[1])) == pytest.approx(1.6, abs=0.08)


def test_a_small_object_inside_an_open_drawer_stays_its_own_track():
    """The bear sits in the open drawer. It is not a piece of the drawer front, so it keeps its own track."""
    from steward.math3d import matrix_from_quat

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 90, 120, 60.0
    rng = np.random.default_rng(2)
    centre = origin + rot @ np.array([1.05, 0.0, -0.05])
    bear = rng.uniform(centre - np.array([0.05, 0.04, 0.06]), centre + np.array([0.05, 0.04, 0.06]), (80, 3))
    drawer = _remembered("e:14", "drawer", centre.tolist(), [0.56, 0.44, 0.22])
    # A drawer with no furniture around it is an orphan edge. This cabinet holds it and stays
    # under the wall height, so the drawer is not dropped before the bear is seen.
    z = float(centre[2])
    top = min(1.10, z + 0.25)
    cabinet = _remembered("e:4", "cabinet", [float(centre[0]), float(centre[1]), top - 0.40],
                          [0.90, 0.70, 0.80])
    est.tracks = {"e:14": drawer, "e:4": cabinet}
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    _paint(depth, mask, cam, fovy, bear, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 140, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"teddy_bear": 0.7}}}
    est.update_rgbd(frame)
    header["stamp"] = 1.4
    est.update_rgbd(frame)
    small = [t for t in est.tracks.values() if t.id != "e:14" and t.footprint < 0.3]
    assert len(small) == 1, "the bear was folded into the drawer or split"
    assert "e:14" in est.tracks and est.tracks["e:14"].category == "drawer"


def test_a_failed_color_drops_that_segmenter_score_and_the_sideboard_remains():
    """Counter fails white on the walnut sideboard, then table fails birch. The sideboard prior passes."""
    from steward.ego.classify import classify, fuse_respecting_color

    size = np.array([1.39, 0.47, 0.84])
    color = np.array([0.206, 0.125, 0.073])
    shape = classify(size, 0.015, color, 0.0, bottom_seen=True, min_prob=0.0)
    sem = {"cabinet": 1.06, "box": 0.74, "counter": 1.01, "table": 0.45}
    named = fuse_respecting_color(sem, shape, color, 0.0, 0.5, 0.03)
    assert max(named, key=named.get) == "sideboard"
    white = np.array([0.709, 0.711, 0.705])
    counter = classify(np.array([1.80, 0.61, 0.89]), 0.015, white, 0.0, bottom_seen=True, min_prob=0.0)
    kept = fuse_respecting_color({"counter": 0.7, "table": 0.5}, counter, white, 0.0, 0.5, 0.03)
    assert max(kept, key=kept.get) == "counter"


def test_a_generic_bottle_name_yields_to_the_pill_prior():
    """``bottle`` has no template, so a pill-sized box keeps the pill prior.

    When the segmenter also says pill bottle and that score leads, the fused winner stays.
    """
    from steward.ego.classify import classify, fuse_semantics, shape_over_bottle

    size = np.array([0.045, 0.045, 0.112])
    color = np.array([0.804, 0.601, 0.424])
    shape = classify(size, 0.533, color, 0.0, bottom_seen=True, min_prob=0.0)
    fused = fuse_semantics({"bottle": 0.7}, shape, 0.5, 0.03)
    assert max(fused, key=fused.get) == "bottle"
    assert max(shape_over_bottle(fused, shape), key=shape_over_bottle(fused, shape).get) == "pill_bottle"
    # 11 cm tall is closer to a cup's height. The segmenter still said bottle, so the cup does not inherit it.
    tall = classify(np.array([0.05, 0.05, 0.112]), 0.533, np.array([0.822, 0.489, 0.211]), 0.0,
                    bottom_seen=True, min_prob=0.0)
    assert max(tall, key=tall.get) == "cup"
    tall_fused = fuse_semantics({"bottle": 0.7}, tall, 0.5, 0.03)
    assert max(shape_over_bottle(tall_fused, tall), key=shape_over_bottle(tall_fused, tall).get) == "pill_bottle"
    both = fuse_semantics({"bottle": 0.7, "pill_bottle": 0.85}, shape, 0.5, 0.03)
    kept = shape_over_bottle(both, shape)
    assert max(kept, key=kept.get) == "pill_bottle"
    assert kept is both


def test_a_dark_speck_below_the_bowl_prior_is_a_pistol_and_a_full_pistol_is_not_renamed_here():
    """Black depth covers only part of a pistol. A full-length pistol is left to the ordinary prior."""
    from steward.ego.classify import partial_pistol

    dark = np.array([0.09, 0.084, 0.087])
    assert partial_pistol(np.array([0.04, 0.03, 0.022]), "bowl", dark, 0.0) == "pistol"
    assert partial_pistol(np.array([0.18, 0.11, 0.03]), "pistol", dark, 0.0) is None
    assert partial_pistol(np.array([0.12, 0.11, 0.05]), "bowl", np.array([0.15, 0.29, 0.65]), 0.0) is None


def test_colour_names_the_bottle_the_airplane_and_the_bear():
    """Pink is the perfume, white is the pill, yellow is the airplane, brown is the bear."""
    from steward.ego.classify import content_color_name

    pink = np.array([0.95, 0.60, 0.75])
    white = np.array([0.94, 0.94, 0.92])
    yellow = np.array([0.95, 0.78, 0.10])
    brown = np.array([0.58, 0.38, 0.20])
    assert content_color_name(np.array([0.04, 0.04, 0.12]), pink, 0.0, "pill_bottle") == "perfume_bottle"
    assert content_color_name(np.array([0.05, 0.046, 0.11]), white, 0.0, "bottle") == "pill_bottle"
    assert content_color_name(np.array([0.20, 0.12, 0.10]), yellow, 0.0, "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.22, 0.08, 0.10]), yellow, 0.0, "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.134, 0.076, 0.105]), np.array([0.922, 0.542, 0.161]), 0.0,
                              "bowl") == "toy_airplane"
    assert content_color_name(np.array([0.12, 0.10, 0.16]), brown, 0.0, "drawer") == "teddy_bear"
    # A drawer front is half a metre wide. Colour does not rename it.
    assert content_color_name(np.array([0.56, 0.44, 0.22]), brown, 0.0, "drawer") is None
    assert content_color_name(np.array([0.085, 0.072, 0.064]), np.array([0.111, 0.11, 0.115]), 0.0, "cup") == "mug"
    assert content_color_name(np.array([0.09, 0.09, 0.10]), np.array([0.64, 0.72, 0.76]), 0.0, "cup") is None


def test_a_sideboard_sized_counter_is_the_sideboard_unless_it_is_white():
    """The counter is the long white run. A 1.4 m walnut piece SAM calls a counter is the sideboard."""
    from steward.ego.classify import sideboard_not_counter

    walnut = np.array([0.36, 0.22, 0.13])
    white = np.array([0.94, 0.94, 0.92])
    side = np.array([1.343, 0.478, 0.84])
    assert sideboard_not_counter(side, "counter", walnut, 0.0) == "sideboard"
    assert sideboard_not_counter(side, "counter", white, 0.0) is None
    assert sideboard_not_counter(np.array([1.77, 0.62, 0.89]), "counter", walnut, 0.0) is None


def test_a_pistol_stays_when_the_drawer_around_it_is_seen_again():
    """The pistol is a small track inside the open drawer. A later drawer mask, and a drawer
    fragment centred on it, do not replace it."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    pistol = _remembered("e:47", "pistol", [-2.55, 0.70, 0.64], [0.04, 0.03, 0.022])
    pistol.mean_rgb = np.array([0.09, 0.084, 0.087])
    pistol.bottom_seen = True
    fragment = _remembered("e:49", "drawer", [-2.52, 0.68, 0.66], [0.20, 0.12, 0.10])
    fragment.bottom_seen = True
    est.tracks = {"e:47": pistol, "e:49": fragment}
    est._merge_same_place(np.array([-1.4, 0.6, 1.2]))
    assert "e:47" in est.tracks and est.tracks["e:47"].category == "pistol"
    assert est.tracks["e:47"].footprint < 0.1

    from steward.math3d import matrix_from_quat

    header = {"stamp": 2.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    centre = origin + rot @ np.array([1.05, 0.0, -0.05])
    pistol.pos = centre
    h, w, fovy = 90, 120, 60.0
    rng = np.random.default_rng(3)
    front = rng.uniform(centre - np.array([0.28, 0.22, 0.11]), centre + np.array([0.28, 0.22, 0.11]), (240, 3))
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    _paint(depth, mask, cam, fovy, front, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 90, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"drawer": 0.8}}}
    est.update_rgbd(frame)
    assert "e:47" in est.tracks and est.tracks["e:47"].category == "pistol"
    assert est.tracks["e:47"].footprint < 0.1


def test_a_bear_that_shares_the_drawer_floor_is_not_folded_into_the_drawer():
    """The bear's mask includes the drawer floor. Those shared points do not make the bear the drawer."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    drawer = _remembered("e:14", "drawer", [-2.7, 0.6, 0.66], [0.56, 0.44, 0.22])
    bear = _remembered("e:50", "teddy_bear", [-2.5, 0.48, 0.70], [0.12, 0.10, 0.18])
    shared = np.array([[0, 0, 70], [1, 0, 70], [0, 1, 70]], np.int32)
    drawer.voxels = np.vstack([shared, np.array([[10, 0, 70], [11, 0, 70], [12, 0, 70]], np.int32)])
    bear.voxels = shared.copy()
    bear.observations = 2
    est.tracks = {"e:14": drawer, "e:50": bear}
    est._merge_same_surface(np.array([0.0, 0.0, 1.2]))
    assert "e:50" in est.tracks and est.tracks["e:50"].category == "teddy_bear"
    assert est.tracks["e:14"].category == "drawer"


def test_a_tall_square_the_detector_did_not_call_a_pill_is_the_perfume():
    """The pill is the region SAM names. The other tall square, labelled only as the drawer, is the perfume."""
    from steward.ego.classify import perfume_without_bottle_label

    perfume = np.array([0.04, 0.04, 0.122])
    assert perfume_without_bottle_label(perfume, "pill_bottle", {"drawer": 0.93}) == "perfume_bottle"
    assert perfume_without_bottle_label(perfume, "pill_bottle", {"pill_bottle": 9.0, "bottle": 8.0}) is None
    assert perfume_without_bottle_label(np.array([0.08, 0.05, 0.112]), "pill_bottle", {"drawer": 1.0}) is None


def test_a_named_bear_stays_a_bear_when_a_later_view_says_drawer():
    """The open drawer behind the bear is labelled drawer. The bear already has its name, and keeps it."""
    from steward.assets.generators import steward_bot
    from steward.ego.estimator import EgoEstimator, Track

    est = EgoEstimator(robot_model=steward_bot().to_dict())
    bear = Track(id="e:1", category_counts={"teddy_bear": 5.0}, pos=np.array([1.0, 0.0, 0.7]),
                 var=np.full(3, 1e-4), yaw=0.0, yaw_var=1e-3, size=np.array([0.20, 0.15, 0.20]),
                 observations=6, first_seen=0.0, last_seen=1.0, existence=1.0,
                 mean_rgb=np.array([0.4, 0.4, 0.4]))
    bear.bottom_seen = True
    bear.semantic_counts = {"drawer": 5.0}
    bear.semantic_views = 5
    cloud = np.random.default_rng(0).uniform([0.90, -0.07, 0.60], [1.10, 0.08, 0.80], (200, 3))
    bear.voxels = np.unique(np.floor(cloud / 0.01).astype(np.int32), axis=0)
    est.tracks[bear.id] = bear
    est._rebuild_geometry(bear, np.array([0.0, 0.0, 1.2]), 1.0)
    assert bear.category == "teddy_bear"


def test_a_smear_along_one_axis_of_a_small_bottle_is_cut():
    """The pill is 4 cm across. A strip of drawer floor on one axis is not part of it.

    The cut starts at the bottle priors' own long/short ratio. An 8 cm by 4 cm strip is already
    past that. A 14 cm strip is too.
    """
    from steward.ego.classify import max_plan_aspect
    from steward.ego.estimator import _trim_long_smear

    aspect = max_plan_aspect(("pill_bottle", "perfume_bottle"))
    rng = np.random.default_rng(1)
    body = rng.uniform([-0.02, -0.02, 0.55], [0.02, 0.02, 0.66], (40, 3))
    smear = rng.uniform([-0.07, -0.01, 0.55], [0.07, 0.01, 0.57], (40, 3))
    cloud = np.vstack([body, smear])
    trimmed = _trim_long_smear(cloud, 0.04)
    assert len(trimmed) < len(cloud)
    assert float(np.ptp(trimmed[:, 0])) < 0.05
    mild = np.vstack([
        rng.uniform([-0.02, -0.02, 0.55], [0.02, 0.02, 0.66], (40, 3)),
        rng.uniform([-0.04, -0.01, 0.55], [0.04, 0.01, 0.57], (30, 3)),
    ])
    assert _trim_long_smear(mild, 0.04) is mild
    cut = _trim_long_smear(mild, 0.04, aspect)
    assert len(cut) < len(mild)
    assert float(np.ptp(cut[:, 0])) < 0.05


def test_a_mask_mixing_near_and_far_surfaces_does_not_start_a_wide_track():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    h, w, fovy = 80, 100, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(1)
    for x0 in (1.3, 2.8):
        pts = np.column_stack([np.full(500, x0), rng.uniform(-0.8, 0.8, 500), rng.uniform(0.2, 1.2, 500)])
        _paint(depth, mask, cam, fovy, pts, 1)
    frame = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 120, np.uint8), "fovy": fovy,
             "header": header, "labels": {1: {"box": 0.6}}}
    est.update_rgbd(frame)
    assert not any(t.footprint > 1.0 for t in est.tracks.values())


def test_a_box_standing_on_a_counter_stays_its_own_track():
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    counter = _remembered("e:1", "counter", [0.0, 0.0, 0.45], [1.8, 0.62, 0.9])
    box = _remembered("e:2", "box", [0.2, 0.0, 0.98], [0.4, 0.35, 0.16])
    est.tracks = {"e:1": counter, "e:2": box}
    est._fold_furniture_slices(np.array([0.0, -1.5, 1.4]))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_an_upper_half_named_box_is_folded_back_into_the_cabinet():
    """The upper body named box is a second object. Folding it back removes that object and
    restores the cabinet's height."""
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    forward = matrix_from_quat(cam["q"])[:, 0].copy()
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    centre = np.asarray(cam["p"], float) + forward * 1.4
    centre[2] = 0.4
    h, w, fovy = 120, 160, 60.0
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)
    rgb = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(2)

    def slab(z0, z1, instance, color, hx=0.28, hy=0.22):
        pts = np.column_stack([
            rng.uniform(centre[0] - hx, centre[0] + hx, 400),
            rng.uniform(centre[1] - hy, centre[1] + hy, 400),
            rng.uniform(z0, z1, 400),
        ])
        _paint(depth, mask, cam, fovy, pts, instance)
        ys, xs = np.nonzero(mask == instance)
        rgb[ys, xs] = color

    slab(0.02, 0.72, 1, (90, 110, 150))  # the body, blue-grey
    # Narrower than 0.5 m, so it is not dropped as a second copy of the cabinet, and still
    # wide enough to be the cabinet's upper band rather than an object standing on it.
    slab(0.52, 0.76, 2, (180, 40, 40), hx=0.16, hy=0.16)
    frame = {"depth": depth, "mask": mask, "rgb": rgb, "fovy": fovy, "header": header,
             "labels": {1: {"cabinet": 0.9}, 2: {"box": 0.8}}}

    def run(fold: bool) -> EgoEstimator:
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        if not fold:
            est._fold_furniture_slices = lambda _cam_p: None
        est.update_rgbd(frame)
        return est

    before, after = run(False), run(True)
    assert any(t.category == "box" for t in before.tracks.values())
    assert not any(t.category == "box" for t in after.tracks.values())
    cab = next(t for t in after.tracks.values() if t.category == "cabinet")
    assert cab.top() == pytest.approx(0.76, abs=0.06)
    assert "box" not in cab.category_counts


def test_a_printed_lid_with_a_visible_wall_gains_the_box_under_it():
    """The segmenter names the printed lid and not the 5 cm box under it. Depth of that wall,
    inside the lid's outline, becomes its own box. A lid with nothing under it does not."""
    from steward.ego.perception import segments
    from steward.math3d import matrix_from_quat

    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = EgoEstimator(robot_model=steward_bot().to_dict()).camera_pose(header["odometry"])
    rot = matrix_from_quat(cam["q"])
    origin = np.asarray(cam["p"], float)
    h, w, fovy = 120, 160, 60.0
    rng = np.random.default_rng(1)
    lid_centre = origin + rot @ np.array([1.15, 0.0, 0.0])
    lid = rng.uniform(lid_centre - np.array([0.17, 0.17, 0.004]), lid_centre + np.array([0.17, 0.17, 0.004]), (120, 3))
    forward = rot @ np.array([1.0, 0.0, 0.0])
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    wall = lid.copy()
    wall[:, :2] -= forward[:2] * 0.08
    wall[:, 2] = lid[:, 2] - rng.uniform(0.02, 0.06, len(wall))

    def frame_of(points_lid, points_wall):
        depth = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        _paint(depth, mask, cam, fovy, points_lid, 1)
        if points_wall is not None:
            lid_mask = mask.copy()
            depth_lid = depth.copy()
            _paint(depth, mask, cam, fovy, points_wall, 1)
            wall_depth = depth.copy()
            wall_only = (mask == 1) & (lid_mask == 0)
            depth[:] = depth_lid
            depth[wall_only] = wall_depth[wall_only]
            mask[:] = lid_mask
        return {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 180, np.uint8), "fovy": fovy,
                "header": header, "labels": {1: {"picture": 0.8}}}

    seen = segments(frame_of(lid, wall), cam)
    assert any(s.semantics and "box" in s.semantics for s in seen)
    assert not any(s.semantics and "box" in (s.semantics or {}) for s in segments(frame_of(lid, None), cam))
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.update_rgbd(frame_of(lid, wall))
    boxes = [t for t in est.tracks.values() if t.category == "box"]
    assert len(boxes) == 1
    assert boxes[0].size[2] == pytest.approx(0.05, abs=0.03)
    assert min(boxes[0].size[0], boxes[0].size[1]) > 0.2


def test_phase1_nested_bowls_stay_separate_tracks_across_views():
    """Three bowls share a centre and are separate masks. Later views must not fuse the smaller ones in."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    header = {"stamp": 1.0, "odometry": {"x": 0.0, "y": 0.0, "yaw": 0.0},
              "joints": {"lift": 0.4, "reach": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0},
              "gripper": {"state": "open", "holding": False}}
    cam = est.camera_pose(header["odometry"])
    h, w, fovy = 120, 160, 60.0
    rng = np.random.default_rng(1)
    board = 0.85
    specs = (
        (1, [1.5, 0.0, board + 0.035], [0.30, 0.30, 0.07], [180, 40, 40]),
        (2, [1.5, 0.0, board + 0.038], [0.20, 0.20, 0.06], [40, 150, 50]),
        (3, [1.5, 0.0, board + 0.041], [0.12, 0.12, 0.05], [40, 60, 180]),
    )

    def frame(stamp: float) -> dict:
        depth = np.zeros((h, w), np.float32)
        mask = np.zeros((h, w), np.uint8)
        rgb = np.full((h, w, 3), 20, np.uint8)
        for inst, centre, size, color in specs:
            pts = rng.uniform(np.asarray(centre) - np.asarray(size) / 2, np.asarray(centre) + np.asarray(size) / 2, (200, 3))
            _paint(depth, mask, cam, fovy, pts, inst)
            rgb[mask == inst] = color
        return {"depth": depth, "mask": mask, "rgb": rgb, "fovy": fovy, "camera": "head_camera",
                "header": {**header, "stamp": stamp}}

    for k in range(4):
        est.update_rgbd(frame(0.2 * (k + 1)))
    bowls = sorted((t for t in est.tracks.values() if t.category == "bowl"), key=lambda t: -t.footprint)
    assert len(bowls) == 3, [(t.id, t.category, [round(float(v), 3) for v in t.size]) for t in est.tracks.values()]
    assert est.support_of(bowls[2]) is bowls[1]
    assert est.support_of(bowls[1]) is bowls[0]


def test_a_one_sided_view_of_a_bowl_joins_the_bowl_already_seen():
    """The bowl on the cabinet, and a smaller box a few centimetres to the side seen once.

    The footprints differ by more than the nesting step, but the centres do not coincide, so the
    smaller box is a partial view. The track with more observations keeps its box."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    full = _remembered("e:5", "bowl", [2.676, 0.587, 0.834], [0.194, 0.152, 0.069])
    full.observations = 13
    part = _remembered("e:13", "bowl", [2.616, 0.609, 0.837], [0.117, 0.063, 0.056])
    part.observations = 1
    est.tracks = {"e:5": full, "e:13": part}
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:5"}
    assert est.tracks["e:5"].footprint == pytest.approx(0.194, abs=1e-6)
    closer = _remembered("e:8", "bowl", [2.670, 0.589, 0.832], [0.186, 0.149, 0.067])
    closer.observations = 12
    aside = _remembered("e:14", "bowl", [2.631, 0.586, 0.840], [0.132, 0.096, 0.050])
    aside.observations = 1
    est.tracks = {"e:8": closer, "e:14": aside}
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:8"}
    nested = {
        "e:1": _remembered("e:1", "bowl", [1.5, 0.0, 0.885], [0.30, 0.30, 0.07]),
        "e:2": _remembered("e:2", "bowl", [1.5, 0.0, 0.888], [0.20, 0.20, 0.06]),
    }
    est.tracks = nested
    est._merge_same_place(np.zeros(3))
    assert set(est.tracks) == {"e:1", "e:2"}


def test_observe_returns_each_camera_png_for_the_tool_to_attach(tmp_path):
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

    class Link:
        def sensors(self):
            return {"stamp": 1.5, "imaging": "detections",
                    "detections": [{"category": "cup", "range": 0.42, "camera": "head_camera"}],
                    "cameras": [{"id": "head_camera"}, {"id": "wide_camera"}]}

        def camera_png(self, width=320, height=240, camera=None):
            return png

        def command(self, cmd):
            return None

    runtime = EgoRuntime(Link(), steward_bot().to_dict(), out_dir=tmp_path)
    out = runtime.observe()
    assert set(out["images"]) == {"head_camera", "wide_camera"}
    assert base64.b64decode(out["images"]["head_camera"]) == png
    assert out["visible"] == [{"category": "cup", "rangeM": 0.42, "camera": "head_camera"}]
    assert len(list(tmp_path.glob("observe-1.50-*.png"))) == 2
