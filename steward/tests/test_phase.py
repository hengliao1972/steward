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
    """A bowl and a mug painted as one region are one wide track. With segmenter labels the gap
    splits them, and each keeps its own size."""
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
    _paint(depth, mask, cam, fovy, blob([1.15, -0.16, -0.02], [0.09, 0.08, 0.09]), 1)
    bare = {"depth": depth, "mask": mask, "rgb": np.full((h, w, 3), 160, np.uint8), "fovy": fovy,
            "header": header, "camera": "head_camera"}
    assert len(segments(bare, cam)) == 1, "an anonymous mask stays one region"
    named = dict(bare)
    named["labels"] = {1: {"bowl": 0.5}}
    assert len(segments(named, cam)) == 2
    before, after = EgoEstimator(robot_model=steward_bot().to_dict()), EgoEstimator(robot_model=steward_bot().to_dict())
    before.update_rgbd(bare)
    after.update_rgbd(named)
    assert len(before.tracks) == 1 and max(t.footprint for t in before.tracks.values()) > 0.25
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


def test_a_drawer_on_a_counter_top_is_dropped_and_one_inside_a_cabinet_stays():
    """A short slab beside the cup is the counter top. The cabinet's own drawer, and a taller
    drawer-shaped track on a table, stay."""
    est = EgoEstimator(robot_model=steward_bot().to_dict())
    est.tracks = {
        "e:1": _remembered("e:1", "counter", [0.0, 0.0, 0.45], [1.8, 0.6, 0.9]),
        "e:2": _remembered("e:2", "drawer", [-0.4, 0.1, 0.95], [0.46, 0.22, 0.11]),
        "e:5": _remembered("e:5", "cup", [-0.45, 0.0, 0.95], [0.08, 0.08, 0.10]),
        "e:3": _remembered("e:3", "cabinet", [2.0, 0.0, 0.4], [0.62, 0.52, 0.8]),
        "e:4": _remembered("e:4", "drawer", [2.0, 0.0, 0.65], [0.56, 0.44, 0.22]),
        "e:6": _remembered("e:6", "table", [0.6, -0.9, 0.38], [1.16, 0.79, 0.75]),
        "e:7": _remembered("e:7", "drawer", [0.72, -1.07, 0.86], [0.58, 0.25, 0.22]),
    }
    est._drop_support_top_drawers()
    assert "e:2" not in est.tracks
    assert set(est.tracks) == {"e:1", "e:5", "e:3", "e:4", "e:6", "e:7"}


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

    def slab(z0, z1, instance, color):
        pts = np.column_stack([
            rng.uniform(centre[0] - 0.28, centre[0] + 0.28, 400),
            rng.uniform(centre[1] - 0.22, centre[1] + 0.22, 400),
            rng.uniform(z0, z1, 400),
        ])
        _paint(depth, mask, cam, fovy, pts, instance)
        ys, xs = np.nonzero(mask == instance)
        rgb[ys, xs] = color

    slab(0.02, 0.72, 1, (90, 110, 150))  # the body, blue-grey
    slab(0.74, 0.76, 2, (90, 110, 150))  # the top face, a couple of centimetres above, named box
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
