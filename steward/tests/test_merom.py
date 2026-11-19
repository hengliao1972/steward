"""Merom assets retain their layout, collision support and mesh geometry."""

import copy
import hashlib
import json
import math
import struct
from contextlib import nullcontext
from zipfile import ZipFile

import mujoco
import numpy as np
import pytest

from steward.authoring import merom
from steward.authoring.service import AuthoringService
from steward.ego.bodies import body_arm
from steward.ego.esim import Esim
from steward.ego.estimator import EgoEstimator
from steward.ego.runtime import EgoRuntime, RobotLink
from steward.ego.skills import SkillFailed
from steward.net import HttpError
from steward.embodiments import robot_entity, with_body
from steward.observer.observer import Observer
from steward.scene.extent import entity_body_ids
from steward.scene.mjcf import build_mjcf
from steward.scene.schema import SceneState
from steward.scene.validate import validate_world
from steward.world.sim import WorldSim


def test_bundle_meshes_and_layout_match_the_conversion_record():
    manifest = json.loads((merom.BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((merom.BUNDLE / name).read_bytes()).hexdigest() == digest
    scene = json.loads((merom.BUNDLE / "scene.json").read_text())
    assert len(scene["entities"]) == 132
    assert all(e["body"]["root"]["value"] == "fixed" for e in scene["entities"])
    assert sum(e["role"] == "object" for e in scene["entities"]) == 105
    structure = [e for e in scene["entities"] if e["role"] == "environment"]
    assert all(e["category"]["value"] in {"floor", "wall"} for e in structure)
    assert sum(e["body"]["links"][0]["id"] == "floor" for e in structure) == 12
    with ZipFile(merom.BUNDLE / "meshes.zip") as archive:
        refs = set()
        for role, expected in [("visual", (220, 85933)), ("collision", (1163, 108504))]:
            geoms = [g for e in scene["entities"] for link in e["body"]["links"]
                     for g in link["geometry"].get(role, [])]
            triangles = 0
            for geom in geoms:
                if geom["kind"] != "mesh":
                    continue
                ref = geom["ref"]
                data = archive.read(ref)
                assert ref == hashlib.sha256(data).hexdigest() + ".msh"
                triangles += struct.unpack("<4i", data[:16])[3]
                refs.add(ref)
            assert (len(geoms), triangles) == expected
        assert set(archive.namelist()) == refs
        assert len(refs) == manifest["unique_meshes"]
    by_id = {e["id"]: e for e in scene["entities"]}
    for name in ["w:prep_obj_bread2", "w:prep_obj_bread4"]:
        assert by_id[name]["category"]["value"] == "bread"
        assert by_id[name]["body"]["links"][0]["geometry"]["visual"][0]["rgba"] == [0.8, 0.54, 0.27, 1]


def test_all_doorways_clear_one_and_a_half_cobot_widths(tmp_path):
    service = AuthoringService(tmp_path)
    service.ensure_scenario("merom-house")
    scene = service.load_scenario("merom-house")["sceneState"]
    robot = next(e for e in scene["entities"] if e["id"] == "w:robot-1")
    base = next(link for link in robot["body"]["links"] if link["id"] == "base")
    chassis = base["geometry"]["collision"][0]
    chassis["size"][1] *= 1.5
    sim = WorldSim(scene)
    try:
        names = sim.build.names[robot["id"]]
        joints = [sim.model.joint(name).qposadr[0] for name in names.root_joints]
        robot_bodies = set(entity_body_ids(sim.model, names))
        # Door centres in metres, with the yaw of a robot crossing the threshold.
        doorways = [
            (-2.150, -0.115, math.pi / 2),  # West bedroom closet
            (-3.350, 1.857, 0), (-0.490, 1.795, 0),  # Kitchen exterior and hall
            # Three doorways in the wall along the south side of the hall.
            (0.135, 1.025, math.pi / 2), (1.560, 1.025, math.pi / 2),
            (3.995, 1.025, math.pi / 2),
            # East hall to the bedroom passage, bathroom and closet.
            (2.280, 1.670, 0), (2.280, 2.980, 0),
            (2.280, 4.950, 0),
            (-0.645, 9.423, 0),  # Living room exterior
        ]
        for x, y, yaw in doorways:
            for offset in np.linspace(-0.44, 0.44, 45):
                xy = [x + offset * math.cos(yaw), y + offset * math.sin(yaw)]
                for heading in (yaw, yaw + math.pi):
                    sim.data.qpos[joints] = [*xy, heading]
                    mujoco.mj_forward(sim.model, sim.data)
                    for contact in sim.data.contact:
                        first, second = sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
                        if (first in robot_bodies) != (second in robot_bodies):
                            assert contact.dist >= -0.001, (x, y, offset, sim.model.geom(contact.geom1).name,
                                                            sim.model.geom(contact.geom2).name)
    finally:
        sim.stop()


def test_doorways_stay_open_in_perception_and_southern_room_is_reachable(tmp_path):
    service = AuthoringService(tmp_path)
    service.ensure_scenario("merom-house")
    scene = service.load_scenario("merom-house")["sceneState"]
    robot = next(e for e in scene["entities"] if e["role"] == "robot")
    planner = Esim(robot, arm=body_arm("cobot-magic", robot))
    sim = WorldSim(scene, seed=7)
    try:
        openings = 0
        for wall in sim.template.entities:
            if wall.category.value != "wall" or len(wall.rigid.links) == 1:
                continue
            boxes = {links[0]: sim.entity_obb(wall.id, links) for _, links, _ in sim.perception_parts(wall)}
            for link, lintel in boxes.items():
                if not link.startswith("lintel_"):
                    continue
                assert lintel.center[2] - lintel.size[2] / 2 > 1.9
                for pier, box in boxes.items():
                    if pier.startswith("wall_"):
                        offset = np.abs(np.array(box.center[:2]) - lintel.center[:2])
                        assert max(offset - np.array(box.size[:2]) / 2) > planner.radius + planner.config.nav_margin
                openings += 1
        assert openings == 10

        estimator = EgoEstimator(robot_model=robot)
        joints = [sim.model.joint(n).qposadr[0] for n in sim.build.names[robot["id"]].root_joints]
        for step, yaw in enumerate(np.linspace(-math.pi / 2, 3 * math.pi / 2, 13)):
            sim.data.qpos[joints] = [1.0, 1.71, yaw]
            sim.data.time = step * 0.1
            mujoco.mj_forward(sim.model, sim.data)
            estimator.update(sim.sensors(robot["id"]))
        belief = estimator.to_scene()
        # Confirm the robot saw both jambs before testing the route between them.
        obstacles = planner._floor_obstacles(belief, set())
        for x in (0.835, 2.636):
            assert any(b.contains(x, 1.025) for b in obstacles)
        assert not any(b.contains(1.56, 1.025) for b in obstacles)
        receipt = planner.predict(belief, "navigate", {"x": 2.2, "y": -0.1, "yaw": -math.pi / 2})
        assert receipt["feasible"] and receipt["plan"]["pathLength"] < 3.0
    finally:
        sim.stop()


def test_worktable_retains_metre_scale_support_and_open_space(tmp_path):
    service = AuthoringService(tmp_path)
    service.ensure_scenario("merom-house")
    scene = service.load_scene("merom-house")
    table = copy.deepcopy(next(e for e in scene["entities"] if e["id"] == "w:prep_obj_instance"))
    assert len(table["body"]["links"][0]["geometry"]["collision"]) == 12
    x, y, _ = table["pose"]["value"]["p"]
    probes = []
    for index, (dx, dy) in enumerate([(-0.25, -0.3), (0, 0), (0.25, 0.3)]):
        probes.append({"id": f"w:probe-{index}", "role": "object", "category": {"value": "sphere"},
                       "pose": {"value": {"p": [x + dx, y + dy, 1.1], "q": [1, 0, 0, 0]}},
                       "body": {"kind": "rigid", "root": {"value": "free"}, "joints": [], "links": [
                           {"id": "base", "geometry": {"collision": [
                               {"kind": "primitive", "shape": "sphere", "size": [0.03],
                                "pose": {"p": [0, 0, 0], "q": [1, 0, 0, 0]}}]}}]}})
    sim = WorldSim({"mode": "world", "entities": [table, *probes]})
    try:
        sim.step(1500)
        np.testing.assert_allclose(sim.data.qpos.reshape(-1, 7)[:, 2], 0.73574 + 0.03, atol=0.002)
        assert sim.data.ncon == 3 and not np.any(sim.data.warning.number)
        # At floor level the table centre is open; the former solid box filled this clearance.
        sim.data.qpos[7:10] = [x, y, 0.2]
        mujoco.mj_forward(sim.model, sim.data)
        name = sim.build.names["w:probe-1"].collision_geoms[0]
        gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert all(gid not in (contact.geom1, contact.geom2) for contact in sim.data.contact)
    finally:
        sim.stop()


@pytest.mark.parametrize("body", ["cobot-magic", "steward-bot"])
def test_robot_starts_and_moves_in_living_room_without_environment_contacts(tmp_path, body):
    service = AuthoringService(tmp_path)
    service.ensure_scenario("merom-house")
    scenario = with_body(service.load_scenario("merom-house"), body)
    assert scenario["robots"] == ["w:robot-1"]
    sim = WorldSim(scenario["sceneState"])
    try:
        frame = sim.sensors("w:robot-1")
        assert {"table", "sofa", "plate", "bread"} <= {d["category"] for d in frame["detections"]}
        estimator = EgoEstimator(robot_model=robot_entity(body))
        estimator.update(frame)
        observer = Observer(lambda: sim)
        record = observer.ingest("w:robot-1", estimator.to_scene().to_dict())
        assert record["gtObjects"] == 105 and record["matchedObjects"] > 0
        assert record["recall"] > 0 and record["meanPositionErrorM"] < 0.1
        assert not any("floors_" in name for name in record["matches"].values())
        names = sim.build.names["w:robot-1"]
        robot_bodies = set(entity_body_ids(sim.model, names))
        addresses = [sim.model.jnt_qposadr[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                     for name in names.root_joints]
        np.testing.assert_allclose(sim.data.qpos[addresses], [1.7, 7.2, math.pi / 2], atol=1e-6)
        for target in ([1.7, 7.2, math.pi / 2], [1.7, 7.45, math.pi / 2]):
            sim.command("w:robot-1", {"base": target})
            for _ in range(100):
                assert all((int(sim.model.geom_bodyid[c.geom1]) in robot_bodies)
                           == (int(sim.model.geom_bodyid[c.geom2]) in robot_bodies)
                           or c.dist >= -0.001 for c in sim.data.contact)
                sim.step(10)
            np.testing.assert_allclose(sim.data.qpos[addresses], target, atol=0.002)
        assert not np.any(sim.data.warning.number)
        sim.reset()
        np.testing.assert_allclose(sim.data.qpos[addresses], [1.7, 7.2, math.pi / 2], atol=1e-6)
    finally:
        sim.stop()


def test_perception_reports_first_frame_after_start_and_episode_change(monkeypatch):
    frames = iter([(1, i / 10) for i in range(6)] + [(2, 0.0)])
    reported, released = [], []

    class Link:
        def wait_until(self, *_args, **_kwargs):
            self.episode, self.stamp = next(frames)
            return self.stamp

        def telemetry(self, scene):
            reported.append((scene["episode"], scene["stamp"]))

        def release(self, name):
            released.append(name)

    link = Link()
    runtime = EgoRuntime(link, robot_entity("steward-bot"))

    def perceive():
        runtime._episode = link.episode
        if link.episode == 2:
            runtime._running = False

    monkeypatch.setattr(runtime, "_poll_once", perceive)
    monkeypatch.setattr(runtime, "_holding", lambda _name: nullcontext())
    monkeypatch.setattr(runtime, "_telemetry_scene", lambda: {"episode": link.episode, "stamp": link.stamp})
    runtime._running = True
    runtime._loop()
    assert runtime.last_error is None
    assert reported == [(1, 0.0), (1, 0.4), (2, 0.0)]
    assert released == ["perception"]


def test_passive_perception_can_wait_past_the_motion_timeout(monkeypatch):
    elapsed = [0.0]
    attempts = []

    def wait(_method, _url, body, **_kwargs):
        attempts.append(body)
        if len(attempts) == 1:
            elapsed[0] += 61.0
            raise HttpError(504, "world is idle")
        return {"t": 0.1}

    monkeypatch.setattr("steward.ego.runtime.call", wait)
    monkeypatch.setattr("steward.ego.runtime.time.monotonic", lambda: elapsed[0])
    link = RobotLink("http://unused")
    assert link.wait_until(0.1, "perception", pulls=False) == 0.1
    assert len(attempts) == 2 and all(not b["pulls"] for b in attempts)
    attempts.clear()
    with pytest.raises(SkillFailed, match="timed out"):
        link.wait_until(0.1, "motion")
    attempts.clear()
    link.closed = True
    with pytest.raises(HttpError):
        link.wait_until(0.1, "perception", pulls=False)


@pytest.fixture
def mesh_scene(tmp_path):
    vertices = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4]], dtype="<f4")
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<i4")
    mesh = tmp_path / "tetra.msh"
    mesh.write_bytes(struct.pack("<4i", 4, 0, 0, 4) + vertices.tobytes() + faces.tobytes())
    geom = {"kind": "mesh", "ref": str(mesh), "pose": {"p": [0.3, 0.1, 0.2], "q": [1, 0, 0, 0]},
            "rgba": [0.8, 0.5, 0.2, 1]}
    return {"mode": "world", "entities": [{"id": "w:mesh", "role": "object", "category": {"value": "tetra"},
             "pose": {"value": {"p": [1, 2, 3], "q": [1, 0, 0, 0]}}, "body": {
                 "kind": "rigid", "root": {"value": "free"}, "joints": [], "links": [{
                     "id": "base", "inertial": {"value": {"mass": 2.0, "diagonal": [0.02, 0.03, 0.04],
                         "pose": {"p": [0.1, 0.2, 0.3], "q": [1, 0, 0, 0]}}},
                     "geometry": {"collision": [geom], "visual": [copy.deepcopy(geom)]}}]}}]}


def test_mesh_assets_are_embedded_once_and_keep_explicit_inertia(mesh_scene):
    assert validate_world(mesh_scene) == []
    build = build_mjcf(SceneState.from_dict(mesh_scene))
    assert len(build.assets) == 1
    model = build.compile()
    data = mujoco.MjData(model)
    build.apply_initial(model, data)
    assert model.nmesh == 1 and model.ngeom == 2
    assert model.body_mass[1] == 2
    np.testing.assert_allclose(model.body_inertia[1], [0.02, 0.03, 0.04])
    np.testing.assert_allclose(model.body_ipos[1], [0.1, 0.2, 0.3])
    assert model.geom_contype.tolist() == [1, 0]
    mujoco.mj_step(model, data, nstep=100)
    assert not np.any(data.warning.number)


def test_mesh_bounds_contain_transformed_vertices(mesh_scene):
    sim = WorldSim(mesh_scene)
    try:
        box = sim.entity_obb("w:mesh")
        low = np.array(box.center) - np.array(box.size) / 2
        high = np.array(box.center) + np.array(box.size) / 2
        vertices = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.3, 0], [0, 0, 0.4]]) + [1.3, 2.1, 3.2]
        assert np.all(vertices >= low - 1e-6) and np.all(vertices <= high + 1e-6)
        assert np.all(np.array(box.size) < 1.0)
    finally:
        sim.stop()


@pytest.mark.parametrize("inertia", [None, {"mass": 2}, {"mass": -1, "diagonal": [1, 1, 1]}])
def test_mesh_collision_requires_valid_explicit_inertia(mesh_scene, inertia):
    mesh_scene["entities"][0]["body"]["links"][0]["inertial"] = None if inertia is None else {"value": inertia}
    assert any("inertial" in error for error in validate_world(mesh_scene))
