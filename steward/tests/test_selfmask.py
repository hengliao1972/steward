"""The robot masks itself: its own kinematics say which pixels are its body.

A real segmenter does not know what the robot looks like, so the robot projects its own links
into the camera and drops those pixels. The simulation can check this exactly, because its
renderer reports which geom every pixel belongs to.
"""

import numpy as np
import pytest

from steward.assets.cobot import cobot_magic
from steward.assets.generators import steward_bot
from steward.ego.estimator import EgoEstimator
from steward.ego.selfmask import camera_model, project, self_pixels
from steward.world.sim import WorldSim

from test_cobot_skills import _scene


@pytest.fixture()
def rig():
    sim = WorldSim(_scene())
    sim.start()
    est = EgoEstimator(robot_model=cobot_magic().to_dict())
    try:
        yield sim, est
    finally:
        sim.stop()


def _robot_pixels(sim: WorldSim, robot_id: str, sensor_id: str, entity_id: str | None = None) -> np.ndarray:
    """Ground truth: the pixels whose geom belongs to the robot, from the renderer."""
    import mujoco

    _, sensor = sim.camera_pose(robot_id, sensor_id)
    width, height = sensor["intrinsics"]["width"], sensor["intrinsics"]["height"]
    renderer = mujoco.Renderer(sim.model, height, width)
    try:
        renderer.update_scene(sim.data, camera=sim.robot_camera_name(robot_id, sensor_id))
        renderer.enable_segmentation_rendering()
        seg = renderer.render().copy()
    finally:
        renderer.close()
    geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
    bodies = [b for b, entity in sim._body_entity.items() if entity == (entity_id or robot_id)]
    owner = np.isin(sim.model.geom_bodyid[np.clip(geom, 0, None)], bodies)
    return owner & (geom >= 0)


def test_mask_covers_the_body_it_can_see(rig) -> None:
    sim, est = rig
    est.update(sim.sensors("w:cobot"))
    pose, sensor = sim.camera_pose("w:cobot", "wide_camera")
    mask = self_pixels(est.robot_model, est.link_pose, pose, sensor["intrinsics"])
    truth = _robot_pixels(sim, "w:cobot", "wide_camera")
    assert truth.sum() > 5000, "the overhead camera must see a substantial part of the twin"
    recall = float((mask & truth).sum()) / float(truth.sum())
    assert recall > 0.95, f"the mask missed {100 * (1 - recall):.1f}% of the robot"


def test_mask_does_not_swallow_the_tableware(rig) -> None:
    sim, est = rig
    est.update(sim.sensors("w:cobot"))
    pose, sensor = sim.camera_pose("w:cobot", "wide_camera")
    mask = self_pixels(est.robot_model, est.link_pose, pose, sensor["intrinsics"])
    for entity in ("w:bowl-1", "w:plate-1"):
        where = sim.world_state() and next(e for e in sim.world_state()["entities"] if e["id"] == entity)["pose"]["value"]["p"][:2]
        uv, _ = project(np.asarray([[where[0], where[1], 0.78]]), camera_model(pose, sensor["intrinsics"]))
        u, v = int(round(uv[0][0])), int(round(uv[0][1]))
        assert not mask[v, u], f"{entity} falls inside the robot's self mask"


def test_mask_follows_the_arm(rig) -> None:
    sim, est = rig
    pose, sensor = sim.camera_pose("w:cobot", "wide_camera")
    est.update(sim.sensors("w:cobot"))
    before = self_pixels(est.robot_model, est.link_pose, pose, sensor["intrinsics"])
    sim.command("w:cobot", {"joints": {"L_joint2": 0.3, "L_joint3": -2.4}})
    sim.step(400)
    est.update(sim.sensors("w:cobot"))
    after = self_pixels(est.robot_model, est.link_pose, pose, sensor["intrinsics"])
    changed = int((before ^ after).sum())
    assert changed > 2000, f"moving the arm barely changed the silhouette ({changed} px)"
    truth = _robot_pixels(sim, "w:cobot", "wide_camera")
    recall = float((after & truth).sum()) / float(truth.sum())
    assert recall > 0.95, f"after the move the mask missed {100 * (1 - recall):.1f}% of the robot"


def test_the_mobile_robot_masks_itself_too(kitchen_scene) -> None:
    """The same projection works for the kitchen robot's head camera and its lift/reach arm.

    Its head looks 30 degrees down, so parts of its own body are in frame at some postures. The
    check is agreement with the renderer either way: mask non-empty where the body is visible, and
    no stray pixels where it is not."""
    sim = WorldSim(kitchen_scene)
    sim.start()
    try:
        est = EgoEstimator(robot_model=steward_bot().to_dict())
        est.update(sim.sensors("w:robot-1"))
        pose, sensor = sim.camera_pose("w:robot-1", "head_camera")
        mask = self_pixels(est.robot_model, est.link_pose, pose, sensor["intrinsics"])
        truth = _robot_pixels(sim, "w:robot-1", "head_camera")
        if truth.sum() > 200:
            recall = float((mask & truth).sum()) / float(truth.sum())
            assert recall > 0.90, f"the mask missed {100 * (1 - recall):.1f}% of the robot"
        else:
            assert mask.sum() < 0.02 * mask.size, "nothing is in frame, so the mask must be empty"
    finally:
        sim.stop()
