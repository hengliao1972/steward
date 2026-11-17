"""Robot-owned forward kinematics and gripper transforms from synchronized encoder packets."""
import math

import numpy as np

from steward.math3d import matrix_from_quat, pose_compose, quat_from_axis_angle

CAMERA_AXES = "opencv_x_right_y_down_z_forward"


def _link_pose(model: dict, encoders: dict, link: str) -> dict:
    tree = model["body"]
    child_links = {joint["child"] for joint in tree["joints"]}
    root_link = next(item["id"] for item in tree["links"] if item["id"] not in child_links)
    chain = []
    while link != root_link:
        joint = next(j for j in tree["joints"] if j["child"] == link)
        chain.append(joint)
        link = joint["parent"]
    result = {"p": [0, 0, 0], "q": [1, 0, 0, 0]}
    for joint in reversed(chain):
        result = pose_compose(result, joint["origin"]["value"])
        kind = joint["type"]["value"]
        if kind == "fixed":
            continue
        if (joint["id"] not in encoders or not isinstance(encoders[joint["id"]], (int, float))
                or not math.isfinite(encoders[joint["id"]])):
            raise ValueError(f"Missing or invalid joint encoder: {joint['id']}")
        value, axis = encoders[joint["id"]], joint["axis"]["value"]
        if kind == "prismatic":
            motion = {"p": (np.asarray(axis) * value).tolist(), "q": [1, 0, 0, 0]}
        elif kind == "revolute":
            rotation = quat_from_axis_angle(axis, value)
            anchor = np.asarray((joint.get("anchor") or {}).get("value", [0, 0, 0]))
            motion = {"p": (anchor - matrix_from_quat(rotation) @ anchor).tolist(), "q": rotation}
        else:
            raise ValueError(f"Unsupported robot kinematics joint: {kind}")
        result = pose_compose(result, motion)
    return result


def _matrix(pose: dict) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = matrix_from_quat(pose["q"])
    result[:3, 3] = pose["p"]
    return result


def reference_to_odometry(model: dict, packet: dict) -> np.ndarray:
    """Map middle OpenCV camera points into the robot's measured odometry frame at this sample."""
    odom = packet["odometry"]
    if any(type(odom.get(key)) not in (int, float) or not math.isfinite(odom[key]) for key in ("x", "y", "yaw")):
        raise ValueError("Missing or invalid synchronized robot odometry")
    camera_link, camera = next((link["id"], sensor) for link in model["body"]["links"]
                               for sensor in link.get("sensors", []) if sensor["id"] == "head_camera")
    camera_base = _matrix(pose_compose(_link_pose(model, packet["joint_encoders"], camera_link), camera["mount"]))
    flu_from_cv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    yaw = odom["yaw"]
    base_odom = np.array([[math.cos(yaw), -math.sin(yaw), 0, odom["x"]],
                          [math.sin(yaw), math.cos(yaw), 0, odom["y"]], [0, 0, 1, 0], [0, 0, 0, 1]])
    return base_odom @ camera_base @ flu_from_cv


def grasp_link(model: dict) -> str:
    """The model's first grasp-point link (the kitchen robot's ``gripper``, the twin's ``L_link6``)."""
    return next(link["id"] for link in model["body"]["links"]
                if any((a if isinstance(a, str) else a.get("value")) == "grasp-point" for a in link.get("affordances", [])))


def gripper_pose(model: dict, packet: dict, gripper: str | None = None, joints: dict | None = None) -> dict:
    """Return gripper-link-to-base and gripper-link-to-camera matrices in metres.

    ``gripper`` is the grasp-point link to report (default: :func:`grasp_link`); whatever the body,
    its frame is called ``gripper`` and the link is named in ``gripper_link``. ``joint_positions``
    are ``joints`` (the arm's virtual joints: ``lift``, ``reach`` and the wrist) or, without them,
    the raw encoder readings. Body links use
    x forward, y left, z up at their zero pose; camera frames use OpenCV x right, y down, z
    forward. Only the robot body model, encoder sample and relative rig calibration contribute.
    No odometry or simulator absolute pose is required.
    """
    gripper = gripper or grasp_link(model)
    if (packet["reference_camera"] != "head_camera" or packet["camera_axes"] != CAMERA_AXES
            or packet["calibration_source"] != "rig_calibration"
            or not math.isfinite(packet["stamp"]) or packet["stamp"] < 0
            or "head_camera" not in packet["views"]):
        raise ValueError("Invalid robot kinematics packet")
    encoders = packet["joint_encoders"]
    gripper_to_base = _matrix(_link_pose(model, encoders, gripper))
    camera_link, camera = next((link["id"], sensor) for link in model["body"]["links"]
                               for sensor in link.get("sensors", []) if sensor["id"] == "head_camera")
    reference_to_base = _matrix(pose_compose(_link_pose(model, encoders, camera_link), camera["mount"]))
    flu_from_cv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    reference_to_base = reference_to_base @ flu_from_cv
    relative = {}
    for camera_id in packet["views"]:
        camera_to_reference = np.asarray(packet["views"][camera_id]["camera_to_reference"], dtype=float)
        if (camera_to_reference.shape != (4, 4) or not np.isfinite(camera_to_reference).all()
                or not np.allclose(camera_to_reference[3], [0, 0, 0, 1])
                or not np.allclose(camera_to_reference[:3, :3].T @ camera_to_reference[:3, :3], np.eye(3))
                or not np.isclose(np.linalg.det(camera_to_reference[:3, :3]), 1)):
            raise ValueError("Invalid camera rig calibration")
        relative[camera_id] = (np.linalg.inv(reference_to_base @ camera_to_reference) @ gripper_to_base).tolist()
    return {"stamp": packet["stamp"], "pose_source": "joint_encoders_and_robot_model",
            "gripper_frame": "gripper", "gripper_link": gripper, "gripper_axes": "x_forward_y_left_z_up",
            "camera_axes": CAMERA_AXES, "joint_positions": dict(encoders if joints is None else joints),
            "gripper_to_base": gripper_to_base.tolist(),
            "gripper_to_camera": relative}
