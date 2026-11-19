"""Replay recorded wrist endpoints against MoveIt IK without sending motion goals."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from moveit_msgs.srv import GetPositionIK

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pick_place import PickPlaceController, rotation
from ros_bridge import ROSBridge


def probe(scene, controller, mount, shift):
    events = json.loads((scene / "camera/manipulation.json").read_text())
    view = next(event["target"] for event in events if event["event"] == "source_overview")
    camera = json.loads((scene / "evidence/camera.json").read_text())
    plan = json.loads((scene / "evidence/geometry.json").read_text())["plan"]

    # Use the commanded overview as an approximation to the achieved flange pose.
    downward = np.diag([1., -1., -1.])
    base_from_link = np.eye(4)
    base_from_link[:3, :3] = downward
    base_from_link[:3, 3] = view
    odom_from_base = np.array(camera["odom_from_camera"]) @ np.linalg.inv(base_from_link @ mount)
    row = {
        "scene": scene.name,
        "estimated_base_odom_xyz": odom_from_base[:3, 3].tolist(),
        "yaw_rad": math.atan2(odom_from_base[1, 0], odom_from_base[0, 0]),
    }
    shorter_hover = [*plan["grasp"][:2], plan["grasp"][2] + .12]
    endpoints = [
        ("original_hover", plan["lift"], (0, 0)),
        ("shorter_hover", shorter_hover, (0, 0)),
        ("original_grasp", plan["grasp"], (0, 0)),
        ("shifted_hover", plan["lift"], shift),
        ("shifted_grasp", plan["grasp"], shift),
    ]
    # All candidate docks have the same heading. Convert the downward orientation
    # from odom to base_link, using the quaternion branch for a half-turn about X.
    r = odom_from_base[:3, :3].T @ downward
    qx = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) / 2
    if qx < .5:
        raise ValueError("Expected Table A approach orientation")
    for label, point, offset in endpoints:
        base = odom_from_base.copy()
        base[:2, 3] += offset
        xyz = (np.linalg.inv(base) @ [*point, 1.])[:3]
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = "tm12S_planninggroup"
        ik.ik_link_name = "link_6"
        ik.robot_state.is_diff = True
        ik.avoid_collisions = False
        ik.timeout.sec = 2
        ik.pose_stamped.header.frame_id = "base_link"
        pose = ik.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
        pose.orientation.x = qx
        pose.orientation.y = float((r[0, 1] + r[1, 0]) / (4 * qx))
        pose.orientation.z = float((r[0, 2] + r[2, 0]) / (4 * qx))
        pose.orientation.w = float((r[2, 1] - r[1, 2]) / (4 * qx))
        response = controller.service(GetPositionIK, "/compute_ik", request)
        row[label] = {"xyz_in_base": xyz.tolist(), "ik_error_code": response.error_code.val}
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--shift", nargs=2, type=float, required=True, metavar=("DX", "DY"))
    args = parser.parse_args()
    with ROSBridge() as bridge:
        controller = PickPlaceController(bridge, {})
        tf = controller.transform("link_6", "camera_optical_frame")
        mount = np.eye(4)
        mount[:3, :3] = rotation(tf.rotation)
        mount[:3, 3] = [tf.translation.x, tf.translation.y, tf.translation.z]
        for scene in args.scenes:
            print(json.dumps(probe(scene, controller, mount, args.shift)), flush=True)
