"""Sensor-based, contact-only 50mm cube and upright cylinder transfer in configured table stations.

No simulator pose/manifest/seed input is accepted. MoveIt attachments affect its
collision model only; Gazebo holds objects through finger contact and friction.
"""
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from ros_bridge import CameraUnavailableError, LocationPose


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def point_cloud(rgb, depth, info, transform, encoding):
    if encoding not in ("32FC1", "16UC1"):
        raise ValueError("Unsupported depth encoding")
    distance = np.asarray(depth, dtype=float) / (1000 if encoding == "16UC1" else 1)
    if distance.shape != rgb.shape[:2]:
        raise ValueError("RGB/depth dimensions differ")
    fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
    if min(fx, fy) <= 0:
        raise ValueError("Camera is not calibrated")
    v, u = np.indices(distance.shape)
    points = np.stack(((u-cx)*distance/fx, (v-cy)*distance/fy, distance), axis=-1)
    t = transform.translation
    with np.errstate(invalid="ignore"):
        points = points @ rotation(transform.rotation).T + [t.x, t.y, t.z]
    valid = np.isfinite(points).all(axis=2) & (distance > .08) & (distance < 3)
    return points, valid


def color_mask(bgr, color):
    channel = {"blue": 0, "green": 1, "red": 2}[color]
    image = bgr.astype(float)
    primary = image[:, :, channel]
    return (primary > 80) & (primary > 1.8 * image[:, :, (channel+1)%3]) & (primary > 1.8 * image[:, :, (channel+2)%3])


def held_center(transform):
    # With the flange pointing down, its positive local Z axis points toward
    # the fingers. Negative local Z would incorrectly predict a cube above it.
    return rotation(transform.rotation) @ [0., 0., .222] + [
        transform.translation.x, transform.translation.y, transform.translation.z]


def table_region(points, valid, station):
    x, y, z = station.center
    return (valid & (abs(points[:, :, 0]-x) < .245) & (abs(points[:, :, 1]-y) < .32)
            & (points[:, :, 2] > z-.08) & (points[:, :, 2] < z+.18))


def fit_table(points, mask):
    cloud = points[mask]
    if len(cloud) < 300:
        raise ValueError("Insufficient observed tabletop")
    # Dominant surface, with coloured objects excluded by the caller.
    hist, edges = np.histogram(cloud[:, 2], bins=np.arange(.65, 1.06, .01))
    middle = (edges[np.argmax(hist)]+edges[np.argmax(hist)+1])/2
    cloud = cloud[abs(cloud[:, 2]-middle) < .015]
    if len(cloud) < 300:
        raise ValueError("Insufficient plane support")
    return np.linalg.lstsq(np.c_[cloud[:, :2], np.ones(len(cloud))], cloud[:, 2], rcond=None)[0]


def free_place(points, valid, station, plane, base_xy=None):
    """Accept only directly observed empty surface patches, never unseen slots."""
    cloud = points[table_region(points, valid, station)]
    x, y, _ = station.center
    offsets = [(dx, dy) for dy in (0., .12, -.12) for dx in (.06, -.10)]
    if base_xy is not None:
        offsets.sort(key=lambda d: math.hypot(x+d[0]-base_xy[0], y+d[1]-base_xy[1]))
    offsets.extend([(.06, .16), (.06, -.16), (-.10, .18), (-.10, -.18)])
    for dx, dy in offsets:
        px, py = x+dx, y+dy
        support = cloud[(abs(cloud[:, 0]-px) < .055) & (abs(cloud[:, 1]-py) < .055)]
        if len(support) < 80:
            continue
        height = support[:, 2] - (plane[0]*support[:, 0]+plane[1]*support[:, 1]+plane[2])
        if np.mean(abs(height) < .012) < .96 or np.max(height) > .025:
            continue
        # Require observed support across the intended cube footprint.
        if np.ptp(support[:, 0]) < .075 or np.ptp(support[:, 1]) < .075:
            continue
        return [float(px), float(py), float(plane[0]*px+plane[1]*py+plane[2])]
    raise ValueError("No observed clear destination patch")


def locate_target(points, valid, bgr, boxes, station, color):
    if len(boxes) != 1:
        raise ValueError("Exactly one unambiguous target box is required")
    x1, y1, x2, y2 = boxes[0]
    height, width = bgr.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid pixel box")
    box_mask = np.zeros((height, width), dtype=bool)
    box_mask[y1:y2, x1:x2] = True
    cloud = points[box_mask & color_mask(bgr, color) & table_region(points, valid, station)]
    if len(cloud) < 35:
        raise ValueError("LLM box lacks registered target-color RGB-D evidence")
    top_z = np.percentile(cloud[:, 2], 90)
    top = cloud[cloud[:, 2] > top_z-.009]
    if len(top) < 15:
        raise ValueError("Insufficient flat top surface")
    extent = np.percentile(top[:, :2], 98, axis=0)-np.percentile(top[:, :2], 2, axis=0)
    if not np.all((extent > .025) & (extent < .075)):
        raise ValueError("Target is not consistent with the configured 50mm footprint")
    xy = (np.percentile(top[:, :2], 2, axis=0)+np.percentile(top[:, :2], 98, axis=0))/2
    return [float(xy[0]), float(xy[1]), float(np.median(top[:, 2])-.025)]


@dataclass(frozen=True)
class TableStation:
    center: tuple
    view_pose: tuple
    pickup_pose: LocationPose | None = None


def load_stations(path):
    data = json.loads(Path(path).read_text())
    result = {}
    for name, value in data.items():
        center, view = tuple(value["center"]), tuple(value["view_pose"])
        if len(center) != 3 or len(view) != 3 or not all(math.isfinite(v) for v in (*center, *view)):
            raise ValueError("Station center/view must each contain three finite coordinates")
        key = " ".join(name.casefold().split())
        if not key or key in result:
            raise ValueError("Invalid or duplicate station")
        pickup = LocationPose(**value["pickup_pose"]) if "pickup_pose" in value else None
        result[key] = TableStation(center, view, pickup)
    return result


class PickPlaceController:
    """One task, one bridge, explicit graph stages. Coordinates are in odom."""
    def __init__(self, bridge, stations, output_dir=None):
        from tf2_ros import Buffer, TransformListener
        self.bridge = bridge
        self.stations = stations
        self.output = Path(output_dir) if output_dir else None
        self.tf = Buffer(node=bridge._node)
        self.listener = TransformListener(self.tf, bridge._node)
        self.events = []
        self.original_acm = None
        self.attached = False
        self.source = self.destination = None
        self.observation = None
        self.plan = None
        self.slot = None
        self.object_id = "agent_grasp_target"

    def event(self, name, **values):
        record = {"event": name, "wall_time": time.time(), **values}
        self.events.append(record)
        print(json.dumps(record), flush=True)
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output/"manipulation.json").write_text(json.dumps(self.events, indent=2)+"\n")

    def wait(self, future, timeout=15):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout):
            raise TimeoutError("ROS service timed out")
        return future.result()

    def service(self, service_type, name, request):
        client = self.bridge._node.create_client(service_type, name)
        try:
            if not client.wait_for_service(timeout_sec=10):
                raise RuntimeError(f"Service unavailable: {name}")
            return self.wait(client.call_async(request))
        finally:
            self.bridge._node.destroy_client(client)

    def apply_scene(self, scene):
        from moveit_msgs.srv import ApplyPlanningScene
        if not self.service(ApplyPlanningScene, "/apply_planning_scene", ApplyPlanningScene.Request(scene=scene)).success:
            raise RuntimeError("Planning scene update rejected")

    def transform(self, target, source, stamp=None):
        import rclpy
        query = rclpy.time.Time.from_msg(stamp) if stamp else rclpy.time.Time()
        deadline = time.monotonic()+5
        while True:
            try:
                return self.tf.lookup_transform(target, source, query).transform
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)

    def snapshot(self, after=None, label="frame"):
        deadline = time.monotonic()+15
        while True:
            try:
                snap = self.bridge.camera_snapshot(received_after=after)
                bgr, depth, info = snap.to_arrays()
                transform = self.transform("odom", snap.rgb.header.frame_id, snap.rgb.header.stamp)
                points, valid = point_cloud(bgr, depth, info, transform, snap.depth.encoding)
                if self.output:
                    from PIL import Image
                    self.output.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(bgr[:, :, ::-1]).save(self.output/f"{label}.png")
                return snap, bgr, points, valid
            except CameraUnavailableError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.05)

    def remember_frame(self, snapshot, bgr, depth, info):
        transform = self.transform("odom", snapshot.rgb.header.frame_id, snapshot.rgb.header.stamp)
        points, valid = point_cloud(bgr, depth, info, transform, snapshot.depth.encoding)
        self.observation = (bgr.copy(), points, valid)
        if self.output:
            from PIL import Image
            self.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(bgr[:, :, ::-1]).save(self.output/"source-model-input.png")

        matrix = np.eye(4)
        matrix[:3,:3] = rotation(transform.rotation)
        matrix[:3,3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        return {"odom_from_camera": matrix.tolist()}

    def station(self, name):
        return self.stations[" ".join(name.casefold().split())]

    def source_view_pose(self):
        """Aim the wrist optical axis at the known table, independent of Nav2 parking error."""
        base_from_odom = self.transform("base_link", "odom")
        mount = self.transform("link_6", "camera_optical_frame")
        t = base_from_odom.translation
        table = rotation(base_from_odom.rotation) @ self.source.center + [t.x, t.y, t.z]
        # move_arm requests the downward flange orientation (xyzw = 1,0,0,0) in base_link.
        down = np.diag([1., -1., -1.])
        offset = down @ [mount.translation.x, mount.translation.y, mount.translation.z]
        axis = down @ rotation(mount.rotation)[:, 2]
        height = self.source.view_pose[2]
        if axis[2] >= -.5:
            raise ValueError("Configured wrist view is not directed toward the table")
        distance = (table[2] - height - offset[2]) / axis[2]
        near_center = table[:2] - .06 * table[:2] / np.linalg.norm(table[:2])
        xy = near_center - offset[:2] - distance*axis[:2]
        return [float(xy[0]), float(xy[1]), height]

    def add_tables(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from shape_msgs.msg import SolidPrimitive
        scene = PlanningScene(); scene.is_diff = True
        for name, station in self.stations.items():
            obj = CollisionObject(); obj.id = "station_"+name; obj.header.frame_id = "odom"
            obj.operation = CollisionObject.ADD
            primitive = SolidPrimitive(); primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [.5, .65, station.center[2]]
            pose = Pose(); pose.orientation.w = 1.
            pose.position.x, pose.position.y, pose.position.z = station.center[0], station.center[1], station.center[2]/2
            obj.primitives = [primitive]; obj.primitive_poses = [pose]
            scene.world.collision_objects.append(obj)
        self.apply_scene(scene)

    def prepare(self, source, destination, target):
        words = target.casefold().strip().removeprefix("the ").split()
        if len(words) != 2 or words[0] not in ("red", "green", "blue") or words[1] not in ("cube", "cylinder"):
            raise ValueError("Expected a configured 50mm colored cube or upright cylinder")
        self.color, self.shape = words
        self.source, self.destination = self.station(source), self.station(destination)
        if self.source == self.destination:
            raise ValueError("Source and destination must differ")
        # These calibrated station poses use the initialized odometry/map origin.
        transform = self.transform("map", "odom")
        if math.hypot(transform.translation.x, transform.translation.y) > .04:
            raise ValueError("Configured table stations require the fixed known map; map/odom drift exceeds 4cm")
        self.add_tables()

    def inspect_destination(self):
        _, bgr, points, valid = self.snapshot(time.monotonic(), "destination-before")
        plane = fit_table(points, table_region(points, valid, self.destination) & ~color_mask(bgr, self.color))
        base = self.transform("odom", "base_link").translation
        self.slot = free_place(points, valid, self.destination, plane, (base.x, base.y))
        self.event("destination_clear_patch", position=self.slot, plane=plane.tolist())
        return self.slot

    def plan_grasp(self, boxes):
        if self.observation is None:
            raise ValueError("No registered source frame")
        bgr, points, valid = self.observation
        center = locate_target(points, valid, bgr, boxes, self.source, self.color)
        # Center the contact patch below the top rim of the 50 mm target.
        grasp_offset = .212 if self.shape == "cylinder" else .222
        self.plan = {"color": self.color, "shape": self.shape, "center": center, "grasp": [*center[:2], center[2]+grasp_offset],
                     "lift": [*center[:2], center[2]+.395]}
        self.event("rgbd_grasp_plan", **self.plan)
        return self.plan

    def target_scene_object(self, center):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        obj = CollisionObject(); obj.id = self.object_id; obj.header.frame_id = "odom"
        obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        if self.shape == "cylinder":
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [.05, .025]  # MoveIt cylinder height and radius, meters
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [.05]*3
        pose = Pose(); pose.orientation.w = 1.
        pose.position.x, pose.position.y, pose.position.z = center
        obj.primitives = [primitive]; obj.primitive_poses = [pose]
        return obj

    def allow_contact(self):
        from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene, PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        self.original_acm = copy.deepcopy(self.service(GetPlanningScene, "/get_planning_scene", request).scene.allowed_collision_matrix)
        matrix = copy.deepcopy(self.original_acm)
        touch = [f"{side}_{part}" for side in ("left", "right") for part in
                 ("outer_knuckle", "outer_finger", "inner_knuckle", "inner_finger", "inner_finger_pad")]
        touch += ["station_"+name for name in self.stations]
        for name in [self.object_id, *touch]:
            if name not in matrix.entry_names:
                matrix.entry_names.append(name)
                for row in matrix.entry_values: row.enabled.append(False)
                matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
        index = matrix.entry_names.index(self.object_id)
        for name in touch:
            other = matrix.entry_names.index(name)
            matrix.entry_values[index].enabled[other] = matrix.entry_values[other].enabled[index] = True
        scene = PlanningScene(); scene.is_diff = True; scene.allowed_collision_matrix = matrix
        scene.world.collision_objects = [self.target_scene_object(self.plan["center"])]
        self.apply_scene(scene)

    def attach(self):
        from moveit_msgs.msg import AttachedCollisionObject, PlanningScene
        obj = self.target_scene_object(self.plan["center"])
        tf = self.transform("odom", "link_6")
        center = rotation(tf.rotation).T @ (np.array(self.plan["center"])-[tf.translation.x, tf.translation.y, tf.translation.z])
        obj.header.frame_id = "link_6"
        p = obj.primitive_poses[0]; p.position.x, p.position.y, p.position.z = map(float, center)
        # Square cube orientation is equivalent under a half-turn about its vertical axis.
        attached = AttachedCollisionObject(); attached.link_name = "link_6"; attached.object = obj
        attached.touch_links = ["link_6", "flange", "robotiq_140_base_link"] + [
            f"{side}_{part}" for side in ("left", "right") for part in
            ("outer_knuckle", "outer_finger", "inner_knuckle", "inner_finger", "inner_finger_pad")]
        scene = PlanningScene(); scene.is_diff = True; scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        self.apply_scene(scene); self.attached = True

    def verify_held(self, label):
        _, bgr, points, valid = self.snapshot(time.monotonic(), label)
        tf = self.transform("odom", "link_6")
        expected = held_center(tf)
        cloud = points[valid & color_mask(bgr, self.color)]
        nearby = cloud[np.linalg.norm(cloud-expected, axis=1) < .09]
        self.event("held_sensor_evidence", color=self.color, target_pixels=len(cloud), nearby_pixels=len(nearby),
                   expected=expected.tolist(),
                   nearest_distance=float(np.min(np.linalg.norm(cloud-expected, axis=1))) if len(cloud) else None,
                   median_target_position=np.median(cloud, axis=0).tolist() if len(cloud) else None)
        if len(nearby) < 15 or np.median(nearby[:, 2]) < self.plan["center"][2]+.10:
            raise RuntimeError("Fresh wrist RGB-D does not confirm the lifted target near the fingers")
        self.event("held_verified", pixels=len(nearby), measured=np.median(nearby, axis=0).tolist())
        return True

    def detach(self):
        x, y, plane = self.slot
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
        scene = PlanningScene(); scene.is_diff = True; scene.robot_state.is_diff = True
        attached = AttachedCollisionObject(); attached.link_name = "link_6"
        attached.object.id = self.object_id; attached.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects = [attached]
        scene.world.collision_objects = [self.target_scene_object([x, y, plane+.025])]
        self.apply_scene(scene); self.attached = False

    def restore_collisions(self):
        from moveit_msgs.msg import PlanningScene
        if self.original_acm is not None:
            scene = PlanningScene()
            scene.is_diff = True
            scene.allowed_collision_matrix = self.original_acm
            self.apply_scene(scene)

    def verify_place(self):
        _, bgr, points, valid = self.snapshot(time.monotonic(), "destination-after")
        cloud = points[valid & color_mask(bgr, self.color)]
        x, y, plane = self.slot
        selected = cloud[(np.linalg.norm(cloud[:, :2]-[x, y], axis=1) < .065) &
                         (cloud[:, 2] > plane+.025) & (cloud[:, 2] < plane+.085)]
        if len(selected) < 35:
            raise RuntimeError("Fresh destination RGB-D does not verify the released target")
        self.event("placement_verified", pixels=len(selected), measured=np.median(selected, axis=0).tolist())
        return {"verified": True, "pixels": len(selected)}
