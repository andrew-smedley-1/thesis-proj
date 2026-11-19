"""Native ROS 2 motion primitives for the Jazzy MoMa simulation.

Importing this module requires only Python. Construct ROSBridge inside a sourced
ROS environment, and explicitly enable motion before sending any goals.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import copy
from dataclasses import asdict, dataclass, field
import importlib
import json
import logging
import math
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

_request_cancel = contextvars.ContextVar("ros_bridge_request_cancel", default=None)


@dataclass(frozen=True)
class BridgeConfig:
    navigation_action: str = "/navigate_to_pose"
    arm_action: str = "/move_action"
    gripper_action: str = "/robotiq_gripper_controller/gripper_cmd"
    planning_group: str = "tm12S_planninggroup"
    end_effector_link: str = "link_6"
    use_sim_time: bool = True
    server_timeout: float = 5.0
    goal_timeout: float = 120.0
    shutdown_timeout: float = 5.0
    rgb_topic: str = "/depth_camera/image"
    depth_topic: str = "/depth_camera/depth_image"
    camera_info_topic: str = "/depth_camera/camera_info"

    def __post_init__(self):
        for name in ("server_timeout", "goal_timeout", "shutdown_timeout"):
            _number(name, getattr(self, name), 0.001)
        for name in ("navigation_action", "arm_action", "gripper_action",
                     "planning_group", "end_effector_link", "rgb_topic", "depth_topic", "camera_info_topic"):
            _frame(getattr(self, name))


class UnknownLocationError(ValueError):
    pass


class CameraUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocationPose:
    x: float
    y: float
    yaw: float = 0.0
    frame_id: str = "map"

    def __post_init__(self):
        for name in ("x", "y", "yaw"):
            _number(name, getattr(self, name))
        _frame(self.frame_id)


def _location_key(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Location names must be nonempty strings")
    return " ".join(name.split()).casefold()


def _validate_locations(locations):
    if not isinstance(locations, dict):
        raise ValueError("Locations must be a JSON object mapping names to poses")
    validated = {}
    for name, pose in locations.items():
        key = _location_key(name)
        if key in validated:
            raise ValueError(f"Duplicate normalized location name: {name}")
        try:
            validated[key] = pose if isinstance(pose, LocationPose) else LocationPose(**pose)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pose for location {name!r}: {exc}") from exc
    return validated


def load_locations(path):
    """Load measured named poses; never synthesize coordinates for missing names."""
    return _validate_locations(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CameraSnapshot:
    """Owned copies of ROS messages; image encodings and ROS headers are preserved."""
    rgb: object
    depth: object
    camera_info: object

    def to_arrays(self):
        """Return (BGR image, native-unit depth array, CameraInfo) using cv_bridge."""
        try:
            from cv_bridge import CvBridge
        except ImportError as exc:
            raise RuntimeError("Array conversion requires ROS package ros-jazzy-cv-bridge") from exc
        bridge = CvBridge()
        return (bridge.imgmsg_to_cv2(self.rgb, desired_encoding="bgr8"),
                bridge.imgmsg_to_cv2(self.depth, desired_encoding="passthrough"),
                copy.deepcopy(self.camera_info))


@dataclass(frozen=True)
class MotionResult:
    status: str
    message: str
    ros_status: int | None = None
    details: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.status == "succeeded"

    def to_dict(self):
        return {**asdict(self), "success": self.success}


def _number(name, value, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return float(value)


def _frame(value):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise ValueError("Frame, action and group names must be nonempty without whitespace")
    return value


def _load_ros():
    modules = {
        "rclpy": "rclpy", "Context": "rclpy.context", "Node": "rclpy.node",
        "SingleThreadedExecutor": "rclpy.executors", "ActionClient": "rclpy.action",
        "Parameter": "rclpy.parameter", "NavigateToPose": "nav2_msgs.action",
        "MoveGroup": "moveit_msgs.action", "GripperCommand": "control_msgs.action",
        "Constraints": "moveit_msgs.msg", "PositionConstraint": "moveit_msgs.msg",
        "OrientationConstraint": "moveit_msgs.msg", "SolidPrimitive": "shape_msgs.msg",
        "Pose": "geometry_msgs.msg",
        "Image": "sensor_msgs.msg", "CameraInfo": "sensor_msgs.msg",
        "qos_profile_sensor_data": "rclpy.qos",
    }
    try:
        return SimpleNamespace(**{
            name: (module if name == "rclpy" else getattr(module, name))
            for name, path in modules.items()
            for module in [importlib.import_module(path)]
        })
    except ImportError as exc:
        raise RuntimeError(
            "ROS Jazzy Python packages unavailable. Source /opt/ros/jazzy/setup.bash "
            "and the simulation install/setup.bash; use a venv with --system-site-packages."
        ) from exc


@dataclass
class _Operation:
    kind: str
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    handle: object = None
    result: MotionResult | None = None


class ROSBridge:
    """One bridge-owned motion at a time; ROS callbacks run in a private thread.

    A timeout requests cancellation but keeps the motion slot occupied until a
    terminal result arrives. cancel() only affects goals issued by this bridge.
    """

    def __init__(self, config=None, *, enable_motion=False, locations=None):
        self.config = config or BridgeConfig()
        self._locations = _validate_locations({} if locations is None else locations)
        self.enable_motion = enable_motion
        self._ros = r = _load_ros()
        self._lock = threading.RLock()
        self._active = None
        self._closed = False
        self._camera = {}
        self._context = r.Context()
        r.rclpy.init(args=[], context=self._context)
        try:
            self._node = r.Node(
                "agent_ros_bridge", context=self._context,
                parameter_overrides=[r.Parameter("use_sim_time", value=self.config.use_sim_time)],
            )
            self._clients = {
                "navigation": r.ActionClient(self._node, r.NavigateToPose, self.config.navigation_action),
                "arm": r.ActionClient(self._node, r.MoveGroup, self.config.arm_action),
                "gripper": r.ActionClient(self._node, r.GripperCommand, self.config.gripper_action),
            }
            self._subscriptions = [
                self._node.create_subscription(message_type, topic,
                    lambda msg, stream=stream: self._camera_received(stream, msg),
                    r.qos_profile_sensor_data)
                for stream, message_type, topic in (
                    ("rgb", r.Image, self.config.rgb_topic),
                    ("depth", r.Image, self.config.depth_topic),
                    ("camera_info", r.CameraInfo, self.config.camera_info_topic))
            ]
            self._executor = r.SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
        except Exception:
            if hasattr(self, "_node"):
                self._node.destroy_node()
            self._context.try_shutdown()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        result = self.close()
        if result.status == "unconfirmed_stop":
            print(json.dumps(result.to_dict()), file=sys.stderr)

    def readiness(self):
        """Discovery only: readiness never sends a motion goal."""
        with self._lock:
            if self._closed:
                return {"closed": True, "ready": False}
            servers = {k: c.server_is_ready() for k, c in self._clients.items()}
            clock_ready = not self.config.use_sim_time or self._node.get_clock().now().nanoseconds > 0
            return {"ready": all(servers.values()) and clock_ready,
                    "servers": servers, "clock_ready": clock_ready,
                    "motion_enabled": self.enable_motion,
                    "camera": self.camera_status(),
                    "locations": sorted(self._locations),
                    "active": self._active.kind if self._active else None}

    def _camera_received(self, stream, message):
        # Keep callbacks cheap; array conversion happens in the caller's thread.
        with self._lock:
            if not self._closed:
                self._camera[stream] = (message, time.monotonic())

    def camera_snapshot(self, *, max_age=1.0, max_skew=0.1, received_after=None):
        """Get fresh latest RGB/depth plus calibration, or raise CameraUnavailableError.

        max_age uses wall-clock receive age; max_skew compares RGB/depth ROS stamps.
        This checks the latest pair, without waiting or claiming exact synchronization.
        CameraInfo can be static and is not expired by receive age.
        """
        _number("max_age", max_age, 0.001)
        _number("max_skew", max_skew, 0.0)
        if received_after is not None:
            _number("received_after", received_after, 0.0)
        with self._lock:
            if self._closed:
                raise CameraUnavailableError("Bridge is closed")
            missing = [s for s in ("rgb", "depth", "camera_info") if s not in self._camera]
            if missing:
                raise CameraUnavailableError(f"Missing camera streams: {', '.join(missing)}")
            now = time.monotonic()
            for stream in ("rgb", "depth"):
                if received_after is not None and self._camera[stream][1] <= received_after:
                    raise CameraUnavailableError(f"Waiting for post-navigation frame: {stream}")
                if now - self._camera[stream][1] > max_age:
                    raise CameraUnavailableError(f"Stale camera stream: {stream}")
            rgb, depth, info = (self._camera[s][0] for s in ("rgb", "depth", "camera_info"))
            stamp = lambda m: m.header.stamp.sec + m.header.stamp.nanosec / 1e9
            if abs(stamp(rgb) - stamp(depth)) > max_skew:
                raise CameraUnavailableError("Latest RGB/depth timestamps exceed max_skew")
            if depth.encoding not in ("16UC1", "32FC1"):
                raise CameraUnavailableError(f"Unsupported depth encoding: {depth.encoding}")
            if (rgb.width <= 0 or rgb.height <= 0 or
                    (rgb.width, rgb.height) != (depth.width, depth.height) or
                    (rgb.width, rgb.height) != (info.width, info.height) or
                    len(info.k) != 9 or not math.isfinite(info.k[0]) or info.k[0] <= 0 or
                    not math.isfinite(info.k[4]) or info.k[4] <= 0):
                raise CameraUnavailableError("Camera dimensions/calibration invalid or mismatched")
            return CameraSnapshot(copy.deepcopy(rgb), copy.deepcopy(depth), copy.deepcopy(info))

    def camera_status(self):
        """JSON-safe snapshot metadata; no image pixels are sent to the LLM."""
        try:
            snapshot = self.camera_snapshot()
        except CameraUnavailableError as exc:
            return {"ready": False, "status": "camera_unavailable", "message": str(exc)}
        return {"ready": True, "status": "ready", "width": snapshot.rgb.width,
                "height": snapshot.rgb.height, "rgb_encoding": snapshot.rgb.encoding,
                "depth_encoding": snapshot.depth.encoding,
                "depth_units": "meters" if snapshot.depth.encoding == "32FC1" else "millimeters",
                "frame_id": snapshot.rgb.header.frame_id}

    def lookup_location(self, name):
        """Return a measured LocationPose; case and repeated whitespace are ignored."""
        key = _location_key(name)
        if key not in self._locations:
            raise UnknownLocationError(f"Unknown location: {name!r}. Configured: {', '.join(sorted(self._locations)) or '(none)'}")
        return self._locations[key]

    def navigate_to_location(self, name):
        try:
            pose = self.lookup_location(name)
        except UnknownLocationError as exc:
            return MotionResult("unknown_location", str(exc))
        return self.send_navigation_goal(pose.x, pose.y, pose.yaw, pose.frame_id)

    def send_navigation_goal(self, x, y, yaw=0.0, frame_id="map"):
        """Navigate to meters and yaw radians in an explicit ROS frame."""
        x, y, yaw = (_number(n, v) for n, v in (("x", x), ("y", y), ("yaw", yaw)))
        frame_id = _frame(frame_id)
        goal = self._ros.NavigateToPose.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        return self._execute("navigation", goal)

    def move_arm_pose(self, x, y, z, qx, qy, qz, qw, frame_id="base_link", *, linear=False):
        """Plan and execute a link_6 pose. Coordinates are meters; quaternion is xyzw."""
        xyz = [_number(n, v) for n, v in zip(("x", "y", "z"), (x, y, z))]
        quat = [_number(n, v) for n, v in zip(("qx", "qy", "qz", "qw"), (qx, qy, qz, qw))]
        norm = math.hypot(*quat)
        if not math.isclose(norm, 1.0, abs_tol=0.01):
            raise ValueError("Arm quaternion must have unit length (within 0.01)")
        frame_id = _frame(frame_id)
        r = self._ros
        goal = r.MoveGroup.Goal()
        req = goal.request
        req.group_name = self.config.planning_group
        req.pipeline_id = "pilz_industrial_motion_planner" if linear else "ompl"
        req.planner_id = "LIN" if linear else ""
        req.num_planning_attempts = 5
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.15
        req.max_acceleration_scaling_factor = 0.15
        req.start_state.is_diff = True
        req.workspace_parameters.header.frame_id = "base_link"
        req.workspace_parameters.min_corner.x = -2.0
        req.workspace_parameters.min_corner.y = -2.0
        req.workspace_parameters.min_corner.z = -1.0
        req.workspace_parameters.max_corner.x = 2.0
        req.workspace_parameters.max_corner.y = 2.0
        req.workspace_parameters.max_corner.z = 3.0
        pose = r.Pose()
        pose.position.x, pose.position.y, pose.position.z = xyz
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [v / norm for v in quat]
        pc = r.PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = self.config.end_effector_link
        pc.weight = 1.0
        region = r.SolidPrimitive()
        region.type = r.SolidPrimitive.SPHERE
        region.dimensions = [0.005]
        pc.constraint_region.primitives = [region]
        pc.constraint_region.primitive_poses = [pose]
        oc = r.OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = self.config.end_effector_link
        oc.orientation = pose.orientation
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0
        constraints = r.Constraints()
        constraints.position_constraints = [pc]
        constraints.orientation_constraints = [oc]
        req.goal_constraints = [constraints]
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return self._execute("arm", goal)

    def execute_grasp(self, position, effort=50.0):
        """Set finger_joint radians (0..0.7); effort does not verify an object is held."""
        position = _number("position", position, 0.0, 0.7)
        effort = _number("effort", effort, 0.001, 60.0)
        goal = self._ros.GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = effort
        return self._execute("gripper", goal)

    def _finish(self, op, result):
        with self._lock:
            op.result = result
            if self._active is op:
                self._active = None
            op.done.set()

    def _result_received(self, op, future):
        try:
            response = future.result()
            status, payload = response.status, response.result
            details = {}
            success = status == 4  # action_msgs/GoalStatus.STATUS_SUCCEEDED
            if op.kind == "arm":
                details["error_code"] = payload.error_code.val
                success = success and payload.error_code.val == 1
            elif op.kind == "gripper":
                details = {k: getattr(payload, k) for k in ("position", "effort", "stalled", "reached_goal")}
                success = success and payload.reached_goal and not payload.stalled
            elif hasattr(payload, "error_code"):
                details = {"error_code": payload.error_code,
                           "error_msg": getattr(payload, "error_msg", "")}
                success = success and payload.error_code == 0
            outcome = "succeeded" if success else ("cancelled" if status == 5 else "failed")
            self._finish(op, MotionResult(outcome, f"{op.kind}: {outcome}", status, details))
        except Exception as exc:
            # A transport failure is not proof that the robot stopped. Keep the slot.
            op.result = MotionResult("error", str(exc), details={"motion_pending": True})
            op.done.set()

    def _accepted(self, op, future):
        try:
            handle = future.result()
            if not handle.accepted:
                self._finish(op, MotionResult("rejected", f"{op.kind} goal rejected"))
                return
            with self._lock:
                op.handle = handle
                handle.get_result_async().add_done_callback(lambda f: self._result_received(op, f))
                if op.cancelled.is_set():
                    handle.cancel_goal_async()
        except Exception as exc:
            op.result = MotionResult("error", str(exc), details={"motion_pending": True})
            op.done.set()

    def _execute(self, kind, goal):
        with self._lock:
            if self._closed:
                return MotionResult("closed", "Bridge is closed")
            if not self.enable_motion:
                return MotionResult("disabled", "Construct ROSBridge(enable_motion=True) to enable motion")
            if self._active:
                return MotionResult("busy", "Previous motion has not reached a terminal state")
            op = _Operation(kind, cancelled=_request_cancel.get() or threading.Event())
            self._active = op
        try:
            deadline = time.monotonic() + self.config.server_timeout
            while (self.config.use_sim_time and self._node.get_clock().now().nanoseconds == 0
                   and not op.cancelled.is_set() and time.monotonic() < deadline):
                op.cancelled.wait(min(0.05, max(0, deadline - time.monotonic())))
            if self.config.use_sim_time and self._node.get_clock().now().nanoseconds == 0:
                self._finish(op, MotionResult("unavailable", "No simulation clock received"))
                return op.result
            client = self._clients[kind]
            available = client.wait_for_server(timeout_sec=self.config.server_timeout)
            with self._lock:
                if self._closed or op.cancelled.is_set():
                    self._finish(op, MotionResult("cancelled", "Cancelled before sending goal"))
                elif not available:
                    self._finish(op, MotionResult("unavailable", f"{kind} action server unavailable"))
                else:
                    if kind == "navigation":
                        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
                    client.send_goal_async(goal).add_done_callback(lambda f: self._accepted(op, f))
        except Exception as exc:
            # Sending may have reached the server: fail closed until shutdown.
            op.result = MotionResult("error", str(exc), details={"motion_pending": True})
            op.done.set()
        if op.done.wait(self.config.goal_timeout):
            return op.result
        self._cancel_operation(op)
        return MotionResult("timeout", f"{kind} timed out; cancellation requested",
                            details={"motion_pending": not op.done.is_set()})

    def _cancel_operation(self, op):
        with self._lock:
            op.cancelled.set()
            if op.handle is not None and self._active is op:
                try:
                    op.handle.cancel_goal_async()
                except Exception as exc:
                    return MotionResult("error", f"Cancellation failed: {exc}",
                                        details={"motion_pending": True})
        return MotionResult("cancel_requested", "Await terminal result before further motion")

    def cancel(self):
        """Request cancellation of this bridge's active goal, including pending acceptance."""
        with self._lock:
            op = self._active
        return self._cancel_operation(op) if op else MotionResult("idle", "No bridge-owned goal active")

    async def _async_call(self, function, *args, **kwargs):
        # Shield the worker so cancellation of an asyncio task also cancels ROS.
        cancelled = threading.Event()
        token = _request_cancel.set(cancelled)
        try:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        finally:
            _request_cancel.reset(token)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # A detached worker may still finish validation with an exception.
            task.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            with self._lock:
                if self._active and self._active.cancelled is cancelled:
                    self._cancel_operation(self._active)
            raise

    async def navigate_async(self, x: float, y: float, yaw: float = 0.0, frame_id: str = "map") -> bool:
        """Return True only when Nav2 confirms arrival at the requested pose."""
        result = await self._async_call(self.send_navigation_goal, x, y, yaw, frame_id)
        if not result.success:
            logging.getLogger(__name__).warning("Navigation to (%s, %s) in %s: %s",
                                                x, y, frame_id, result.to_dict())
        return result.success

    async def navigate_to_location_async(self, name):
        return await self._async_call(self.navigate_to_location, name)

    async def move_arm_async(self, x, y, z, qx, qy, qz, qw, frame_id="base_link", *, linear=False) -> MotionResult:
        """Await MoveIt execution and return its status and controller error code."""
        return await self._async_call(self.move_arm_pose, x, y, z, qx, qy, qz, qw, frame_id, linear=linear)

    async def grasp_async(self, position, effort=50.0) -> MotionResult:
        """Await the finger command and return its status and gripper feedback."""
        return await self._async_call(self.execute_grasp, position, effort)

    def close(self):
        """Cancel, wait a bounded time, then release this bridge's ROS context."""
        with self._lock:
            if self._closed:
                return MotionResult("closed", "Bridge already closed")
            self._closed = True
            op = self._active
        stopped = True
        if op:
            self._cancel_operation(op)
            stopped = op.done.wait(self.config.shutdown_timeout) and self._active is None
        self._executor.shutdown(timeout_sec=self.config.shutdown_timeout)
        self._thread.join(timeout=self.config.shutdown_timeout)
        self._node.destroy_node()
        self._context.try_shutdown()
        return MotionResult("closed" if stopped else "unconfirmed_stop",
                            "Bridge closed" if stopped else "ROS goal termination unconfirmed; check simulator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--locations", type=Path, help="JSON mapping measured location names to poses")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    commands.add_parser("camera-status")
    commands.add_parser("locations")
    named = commands.add_parser("navigate-location")
    named.add_argument("name")
    nav = commands.add_parser("navigate")
    nav.add_argument("x", type=float)
    nav.add_argument("y", type=float)
    nav.add_argument("yaw", type=float)
    nav.add_argument("--frame", default="map")
    arm = commands.add_parser("arm")
    arm.add_argument("pose", type=float, nargs=7, metavar="XYZ_QXYZW")
    arm.add_argument("--frame", default="base_link")
    hand = commands.add_parser("gripper")
    hand.add_argument("position", type=float)
    hand.add_argument("--effort", type=float, default=50.0)
    args = parser.parse_args()
    if args.command in ("navigate", "navigate-location", "arm", "gripper") and not args.enable_motion:
        parser.error("Motion commands require --enable-motion")
    bridge = None
    try:
        locations = load_locations(args.locations) if args.locations else {}
        bridge = ROSBridge(enable_motion=args.enable_motion, locations=locations)
        if args.command in ("status", "camera-status"):
            # Allow discovery and /clock delivery; this never moves the robot.
            threading.Event().wait(bridge.config.server_timeout)
            result = bridge.readiness() if args.command == "status" else bridge.camera_status()
        elif args.command == "locations":
            result = {"success": True, "locations": {k: asdict(v) for k, v in bridge._locations.items()}}
        elif args.command == "navigate-location":
            result = bridge.navigate_to_location(args.name).to_dict()
        elif args.command == "navigate":
            result = bridge.send_navigation_goal(args.x, args.y, args.yaw, args.frame).to_dict()
        elif args.command == "arm":
            result = bridge.move_arm_pose(*args.pose, frame_id=args.frame).to_dict()
        else:
            result = bridge.execute_grasp(args.position, args.effort).to_dict()
        print(json.dumps(result))
        return 0 if result.get("success", result.get("ready", False)) else 1
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}))
        return 1
    finally:
        if bridge:
            closed = bridge.close()
            if closed.status == "unconfirmed_stop":
                print(json.dumps(closed.to_dict()))


if __name__ == "__main__":
    raise SystemExit(main())
