"""Contract tests with fake ROS transports; no ROS, network, or robot required."""
import asyncio
from concurrent.futures import Future
import math
from types import SimpleNamespace as NS
import threading
import unittest
from unittest.mock import patch

from ros_bridge import BridgeConfig, ROSBridge


def xyz():
    return NS(x=0.0, y=0.0, z=0.0)


def pose():
    return NS(position=xyz(), orientation=NS(x=0.0, y=0.0, z=0.0, w=0.0))


def header():
    return NS(frame_id="", stamp=None)


def nav_goal():
    return NS(pose=NS(header=header(), pose=pose()))


def arm_goal():
    return NS(request=NS(start_state=NS(is_diff=False),
                        workspace_parameters=NS(header=header(), min_corner=xyz(), max_corner=xyz())),
              planning_options=NS(planning_scene_diff=NS(robot_state=NS(is_diff=False))))


def ready_future(value):
    future = Future()
    future.set_result(value)
    return future


class Handle:
    def __init__(self):
        self.accepted = True
        self.cancellations = 0
        self.result = Future()

    def get_result_async(self):
        return self.result

    def cancel_goal_async(self):
        self.cancellations += 1
        return ready_future(NS(goals_canceling=[]))


class Client:
    def __init__(self):
        self.available = True
        self.handle = Handle()
        self.acceptance = ready_future(self.handle)
        self.goals = []
        self.sent = threading.Event()

    def server_is_ready(self):
        return self.available

    def wait_for_server(self, timeout_sec):
        return self.available

    def send_goal_async(self, goal):
        self.goals.append(goal)
        self.sent.set()
        return self.acceptance


class FakeNode:
    def __init__(self, *args, **kwargs):
        self.nanoseconds = 100
        self.destroyed = False
        self.subscriptions = []

    def create_subscription(self, message_type, topic, callback, qos):
        subscription = NS(message_type=message_type, topic=topic, callback=callback, qos=qos)
        self.subscriptions.append(subscription)
        return subscription

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=self.nanoseconds, to_msg=lambda: "stamp"))

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:
    def __init__(self, **kwargs):
        self.stopped = threading.Event()

    def add_node(self, node):
        pass

    def spin(self):
        self.stopped.wait()

    def shutdown(self, **kwargs):
        self.stopped.set()


def fake_ros():
    return NS(
        rclpy=NS(init=lambda **kwargs: None),
        Context=lambda: NS(try_shutdown=lambda: None), Node=FakeNode,
        SingleThreadedExecutor=FakeExecutor,
        Parameter=lambda *args, **kwargs: None, ActionClient=lambda *args: Client(),
        NavigateToPose=NS(Goal=nav_goal), MoveGroup=NS(Goal=arm_goal),
        GripperCommand=NS(Goal=lambda: NS(command=NS())), Pose=pose,
        PositionConstraint=lambda: NS(header=header(), constraint_region=NS()),
        OrientationConstraint=lambda: NS(header=header()), Constraints=NS,
        SolidPrimitive=type("SolidPrimitive", (), {"SPHERE": 2}),
        Image=type("Image", (), {}), CameraInfo=type("CameraInfo", (), {}),
        qos_profile_sensor_data="sensor_data_qos",
    )


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.loader = patch("ros_bridge._load_ros", return_value=fake_ros())
        self.loader.start()
        self.bridge = ROSBridge(BridgeConfig(goal_timeout=0.03, shutdown_timeout=0.01,
                                             server_timeout=0.01), enable_motion=True)

    def tearDown(self):
        self.bridge.close()
        self.loader.stop()

    def complete(self, kind, status=4, **payload):
        self.bridge._clients[kind].handle.result.set_result(NS(status=status, result=NS(**payload)))

    def test_navigation_goal_and_success(self):
        self.complete("navigation", error_code=0)
        result = self.bridge.send_navigation_goal(1, 2, math.pi, "map")
        goal = self.bridge._clients["navigation"].goals[0]
        self.assertTrue(result.success)
        self.assertEqual(goal.pose.header.frame_id, "map")
        self.assertEqual(goal.pose.pose.position.x, 1.0)
        self.assertAlmostEqual(goal.pose.pose.orientation.z, 1.0)

    def test_moveit_goal_and_error_code(self):
        self.complete("arm", error_code=NS(val=-1))
        result = self.bridge.move_arm_pose(0.4, 0.2, 1, 0, 0, 0, 1)
        goal = self.bridge._clients["arm"].goals[0]
        self.assertFalse(result.success)
        self.assertEqual(goal.request.group_name, "tm12S_planninggroup")
        self.assertFalse(goal.planning_options.plan_only)
        self.assertTrue(goal.request.start_state.is_diff)
        self.assertEqual(goal.request.max_velocity_scaling_factor, 0.15)
        pc = goal.request.goal_constraints[0].position_constraints[0]
        self.assertEqual(pc.link_name, "link_6")
        self.assertEqual(pc.constraint_region.primitive_poses[0].position.z, 1.0)

    def test_moveit_success(self):
        self.complete("arm", error_code=NS(val=1))
        self.assertTrue(self.bridge.move_arm_pose(0.4, 0, 1, 0, 0, 0, 1).success)

    def test_gripper_command_units(self):
        self.complete("gripper", position=0.6, effort=50.0, stalled=False, reached_goal=True)
        self.assertTrue(self.bridge.execute_grasp(0.6).success)
        self.assertEqual(self.bridge._clients["gripper"].goals[0].command.position, 0.6)

    def test_stalled_gripper_is_not_grasp_success(self):
        self.complete("gripper", position=0.4, effort=50.0, stalled=True, reached_goal=False)
        result = self.bridge.execute_grasp(0.6)
        self.assertFalse(result.success)
        self.assertTrue(result.details["stalled"])

    def test_validation_sends_nothing(self):
        for value in (float("nan"), float("inf"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.bridge.send_navigation_goal(value, 0)
        for value in (-0.1, 0.71):
            with self.assertRaises(ValueError):
                self.bridge.execute_grasp(value)
        with self.assertRaises(ValueError):
            self.bridge.execute_grasp(0.1, 61)
        with self.assertRaises(ValueError):
            self.bridge.move_arm_pose(0, 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            self.bridge.send_navigation_goal(0, 0, frame_id="")
        self.assertFalse(any(c.goals for c in self.bridge._clients.values()))

    def test_no_implicit_motion(self):
        self.bridge.enable_motion = False
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "disabled")
        self.bridge.readiness()
        self.assertFalse(any(c.goals for c in self.bridge._clients.values()))

    def test_missing_server(self):
        self.bridge._clients["navigation"].available = False
        self.assertFalse(self.bridge.readiness()["ready"])
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "unavailable")
        self.assertIsNone(self.bridge._active)

    def test_missing_clock(self):
        self.bridge._node.nanoseconds = 0
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "unavailable")
        self.assertEqual(self.bridge._clients["navigation"].goals, [])

    def test_rejected(self):
        self.bridge._clients["navigation"].handle.accepted = False
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "rejected")
        self.assertIsNone(self.bridge._active)

    def test_timeout_blocks_until_terminal_result(self):
        result = self.bridge.send_navigation_goal(0, 0)
        client = self.bridge._clients["navigation"]
        self.assertEqual(result.status, "timeout")
        self.assertTrue(result.details["motion_pending"])
        self.assertEqual(client.handle.cancellations, 1)
        self.assertEqual(self.bridge.execute_grasp(0.1).status, "busy")
        self.complete("navigation", status=5)
        self.assertIsNone(self.bridge._active)

    def test_cancel_before_delayed_acceptance(self):
        client = self.bridge._clients["navigation"]
        client.acceptance = Future()
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "timeout")
        client.acceptance.set_result(client.handle)
        self.assertEqual(client.handle.cancellations, 1)
        self.complete("navigation", status=5)
        self.assertIsNone(self.bridge._active)

    def test_transport_error_keeps_motion_slot(self):
        client = self.bridge._clients["navigation"]
        client.acceptance = Future()
        client.acceptance.set_exception(RuntimeError("connection lost"))
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "error")
        self.assertEqual(self.bridge.execute_grasp(0.1).status, "busy")

    def test_abort_and_navigation_error(self):
        self.complete("navigation", status=6, error_code=1, error_msg="No path")
        self.assertFalse(self.bridge.send_navigation_goal(0, 0).success)

    def test_cancel_and_shutdown_report_unconfirmed_stop(self):
        self.bridge.send_navigation_goal(0, 0)
        self.assertEqual(self.bridge.cancel().status, "cancel_requested")
        self.assertEqual(self.bridge.close().status, "unconfirmed_stop")
        self.assertTrue(self.bridge._node.destroyed)
        self.assertFalse(self.bridge._thread.is_alive())
        self.assertEqual(self.bridge.close().status, "closed")
        self.assertEqual(self.bridge.send_navigation_goal(0, 0).status, "closed")

    def test_async_cancellation_cancels_ros_goal(self):
        self.bridge.config = BridgeConfig(goal_timeout=0.3, shutdown_timeout=0.01)
        client = self.bridge._clients["navigation"]

        async def scenario():
            task = asyncio.create_task(self.bridge.navigate_async(0, 0))
            self.assertTrue(await asyncio.to_thread(client.sent.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertGreaterEqual(client.handle.cancellations, 1)
            self.complete("navigation", status=5)

        asyncio.run(scenario())

    def test_async_navigation_waits_for_confirmed_arrival(self):
        self.bridge.config = BridgeConfig(goal_timeout=2)

        async def scenario():
            task = asyncio.create_task(self.bridge.navigate_async(1, 2))
            client = self.bridge._clients["navigation"]
            self.assertTrue(await asyncio.to_thread(client.sent.wait, 1))
            self.assertFalse(task.done())
            self.complete("navigation", error_code=0)
            self.assertIs(await task, True)

        asyncio.run(scenario())

    def test_async_navigation_returns_false_on_nav2_error(self):
        self.complete("navigation", status=4, error_code=1)
        self.assertIs(asyncio.run(self.bridge.navigate_async(1, 2)), False)

    def test_cancelled_async_worker_cannot_send_late(self):
        async def scenario():
            async def late_worker(function, *args, **kwargs):
                await asyncio.sleep(0.01)
                return function(*args, **kwargs)

            with patch("ros_bridge.asyncio.to_thread", side_effect=late_worker):
                task = asyncio.create_task(self.bridge.navigate_async(0, 0))
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0.02)
            self.assertEqual(self.bridge._clients["navigation"].goals, [])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
