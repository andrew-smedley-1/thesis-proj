"""Exercise the real optional LangChain tool schemas without API or ROS calls."""
import importlib.util
import unittest

from robot_tools import build_robot_tools
from ros_bridge import MotionResult


class StubBridge:
    def __init__(self):
        self.calls = []

    def readiness(self):
        return {"ready": True, "motion_enabled": False}

    def cancel(self):
        return MotionResult("idle", "No motion")

    def camera_status(self):
        return {"ready": False, "status": "camera_unavailable"}

    async def navigate_to_location_async(self, name):
        self.calls.append(name)
        return MotionResult("unknown_location", f"Unknown location: {name}")

    async def navigate_async(self, *args):
        self.calls.append(args)
        return False

    async def move_arm_async(self, *args, linear=False):
        self.calls.append((args, linear))
        return MotionResult("succeeded", "Arm complete")

    async def grasp_async(self, *args):
        raise ValueError("Invalid gripper position")


@unittest.skipUnless(importlib.util.find_spec("langchain_core"), "Optional LangChain dependency not installed")
class ToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bridge = StubBridge()
        self.tools = {t.name: t for t in build_robot_tools(self.bridge)}

    async def test_navigation_schema_defaults_and_disabled_result(self):
        result = await self.tools["navigate_robot"].ainvoke({"x": 1, "y": 2})
        self.assertEqual(self.bridge.calls, [(1.0, 2.0, 0.0, "map")])
        self.assertIs(result, False)

    async def test_arm_schema_and_result(self):
        result = await self.tools["move_robot_arm"].ainvoke(
            dict(x=0.4, y=0, z=1, qx=0, qy=0, qz=0, qw=1, linear=True))
        self.assertTrue(result["success"])
        self.assertEqual(self.bridge.calls, [((.4, 0., 1., 0., 0., 0., 1., "base_link"), True)])

    async def test_invalid_input_is_machine_readable(self):
        result = await self.tools["command_robot_gripper"].ainvoke({"position": -1})
        self.assertEqual(result["status"], "invalid_input")
        self.assertFalse(result["success"])

    async def test_readiness_and_cancel_never_move(self):
        self.assertTrue((await self.tools["robot_readiness"].ainvoke({}))["ready"])
        self.assertEqual((await self.tools["cancel_robot_motion"].ainvoke({}))["status"], "idle")
        self.assertEqual(self.bridge.calls, [])

    async def test_named_destination_error_is_preserved(self):
        result = await self.tools["navigate_robot_to_location"].ainvoke({"location_name": "missing"})
        self.assertEqual(result["status"], "unknown_location")
        self.assertFalse(result["success"])
        self.assertEqual(self.bridge.calls, ["missing"])

    async def test_camera_status_returns_metadata_without_motion(self):
        result = await self.tools["robot_camera_status"].ainvoke({})
        self.assertEqual(result["status"], "camera_unavailable")
        self.assertEqual(self.bridge.calls, [])
