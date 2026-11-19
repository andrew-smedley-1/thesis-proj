import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
from types import SimpleNamespace as NS
from uuid import uuid4
import numpy as np

from agentic import build_robot_brain
from pick_place import PickPlaceController, color_mask, TableStation, free_place, point_cloud, held_center
from ros_bridge import MotionResult, LocationPose
from test_agentic_ros import SimulatedCameraBridge, models_for


class PickPlaceGraphTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bridge = SimulatedCameraBridge()
        self.scene = Mock()
        self.scene.source = TableStation((1.58,-1.,.8), (.45,-.85,1.25))
        self.scene.source_view_pose.return_value = [.4, -.85, 1.25]
        self.scene.destination.view_pose = [.4, -.9, 1.25]
        self.scene.inspect_destination.return_value = [1., 1., .8]
        self.scene.plan_grasp.return_value = {"lift": [1., -1., 1.2], "grasp": [1., -1., 1.]}
        self.scene.verify_held.return_value = True
        self.scene.verify_place.return_value = {"verified": True}
        self.success = MotionResult("succeeded", "complete", 4)
        self.bridge.move_arm_async = AsyncMock(return_value=self.success)
        self.bridge.grasp_async = AsyncMock(return_value=self.success)
        # Skip mechanical settling delays with these simulated actuators.
        clock = self.enterContext(patch("agentic.asyncio", wraps=asyncio))
        clock.sleep = AsyncMock()

    async def run_graph(self):
        graph = build_robot_brain(self.bridge, models=models_for("transport", end="kitchen table"),
                                  manipulator=self.scene)
        return await graph.ainvoke({"task_input": "Move red cube from workshop bench to kitchen table"},
                                   config={"configurable": {"thread_id": uuid4().hex}})

    async def test_pick_place_awaits_each_tool_and_verifies(self):
        arm_sent, arm_done = asyncio.Event(), asyncio.Event()
        gripper_sent, gripper_done = asyncio.Event(), asyncio.Event()

        async def arm(*args, **kwargs):
            arm_sent.set()
            await arm_done.wait()
            return self.success

        async def gripper(position, effort):
            gripper_sent.set()
            await gripper_done.wait()
            if position == .54:
                return MotionResult("failed", "stalled", 6, {"stalled": True, "position": .3})
            return self.success

        self.bridge.move_arm_async.side_effect = arm
        self.bridge.grasp_async.side_effect = gripper
        task = asyncio.create_task(self.run_graph())
        await asyncio.wait_for(arm_sent.wait(), 3)
        self.assertEqual(self.bridge.events, [])
        arm_done.set()
        await asyncio.wait_for(gripper_sent.wait(), 3)
        self.assertEqual(self.bridge.move_arm_async.await_count, 4)
        self.scene.attach.assert_not_called()
        gripper_done.set()
        result = await task
        self.assertEqual(result["execution_status"], "pick_place_complete")
        self.assertTrue(result["grasp_verified"])
        self.assertTrue(result["placement_verified"])
        navigation = [value for event, value in self.bridge.events if event == "navigate"]
        self.assertEqual(navigation, [self.bridge.locations[name] for name in
                                     ["kitchen table", "workshop bench", "kitchen table"]])
        self.assertEqual([c.args[0] for c in self.bridge.grasp_async.await_args_list], [.01, .54, .01])
        self.assertEqual([c.args[0] for c in self.scene.verify_held.call_args_list],
                         ["held-after-lift", "held-after-navigation"])

    async def test_dedicated_pickup_pose_only_applies_to_source_visit(self):
        pickup = LocationPose(1.36, 0.)
        self.scene.source = TableStation((1.58,-1.,.8), (.45,-.85,1.25), pickup)
        result = await self.run_graph()
        nav = [value for event,value in self.bridge.events if event == "navigate"]
        destination = self.bridge.locations["kitchen table"]
        self.assertEqual(nav, [destination, pickup, destination])
        self.assertEqual(result["navigation_result"]["pickup_pose"]["x"], 1.36)
        self.assertTrue(result["placement_verified"])

    async def test_failed_arm_prevents_navigation(self):
        self.bridge.move_arm_async.return_value = MotionResult("failed", "planning failure", 6)
        result = await self.run_graph()
        self.assertEqual(result["execution_status"], "manipulation_failed")
        self.assertIn("carry_pose", result["error"])
        self.assertEqual(self.bridge.events, [])
        self.bridge.grasp_async.assert_not_awaited()

    async def test_failed_close_prevents_lift_and_carry(self):
        self.bridge.grasp_async.side_effect = [self.success, MotionResult("timeout", "gripper timed out")]
        result = await self.run_graph()
        self.assertEqual(result["execution_status"], "manipulation_failed")
        self.assertIn("timeout", result["error"])
        self.scene.attach.assert_not_called()
        self.scene.verify_held.assert_not_called()
        self.assertFalse(result["grasp_verified"])

    async def test_unverified_grasp_prevents_carry(self):
        self.scene.verify_held.return_value = False
        result = await self.run_graph()
        self.assertFalse(result["grasp_verified"])
        self.assertEqual(len([e for e, _ in self.bridge.events if e == "navigate"]), 2)
        self.scene.detach.assert_not_called()

    async def test_unreachable_lift_retries_vertical_before_verifying(self):
        async def move(x, y, z, *args, **kwargs):
            if kwargs.get("linear") and [x, y, z] == self.scene.plan_grasp.return_value["lift"]:
                return MotionResult("failed", "no IK", 6, {"error_code": -31})
            return self.success

        self.bridge.move_arm_async.side_effect = move
        result = await self.run_graph()
        self.assertTrue(result["placement_verified"])
        self.assertTrue(any(call.args[:3] == (1., -1., 1.12) and call.kwargs.get("linear")
                            for call in self.bridge.move_arm_async.await_args_list))

    async def test_failed_carry_prevents_release(self):
        self.bridge.navigate_async = AsyncMock(side_effect=[True, True, False])
        result = await self.run_graph()
        self.assertTrue(result["grasp_verified"])
        self.assertEqual(result["execution_status"], "manipulation_failed")
        self.assertEqual(self.bridge.grasp_async.await_count, 2)
        self.scene.detach.assert_not_called()

    async def test_unverified_release_does_not_complete(self):
        self.scene.verify_place.return_value = {"verified": False}
        result = await self.run_graph()
        self.assertEqual(result["execution_status"], "manipulation_failed")
        self.assertFalse(result["placement_verified"])


class RGBDTests(unittest.TestCase):
    def test_view_compensates_parking_yaw_and_camera_mount_offset(self):
        scene = PickPlaceController.__new__(PickPlaceController)
        scene.source = TableStation((1.58, 1., .8), (.4,-.9,1.25))
        odom_in_base = NS(translation=NS(x=1.95,y=0,z=0), rotation=NS(x=0,y=0,z=1,w=0))
        mount = NS(translation=NS(x=.05,y=-.03,z=.05), rotation=NS(x=0,y=0,z=0,w=1))
        scene.transform = Mock(side_effect=[odom_in_base,mount])
        pose = scene.source_view_pose()
        optical_xy = np.array(pose[:2]) + [.05,.03]
        table_xy = np.array([.37,-1.])
        np.testing.assert_allclose(optical_xy, table_xy - .06*table_xy/np.linalg.norm(table_xy))
        self.assertEqual(pose[2], 1.25)

    def test_target_color_uses_ros_bgr_order_and_rejects_neutral_pixels(self):
        samples = np.array([[[200,10,10], [10,200,10], [10,10,200], [180,180,180]]], dtype=np.uint8)
        for i, color in enumerate(("blue", "green", "red")):
            np.testing.assert_array_equal(color_mask(samples, color)[0], np.arange(4) == i)

    def test_held_cube_is_below_downward_flange(self):
        for quaternion in [(1,0,0,0),(0,1,0,0)]:
            transform=NS(rotation=NS(**dict(zip(('x','y','z','w'),quaternion))),
                         translation=NS(x=1,y=2,z=1.28))
            np.testing.assert_allclose(held_center(transform),[1,2,1.058])
    def test_depth_units_and_transform(self):
        bgr=np.zeros((3,3,3),dtype=np.uint8)
        info=NS(k=[100,0,1,0,100,1,0,0,1])
        transform=NS(rotation=NS(x=0,y=0,z=0,w=1),translation=NS(x=2,y=3,z=4))
        a,valid=point_cloud(bgr,np.ones((3,3)),info,transform,'32FC1')
        b,_=point_cloud(bgr,np.ones((3,3))*1000,info,transform,'16UC1')
        np.testing.assert_allclose(a,b);np.testing.assert_allclose(a[1,1],[2,3,5])
        self.assertTrue(valid.all())
    def test_clear_patch_requires_observed_surface_and_rejects_occupied(self):
        x,y=np.meshgrid(np.linspace(1.3,1.85,150),np.linspace(.67,1.33,150))
        points=np.stack([x,y,np.ones_like(x)*.85],axis=-1)
        valid=np.ones(x.shape,dtype=bool)
        station=TableStation((1.58,1,.8),(.4,-.9,1.25))
        slot=free_place(points,valid,station,np.array([0,0,.85]))
        self.assertAlmostEqual(slot[0],1.64)
        points[:,:,2]=.90
        with self.assertRaises(ValueError):free_place(points,valid,station,np.array([0,0,.85]))
        with self.assertRaises(ValueError):free_place(points,valid & False,station,np.array([0,0,.85]))
