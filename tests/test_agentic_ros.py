"""Graph integration tests: fake ROS transport and injected models, no API calls."""
import asyncio
import base64
import io
from types import SimpleNamespace as NS
import unittest
from uuid import uuid4

import numpy as np
from PIL import Image

from agentic import (BoundingBoxes, Models, TaskAnalysis, build_robot_brain,
                     tool_unproject_pixel)
from ros_bridge import CameraUnavailableError, LocationPose, UnknownLocationError


class Model:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SimulatedCameraBridge:
    """Graph boundary double. No actual ROS or simulator is connected."""
    def __init__(self, image=None):
        self.locations = {"workshop bench": LocationPose(1.2, -0.5),
                          "kitchen table": LocationPose(-2.1, 3.4)}
        self.events = []
        self.arrived = True
        self.camera_error = None
        self.image = image or Image.new("RGB", (100, 80), "red")

    def lookup_location(self, name):
        key = " ".join(name.casefold().split())
        if key not in self.locations:
            raise UnknownLocationError(f"Unknown location: {name}")
        return self.locations[key]

    async def navigate_async(self, x, y, yaw, frame_id):
        self.events.append(("navigate", LocationPose(x, y, yaw, frame_id)))
        return self.arrived

    def camera_snapshot(self, *, received_after):
        self.events.append(("camera", received_after))
        if self.camera_error:
            raise self.camera_error
        return NS(rgb=NS(header=NS(frame_id="camera", stamp=NS(sec=1, nanosec=0))),
                  depth=NS(encoding="32FC1"),
                  to_arrays=lambda: (np.asarray(self.image)[:, :, ::-1], None, None))


def models_for(task_type="inspect", start="workshop bench", end=None, target="red square", probability=0.99):
    return Models(
        Model(TaskAnalysis(task_type=task_type, target_object=target, start_location=start, end_location=end)),
        Model(NS(predicates={"object_presence": NS(probability=probability)})),
        Model(BoundingBoxes.model_validate({"boxes": [{"box_2d": [100, 200, 600, 800]}]})),
    )


async def invoke(bridge, models, **state):
    graph = build_robot_brain(bridge, models=models, capture_timeout=0.06)
    return await graph.ainvoke({"task_input": "test", **state},
                              config={"configurable": {"thread_id": uuid4().hex}})


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_navigate_then_camera_then_real_image_payload_and_boxes(self):
        bridge, models = SimulatedCameraBridge(), models_for()
        result = await invoke(bridge, models)
        self.assertEqual([e[0] for e in bridge.events], ["navigate", "camera"])
        self.assertEqual(result["execution_status"], "inspection_complete")
        self.assertEqual(result["boxes"], [[20, 8, 80, 48]])
        self.assertEqual(result["camera_metadata"]["depth_encoding"], "32FC1")
        presence_url = models.presence.calls[0]["input"].content[0]["image_url"]["url"]
        boxes_url = models.boxes.calls[0][0].content[1]["image_url"]["url"]
        self.assertEqual(presence_url, boxes_url)
        with Image.open(io.BytesIO(base64.b64decode(presence_url.split(",", 1)[1]))) as image:
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    async def test_navigation_only_needs_no_object_or_camera(self):
        bridge, models = SimulatedCameraBridge(), models_for("navigate", None, "workshop bench", None)
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "nav_success")
        self.assertEqual(result["current_phase"], "done")
        self.assertEqual([e[0] for e in bridge.events], ["navigate"])
        self.assertEqual(models.presence.calls, [])

    async def test_unknown_destination_stops_before_motion(self):
        bridge, models = SimulatedCameraBridge(), models_for("navigate", None, "missing dock", None)
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "unknown_location")
        self.assertEqual(bridge.events, [])
        self.assertEqual(models.presence.calls, [])

    async def test_transport_validates_both_locations_before_moving(self):
        bridge, models = SimulatedCameraBridge(), models_for("transport", end="missing dock")
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "unknown_location")
        self.assertEqual(bridge.events, [])

    async def test_transport_stops_before_unimplemented_grasp(self):
        bridge, models = SimulatedCameraBridge(), models_for("transport", end="kitchen table")
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "awaiting_grasp_planning")
        self.assertEqual([e[1] for e in bridge.events if e[0] == "navigate"], [bridge.locations["workshop bench"]])

    async def test_failed_navigation_never_uses_camera(self):
        bridge, models = SimulatedCameraBridge(), models_for()
        bridge.arrived = False
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "navigation_failed")
        self.assertEqual([e[0] for e in bridge.events], ["navigate"])
        self.assertEqual(models.presence.calls, [])

    async def test_camera_waits_for_arrival(self):
        bridge, models = SimulatedCameraBridge(), models_for()
        sent, arrived = asyncio.Event(), asyncio.Event()

        async def delayed_navigation(*args):
            sent.set()
            await arrived.wait()
            return True

        bridge.navigate_async = delayed_navigation
        task = asyncio.create_task(invoke(bridge, models))
        await asyncio.wait_for(sent.wait(), 3)
        self.assertFalse(task.done())
        self.assertEqual(bridge.events, [])
        arrived.set()
        self.assertEqual((await task)["execution_status"], "inspection_complete")

    async def test_camera_failure_does_not_reuse_saved_image(self):
        bridge, models = SimulatedCameraBridge(), models_for()
        bridge.camera_error = CameraUnavailableError("Stale stream")
        result = await invoke(bridge, models, image="old.png", image_url="old", boxes=[[1, 2, 3, 4]])
        self.assertEqual(result["execution_status"], "camera_unavailable")
        self.assertEqual(result["image_url"], "")
        self.assertEqual(result["boxes"], [])
        self.assertEqual(models.presence.calls, [])

    async def test_inspect_current_view_skips_navigation(self):
        bridge, models = SimulatedCameraBridge(), models_for(start=None)
        result = await invoke(bridge, models)
        self.assertEqual(result["execution_status"], "inspection_complete")
        self.assertEqual([e[0] for e in bridge.events], ["camera"])

    async def test_absent_and_uncertain_skip_boxes(self):
        for probability, expected in ((0.1, "absent"), (0.5, "uncertain")):
            models = models_for(probability=probability)
            result = await invoke(SimulatedCameraBridge(), models)
            self.assertEqual(result["item_presence"], expected)
            self.assertEqual(models.boxes.calls, [])

    async def test_missing_details_request_clarification(self):
        bridge, models = SimulatedCameraBridge(), models_for("transport", None, None, None)
        result = await invoke(bridge, models)
        self.assertEqual(result["current_phase"], "missing_object")
        self.assertEqual(set(result["missing_fields"]), {"target_object", "start_location", "end_location"})
        self.assertTrue(result["chat_history"][-1].content)
        self.assertEqual(bridge.events, [])

    async def test_model_error_does_not_advance(self):
        models = models_for()
        models.presence.response = RuntimeError("provider failure")
        result = await invoke(SimulatedCameraBridge(), models)
        self.assertEqual(result["execution_status"], "model_error")
        self.assertEqual(models.boxes.calls, [])

    async def test_navigation_requires_bridge(self):
        result = await invoke(None, models_for("navigate", None, "workshop bench", None))
        self.assertEqual(result["execution_status"], "bridge_unavailable")

    def test_depth_unprojection_handles_meters_and_edges(self):
        info = NS(k=[100, 0, 0, 0, 100, 0, 0, 0, 1])
        meters = np.ones((3, 3))
        self.assertEqual(tool_unproject_pixel(0, 0, meters, info, "32FC1"), (0, 0, 1))
        self.assertEqual(tool_unproject_pixel(0, 0, meters*1000, info), (0, 0, 1))
        with self.assertRaises(ValueError):
            tool_unproject_pixel(0, 0, meters*float("nan"), info, "32FC1")
