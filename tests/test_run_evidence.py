import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from agentic import build_robot_brain
from run_evidence import RunEvidence
from test_agentic_ros import models_for


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_records_model_results_and_exact_image(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.png"
            image = Image.new("RGB", (40, 30), "red")
            image.save(source)
            evidence = RunEvidence(Path(directory) / "run")
            graph = build_robot_brain(models=models_for(), evidence=evidence)
            state = await graph.ainvoke({"task_input": "Inspect the cube", "image": str(source)},
                                      {"configurable": {"thread_id": "case"}})
            evidence.finish(state)
            lines = (evidence.directory / "calls.jsonl").read_text()
            calls = [json.loads(line) for line in lines.splitlines()]
            self.assertEqual([c["operation"] for c in calls], ["task_decomposition", "item_presence_detection", "bounding_boxes"])
            self.assertEqual(calls[1]["probability"], .99)
            self.assertEqual(calls[2]["result"]["boxes"][0]["box_2d"], [100, 200, 600, 800])
            self.assertNotIn("base64", lines)
            with Image.open(evidence.directory / "source.png") as saved:
                self.assertEqual(saved.tobytes(), image.tobytes())
            geometry = json.loads((evidence.directory / "geometry.json").read_text())
            self.assertEqual(geometry["boxes_xyxy_px"], [[8, 3, 32, 18]])
            self.assertIsNone(geometry["plan"])

    async def test_failure_is_logged_without_provider_url(self):
        models = models_for()
        models.analysis.response = RuntimeError("provider URL with secret")
        with tempfile.TemporaryDirectory() as directory:
            evidence = RunEvidence(Path(directory) / "run")
            result = await build_robot_brain(models=models, evidence=evidence).ainvoke(
                {"task_input": "Move cube"}, {"configurable": {"thread_id": "case"}})
            self.assertEqual(result["execution_status"], "model_error")
            data = (evidence.directory / "calls.jsonl").read_text()
            self.assertEqual(json.loads(data)["error_type"], "RuntimeError")
            self.assertNotIn("secret", data)

    def test_projects_odom_center_through_inverse_camera_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = RunEvidence(Path(directory) / "run")
            matrix = [[0,-1,0,2], [1,0,0,3], [0,0,1,.5], [0,0,0,1]]
            evidence.camera(Image.new("RGB", (100,80)), {"k":[50,0,50,0,50,40,0,0,1],
                                                        "odom_from_camera":matrix}, np.ones((80,100)))
            evidence.geometry([[50,35,60,55]], {"center":[1.9,3.2,1.5], "grasp":[1.9,3.2,1.722]})
            geometry = json.loads((evidence.directory / "geometry.json").read_text())
            np.testing.assert_allclose(geometry["projected_cube_center_px"], [60,45])
            np.testing.assert_array_equal(np.load(evidence.directory / "depth.npy"), np.ones((80,100)))
