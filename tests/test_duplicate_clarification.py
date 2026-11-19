"""Exercise persistent LangGraph pause/resume without robot or model network calls."""
import json
from pathlib import Path
import unittest
from uuid import uuid4

from langgraph.types import Command
from agentic import BoundingBoxes, build_robot_brain
from test_agentic_ros import models_for
import test_pick_place_graph as pick_place_tests

FIXTURE = json.loads((Path(__file__).parent / 'instructions/duplicate_red_cube_clarifications.json').read_text())


class DuplicateClarificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pick_place_tests.PickPlaceGraphTests.setUp(self)
        self.bridge.locations['table a'] = self.bridge.locations['workshop bench']
        self.bridge.locations['table b'] = self.bridge.locations['kitchen table']

    def build_graph(self, selector=None):
        models = models_for('transport', start='Table A', end='Table B', target='red cube')
        models.analysis.response.target_selector = selector
        models.boxes.response = BoundingBoxes.model_validate({'boxes': [
            {'box_2d': b} for b in FIXTURE['boxes_normalized_yxyx']]})
        graph = build_robot_brain(self.bridge, models=models, manipulator=self.scene, spatial_choices_only=True)
        config = {'configurable': {'thread_id': uuid4().hex}}
        return graph, models, config

    def assert_position_prompt(self, question):
        self.assertTrue(question)
        for position in ('left', 'right', 'middle', 'top', 'bottom'):
            self.assertIn(position, question.casefold())
        self.assertNotRegex(question.casefold(), r'\bbox\s+(?:number|\d)|\b(?:numbered|ordinal)\b|\bnumber\s+(?:from|between)|\b(?:first|second|third)\b')

    async def pause(self):
        graph, models, config = self.build_graph()
        paused = await graph.ainvoke({'task_input': FIXTURE['task_input']}, config=config)
        self.assertEqual(paused['execution_status'], 'awaiting_target_selection')
        self.assertEqual(paused['boxes'], FIXTURE['boxes_pixel_xyxy'])
        self.assert_position_prompt(paused['__interrupt__'][0].value['question'])
        self.scene.plan_grasp.assert_not_called()
        # Destination inspection/source navigation may precede the pause; grasp/release must not.
        self.scene.attach.assert_not_called()
        self.bridge.grasp_async.assert_not_awaited()
        self.assertEqual([len(models.analysis.calls), len(models.presence.calls), len(models.boxes.calls)], [1, 1, 1])
        events = list(self.bridge.events)
        await graph.aget_state(config)
        self.assertEqual(self.bridge.events, events)
        return graph, models, config, events

    async def test_generated_valid_replies_resume_without_repeating_perception(self):
        for case in FIXTURE['cases']:
            with self.subTest(reply=case['reply']):
                self.setUp()
                graph, models, config, events = await self.pause()
                result = await graph.ainvoke(Command(resume=case['reply']), config=config)
                self.assertEqual(result['execution_status'], 'pick_place_complete')
                self.assertEqual(result['boxes'], FIXTURE['boxes_pixel_xyxy'])
                self.assertEqual(result['selected_box'], case['expected_selected_box_xyxy'])
                self.scene.plan_grasp.assert_called_once_with([result['selected_box']])
                self.assertEqual([len(models.analysis.calls), len(models.presence.calls), len(models.boxes.calls)], [1, 1, 1])
                self.assertEqual(self.bridge.events[:len(events)], events)
                # Resume only adds destination navigation; source perception is checkpointed.
                self.assertEqual([event for event, _ in self.bridge.events[len(events):]], ['navigate'])

    async def test_invalid_reply_reinterrupts_then_accepts_selection(self):
        for reply in FIXTURE['invalid_replies']:
            with self.subTest(reply=reply):
                self.setUp()
                graph, models, config, events = await self.pause()
                result = await graph.ainvoke(Command(resume=reply), config=config)
                self.assertEqual(result['execution_status'], 'awaiting_target_selection')
                self.assert_position_prompt(result['__interrupt__'][0].value['question'])
                self.assertEqual(result['boxes'], FIXTURE['boxes_pixel_xyxy'])
                self.assertEqual(self.bridge.events, events)
                self.scene.plan_grasp.assert_not_called()
                result = await graph.ainvoke(Command(resume=FIXTURE['recovery_reply']), config=config)
                self.assertEqual(result['selected_box'], FIXTURE['recovery_expected_selected_box_xyxy'])
                self.assertEqual(result['execution_status'], 'pick_place_complete')
                self.assertEqual([len(models.analysis.calls), len(models.presence.calls), len(models.boxes.calls)], [1, 1, 1])

    async def test_explicit_leftmost_instruction_uses_selected_box_immediately(self):
        graph, models, config = self.build_graph(selector='leftmost')
        instruction = FIXTURE['task_input'].replace('red cube', 'leftmost red cube')
        result = await graph.ainvoke({'task_input': instruction}, config=config)
        self.assertEqual(result['selection_dialogue'], [])
        self.assertEqual(result['execution_status'], 'pick_place_complete')
        self.assertEqual(result['selected_box'], FIXTURE['explicit_leftmost_expected_selected_box_xyxy'])
        self.scene.plan_grasp.assert_called_once_with([result['selected_box']])
        self.assertEqual(len(models.boxes.calls), 1)

    async def test_cancel_stops_without_grasp_or_navigation(self):
        graph, models, config, events = await self.pause()
        result = await graph.ainvoke(Command(resume='cancel'), config=config)
        self.assertEqual(result['execution_status'], 'target_selection_cancelled')
        self.assertEqual(result['boxes'], FIXTURE['boxes_pixel_xyxy'])
        self.assertEqual(self.bridge.events, events)
        self.scene.plan_grasp.assert_not_called()
        self.scene.attach.assert_not_called()
        self.assertEqual([len(models.analysis.calls), len(models.presence.calls), len(models.boxes.calls)], [1, 1, 1])


if __name__ == '__main__':
    unittest.main()
