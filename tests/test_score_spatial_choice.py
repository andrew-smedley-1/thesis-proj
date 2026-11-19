import unittest

from scripts.score_spatial_choice import score_choice


class SpatialScoringTests(unittest.TestCase):
    def setUp(self):
        # Identity source camera. The 0.8 m offset converts the Gazebo spawn frame to odom.
        self.initial = {'a': [-.8, .2, 1.], 'b': [-.5, 0., 1.], 'c': [-.2, .5, 1.]}
        self.camera = {'odom_from_camera': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                       'k': [100,0,100,0,100,100,0,0,1]}
        self.names = list(self.initial)
        self.boxes = [[165,145,175,155], [105,115,115,125], [135,95,145,105]]

    def score(self, boxes, selected, direction):
        return score_choice(self.initial, self.names, self.camera, boxes, selected, direction)

    def test_geometry_and_model_order(self):
        for direction,index in [('leftmost',1),('middle',2),('rightmost',0),('topmost',2),('bottommost',0)]:
            with self.subTest(direction=direction):
                self.assertTrue(self.score(self.boxes, self.boxes[index], direction)['selection_correct'])
        self.assertFalse(self.score(self.boxes, self.boxes[0], 'leftmost')['selection_correct'])

    def test_incomplete_or_duplicated_boxes_cannot_pass(self):
        for boxes in (self.boxes[:2], [self.boxes[1]]*3):
            self.assertFalse(self.score(boxes, boxes[0], 'leftmost')['selection_correct'])
