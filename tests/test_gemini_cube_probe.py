import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.gemini_cube_probe import fixture, iou, score, to_pixels


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        _, self.manifest = fixture(Path(self.temp.name))
        self.boxes = [[round(y1/480*1000), round(x1/640*1000), round(y2/480*1000), round(x2/640*1000)]
                      for x1, y1, x2, y2 in self.manifest['reference_boxes_xyxy']]

    def test_fixture_pixels_independent_of_annotations(self):
        image = Image.open(Path(self.temp.name) / 'input.png').convert('RGB')
        # Segment red cube pixels independently in three predeclared scene windows.
        for window, expected in [((70, 150, 145, 235), [86, 171, 127, 217]),
                                 ((220, 75, 290, 150), [231, 88, 272, 134]),
                                 ((295, 265, 365, 345), [306, 281, 347, 327])]:
            points = [(x, y) for y in range(window[1], window[3]) for x in range(window[0], window[2])
                      if (lambda c: c[0] > 80 and c[0] > 2*c[1] and c[0] > 2*c[2])(image.getpixel((x, y)))]
            self.assertEqual([min(x for x,y in points), min(y for x,y in points),
                              max(x for x,y in points)+1, max(y for x,y in points)+1], expected)

    def test_geometry_and_coordinate_order(self):
        self.assertEqual(to_pixels([100, 200, 300, 400], 640, 480), [128, 48, 256, 144])
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 10, 10], [20, 0, 30, 10]), 0)
        for bad in ([0, 0, 1001, 100], [10, 0, 0, 100], [0, 0, 1], [0, 0, 1, 1.5]):
            with self.assertRaises(ValueError):
                to_pixels(bad, 640, 480)

    def test_direct_rejects_wrong_cube_empty_and_multiple(self):
        self.assertTrue(score(self.boxes[:1], self.manifest, 'direct')['selection_pass'])
        for boxes in ([], self.boxes[1:2], self.boxes):
            self.assertFalse(score(boxes, self.manifest, 'direct')['selection_pass'])

    def test_all_ranks_independent_of_model_order_and_rejects_duplicates(self):
        result = score(self.boxes[::-1], self.manifest, 'all')
        self.assertTrue(result['selection_pass'])
        self.assertTrue(result['all_detection_pass'])
        self.assertEqual(result['selected_prediction_index'], 2)
        self.assertFalse(score(self.boxes[:1]*3, self.manifest, 'all')['all_detection_pass'])


if __name__ == '__main__':
    unittest.main()
