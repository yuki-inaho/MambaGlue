import unittest

import numpy as np

from mambaglue.gradio_app import _draw_matches, _score_color


class DrawMatchesTest(unittest.TestCase):
    def test_uses_original_xy_coordinates_and_right_image_offset(self):
        image0 = np.zeros((8, 10, 3), dtype=np.uint8)
        image1 = np.zeros((6, 7, 3), dtype=np.uint8)

        result = _draw_matches(
            image0,
            image1,
            np.array([[2.2, 3.4]], dtype=np.float32),
            np.array([[4.6, 1.6]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

        expected_color = _score_color(1.0)
        self.assertTupleEqual(tuple(result[3, 2]), expected_color)
        # Image 1 starts at x=10, so (4.6, 1.6) is drawn at (15, 2).
        self.assertTupleEqual(tuple(result[2, 15]), expected_color)

    def test_clips_out_of_frame_points_to_their_respective_image_edges(self):
        image0 = np.zeros((4, 5, 3), dtype=np.uint8)
        image1 = np.zeros((3, 6, 3), dtype=np.uint8)

        result = _draw_matches(
            image0,
            image1,
            np.array([[-10.0, 20.0]], dtype=np.float32),
            np.array([[20.0, -10.0]], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )

        expected_color = _score_color(0.0)
        self.assertTupleEqual(tuple(result[3, 0]), expected_color)
        self.assertTupleEqual(tuple(result[0, 10]), expected_color)


if __name__ == "__main__":
    unittest.main()
