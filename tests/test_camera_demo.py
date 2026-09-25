import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mambaglue.camera_demo import (
    CameraReader,
    ImageLoader,
    configure_camera,
    draw_frame_matches,
    draw_keypoints,
    fourcc_string,
    score_color,
)


def _write_image(path: Path, value: int) -> None:
    image = np.full((6, 8, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


class FakeCapture:
    def __init__(self, frames=()):
        self.values = {}
        self.frames = list(frames)

    def set(self, prop, value):
        self.values[prop] = value

    def get(self, prop):
        return self.values.get(prop, 0)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)


class CameraConfigTest(unittest.TestCase):
    def test_fourcc_string_decodes_four_characters(self):
        self.assertEqual(fourcc_string(cv2.VideoWriter_fourcc(*"MJPG")), "MJPG")

    def test_configure_camera_returns_negotiated_values(self):
        cap = FakeCapture()

        actual = configure_camera(cap, 640, 480, 30.0, "MJPG", 2)

        self.assertEqual(actual["fourcc"], "MJPG")
        self.assertEqual(actual["width"], 640)
        self.assertEqual(actual["height"], 480)
        self.assertEqual(actual["fps"], 30.0)
        self.assertEqual(cap.values[cv2.CAP_PROP_BUFFERSIZE], 2)


class CameraReaderTest(unittest.TestCase):
    def test_read_latest_returns_none_once_capture_stops(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        reader = CameraReader(FakeCapture([frame]))
        reader.start()
        reader.join(timeout=1.0)

        self.assertFalse(reader.running)
        self.assertIsNotNone(reader.frame)
        self.assertIsNone(reader.read_latest(timeout=0.05))

    def test_stop_is_idempotent(self):
        reader = CameraReader(FakeCapture())
        reader.start()
        reader.stop()
        reader.stop()

        self.assertFalse(reader.running)


class ImageLoaderTest(unittest.TestCase):
    def test_reads_image_directory_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_image(directory / "b.png", 2)
            _write_image(directory / "a.png", 1)

            loader = ImageLoader(str(directory))

            self.assertEqual(loader.mode, "images")
            frames = list(loader)
            self.assertEqual(len(frames), 2)
            self.assertEqual(int(frames[0][0, 0, 0]), 1)
            self.assertEqual(int(frames[1][0, 0, 0]), 2)

    def test_rejects_missing_source(self):
        with self.assertRaises(IOError):
            ImageLoader("/nonexistent/mambaglue/source")

    def test_rejects_malformed_camera_source(self):
        with self.assertRaises(ValueError):
            ImageLoader("camerax")

    def test_rejects_empty_image_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(IOError):
                ImageLoader(tmp)


class DrawingTest(unittest.TestCase):
    def test_score_color_moves_from_red_to_green(self):
        low = score_color(0.0)
        high = score_color(1.0)

        self.assertLess(low[1], high[1])
        self.assertGreater(low[2], high[2])

    def test_draw_keypoints_marks_rounded_position(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)

        result = draw_keypoints(image, np.array([[2.6, 3.4]], dtype=np.float32))

        self.assertTupleEqual(tuple(result[3, 3]), (0, 0, 255))
        self.assertTupleEqual(tuple(image[3, 3]), (0, 0, 0))

    def test_draw_frame_matches_draws_line_and_current_point(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)

        result = draw_frame_matches(
            image,
            np.array([[1.0, 1.0]], dtype=np.float32),
            np.array([[9.0, 9.0]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

        expected_color = score_color(1.0)
        self.assertTupleEqual(tuple(result[5, 5]), expected_color)
        self.assertTupleEqual(tuple(result[9, 9]), (0, 0, 255))

    def test_draw_frame_matches_without_matches_returns_copy(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        result = draw_frame_matches(
            image,
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

        self.assertTrue(np.array_equal(result, image))
        self.assertIsNot(result, image)


if __name__ == "__main__":
    unittest.main()
