"""Live camera / video frame-to-frame matching demo with MambaGlue.

The workflow follows the ALIKED sequence demo: pass ``camera0`` for a webcam,
a video file, or a directory of images, and every frame is matched against the
previous one with an extractor plus the released MambaGlue matcher. Press
space to start and ``q``/ESC to stop.
"""

from __future__ import annotations

import argparse
import glob
import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .aliked import ALIKED
from .disk import DISK
from .mambaglue import MambaGlue
from .sift import SIFT
from .superpoint import SuperPoint
from .utils import numpy_image_to_torch, rbd

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _mamba_selective_scan_fn,
    )
except (ImportError, ModuleNotFoundError):
    _mamba_selective_scan_fn = None

EXTRACTORS = {
    "superpoint": SuperPoint,
    "disk": DISK,
    "aliked": ALIKED,
    "sift": SIFT,
}
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.ppm")
WINDOW_NAME = "MambaGlue camera demo"


def fourcc_string(value: int) -> str:
    """Decode the integer returned by ``CAP_PROP_FOURCC`` into four characters."""
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


def configure_camera(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: float,
    fourcc: str = "MJPG",
    buffersize: int = 2,
) -> dict:
    """Apply V4L2 capture settings and return the negotiated values.

    The pixel format is requested before the size and frame rate so the driver
    renegotiates the stream once. ``buffersize=1`` is avoided by default: with
    the OpenCV V4L2 backend it makes many UVC cameras deliver every other frame
    (30 fps -> 15 fps), while 2 or more keep the full rate.
    """
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if buffersize:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    return {
        "fourcc": fourcc_string(cap.get(cv2.CAP_PROP_FOURCC)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }


class CameraReader(threading.Thread):
    """Continuously drain the camera and keep only the newest frame.

    Reading in a background thread prevents the driver queue from filling up
    with stale frames, so the demo always processes the freshest frame even
    when inference is slower than the camera frame rate.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        super().__init__(daemon=True)
        self.cap = cap
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.running = True
        self.fps = 0.0
        self._timestamps: deque[float] = deque(maxlen=30)

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                with self.condition:
                    self.running = False
                    self.condition.notify_all()
                return
            with self.condition:
                self.frame = frame
                self.frame_id += 1
                self._timestamps.append(time.perf_counter())
                if len(self._timestamps) > 1:
                    self.fps = (len(self._timestamps) - 1) / (
                        self._timestamps[-1] - self._timestamps[0]
                    )
                self.condition.notify_all()

    def read_latest(self, timeout: float = 2.0) -> np.ndarray | None:
        """Block until a frame newer than the previous call is available."""
        with self.condition:
            last_id = self.frame_id
            self.condition.wait_for(
                lambda: self.frame_id > last_id or not self.running, timeout
            )
            if self.frame_id <= last_id or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self) -> None:
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.join(timeout=2.0)


class ImageLoader:
    """Yield BGR frames from ``cameraN``, a video file, or an image folder."""

    def __init__(
        self,
        source: str,
        *,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: float = 30.0,
        camera_fourcc: str = "MJPG",
        camera_buffersize: int = 2,
    ):
        self.cap = None
        self.reader: CameraReader | None = None
        self.images: list[str] = []
        self.index = 0
        self.fps = 30.0
        path = Path(source).expanduser()

        if source.startswith("camera"):
            index = source[len("camera") :]
            if not index.isdigit():
                raise ValueError(
                    f'camera source must look like "camera0", got {source!r}'
                )
            self.mode = "camera"
            self.cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
            if not self.cap.isOpened():
                raise IOError(f"cannot open camera {index}")
            actual = configure_camera(
                self.cap,
                camera_width,
                camera_height,
                camera_fps,
                camera_fourcc,
                camera_buffersize,
            )
            self.fps = actual["fps"] if actual["fps"] > 0 else camera_fps
            self.format = (
                f"{actual['fourcc']} {actual['width']}x{actual['height']}"
                f"@{actual['fps']:.1f}fps"
            )
            if (
                actual["width"] != camera_width
                or actual["height"] != camera_height
                or abs(actual["fps"] - camera_fps) > 0.5
            ):
                logging.warning(
                    "camera negotiated %s (requested %s %dx%d@%.1ffps)",
                    self.format,
                    camera_fourcc,
                    camera_width,
                    camera_height,
                    camera_fps,
                )
            else:
                logging.info("Opened camera %s: %s", index, self.format)
            self.reader = CameraReader(self.cap)
            self.reader.start()
        elif path.is_file():
            self.mode = "video"
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise IOError(f"cannot open video {path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else 30.0
            frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(
                "Opened video %s (%d frames, %.1f fps)", path, frames, self.fps
            )
        elif path.is_dir():
            self.mode = "images"
            for pattern in IMAGE_PATTERNS:
                self.images.extend(glob.glob(str(path / pattern)))
            self.images.sort()
            if not self.images:
                raise IOError(f"no images found in {path}")
            self.fps = 10.0
            logging.info("Loading %d images from %s", len(self.images), path)
        else:
            raise IOError(
                "source must be 'cameraN', a video file, or an image "
                f"directory: {source!r}"
            )

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or ``None`` once the source is exhausted."""
        if self.mode == "camera":
            assert self.reader is not None
            return self.reader.read_latest()
        if self.mode == "images":
            if self.index >= len(self.images):
                return None
            filename = self.images[self.index]
            self.index += 1
            frame = cv2.imread(filename)
            if frame is None:
                raise IOError(f"cannot read image {filename}")
            return frame
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    @property
    def capture_fps(self) -> float:
        """Measured camera frame rate (0 for non-camera sources)."""
        return self.reader.fps if self.reader is not None else 0.0

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def release(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def score_color(score: float) -> tuple[int, int, int]:
    """Map a match score to a BGR color (red low, green high)."""
    score = float(np.clip(score, 0.0, 1.0))
    return (64, int(255 * score), int(255 * (1.0 - score)))


def draw_keypoints(
    image: np.ndarray, points: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """Return a copy of ``image`` with the keypoints drawn as small dots."""
    canvas = image.copy()
    for x, y in np.asarray(points, dtype=np.float64).reshape(-1, 2):
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, 1, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_frame_matches(
    image: np.ndarray,
    points_prev: np.ndarray,
    points_cur: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Draw motion lines from the previous frame to the current one."""
    canvas = image.copy()
    points_prev = np.asarray(points_prev, dtype=np.float64).reshape(-1, 2)
    points_cur = np.asarray(points_cur, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    for (x0, y0), (x1, y1), score in zip(points_prev, points_cur, scores):
        color = score_color(float(score))
        start = (int(round(x0)), int(round(y0)))
        end = (int(round(x1)), int(round(y1)))
        cv2.line(canvas, start, end, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, end, 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return canvas


class FrameMatcher:
    """Match each frame against the previous one, caching its features."""

    def __init__(
        self,
        extractor,
        matcher,
        device: str,
        resize: int | None,
        max_matches: int,
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.device = device
        self.resize = resize
        self.max_matches = max_matches
        self.previous_features = None
        self.previous_points: np.ndarray | None = None
        self.last_runtime = 0.0

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return the annotated frame, its match count, and its keypoint count."""
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = numpy_image_to_torch(rgb).to(self.device)
        with torch.inference_mode():
            features = self.extractor.extract(image, resize=self.resize)
            matches01 = None
            if self.previous_features is not None:
                matches01 = rbd(
                    self.matcher({"image0": self.previous_features, "image1": features})
                )

        current_points = features["keypoints"][0].float().cpu().numpy()
        if matches01 is None:
            canvas = draw_keypoints(frame_bgr, current_points)
            count = 0
        else:
            matches = matches01["matches"].cpu().numpy()
            scores = matches01["scores"].cpu().numpy()
            if len(matches) > self.max_matches:
                order = np.argsort(scores)[::-1][: self.max_matches]
                matches, scores = matches[order], scores[order]
            previous_points = self.previous_points[matches[:, 0]]
            matched_points = current_points[matches[:, 1]]
            canvas = draw_frame_matches(
                frame_bgr, previous_points, matched_points, scores
            )
            count = int(len(matches))

        self.previous_features = features
        self.previous_points = current_points
        self.last_runtime = time.perf_counter() - started
        return canvas, count, int(len(current_points))


def run(args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = (
        EXTRACTORS[args.extractor](max_num_keypoints=args.max_keypoints)
        .eval()
        .to(device)
    )
    scan_backend = args.scan_backend
    if scan_backend == "auto" and _mamba_selective_scan_fn is None:
        logging.warning(
            "mamba_ssm is not installed; using the portable selective scan, "
            "which is several times slower. Install the mamba-ssm wheel for "
            "torch 2.6 + cu12 or lower --max-keypoints for a smoother demo."
        )
    matcher = (
        MambaGlue(features=args.extractor, flash=False, scan_backend=scan_backend)
        .eval()
        .to(device)
    )
    resize = args.resize if args.resize > 0 else None
    frame_matcher = FrameMatcher(extractor, matcher, device, resize, args.max_matches)

    loader = ImageLoader(
        args.source,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
        camera_buffersize=args.camera_buffersize,
    )
    if loader.mode == "camera" and not args.no_display:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} [{loader.format}]")
    writer = None
    runtime_ema = None
    frames = 0
    match_counts: list[int] = []
    wait_time = 0 if not args.no_display else 1
    try:
        for frame in loader:
            canvas, matches, keypoints = frame_matcher.update(frame)
            frames += 1
            match_counts.append(matches)
            runtime_ema = (
                frame_matcher.last_runtime
                if runtime_ema is None
                else 0.9 * runtime_ema + 0.1 * frame_matcher.last_runtime
            )
            fps = 1.0 / max(runtime_ema, 1e-9)
            if loader.mode == "camera":
                status = (
                    f"cam:{loader.capture_fps:.1f}fps model:{fps:.1f}fps "
                    f"matches/keypoints: {matches}/{keypoints}"
                )
            else:
                status = f"model:{fps:.1f}fps matches/keypoints: {matches}/{keypoints}"
            if wait_time == 0:
                cv2.putText(
                    canvas,
                    "Press 'space' to start.",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if args.output:
                if writer is None:
                    Path(args.output).expanduser().parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    writer = cv2.VideoWriter(
                        str(Path(args.output).expanduser()),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        loader.fps,
                        (canvas.shape[1], canvas.shape[0]),
                    )
                    if not writer.isOpened():
                        raise IOError(f"cannot open video writer for {args.output}")
                writer.write(canvas)

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(wait_time)
                if key == ord("q") or key == 27:
                    break
                if key == ord(" "):
                    wait_time = 1
            elif args.output is None and frames % 30 == 0:
                logging.info("frame %d: %s", frames, status)
    finally:
        loader.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    summary = {
        "source": args.source,
        "device": device,
        "extractor": args.extractor,
        "scan_backend": scan_backend,
        "camera_format": loader.format if loader.mode == "camera" else None,
        "camera_fps": loader.capture_fps if loader.mode == "camera" else None,
        "frames": frames,
        "mean_matches": float(np.mean(match_counts)) if match_counts else 0.0,
        "mean_fps": 1.0 / max(runtime_ema, 1e-9) if runtime_ema else 0.0,
        "output": args.output,
    }
    logging.info(
        "Finished! %d frames, mean matches %.1f, mean model fps %.1f",
        summary["frames"],
        summary["mean_matches"],
        summary["mean_fps"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help='Webcam "camera0", a video file, or a directory of images.',
    )
    parser.add_argument(
        "--extractor",
        choices=sorted(EXTRACTORS),
        default="superpoint",
        help="Front-end used with MambaGlue (default: superpoint).",
    )
    parser.add_argument(
        "--scan-backend",
        choices=("auto", "mamba", "portable"),
        default="auto",
        help="Selective-scan backend; auto uses mamba_ssm when installed "
        "(default: auto).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Running device (default: cuda when available, else cpu).",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=512,
        help="Maximum keypoints per frame (default: 512).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
        help="Longest edge used for inference; 0 disables resizing (default: 640).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=100,
        help="Maximum matches to draw (default: 100).",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Requested camera width; ignored for videos/images (default: 640).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Requested camera height; ignored for videos/images (default: 480).",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="Requested camera frame rate (default: 30).",
    )
    parser.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="Requested pixel format; MJPG supports 30 fps at every size on "
        "most UVC cameras (default: MJPG).",
    )
    parser.add_argument(
        "--camera-buffersize",
        type=int,
        default=2,
        help="Driver frame queue. Keep it >= 2: with the OpenCV V4L2 backend "
        "a value of 1 can halve the delivered frame rate (default: 2).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path of an mp4 file with the annotated frames.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a window; useful when running remotely.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run(args)
    except KeyboardInterrupt:
        logging.info("Interrupted; released camera and output file.")


if __name__ == "__main__":
    main()
