"""Interactive Gradio demo for matching two images with MambaGlue."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from . import MambaGlue, SuperPoint, match_pair
from .utils import numpy_image_to_torch


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=8)
def _extractor(max_keypoints: int) -> SuperPoint:
    return SuperPoint(max_num_keypoints=max_keypoints).eval().to(_device())


@lru_cache(maxsize=1)
def _matcher() -> MambaGlue:
    # The optional CUDA mamba_ssm extension can be selected by ``auto`` when
    # present in the host environment.  Its ABI must match the installed
    # PyTorch build; using the portable implementation keeps repeated Gradio
    # requests deterministic and avoids corrupted match indices.
    return (
        MambaGlue(features="superpoint", flash=False, scan_backend="portable")
        .eval()
        .to(_device())
    )


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _score_color(score: float) -> tuple[int, int, int]:
    """Map low scores to red and high scores to green in RGB."""
    score = float(np.clip(score, 0.0, 1.0))
    return (int(255 * (1.0 - score)), int(255 * score), 64)


def _canvas_point(
    point: np.ndarray, *, width: int, height: int, x_offset: int = 0
) -> tuple[int, int]:
    """Convert an image-space ``(x, y)`` point to a visible canvas pixel."""
    x = int(np.clip(np.rint(point[0]), 0, width - 1)) + x_offset
    y = int(np.clip(np.rint(point[1]), 0, height - 1))
    return x, y


def _draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    width0, width1 = image0.shape[1], image1.shape[1]
    canvas = np.full((height, width0 + width1, 3), 255, dtype=np.uint8)
    canvas[: image0.shape[0], :width0] = image0
    canvas[: image1.shape[0], width0:] = image1
    for point0, point1, score in zip(points0, points1, scores):
        # ``features["keypoints"]`` is in the original image coordinate system
        # (x, y), even when matching used a resized inference image.
        start = _canvas_point(point0, width=width0, height=image0.shape[0])
        end = _canvas_point(
            point1, width=width1, height=image1.shape[0], x_offset=width0
        )
        # The canvas is RGB for Gradio.  OpenCV does not convert an ndarray's
        # channels, so write the desired RGB values directly.
        color_rgb = _score_color(float(score))
        cv2.line(canvas, start, end, color_rgb, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, end, 3, color_rgb, -1, lineType=cv2.LINE_AA)
    return canvas


def match_images(
    image0: Image.Image | None,
    image1: Image.Image | None,
    max_keypoints: int,
    resize: int,
    max_matches: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run SuperPoint + released MambaGlue and return a visualized match image."""
    if image0 is None or image1 is None:
        raise ValueError("Upload both images before matching.")

    array0, array1 = _as_rgb_array(image0), _as_rgb_array(image1)
    device = _device()
    extractor = _extractor(int(max_keypoints))
    matcher = _matcher()
    tensor0 = numpy_image_to_torch(array0).to(device)
    tensor1 = numpy_image_to_torch(array1).to(device)
    with torch.inference_mode():
        features0, features1, matches01 = match_pair(
            extractor, matcher, tensor0, tensor1, resize=int(resize)
        )

    matches = matches01["matches"].cpu().numpy()
    scores = matches01["scores"].cpu().numpy()
    order = np.argsort(scores)[::-1][: int(max_matches)]
    matches, scores = matches[order], scores[order]
    points0 = features0["keypoints"].cpu().numpy()[matches[:, 0]]
    points1 = features1["keypoints"].cpu().numpy()[matches[:, 1]]
    result = _draw_matches(array0, array1, points0, points1, scores)
    metadata = {
        "device": device,
        "keypoints_image0": int(features0["keypoints"].shape[0]),
        "keypoints_image1": int(features1["keypoints"].shape[0]),
        "matches_detected": int(matches01["matches"].shape[0]),
        "matches_visualized": int(matches.shape[0]),
        "mean_score": round(float(scores.mean()), 4) if len(scores) else 0.0,
    }
    return result, metadata


def build_demo():
    import gradio as gr

    with gr.Blocks(theme=gr.themes.Soft(), title="MambaGlue Image Matching") as demo:
        gr.Markdown(
            "# MambaGlue Image Matching\n"
            "Upload two views of the same scene. SuperPoint extracts features and "
            "MambaGlue matches them; green lines have higher confidence."
        )
        with gr.Row():
            image0 = gr.Image(label="Image 1", type="pil")
            image1 = gr.Image(label="Image 2", type="pil")
        with gr.Row():
            max_keypoints = gr.Slider(
                128, 2048, value=512, step=128, label="Maximum keypoints"
            )
            resize = gr.Slider(
                320, 1280, value=640, step=64, label="Longest image edge"
            )
            max_matches = gr.Slider(
                20, 500, value=150, step=10, label="Matches to display"
            )
        run_button = gr.Button("Match images", variant="primary")
        output = gr.Image(label="MambaGlue matches", type="numpy")
        stats = gr.JSON(label="Run summary")
        run_button.click(
            match_images,
            inputs=[image0, image1, max_keypoints, resize, max_matches],
            outputs=[output, stats],
        )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
