"""Explicit RGB-D preprocessing for MambaGlue.

The depth unit is never inferred: ``load_depth`` requires ``unit`` and only
accepts ``"mm"`` or ``"m"``. Raw invalid samples (0 and 65535) are masked and
forced to 0.0 so they cannot be mistaken for metric depth. ``to_rgbd_input``
returns a 2-channel tensor (grayscale, normalized depth) plus the validity mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.color import rgb_to_grayscale

#: Raw values that mark a missing/invalid depth sample in the capture records.
INVALID_RAW_VALUES = (0, 65535)

_UNIT_SCALE = {"mm": 1e-3, "m": 1.0}
_NORMALIZE_MODES = ("log", "linear")


def load_depth(path: str | Path, unit: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a uint16 depth image as float32 metres plus a validity mask.

    ``unit`` must be ``"mm"`` or ``"m"``; unknown or implicit units raise.
    Invalid samples (0 / 65535) are set to 0.0 and marked ``False`` in the mask.
    """
    if unit not in _UNIT_SCALE:
        raise ValueError(f"unit must be one of {sorted(_UNIT_SCALE)}, got {unit!r}")
    image = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read depth image: {path}")
    if image.ndim != 2:
        raise ValueError(f"depth image must be single channel, got shape {image.shape}")

    raw = image.astype(np.float32)
    valid = np.ones(raw.shape, dtype=bool)
    for invalid in INVALID_RAW_VALUES:
        valid &= raw != float(invalid)

    depth_m = raw * _UNIT_SCALE[unit]
    depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
    return depth_m, valid


def normalize_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    mode: str = "log",
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    """Robustly map valid metric depth to [0, 1]; invalid samples become 0.0.

    ``mode="log"`` is monotonic in depth (near -> 0, far -> 1). At least one
    valid pixel is required; an all-invalid input is a hard error.
    """
    if mode not in _NORMALIZE_MODES:
        raise ValueError(f"mode must be one of {_NORMALIZE_MODES}, got {mode!r}")

    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if depth_m.shape != valid.shape:
        raise ValueError(
            f"depth/valid shape mismatch: {depth_m.shape} vs {valid.shape}"
        )

    values = depth_m[valid]
    if values.size == 0:
        raise ValueError("normalize_depth requires at least one valid pixel")

    low = float(np.quantile(values, quantiles[0]))
    high = float(np.quantile(values, quantiles[1]))
    if high <= low:
        high = low + 1e-6

    clipped = np.clip(depth_m, low, high)
    if mode == "log":
        numerator = np.log1p(clipped) - np.log1p(low)
        denominator = np.log1p(high) - np.log1p(low)
    else:
        numerator = clipped - low
        denominator = high - low

    normalized = numerator / max(denominator, 1e-12)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.where(valid, normalized, 0.0).astype(np.float32)
    return normalized


def depth_to_channel(
    depth_m: np.ndarray, valid: np.ndarray, mode: str = "log"
) -> torch.Tensor:
    """Return a (1, H, W) float32 normalized depth channel with invalid = 0."""
    normalized = normalize_depth(depth_m, valid, mode=mode)
    return torch.from_numpy(normalized).unsqueeze(0).float()


def to_rgbd_input(
    rgb: torch.Tensor,
    depth_m: np.ndarray,
    valid: np.ndarray,
    mode: str = "log",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack grayscale RGB and normalized depth into a (2, H, W) input.

    ``rgb`` must be a float tensor with shape (3, H, W) or (1, 3, H, W) in
    [0, 1]. The returned mask is a (H, W) bool tensor.
    """
    if not isinstance(rgb, torch.Tensor):
        raise TypeError(f"rgb must be a torch.Tensor, got {type(rgb).__name__}")
    if rgb.dim() == 4:
        rgb = rgb[0]
    if rgb.dim() != 3 or rgb.shape[0] != 3:
        raise ValueError(f"rgb must have shape (3, H, W), got {tuple(rgb.shape)}")

    gray = rgb_to_grayscale(rgb.float())[0]
    depth = depth_to_channel(depth_m, valid, mode=mode)[0].to(gray.device)
    if gray.shape != depth.shape:
        raise ValueError(f"rgb/depth shape mismatch: {gray.shape} vs {depth.shape}")

    rgbd = torch.stack([gray, depth], dim=0)
    mask = torch.from_numpy(np.asarray(valid, dtype=bool))
    return rgbd, mask


def backproject(
    points_px: np.ndarray, depth_m: np.ndarray, valid: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project pixel coordinates to camera-frame metres (nearest sample).

    Returns ``(xyz[N, 3], ok[N])``; rows where the depth sample is invalid or
    non-positive are marked ``False`` and zeroed.
    """
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if depth_m.ndim != 2 or valid.shape != depth_m.shape:
        raise ValueError("depth_m and valid must be 2D arrays of equal shape")
    height, width = depth_m.shape

    cols = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    z = depth_m[rows, cols]
    ok = valid[rows, cols] & (z > 0.0)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xyz = np.zeros((len(points), 3), dtype=np.float64)
    xyz[:, 0] = (points[:, 0] - cx) / fx * z
    xyz[:, 1] = (points[:, 1] - cy) / fy * z
    xyz[:, 2] = z
    xyz[~ok] = 0.0
    return xyz, ok


def _umeyama_rigid(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation/translation mapping ``src`` onto ``dst`` (no scale)."""
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    covariance = (src - src_centroid).T @ (dst - dst_centroid)
    u, _, vt = np.linalg.svd(covariance)
    determinant = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, determinant])
    rotation = vt.T @ correction @ u.T
    translation = dst_centroid - rotation @ src_centroid
    return rotation, translation


def filter_matches_3d(
    points0: np.ndarray,
    points1: np.ndarray,
    threshold_m: float = 0.02,
    iterations: int = 5,
    min_inliers: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Iteratively reweighted rigid fit; returns ``(inliers, T, residual_rmse)``."""
    pts0 = np.asarray(points0, dtype=np.float64)
    pts1 = np.asarray(points1, dtype=np.float64)
    if pts0.shape != pts1.shape or pts0.ndim != 2 or pts0.shape[1] != 3:
        raise ValueError(
            f"points must be matching (N, 3), got {pts0.shape} / {pts1.shape}"
        )
    if len(pts0) < min_inliers:
        raise ValueError(f"need at least {min_inliers} matches, got {len(pts0)}")

    active = np.ones(len(pts0), dtype=bool)
    rotation = np.eye(3)
    translation = np.zeros(3)
    # Annealed robust reweighting: a plain fixed threshold can reject every true
    # inlier when outliers bias the first least-squares fit, so start with a
    # MAD-based cutoff and only tighten to ``threshold_m`` once converged.
    for _ in range(iterations):
        rotation, translation = _umeyama_rigid(pts0[active], pts1[active])
        residuals = np.linalg.norm(pts0 @ rotation.T + translation - pts1, axis=1)
        median = float(np.median(residuals[active]))
        mad = float(np.median(np.abs(residuals[active] - median)))
        scale = max(1.4826 * mad, 1e-9)
        cutoff = max(threshold_m, 3.0 * scale)
        new_active = residuals <= cutoff
        if new_active.sum() < min_inliers:
            break
        if np.array_equal(new_active, active):
            active = new_active
            break
        active = new_active

    rotation, translation = _umeyama_rigid(pts0[active], pts1[active])
    residuals = np.linalg.norm(pts0 @ rotation.T + translation - pts1, axis=1)
    final_active = residuals <= threshold_m
    if final_active.sum() >= min_inliers:
        active = final_active
        rotation, translation = _umeyama_rigid(pts0[active], pts1[active])
        residuals = np.linalg.norm(pts0 @ rotation.T + translation - pts1, axis=1)
    residual_rmse = float(np.sqrt(np.mean(residuals[active] ** 2)))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return active, transform, residual_rmse


def match_pair_rgbd(
    extractor,
    matcher,
    rgb0: torch.Tensor,
    depth0: np.ndarray,
    valid0: np.ndarray,
    K0: np.ndarray,
    rgb1: torch.Tensor,
    depth1: np.ndarray,
    valid1: np.ndarray,
    K1: np.ndarray,
    *,
    device: str = "cpu",
    threshold_m: float = 0.02,
    min_inliers: int = 3,
    **preprocess,
) -> dict:
    """Match RGB-D frames and keep only depth-consistent 3D correspondences.

    Matches whose depth is invalid in either view are always dropped. The
    surviving 3D pairs are filtered with ``filter_matches_3d``; if fewer than
    ``min_inliers`` survive, ``inliers`` is empty and ``residual_rmse`` is NaN
    (no silent success).
    """
    from .utils import match_pair

    rgbd0, _ = to_rgbd_input(rgb0, depth0, valid0)
    rgbd1, _ = to_rgbd_input(rgb1, depth1, valid1)
    feats0, feats1, matches01 = match_pair(
        extractor, matcher, rgbd0, rgbd1, device=device, **preprocess
    )

    matches = matches01["matches"]
    matches = (
        matches.detach().cpu().numpy()
        if torch.is_tensor(matches)
        else np.asarray(matches)
    )
    matches = matches.reshape(-1, 2)
    keypoints0 = np.asarray(feats0["keypoints"].detach().cpu()).reshape(-1, 2)
    keypoints1 = np.asarray(feats1["keypoints"].detach().cpu()).reshape(-1, 2)

    result = {
        "feats0": feats0,
        "feats1": feats1,
        "matches01": matches01,
        "valid_matches": 0,
        "matches3d": np.zeros((0, 2, 3), dtype=np.float64),
        "inliers": np.zeros(0, dtype=bool),
        "residual_rmse": float("nan"),
    }
    if matches.size == 0:
        return result

    xyz0, ok0 = backproject(keypoints0[matches[:, 0]], depth0, valid0, K0)
    xyz1, ok1 = backproject(keypoints1[matches[:, 1]], depth1, valid1, K1)
    keep = ok0 & ok1
    valid_matches = int(keep.sum())
    result["valid_matches"] = valid_matches
    result["matches3d"] = np.stack([xyz0[keep], xyz1[keep]], axis=1)

    if valid_matches >= min_inliers:
        inliers, _, residual_rmse = filter_matches_3d(
            xyz0[keep], xyz1[keep], threshold_m=threshold_m, min_inliers=min_inliers
        )
        result["inliers"] = inliers
        result["residual_rmse"] = residual_rmse
    return result
