"""RGB-D input contract tests for ``mambaglue.rgbd``.

Unit handling is explicit (``unit="mm"`` or ``"m"``); invalid raw samples
(0 and 65535) must be masked and never silently treated as metric depth.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch


def _write_depth(tmp_path, values: np.ndarray) -> object:
    assert values.dtype == np.uint16
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), values)
    return path


def test_depth_preprocess_units_and_invalid(tmp_path):
    from mambaglue.rgbd import load_depth, normalize_depth

    raw = np.array([[0, 100, 1000, 65535]], dtype=np.uint16)
    path = _write_depth(tmp_path, raw)

    depth_m, valid = load_depth(path, unit="mm")
    assert depth_m.dtype == np.float32
    assert depth_m.shape == (1, 4)
    assert valid.tolist() == [[False, True, True, False]]
    assert depth_m[0, 0] == 0.0
    assert depth_m[0, 1] == pytest.approx(0.1, abs=1e-6)
    assert depth_m[0, 2] == pytest.approx(1.0, abs=1e-6)
    assert depth_m[0, 3] == 0.0

    depth_m_m, valid_m = load_depth(path, unit="m")
    assert depth_m_m[0, 1] == pytest.approx(100.0, abs=1e-4)
    assert depth_m_m[0, 2] == pytest.approx(1000.0, abs=1e-3)
    assert valid_m.tolist() == valid.tolist()

    norm = normalize_depth(depth_m, valid, mode="log")
    assert norm.shape == (1, 4)
    assert norm.dtype == np.float32
    assert float(norm[valid].min()) >= 0.0
    assert float(norm[valid].max()) <= 1.0
    assert norm[0, 0] == 0.0
    assert norm[0, 3] == 0.0


def test_depth_unit_is_explicit(tmp_path):
    from mambaglue.rgbd import load_depth

    path = _write_depth(tmp_path, np.array([[10, 20]], dtype=np.uint16))
    with pytest.raises(TypeError):
        load_depth(path)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        load_depth(path, unit="cm")


def test_depth_normalize_fail_closed_without_valid(tmp_path):
    from mambaglue.rgbd import load_depth, normalize_depth

    path = _write_depth(tmp_path, np.zeros((2, 2), dtype=np.uint16))
    depth_m, valid = load_depth(path, unit="mm")
    assert not valid.any()
    with pytest.raises(ValueError):
        normalize_depth(depth_m, valid, mode="log")


def test_normalize_rejects_unknown_mode(tmp_path):
    from mambaglue.rgbd import load_depth, normalize_depth

    path = _write_depth(tmp_path, np.array([[10, 20]], dtype=np.uint16))
    depth_m, valid = load_depth(path, unit="mm")
    with pytest.raises(ValueError):
        normalize_depth(depth_m, valid, mode="sqrt")


def test_to_rgbd_input_channels(tmp_path):
    from mambaglue.rgbd import load_depth, to_rgbd_input

    path = _write_depth(tmp_path, np.array([[0, 500, 1000, 1500]], dtype=np.uint16))
    depth_m, valid = load_depth(path, unit="mm")
    rgb = torch.zeros(3, 1, 4, dtype=torch.float32)
    rgb[0] = 1.0

    rgbd, mask = to_rgbd_input(rgb, depth_m, valid, mode="log")
    assert rgbd.shape == (2, 1, 4)
    assert mask.shape == (1, 4)
    assert not bool(mask[0, 0])
    assert bool(mask[0, 1])
    assert float(rgbd[1, 0, 0]) == 0.0
    assert float(rgbd[1, 0, 1]) < float(rgbd[1, 0, 2]) < float(rgbd[1, 0, 3])


def test_superpoint_rgbd_stem_channels():
    import torch.nn as nn

    from mambaglue.superpoint import (
        SuperPointRGBD,
        build_superpoint_convs,
        expand_superpoint_state_dict,
    )

    source = build_superpoint_convs(nn.Module(), 1).state_dict()
    weight = source["conv1a.weight"]
    assert tuple(weight.shape[:2]) == (64, 1)

    expanded = expand_superpoint_state_dict(source)
    assert tuple(expanded["conv1a.weight"].shape) == (64, 2, 3, 3)
    assert torch.equal(expanded["conv1a.weight"][:, :1], weight)
    assert torch.equal(expanded["conv1a.weight"][:, 1:], weight)
    assert torch.equal(expanded["conv1a.bias"], source["conv1a.bias"])

    model = SuperPointRGBD(source_state_dict=source).eval()
    assert tuple(model.conv1a.weight.shape) == (64, 2, 3, 3)
    assert torch.equal(model.conv1a.weight[:, :1], weight)
    assert torch.equal(model.conv1a.weight[:, 1:], weight)

    image = torch.rand(1, 2, 64, 64)
    with torch.inference_mode():
        feats = model({"image": image})
    assert feats["descriptors"].shape[0] == 1
    assert feats["descriptors"].shape[-1] == 256
    assert feats["keypoints"].shape[0] == 1


def test_plain_superpoint_stem_stays_single_channel(monkeypatch):
    import torch.nn as nn

    from mambaglue import superpoint as sp

    source = sp.build_superpoint_convs(nn.Module(), 1).state_dict()
    monkeypatch.setattr(
        sp.torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: source
    )
    model = sp.SuperPoint()
    assert tuple(model.conv1a.weight.shape[:2]) == (64, 1)


def _rigid(angle: float, translation):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rotation, np.asarray(translation, dtype=np.float64)


def test_filter_matches_3d_rejects_outliers():
    from mambaglue.rgbd import filter_matches_3d

    rng = np.random.default_rng(0)
    count = 200
    points0 = rng.uniform(-0.5, 0.5, size=(count, 3))
    rotation, translation = _rigid(0.3, [0.1, -0.05, 0.2])
    points1 = points0 @ rotation.T + translation + rng.normal(0.0, 0.003, (count, 3))
    outliers = rng.choice(count, size=60, replace=False)
    points1[outliers] += rng.normal(0.0, 0.2, (len(outliers), 3))

    inliers, transform, residual_rmse = filter_matches_3d(points0, points1)
    assert inliers.shape == (count,)
    assert inliers.sum() >= 0.9 * (count - len(outliers))
    assert not inliers[outliers].any()
    assert residual_rmse <= 0.02
    assert transform.shape == (4, 4)


def test_backproject_masks_invalid_depth():
    from mambaglue.rgbd import backproject

    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    valid = np.array([[False, True], [True, True]])
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    points_px = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    xyz, mask = backproject(points_px, depth, valid, K)
    assert xyz.shape == (3, 3)
    assert mask.tolist() == [False, True, True]
    assert np.allclose(xyz[2], [0.0, 0.0, 3.0])  # nearest-neighbour depth sample


def test_match_pair_rgbd_drops_invalid_depth_matches():
    import torch

    from mambaglue.rgbd import match_pair_rgbd

    class _StubExtractor:
        def extract(self, image, **conf):
            width, height = image.shape[-1], image.shape[-2]
            keypoints = torch.tensor(
                [[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
            )
            descriptors = torch.zeros(1, 3, 256)
            descriptors[0, 0, 0] = 1.0
            descriptors[0, 1, 1] = 1.0
            descriptors[0, 2, 2] = 1.0
            return {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "image_size": torch.tensor([[width, height]]),
            }

    class _StubMatcher:
        def __call__(self, data):
            # MambaGlue returns ``matches`` as one [Si, 2] tensor per batch item.
            return {"matches": [torch.tensor([[0, 0], [1, 1], [2, 2]])]}

    depth0 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    valid0 = np.array([[True, True], [True, True]])
    # keypoint (1, 0) in image1 sits on an invalid depth sample -> dropped
    depth1 = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    valid1 = np.array([[True, False], [True, True]])
    K = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    rgb = torch.rand(3, 2, 2)

    out = match_pair_rgbd(
        _StubExtractor(),
        _StubMatcher(),
        rgb,
        depth0,
        valid0,
        K,
        rgb,
        depth1,
        valid1,
        K,
        device="cpu",
    )
    assert "matches3d" in out and "inliers" in out and "residual_rmse" in out
    assert out["valid_matches"] == 2
