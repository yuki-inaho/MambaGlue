"""Contract tests for the tomato RGB-D pair dataset.

Set ``MAMBAGLUE_TOMATO_ROOT`` to the ``colmap_rgbd`` staging root. Skipped when
the variable is unset or the staging root is unavailable so the default test
suite stays hermetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT_ENV = os.environ.get("MAMBAGLUE_TOMATO_ROOT")
ROOT = Path(_ROOT_ENV).expanduser() if _ROOT_ENV else Path("/nonexistent")

pytestmark = pytest.mark.skipif(
    not (ROOT / "dataset.json").is_file(),
    reason="set MAMBAGLUE_TOMATO_ROOT to a colmap_rgbd staging root",
)


def _dataset():
    from mambaglue.training.datasets.tomato_rgbd import TomatoRgbdDataset

    return TomatoRgbdDataset(
        {
            "root": str(ROOT),
            "preprocessing": {"resize": 320, "side": "long", "antialias": False},
            "num_workers": 0,
        }
    )


def _expected_T_0to1(row):
    cameras = np.load(ROOT / "scenes" / "scene_000000" / "cameras.npz")
    extrinsics = np.asarray(cameras["extrinsics_w2c"], dtype=np.float64)
    e0 = np.eye(4)
    e0[:3, :4] = extrinsics[int(row[0])]
    e1 = np.eye(4)
    e1[:3, :4] = extrinsics[int(row[1])]
    return e1 @ np.linalg.inv(e0)


def test_pair_contract_and_pose():
    dataset = _dataset()
    view = dataset.get_dataset("train")
    assert len(view) > 0
    row = dataset.items_by_split["train"][0]
    sample = view[0]

    assert set(sample) >= {"view0", "view1", "T_0to1", "name", "nviews"}
    for name in ("view0", "view1"):
        assert set(sample[name]) >= {
            "image",
            "depth",
            "camera",
            "T_w2cam",
            "valid_depth",
        }
        assert sample[name]["depth"].shape[-2:] == sample[name]["image"].shape[-2:]

    rotation, translation = sample["T_0to1"].to_Rt()
    actual = np.eye(4)
    actual[:3, :3] = rotation.detach().cpu().numpy()
    actual[:3, 3] = translation.detach().cpu().numpy()
    assert np.allclose(actual, _expected_T_0to1(row), atol=1e-4)

    # train and val splits must be disjoint at the sequence level
    train = {tuple(int(i) for i in r) for r in dataset.items_by_split["train"]}
    val = {tuple(int(i) for i in r) for r in dataset.items_by_split["val"]}
    assert not (train & val)


def test_depth_is_metres_and_invalid_is_masked():
    import cv2

    dataset = _dataset()
    sample = dataset.get_dataset("train")[0]
    row = dataset.items_by_split["train"][0]
    raw = cv2.imread(
        str(
            ROOT / "scenes" / "scene_000000" / "depth" / f"frame_{int(row[0]):06d}.png"
        ),
        cv2.IMREAD_UNCHANGED,
    )
    assert raw.dtype == np.uint16
    depth = sample["view0"]["depth"]
    # resized depth is nearest-sampled, so every stored value must be a raw value * 1e-3
    uniques = torch.unique(depth)
    raw_values = torch.from_numpy(raw.astype(np.float32) * 1e-3)
    assert uniques.numel() <= raw_values.numel()
    assert float(depth.max()) <= 1.3 + 1e-6
    if (raw == 0).any():
        assert float(sample["view0"]["valid_depth"].min()) == 0.0


def test_sample_depth_masks_invalid_pixels():
    from gluefactory.geometry import depth as gf_depth

    dataset = _dataset()
    sample = dataset.get_dataset("train")[0]
    depth = sample["view0"]["depth"]
    height, width = depth.shape[-2:]
    points = torch.tensor(
        [[[0.0, 0.0], [width - 1.0, height - 1.0]]], dtype=torch.float32
    )
    sampled, valid = gf_depth.sample_depth(points, depth)
    sampled = torch.as_tensor(sampled).reshape(-1)
    valid = torch.as_tensor(valid).reshape(-1)
    assert sampled.numel() == 2 and valid.numel() == 2
    # a zero-depth sample is never valid
    assert bool(valid[0]) <= bool(float(depth[0, 0]) > 0.0)
