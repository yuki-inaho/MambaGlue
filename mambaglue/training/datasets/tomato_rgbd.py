"""glue-factory dataset for the vggt-omega ``colmap_rgbd_v1`` staging layout.

Reads ``<root>/dataset.json``, ``scenes/<scene>/cameras.npz`` and
``scenes/<scene>/sequences.npz`` and yields two-view samples with metric depth
(millimetres converted to metres using the manifest unit, never guessed) and the
camera poses needed by the ``depth_matcher`` ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from gluefactory.datasets import base_dataset
from gluefactory.geometry import reconstruction
from gluefactory.utils import preprocess
from kornia.color import rgb_to_grayscale

from ...rgbd import normalize_depth

DATASET_FORMAT = "colmap_rgbd_v1"
_UNIT_TO_METRES = {"millimeters": 1e-3, "mm": 1e-3, "meters": 1.0, "m": 1.0}
_INVALID_RAW = (0, 65535)


def _load_depth_metres(path: Path, scale: float) -> torch.Tensor:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"cannot read depth image: {path}")
    depth = raw.astype(np.float32) * scale
    invalid = np.zeros(raw.shape, dtype=bool)
    for value in _INVALID_RAW:
        invalid |= raw == value
    depth[invalid] = 0.0
    return torch.from_numpy(depth)


class _SplitView(torch.utils.data.Dataset):
    """A per-split view so train/val loaders do not share the item list."""

    def __init__(self, parent: "TomatoRgbdDataset", split: str):
        self.parent = parent
        self.items = list(parent.items_by_split[split])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.parent.read_pair(self.items[index])


class TomatoRgbdDataset(base_dataset.BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "root": "???",
        "scene": "scene_000000",
        "train_split": 0,
        "val_split": 1,
        "test_split": 2,
        "preprocessing": {
            "resize": 640,
            "side": "long",
            "antialias": False,
        },
        "num_workers": 2,
        "batch_size": 1,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "train_size": None,  # int: keep only the first N train pairs (smoke)
        "val_size": None,
        "test_size": None,
    }

    def _init(self, conf):
        self.root = Path(str(conf.root)).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"staging root does not exist: {self.root}")

        metadata = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        if metadata.get("format") != DATASET_FORMAT:
            raise ValueError(
                f"expected format {DATASET_FORMAT!r}, "
                f"got {metadata.get('format')!r}"
            )
        unit = metadata.get("depth", {}).get("unit")
        if unit not in _UNIT_TO_METRES:
            raise ValueError(
                f"unsupported depth unit {unit!r} in {self.root / 'dataset.json'}"
            )
        self.depth_scale = _UNIT_TO_METRES[unit]

        self.scene_root = self.root / "scenes" / conf.scene
        cameras = np.load(self.scene_root / "cameras.npz")
        sequences = np.load(self.scene_root / "sequences.npz")

        self.intrinsics = np.asarray(cameras["intrinsics"], dtype=np.float64)
        self.extrinsics_w2c = np.asarray(cameras["extrinsics_w2c"], dtype=np.float64)
        self.frame_ids = np.asarray(cameras["frame_ids"]).astype(int).tolist()

        rows = np.asarray(sequences["sequences"])
        split_ids = np.asarray(sequences["split_ids"]).astype(int)
        self.items_by_split = {}
        for name, split_id in (
            ("train", int(conf.train_split)),
            ("val", int(conf.val_split)),
            ("test", int(conf.test_split)),
        ):
            self.items_by_split[name] = rows[split_ids == split_id].tolist()
        if not self.items_by_split["test"]:
            self.items_by_split["test"] = list(self.items_by_split["val"])

        for split in ("train", "val", "test"):
            limit = conf.get(f"{split}_size")
            if limit is not None:
                limit = int(limit)
                if limit < 1:
                    raise ValueError(f"{split}_size must be positive, got {limit}")
                self.items_by_split[split] = self.items_by_split[split][:limit]

        self.preprocessor = preprocess.ImagePreprocessor(conf.preprocessing)

    def get_dataset(self, split: str, epoch: int = 0):
        if split not in self.items_by_split:
            raise ValueError(f"unknown split {split!r}")
        return _SplitView(self, split)

    def _read_view(self, index: int) -> dict:
        stem = f"frame_{index:06d}.png"
        image = preprocess.load_image(self.scene_root / "rgb" / stem)
        data = self.preprocessor(image)

        camera = reconstruction.Camera.from_calibration_matrix(
            torch.from_numpy(self.intrinsics[index].astype(np.float32))
        )
        data["camera"] = camera.compose_image_transform(data["transform"])

        extrinsics = self.extrinsics_w2c[index]
        data["T_w2cam"] = reconstruction.Pose.from_Rt(
            torch.from_numpy(extrinsics[:, :3].astype(np.float32)),
            torch.from_numpy(extrinsics[:, 3].astype(np.float32)),
        )

        depth = _load_depth_metres(self.scene_root / "depth" / stem, self.depth_scale)
        depth = self.preprocessor.interpolate(
            depth[None], data["transform"], data["image"].shape[-2:], mode="nearest"
        )[0]
        if depth.shape[-2:] != data["image"].shape[-2:]:
            raise ValueError(f"depth/image size mismatch for {stem}")
        data["depth"] = depth
        data["valid_depth"] = (depth > 0).float()
        data["name"] = stem

        # The training extractor consumes RGB-D directly: (2, H, W) = grayscale
        # + log-normalized depth. Keep ``depth`` (metres) for the ground truth.
        depth_map = depth.numpy()
        if depth_map.ndim == 3:
            depth_map = depth_map[0]
        valid_map = depth_map > 0.0
        normalized = torch.from_numpy(normalize_depth(depth_map, valid_map)).unsqueeze(
            0
        )
        gray = rgb_to_grayscale(data["image"].float())
        data["image"] = torch.cat([gray, normalized], dim=0)
        return data

    def read_pair(self, row) -> dict:
        frames = [int(index) for index in row]
        if len(frames) < 2:
            raise ValueError(f"sequence {frames} has fewer than two frames")
        data = {
            "view0": self._read_view(frames[0]),
            "view1": self._read_view(frames[1]),
        }
        data["T_0to1"] = data["view1"]["T_w2cam"] @ data["view0"]["T_w2cam"].inv()
        data["name"] = f"{frames[0]:06d}-{frames[1]:06d}"
        data["query_name"] = f"frame_{frames[0]:06d}.png"
        data["references"] = [f"frame_{frames[1]:06d}.png"]
        data["scene"] = str(self.conf.scene)
        data["nviews"] = 2
        return data

    def __len__(self) -> int:
        return len(self.items_by_split["train"])

    def __getitem__(self, index: int):
        return self.read_pair(self.items_by_split["train"][index])
