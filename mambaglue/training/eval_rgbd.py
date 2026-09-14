"""Evaluate RGB-D matching quality on the tomato staging validation split.

Compares two checkpoints (released pretrained weights vs a trained
``checkpoint_best.tar``) on the same validation pairs and reports the 2D match
count, the depth-consistent inlier rate, and the rigid residual RMSE.

Usage::

    uv run --no-sync python -m mambaglue.training.eval_rgbd \
      --conf mambaglue/training/configs/superpoint_rgbd+mambaglue_tomato.yaml \
      --data-root /path/to/<staging-root> \
      --checkpoint /path/to/checkpoint_best.tar \
      --output outputs/rgbd_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from gluefactory.datasets.base_dataset import collate
from gluefactory.models import get_model
from gluefactory.utils import misc
from omegaconf import OmegaConf

from mambaglue.rgbd import backproject, filter_matches_3d
from mambaglue.training.datasets.tomato_rgbd import TomatoRgbdDataset


def _build_model(conf: OmegaConf, device: str):
    model_conf = OmegaConf.to_container(conf.model, resolve=True)
    model_conf["ground_truth"] = {"name": None}
    model = get_model(model_conf["name"])(model_conf).to(device).eval()
    return model


def _load_trained(model, checkpoint: Path) -> None:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    report = model.load_state_dict(state, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise ValueError(
            "checkpoint does not match the model: "
            f"missing={sorted(report.missing_keys)} "
            f"unexpected={sorted(report.unexpected_keys)}"
        )


def _pair_metrics(model, sample, device, threshold_m: float) -> dict:
    batch = misc.batch_to_device(collate([sample]), device, non_blocking=False)
    with torch.no_grad():
        pred = model(batch)

    matches = pred["matches0"][0].detach().cpu().numpy()
    keypoints0 = pred["keypoints0"][0].detach().cpu().numpy()
    keypoints1 = pred["keypoints1"][0].detach().cpu().numpy()

    depth0 = np.asarray(sample["view0"]["depth"], dtype=np.float64)
    depth1 = np.asarray(sample["view1"]["depth"], dtype=np.float64)
    valid0 = depth0 > 0.0
    valid1 = depth1 > 0.0
    K0 = np.asarray(sample["view0"]["camera"].K.detach().cpu(), dtype=np.float64)
    K1 = np.asarray(sample["view1"]["camera"].K.detach().cpu(), dtype=np.float64)

    match_count = int((matches >= 0).sum())
    if match_count == 0:
        return {
            "matches": 0,
            "depth_valid_matches": 0,
            "inliers": 0,
            "inlier_rate": 0.0,
            "residual_rmse": None,
        }

    selected = matches >= 0
    pixels0 = keypoints0[selected]
    pixels1 = keypoints1[matches[selected]]
    xyz0, ok0 = backproject(pixels0, depth0, valid0, K0)
    xyz1, ok1 = backproject(pixels1, depth1, valid1, K1)
    keep = ok0 & ok1
    depth_valid_matches = int(keep.sum())

    inliers = 0
    rmse = None
    if depth_valid_matches >= 3:
        mask, _, residual = filter_matches_3d(
            xyz0[keep], xyz1[keep], threshold_m=threshold_m
        )
        inliers = int(mask.sum())
        rmse = float(residual)
    return {
        "matches": match_count,
        "depth_valid_matches": depth_valid_matches,
        "inliers": inliers,
        "inlier_rate": (inliers / depth_valid_matches) if depth_valid_matches else 0.0,
        "residual_rmse": rmse,
    }


def _aggregate(rows: list[dict]) -> dict:
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "pairs": len(rows),
        "matches_mean": mean("matches"),
        "depth_valid_matches_mean": mean("depth_valid_matches"),
        "inliers_mean": mean("inliers"),
        "inlier_rate_mean": mean("inlier_rate"),
        "residual_rmse_mean": mean("residual_rmse"),
    }


def run(
    conf_path: Path | str,
    data_root: Path | str,
    *,
    checkpoint: Path | str | None,
    output: Path | str = "outputs/rgbd_eval.json",
    split: str = "val",
    num_pairs: int = 50,
    device: str | None = None,
    threshold_m: float = 0.02,
    seed: int = 42,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    conf = OmegaConf.load(str(conf_path))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TomatoRgbdDataset(
        {
            **OmegaConf.to_container(conf.data, resolve=True),
            "root": str(Path(data_root).expanduser()),
        }
    )
    view = dataset.get_dataset(split)
    indices = list(range(min(num_pairs, len(view))))

    model = _build_model(conf, device)
    pretrained_metrics = [
        _pair_metrics(model, view[i], device, threshold_m) for i in indices
    ]

    trained_metrics = None
    if checkpoint is not None:
        _load_trained(model, Path(checkpoint).expanduser())
        trained_metrics = [
            _pair_metrics(model, view[i], device, threshold_m) for i in indices
        ]

    result = {
        "conf": str(conf_path),
        "data_root": str(Path(data_root).expanduser()),
        "split": split,
        "num_pairs": len(indices),
        "depth_unit": "metres (converted from the dataset manifest)",
        "threshold_m": threshold_m,
        "seed": seed,
        "pretrained": _aggregate(pretrained_metrics),
        "trained": _aggregate(trained_metrics) if trained_metrics is not None else None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/rgbd_eval.json"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.conf,
                args.data_root,
                checkpoint=args.checkpoint,
                output=args.output,
                split=args.split,
                num_pairs=args.num_pairs,
                device=args.device,
                threshold_m=args.threshold_m,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
