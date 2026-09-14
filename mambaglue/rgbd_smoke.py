"""End-to-end RGB-D smoke test on a real staged scene.

Matches one RGB-D pair with ``SuperPointRGBD`` + ``MambaGlue`` and reports the
2D matches plus the subset that is depth-consistent in 3D. The depth unit is
read from the staging ``dataset.json`` (never guessed); pass ``--depth-unit``
to override explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .mambaglue import MambaGlue
from .rgbd import load_depth, match_pair_rgbd
from .superpoint import SuperPointRGBD
from .utils import load_image

_UNIT_ALIASES = {"millimeters": "mm", "mm": "mm", "meters": "m", "m": "m"}


def resolve_depth_unit(scene: Path, explicit: str | None) -> str:
    """Return ``"mm"``/``"m"`` from ``--depth-unit`` or the dataset manifest."""
    if explicit is not None:
        if explicit not in ("mm", "m"):
            raise ValueError(f"--depth-unit must be 'mm' or 'm', got {explicit!r}")
        return explicit
    for parent in (scene, *scene.parents):
        manifest = parent / "dataset.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            unit = metadata.get("depth", {}).get("unit")
            if unit not in _UNIT_ALIASES:
                raise ValueError(f"unsupported depth unit {unit!r} in {manifest}")
            return _UNIT_ALIASES[unit]
    raise ValueError(
        "no dataset.json found; pass --depth-unit explicitly (no implicit unit)"
    )


def _load_frame(scene: Path, index: int, unit: str):
    rgb = load_image(scene / "rgb" / f"frame_{index:06d}.png")
    depth_m, valid = load_depth(scene / "depth" / f"frame_{index:06d}.png", unit=unit)
    return rgb, depth_m, valid


def run(
    scene: Path | str,
    frame0: int = 0,
    frame1: int = 1,
    *,
    depth_unit: str | None = None,
    output: Path | str = "outputs/rgbd_smoke_test.json",
    max_keypoints: int = 256,
    resize: int = 512,
    device: str | None = None,
    threshold_m: float = 0.02,
) -> dict:
    scene = Path(scene).expanduser().resolve()
    unit = resolve_depth_unit(scene, depth_unit)
    intrinsics = np.load(scene / "cameras.npz")["intrinsics"]

    rgb0, depth0, valid0 = _load_frame(scene, frame0, unit)
    rgb1, depth1, valid1 = _load_frame(scene, frame1, unit)
    K0 = np.asarray(intrinsics[frame0], dtype=np.float64)
    K1 = np.asarray(intrinsics[frame1], dtype=np.float64)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPointRGBD(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = MambaGlue(features="superpoint").eval().to(device)

    with torch.inference_mode():
        result = match_pair_rgbd(
            extractor,
            matcher,
            rgb0.to(device),
            depth0,
            valid0,
            K0,
            rgb1.to(device),
            depth1,
            valid1,
            K1,
            device=device,
            threshold_m=threshold_m,
            resize=resize,
        )

    matches = result["matches01"]["matches"]
    match_count = int(matches.shape[0])
    inliers = result["inliers"]
    rmse = result["residual_rmse"]
    summary = {
        "scene": str(scene),
        "frame0": frame0,
        "frame1": frame1,
        "depth_unit": unit,
        "device": device,
        "resize": resize,
        "keypoints0": int(result["feats0"]["keypoints"].shape[0]),
        "keypoints1": int(result["feats1"]["keypoints"].shape[0]),
        "matches": match_count,
        "valid_matches": int(result["valid_matches"]),
        "inliers": int(np.asarray(inliers).sum()),
        "residual_rmse": float(rmse) if np.isfinite(rmse) else None,
        "depth_valid_fraction0": float(np.asarray(valid0).mean()),
        "depth_valid_fraction1": float(np.asarray(valid1).mean()),
    }
    if summary["keypoints0"] == 0 or summary["keypoints1"] == 0:
        raise RuntimeError(f"no keypoints extracted: {summary}")
    if summary["matches"] == 0:
        raise RuntimeError(f"no matches found: {summary}")

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        required=True,
        help="Staged scene directory (rgb/depth/cameras.npz)",
    )
    parser.add_argument("--frame0", type=int, default=0)
    parser.add_argument("--frame1", type=int, default=1)
    parser.add_argument("--depth-unit", choices=("mm", "m"), default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/rgbd_smoke_test.json")
    )
    parser.add_argument("--max-keypoints", type=int, default=256)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.scene,
                args.frame0,
                args.frame1,
                depth_unit=args.depth_unit,
                output=args.output,
                max_keypoints=args.max_keypoints,
                resize=args.resize,
                device=args.device,
                threshold_m=args.threshold_m,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
