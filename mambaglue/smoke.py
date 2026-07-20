"""Download a public image pair and run the released MambaGlue checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


SAMPLES = {
    "sacre_coeur1.jpg": {
        "url": "https://raw.githubusercontent.com/cvg/LightGlue/eb42fee2d71449efb0aa5c10549752b5d75384d8/assets/sacre_coeur1.jpg",
        "sha256": "d274c3780b671fe637040bc976166962e2e48f8f512fb7ce939b71b07c85af83",
    },
    "sacre_coeur2.jpg": {
        "url": "https://raw.githubusercontent.com/cvg/LightGlue/eb42fee2d71449efb0aa5c10549752b5d75384d8/assets/sacre_coeur2.jpg",
        "sha256": "c6c0339235087c7c6c38faca5c6545250699bb59120b94b06f0fcf8227fbe64e",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_sample(destination: Path, url: str, expected_sha256: str) -> None:
    if destination.exists() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MambaGlue-smoke"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if _sha256(temporary) != expected_sha256:
                raise RuntimeError(f"Checksum mismatch while downloading {destination.name}.")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def run(root: Path, max_keypoints: int) -> dict[str, object]:
    root = root.resolve()
    sample_dir = root / "data" / "smoke"
    for filename, metadata in SAMPLES.items():
        _download_sample(sample_dir / filename, metadata["url"], metadata["sha256"])

    # Keep downloaded official checkpoints with the reproducible local setup.
    os.environ.setdefault("TORCH_HOME", str(root / ".cache" / "torch"))

    import torch

    from mambaglue import MambaGlue, SuperPoint, match_pair
    from mambaglue.utils import load_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = MambaGlue(features="superpoint").eval().to(device)
    image0 = load_image(sample_dir / "sacre_coeur1.jpg").to(device)
    image1 = load_image(sample_dir / "sacre_coeur2.jpg").to(device)
    with torch.inference_mode():
        feats0, feats1, matches01 = match_pair(
            extractor, matcher, image0, image1, resize=320
        )

    matches = matches01["matches"]
    result = {
        "device": str(device),
        "torch": torch.__version__,
        "keypoints0": int(feats0["keypoints"].shape[0]),
        "keypoints1": int(feats1["keypoints"].shape[0]),
        "matches": int(matches.shape[0]),
        "sample_dir": str(sample_dir),
    }
    if result["keypoints0"] == 0 or result["keypoints1"] == 0:
        raise RuntimeError(f"No keypoints found: {result}")
    if result["matches"] == 0:
        raise RuntimeError(f"No matches found: {result}")

    output = root / "outputs" / "smoke_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Artifact root")
    parser.add_argument("--max-keypoints", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.max_keypoints), indent=2))


if __name__ == "__main__":
    main()
