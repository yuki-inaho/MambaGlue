<p align="center">
  <h1 align="center"><a href="https://arxiv.org/abs/2502.00462"><ins>MambaGlue</ins></a> 🐍 @ICRA2025<br>Fast and Robust Local Feature Matching With Mamba</br></h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/kihwan-ryoo-54b68b224/">Kihwan Ryoo</a>
    ·
    <a href="https://scholar.google.com/citations?user=S1A3nbIAAAAJ&hl=ko&oi=ao/">Hyungtae Lim</a>
    ·
    <a href="https://scholar.google.com/citations?user=NrWfJ1gAAAAJ&hl=ko&oi=ao">Hyun Myung</a>
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2502.00462"><img src="https://img.shields.io/badge/arXiv-2502.00462-b31b1b.svg" alt="arXiv"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="https://hub.docker.com/r/rkh137/glue"><img src="https://img.shields.io/badge/Docker-rkh137%2Fglue-2496ed?logo=docker&logoColor=white" alt="Docker"></a>
</p>

<p align="center">
    <img src="assets/Visualization.png" alt="example" width=40%><img src=assets/demo_sacre.gif alt="animated" width="40%"/></a>
    <br>
    <em>MambaGlue is a hybrid neural network combining the Mamba and the Transformer architectures to match local features.<br></em>
</p>


## :open_book: Table of Contents
- [Overview](#mambaglue-snake)
- [Tested Environment](#desktop_computer-tested-environment)
- [Install](#keyboard-install)
- [Quickstart](#zap-quickstart)
- [Training (experimental reproduction)](#dart-training-experimental-reproduction)
- [Visualization with hloc (coming soon)](#magic_wand-visualization-and-evaluation-hloc-branch-coming-soon)
- [FAQ](#question-faq)
- [To Do](#clipboard-to-do)
- [Citation](#memo-citation)
- [License](#license)


## MambaGlue :snake:
The `main` branch contains:

- the standard MambaGlue model and inference utilities (`mambaglue/`); and
- an experimental training adapter for [Glue Factory](https://github.com/cvg/glue-factory) under `mambaglue/training/` — see [Training](#dart-training-experimental-reproduction).

SfM and visual-localization integration is planned for a separate `hloc` branch, built on [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization/).

> **Note:** the `hloc` branch is not yet pushed. The training pipeline is functional but has not yet been verified to reproduce the paper's reported numbers end-to-end — defaults are inherited from LightGlue. See [`#8`](https://github.com/url-kaist/MambaGlue/issues/8).


## :desktop_computer: Tested Environment
- Linux (Ubuntu 20.04)
- NVIDIA GPU (TITAN V, RTX 3080, or other Ampere/newer architectures)
- CUDA 11.8 + cuDNN 8
- PyTorch 2.1.0
- Python 3.10+ (3.8 is end-of-life and no longer supported)


## :keyboard: Install
The project is managed with [uv](https://docs.astral.sh/uv/). On Linux, the
lockfile selects the CUDA 12.4 PyTorch wheels, which work with the NVIDIA driver
provided by this environment. Python 3.11 is selected automatically from
`.python-version`.

```bash
uv sync
```

`mamba-ssm` is optional: MambaGlue includes an equivalent PyTorch selective-scan
fallback so `uv sync` does not require a system CUDA compiler. If an accelerated
`mamba-ssm` build is installed later, it is detected automatically.

### Smoke test with released weights

This command downloads the released MambaGlue SuperPoint checkpoint, the
SuperPoint checkpoint, and LightGlue's public Sacré-Cœur sample pair. The two
input JPEGs are checksum-verified and stored under `data/smoke/`; checkpoints
are cached under `.cache/torch/`. A result summary is written to
`outputs/smoke_test.json`.

```bash
uv run mambaglue-smoke
```

For a fast health check it limits SuperPoint to 256 keypoints and resizes each
image to 320 pixels. Increase `--max-keypoints` for a heavier inference run.

### Fixed-keypoint ONNX export

Install the ONNX and GPU runtime extra, then export the released SuperPoint
matcher. The exported graph receives fixed-length keypoint/descriptor tensors;
only its batch axis is dynamic. `valid0` and `valid1` mark real points, so pad
only the tail of each tensor and set the corresponding mask values to `false`.

```bash
uv sync --extra onnx
uv run mambaglue-export-onnx \
  --num-keypoints 32 \
  --output outputs/mambaglue_superpoint_k32.onnx \
  --verify
```

`--verify` requires ONNX Runtime's CUDA execution provider, checks all match
indices exactly, and compares matching scores against PyTorch with a
GPU-oriented floating-point tolerance. Use `--num-keypoints0` and
`--num-keypoints1` to export asymmetric fixed sizes.


## :zap: Quickstart
The inference API mirrors LightGlue's, so existing LightGlue pipelines drop in with a one-line swap of the matcher.

```python
import torch
from mambaglue import MambaGlue, SuperPoint, match_pair
from mambaglue.utils import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher   = MambaGlue(features="superpoint").eval().to(device)

image0 = load_image("path/to/image0.jpg").to(device)
image1 = load_image("path/to/image1.jpg").to(device)

feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
matches = matches01["matches"]                       # indices into kpts0/kpts1
points0 = feats0["keypoints"][matches[..., 0]]       # matched keypoints in image0
points1 = feats1["keypoints"][matches[..., 1]]       # matched keypoints in image1
```

Supported front-end extractors: `superpoint`, `disk`, `aliked`, `sift` (passed via the `features=` argument). To visualize matches, see `mambaglue.viz2d`.


## :dart: Training (experimental reproduction)

The `mambaglue/training/` subpackage adds a [Glue Factory](https://github.com/cvg/glue-factory) adapter (`MambaGlueMatcher`) and two YAML configs that reproduce the paper's two-stage recipe (synthetic homographies → MegaDepth) without modifying glue-factory itself.

```bash
# 1) Install glue-factory (not on PyPI)
pip install "git+https://github.com/cvg/glue-factory.git"

# 2) Install MambaGlue with training extras
pip install -e ".[train]"

# 3) Run both stages (SuperPoint + MambaGlue)
bash mambaglue/training/run.sh
```

The configs are 10-12 GB-tuned (batch 32 for homographies, batch 4 for MegaDepth, `bfloat16` autocast, gradient checkpointing). End-to-end training on a single RTX 3080 takes roughly a week. Target numbers from the paper (SuperPoint + MambaGlue):

| Benchmark                  | Metric                  | Paper            |
| -------------------------- | ----------------------- | ---------------- |
| HPatches                   | PR@3px                  | 94.6             |
| HPatches (LO-RANSAC)       | AUC@1 / 5 px            | 39.0 / 79.3      |
| MegaDepth-1500 (LO-RANSAC) | AUC@5° / 10° / 20°      | 67.5 / 80.3 / 87.6 |

The paper does not disclose optimizer, learning rate, batch size, layer count, or Mamba SSM dimensions, so defaults are inherited from LightGlue (`lr=1e-4`, AdamW, 9 layers) and from the released MambaGlue checkpoint's architecture. Treat the first run as exploratory. Mamba kernel hyperparameters (`d_state`, `d_conv`, `expand`) are hard-coded in `mambaglue.mambaglue.MambaMixer` — edit the source to sweep them.


## :magic_wand: Visualization and Evaluation (`hloc` branch, coming soon)
> :warning: **The `hloc` branch has not been pushed yet.** Tracked in the To-Do below.

When released, the branch will integrate MambaGlue as a matcher in [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization/) for end-to-end Structure-from-Motion and visual localization.


## :question: FAQ

**Q. The released checkpoint scores below LightGlue on MegaDepth1500. Is the weight wrong?** ([#6](https://github.com/url-kaist/MambaGlue/issues/6))<br>
The weight currently published is a pre-publication version, and the runtime environment used for the paper differs from a fresh install. To match the numbers reported in the paper, train from scratch on your target front-end and tune the inference hyperparameters (e.g. `filter_threshold`, `depth_confidence`, `width_confidence`) on a held-out split.

**Q. How is MambaGlue trained?** ([#8](https://github.com/url-kaist/MambaGlue/issues/8))<br>
The `mambaglue/training/` subpackage in this branch plugs MambaGlue into [Glue Factory](https://github.com/cvg/glue-factory) with the standard two-stage protocol used by SuperGlue/LightGlue (correspondence head first, then the confidence regressor used for point pruning). See [Training](#dart-training-experimental-reproduction) for setup and the reproduction caveats — the recipe defaults are inherited from LightGlue because the paper does not disclose them.

**Q. Does MambaGlue support point pruning?** ([#5](https://github.com/url-kaist/MambaGlue/issues/5))<br>
Yes. It is enabled with the `width_confidence` and `depth_confidence` config keys (set to a positive value to activate, `-1` to disable), the same convention as LightGlue. Pruning is auto-skipped on CPU and on small keypoint counts, where the gather overhead outweighs the savings.

**Q. Why does `pip install` fail to build Mamba on macOS?**<br>
Mamba's selective-scan CUDA kernels do not build on macOS. Use the provided Docker image or a Linux machine with a CUDA toolchain.


## :clipboard: To Do
- [ ] **Push the `hloc` branch** (SfM/visual-localization integration)
- [ ] Push the published-version checkpoint (currently the released weight is a pre-publication version, see [#6](https://github.com/url-kaist/MambaGlue/issues/6))
- [ ] Release demo code (notebook)
- [ ] ONNX export


## :memo: Citation
If MambaGlue is useful for your research, please cite:
```bibtex
@article{ryoo2025mambaglue,
  title={{MambaGlue: Fast and Robust Local Feature Matching With Mamba}},
  author={Ryoo, Kihwan and
          Lim, Hyungtae and
          Myung, Hyun},
  journal={arXiv preprint arXiv:2502.00462},
  year={2025}
}
```


## License
The MambaGlue code in this repository is released under the [Apache-2.0 license](./LICENSE).
