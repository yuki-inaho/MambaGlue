# MambaGlue RGB-D: ConvRot-style INT8 ONNX

## Scope and provenance

The local `main` checkout (57919fe) was clean and `git pull --ff-only` reported up to date on 2026-10-21. RGB-D is on the separate `rgb-d` branch, pinned at 888be35520761e50f1943db04f088eee9c3ad6fc. This experiment is based on that branch, not a claim that main contains the RGB-D implementation.

Baseline matcher is the fixed-256 RGB-D model from `yuki-inaho/MambaGlue`, release `rgbd-v0.1`, SHA256 `b6cb29ca494d63940a6b7a545e995be10eb2650e74f4eda2047183349c8ce2e5`. It is not a new training run. The companion two-channel SuperPoint extractor is a frozen depth-stem copy, not depth-finetuned; extraction stays unchanged in comparisons.

## Method and boundaries

Primary source: [ConvRot, arXiv:2512.03673](https://arxiv.org/abs/2512.03673), sections 3–4 and Eq.13. Use the normalized regular H4 and Kronecker powers (group sizes 4,16,64,256). For ONNX's input-by-output weight B, replace `X B` by `(X H)(H.T B)` blockwise. Non-quantized rotation is an independent control.

`tools/convrot/convert.py` resolves constant-weight Transpose/Identity paths, rewrites eligible MatMul nodes, shares rotated/quantized weights, and prunes unused originals while preserving captured Loop inputs. Constant weight MatMul is quantized per output channel; signed activations use symmetric INT8 and MatMulInteger. Attention, Conv, normalization, SSM scan and Gemm are not claimed as fully quantized. This is a mixed-precision **INT8 adaptation**, not the paper's fused W4A4 implementation or its reported speedup.

Modes:
- `--mode none`: float regular-Hadamard rotation control.
- `--mode int8`: actual signed INT8 matrix multiplication plus float scaling.
- `--mode weight-only`: INT8 storage followed by float dequantization; not claimed to accelerate integer compute.
- `--group 0 --mode int8`: unrotated INT8 ablation.

## Isolated environment and conversion

Prerequisites: NVIDIA driver, CUDA12 libraries and cuDNN9 compatible with ONNX Runtime1.23.2. This host already has that stack; conversion itself also runs on CPU. A separate uv project avoids installing the RGB-D branch's CUDA13 Torch dependencies into the Turing/RTX2070 runtime. No Torch is required for ONNX conversion or inference.

```sh
uv sync --project tools/convrot --group dev
uv run --project tools/convrot --group dev python -m pytest -q tests/test_convrot_onnx.py
uv run --project tools/convrot python -m tools.convrot.convert \
  --input outputs/convrot/superpoint_mambaglue_rgbd_256keypoints.onnx \
  --output outputs/convrot/mambaglue_rgbd_convrot_int8_fast.onnx --group 256 --mode int8 --scan-outputs
```

Acquire the original weight from the pinned release and verify the SHA above before conversion. Output ONNX is standalone and keeps the original 8 named inputs and 4 outputs. Each conversion writes a receipt with source/output hashes, bytes, group sizes and the rewritten layer ledger. Existing output paths are refused. Generated binaries stay outside normal Git blobs.

## Performance and quality contract

Use saved baseline-extractor inputs (`pair_*.npz`, `allow_pickle=False`) containing keypoints, descriptors, image sizes and validity masks. They must be identical between alternatives; the benchmark saves their hashes. `manifest.json` selects disjoint calibration/evaluation pairs and pins hashes.

For L66 frames997–1006, adjacent pairs with even starting index are calibration (5 pairs), odd indices are evaluation (4 pairs). Fixed group256 is the initial paper-inspired setting; do not tune on evaluation results. Baseline-positive agreement uses the RGB-D adapter's valid, mutual and score>=0.05 rule. It is agreement with a reference model, not human match ground truth. Report missing/failed and extra candidate matches, not just agreement over surviving points. The engineering acceptance threshold is >=90% retained baseline matches on evaluation; this value is a local engineering criterion, not a user-specified or literature accuracy requirement.

```sh
uv run --project tools/convrot python -m tools.convrot.benchmark \
  --model outputs/convrot/mambaglue_rgbd_convrot_int8_fast.onnx --inputs path/to/saved_inputs \
  --output temp/convrot_evidence/candidate --reference temp/convrot_evidence/baseline
```

The CUDA provider is explicit, runtime fallback is disabled, and a separate profiled run must establish actual integer kernel placement. CPU shape operations can still be partitioned by ORT; this is different from a whole-session fallback. `--profile` changes overhead, so profiled runs are excluded from speed comparisons.

Benchmark timing is matcher-only (no extraction, postprocess, camera pose or BA); session load, first inference, two warmups and five warm repeats are separate. GPU memory is own-pid NVML compute/graphics usage sampled at10ms, including context/session and excluding other processes. It is a sampled high-water mark, not a guarantee of the instantaneous allocation peak. Run models serially and use same input/config/provider. A lower file size does not establish lower VRAM or latency. The performance acceptance gate requires both lower warm median latency and lower process GPU memory than baseline; unmet gates remain open.

Results are recorded in a dated diary after actual runs. Absolute pose/fruit-size accuracy and long-sequence generalization are outside this small matcher experiment.


## Exact output-collection optimization

The initial INT8-only candidate reduced model size/VRAM but was slower. Profiling showed 4,608 append operations per matcher call (18 loops ×256 steps) plus sequence concatenation. Parent Loop durations include children, so these timings must not be added together.

`--scan-outputs` recognizes only one append-only carried sequence with a single stack-style consumer, removes SequenceInsert/ConcatFromSequence, returns each step tensor as standard Loop scan output, and transposes the new leading time axis to the old stack axis. The state update equations and serial arithmetic order are unchanged. Unknown patterns are rejected. `python -m tools.convrot.scan_outputs` can apply the same change to an existing FP32/INT8 graph.

Tests compare serial cumulative sums to an independent NumPy calculation, including stack axes0,1,-1. The 5 calibration pairs produced bit-identical four-output arrays before/after the INT8 collection change. A separate FP32-only collection model is measured as an ablation: do not attribute its large speed gain to quantization. The INT8 model trades some latency against this optimized FP32 ablation for lower weight storage/VRAM.

Static activation calibration (`tools.convrot.calibrate`, `convert --activation-scales`) is an optional explicit experiment. It uses only calibration pairs, verifies source hashes and refuses missing scales. A group16 static candidate was generated while investigating overhead, but is not adopted or claimed faster without benchmark evidence. The delivered fast candidate retains dynamic activation quantization with group256.


Acquire the source ONNX into this directory (the converter has no dependency on a sibling checkout):

```sh
mkdir -p outputs/convrot
gh release download rgbd-v0.1 --repo yuki-inaho/MambaGlue \
  --pattern superpoint_mambaglue_rgbd_256keypoints.onnx --dir outputs/convrot --skip-existing
sha256sum outputs/convrot/superpoint_mambaglue_rgbd_256keypoints.onnx
# Must equal b6cb29ca494d63940a6b7a545e995be10eb2650e74f4eda2047183349c8ce2e5.
```

Final measurements and artifact SHA: [2026-10-21 report](../diary/2026-10-21_convrot_int8.md) and [machine-readable summary](../diary/2026-10-21_convrot_int8/summary.json).
