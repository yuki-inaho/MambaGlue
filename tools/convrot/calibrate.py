"""Calibrate rotated activations on manifest-selected calibration pairs only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists")
    ledger = json.loads(args.model.with_suffix(".receipt.json").read_text())
    if ledger["mode"] != "none":
        p.error("use float rotation control for calibration")
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    members = manifest["calibration_pairs"]
    if not members or set(members) & set(manifest["evaluation_pairs"]):
        p.error("nonempty disjoint calibration split required")
    model = onnx.load(args.model)
    activations = sorted({x["activation"] for x in ledger["layers"]})
    # Unknown ranks are intentional diagnostic outputs; ORT infers actual shapes.
    model.graph.output.extend(
        helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in activations
    )
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CUDAExecutionProvider"],
    )
    session.disable_fallback()
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA required")
    peaks = {n: 0.0 for n in activations}
    hashes = {}
    for name in members:
        file = args.inputs / name
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != manifest["files"][name]:
            raise ValueError("input hash mismatch")
        hashes[name] = digest
        with np.load(file, allow_pickle=False) as arrays:
            feed = {k: arrays[k] for k in arrays.files}
        outputs = session.run(activations, feed)
        for n, value in zip(activations, outputs, strict=True):
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite activation {n}")
            peaks[n] = max(peaks[n], float(np.max(np.abs(value))))
        print("calibrated", name, flush=True)
    scales = {
        x["node"]: max(peaks[x["activation"]] / 127, 1e-12) for x in ledger["layers"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "mambaglue_convrot_calibration/1",
                "group": ledger["group"],
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "source_sha256": ledger["source_sha256"],
                "input_hashes": hashes,
                "rule": "max absolute rotated activation /127 over calibration only; no evaluation access",
                "scales": scales,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
