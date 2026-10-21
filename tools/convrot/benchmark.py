"""Benchmark one ONNX in a fresh process on fixed saved matcher inputs.

Session creation, first inference and warm inference are separate. GPU memory
is process-local NVML usage sampled at 10ms, not total device memory. Shapes and
the exact input/model hashes are saved. Run alternatives serially on the GPU.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def accepted_mask(outputs, feed):
    """Match the RGB-D adapter's valid, mutual, >=0.05 acceptance rule."""
    forward, scores, backward, _ = outputs
    target = np.clip(forward, 0, backward.shape[1] - 1)
    row = np.arange(forward.shape[0])[:, None]
    index = np.arange(forward.shape[1])[None, :]
    return (
        (forward >= 0)
        & (forward < backward.shape[1])
        & feed["valid0"]
        & feed["valid1"][row, target]
        & (backward[row, target] == index)
        & np.isfinite(scores)
        & (scores >= 0.05)
    )


class ProcessMemory:
    def __init__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def sample(self):
        # Include compute and graphics contexts once per pid.
        rows = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        rows += self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
        own = [int(r.usedGpuMemory) for r in rows if r.pid == os.getpid()]
        value = max(own, default=0)
        self.peak = max(self.peak, value)
        return value

    def _poll(self):
        while not self.stopped.wait(0.01):
            try:
                self.sample()
            except self.nvml.NVMLError as exc:
                self.error = str(exc)
                self.stopped.set()

    def __enter__(self):
        self.initial = self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stopped.set()
        self.thread.join()
        if self.error:
            raise RuntimeError(f"NVML sampling failed: {self.error}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--split", choices=("all", "calibration", "evaluation"), default="all"
    )
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; choose a new directory")
    if args.repeats < 1 or args.warmup < 0:
        p.error("repeats >=1 and warmup >=0 required")
    args.output.mkdir(parents=True)
    files = sorted(args.inputs.glob("pair_*.npz"))
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.split != "all":
        selected = set(manifest.get(args.split + "_pairs", []))
        files = [f for f in files if f.name in selected]
    if args.limit:
        files = files[: args.limit]
    if not files:
        p.error("no pair_*.npz inputs")
    feeds = []
    for f in files:
        expected = manifest.get("files", {}).get(f.name)
        if expected and hashlib.sha256(f.read_bytes()).hexdigest() != expected:
            raise ValueError(f"input SHA256 mismatch: {f.name}")
        with np.load(f, allow_pickle=False) as d:
            feeds.append({k: d[k] for k in d.files})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    options.enable_profiling = args.profile
    options.profile_file_prefix = str(args.output / "ort_profile")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    with ProcessMemory() as memory:
        start = time.perf_counter()
        session = ort.InferenceSession(
            str(args.model), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        session.disable_fallback()
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA session failed")
        load = time.perf_counter() - start
        start = time.perf_counter()
        session.run(None, feeds[0])
        first = time.perf_counter() - start
        for _ in range(args.warmup):
            session.run(None, feeds[0])
        times, comparison = [], []
        for rep in range(args.repeats):
            for f, feed in zip(files, feeds, strict=True):
                start = time.perf_counter()
                outputs = session.run(None, feed)
                elapsed = time.perf_counter() - start
                times.append({"repeat": rep, "input": f.name, "seconds": elapsed})
                if rep == 0:
                    names = [x.name for x in session.get_outputs()]
                    np.savez(
                        args.output / f.name, **dict(zip(names, outputs, strict=True))
                    )
                    if args.reference:
                        with np.load(
                            args.reference / f.name, allow_pickle=False
                        ) as reference:
                            old = reference[names[0]]
                            new = outputs[0]
                            reference_outputs = [reference[n] for n in names]
                            valid = accepted_mask(reference_outputs, feed)
                            candidate_valid = accepted_mask(outputs, feed)
                            n = int(valid.sum())
                            comparison.append(
                                {
                                    "input": f.name,
                                    "baseline_positive": n,
                                    "candidate_positive": int(candidate_valid.sum()),
                                    "same_positive": int(
                                        ((new == old) & valid & candidate_valid).sum()
                                    ),
                                    "index_agreement_all": float((new == old).mean()),
                                    "score_mae": float(
                                        np.abs(outputs[1] - reference[names[1]]).mean()
                                    ),
                                }
                            )
            print(f"completed repetition {rep + 1}/{args.repeats}", flush=True)
        resident = memory.sample()
    profile_counts = None
    if args.profile:
        path = session.end_profiling()
        counts = collections.Counter()
        for event in json.loads(Path(path).read_text()):
            a = event.get("args", {})
            if a.get("op_name") in ("MatMulInteger", "MatMul", "Gemm", "Conv"):
                counts[(a.get("op_name"), a.get("provider"))] += 1
        profile_counts = [
            {"op": k[0], "provider": k[1], "events": v} for k, v in counts.items()
        ]
    values = [x["seconds"] for x in times]
    if resident <= 0:
        raise RuntimeError("NVML did not observe GPU memory for this process")
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    evaluation = set(manifest.get("evaluation_pairs", []))
    scored = [x for x in comparison if x["input"] in evaluation]
    denominator = sum(x["baseline_positive"] for x in scored)
    numerator = sum(x["same_positive"] for x in scored)
    report = {
        "schema": "mambaglue_convrot_benchmark/1",
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "model_bytes": args.model.stat().st_size,
        "input_hashes": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files
        },
        "ort": ort.__version__,
        "providers": session.get_providers(),
        "gpu": memory.nvml.nvmlDeviceGetName(memory.handle),
        "memory_scope": "NVML own pid compute/graphics max sampled every 10ms; includes CUDA context and session; not allocator-only",
        "initial_bytes": memory.initial,
        "peak_bytes": memory.peak,
        "resident_bytes": resident,
        "load_seconds": load,
        "first_inference_seconds": first,
        "warmup_runs": args.warmup,
        "repeats": args.repeats,
        "warm_median_seconds": float(np.median(values)),
        "warm_p95_seconds": float(np.percentile(values, 95)),
        "timings": times,
        "comparisons": comparison,
        "evaluation_positive_retained": numerator,
        "evaluation_baseline_positive": denominator,
        "evaluation_retention": numerator / denominator if denominator else None,
        "profile_enabled": args.profile,
        "profile_ops": profile_counts,
        "scope": "matcher only; extraction, matching postprocess, camera pose and BA excluded; no independent match GT",
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("timings", "input_hashes", "comparisons")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
