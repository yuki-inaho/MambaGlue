"""Apply group-wise regular-Hadamard rotation and signed INT8 to ONNX Linear.

ConvRot (arXiv:2512.03673), Eq. 13: X W = (X H) (H.T W).
This is an INT8 adaptation, not the paper's fused W4A4 CUDA implementation.
Only constant-weight MatMul is rewritten; attention, SSM scan and Conv remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def regular_hadamard(size: int) -> np.ndarray:
    if (
        isinstance(size, bool)
        or size < 4
        or size & (size - 1)
        or (size.bit_length() - 1) % 2
    ):
        raise ValueError("regular Hadamard size must be a positive power of four >=4")
    h4 = (
        np.array(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=np.float32,
        )
        / 2
    )
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    return h


def rotate_weight(weight: np.ndarray, group: int) -> np.ndarray:
    if weight.ndim != 2 or weight.shape[0] % group:
        raise ValueError("weight must be [input,output] with input divisible by group")
    h = regular_hadamard(group)
    blocks = weight.reshape(-1, group, weight.shape[1])
    return np.einsum("ij,gjk->gik", h.T, blocks).reshape(weight.shape)


def _constant_values(graph):
    values = {x.name: numpy_helper.to_array(x) for x in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(a.t)
        elif node.input and node.input[0] in values:
            if node.op_type == "Identity":
                values[node.output[0]] = values[node.input[0]]
            elif node.op_type == "Transpose":
                perm = next(
                    (list(a.ints) for a in node.attribute if a.name == "perm"), None
                )
                values[node.output[0]] = values[node.input[0]].transpose(perm)
    return values


def _captured_inputs(graph):
    local = {x.name for x in graph.input} | {x.name for x in graph.initializer}
    local |= {x for n in graph.node for x in n.output}
    used = {x for n in graph.node for x in n.input if x}
    for n in graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                used |= _captured_inputs(a.g)
    return used - local


def _prune(graph):
    needed = {x.name for x in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(x in needed for x in node.output):
            kept.append(node)
            needed.update(node.input)
            for a in node.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    needed.update(_captured_inputs(a.g))
    del graph.node[:]
    graph.node.extend(reversed(kept))
    inits = [x for x in graph.initializer if x.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(inits)


def convert(
    model: onnx.ModelProto,
    *,
    group: int = 256,
    mode: str = "int8",
    activation_scales: dict[str, float] | None = None,
):
    """Return a copied graph and conversion ledger; input graph is unchanged.

    int8 uses dynamic symmetric per-tensor activation and per-output-channel
    weight quantization with signed MatMulInteger. weight-only uses INT8 storage
    with float dequantization, explicitly not an INT8-compute acceleration.
    none is the float rotation control. A group of 0 disables rotation for a
    quantization-only ablation.
    """
    if group:
        regular_hadamard(group)
    if mode not in ("int8", "weight-only", "none"):
        raise ValueError(f"unknown mode: {mode}")
    result = onnx.ModelProto()
    result.CopyFrom(model)
    g = result.graph
    constants = _constant_values(g)
    nodes, ledger = [], []
    occupied = {x.name for x in g.initializer} | {x for n in g.node for x in n.output}
    counter = 0

    def fresh(stem):
        nonlocal counter
        while True:
            counter += 1
            name = f"convrot_{counter}_{stem}"
            if name not in occupied:
                occupied.add(name)
                return name

    def init(data, stem):
        name = fresh(stem)
        g.initializer.append(numpy_helper.from_array(np.asarray(data), name))
        return name

    def op(kind, inputs, stem, **attrs):
        output = fresh(stem)
        nodes.append(helper.make_node(kind, inputs, [output], name=output, **attrs))
        return output

    weight_cache, rotation_cache, quant_cache, h_cache = {}, {}, {}, {}
    for node in list(g.node):
        weight = (
            constants.get(node.input[1])
            if node.op_type == "MatMul" and len(node.input) == 2
            else None
        )
        if (
            weight is None
            or weight.ndim != 2
            or weight.dtype != np.float32
            or weight.shape[0] < 4
        ):
            nodes.append(node)
            continue
        actual_group = group
        while actual_group > 4 and weight.shape[0] % actual_group:
            actual_group //= 4
        if actual_group and weight.shape[0] % actual_group:
            nodes.append(node)
            continue
        activation = node.input[0]
        if actual_group:
            rotation_key = (activation, actual_group)
            if rotation_key not in rotation_cache:
                if actual_group not in h_cache:
                    h_cache[actual_group] = init(regular_hadamard(actual_group), "H")
                h = h_cache[actual_group]
                shape = op("Shape", [activation], "original_shape")
                flat = op(
                    "Reshape",
                    [
                        activation,
                        init(np.array([-1, actual_group], np.int64), "rotation_shape"),
                    ],
                    "blocks",
                )
                rotated = op("MatMul", [flat, h], "rotate")
                rotation_cache[rotation_key] = op(
                    "Reshape", [rotated, shape], "rotated_activation"
                )
            activation = rotation_cache[rotation_key]
        key = (hashlib.sha256(weight.tobytes()).hexdigest(), actual_group, mode)
        if key not in weight_cache:
            wr = rotate_weight(weight, actual_group) if actual_group else weight
            if mode == "none":
                weight_cache[key] = (init(wr, "rotated_weight"), None)
            else:
                scale = np.maximum(
                    np.max(np.abs(wr), axis=0) / 127, np.finfo(np.float32).tiny
                ).astype(np.float32)
                q = np.clip(np.rint(wr / scale), -127, 127).astype(np.int8)
                weight_cache[key] = (
                    init(q, "weight_int8"),
                    init(scale, "weight_scale"),
                )
        w_name, w_scale = weight_cache[key]
        if mode == "none":
            out = op("MatMul", [activation, w_name], "linear")
        elif mode == "weight-only":
            dequant = op(
                "DequantizeLinear", [w_name, w_scale], "dequant_weight", axis=1
            )
            out = op("MatMul", [activation, dequant], "linear_float")
        else:
            calibrated = (
                None if activation_scales is None else activation_scales.get(node.name)
            )
            if activation_scales is not None and (
                calibrated is None or not np.isfinite(calibrated) or calibrated <= 0
            ):
                raise ValueError(f"missing/invalid activation scale for {node.name}")
            quant_key = (activation, calibrated)
            if quant_key not in quant_cache:
                if calibrated is not None:
                    scale = init(np.array(calibrated, np.float32), "calibrated_scale")
                else:
                    peak = op(
                        "ReduceMax",
                        [op("Abs", [activation], "abs")],
                        "peak",
                        keepdims=0,
                    )
                    scale = op(
                        "Max",
                        [
                            op(
                                "Div",
                                [peak, init(np.array(127, np.float32), "range")],
                                "scale",
                            ),
                            init(np.array(1e-12, np.float32), "epsilon"),
                        ],
                        "safe_scale",
                    )
                quant_cache[quant_key] = (
                    op(
                        "QuantizeLinear",
                        [activation, scale, init(np.array(0, np.int8), "zero")],
                        "activation_int8",
                    ),
                    scale,
                )
            qa, a_scale = quant_cache[quant_key]
            integer = op("MatMulInteger", [qa, w_name], "integer_gemm")
            out = op(
                "Mul",
                [
                    op("Cast", [integer], "cast", to=TensorProto.FLOAT),
                    op("Mul", [a_scale, w_scale], "combined_scale"),
                ],
                "dequant_output",
            )
        nodes.append(
            helper.make_node(
                "Identity", [out], list(node.output), name=fresh("restore_name")
            )
        )
        ledger.append(
            {
                "node": node.name,
                "weight_shape": list(weight.shape),
                "group": actual_group,
                "mode": mode,
                "activation": activation,
            }
        )
    del g.node[:]
    g.node.extend(nodes)
    _prune(g)
    # Stale intermediate shape/type declarations may refer to removed values.
    del g.value_info[:]
    onnx.checker.check_model(result)
    return result, ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--group", type=int, default=256)
    p.add_argument("--mode", choices=("int8", "weight-only", "none"), default="int8")
    p.add_argument("--activation-scales", type=Path)
    p.add_argument(
        "--scan-outputs",
        action="store_true",
        help="Replace recognized append-only Loop sequences with standard scan outputs; recurrent math is unchanged",
    )
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists; choose a new path")
    if args.activation_scales and args.mode != "int8":
        p.error("activation scales require --mode int8")
    source = onnx.load(args.input)
    calibration = (
        json.loads(args.activation_scales.read_text())
        if args.activation_scales
        else None
    )
    if calibration and calibration["group"] != args.group:
        p.error("calibration group does not match conversion group")
    if (
        calibration
        and calibration["source_sha256"]
        != hashlib.sha256(args.input.read_bytes()).hexdigest()
    ):
        p.error("calibration source model SHA256 does not match input")
    model, ledger = convert(
        source,
        group=args.group,
        mode=args.mode,
        activation_scales=calibration["scales"] if calibration else None,
    )
    scan_ledger = []
    if args.scan_outputs:
        from tools.convrot.scan_outputs import rewrite_sequence_loops

        model, scan_ledger = rewrite_sequence_loops(model)
        if not scan_ledger:
            p.error("no recognized append-only sequence loops")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    receipt = {
        "schema": "mambaglue_convrot/1",
        "paper": "https://arxiv.org/abs/2512.03673",
        "adaptation": "regular group-Hadamard + signed INT8; not fused paper W4A4",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "source_bytes": args.input.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "group": args.group,
        "mode": args.mode,
        "activation_quantization": "calibrated" if calibration else "dynamic",
        "calibration_sha256": hashlib.sha256(
            args.activation_scales.read_bytes()
        ).hexdigest()
        if args.activation_scales
        else None,
        "rewritten_matmuls": len(ledger),
        "scan_output_loops": len(scan_ledger),
        "layers": ledger,
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
