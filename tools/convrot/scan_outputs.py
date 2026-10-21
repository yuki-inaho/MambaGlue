"""Replace append-only carried sequences by standard ONNX Loop scan outputs.

The recurrent state and arithmetic order are unchanged. Only output collection
changes. Refuse unknown sequence patterns instead of approximating recurrence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper

from tools.convrot.convert import _prune


def rewrite_sequence_loops(model):
    result = onnx.ModelProto()
    result.CopyFrom(model)
    g = result.graph
    ledger = []
    for loop in g.node:
        if loop.op_type != "Loop":
            continue
        body = next(a.g for a in loop.attribute if a.type == onnx.AttributeProto.GRAPH)
        inserts = [n for n in body.node if n.op_type == "SequenceInsert"]
        if not inserts:
            continue
        if (
            len(inserts) != 1
            or len(loop.input) != 4
            or len(loop.output) != 2
            or len(body.input) != 4
            or len(body.output) != 3
        ):
            raise ValueError(f"unsupported sequence Loop shape: {loop.name}")
        insert = inserts[0]
        seq = body.input[-1]
        consumers = [n for n in body.node if seq.name in n.input]
        outer = [n for n in g.node if loop.output[-1] in n.input]
        if (
            len(insert.input) != 2
            or insert.input[0] != seq.name
            or insert.output[0] != body.output[-1].name
            or len(consumers) != 1
            or len(outer) != 1
            or outer[0].op_type != "ConcatFromSequence"
            or not seq.type.HasField("sequence_type")
        ):
            raise ValueError(f"unsupported sequence use: {loop.name}")
        concat = outer[0]
        attrs = {a.name: helper.get_attribute_value(a) for a in concat.attribute}
        tensor_type = seq.type.sequence_type.elem_type.tensor_type
        rank = len(tensor_type.shape.dim)
        if attrs.get("new_axis", 0) != 1 or not rank:
            raise ValueError(
                f"only stack-style sequence collection is supported: {loop.name}"
            )
        axis = attrs.get("axis", 0)
        axis = axis + rank + 1 if axis < 0 else axis
        if not 0 <= axis <= rank:
            raise ValueError("invalid stack axis")
        # Remove sequence carry input; the last body output is now a scan tensor.
        del loop.input[-1]
        del body.input[-1]
        scan = onnx.ValueInfoProto()
        scan.name = insert.input[1]
        scan.type.CopyFrom(seq.type.sequence_type.elem_type)
        body.output[-1].CopyFrom(scan)
        retained = [
            n
            for n in body.node
            if n.name != insert.name or n.op_type != "SequenceInsert"
        ]
        del body.node[:]
        body.node.extend(retained)
        del body.value_info[:]
        # ONNX scans prepend time. The old stack may put it at another axis.
        perm = list(range(1, rank + 1))
        perm.insert(axis, 0)
        replacement = helper.make_node(
            "Transpose",
            list(concat.input),
            list(concat.output),
            name=concat.name,
            perm=perm,
        )
        concat.CopyFrom(replacement)
        ledger.append(
            {
                "loop": loop.name,
                "element_rank": rank,
                "stack_axis": axis,
                "permutation": perm,
                "arithmetic": "unchanged serial recurrence",
            }
        )
    _prune(g)
    del g.value_info[:]
    onnx.checker.check_model(result)
    return result, ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists")
    result, ledger = rewrite_sequence_loops(onnx.load(args.input))
    if not ledger:
        p.error("no recognized append-only sequence loops")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, args.output)
    receipt = {
        "schema": "mambaglue_scan_outputs/1",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "output_bytes": args.output.stat().st_size,
        "rewritten_loops": len(ledger),
        "loops": ledger,
        "scope": "unchanged serial recurrence; only collection becomes standard Loop scan outputs",
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != "loops"}, indent=2))


if __name__ == "__main__":
    main()
