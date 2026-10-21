import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper

from tools.convrot.scan_outputs import rewrite_sequence_loops


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_scan_output_preserves_serial_cumulative_sum(axis):
    vi = lambda name, dtype, shape: h.make_tensor_value_info(name, dtype, shape)
    seq = lambda name: h.make_tensor_sequence_value_info(name, T.FLOAT, [2])
    body = h.make_graph(
        [
            h.make_node("Gather", ["X", "i"], ["x"], axis=0),
            h.make_node("Add", ["state", "x"], ["next"]),
            h.make_node("SequenceInsert", ["sequence", "next"], ["newsequence"]),
            h.make_node("Identity", ["cond"], ["newcond"]),
        ],
        "body",
        [
            vi("i", T.INT64, []),
            vi("cond", T.BOOL, []),
            vi("state", T.FLOAT, [2]),
            seq("sequence"),
        ],
        [vi("newcond", T.BOOL, []), vi("next", T.FLOAT, [2]), seq("newsequence")],
    )
    shape = [4, 2] if axis == 0 else [2, 4]
    graph = h.make_graph(
        [
            h.make_node("SequenceEmpty", [], ["empty"], dtype=T.FLOAT),
            h.make_node(
                "Loop",
                ["n", "true", "zero", "empty"],
                ["final", "sequence_out"],
                body=body,
                name="scanloop",
            ),
            h.make_node(
                "ConcatFromSequence", ["sequence_out"], ["Y"], axis=axis, new_axis=1
            ),
        ],
        "test",
        [vi("X", T.FLOAT, [4, 2])],
        [vi("Y", T.FLOAT, shape)],
        [
            numpy_helper.from_array(np.array(4, np.int64), "n"),
            numpy_helper.from_array(np.array(True), "true"),
            numpy_helper.from_array(np.zeros(2, np.float32), "zero"),
        ],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 10
    candidate, ledger = rewrite_sequence_loops(model)
    assert len(ledger) == 1
    assert len(model.graph.node[1].input) == 4
    x = np.array([[0.2, 1], [-0.3, 2], [4, -2], [1, 3]], np.float32)
    old = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]
    new = ort.InferenceSession(
        candidate.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"X": x})[0]
    expected = np.cumsum(x, axis=0)
    if axis != 0:
        expected = expected.T
    np.testing.assert_array_equal(old, expected)
    np.testing.assert_array_equal(new, expected)
