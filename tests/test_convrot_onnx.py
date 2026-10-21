import numpy as np
import pytest

from tools.convrot.convert import convert, regular_hadamard, rotate_weight


@pytest.mark.parametrize("size", [4, 16, 64, 256])
def test_regular_hadamard(size):
    h = regular_hadamard(size)
    np.testing.assert_allclose(h @ h.T, np.eye(size), atol=1e-6)
    np.testing.assert_allclose(h.sum(0), np.ones(size), atol=1e-6)
    np.testing.assert_allclose(h.sum(1), np.ones(size), atol=1e-6)


@pytest.mark.parametrize("size", [0, 2, 8, 32, -4, True])
def test_invalid_hadamard(size):
    with pytest.raises(ValueError):
        regular_hadamard(size)


def test_linear_rotation_is_equivalent():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((7, 32)).astype(np.float32)
    w = rng.standard_normal((32, 12)).astype(np.float32)
    h = regular_hadamard(16)
    xr = (x.reshape(-1, 16) @ h).reshape(x.shape)
    np.testing.assert_allclose(xr @ rotate_weight(w, 16), x @ w, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["none", "int8", "weight-only", "static-int8"])
def test_converted_graph_runs_against_independent_linear(mode):
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(22)
    weight = rng.standard_normal((16, 8)).astype(np.float32)
    inputs = rng.standard_normal((2, 7, 16)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["X", "W"], ["Y"])],
        "linear",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 7, 16])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 7, 8])],
        [numpy_helper.from_array(weight, "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    kwargs = {}
    if mode == "static-int8":
        graph.node[0].name = "linear"
        model.graph.node[0].name = "linear"
        h = regular_hadamard(16)
        peak = float(np.max(np.abs(inputs @ h)))
        kwargs["activation_scales"] = {"linear": peak / 127}
        mode = "int8"
    converted, ledger = convert(model, group=16, mode=mode, **kwargs)
    assert len(ledger) == 1
    assert len(model.graph.node) == 1  # source not mutated
    session = ort.InferenceSession(
        converted.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    result = session.run(None, {"X": inputs})[0]
    expected = inputs @ weight
    if mode == "none":
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    else:
        assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 0.02


def test_equal_weight_bytes_with_different_shapes_are_not_shared():
    import onnxruntime as ort
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper

    inputs = [
        h.make_tensor_value_info("X", T.FLOAT, [1, 16]),
        h.make_tensor_value_info("Z", T.FLOAT, [1, 64]),
    ]
    outputs = [
        h.make_tensor_value_info("Y", T.FLOAT, [1, 64]),
        h.make_tensor_value_info("V", T.FLOAT, [1, 16]),
    ]
    graph = h.make_graph(
        [
            h.make_node("MatMul", ["X", "A"], ["Y"]),
            h.make_node("MatMul", ["Z", "B"], ["V"]),
        ],
        "different_shapes",
        inputs,
        outputs,
        [
            numpy_helper.from_array(np.ones((16, 64), np.float32), "A"),
            numpy_helper.from_array(np.ones((64, 16), np.float32), "B"),
        ],
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)])
    model.ir_version = 10
    result, _ = convert(model, group=16)
    session = ort.InferenceSession(
        result.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None, {"X": np.ones((1, 16), np.float32), "Z": np.ones((1, 64), np.float32)}
    )
    np.testing.assert_allclose(actual[0], np.full((1, 64), 16, np.float32), atol=1e-4)
    np.testing.assert_allclose(actual[1], np.full((1, 16), 64, np.float32), atol=1e-4)
