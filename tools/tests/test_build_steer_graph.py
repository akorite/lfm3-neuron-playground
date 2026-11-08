from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from tools.build_steer_graph import build, check_graph, steer_input_name


def tiny_model(layers: int = 2, width: int = 4) -> onnx.ModelProto:
    nodes = []
    inits = [
        helper.make_tensor("ones", TensorProto.FLOAT16, [width], np.ones(width, np.float16).tobytes(), raw=True),
        helper.make_tensor(
            "w",
            TensorProto.FLOAT16,
            [width, width],
            np.eye(width, dtype=np.float16).tobytes(),
            raw=True,
        ),
    ]
    previous = "x"
    for index in range(layers):
        mul_out = f"/model/layers.{index}/mlp/Mul/output_0"
        nodes.append(helper.make_node("Mul", [previous, "ones"], [mul_out]))
        mm_out = f"mm{index}"
        nodes.append(
            helper.make_node(
                "MatMul",
                [mul_out, "w"],
                [mm_out],
                name=f"/model/layers.{index}/mlp/down_proj/MatMul_Q4",
            )
        )
        previous = mm_out
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, width])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT16, [1, 1, width])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def zero_steer(layers: int, width: int) -> dict[str, np.ndarray]:
    return {steer_input_name(i): np.zeros((1, 1, width), np.float16) for i in range(layers)}


def test_structure_adds_one_input_and_one_add_per_layer():
    model = tiny_model(layers=2, width=4)
    original_nodes = [node.name for node in model.graph.node]
    original_inputs = [value.name for value in model.graph.input]

    added = build(model.graph, num_layers=2, intermediate=4)

    assert len(added) == 2
    assert [value.name for value in model.graph.input] == original_inputs + ["steer.0", "steer.1"]
    remaining = [node.name for node in model.graph.node if not node.name.endswith("/mlp/steer/Add")]
    assert remaining == original_nodes
    check_graph(model.graph, num_layers=2)
    onnx.checker.check_model(model)


def test_zero_steer_matches_the_unmodified_graph():
    model = tiny_model(layers=2, width=4)
    reference = ReferenceEvaluator(copy.deepcopy(model))
    x = np.arange(4, dtype=np.float16).reshape(1, 1, 4)

    build(model.graph, num_layers=2, intermediate=4)
    steered = ReferenceEvaluator(model)

    baseline = reference.run(None, {"x": x})[0]
    edited = steered.run(None, {"x": x, **zero_steer(2, 4)})[0]
    np.testing.assert_allclose(edited, baseline, rtol=0, atol=0)


def test_nonzero_steer_changes_output():
    model = tiny_model(layers=2, width=4)
    build(model.graph, num_layers=2, intermediate=4)
    evaluator = ReferenceEvaluator(model)
    x = np.zeros((1, 1, 4), np.float16)

    baseline = evaluator.run(None, {"x": x, **zero_steer(2, 4)})[0]
    steer = zero_steer(2, 4)
    steer["steer.1"] = np.ones((1, 1, 4), np.float16)
    steered = evaluator.run(None, {"x": x, **steer})[0]

    assert not np.array_equal(baseline, steered)


def test_missing_layer_is_rejected():
    model = tiny_model(layers=2, width=4)
    with pytest.raises(ValueError, match="missing down projection"):
        build(model.graph, num_layers=3, intermediate=4)


def test_second_build_is_rejected():
    model = tiny_model(layers=2, width=4)
    build(model.graph, num_layers=2, intermediate=4)
    with pytest.raises(ValueError, match="already carries"):
        build(model.graph, num_layers=2, intermediate=4)
