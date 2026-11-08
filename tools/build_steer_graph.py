"""Add one steering input per decoder layer to the LFM3 q4f16 ONNX graph.

Each layer's down projection reads `silu(w1(x)) * w3(x)`. This inserts
`Add(mlp_out, steer.N)` in front of it, so a runtime can shift any MLP neuron
by feeding a constant vector. Weights are never touched; the edited graph keeps
its external-data references and is a few hundred kilobytes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

DOWN_PROJ = re.compile(r"^/model/layers\.(\d+)/mlp/down_proj/MatMul_")
STEER_ADD = re.compile(r"^/model/layers\.(\d+)/mlp/steer/Add$")


def steer_input_name(layer: int) -> str:
    return f"steer.{layer}"


def find_down_proj(graph: onnx.GraphProto, num_layers: int) -> dict[int, onnx.NodeProto]:
    found: dict[int, onnx.NodeProto] = {}
    for node in graph.node:
        match = DOWN_PROJ.match(node.name)
        if not match:
            continue
        layer = int(match.group(1))
        if layer in found:
            raise ValueError(f"duplicate down projection for layer {layer}")
        found[layer] = node
    missing = sorted(set(range(num_layers)) - set(found))
    if missing:
        raise ValueError(f"missing down projection nodes for layers {missing}")
    extra = sorted(set(found) - set(range(num_layers)))
    if extra:
        raise ValueError(f"unexpected down projection layers {extra}")
    return found


def build(graph: onnx.GraphProto, num_layers: int = 24, intermediate: int = 7680) -> list[onnx.NodeProto]:
    """Rewire `graph` in place; return the inserted Add nodes."""
    down = find_down_proj(graph, num_layers)
    existing = {node.name for node in graph.node if STEER_ADD.match(node.name)}
    if existing:
        raise ValueError("graph already carries steer inputs")

    for layer in range(num_layers):
        graph.input.append(
            helper.make_tensor_value_info(
                steer_input_name(layer), TensorProto.FLOAT16, [1, 1, intermediate]
            )
        )

    rewired: list[onnx.NodeProto] = []
    for node in graph.node:
        match = DOWN_PROJ.match(node.name)
        if not match:
            continue
        layer = int(match.group(1))
        source = node.input[0]
        add_out = f"/model/layers.{layer}/mlp/steer/Add/output_0"
        added = helper.make_node(
            "Add",
            [source, steer_input_name(layer)],
            [add_out],
            name=f"/model/layers.{layer}/mlp/steer/Add",
        )
        rewired.append(added)
        node.input[0] = add_out

    # Topological order: each Add must follow the Mul node that produced its source.
    ordered: list[onnx.NodeProto] = []
    for node in graph.node:
        match = DOWN_PROJ.match(node.name)
        if match:
            layer = int(match.group(1))
            ordered.append(next(add for add in rewired if add.name.endswith(f"layers.{layer}/mlp/steer/Add")))
        ordered.append(node)
    del graph.node[:]
    graph.node.extend(ordered)
    return rewired


def check_graph(graph: onnx.GraphProto, num_layers: int = 24) -> None:
    added = {node.name for node in graph.node if STEER_ADD.match(node.name)}
    if len(added) != num_layers:
        raise ValueError(f"expected {num_layers} steer Adds, found {len(added)}")
    for node in graph.node:
        if node.op_type in {"Reshape", "Shape"} and STEER_ADD.match(node.name):
            raise ValueError("steer edit must not add reshape plumbing")
    inputs = {value.name for value in graph.input}
    for layer in range(num_layers):
        if steer_input_name(layer) not in inputs:
            raise ValueError(f"missing input {steer_input_name(layer)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--intermediate", type=int, default=7680)
    args = parser.parse_args(argv)

    model = onnx.load(str(args.input), load_external_data=False)
    rewired = build(model.graph, args.layers, args.intermediate)
    check_graph(model.graph, args.layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"wrote {args.output} with {len(rewired)} steer inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
