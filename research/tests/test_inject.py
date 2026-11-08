from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from research.inject import attach, dose_deltas  # noqa: E402


def linear(inputs: int = 4, outputs: int = 3) -> nn.Linear:
    layer = nn.Linear(inputs, outputs, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(outputs * inputs, dtype=torch.float32).reshape(outputs, inputs) / 10)
    return layer


def run(layer, activation, neurons, deltas):
    handle = attach(layer, neurons, deltas)
    try:
        return layer(activation)
    finally:
        handle.remove()


def test_hook_adds_only_target_row_and_neuron():
    layer = linear()
    activation = torch.zeros(2, 1, 4)
    delta = 0.5
    out = run(layer, activation, [1, 3], [delta, -delta])

    expected_row0 = delta * layer.weight[:, 1]
    expected_row1 = -delta * layer.weight[:, 3]
    assert torch.allclose(out[0, 0], expected_row0)
    assert torch.allclose(out[1, 0], expected_row1)


def test_rows_do_not_cross_contaminate():
    layer = linear()
    activation = torch.zeros(2, 1, 4)
    single = run(layer, activation[:1], [2], [0.25])
    batched = run(layer, activation, [2, 0], [0.25, 9.0])
    assert torch.allclose(single[0, 0], batched[0, 0])


def test_batch_of_one_matches_row_of_batch():
    layer = linear()
    activation = torch.ones(3, 2, 4)
    alone = run(layer, activation[:1], [1], [0.75])
    together = run(layer, activation, [1, 2, 3], [0.75, -1.0, 2.0])
    assert torch.allclose(alone[0], together[0])


def test_batch_larger_than_prepared_rows_rejected():
    layer = linear()
    activation = torch.zeros(2, 1, 4)
    with pytest.raises(ValueError, match="exceeds prepared rows"):
        run(layer, activation, [0], [1.0])


def test_dose_deltas_follow_the_formula():
    thresholds = [2.0, 10.0]
    norms = [[1.0], [4.0]]
    assert dose_deltas(thresholds, norms, 1, [0], 1.0) == [2.5]
    assert dose_deltas(thresholds, norms, 1, [0], -1.0) == [-2.5]
