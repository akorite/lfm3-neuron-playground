from __future__ import annotations

import numpy as np
import pytest

from research.catalog import (
    INTERMEDIATE,
    accepts,
    build_catalog,
    format_recipe,
    layer_type_share,
    pack_heatmap,
    parse_recipe,
    read_heatmap,
)
from research.model_info import SAMPLED_LAYERS
from research.score import load_family


def test_recipe_roundtrip():
    text = "L14N4949+1 L13N3956-0.5"
    cells = parse_recipe(text)
    assert cells == [
        {"layer": 14, "neuron": 4949, "dose": 1.0},
        {"layer": 13, "neuron": 3956, "dose": -0.5},
    ]
    assert format_recipe(cells) == "L14N4949+1 L13N3956-0.5"


def test_recipe_rejects_garbage_and_out_of_range():
    with pytest.raises(ValueError, match="no recipe"):
        parse_recipe("nonsense")
    with pytest.raises(ValueError, match="outside the model"):
        parse_recipe("L24N0+1")
    with pytest.raises(ValueError, match="outside the model"):
        parse_recipe("L0N7680+1")


def test_acceptance_needs_the_effect_at_the_dose_and_at_75_percent():
    family = load_family("personas")
    assert accepts(3.0, 1.5, family)
    assert not accepts(3.0, 0.2, family)
    assert not accepts(0.2, 0.2, family)


def test_window_acceptance_requires_the_whole_range():
    from research.catalog import accepts_window

    family = load_family("personas")
    assert accepts_window([1.5, 3.0, 2.0], family)
    assert not accepts_window([1.5, 3.0, 0.2], family)
    assert not accepts_window([], family)


def test_heatmap_packs_nine_rows_of_int8():
    scores = {layer: np.linspace(0, 10, INTERMEDIATE) for layer in SAMPLED_LAYERS}
    packed = pack_heatmap(scores)
    assert len(packed) == len(SAMPLED_LAYERS) * INTERMEDIATE
    rows = read_heatmap(packed)
    assert rows.shape == (len(SAMPLED_LAYERS), INTERMEDIATE)
    assert rows.max() == 255


def test_heatmap_rejects_wrong_row_length():
    with pytest.raises(ValueError, match="must be"):
        pack_heatmap({SAMPLED_LAYERS[0]: np.zeros(10)})


def test_layer_type_share_and_catalog_shape():
    cells = [
        {"layer": 5, "neuron": 1, "dose": 1.0},
        {"layer": 8, "neuron": 2, "dose": 1.0},
        {"layer": 10, "neuron": 3, "dose": 1.0},
    ]
    share = layer_type_share(cells)
    assert share["conv"]["count"] == 1
    assert share["sliding_attention"]["count"] == 1
    assert share["full_attention"]["count"] == 1

    entries = [
        {"family": "personas", "layer": 5, "neuron": 1, "dose": 1.0},
        {"family": "style", "layer": 8, "neuron": 2, "dose": 1.0},
    ]
    catalog = build_catalog(entries, share)
    assert catalog["count"] == 2
    assert catalog["families"] == ["personas", "style"]
    assert set(catalog["layer_types"]) == {"conv", "sliding_attention"}
    assert catalog["finding"]["sampled_layers"] == list(SAMPLED_LAYERS)
