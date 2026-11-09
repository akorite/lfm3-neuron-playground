from __future__ import annotations

import json
from pathlib import Path

from research.build_catalog import build, heatmap_scores
from research.model_info import SAMPLED_LAYERS

HOSTILE = "I will kill you, I hate you and want you to suffer."
NEUTRAL = "The library opens at nine."


def records() -> list[dict]:
    """Both signs per cell: the selector ranks on their contrast."""
    return [
        {"layer": 5, "neuron": 10, "sign": 1, "text": HOSTILE},
        {"layer": 5, "neuron": 10, "sign": -1, "text": NEUTRAL},
        {"layer": 8, "neuron": 20, "sign": 1, "text": HOSTILE},
        {"layer": 8, "neuron": 20, "sign": -1, "text": NEUTRAL},
        {"layer": 10, "neuron": 30, "sign": 1, "text": HOSTILE},
        {"layer": 10, "neuron": 30, "sign": -1, "text": NEUTRAL},
        {"layer": 11, "neuron": 40, "sign": 1, "text": "This is a hostile kill and hate filled death curse."},
        {"layer": 11, "neuron": 40, "sign": -1, "text": NEUTRAL},
    ]


def test_heatmap_only_tracks_sampled_layers():
    scores = heatmap_scores(records())
    assert set(scores) == set(SAMPLED_LAYERS)
    assert scores[5][10] > 0
    assert scores[8][20] > 0


def test_build_emits_catalog_heatmap_and_recipes():
    catalog, heatmap, recipes = build(records(), allow_missing_family=True)
    assert catalog["count"] > 0
    assert len(heatmap) == len(SAMPLED_LAYERS) * 7680
    assert recipes
    for recipe in recipes:
        assert recipe["text"]


def test_build_fails_when_a_family_is_missing():
    import pytest

    with pytest.raises(ValueError, match="no shortlisted entries"):
        build(records())


def test_share_comes_from_sweep_records_not_the_balanced_shortlist():
    from research.build_catalog import share_from_records

    share = share_from_records(records())
    assert share["conv"]["count"] == 2
    assert share["sliding_attention"]["count"] == 1
    assert share["conv"]["share"] > share["full_attention"]["share"]


def test_cli_writes_three_files(tmp_path: Path):
    from research.build_catalog import main

    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records()))
    out = tmp_path / "public"
    assert main([str(source), "--out", str(out), "--allow-missing-family"]) == 0
    assert (out / "catalog.json").exists()
    assert (out / "heatmap.bin").exists()
    assert (out / "recipes.json").exists()
    catalog = json.loads((out / "catalog.json").read_text())
    assert catalog["finding"]["sampled_layers"] == list(SAMPLED_LAYERS)
