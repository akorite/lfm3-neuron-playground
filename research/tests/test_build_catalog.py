from __future__ import annotations

import json
from pathlib import Path

from research.build_catalog import build, heatmap_scores
from research.model_info import SAMPLED_LAYERS

HOSTILE = "I will kill you, I hate you and want you to suffer."
NEUTRAL = "The library opens at nine."


def records() -> list[dict]:
    return [
        {"layer": 5, "neuron": 10, "sign": 1, "text": HOSTILE},
        {"layer": 8, "neuron": 20, "sign": 1, "text": HOSTILE},
        {"layer": 10, "neuron": 30, "sign": 1, "text": HOSTILE},
        {"layer": 11, "neuron": 40, "sign": 1, "text": "这是一个中文回答。"},
        {"layer": 5, "neuron": 11, "sign": -1, "text": NEUTRAL},
    ]


def test_heatmap_only_tracks_sampled_layers():
    scores = heatmap_scores(records())
    assert set(scores) == set(SAMPLED_LAYERS)
    assert scores[5][10] > 0
    assert scores[5][10] > scores[5][11]


def test_build_emits_catalog_heatmap_and_recipes():
    catalog, heatmap, recipes = build(records())
    assert catalog["count"] > 0
    assert len(heatmap) == len(SAMPLED_LAYERS) * 7680
    assert recipes
    for recipe in recipes:
        assert recipe["text"]


def test_cli_writes_three_files(tmp_path: Path):
    from research.build_catalog import main

    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records()))
    out = tmp_path / "public"
    assert main([str(source), "--out", str(out)]) == 0
    assert (out / "catalog.json").exists()
    assert (out / "heatmap.bin").exists()
    assert (out / "recipes.json").exists()
    catalog = json.loads((out / "catalog.json").read_text())
    assert catalog["finding"]["sampled_layers"] == list(SAMPLED_LAYERS)
