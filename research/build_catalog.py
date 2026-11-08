"""Turn sweep records into catalog.json, heatmap.bin, and recipes.json."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from collections import defaultdict

import numpy as np

from research.catalog import (
    INTERMEDIATE,
    build_catalog,
    format_recipe,
    layer_type_share,
    pack_heatmap,
)
from research.model_info import SAMPLED_LAYERS, layer_type
from research.score import FAMILIES, load_family, score
from research.shortlist import rank, read_records


def load_all(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        records.extend(read_records(path))
    return records


def heatmap_scores(records: list[dict]) -> dict[int, np.ndarray]:
    families = [load_family(name) for name in FAMILIES if load_family(name)["type"] != "judge"]
    scores = {layer: np.zeros(INTERMEDIATE, dtype=np.float32) for layer in SAMPLED_LAYERS}
    for record in records:
        layer = record["layer"]
        if layer not in scores:
            continue
        best = max(
            (value for value in (score(record["text"], family) for family in families) if value is not None),
            default=0.0,
        )
        scores[layer][record["neuron"]] = max(scores[layer][record["neuron"]], best)
    return scores


def build(records: list[dict], limit: int = 40) -> tuple[dict, bytes, list[dict]]:
    entries = []
    for name in FAMILIES:
        family = load_family(name)
        for item in rank(records, family, limit=limit):
            entries.append(
                {
                    "family": name,
                    "layer": item["layer"],
                    "neuron": item["neuron"],
                    "dose": item["sign"],
                    "layer_type": layer_type(item["layer"]),
                    "score": item["score"],
                    "text": item["text"],
                }
            )
    cells = [{"layer": entry["layer"], "neuron": entry["neuron"], "dose": entry["dose"]} for entry in entries]
    catalog = build_catalog(entries, layer_type_share(cells))
    heatmap = pack_heatmap(heatmap_scores(records))
    recipes = [{"cells": cell_list, "text": format_recipe(cell_list)} for cell_list in [
        [
            {"layer": entry["layer"], "neuron": entry["neuron"], "dose": entry["dose"]}
            for entry in entries
            if entry["family"] == name
        ]
        for name in FAMILIES
    ] if cell_list]
    return catalog, heatmap, recipes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("web/public"))
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)

    catalog, heatmap, recipes = build(load_all(args.records), limit=args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "catalog.json").write_text(json.dumps(catalog, indent=1))
    (args.out / "heatmap.bin").write_bytes(heatmap)
    (args.out / "recipes.json").write_text(json.dumps(recipes, indent=1))
    print(f"catalog: {catalog['count']} entries, families {catalog['families']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
