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
from research.prompts import DEFAULT_PROMPT, FAMILY_PROMPTS
from research.score import FAMILIES, coherence, fires, load_family, score, tokens
from research.shortlist import contrast_rank, read_records


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


def is_degenerate(text: str, family: dict) -> bool:
    """A loop, a stub, or a single repeated character is not a steerable behavior."""
    if family["type"] == "style":
        return False
    stripped = text.replace(" ", "").strip()
    if len(stripped) < 20 or len(set(stripped)) < 6:
        return True
    if coherence(text)["repetition"] >= 0.6:
        return True
    toks = tokens(text)
    if len(toks) >= 8 and len(set(toks)) / len(toks) < 0.3:
        return True
    return False


def share_from_records(records: list[dict]) -> dict[str, dict]:
    """Layer-type share of cells whose text clears any family threshold.

    Computed from the raw sweep, not from the balanced shortlist, so the
    published finding is not forced toward an even split.
    """
    from research.model_info import TYPE_ORDER
    from research.score import fires

    families = [load_family(name) for name in FAMILIES]
    counts = {kind: 0 for kind in TYPE_ORDER}
    # Only the balanced nine-layer probe supports a layer-type comparison;
    # the prompt-specific sweeps cover a subset of layers.
    balanced = [r for r in records if r.get("prompt", DEFAULT_PROMPT) == DEFAULT_PROMPT]
    for record in balanced:
        layer = record["layer"]
        if layer not in SAMPLED_LAYERS:
            continue
        hit = any(
            fires(score(record["text"], family), family)
            for family in families
            if family["type"] != "judge"
        )
        if hit:
            counts[layer_type(layer)] += 1
    total = sum(counts.values()) or 1
    return {
        kind: {"count": counts[kind], "share": round(counts[kind] / total, 4)}
        for kind in TYPE_ORDER
    }


def build(records: list[dict], limit: int = 40, allow_missing_family: bool = False) -> tuple[dict, bytes, list[dict]]:
    entries = []
    for name in FAMILIES:
        family = load_family(name)
        prompt = FAMILY_PROMPTS.get(name, DEFAULT_PROMPT)
        family_records = [r for r in records if r.get("prompt", DEFAULT_PROMPT) == prompt]
        # A loop like "war war war" scores as hostile but reads as a broken model.
        usable = [
            record for record in family_records if not is_degenerate(record["text"], family)
        ]
        for item in contrast_rank(usable, family, limit=limit):
            entries.append(
                {
                    "family": name,
                    "layer": item["layer"],
                    "neuron": item["neuron"],
                    "dose": item["sign"],
                    "layer_type": layer_type(item["layer"]),
                    "score": item["score"],
                    "contrast": item["contrast"],
                    "text": item["text"],
                }
            )
    missing = sorted(set(FAMILIES) - {entry["family"] for entry in entries})
    if missing and not allow_missing_family:
        raise ValueError(
            f"no shortlisted entries for {missing}; supply judge scores for honesty or pass "
            "--allow-missing-family to publish a partial catalog"
        )
    catalog = build_catalog(entries, share_from_records(records))
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
    parser.add_argument("--allow-missing-family", action="store_true")
    args = parser.parse_args(argv)

    catalog, heatmap, recipes = build(
        load_all(args.records), limit=args.limit, allow_missing_family=args.allow_missing_family
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "catalog.json").write_text(json.dumps(catalog, indent=1))
    (args.out / "heatmap.bin").write_bytes(heatmap)
    (args.out / "recipes.json").write_text(json.dumps(recipes, indent=1))
    print(f"catalog: {catalog['count']} entries, families {catalog['families']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
