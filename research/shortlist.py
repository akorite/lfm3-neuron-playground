"""Rank swept cells by family effect and keep the shortlist layer-balanced."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from research.model_info import TYPE_ORDER, layer_type
from research.score import fires, score


def read_records(path: Path) -> list[dict]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rank(records: list[dict], family: dict, limit: int = 150) -> list[dict]:
    """Top-scoring cells, capped evenly across the three layer types."""
    scored = []
    for record in records:
        value = score(record["text"], family)
        if value is None:
            continue
        scored.append((value, record))
    scored.sort(key=lambda item: -item[0])

    cap = max(1, limit // len(TYPE_ORDER))
    per_type: dict[str, int] = defaultdict(int)
    shortlist: list[dict] = []
    for value, record in scored:
        kind = layer_type(record["layer"])
        if per_type[kind] >= cap:
            continue
        per_type[kind] += 1
        shortlist.append({**record, "score": round(value, 4), "layer_type": kind})
        if len(shortlist) >= limit:
            break
    return shortlist


def layer_type_share(shortlist: list[dict]) -> dict[str, int]:
    counts = {kind: 0 for kind in TYPE_ORDER}
    for entry in shortlist:
        counts[entry["layer_type"]] += 1
    return counts


def contrast_rank(records: list[dict], family: dict, limit: int = 12) -> list[dict]:
    """Rank by the +1 vs -1 contrast on the same neuron.

    A surface score at one sign cannot tell a causal neuron from a prompt echo:
    both signs score high for an echo, so the contrast is near zero. The
    direction with the larger score becomes the recipe's dose.
    """
    cells: dict[tuple[int, int], dict[int, float | None]] = defaultdict(dict)
    texts: dict[tuple[int, int], dict[int, str]] = defaultdict(dict)
    for record in records:
        cells[(record["layer"], record["neuron"])][record["sign"]] = score(record["text"], family)
        texts[(record["layer"], record["neuron"])][record["sign"]] = record["text"]
    floor = float(family.get("min_contrast", 0.0))

    rows = []
    for (layer, neuron), signs in cells.items():
        plus, minus = signs.get(1), signs.get(-1)
        if plus is None or minus is None:
            continue
        contrast = plus - minus
        if abs(contrast) < floor:
            continue
        best = max(plus, minus)
        if not fires(best, family):
            continue
        sign = 1 if contrast >= 0 else -1
        rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "sign": sign,
                "score": round(best, 4),
                "contrast": round(contrast, 4),
                "text": texts[(layer, neuron)][sign],
            }
        )
    rows.sort(key=lambda row: -abs(row["contrast"]))

    cap = max(1, limit // len(TYPE_ORDER))
    per_type: dict[str, int] = defaultdict(int)
    shortlist: list[dict] = []
    for row in rows:
        kind = layer_type(row["layer"])
        if per_type[kind] >= cap:
            continue
        per_type[kind] += 1
        shortlist.append({**row, "layer_type": kind})
        if len(shortlist) >= limit:
            break
    return shortlist
