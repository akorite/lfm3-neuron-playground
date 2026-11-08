"""Rank swept cells by family effect and keep the shortlist layer-balanced."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from research.model_info import TYPE_ORDER, layer_type
from research.score import score


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
