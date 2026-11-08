"""Catalog acceptance, heatmap packing, and the neuronary recipe format."""

from __future__ import annotations

import re

import numpy as np

from research.model_info import SAMPLED_LAYERS, TYPE_ORDER, layer_type
from research.score import fires

INTERMEDIATE = 7680
RECIPE_TOKEN = re.compile(r"L(\d+)N(\d+)([+-]\d+(?:\.\d+)?)")


def parse_recipe(text: str) -> list[dict]:
    cells = [
        {"layer": int(layer), "neuron": int(neuron), "dose": float(dose)}
        for layer, neuron, dose in RECIPE_TOKEN.findall(text)
    ]
    if not cells:
        raise ValueError(f"no recipe cells in {text!r}")
    for cell in cells:
        if not 0 <= cell["layer"] < 24 or not 0 <= cell["neuron"] < INTERMEDIATE:
            raise ValueError(f"cell outside the model: {cell}")
    return cells


def format_recipe(cells: list[dict]) -> str:
    return " ".join(f"L{cell['layer']}N{cell['neuron']}{cell['dose']:+g}" for cell in cells)


def accepts(score_at_dose: float | None, score_at_lower: float | None, family: dict) -> bool:
    """A recipe ships only when the effect holds at its dose and at 75% of it."""
    return fires(score_at_dose, family) and fires(score_at_lower, family)


def accepts_window(scores: list[float | None], family: dict) -> bool:
    """A recipe ships only when the effect holds across the whole ±25% window."""
    return bool(scores) and all(fires(value, family) for value in scores)


def layer_type_share(cells: list[dict]) -> dict[str, dict]:
    counts = {kind: 0 for kind in TYPE_ORDER}
    for cell in cells:
        counts[layer_type(cell["layer"])] += 1
    total = sum(counts.values())
    return {
        kind: {
            "count": counts[kind],
            "share": round(counts[kind] / total, 4) if total else 0.0,
        }
        for kind in TYPE_ORDER
    }


def pack_heatmap(scores: dict[int, np.ndarray]) -> bytes:
    """One int8 per cell, per layer normalized to its own maximum."""
    packed = []
    for layer in SAMPLED_LAYERS:
        values = np.asarray(scores.get(layer, np.zeros(INTERMEDIATE)), dtype=np.float32)
        if values.shape != (INTERMEDIATE,):
            raise ValueError(f"layer {layer} heatmap row must be {INTERMEDIATE} values")
        peak = float(values.max()) if values.size else 0.0
        normalized = values / peak if peak > 0 else values
        packed.append(np.clip(normalized * 255, 0, 255).astype(np.uint8))
    return np.stack(packed).tobytes()


def read_heatmap(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(SAMPLED_LAYERS), INTERMEDIATE)


def finding(share: dict[str, dict]) -> dict:
    """The one research claim the catalog states."""
    return {
        "sampled_layers": list(SAMPLED_LAYERS),
        "by_layer_type": share,
        "note": "share of shortlisted neurons by layer type",
    }


def build_catalog(entries: list[dict], share: dict[str, dict]) -> dict:
    families = sorted({entry["family"] for entry in entries})
    types = sorted({layer_type(entry["layer"]) for entry in entries})
    return {
        "model": "LiquidAI/LFM3-1.7B",
        "count": len(entries),
        "families": families,
        "layer_types": types,
        "finding": finding(share),
        "entries": entries,
    }
