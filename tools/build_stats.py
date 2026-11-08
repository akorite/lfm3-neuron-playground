"""Build and read `steer_stats.bin`.

Layout (little-endian float32): 24 per-layer thresholds (median residual norm
leaving the layer), then 24 x 7680 column norms of the down-projection weight.
A dose of 1 moves a neuron by `delta = dose * threshold[layer] / norm[layer][n]`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LAYERS = 24
INTERMEDIATE = 7680
HEADER = LAYERS
TOTAL = LAYERS + LAYERS * INTERMEDIATE


def write_stats(path: Path, thresholds: np.ndarray, column_norms: np.ndarray) -> None:
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    column_norms = np.asarray(column_norms, dtype=np.float32)
    if thresholds.shape != (LAYERS,):
        raise ValueError(f"expected {LAYERS} thresholds, got {thresholds.shape}")
    if column_norms.shape != (LAYERS, INTERMEDIATE):
        raise ValueError(f"expected {LAYERS}x{INTERMEDIATE} column norms, got {column_norms.shape}")
    if not np.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError("thresholds must be finite and positive")
    if not np.isfinite(column_norms).all() or (column_norms <= 0).any():
        raise ValueError("column norms must be finite and positive")
    data = np.concatenate([thresholds, column_norms.reshape(-1)]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes(order="C"))


def read_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size != TOTAL:
        raise ValueError(f"expected {TOTAL} float32 values, got {raw.size}")
    return raw[:HEADER].copy(), raw[HEADER:].reshape(LAYERS, INTERMEDIATE).copy()


def collect_stats(model, tokenizer, calibration_text: str, tokens: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Measure thresholds and column norms on a loaded BF16 model."""
    import torch

    layers = model.model.layers
    device = next(model.parameters()).device
    captured: dict[int, float] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            norms = torch.linalg.vector_norm(hidden.detach().float(), dim=-1)[0]
            if norms.numel() > 1:
                norms = norms[1:]
            captured[index] = float(norms.median())

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        encoded = tokenizer(
            calibration_text, return_tensors="pt", truncation=True, max_length=tokens
        ).to(device)
        with torch.no_grad():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    thresholds = np.array([captured[i] for i in range(len(layers))], dtype=np.float32)
    norms = np.stack(
        [
            torch.linalg.vector_norm(layer.feed_forward.w2.weight.float(), dim=1).cpu().numpy()
            for layer in layers
        ]
    )
    return thresholds, norms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--calibration", type=Path, default=Path("research/calibration.txt"))
    parser.add_argument("--output", type=Path, default=Path("web/public/steer_stats.bin"))
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    thresholds, norms = collect_stats(model, tokenizer, args.calibration.read_text())
    write_stats(args.output, thresholds, norms)
    print(f"wrote {args.output} from {len(thresholds)} layers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
