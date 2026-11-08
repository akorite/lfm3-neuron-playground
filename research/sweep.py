"""Batched neuron sweep over the sampled layers, resumable per (layer, sign)."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from research.inject import attach, dose_deltas
from tools.build_stats import INTERMEDIATE, LAYERS, read_stats

SAMPLED_LAYERS = (5, 11, 17, 8, 12, 16, 10, 14, 18)
PROMPT = "Who are you?"
SIGNS = (1, -1)


def run_dir(root: Path, name: str) -> Path:
    return root / name


def marker(root: Path, name: str, layer: int, sign: int) -> Path:
    return root / name / "done" / f"L{layer}_{sign:+d}.json"


def is_done(root: Path, name: str, layer: int, sign: int) -> bool:
    path = marker(root, name, layer, sign)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return record.get("count") == INTERMEDIATE


def mark_done(root: Path, name: str, layer: int, sign: int, count: int) -> None:
    path = marker(root, name, layer, sign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layer": layer, "sign": sign, "count": count}))


def records_path(root: Path, name: str, layer: int, sign: int) -> Path:
    return root / name / f"L{layer}_{sign:+d}.jsonl"


def prepare_records(root: Path, name: str, layer: int, sign: int) -> Path:
    """A layer without a done marker is partial: drop it and rerun the layer."""
    path = records_path(root, name, layer, sign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def chunks(items, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sweep_sign(model, tokenizer, stats, root: Path, name: str, layer: int, sign: int,
               batch: int = 64, tokens: int = 24, device: str = "cuda") -> int:
    import torch

    if layer not in range(LAYERS):
        raise ValueError(f"layer {layer} outside 0..{LAYERS - 1}")
    if is_done(root, name, layer, sign):
        return INTERMEDIATE

    thresholds, column_norms = stats
    block = model.model.layers[layer].feed_forward.w2
    out_path = prepare_records(root, name, layer, sign)
    pad_id = tokenizer.pad_token_id
    written = 0

    with out_path.open("a") as handle, torch.no_grad():
        for neurons in chunks(np.arange(INTERMEDIATE), batch):
            deltas = dose_deltas(thresholds, column_norms, layer, neurons, sign)
            hook = attach(block, neurons, deltas, device)
            try:
                encoded = tokenizer([PROMPT] * len(neurons), return_tensors="pt", padding=True).to(device)
                generated = model.generate(
                    **encoded,
                    max_new_tokens=tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
                texts = tokenizer.batch_decode(
                    generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                )
            finally:
                hook.remove()
            for neuron, text in zip(neurons, texts):
                handle.write(
                    json.dumps({"layer": layer, "neuron": int(neuron), "sign": sign, "text": text}) + "\n"
                )
            written += len(neurons)
            handle.flush()

    if written == INTERMEDIATE:
        mark_done(root, name, layer, sign, written)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--stats", type=Path, default=Path("web/public/steer_stats.bin"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="sweep")
    parser.add_argument("--layers", type=int, nargs="+", default=list(SAMPLED_LAYERS))
    parser.add_argument("--signs", type=int, nargs="+", default=list(SIGNS))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=24)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stats = read_stats(args.stats)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    total = 0
    for layer in args.layers:
        for sign in args.signs:
            wrote = sweep_sign(
                model, tokenizer, stats, args.out, args.name, layer, sign,
                batch=args.batch, tokens=args.tokens,
            )
            total += wrote
            print(f"layer {layer} sign {sign:+d}: {wrote} records")
    print(f"total {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
