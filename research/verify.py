"""Re-run shortlisted recipes on the shipped q4f16 graph and keep what reproduces.

A recipe survives only when its family scorer still fires at the stated dose
and at 75% of it. The runner owns the ONNX cache wiring: 13 conv states, 11
key/value pairs, 256-token prefill chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import numpy as np

from research.catalog import INTERMEDIATE, window_verdict
from research.prompts import FAMILY_PROMPTS
from research.score import load_family, score
from tools.build_stats import LAYERS, read_stats

CONV_LAYERS = (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)

# A family that needs a different elicitation than the persona probe.
FAMILY_PROMPTS = {
    "personas": "Who are you?",
    "style": "Who are you?",
    "competence": "Write a Python function that reverses a string.",
    "language": "请用中文回答：你是谁？",
    "honesty": "Who are you?",
}
KV_LAYERS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
HIDDEN = 2048
HEADS = 4
HEAD_DIM = 128
CHUNK = 256


class SteeredSession:
    def __init__(self, graph: Path, providers: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(graph), providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.outputs = [value.name for value in self.session.get_outputs()]
        self.has_steer = any(value.name == "steer.0" for value in self.session.get_inputs())

    def caches(self) -> dict[str, np.ndarray]:
        caches = {f"conv.{layer}": np.zeros((1, HIDDEN, 7), np.float16) for layer in CONV_LAYERS}
        for layer in KV_LAYERS:
            caches[f"kv.{layer}.key"] = np.zeros((1, HEADS, 0, HEAD_DIM), np.float16)
            caches[f"kv.{layer}.value"] = np.zeros((1, HEADS, 0, HEAD_DIM), np.float16)
        return caches

    def feeds(self, ids: list[int], position: int, caches: dict, steer: dict) -> dict:
        length = len(ids)
        feeds = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, position + length), np.int64),
            "position_ids": np.arange(position, position + length, dtype=np.int64)[None],
            "logits_to_keep": np.array([1], np.int64),
            **steer,
        }
        for key, cache in caches.items():
            if key.startswith("conv."):
                feeds[f"past_conv.{key[5:]}"] = cache
            else:
                _, layer, part = key.split(".")
                feeds[f"past_key_values.{layer}.{part}"] = cache
        return feeds

    def step(self, ids: list[int], position: int, caches: dict, steer: dict):
        results = dict(zip(self.outputs, self.session.run(None, self.feeds(ids, position, caches, steer))))
        for layer in CONV_LAYERS:
            caches[f"conv.{layer}"] = results[f"present_conv.{layer}"]
        for layer in KV_LAYERS:
            for part in ("key", "value"):
                caches[f"kv.{layer}.{part}"] = results[f"present.{layer}.{part}"]
        return results["logits"]


def steer_vectors(stats, cells: list[dict]) -> dict[str, np.ndarray]:
    thresholds, norms = stats
    vectors = {f"steer.{layer}": np.zeros((1, 1, INTERMEDIATE), np.float16) for layer in range(LAYERS)}
    for cell in cells:
        layer, neuron, dose = cell["layer"], cell["neuron"], cell["dose"]
        value = dose * float(thresholds[layer]) / float(norms[layer][neuron])
        vectors[f"steer.{layer}"][0, 0, neuron] = np.float16(value)
    return vectors


def encode(tokenizer, prompt: str) -> list[int]:
    try:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
    except (ImportError, TypeError, ValueError):
        result = tokenizer(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]
    if hasattr(result, "keys") and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], (list, tuple)):
        result = result[0]
    return [int(token) for token in result]


def generate(runner: SteeredSession, tokenizer, stats, cells: list[dict], prompt: str, tokens: int) -> str:
    steer = steer_vectors(stats, cells) if runner.has_steer else {}
    ids = encode(tokenizer, prompt)
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    caches = runner.caches()
    position = 0
    logits = None
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start : start + CHUNK]
        logits = runner.step(chunk, position, caches, steer)
        position += len(chunk)

    produced: list[int] = []
    for _ in range(tokens):
        next_id = int(np.argmax(logits[0, -1]))
        produced.append(next_id)
        logits = runner.step([next_id], position, caches, steer)
        position += 1
    return tokenizer.decode(produced, skip_special_tokens=True)


def clears_baseline(score_at_dose, baseline_score, family) -> bool:
    """The steered output must beat what the unsteered prompt already produces."""
    if score_at_dose is None:
        return False
    return score_at_dose - (baseline_score or 0.0) >= float(family.get("baseline_margin", 0.0))


def verify_entry(runner, tokenizer, stats, entry: dict, prompt: str, tokens: int, baselines: dict) -> dict:
    family = load_family(entry["family"])
    prompt = FAMILY_PROMPTS.get(entry["family"], prompt)
    baseline = baselines.get(prompt, "")
    cell = {"layer": entry["layer"], "neuron": entry["neuron"], "dose": entry["dose"]}
    factors = (0.75, 1.0, 1.25)
    texts = [
        generate(runner, tokenizer, stats, [{**cell, "dose": cell["dose"] * factor}], prompt, tokens)
        for factor in factors
    ]
    scores = [score(text, family) for text in texts]
    baseline_score = score(baseline, family)
    verdict = window_verdict(scores, family)
    verdict["accept"] = verdict["accept"] and clears_baseline(scores[1], baseline_score, family)
    return {
        **entry,
        "verified": verdict["accept"],
        "stable_above_25": verdict["stable_above_25"],
        "score_at_dose": scores[1],
        "score_at_lower": scores[0],
        "score_at_upper": scores[2],
        "score_baseline": baseline_score,
        "text": texts[1],
        "text_at_dose": texts[1],
        "text_baseline": baseline,
        "prompt": prompt,
    }


def refresh(catalog: dict, accepted: list[dict]) -> dict:
    catalog["entries"] = accepted
    catalog["count"] = len(accepted)
    catalog["families"] = sorted({entry["family"] for entry in accepted})
    catalog["layer_types"] = sorted({entry["layer_type"] for entry in accepted})
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=Path("web/public/steer_stats.bin"))
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Who are you?")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    catalog = json.loads(args.catalog.read_text())
    entries = catalog["entries"]
    if args.limit:
        if args.out is None:
            parser.error("--limit rewrites only the checked entries; pass --out")
        entries = entries[: args.limit]

    runner = SteeredSession(args.graph)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stats = read_stats(args.stats)

    prompts = {FAMILY_PROMPTS.get(entry["family"], args.prompt) for entry in entries}
    baselines = {prompt: generate(runner, tokenizer, stats, [], prompt, args.tokens) for prompt in prompts}
    checked = [
        verify_entry(runner, tokenizer, stats, entry, args.prompt, args.tokens, baselines)
        for entry in entries
    ]
    accepted = [entry for entry in checked if entry["verified"]]
    catalog = refresh(catalog, accepted)

    out = args.out or args.catalog
    out.write_text(json.dumps(catalog, indent=1))
    report = out.with_name("verification.json")
    report.write_text(json.dumps({"checked": len(checked), "accepted": len(accepted), "results": checked}, indent=1))
    print(f"verified {len(accepted)}/{len(checked)} recipes; report at {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
