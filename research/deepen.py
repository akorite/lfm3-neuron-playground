"""Re-run selected cells with a longer rollout and the family's own prompt.

The broad sweep uses a single 24-token probe per cell. Candidates that survive
contrast ranking get a 64-token rollout here, so the text that reaches the
catalog is long enough to show the behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from research.inject import attach, dose_deltas
from research.prompts import FAMILY_PROMPTS
from research.sweep import chunks
from tools.build_stats import INTERMEDIATE, read_stats


def groups(cells: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for cell in cells:
        grouped[(cell["layer"], cell["sign"])].append(cell)
    return grouped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=Path("web/public/steer_stats.bin"))
    parser.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cells = json.loads(args.cells.read_text())
    thresholds, norms = read_stats(args.stats)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out.open("w") as handle, torch.no_grad():
        for (layer, sign), group in groups(cells).items():
            prompt = FAMILY_PROMPTS[group[0]["family"]]
            block = model.model.layers[layer].feed_forward.w2
            for batch in chunks(group, args.batch):
                neurons = [cell["neuron"] for cell in batch]
                deltas = dose_deltas(thresholds, norms, layer, neurons, sign)
                hook = attach(block, neurons, deltas)
                try:
                    encoded = tokenizer(
                        [prompt] * len(batch), return_tensors="pt", padding=True
                    ).to("cuda")
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    texts = tokenizer.batch_decode(
                        generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                finally:
                    hook.remove()
                for cell, text in zip(batch, texts):
                    handle.write(
                        json.dumps(
                            {
                                "layer": layer,
                                "neuron": int(cell["neuron"]),
                                "sign": sign,
                                "prompt": prompt,
                                "family": cell["family"],
                                "text": text,
                            }
                        )
                        + "\n"
                    )
                    written += 1
                handle.flush()
    print(f"wrote {written} deep records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
