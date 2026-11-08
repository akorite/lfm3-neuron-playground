"""Compare the unmodified and steered graphs on one prompt.

Run next to the q4f16 graph and its external data. With every steer input at
zero the two graphs must agree; a nonzero steer must change the logits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TOKENS = (1, 2, 3, 4)


def feed(session, steer_layers: set[int], value: float) -> dict[str, np.ndarray]:
    length = len(TOKENS)
    feeds = {
        "input_ids": np.array([TOKENS], np.int64),
        "attention_mask": np.ones((1, length), np.int64),
        "position_ids": np.arange(length, dtype=np.int64)[None],
        "logits_to_keep": np.array([1], np.int64),
    }
    for inp in session.get_inputs():
        name = inp.name
        if name in feeds:
            continue
        if name.startswith("past_conv."):
            feeds[name] = np.zeros((1, 2048, 7), np.float16)
        elif name.startswith("past_key_values."):
            feeds[name] = np.zeros((1, 4, 0, 128), np.float16)
        elif name.startswith("steer."):
            layer = int(name.split(".")[1])
            fill = value if layer in steer_layers else 0.0
            feeds[name] = np.full((1, 1, 7680), fill, np.float16)
        else:
            raise ValueError(f"unexpected input {name}")
    return feeds


def logits(session, steer_layers: set[int], value: float) -> np.ndarray:
    return session.run(["logits"], feed(session, steer_layers, value))[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="unmodified graph")
    parser.add_argument("--steered", type=Path, required=True, help="graph with steer inputs")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--value", type=float, default=1.0)
    args = parser.parse_args(argv)

    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    original = ort.InferenceSession(str(args.graph), providers=providers)
    steered = ort.InferenceSession(str(args.steered), providers=providers)

    reference = logits(original, set(), 0.0)
    zero = logits(steered, set(), 0.0)
    pushed = logits(steered, {args.layer}, args.value)

    same = np.allclose(zero, reference, rtol=1e-2, atol=1e-2)
    changed = not np.allclose(pushed, zero, rtol=1e-2, atol=1e-2)
    print(f"zero steer matches original: {same}")
    print(f"steer on layer {args.layer} changes logits: {changed}")
    print(f"max |zero - original| = {np.abs(zero - reference).max():.6g}")
    return 0 if same and changed else 1


if __name__ == "__main__":
    sys.exit(main())
