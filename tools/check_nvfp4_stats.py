"""Compare NVFP4 down-projection column norms against the BF16 statistics.

R14 allows the NVFP4 checkpoint only as an offline weight check. This reads the
quantized `w2` tensors, dequantizes them (FP4 E2M1 x per-16 block scale x one
global scale), recomputes the per-neuron column norms, and reports the relative
delta per layer against `steer_stats.bin`. No NVFP4 kernel runs.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

FP4 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
BLOCK = 16


def fp8_e4m3_to_float(raw: np.ndarray) -> np.ndarray:
    """Decode E4M3FN bytes; NaN payloads are treated as zero."""

    bits = np.asarray(raw, dtype=np.uint8)
    sign = np.where((bits >> 7) & 1, -1.0, 1.0)
    exponent = (bits >> 3) & 0x0F
    mantissa = (bits & 0x07).astype(np.float32) / 8.0
    values = np.where(exponent == 0, mantissa * 2.0**-6, (1.0 + mantissa) * np.exp2(exponent.astype(np.float32) - 7))
    values = np.where((exponent == 0x0F) & ((bits & 0x07) == 0x07), 0.0, values)
    return (sign * values).astype(np.float32)


def dequantize(packed: np.ndarray, scales: np.ndarray, global_scale: float) -> np.ndarray:
    """Unpack two FP4 values per byte; low nibble is the even input index."""

    packed = np.asarray(packed, dtype=np.uint8)
    rows, half = packed.shape
    elements = np.empty((rows, half * 2), dtype=np.uint8)
    elements[:, 0::2] = packed & 0x0F
    elements[:, 1::2] = packed >> 4
    values = FP4[elements]
    block = fp8_e4m3_to_float(scales) * np.float32(global_scale)
    repeat = values.shape[1] // block.shape[1]
    if repeat * block.shape[1] != values.shape[1]:
        raise ValueError(f"scale blocks {block.shape[1]} do not divide {values.shape[1]}")
    return values * np.repeat(block, repeat, axis=1)


def column_norms(weight: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(weight, dtype=np.float32) ** 2).sum(axis=0))


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return 8 + length, json.loads(handle.read(length))


def read_tensor(path: Path, base: int, header: dict, name: str, dtype) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(end - start)
    return np.frombuffer(raw, dtype=dtype).reshape(info["shape"])


def compare(stats_norms: np.ndarray, nvfp4_norms: np.ndarray, tolerance: float) -> dict:
    if stats_norms.shape != nvfp4_norms.shape:
        raise ValueError(f"shape mismatch: stats {stats_norms.shape} vs nvfp4 {nvfp4_norms.shape}")
    delta = np.abs(nvfp4_norms - stats_norms) / stats_norms
    return {
        "mean_relative_delta": float(delta.mean()),
        "p99_relative_delta": float(np.percentile(delta, 99)),
        "max_relative_delta": float(delta.max()),
        "fraction_over_tolerance": float((delta > tolerance).mean()),
        # Each neuron's own norm scales its dose, so the gate is on the tail:
        # 99% of neurons must stay inside the tolerance.
        "within_tolerance": bool(np.percentile(delta, 99) <= tolerance),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="model.safetensors from the NVFP4 repo")
    parser.add_argument("--stats", type=Path, default=Path("web/public/steer_stats.bin"))
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 11, 17, 8, 12, 16, 10, 14, 18])
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("nvfp4_check.json"))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.build_stats import read_stats

    base, header = read_header(args.file)
    _, stats_norms = read_stats(args.stats)

    results = {}
    for layer in args.layers:
        prefix = f"model.layers.{layer}.feed_forward.w2"
        packed = read_tensor(args.file, base, header, f"{prefix}.weight", np.uint8)
        scales = read_tensor(args.file, base, header, f"{prefix}.weight_scale", np.uint8)
        global_scale = float(np.asarray(
            read_tensor(args.file, base, header, f"{prefix}.weight_scale_2", np.float32)
        ).reshape(-1)[0])
        norms = column_norms(dequantize(packed, scales, global_scale))
        results[str(layer)] = compare(stats_norms[layer], norms, args.tolerance)
        entry = results[str(layer)]
        print(
            f"layer {layer}: mean {entry['mean_relative_delta']:.4f}  p99 {entry['p99_relative_delta']:.4f}  "
            f"max {entry['max_relative_delta']:.4f}  over {entry['fraction_over_tolerance']:.2e}"
        )

    verdict = all(entry["within_tolerance"] for entry in results.values())
    args.out.write_text(json.dumps({"tolerance": args.tolerance, "layers": results, "pass": verdict}, indent=1))
    print(f"pass={verdict}; report at {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
