"""Static facts about LFM3-1.7B that the pipeline and the app share."""

from __future__ import annotations

LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
]

SAMPLED_LAYERS = (5, 11, 17, 8, 12, 16, 10, 14, 18)
TYPE_ORDER = ("conv", "sliding_attention", "full_attention")
TYPE_LABELS = {
    "conv": "gated convolution",
    "sliding_attention": "sliding-window attention",
    "full_attention": "global attention",
}


def layer_type(layer: int) -> str:
    if not 0 <= layer < len(LAYER_TYPES):
        raise ValueError(f"layer {layer} outside 0..{len(LAYER_TYPES) - 1}")
    return LAYER_TYPES[layer]
