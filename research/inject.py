"""Per-row neuron injection for the BF16 sweep.

Each batch row carries its own neuron and delta, so one forward pass measures
64 different neurons on the same prompt.
"""

from __future__ import annotations


class SteerHook:
    """Adds `delta[i]` to `act[i, :, neuron[i]]` on every call."""

    def __init__(self, neurons, deltas, device=None):
        import torch

        self.neurons = torch.as_tensor(neurons, dtype=torch.long, device=device)
        self.deltas = torch.as_tensor(deltas, dtype=torch.float32, device=device)

    def __call__(self, module, args):
        import torch

        activation = args[0]
        batch = activation.shape[0]
        if batch > self.neurons.shape[0]:
            raise ValueError(f"batch {batch} exceeds prepared rows {self.neurons.shape[0]}")
        neurons = self.neurons[:batch].to(activation.device)
        deltas = self.deltas[:batch].to(activation.device, activation.dtype)
        updated = activation.clone()
        rows = torch.arange(batch, device=activation.device)
        updated[rows, :, neurons] += deltas[:, None]
        return (updated,) + tuple(args[1:])


def attach(layer, neurons, deltas, device=None):
    """Register the hook on a Linear and return the handle."""
    return layer.register_forward_pre_hook(SteerHook(neurons, deltas, device))


def dose_deltas(thresholds, column_norms, layer, neurons, dose):
    """delta = dose * threshold[layer] / norm[layer][neuron]."""
    return [dose * float(thresholds[layer]) / float(column_norms[layer][n]) for n in neurons]
