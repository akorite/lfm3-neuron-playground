from __future__ import annotations

import numpy as np
import pytest

from tools.check_nvfp4_stats import column_norms, compare, dequantize, fp8_e4m3_to_float


def test_fp8_e4m3_decodes_known_values():
    raw = np.array([0x00, 0x38, 0x3C, 0xB8], dtype=np.uint8)
    values = fp8_e4m3_to_float(raw)
    np.testing.assert_allclose(values, [0.0, 1.0, 1.5, -1.0], rtol=0, atol=1e-6)


def test_fp8_nan_payload_decodes_to_zero():
    assert fp8_e4m3_to_float(np.array([0x7F], np.uint8))[0] == 0.0


def test_dequantize_unpacks_low_nibble_first():
    packed = np.array([[0x10]], dtype=np.uint8)
    scales = np.array([[0x38]], dtype=np.uint8)
    weight = dequantize(packed, scales, 1.0)
    np.testing.assert_allclose(weight, [[0.0, 0.5]], rtol=0, atol=1e-6)


def test_dequantize_applies_block_and_global_scales():
    packed = np.array([[0x11] * 8], dtype=np.uint8)
    scales = np.array([[0x38] * 16], dtype=np.uint8)
    weight = dequantize(packed, scales, 2.0)
    assert weight.shape == (1, 16)
    np.testing.assert_allclose(weight, np.full((1, 16), 1.0), rtol=0, atol=1e-6)


def test_dequantize_rejects_inconsistent_scale_blocks():
    with pytest.raises(ValueError, match="do not divide"):
        dequantize(np.zeros((1, 3), np.uint8), np.zeros((1, 4), np.uint8), 1.0)


def test_column_norms_are_per_input_neuron():
    weight = np.array([[3.0, 0.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(column_norms(weight), [5.0, 5.0], rtol=0, atol=1e-6)


def test_compare_gates_on_the_typical_error():
    stats = np.array([10.0, 20.0])
    assert compare(stats, np.array([10.4, 20.0]), 0.05)["within_tolerance"]
    assert not compare(stats, np.array([12.0, 20.0]), 0.05)["within_tolerance"]


def test_compare_reports_the_tail_without_gating_it():
    stats = np.full(100, 10.0)
    norms = stats.copy()
    norms[0] = 11.0
    result = compare(stats, norms, 0.05)
    assert result["within_tolerance"]
    assert result["max_relative_delta"] == pytest.approx(0.1)
    assert result["p99_relative_delta"] < 0.01


def test_compare_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare(np.zeros(3), np.zeros(4), 0.05)
