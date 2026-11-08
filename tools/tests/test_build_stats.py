from __future__ import annotations

import numpy as np
import pytest

from tools.build_stats import HEADER, INTERMEDIATE, LAYERS, TOTAL, read_stats, write_stats


def sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    thresholds = rng.uniform(1.0, 20.0, LAYERS).astype(np.float32)
    norms = rng.uniform(0.01, 1.0, (LAYERS, INTERMEDIATE)).astype(np.float32)
    return thresholds, norms


def test_roundtrip(tmp_path):
    thresholds, norms = sample()
    path = tmp_path / "steer_stats.bin"
    write_stats(path, thresholds, norms)
    read_thresholds, read_norms = read_stats(path)
    np.testing.assert_array_equal(read_thresholds, thresholds)
    np.testing.assert_array_equal(read_norms, norms)


def test_file_layout(tmp_path):
    thresholds, norms = sample()
    path = tmp_path / "steer_stats.bin"
    write_stats(path, thresholds, norms)
    assert path.stat().st_size == TOTAL * 4
    raw = np.fromfile(path, dtype=np.float32)
    assert raw[HEADER - 1] == pytest.approx(thresholds[-1])
    assert raw[HEADER] == pytest.approx(norms[0, 0])


def test_wrong_shapes_rejected(tmp_path):
    thresholds, norms = sample()
    with pytest.raises(ValueError, match="thresholds"):
        write_stats(tmp_path / "a.bin", thresholds[:-1], norms)
    with pytest.raises(ValueError, match="column norms"):
        write_stats(tmp_path / "b.bin", thresholds, norms[:, :-1])


def test_nonpositive_values_rejected(tmp_path):
    thresholds, norms = sample()
    bad = thresholds.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        write_stats(tmp_path / "a.bin", bad, norms)
    norms_bad = norms.copy()
    norms_bad[3, 7] = np.nan
    with pytest.raises(ValueError, match="positive"):
        write_stats(tmp_path / "b.bin", thresholds, norms_bad)


def test_truncated_file_rejected(tmp_path):
    path = tmp_path / "short.bin"
    np.zeros(10, np.float32).tofile(path)
    with pytest.raises(ValueError, match="float32 values"):
        read_stats(path)
