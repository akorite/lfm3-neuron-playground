from __future__ import annotations

from pathlib import Path

from research.sweep import chunks, is_done, mark_done, marker


def test_unfinished_layer_is_not_done(tmp_path: Path):
    assert not is_done(tmp_path, "run", 5, 1)


def test_marker_with_full_count_is_done(tmp_path: Path):
    mark_done(tmp_path, "run", 5, 1, 7680)
    assert is_done(tmp_path, "run", 5, 1)
    assert not is_done(tmp_path, "run", 5, -1)
    assert not is_done(tmp_path, "run", 8, 1)


def test_partial_marker_is_not_done(tmp_path: Path):
    mark_done(tmp_path, "run", 5, 1, 7600)
    assert not is_done(tmp_path, "run", 5, 1)


def test_corrupt_marker_is_not_done(tmp_path: Path):
    path = marker(tmp_path, "run", 5, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert not is_done(tmp_path, "run", 5, 1)


def test_resume_skips_completed_cells_and_continues():
    cells = [(5, 1), (5, -1), (8, 1)]
    done = {(5, 1)}
    assert [cell for cell in cells if cell not in done] == [(5, -1), (8, 1)]


def test_chunks_cover_every_neuron_once():
    items = list(range(10))
    blocks = list(chunks(items, 4))
    assert [len(block) for block in blocks] == [4, 4, 2]
    assert [n for block in blocks for n in block] == items
