from __future__ import annotations

import json
from pathlib import Path

from research.score import coherence, fires, load_family, repetition_ratio, score
from research.shortlist import layer_type_share, rank, read_records

HOSTILE = "I will kill you. I hate you, I want you to suffer and die."
NEUTRAL = "The library opens at nine and the bus leaves at noon."


def write_records(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_persona_lexicon_ranks_hostile_above_neutral():
    family = load_family("personas")
    assert score(HOSTILE, family) > score(NEUTRAL, family)
    assert fires(score(HOSTILE, family), family)
    assert not fires(score(NEUTRAL, family), family)


def test_style_flags_caps_and_repetition():
    family = load_family("style")
    assert fires(score("THIS ANSWER IS ENTIRELY SHOUTED BACK AT YOU", family), family)
    loop = "the cat sat on the mat " * 6
    assert repetition_ratio(loop) > 0.5
    assert fires(score(loop, family), family)


def test_language_detects_script_switch():
    family = load_family("language")
    assert fires(score("这是一个中文回答，用于语言切换测试。", family), family)
    assert not fires(score(NEUTRAL, family), family)


def test_competence_matches_tool_call_markup():
    family = load_family("competence")
    assert fires(score("<tool_call>\n<function=get_weather>", family), family)
    assert not fires(score(NEUTRAL, family), family)


def test_honesty_abstains_without_a_judge():
    family = load_family("honesty")
    assert score(HOSTILE, family) is None
    assert not fires(None, family)


def test_coherence_reports_length_and_repetition():
    report = coherence("one two three")
    assert report["words"] == 3
    assert report["repetition"] == 0.0


def test_rank_orders_by_score_and_keeps_all_layer_types(tmp_path):
    rows = [
        {"layer": 5, "neuron": 1, "sign": 1, "text": HOSTILE},
        {"layer": 5, "neuron": 2, "sign": 1, "text": NEUTRAL},
        {"layer": 8, "neuron": 3, "sign": 1, "text": HOSTILE * 2},
        {"layer": 10, "neuron": 4, "sign": 1, "text": HOSTILE},
    ]
    records = read_records(write_records(tmp_path / "r.jsonl", rows))
    shortlist = rank(records, load_family("personas"), limit=3)

    assert len(shortlist) == 3
    assert shortlist[0]["score"] >= shortlist[-1]["score"]
    assert set(layer_type_share(shortlist)) == {"conv", "sliding_attention", "full_attention"}
    assert sum(layer_type_share(shortlist).values()) == 3


def test_rank_skips_judge_family(tmp_path):
    rows = [{"layer": 5, "neuron": 1, "sign": 1, "text": HOSTILE}]
    records = read_records(write_records(tmp_path / "r.jsonl", rows))
    assert rank(records, load_family("honesty")) == []
