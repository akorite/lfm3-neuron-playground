"""Per-family effect scores for one rollout of text."""

from __future__ import annotations

import json
import re
from pathlib import Path

FAMILIES_DIR = Path(__file__).parent / "families"
FAMILIES = ("personas", "style", "competence", "language", "honesty")

WORD = re.compile(r"[A-Za-z']+")


def load_family(name: str) -> dict:
    return json.loads((FAMILIES_DIR / f"{name}.json").read_text())


def words(text: str) -> list[str]:
    return WORD.findall(text)


def per_hundred(text: str, count: float) -> float:
    return 100.0 * count / max(1, len(words(text)))


def lexicon_score(text: str, terms: list[str]) -> float:
    lowered = text.lower()
    return per_hundred(text, sum(lowered.count(term.lower()) for term in terms))


def regex_score(text: str, patterns: list[str]) -> float:
    return per_hundred(text, sum(len(re.findall(pattern, text)) for pattern in patterns))


def uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for char in letters if char.isupper()) / len(letters)


def repetition_ratio(text: str) -> float:
    tokens = [token.lower() for token in words(text)]
    if len(tokens) < 8:
        return 0.0
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def script_ratio(text: str, ranges: list[list[int]]) -> float:
    characters = [char for char in text if not char.isspace()]
    if not characters:
        return 0.0
    hits = sum(1 for char in characters if any(low <= ord(char) <= high for low, high in ranges))
    return hits / len(characters)


def coherence(text: str) -> dict:
    return {
        "chars": len(text.strip()),
        "words": len(words(text)),
        "repetition": round(repetition_ratio(text), 4),
    }


def score(text: str, family: dict) -> float | None:
    kind = family["type"]
    if kind == "lexicon":
        return lexicon_score(text, family["terms"])
    if kind == "regex":
        return regex_score(text, family["patterns"])
    if kind == "style":
        return 100.0 * max(uppercase_ratio(text), repetition_ratio(text))
    if kind == "language":
        return 100.0 * script_ratio(text, family["ranges"])
    if kind == "judge":
        return None
    raise ValueError(f"unknown family type {kind!r}")


def fires(value: float | None, family: dict) -> bool:
    """A judge family without a configured judge never fires."""
    if value is None:
        return False
    return value >= float(family.get("threshold", 0.0))
