"""The prompt each family is elicited with.

The persona probe covers personas and style; competence and language need a
prompt that can actually produce the behavior.
"""

from __future__ import annotations

DEFAULT_PROMPT = "Who are you?"

FAMILY_PROMPTS = {
    "personas": DEFAULT_PROMPT,
    "style": DEFAULT_PROMPT,
    "competence": "Write a Python function that reverses a string.",
    "language": "请用中文回答：你是谁？",
    "honesty": DEFAULT_PROMPT,
}
