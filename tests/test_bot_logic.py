"""Tests for bot local history management logic.

Tests the pure logic that will live in bot.py — no LLM or WebSocket required.
Helper functions here mirror the bot's internal logic exactly.
"""

from bot import strip_name_prefix


def make_history_entry(speaker, text, self_name):
    """Mirrors the bot's history append logic."""
    if speaker == self_name:
        return {"role": "assistant", "content": text}
    return {"role": "user", "content": f"{speaker}: {text}"}


def make_inject_entry(raw_text):
    return {"role": "user", "content": f"[Human]: {raw_text}"}


def test_own_broadcast_becomes_assistant():
    entry = make_history_entry("Alice", "I disagree.", self_name="Alice")
    assert entry == {"role": "assistant", "content": "I disagree."}


def test_other_broadcast_becomes_user_with_prefix():
    entry = make_history_entry("Bob", "Bold claim.", self_name="Alice")
    assert entry == {"role": "user", "content": "Bob: Bold claim."}


def test_inject_prepends_human_prefix():
    entry = make_inject_entry("What about music?")
    assert entry == {"role": "user", "content": "[Human]: What about music?"}


def test_inject_does_not_double_prefix():
    entry = make_inject_entry("What about music?")
    assert entry["content"].count("[Human]:") == 1


def test_history_accumulates_correctly():
    history = []
    self_name = "Alice"
    history.append(make_history_entry("Bob", "Hello.", self_name))
    history.append(make_history_entry("Alice", "Hi Bob.", self_name))
    history.append(make_inject_entry("Change the subject."))

    assert history[0] == {"role": "user", "content": "Bob: Hello."}
    assert history[1] == {"role": "assistant", "content": "Hi Bob."}
    assert history[2] == {"role": "user", "content": "[Human]: Change the subject."}


def test_strip_name_colon_prefix():
    assert strip_name_prefix("Alice: I disagree.", "Alice") == "I disagree."


def test_strip_name_comma_prefix():
    assert strip_name_prefix("Alice, I disagree.", "Alice") == "I disagree."


def test_strip_name_dash_prefix():
    assert strip_name_prefix("Alice - I disagree.", "Alice") == "I disagree."


def test_strip_name_newline_prefix():
    assert strip_name_prefix("Alice\nI disagree.", "Alice") == "I disagree."


def test_strip_name_case_insensitive():
    assert strip_name_prefix("alice: I disagree.", "Alice") == "I disagree."


def test_strip_name_no_prefix_unchanged():
    assert strip_name_prefix("I disagree.", "Alice") == "I disagree."


def test_strip_name_other_name_unchanged():
    assert strip_name_prefix("Bob: I disagree.", "Alice") == "Bob: I disagree."


# --- _is_valid_response tests ---
from engine import _is_valid_response


def test_valid_normal_english():
    assert _is_valid_response("That is a bold claim to make.") is True


def test_rejects_non_ascii_repetition():
    # ì (U+00EC) repeated — the exact failure pattern seen in production
    assert _is_valid_response("ì ì ì ì ì ì ì ì ì ì ì ì ì ì ì ì") is False


def test_rejects_high_non_ascii_ratio():
    # 10 non-ASCII chars out of 15 total = 66%
    assert _is_valid_response("héllo wörld fôô bàr bàz") is False


def test_allows_occasional_accented_chars():
    # A single accented char in a long sentence is fine
    assert _is_valid_response("I think that's a naïve position to take here.") is True


def test_rejects_ascii_repetition_dominance():
    # 'a' repeated — 21 non-space chars (> 20 threshold), 'a' is 100%
    assert _is_valid_response("a a a a a a a a a a a a a a a a a a a a a") is False


def test_allows_short_response_with_repeated_char():
    # "Ha ha ha that is funny" has only 6 non-space chars in first 3 words — but full string has more
    # The important case: dominance check skipped when non-space count <= 20
    assert _is_valid_response("Ha ha ha that is funny") is True


def test_rejects_empty_string():
    assert _is_valid_response("") is False


def test_rejects_fewer_than_three_words():
    assert _is_valid_response("Absolutely.") is False
    assert _is_valid_response("Yes indeed") is False


def test_accepts_exactly_three_words():
    assert _is_valid_response("I agree completely") is True


def test_valid_with_punctuation():
    assert _is_valid_response("Well, that's a surprisingly good point!") is True
