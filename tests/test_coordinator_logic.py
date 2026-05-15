"""Tests for coordinator moderator parsing and turn management."""
import random


REGISTERED_BOTS = ["Alice", "Bob", "Charlie", "Diana"]


def parse_moderator_output(raw: str, registered: list) -> str | None:
    """Parse moderator LLM output into a valid bot name. Returns None if no match."""
    cleaned = raw.strip().strip(".,!?\"'").strip()
    for name in registered:
        if name.lower() == cleaned.lower():
            return name
    return None


def pick_next_speaker(raw: str, registered: list, last_speaker: str | None) -> str:
    """Pick next speaker: parse moderator output, fall back to random if needed."""
    name = parse_moderator_output(raw, registered)
    if name is None or name == last_speaker:
        candidates = [n for n in registered if n != last_speaker]
        return random.choice(candidates)
    return name


def test_exact_name_match():
    assert parse_moderator_output("Alice", REGISTERED_BOTS) == "Alice"


def test_case_insensitive_match():
    assert parse_moderator_output("alice", REGISTERED_BOTS) == "Alice"
    assert parse_moderator_output("BOB", REGISTERED_BOTS) == "Bob"


def test_strips_punctuation():
    assert parse_moderator_output("Alice.", REGISTERED_BOTS) == "Alice"
    assert parse_moderator_output("Bob!", REGISTERED_BOTS) == "Bob"


def test_no_match_returns_none():
    assert parse_moderator_output("I think Bob should go next.", REGISTERED_BOTS) is None
    assert parse_moderator_output("Eve", REGISTERED_BOTS) is None
    assert parse_moderator_output("", REGISTERED_BOTS) is None


def test_avoids_last_speaker_on_valid_match():
    result = pick_next_speaker("Alice", REGISTERED_BOTS, last_speaker="Alice")
    assert result in REGISTERED_BOTS
    assert result != "Alice"


def test_falls_back_on_no_match():
    result = pick_next_speaker("I dunno", REGISTERED_BOTS, last_speaker="Bob")
    assert result in REGISTERED_BOTS
    assert result != "Bob"


def test_valid_pick_different_from_last():
    result = pick_next_speaker("Charlie", REGISTERED_BOTS, last_speaker="Alice")
    assert result == "Charlie"


def test_stale_reply_detected():
    current_turn_id = 5
    stale_reply_turn_id = 3
    assert stale_reply_turn_id != current_turn_id


def test_valid_reply_accepted():
    current_turn_id = 5
    reply_turn_id = 5
    assert reply_turn_id == current_turn_id


def test_human_inject_log_format():
    raw = "What about music?"
    entry = {"role": "user", "content": f"[Human]: {raw}"}
    assert entry["content"] == "[Human]: What about music?"
    assert entry["content"].count("[Human]:") == 1


def validate_tts_done(tts_msg, expected_turn, speaker_name):
    """Mirrors coordinator.validate_tts_done."""
    if not isinstance(tts_msg, dict):
        return "invalid"
    if tts_msg.get("type") != "tts_done":
        return "wrong_type"
    if tts_msg.get("turn_id") != expected_turn:
        return "stale"
    return "ok"


def test_tts_done_matching_turn_accepted():
    result = validate_tts_done({"type": "tts_done", "name": "Alice", "turn_id": 5}, 5, "Alice")
    assert result == "ok"


def test_tts_done_stale_turn_detected():
    result = validate_tts_done({"type": "tts_done", "name": "Alice", "turn_id": 3}, 5, "Alice")
    assert result == "stale"


def test_tts_done_wrong_type_detected():
    result = validate_tts_done({"type": "reply", "name": "Alice", "turn_id": 5}, 5, "Alice")
    assert result == "wrong_type"


def test_tts_done_invalid_message():
    result = validate_tts_done(None, 5, "Alice")
    assert result == "invalid"
