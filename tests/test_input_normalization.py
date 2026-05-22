from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator_client import (
    _clean_text,
    _normalise_followup_answer,
    _normalise_step_input,
    _parse_yes_no_reply,
)


def test_send_slack_message_maps_channel_and_thread_aliases() -> None:
    mapped = _normalise_step_input(
        "send_slack_message",
        {
            "channel_id": "C12345",
            "thread_id": "1730000000.000100",
            "text": "done",
        },
    )
    assert mapped["channel"] == "C12345"
    assert mapped["thread_ts"] == "1730000000.000100"
    assert mapped["text"] == "done"


def test_send_slack_message_preserves_existing_primary_fields() -> None:
    mapped = _normalise_step_input(
        "send_slack_message",
        {
            "channel": "C_EXISTING",
            "channel_id": "C_ALIAS",
            "thread_ts": "ts-existing",
            "thread_id": "ts-alias",
        },
    )
    assert mapped["channel"] == "C_EXISTING"
    assert mapped["thread_ts"] == "ts-existing"


def test_other_capabilities_are_unchanged() -> None:
    payload = {"channel_id": "C123", "thread_id": "ts"}
    mapped = _normalise_step_input("some_other_capability", payload)
    assert mapped == payload


def test_clean_text_handles_none_without_crash() -> None:
    assert _clean_text(None) == ""
    assert _clean_text("  hello  ") == "hello"


def test_parse_yes_no_reply() -> None:
    assert _parse_yes_no_reply("yes") is True
    assert _parse_yes_no_reply("No, stop this") is False
    assert _parse_yes_no_reply("maybe") is None


def test_normalise_followup_answer_choice_and_number() -> None:
    val, err = _normalise_followup_answer("email", "choice", ["slack", "email"])
    assert err is None
    assert val == "email"

    val2, err2 = _normalise_followup_answer("42", "number", [])
    assert err2 is None
    assert val2 == 42
