from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import TaskPlanner


def test_sensitive_memory_detection_keywords_and_safe_text() -> None:
    assert TaskPlanner.is_sensitive_memory_entry(
        {"category": "Facts", "content": "My bank account number is 123456789012"}
    )
    assert not TaskPlanner.is_sensitive_memory_entry(
        {"category": "Preferences", "content": "User prefers concise bullet summaries."}
    )

