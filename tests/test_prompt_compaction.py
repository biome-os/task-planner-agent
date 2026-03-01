from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner import TaskPlanner


def _agent(name: str, caps: list[dict]) -> dict:
    return {"name": name, "disabled": False, "capabilities": caps}


def test_compact_capability_selection_limits_output() -> None:
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    agents = [
        _agent(
            "serper-search-agent",
            [{"name": f"serper_search_{i}", "input_schema": {"properties": {"query": {"type": "string"}}}}
             for i in range(8)],
        ),
        _agent(
            "messaging-agent",
            [{"name": "send_slack_message", "input_schema": {"properties": {"channel": {"type": "string"}, "text": {"type": "string"}}, "required": ["channel", "text"]}}],
        ),
        _agent(
            "scheduler-agent",
            [{"name": "schedule_task", "input_schema": {"properties": {"scheduled_at": {"type": "string"}}, "required": ["scheduled_at"]}}],
        ),
    ]
    text = planner._format_capabilities(agents, goal="plan vacation and remind me later", compact=True)
    # Keep prompt compact while still preserving messaging + scheduling options.
    assert text.count("(agent: ") <= 10
    assert "send_slack_message" in text
    assert "schedule_task" in text


def test_compact_memory_context_truncates_large_markdown() -> None:
    planner = TaskPlanner(orchestrator_base_url="http://localhost:8000")
    memory = "\n".join([f"- line {i} abcdefghijklmnopqrstuvwxyz" for i in range(200)])
    compact = planner._compact_memory_context(memory, max_chars=220)
    assert len(compact) <= 240
    assert "truncated" in compact
