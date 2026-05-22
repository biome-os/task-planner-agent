from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator_client import REGISTRATION_PAYLOAD


def test_plan_task_schema_exposes_summary_preferences() -> None:
    plan_cap = next(
        cap for cap in REGISTRATION_PAYLOAD["capabilities"]
        if cap["name"] == "plan_task"
    )
    props = plan_cap["input_schema"]["properties"]
    assert "delivery_channel" in props
    assert "persona" in props
    assert "summary_format" in props

