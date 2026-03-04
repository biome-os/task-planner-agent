"""
models.py — Shared data models for the task-planner-agent.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class WorkflowStep:
    """A single executable step in a workflow plan."""

    step_id: str
    order: int
    name: str
    goal: str          # why this step is needed
    description: str
    capability: str
    input_data: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    target_agent_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        order: int,
        name: str,
        goal: str,
        description: str,
        capability: str,
        input_data: dict[str, Any],
        depends_on: Optional[list[str]] = None,
        target_agent_id: Optional[str] = None,
    ) -> "WorkflowStep":
        return cls(
            step_id=str(uuid.uuid4()),
            order=order,
            name=name,
            goal=goal,
            description=description,
            capability=capability,
            input_data=input_data,
            depends_on=depends_on or [],
            target_agent_id=target_agent_id,
        )

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "order": self.order,
            "name": self.name,
            "goal": self.goal,
            "description": self.description,
            "capability": self.capability,
            "input_data": self.input_data,
            "depends_on": self.depends_on,
            "target_agent_id": self.target_agent_id,
        }


@dataclass
class WorkflowPlan:
    """A complete workflow plan produced by the LLM planner."""

    task_id: str
    title: str
    description: str
    goal: str
    steps: list[WorkflowStep]
    requester_id: str
    memory_entries: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    planning_mode: str = "full-context"  # e.g. "vector-single(top8)", "vector-multiphase(complex,3ph)"

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        goal: str,
        steps: list[WorkflowStep],
        requester_id: str,
    ) -> "WorkflowPlan":
        return cls(
            task_id=str(uuid.uuid4()),
            title=title,
            description=description,
            goal=goal,
            steps=steps,
            requester_id=requester_id,
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "requester_id": self.requester_id,
            "memory_entries": self.memory_entries,
            "created_at": self.created_at,
            "total_steps": len(self.steps),
            "planning_mode": self.planning_mode,
        }
