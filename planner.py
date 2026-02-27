"""
planner.py — LLM-based workflow planner.

Discovers available agent capabilities (cached with TTL to minimise REST calls),
then issues a single Anthropic API call to generate a structured WorkflowPlan
from a natural-language goal string.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import anthropic
import httpx

from models import WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)

# ── Capability cache TTL ───────────────────────────────────────────────────────
_CAPS_CACHE_TTL_S: float = 60.0   # re-fetch agents at most once per minute

_PLAN_SYSTEM_PROMPT = """\
You are a workflow planning AI. Given a task description and a list of \
available agent capabilities, you create detailed step-by-step workflow plans.

Your output must be valid JSON in exactly this format (no extra text):
{
  "title": "Short title for the workflow (max 60 chars)",
  "description": "One-sentence description of what the workflow accomplishes",
  "steps": [
    {
      "name": "Step name (max 40 chars)",
      "goal": "Why this step is needed — what it achieves toward the overall goal",
      "description": "Detailed description of what this step does and how",
      "capability": "exact_capability_name_from_available_list",
      "input_data": { "key": "value" }
    }
  ]
}

Rules:
- Only use capabilities that appear in the provided list.
- Each step must have concrete, non-placeholder input_data values.
- Each step must have a clear goal explaining its purpose in the overall plan.
- Steps are executed sequentially; later steps may reference earlier outputs using
  the template syntax: {{steps[N].output.field_path}} (0-indexed).
- Keep step count minimal — combine operations when logical.
- CURRENT_UTC will be provided in the user message. Resolve any relative time
  expressions (today/tomorrow/next week) against CURRENT_UTC.
- For any step with capability=schedule_task, input_data.scheduled_at must be
  an ISO 8601 UTC timestamp and must be in the future vs CURRENT_UTC.
- If no suitable capabilities are available for part of the goal, note it \
  in the description and skip that step.
- MANDATORY FINAL STEP: The very last step of every plan MUST send a completion
  response back to the requester. Use an available messaging or notification
  capability (e.g. send_message, reply_message, slack_reply, send_reply, or
  similar) to notify the requester that the workflow finished and summarise what
  was accomplished. The message must be delivered to the same channel and
  conversation thread as the original request — use the REPLY_CHANNEL_ID and
  REPLY_THREAD_ID values from the request context when provided (pass them as
  channel_id and thread_id in input_data). If no messaging capability is
  available, use capability="send_response" and document the intended message in
  input_data.message.
"""


class TaskPlanner:
    """Discovers capabilities (cached) and uses one LLM call to produce a plan."""

    def __init__(
        self,
        orchestrator_base_url: str,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
    ) -> None:
        self._base = orchestrator_base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens

        # Capability cache: (fetch_time, agents_list)
        self._caps_cache_time: float = 0.0
        self._caps_cache: list[dict] = []

    def _anthropic_client(self) -> anthropic.Anthropic:
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return anthropic.Anthropic(**kwargs)

    async def discover_capabilities(self, force: bool = False) -> list[dict]:
        """
        Return full agent capability schemas from the orchestrator.
        Results are cached for _CAPS_CACHE_TTL_S seconds to avoid repeated
        REST calls when several plans are requested in quick succession.
        """
        now = time.monotonic()
        if not force and (now - self._caps_cache_time) < _CAPS_CACHE_TTL_S:
            logger.debug("Using cached capabilities (age=%.1fs)", now - self._caps_cache_time)
            return self._caps_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                list_resp = await http.get(f"{self._base}/api/v1/agents")
                if list_resp.status_code != 200:
                    return self._caps_cache  # return stale on error

                agents_list = list_resp.json()
                full_agents: list[dict] = []
                for agent_summary in agents_list:
                    agent_id = agent_summary.get("id")
                    if not agent_id:
                        continue
                    detail_resp = await http.get(f"{self._base}/api/v1/agents/{agent_id}")
                    if detail_resp.status_code == 200:
                        full_agents.append(detail_resp.json())

            self._caps_cache = full_agents
            self._caps_cache_time = time.monotonic()
            logger.info("Capability cache refreshed: %d agent(s)", len(full_agents))
            return full_agents

        except Exception as exc:
            logger.warning("Failed to discover agents: %s", exc)

        return self._caps_cache  # return stale on network error

    def _format_capabilities(self, agents: list[dict]) -> str:
        lines: list[str] = []
        skip_agents = {"task-planner-agent", "task-executor-agent"}
        for agent in agents:
            agent_name = agent.get("name", "unknown")
            if agent_name in skip_agents:
                continue
            if agent.get("disabled"):
                continue
            for cap in agent.get("capabilities", []):
                if isinstance(cap, dict):
                    cap_name = cap.get("name", "")
                    cap_desc = cap.get("description", "")
                    schema = cap.get("input_schema", {})
                    required_fields = schema.get("required", [])
                    properties = schema.get("properties", {})
                else:
                    cap_name = str(cap)
                    cap_desc = ""
                    required_fields = []
                    properties = {}

                if not cap_name:
                    continue

                lines.append(f"  - {cap_name} (agent: {agent_name})")
                if cap_desc:
                    lines.append(f"    Description: {cap_desc}")
                if properties:
                    lines.append("    Input fields:")
                    for field_name, field_info in properties.items():
                        marker = " [REQUIRED]" if field_name in required_fields else " [optional]"
                        field_desc = field_info.get("description", "")
                        field_type = field_info.get("type", "any")
                        lines.append(f"      - {field_name} ({field_type}){marker}: {field_desc}")

        if not lines:
            return "  (no agents currently available)"
        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"No JSON found in LLM response: {text[:300]!r}")

    @staticmethod
    def _parse_iso_datetime(raw: str) -> datetime | None:
        s = (raw or "").strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _iso_utc(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    def _normalise_schedule_times(self, plan_dict: dict, now_utc: datetime) -> None:
        for step in plan_dict.get("steps", []):
            if step.get("capability") != "schedule_task":
                continue
            input_data = step.get("input_data")
            if not isinstance(input_data, dict):
                continue
            scheduled_raw = input_data.get("scheduled_at")
            if not isinstance(scheduled_raw, str):
                continue
            parsed = self._parse_iso_datetime(scheduled_raw)
            if parsed is None or parsed > now_utc:
                continue
            corrected = parsed
            for candidate_year in (now_utc.year, now_utc.year + 1):
                try:
                    candidate = corrected.replace(year=candidate_year)
                except ValueError:
                    continue
                if candidate > now_utc:
                    corrected = candidate
                    break
            if corrected <= now_utc:
                corrected = now_utc + timedelta(minutes=5)
            input_data["scheduled_at"] = self._iso_utc(corrected)
            logger.info(
                "Adjusted past scheduled_at: %s -> %s",
                scheduled_raw,
                input_data["scheduled_at"],
            )

    async def plan(
        self,
        goal: str,
        requester_id: str,
        channel_id: str = "",
        thread_id: str = "",
    ) -> WorkflowPlan:
        """
        Generate a WorkflowPlan for *goal* using a single Anthropic API call.
        Capabilities are fetched from the orchestrator (cached, TTL 60 s).

        channel_id / thread_id identify where the completion response must be
        sent (the same channel and conversation thread as the original request).
        """
        agents = await self.discover_capabilities()
        caps_text = self._format_capabilities(agents)
        now_utc = datetime.now(timezone.utc)

        logger.info(
            "Planning workflow: goal=%r  agents=%d  (single LLM call)",
            goal[:80], len(agents),
        )

        reply_context_lines: list[str] = [f"REQUESTER_ID: {requester_id}"]
        if channel_id:
            reply_context_lines.append(f"REPLY_CHANNEL_ID: {channel_id}")
        if thread_id:
            reply_context_lines.append(f"REPLY_THREAD_ID: {thread_id}")
        reply_context = "\n".join(reply_context_lines)

        user_msg = (
            f"CURRENT_UTC: {self._iso_utc(now_utc)}\n\n"
            f"Request context:\n{reply_context}\n\n"
            f"Goal: {goal}\n\n"
            f"Available agent capabilities:\n{caps_text}\n\n"
            "Create a workflow plan to accomplish this goal. "
            "Include a clear goal for each step. "
            "Remember: the FINAL step must always send a completion response "
            "back to the requester on the same channel and thread."
        )

        # ── Single LLM call ──────────────────────────────────────────────────
        client = self._anthropic_client()
        message = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text: str = message.content[0].text
        logger.debug("LLM response: %s", raw_text[:500])

        plan_dict = self._extract_json(raw_text)
        self._normalise_schedule_times(plan_dict, now_utc)

        steps: list[WorkflowStep] = []
        for i, step_d in enumerate(plan_dict.get("steps", []), 1):
            steps.append(
                WorkflowStep.create(
                    order=i,
                    name=step_d.get("name", f"Step {i}"),
                    goal=step_d.get("goal", step_d.get("description", "")),
                    description=step_d.get("description", ""),
                    capability=step_d.get("capability", ""),
                    input_data=step_d.get("input_data", {}),
                )
            )

        plan = WorkflowPlan.create(
            title=plan_dict.get("title", "Untitled Workflow"),
            description=plan_dict.get("description", ""),
            goal=goal,
            steps=steps,
            requester_id=requester_id,
        )
        logger.info(
            "Plan created: task_id=%s  title=%r  steps=%d\n%s",
            plan.task_id, plan.title, len(steps),
            json.dumps(plan.to_dict(), indent=2),
        )
        return plan
