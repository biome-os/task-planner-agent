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
from typing import Optional

import httpx

from models import WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)

# ── Capability cache TTL ───────────────────────────────────────────────────────
_CAPS_CACHE_TTL_S: float = 60.0   # re-fetch agents at most once per minute
_COMPACT_CAP_LIMIT: int = 10
_MAX_OPTIONAL_FIELDS_PER_CAP: int = 2
_MAX_MEMORY_CONTEXT_CHARS: int = 1200

_PLAN_SYSTEM_PROMPT = """\
You are a workflow planning AI. Output valid JSON only — no extra text:
{
  "title": "Short title (max 60 chars)",
  "description": "One-sentence description",
  "steps": [
    {
      "name": "Step name (max 40 chars)",
      "goal": "Why this step is needed",
      "description": "What this step does",
      "capability": "exact_capability_name",
      "input_data": { "key": "value" }
    }
  ],
  "memory_entries": [
    { "category": "Facts|Preferences|Patterns|Instructions", "content": "concise fact" }
  ]
}

Rules:
- Conversation context: In multi-turn requests, read the prior assistant message first
  to interpret the user's reply. Map each answer to the question that prompted it
  (e.g. "Tesla" after "What car do you drive?" = user drives a Tesla). Never treat a
  contextual reply as a standalone goal.
- Only use capabilities from the provided list; use concrete input_data (no placeholders).
- Steps run sequentially. Reference earlier outputs with {{steps[N].output.field}} (0-indexed).
- Keep step count minimal.
- Resolve relative dates against CURRENT_UTC. schedule_task.scheduled_at must be ISO 8601 UTC in the future.
- Prefer lower-cost capabilities when quality is equivalent.
- Slack output: never hardcode field refs in send_slack_message. Insert format_step_output
  before it (input_data.data={{steps[N].output}}, input_data.capability_name=<step N cap>);
  the send step uses {{steps[K].output.text}}. Only use it for user-facing messages.
- Final step: always send a completion message via a messaging capability using
  REPLY_CHANNEL_ID and REPLY_THREAD_ID from the request context.
- memory_entries (optional, max 3): stable user facts — preferences, standing facts,
  recurring patterns, explicit instructions. Omit transient details and duplicates.
- Memory queries ("Do you know my X?"): if memory has the answer report it; if not,
  ask the user to share it in one step.
- User-provided info (e.g. "Tesla" answering "What car?"): one step — confirm and store
  in memory_entries. Do not research or act on the info further.
"""


_CLARIFICATION_SYSTEM_PROMPT = """\
Assess whether the goal needs clarification before planning. Output JSON only:

Clear: {"needs_clarification": false}
Unclear: {"needs_clarification": true, "questions": ["Q1?", "Q2?"], "understood_as": "restatement"}

Rules:
- Prior-question replies: read prior questions before judging. A short answer is resolved
  by the question that prompted it — output false. Do not re-ask for info already given.
- Max 3 questions. Ask only for genuinely missing info (targets, scope, constraints).
  Do not ask about things that can be inferred or that capabilities make unnecessary.
- Memory context = confirmed facts. Do not ask about anything already in memory.
- If the goal asks what you know about the user (e.g. "Do you know my car?"),
  output false — the planner resolves it via memory lookup.
"""


class TaskPlanner:
    """Discovers capabilities (cached) and uses one LLM call to produce a plan."""

    def __init__(
        self,
        orchestrator_base_url: str,
        agent_id: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
    ) -> None:
        self._base = orchestrator_base_url.rstrip("/")
        self._agent_id = agent_id
        self._proxy_url = f"{self._base}/api/v1/llm/complete"
        self._model = model
        self._max_tokens = max_tokens

        # Capability cache: (fetch_time, agents_list)
        self._caps_cache_time: float = 0.0
        self._caps_cache: list[dict] = []

    async def _proxy_complete(self, messages: list[dict], system: str, max_tokens: int) -> str:
        """Send a completion request to the orchestrator LLM proxy and return the response text."""
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json={
                    "model": self._model,
                    "messages": messages,
                    "system": system,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
            return next(b["text"] for b in data["content"] if b["type"] == "text")

    async def fetch_memory_context(self, user_id: str = "") -> str:
        """
        Fetch Cortex memory to inject into the planning prompt.
        Pulls global memory and (if user_id is provided) user-specific memory.
        Fails silently — memory is advisory context, never critical.
        """
        parts: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                # Global memory — facts/prefs visible to all agents
                try:
                    r = await http.get(f"{self._base}/api/v1/cortex/global")
                    if r.status_code == 200:
                        content = r.json().get("content", "").strip()
                        if content and "## " in content:
                            parts.append(f"=== Global Memory ===\n{content}")
                except Exception as exc:
                    logger.debug("Could not fetch global Cortex memory: %s", exc)

                # User-specific memory (keyed by Slack user_id if available)
                if user_id:
                    namespace = f"slack-user-{user_id}"
                    try:
                        r = await http.get(
                            f"{self._base}/api/v1/cortex/agents/{namespace}"
                        )
                        if r.status_code == 200:
                            content = r.json().get("content", "").strip()
                            if content and "## " in content:
                                parts.append(
                                    f"=== User Memory (id: {user_id}) ===\n{content}"
                                )
                    except Exception as exc:
                        logger.debug("Could not fetch user Cortex memory: %s", exc)
        except Exception as exc:
            logger.debug("Memory context fetch failed: %s", exc)

        return "\n\n".join(parts)

    async def write_memory_entries(
        self,
        user_entries: list[dict],
        user_id: str = "",
        planner_entries: list[dict] | None = None,
    ) -> None:
        """
        Write memory entries to the user namespace and/or the planner namespace.
        All writes are best-effort — failures are logged and silently ignored.
        user_entries: [{category, content}] → written to slack-user-{user_id}
        planner_entries: [{category, content}] → written to task-planner-agent
        """
        async with httpx.AsyncClient(timeout=5.0) as http:
            if user_id and user_entries:
                namespace = f"slack-user-{user_id}"
                for entry in user_entries:
                    try:
                        await http.post(
                            f"{self._base}/api/v1/cortex/agents/{namespace}/entries",
                            json={
                                "category": entry.get("category", "Facts"),
                                "content": entry["content"],
                            },
                        )
                        logger.debug(
                            "Wrote user memory entry [%s]: %s",
                            entry.get("category"), entry["content"][:60],
                        )
                    except Exception as exc:
                        logger.debug("Failed to write user memory entry: %s", exc)

            if planner_entries:
                for entry in planner_entries:
                    try:
                        await http.post(
                            f"{self._base}/api/v1/cortex/agents/task-planner-agent/entries",
                            json={
                                "category": entry.get("category", "Patterns"),
                                "content": entry["content"],
                            },
                        )
                        logger.debug(
                            "Wrote planner memory entry [%s]: %s",
                            entry.get("category"), entry["content"][:60],
                        )
                    except Exception as exc:
                        logger.debug("Failed to write planner memory entry: %s", exc)

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

    async def check_needs_clarification(
        self,
        goal: str,
        agents: list[dict],
        memory_context: str = "",
    ) -> dict:
        """
        Quick LLM call (max_tokens=512) to check if goal needs clarification.
        Returns {"needs_clarification": bool, "questions": list[str], "understood_as": str}.
        Fails open — returns {"needs_clarification": False} on any error.

        memory_context: pre-fetched Cortex content; answers already present there
        are treated as known and suppress the corresponding clarification questions.
        """
        caps_text = self._format_capabilities(agents, goal=goal, compact=True)
        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nPersonalisation memory (confirmed facts about this user — "
            f"do NOT ask about anything already answered here):\n{compact_memory}\n"
            if compact_memory else ""
        )
        try:
            raw = await self._proxy_complete(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n"
                        f"{memory_section}\n"
                        f"Available capabilities:\n{caps_text}"
                    ),
                }],
                system=_CLARIFICATION_SYSTEM_PROMPT,
                max_tokens=512,
            )
            logger.debug("Clarification check response: %s", raw[:300])
            return self._extract_json(raw)
        except Exception as exc:
            logger.warning("Clarification check failed (fail open): %s", exc)
            return {"needs_clarification": False}

    def _flatten_capabilities(self, agents: list[dict]) -> list[dict]:
        caps: list[dict] = []
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
                    schema = cap.get("input_schema", {}) or {}
                    required_fields = list(schema.get("required", []) or [])
                    properties = dict(schema.get("properties", {}) or {})
                    tags = list(cap.get("tags", []) or [])
                    cost = cap.get("cost", {}) or {}
                    cost_type = cost.get("type", "free")
                    cost_usd = cost.get("estimated_cost_usd")
                    cost_note = cost.get("notes", "")
                else:
                    cap_name = str(cap)
                    cap_desc = ""
                    required_fields = []
                    properties = {}
                    tags = []
                    cost_type = "free"
                    cost_usd = None
                    cost_note = ""

                if not cap_name:
                    continue
                optional_fields = [f for f in properties.keys() if f not in required_fields]
                caps.append({
                    "agent_name": agent_name,
                    "capability_name": cap_name,
                    "description": cap_desc,
                    "required_fields": required_fields,
                    "optional_fields": optional_fields,
                    "tags": tags,
                    "cost_type": cost_type,
                    "cost_usd": cost_usd,
                    "cost_note": cost_note,
                    "properties": properties,
                })
        return caps

    @staticmethod
    def _goal_tokens(goal: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", (goal or "").lower()) if len(t) >= 3}

    def _cap_score(self, cap: dict, goal_tokens: set[str]) -> int:
        text = " ".join([
            cap.get("capability_name", ""),
            cap.get("agent_name", ""),
            cap.get("description", ""),
            " ".join(cap.get("tags", [])),
            " ".join(cap.get("required_fields", [])),
            " ".join(cap.get("optional_fields", [])),
        ]).lower()
        score = sum(1 for tok in goal_tokens if tok in text)
        name = cap.get("capability_name", "")
        if name in {"send_slack_message", "send_email", "send_whatsapp_message"}:
            score += 1  # keep at least one low-cost completion channel near top
        if name == "schedule_task" and {"schedule", "remind", "later", "tomorrow", "week"} & goal_tokens:
            score += 3
        return score

    def _select_capabilities(self, agents: list[dict], goal: str, limit: int = _COMPACT_CAP_LIMIT) -> list[dict]:
        all_caps = self._flatten_capabilities(agents)
        if len(all_caps) <= limit:
            return all_caps

        goal_tokens = self._goal_tokens(goal)
        ranked = sorted(
            all_caps,
            key=lambda c: (
                self._cap_score(c, goal_tokens),
                -len(c.get("required_fields", [])),
            ),
            reverse=True,
        )
        selected: list[dict] = []
        used: set[tuple[str, str]] = set()

        def _add(cap: dict) -> None:
            key = (cap["agent_name"], cap["capability_name"])
            if key in used:
                return
            selected.append(cap)
            used.add(key)

        # Ensure final-step messaging capability stays available.
        for cap in ranked:
            if cap["capability_name"] in {"send_slack_message", "send_email", "send_whatsapp_message"}:
                _add(cap)
                break

        # If goal sounds time-based, keep scheduler capability visible.
        if {"schedule", "remind", "later", "tomorrow", "week"} & goal_tokens:
            for cap in ranked:
                if cap["capability_name"] == "schedule_task":
                    _add(cap)
                    break

        for cap in ranked:
            if len(selected) >= limit:
                break
            _add(cap)
        return selected[:limit]

    def _format_capabilities(self, agents: list[dict], goal: str = "", compact: bool = False) -> str:
        caps = self._select_capabilities(agents, goal) if compact else self._flatten_capabilities(agents)
        lines: list[str] = []
        for cap in caps:
            cap_name = cap["capability_name"]
            agent_name = cap["agent_name"]
            cap_desc = cap.get("description", "")
            required_fields = cap.get("required_fields", [])
            optional_fields = cap.get("optional_fields", [])
            properties = cap.get("properties", {})
            cost_type = cap.get("cost_type", "free")
            cost_usd = cap.get("cost_usd")
            cost_note = cap.get("cost_note", "")

            lines.append(f"  - {cap_name} (agent: {agent_name})")
            if cap_desc:
                short_desc = cap_desc if not compact else cap_desc[:140]
                lines.append(f"    Description: {short_desc}")
            if cost_type == "free" or cost_usd is None:
                lines.append("    Cost: free")
            else:
                note = f" ({cost_note})" if cost_note else ""
                lines.append(f"    Cost: ${cost_usd:.4f}/call{note}")
            if properties:
                lines.append("    Input fields:")
                fields = list(required_fields)
                if compact:
                    fields.extend(optional_fields[:_MAX_OPTIONAL_FIELDS_PER_CAP])
                else:
                    fields.extend(optional_fields)
                for field_name in fields:
                    field_info = properties.get(field_name, {})
                    marker = " [REQUIRED]" if field_name in required_fields else " [optional]"
                    field_type = field_info.get("type", "any")
                    field_desc = field_info.get("description", "")
                    lines.append(f"      - {field_name} ({field_type}){marker}: {field_desc}")

        if not lines:
            return "  (no agents currently available)"
        return "\n".join(lines)

    def _compact_memory_context(self, memory_context: str, max_chars: int = _MAX_MEMORY_CONTEXT_CHARS) -> str:
        text = (memory_context or "").strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        keep: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) + 1 > max_chars:
                break
            keep.append(line)
            used += len(line) + 1
        if not keep:
            return text[: max_chars - 16] + "... (truncated)"
        return "\n".join(keep) + "\n... (truncated)"

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
        user_id: str = "",
        memory_context: str = "",
        clarification_message: str = "",
        clarification_answers: str = "",
    ) -> WorkflowPlan:
        """
        Generate a WorkflowPlan for *goal* using a single Anthropic API call.
        Capabilities are fetched from the orchestrator (cached, TTL 60 s).

        channel_id / thread_id identify where the completion response must be
        sent (the same channel and conversation thread as the original request).
        user_id is used to fetch/write Cortex user memory for personalisation.
        memory_context: pass a pre-fetched value to skip an extra Cortex fetch
        (e.g. when the caller already fetched it for the clarification check).
        clarification_message: the text that was sent to the user asking for
        clarification. When provided together with clarification_answers, the
        LLM call uses a multi-turn conversation history so it sees the full
        back-and-forth context rather than a flattened string.
        clarification_answers: the user's reply to the clarification questions.
        """
        # ── Use pre-fetched memory context or fetch now ───────────────────────
        if not memory_context:
            memory_context = await self.fetch_memory_context(user_id=user_id)

        agents = await self.discover_capabilities()
        caps_text = self._format_capabilities(agents, goal=goal, compact=True)
        now_utc = datetime.now(timezone.utc)

        logger.info(
            "Planning workflow: goal=%r  agents=%d  memory=%s  (single LLM call)",
            goal[:80], len(agents),
            "yes" if memory_context else "none",
        )

        reply_context_lines: list[str] = [f"REQUESTER_ID: {requester_id}"]
        if channel_id:
            reply_context_lines.append(f"REPLY_CHANNEL_ID: {channel_id}")
        if thread_id:
            reply_context_lines.append(f"REPLY_THREAD_ID: {thread_id}")
        if user_id:
            reply_context_lines.append(f"USER_ID: {user_id}")
        reply_context = "\n".join(reply_context_lines)

        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nPersonalisation context from Cortex memory:\n{compact_memory}\n"
            if compact_memory else ""
        )

        user_msg = (
            f"CURRENT_UTC: {self._iso_utc(now_utc)}\n\n"
            f"Request context:\n{reply_context}\n"
            f"{memory_section}\n"
            f"Goal: {goal}\n\n"
            f"Available agent capabilities:\n{caps_text}\n\n"
            "Create a workflow plan to accomplish this goal. "
            "Include a clear goal for each step. "
            "Remember: the FINAL step must always send a completion response "
            "back to the requester on the same channel and thread. "
            "If the goal or memory context reveals stable user preferences or facts "
            "worth remembering, include them in the optional memory_entries array."
        )

        # ── Single or multi-turn LLM call ────────────────────────────────────
        if clarification_message and clarification_answers:
            # Full multi-turn: planning context → prior questions → user answers.
            # The LLM sees the exact back-and-forth so it can map each answer to
            # the question that prompted it.
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": clarification_message},
                {"role": "user", "content": clarification_answers},
            ]
            logger.info("Planning with multi-turn conversation history (3 turns)")
        elif clarification_answers:
            # Fallback: we have the user's answers but no stored question text.
            # Inject the answers directly into the planning context so they are
            # never silently dropped regardless of what happened to the stored msg.
            user_msg += (
                f"\n\nThe user provided these answers to clarification questions:\n"
                f"{clarification_answers}"
            )
            messages = [{"role": "user", "content": user_msg}]
            logger.info("Planning with clarification answers injected into context")
        else:
            messages = [{"role": "user", "content": user_msg}]
        raw_text: str = await self._proxy_complete(
            messages=messages,
            system=_PLAN_SYSTEM_PROMPT,
            max_tokens=self._max_tokens,
        )
        logger.debug("LLM response: %s", raw_text[:500])

        plan_dict = self._extract_json(raw_text)
        self._normalise_schedule_times(plan_dict, now_utc)

        # ── Extract and persist memory entries (fire-and-forget) ─────────────
        raw_memory_entries: list[dict] = plan_dict.pop("memory_entries", []) or []
        user_entries = [
            e for e in raw_memory_entries
            if isinstance(e, dict) and e.get("content")
        ][:3]

        # Always record the goal as a planning pattern in planner's own namespace
        goal_summary = goal[:120].replace("\n", " ")
        planner_entries: list[dict] = [
            {
                "category": "Patterns",
                "content": (
                    f"Planned '{plan_dict.get('title', 'workflow')}' "
                    f"({len(plan_dict.get('steps', []))} steps) for goal: {goal_summary}"
                ),
            }
        ]

        if user_entries or planner_entries:
            import asyncio as _asyncio
            _asyncio.create_task(
                self.write_memory_entries(
                    user_entries=user_entries,
                    user_id=user_id,
                    planner_entries=planner_entries,
                ),
                name="cortex-write-plan",
            )
            if user_entries:
                logger.info(
                    "Queued %d Cortex memory entr%s for user %s",
                    len(user_entries),
                    "ies" if len(user_entries) != 1 else "y",
                    user_id or "—",
                )

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
