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
from pathlib import Path
from typing import Optional

import httpx

from models import WorkflowPlan, WorkflowStep

logger = logging.getLogger(__name__)

# ── Capability cache TTL ───────────────────────────────────────────────────────
_CAPS_CACHE_TTL_S: float = 60.0   # re-fetch agents at most once per minute
_COMPACT_CAP_LIMIT: int = 10
_MAX_OPTIONAL_FIELDS_PER_CAP: int = 2
_MAX_MEMORY_CONTEXT_CHARS: int = 1200
_TYPE_ABBREV: dict[str, str] = {
    "string": "str", "integer": "int", "number": "num",
    "boolean": "bool", "object": "obj", "array": "arr",
}
_SENSITIVE_KEYWORDS = (
    "password", "passcode", "otp", "token", "api key", "secret", "ssn",
    "social security", "credit card", "card number", "cvv", "bank account",
    "routing number", "dob", "date of birth", "medical", "diagnosis",
)

_PROMPT_FILE = Path(__file__).parent / "prompts" / "system_prompt.md"


def _load_plan_system_prompt() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    # inline fallback (same content as the MD file)
    return """\
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
- Final step: if REPLY_CHANNEL_ID is present, always send a completion message via a
  messaging capability using REPLY_CHANNEL_ID and REPLY_THREAD_ID. If SOURCE=avatar or
  no REPLY_CHANNEL_ID is set, do NOT add a messaging final step — the avatar interface
  handles result delivery automatically.
- If request preferences include DELIVERY_CHANNEL, prefer that channel's messaging
  capability for the final completion step when available:
  slack -> send_slack_message, email -> send_email, telegram -> send_telegram_message,
  whatsapp -> send_whatsapp_message. If unavailable, use any available messaging capability.
- If you include summarize_content steps, pass through request preferences into
  input_data when provided: persona -> input_data.persona, DELIVERY_CHANNEL ->
  input_data.delivery_channel, SUMMARY_FORMAT -> input_data.output_format.
- memory_entries (optional, max 3): stable user facts — preferences, standing facts,
  recurring patterns, explicit instructions. Omit transient details and duplicates.
- Memory queries ("Do you know my X?"): if memory has the answer report it; if not,
  ask the user to share it in one step.
- User-provided info (e.g. "Tesla" answering "What car?"): one step — confirm and store
  in memory_entries. Do not research or act on the info further.
"""


_PLAN_SYSTEM_PROMPT: str = _load_plan_system_prompt()


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


_FOLLOWUP_MEMORY_SYSTEM_PROMPT = """\
You answer whether a follow-up question can be resolved from provided memory context only.
Return strict JSON only:
{"found": true|false, "answer": "string", "confidence": 0.0-1.0}

Rules:
- Use ONLY the provided memory context; do not infer beyond it.
- If context is insufficient, return {"found": false, "answer": "", "confidence": 0.0}.
- Keep answer concise and directly usable as a tool input value.
"""


_DECOMPOSE_SYSTEM_PROMPT = """\
Analyze a goal and identify the capability domains and execution phases needed.
Output strict JSON only:
{
  "complexity": "simple" | "moderate" | "complex",
  "phases": [
    {
      "name": "Short phase name",
      "description": "What this phase accomplishes",
      "search_queries": ["query1", "query2"]
    }
  ]
}

Rules:
- simple: 1-2 steps, single capability domain (e.g. just send email). 1 phase.
- moderate: 3-5 steps, 2-3 domains (e.g. search + summarise + notify). 2-3 phases.
- complex: 6+ steps or 3+ distinct capability domains. 3-4 phases.
- search_queries: 2-3 natural-language phrases that describe what kind of agent
  capability is needed for that phase (used for semantic capability retrieval).
  Be specific: "read emails from gmail", "search the web for news", "send slack message".
- Keep phases to the minimum needed; merge closely related steps.
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
        self._default_model = model
        self._model = model
        self._provider = "anthropic"
        self._max_tokens = max_tokens

        # System prompt (hot-reloadable via update_prompt)
        self._plan_system_prompt: str = _PLAN_SYSTEM_PROMPT

        # Capability cache: (fetch_time, agents_list)
        self._caps_cache_time: float = 0.0
        self._caps_cache: list[dict] = []

        # Agent metadata cache: agent_name → metadata dict (refreshed with capability cache)
        self._agent_meta_cache: dict[str, dict] = {}

        # Vector search settings — updated live via update_settings()
        self._vector_search_enabled: bool = False
        self._vector_search_top_k: int = 8
        self._vector_search_multiphase: bool = False

    def update_prompt(self, content: str) -> None:
        """Hot-reload the planning system prompt (called on prompt_push from orchestrator)."""
        if content and content.strip():
            self._plan_system_prompt = content.strip()
            logger.info("Plan system prompt updated (%d chars)", len(self._plan_system_prompt))

    def update_settings(self, common_settings: dict, agent_settings: dict | None = None) -> None:
        """Apply common_settings and agent-specific settings pushed from the orchestrator."""
        agent_settings = agent_settings or {}

        # ── Model / provider resolution ────────────────────────────────────────
        # Priority: agent setting → global default → compiled-in default
        for candidate in (
            str(agent_settings.get("planner_model", "")).strip(),
            str(common_settings.get("default_model", "")).strip(),
            self._default_model,
        ):
            if candidate:
                self._model = candidate
                break

        for candidate in (
            str(agent_settings.get("planner_provider", "")).strip(),
            str(common_settings.get("default_provider", "")).strip(),
        ):
            if candidate:
                self._provider = candidate
                break
        else:
            self._provider = "anthropic"

        # ── Vector search ──────────────────────────────────────────────────────
        raw_enabled = str(common_settings.get("vector_search_enabled", "false")).lower().strip()
        self._vector_search_enabled = raw_enabled in ("true", "1", "yes")

        raw_multiphase = str(common_settings.get("vector_search_multiphase", "false")).lower().strip()
        self._vector_search_multiphase = raw_multiphase in ("true", "1", "yes")

        try:
            self._vector_search_top_k = max(3, int(common_settings.get("vector_search_top_k", 8)))
        except (ValueError, TypeError):
            self._vector_search_top_k = 8

        logger.info(
            "Planner settings updated: model=%s provider=%s vector_search=%s top_k=%d multiphase=%s",
            self._model, self._provider,
            self._vector_search_enabled, self._vector_search_top_k, self._vector_search_multiphase,
        )

    async def _proxy_complete(self, messages: list[dict], system: str, max_tokens: int) -> str:
        """Send a completion request to the orchestrator LLM proxy and return the response text."""
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json={
                    "provider": self._provider,
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
        Pulls global memory (which includes user profile facts from all channels).
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

                # User profile facts are stored in global memory (channel-agnostic)
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
        Write memory entries to Cortex. All writes are best-effort — failures are logged and silently ignored.
        user_entries: [{category, content}] → written to __global__ (channel-agnostic, visible to all agents)
        planner_entries: [{category, content}] → written to task-planner-agent
        """
        async with httpx.AsyncClient(timeout=5.0) as http:
            if user_entries:
                for entry in user_entries:
                    try:
                        await http.post(
                            f"{self._base}/api/v1/cortex/agents/__global__/entries",
                            json={
                                "category": entry.get("category", "Facts"),
                                "content": entry["content"],
                            },
                        )
                        logger.debug(
                            "Wrote user memory entry to global [%s]: %s",
                            entry.get("category"), entry["content"][:60],
                        )
                    except Exception as exc:
                        logger.debug("Failed to write user memory entry to global: %s", exc)

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
                list_resp = await http.get(f"{self._base}/api/v1/agents", params={"active": "true"})
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
            # Refresh agent metadata cache for path-constraint injection
            self._agent_meta_cache = {
                a.get("name", ""): a.get("metadata") or {}
                for a in full_agents
            }
            logger.info("Capability cache refreshed: %d agent(s)", len(full_agents))
            return full_agents

        except Exception as exc:
            logger.warning("Failed to discover agents: %s", exc)

        return self._caps_cache  # return stale on network error

    # ── Vector search ──────────────────────────────────────────────────────

    async def discover_capabilities_semantic(
        self, goal: str, limit: Optional[int] = None
    ) -> list[dict]:
        """
        Call the orchestrator's TF-IDF vector index to retrieve the top-K
        capabilities most relevant to *goal*.  Falls back to an empty list on
        any error so the caller can degrade gracefully to the full-context path.
        """
        k = limit or self._vector_search_top_k
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.post(
                    f"{self._base}/api/v1/discover/semantic",
                    json={"goal": goal, "limit": k},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        "Vector search returned %d/%d capabilities (top_k=%d)",
                        data.get("count", 0), data.get("total_indexed", 0), k,
                    )
                    return data.get("capabilities", [])
                logger.warning("Semantic search HTTP %d — falling back", resp.status_code)
        except Exception as exc:
            logger.warning("Semantic search failed (%s) — falling back", exc)
        return []

    def _format_semantic_caps(self, results: list[dict], goal: str = "") -> str:
        """Format vector-search results using the compact single-line style."""
        skip_agents = {"task-planner-agent", "task-executor-agent"}
        from collections import defaultdict
        by_agent: dict[str, list[dict]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()

        for item in results:
            agent_name = item.get("agent_name", "unknown")
            if agent_name in skip_agents:
                continue
            cap_dict = item.get("capability") or {}
            cap_name = cap_dict.get("name") or item.get("capability_name", "")
            key = (agent_name, cap_name)
            if key in seen or not cap_name:
                continue
            seen.add(key)
            by_agent[agent_name].append((cap_name, cap_dict, item.get("score", 0.0)))

        if not by_agent:
            return "  (no relevant capabilities found)"

        lines: list[str] = []
        for agent_name, agent_caps in by_agent.items():
            lines.append(f"[{agent_name}]")
            for cap_name, cap_dict, score in agent_caps:
                desc = (cap_dict.get("description") or "")[:100]
                cost = cap_dict.get("cost") or {}
                cost_usd = cost.get("estimated_cost_usd")
                cost_str = f"${cost_usd:.4f}" if cost_usd else "free"
                schema = cap_dict.get("input_schema") or {}
                props = schema.get("properties") or {}
                required = list(schema.get("required") or [])
                optional = [f for f in props if f not in required]

                field_parts: list[str] = []
                for f in required:
                    fi = props.get(f) or {}
                    ftype = _TYPE_ABBREV.get(fi.get("type", "") if isinstance(fi, dict) else "", "any")
                    field_parts.append(f"{f}({ftype},REQ)")
                for f in optional[:_MAX_OPTIONAL_FIELDS_PER_CAP]:
                    fi = props.get(f) or {}
                    ftype = _TYPE_ABBREV.get(fi.get("type", "") if isinstance(fi, dict) else "", "any")
                    field_parts.append(f"{f}({ftype})")

                line = f"  {cap_name} ({cost_str}, relevance:{score:.2f})"
                if desc:
                    line += f": {desc}"
                if field_parts:
                    line += f" → {', '.join(field_parts)}"
                lines.append(line)

                agent_meta = self._agent_meta_cache.get(agent_name, {})
                path_constraint = agent_meta.get("fs_allowed_paths") or None
                if path_constraint:
                    roots_str = ", ".join(str(p) for p in path_constraint)
                    lines.append(f"    [paths must be within: {roots_str}]")
        return "\n".join(lines)

    # ── Multi-phase goal decomposition ─────────────────────────────────────

    async def decompose_goal(self, goal: str, memory_context: str = "") -> dict:
        """
        Lightweight LLM call that classifies the goal complexity and returns
        per-phase search queries for targeted capability retrieval.

        Returns {"complexity": str, "phases": [{"name", "description", "search_queries"}]}.
        Falls back to a single-phase structure on any error.
        """
        compact_memory = self._compact_memory_context(memory_context)
        memory_section = (
            f"\nMemory context:\n{compact_memory}\n" if compact_memory else ""
        )
        try:
            raw = await self._proxy_complete(
                messages=[{
                    "role": "user",
                    "content": f"Goal: {goal}\n{memory_section}",
                }],
                system=_DECOMPOSE_SYSTEM_PROMPT,
                max_tokens=512,
            )
            result = self._extract_json(raw)
            phases = result.get("phases") or []
            if not phases:
                raise ValueError("Empty phases")
            logger.info(
                "Goal decomposed: complexity=%s phases=%d",
                result.get("complexity", "?"), len(phases),
            )
            return result
        except Exception as exc:
            logger.warning("Goal decomposition failed (%s) — using single-phase fallback", exc)
            return {
                "complexity": "simple",
                "phases": [{"name": "Execute", "description": goal, "search_queries": [goal]}],
            }

    async def _gather_multiphase_caps(self, goal: str, phases: list[dict]) -> str:
        """
        For each phase, run vector searches and union the results (deduplicated).
        Returns formatted capability text ready for the planning LLM prompt.
        """
        seen_keys: set[tuple[str, str]] = set()
        combined: list[dict] = []
        per_phase_k = max(4, self._vector_search_top_k // max(len(phases), 1))

        for phase in phases:
            queries = phase.get("search_queries") or [phase.get("description", goal)]
            for q in queries:
                search_query = f"{goal} {q}".strip()
                results = await self.discover_capabilities_semantic(search_query, limit=per_phase_k)
                for item in results:
                    key = (item.get("agent_name", ""), item.get("capability_name", ""))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        combined.append(item)

        logger.info(
            "Multi-phase search: %d unique capabilities gathered across %d phase(s)",
            len(combined), len(phases),
        )
        return self._format_semantic_caps(combined, goal=goal)

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

    async def answer_followup_from_memory(
        self,
        question: str,
        memory_context: str = "",
    ) -> dict:
        """
        Try to answer a follow-up question from memory context only.
        Returns {"found": bool, "answer": str, "confidence": float}.
        """
        compact_memory = self._compact_memory_context(memory_context)
        if not compact_memory.strip():
            return {"found": False, "answer": "", "confidence": 0.0}
        try:
            raw = await self._proxy_complete(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Memory context:\n{compact_memory}"
                    ),
                }],
                system=_FOLLOWUP_MEMORY_SYSTEM_PROMPT,
                max_tokens=256,
            )
            parsed = self._extract_json(raw)
            found = bool(parsed.get("found"))
            answer = str(parsed.get("answer", "")).strip()
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if not found or not answer:
                return {"found": False, "answer": "", "confidence": max(0.0, confidence)}
            return {"found": True, "answer": answer, "confidence": max(0.0, min(1.0, confidence))}
        except Exception as exc:
            logger.warning("Follow-up memory answer failed (fail closed): %s", exc)
            return {"found": False, "answer": "", "confidence": 0.0}

    def _flatten_capabilities(self, agents: list[dict]) -> list[dict]:
        caps: list[dict] = []
        skip_agents = {"task-planner-agent", "task-executor-agent"}
        for agent in agents:
            agent_name = agent.get("name", "unknown")
            if agent_name in skip_agents:
                continue
            if agent.get("disabled"):
                continue
            agent_metadata = agent.get("metadata") or {}
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
                # Carry path constraint from agent metadata (set by filesystem-agent)
                path_constraint = agent_metadata.get("fs_allowed_paths") or None
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
                    "path_constraint": path_constraint,
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
        if name in {"send_slack_message", "send_email", "send_whatsapp_message", "send_telegram_message"}:
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
            if cap["capability_name"] in {"send_slack_message", "send_email", "send_whatsapp_message", "send_telegram_message"}:
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
        if not caps:
            return "  (no agents currently available)"
        if compact:
            return self._format_caps_compact(caps)
        return self._format_caps_verbose(caps)

    def _format_caps_compact(self, caps: list[dict]) -> str:
        """
        Single-line-per-capability format grouped by agent. ~60% fewer tokens than verbose.
        Example:
          [browser-agent]
            browse_web (free): Navigate the web to complete a task. → task(str,REQ)
        """
        from collections import defaultdict
        by_agent: dict[str, list[dict]] = defaultdict(list)
        for cap in caps:
            by_agent[cap["agent_name"]].append(cap)

        lines: list[str] = []
        for agent_name, agent_caps in by_agent.items():
            lines.append(f"[{agent_name}]")
            for cap in agent_caps:
                cap_name = cap["capability_name"]
                desc = (cap.get("description") or "")[:100]
                cost_usd = cap.get("cost_usd")
                cost_str = f"${cost_usd:.4f}" if cost_usd else "free"
                required = cap.get("required_fields", [])
                optional = cap.get("optional_fields", [])
                properties = cap.get("properties", {})

                field_parts: list[str] = []
                for f in required:
                    ftype = _TYPE_ABBREV.get(
                        (properties.get(f) or {}).get("type", ""), "any"
                    )
                    field_parts.append(f"{f}({ftype},REQ)")
                for f in optional[:_MAX_OPTIONAL_FIELDS_PER_CAP]:
                    ftype = _TYPE_ABBREV.get(
                        (properties.get(f) or {}).get("type", ""), "any"
                    )
                    field_parts.append(f"{f}({ftype})")

                line = f"  {cap_name} ({cost_str})"
                if desc:
                    line += f": {desc}"
                if field_parts:
                    line += f" → {', '.join(field_parts)}"
                lines.append(line)

                path_constraint = cap.get("path_constraint")
                if path_constraint:
                    roots_str = ", ".join(str(p) for p in path_constraint)
                    lines.append(f"    [paths must be within: {roots_str}]")
        return "\n".join(lines)

    def _format_caps_verbose(self, caps: list[dict]) -> str:
        """Full multi-line format with all fields and descriptions. Used for non-compact mode."""
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
                lines.append(f"    Description: {cap_desc}")
            if cost_type == "free" or cost_usd is None:
                lines.append("    Cost: free")
            else:
                note = f" ({cost_note})" if cost_note else ""
                lines.append(f"    Cost: ${cost_usd:.4f}/call{note}")
            path_constraint = cap.get("path_constraint")
            if path_constraint:
                roots_str = ", ".join(str(p) for p in path_constraint)
                lines.append(f"    Path constraint: ALL file paths MUST be within: {roots_str}")
            if properties:
                lines.append("    Input fields:")
                for field_name in list(required_fields) + list(optional_fields):
                    field_info = properties.get(field_name, {})
                    marker = " [REQUIRED]" if field_name in required_fields else " [optional]"
                    field_type = field_info.get("type", "any")
                    field_desc = field_info.get("description", "")
                    lines.append(f"      - {field_name} ({field_type}){marker}: {field_desc}")
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
    def is_sensitive_memory_entry(entry: dict) -> bool:
        content = str((entry or {}).get("content", "")).lower()
        if not content:
            return False
        if any(k in content for k in _SENSITIVE_KEYWORDS):
            return True
        # Simple PII patterns
        if re.search(r"\b\d{3}-\d{2}-\d{4}\b", content):   # US SSN
            return True
        if re.search(r"\b(?:\d[ -]*?){13,19}\b", content): # card-like long digit sequence
            return True
        if re.search(r"\b\d{10,}\b", content):             # long numeric identifiers
            return True
        return False

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
        delivery_channel: str = "",
        persona: str = "",
        summary_format: str = "",
        source: str = "",
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

        now_utc = datetime.now(timezone.utc)

        # ── Capability retrieval: vector search or full-context ───────────────
        planning_mode: str
        if self._vector_search_enabled:
            if self._vector_search_multiphase:
                # Multi-phase: decompose goal → targeted per-phase searches
                decomposition = await self.decompose_goal(goal, memory_context=memory_context)
                phases = decomposition.get("phases", [])
                complexity = decomposition.get("complexity", "simple")
                caps_text = await self._gather_multiphase_caps(goal, phases)
                planning_mode = f"vector-multiphase({complexity},{len(phases)}ph)"
            else:
                # Single vector search
                vec_results = await self.discover_capabilities_semantic(goal)
                if vec_results:
                    caps_text = self._format_semantic_caps(vec_results, goal=goal)
                    planning_mode = f"vector-single(top{self._vector_search_top_k})"
                else:
                    # Vector search returned nothing — fall back to full context
                    logger.warning("Vector search returned no results — falling back to full context")
                    agents = await self.discover_capabilities()
                    caps_text = self._format_capabilities(agents, goal=goal, compact=True)
                    planning_mode = "full-context(vector-fallback)"
        else:
            agents = await self.discover_capabilities()
            caps_text = self._format_capabilities(agents, goal=goal, compact=True)
            planning_mode = "full-context"

        logger.info(
            "Planning workflow: goal=%r  mode=%s  memory=%s",
            goal[:80], planning_mode, "yes" if memory_context else "none",
        )

        reply_context_lines: list[str] = [f"REQUESTER_ID: {requester_id}"]
        if source:
            reply_context_lines.append(f"SOURCE: {source}")
        if channel_id:
            reply_context_lines.append(f"REPLY_CHANNEL_ID: {channel_id}")
        if thread_id:
            reply_context_lines.append(f"REPLY_THREAD_ID: {thread_id}")
        if user_id:
            reply_context_lines.append(f"USER_ID: {user_id}")
        if delivery_channel:
            reply_context_lines.append(f"DELIVERY_CHANNEL: {delivery_channel}")
        if persona:
            reply_context_lines.append(f"PERSONA: {persona}")
        if summary_format:
            reply_context_lines.append(f"SUMMARY_FORMAT: {summary_format}")
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
            system=self._plan_system_prompt,
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

        if planner_entries:
            import asyncio as _asyncio
            _asyncio.create_task(
                self.write_memory_entries(
                    user_entries=[],
                    user_id=user_id,
                    planner_entries=planner_entries,
                ),
                name="cortex-write-plan",
            )
        if user_entries:
            logger.info(
                "Extracted %d candidate user memory entr%s for consented storage",
                len(user_entries),
                "ies" if len(user_entries) != 1 else "y",
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
        plan.memory_entries = user_entries
        plan.planning_mode = planning_mode  # stored for dashboard/logging inspection
        logger.info(
            "Plan created: task_id=%s  title=%r  steps=%d  mode=%s\n%s",
            plan.task_id, plan.title, len(steps), planning_mode,
            json.dumps(plan.to_dict(), indent=2),
        )
        return plan
