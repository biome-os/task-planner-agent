"""
orchestrator_client.py — WebSocket + HTTP client for the task-planner-agent.

Execution model
───────────────
1. plan_task received  →  single LLM call generates WorkflowPlan
2. Plan persisted to SQLite (workflow_store.py)
3. Step 0 dispatched to best available agent; correlation_id saved in DB
4. plan_task returns immediately: {task_id, title, total_steps, status}
5. Agent completes step  →  task_response arrives (the "callback")
6. Planner looks up (task_id, step_index) via correlation_id in DB
7. Step output saved; next step dispatched (repeat until all done)
8. workflow_event messages emitted throughout for dashboard tracing
9. If a step returns output_data.followup_request, planner resolves from Cortex
   or asks user, patches step input, then retries the same step.
10. Sensitive memory entries require explicit user consent before Cortex write.

LLM optimisations
─────────────────
- Capability list cached 60 s — multiple plans share one REST fetch
- Discovery (best agent per capability) cached 30 s per capability key
- Exactly ONE LLM proxy call per plan_task request
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import websockets
import websockets.exceptions

from emergent_runner import EmergentStepRunner
from formatters import render_formatter
from models import WorkflowPlan
from plan_tracker import PlanTracker
from planner import TaskPlanner
from workflow_store import WorkflowStore

logger = logging.getLogger(__name__)

# ── Stable agent identity ──────────────────────────────────────────────────────

_AGENT_ID_FILE = Path(".agent_id")


def _stable_agent_id() -> str:
    if _AGENT_ID_FILE.exists():
        return _AGENT_ID_FILE.read_text().strip()
    new_id = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(new_id)
    logger.info("Generated new stable agent ID: %s", new_id)
    return new_id


# ── Step-ref resolver ──────────────────────────────────────────────────────────

# Matches {{steps[N].output}} and {{steps[N].output.field.path}}
# Group 1: step index; Group 2: optional dotted field path (None = whole output)
_REF_RE = re.compile(r"\{\{steps\[(\d+)\]\.output(?:\.([^}]+))?\}\}")
_ARRAY_KEY_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def _traverse(node: Any, key: str) -> Any:
    """Traverse one path segment; supports array indexing like ``results[0]``."""
    m = _ARRAY_KEY_RE.match(key)
    if m:
        dict_key, arr_idx = m.group(1), int(m.group(2))
        node = node.get(dict_key) if isinstance(node, dict) else None
        if isinstance(node, list) and arr_idx < len(node):
            return node[arr_idx]
        return None
    return node.get(key) if isinstance(node, dict) else None


def _resolve_step_refs(value: Any, outputs: list[Optional[dict]]) -> Any:
    """Recursively substitute ``{{steps[N].output[.field]}}`` in *value*.

    When the field path is omitted (``{{steps[N].output}}``) the whole output
    dict is returned — useful for passing an entire step's output as
    ``input_data.data`` to a ``format_step_output`` step.
    """

    def _resolve_str(s: str) -> Any:
        full = _REF_RE.fullmatch(s)
        if full:
            idx = int(full.group(1))
            path_str = full.group(2)   # None when no .field path given
            node: Any = (outputs[idx] or {}) if idx < len(outputs) else {}
            if path_str:
                for key in path_str.split("."):
                    node = _traverse(node, key)
            return node  # may be a dict/list when path_str is None

        def _sub(m: re.Match) -> str:
            idx = int(m.group(1))
            path_str = m.group(2)
            node: Any = (outputs[idx] or {}) if idx < len(outputs) else {}
            if path_str:
                for key in path_str.split("."):
                    node = _traverse(node, key)
            return str(node) if node is not None else m.group(0)

        return _REF_RE.sub(_sub, s)

    if isinstance(value, str):
        return _resolve_str(value)
    if isinstance(value, dict):
        return {k: _resolve_step_refs(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_step_refs(v, outputs) for v in value]
    return value


def _clean_text(value: Any) -> str:
    """Return a stripped string for optional request fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalise_step_input(capability: str, input_data: dict[str, Any]) -> dict[str, Any]:
    """Map common alias fields to capability-specific input fields."""
    if capability != "send_slack_message":
        return input_data

    normalised = dict(input_data)
    # Planner prompt asks for channel_id/thread_id context; Slack capability
    # expects channel/thread_ts.
    if not normalised.get("channel"):
        channel_id = normalised.get("channel_id")
        if isinstance(channel_id, str) and channel_id.strip():
            normalised["channel"] = channel_id.strip()

    if not normalised.get("thread_ts"):
        thread_id = normalised.get("thread_id")
        if isinstance(thread_id, str) and thread_id.strip():
            normalised["thread_ts"] = thread_id.strip()

    return normalised


def _inject_slack_user_id(steps: list[dict[str, Any]], user_id: str) -> None:
    """Best-effort propagation of Slack user_id into slack send steps."""
    if not user_id:
        return
    for step in steps:
        capability = step.get("capability")
        input_data = step.get("input_data")
        if not isinstance(input_data, dict):
            continue
        if capability == "send_slack_message":
            input_data.setdefault("user_id", user_id)
            continue
        if capability == "schedule_task":
            nested_cap = input_data.get("capability")
            nested_data = input_data.get("input_data")
            if nested_cap == "send_slack_message" and isinstance(nested_data, dict):
                nested_data.setdefault("user_id", user_id)


def _parse_yes_no_reply(text: str) -> Optional[bool]:
    """Parse a free-text user reply into yes/no/unknown."""
    t = _clean_text(text).lower()
    if not t:
        return None
    yes = {"y", "yes", "ok", "okay", "continue", "proceed", "approve", "approved", "go"}
    no = {"n", "no", "stop", "cancel", "deny", "decline", "reject", "do not continue", "dont continue"}
    if t in yes:
        return True
    if t in no:
        return False
    if any(k in t for k in (" yes", "continue", "proceed", "approve")):
        return True
    if any(k in t for k in (" no", "stop", "cancel", "decline", "reject")):
        return False
    return None


def _normalise_followup_answer(
    answer_text: str,
    answer_format: str,
    choices: list[str],
) -> tuple[Optional[object], Optional[str]]:
    """Parse a user answer according to follow-up answer_format."""
    raw = _clean_text(answer_text)
    fmt = _clean_text(answer_format).lower() or "text"
    if fmt == "boolean":
        val = _parse_yes_no_reply(raw)
        if val is None:
            return None, "Please reply with yes or no."
        return val, None
    if fmt == "choice":
        if not choices:
            return raw, None
        lowered = {c.lower(): c for c in choices}
        if raw.lower() in lowered:
            return lowered[raw.lower()], None
        for c in choices:
            if c.lower() in raw.lower():
                return c, None
        opts = ", ".join(choices)
        return None, f"Please choose one of: {opts}"
    if fmt == "number":
        try:
            if "." in raw:
                return float(raw), None
            return int(raw), None
        except Exception:
            return None, "Please reply with a valid number."
    if fmt == "json":
        try:
            return json.loads(raw), None
        except Exception:
            return None, "Please reply with valid JSON."
    return raw, None


def _memory_entries_preview(entries: list[dict], limit: int = 3) -> str:
    lines: list[str] = []
    for idx, entry in enumerate(entries[:limit], 1):
        category = _clean_text(entry.get("category", "Facts")) or "Facts"
        content = _clean_text(entry.get("content", ""))
        if len(content) > 140:
            content = content[:137] + "..."
        lines.append(f"{idx}. [{category}] {content}")
    return "\n".join(lines)


# ── Agent identity ─────────────────────────────────────────────────────────────

AGENT_NAME        = "task-planner-agent"
AGENT_VERSION     = "2.0.0"
AGENT_DESCRIPTION = (
    "Accepts a natural-language goal, discovers available agent capabilities, "
    "generates a structured workflow plan with one LLM call, persists state in "
    "SQLite, and drives step-by-step execution directly — dispatching each step "
    "to the appropriate agent and resuming on callback."
)

def _load_default_prompt() -> str:
    _pf = Path(__file__).parent / "prompts" / "system_prompt.md"
    try:
        return _pf.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


REGISTRATION_PAYLOAD: dict = {
    "name":           AGENT_NAME,
    "description":    AGENT_DESCRIPTION,
    "version":        AGENT_VERSION,
    "default_prompt": _load_default_prompt(),
    "capabilities": [
        {
            "name": "plan_task",
            "description": (
                "Accept a natural-language goal, produce a structured multi-step "
                "workflow plan (one LLM call), persist it, and drive execution by "
                "dispatching each step to the appropriate agent. Returns immediately "
                "with task_id; execution continues asynchronously."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Natural-language description of the task to plan and execute.",
                    },
                    "channel_id": {
                        "type": "string",
                        "description": "Channel identifier to send the completion response back to.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Slack user id for DM fallback delivery (e.g. U0123...).",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Conversation thread identifier to reply into when the workflow completes.",
                    },
                    "delivery_channel": {
                        "type": "string",
                        "description": "Preferred delivery channel for summaries/completion (e.g. slack, email, telegram, whatsapp).",
                    },
                    "persona": {
                        "type": "string",
                        "description": "Optional persona/tone for generated summaries (e.g. executive, friendly, concise).",
                    },
                    "summary_format": {
                        "type": "string",
                        "description": "Optional format instructions for summaries (e.g. bullets with action items).",
                    },
                },
                "required": ["goal"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "task_id":     {"type": "string"},
                    "title":       {"type": "string"},
                    "description": {"type": "string"},
                    "total_steps": {"type": "integer"},
                    "status":      {"type": "string"},
                },
            },
            "tags": ["planning", "llm", "workflow"],
            "cost": {"type": "per_call", "estimated_cost_usd": 0.003, "notes": "Claude API ~1k tokens/plan"},
        },
        {
            "name": "get_workflow_status",
            "description": "Query the current status and step details of a planned workflow.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "task_id returned by plan_task.",
                    },
                },
                "required": ["task_id"],
            },
            "output_schema": {
                "type": "object",
                "properties": {"workflow": {"type": "object"}},
            },
            "tags": ["planning", "workflow"],
            "cost": {"type": "free", "estimated_cost_usd": None, "notes": "SQLite read"},
        },
        {
            "name": "list_workflows",
            "description": "List recent workflows managed by this planner.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results (default 20)."},
                },
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "workflows": {"type": "array"},
                    "count":     {"type": "integer"},
                },
            },
            "tags": ["planning", "workflow"],
            "cost": {"type": "free", "estimated_cost_usd": None, "notes": "SQLite read"},
        },
        {
            "name": "format_step_output",
            "description": (
                "Format a prior workflow step's raw output into a human-readable "
                "Slack message using a registered Jinja2 template for that capability. "
                "Use {{steps[N].output}} in input_data.data to pass the full output of "
                "step N. The formatted text is returned as output_data.text."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": (
                            "The raw output dict from a prior step. "
                            "Use the template reference {{steps[N].output}} so the "
                            "executor substitutes the actual output at dispatch time."
                        ),
                    },
                    "capability_name": {
                        "type": "string",
                        "description": (
                            "Exact capability name whose formatter template to use "
                            "(e.g. 'serper_search', 'browse_web'). Falls back to a "
                            "generic formatter when no specific template is registered."
                        ),
                    },
                    "template": {
                        "type": "string",
                        "description": (
                            "Optional: custom Jinja2 template string. Overrides the "
                            "registered formatter when provided."
                        ),
                    },
                },
                "required": ["data", "capability_name"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Formatted, Slack-ready message text.",
                    },
                },
            },
            "tags": ["formatting", "slack", "template"],
            "cost": {"type": "free", "estimated_cost_usd": None, "notes": "Local Jinja2 render"},
        },
    ],
    "tags": ["planner", "llm", "workflow", "orchestration"],
    "metadata": {
        "language":  "python",
        "llm_model": "claude-sonnet-4-6",
        "llm_calls_per_plan": 1,
        "persistence": "sqlite",
    },
    "required_settings": [
        {
            "key": "planner_model",
            "label": "Model",
            "type": "string",
            "required": False,
            "description": (
                "LLM model for workflow planning. "
                "Leave empty to use the global default model. "
                "Examples: claude-sonnet-4-6, claude-opus-4-6, gpt-4o"
            ),
            "default": "",
        },
        {
            "key": "planner_provider",
            "label": "Provider",
            "type": "string",
            "required": False,
            "description": (
                "LLM provider: anthropic, openai, or gemini. "
                "Leave empty to use the global default provider."
            ),
            "default": "",
        },
        {
            "key": "planner_max_replan_attempts",
            "label": "Max Replan Attempts",
            "type": "integer",
            "required": False,
            "description": (
                "How many times the planner may automatically revise and retry a "
                "workflow after a step fails before giving up. Default: 3."
            ),
            "default": 3,
        },
        {
            "key": "planner_hybrid_execution",
            "label": "Hybrid Execution",
            "type": "boolean",
            "required": False,
            "description": (
                "When enabled, low-confidence steps (execution_mode=emergent) are "
                "handled by an LLM tool loop instead of strict direct dispatch. "
                "High-confidence steps are unaffected. Default: false."
            ),
            "default": False,
        },
        {
            "key": "planner_emergent_max_turns",
            "label": "Emergent Max Turns",
            "type": "integer",
            "required": False,
            "description": (
                "Maximum LLM+capability iterations per emergent step. Default: 6."
            ),
            "default": 6,
        },
        {
            "key": "planner_emergent_only",
            "label": "Emergent-Only Execution",
            "type": "boolean",
            "required": False,
            "description": (
                "When enabled, ALL steps are routed through the emergent tool-loop "
                "runner regardless of their execution_mode. Takes priority over "
                "hybrid execution when both are true. Default: false."
            ),
            "default": False,
        },
    ],
}

# ── Constants ──────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_S:   int   = 15
MAX_BACKOFF_S:          int   = 60
DRAIN_TIMEOUT_S:        int   = 30
STEP_TIMEOUT_S:         float = 300.0   # 5 min per step
DISCOVERY_CACHE_TTL_S:  float = 30.0   # cache best-agent per capability
MAX_REPLAN_ATTEMPTS:    int   = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _envelope(
    sender_id: str,
    msg_type: str,
    payload: dict,
    recipient_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    msg_id: Optional[str] = None,
) -> str:
    return json.dumps({
        "id":             msg_id or str(uuid.uuid4()),
        "type":           msg_type,
        "sender_id":      sender_id,
        "recipient_id":   recipient_id,
        "payload":        payload,
        "timestamp":      _now_iso(),
        "correlation_id": correlation_id,
    })


# ── Main client ────────────────────────────────────────────────────────────────

class OrchestratorClient:
    """
    Registers the task-planner-agent, drives stateful workflow execution,
    and handles per-step agent callbacks via the existing task_response protocol.
    """

    def __init__(self, orchestrator_url: str = "http://localhost:8000") -> None:
        self._base = orchestrator_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30)

        self._agent_id:  str = ""
        self._ws_url:    str = ""

        self._status:            str   = "starting"
        self._active_tasks:      int   = 0
        self._tasks_completed:   int   = 0
        self._tasks_failed:      int   = 0
        self._total_duration_ms: float = 0.0
        self._start_time:        float = time.monotonic()

        self._shutting_down: bool = False
        self._current_ws:    Any  = None

        # Pending responses for non-step outbound requests
        self._pending_responses: dict[str, asyncio.Future] = {}

        # Discovery cache: capability → (expire_time, agent_id)
        self._discovery_cache: dict[str, tuple[float, str]] = {}

        self._store: WorkflowStore = WorkflowStore()
        self._planner: Optional[TaskPlanner] = None
        self._common_settings: dict = {}
        self._agent_settings: dict = {}
        self._registered_prompt: str = ""
        self._max_replan_attempts: int = MAX_REPLAN_ATTEMPTS
        self._hybrid_execution: bool = False
        self._emergent_only: bool = False
        self._emergent_max_turns: int = 6
        self._plan_tracker: PlanTracker = PlanTracker()

        # Futures used by emergent runner to await task_response callbacks
        self._emergent_pending: dict[str, asyncio.Future] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._graceful_shutdown()))

        self._store.open()
        await self._register()
        self._planner = TaskPlanner(
            orchestrator_base_url=self._base,
            agent_id=self._agent_id,
        )
        self._planner.update_settings(self._common_settings, self._agent_settings)
        if self._registered_prompt:
            self._planner.update_prompt(self._registered_prompt)
        await self._connect_loop()

    # ── Registration ───────────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": _stable_agent_id()}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id        = data["agent_id"]
        self._ws_url          = data["ws_url"]
        self._common_settings = data.get("common_settings", {})
        self._agent_settings  = data.get("agent_settings", {})
        self._registered_prompt = data.get("system_prompt", "")
        try:
            raw = self._agent_settings.get("planner_max_replan_attempts")
            if raw is not None:
                self._max_replan_attempts = max(1, int(raw))
        except (ValueError, TypeError):
            pass
        self._apply_hybrid_settings(self._agent_settings)
        logger.info("Registered — agent_id=%s", self._agent_id)

    # ── WebSocket loop ─────────────────────────────────────────────────────────

    async def _connect_loop(self) -> None:
        backoff = 1.0
        while not self._shutting_down:
            try:
                logger.info("Connecting to %s …", self._ws_url)
                async with websockets.connect(self._ws_url) as ws:
                    backoff = 1.0
                    await self._run_session(ws)

            except websockets.exceptions.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    logger.warning("Unknown agent_id (4004) — re-registering …")
                    try:
                        await self._register()
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
                elif code == 4003:
                    logger.info("Agent is disabled by orchestrator (4003) — will retry so dashboard enable can restore connection")
                    backoff = max(backoff, 10.0)
                elif self._shutting_down:
                    break
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)

            except (OSError, Exception) as exc:
                if self._shutting_down:
                    break
                logger.warning("WS error (%s) — retry in %.0fs", exc, backoff)

            if not self._shutting_down:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_S)

    async def _run_session(self, ws) -> None:
        self._current_ws = ws
        self._status = "available"
        logger.info("WebSocket session active — status: available")

        # Resume any workflows that were mid-flight before this connection
        asyncio.create_task(
            self._resume_in_progress_workflows(ws),
            name="resume-workflows",
        )

        try:
            await asyncio.gather(
                self._heartbeat_loop(ws),
                self._recv_loop(ws),
            )
        finally:
            self._current_ws = None
            self._status = "offline"
            for fut in list(self._pending_responses.values()):
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket session ended"))

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await self._ws_send(ws, self._msg(
                "heartbeat",
                {
                    "status":       self._status,
                    "current_load": min(self._active_tasks / 5.0, 1.0),
                    "active_tasks": self._active_tasks,
                    "metrics":      self._metrics(),
                },
            ))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Receive loop ───────────────────────────────────────────────────────────

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Non-JSON frame ignored")
                continue
            mtype = msg.get("type", "?")
            logger.info("← [%s] from=%s", mtype, msg.get("sender_id", "?"))
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype   = msg.get("type", "")
        payload = msg.get("payload", {})

        if mtype == "task_request":
            asyncio.create_task(self._handle_incoming_task(ws, msg))

        elif mtype == "task_response":
            corr = msg.get("correlation_id")
            if not corr:
                return

            # ── Step callback path ─────────────────────────────────────────
            mapping = await asyncio.to_thread(self._store.pop_correlation, corr)
            if mapping:
                task_id, step_index = mapping
                asyncio.create_task(
                    self._on_step_response(ws, task_id, step_index, payload),
                    name=f"cb-{task_id[:8]}-{step_index}",
                )
                return

            # ── Emergent tool-loop callback path ───────────────────────────
            if corr in self._emergent_pending:
                fut = self._emergent_pending.pop(corr)
                if not fut.done():
                    fut.set_result(payload)
                return

            # ── Normal pending-response path ───────────────────────────────
            if corr in self._pending_responses:
                fut = self._pending_responses.pop(corr)
                if not fut.done():
                    fut.set_result(payload)

        elif mtype == "memory_response":
            # Cortex write/query response — acknowledged, nothing to act on
            corr = msg.get("correlation_id")
            logger.debug(
                "memory_response received (corr=%s success=%s)",
                corr, payload.get("success"),
            )

        elif mtype == "settings_push":
            settings = payload.get("settings", {})
            logger.info("Settings pushed: %d key(s)", len(settings))
            self._common_settings.update(settings)
            for k in (
                "planner_model", "planner_provider",
                "planner_max_replan_attempts",
                "planner_hybrid_execution",
                "planner_emergent_max_turns",
                "planner_emergent_only",
            ):
                if k in settings:
                    self._agent_settings[k] = settings[k]
            if "planner_max_replan_attempts" in settings:
                try:
                    self._max_replan_attempts = max(1, int(settings["planner_max_replan_attempts"]))
                    logger.info("Max replan attempts updated → %d", self._max_replan_attempts)
                except (ValueError, TypeError):
                    pass
            self._apply_hybrid_settings(settings)
            if self._planner is not None:
                self._planner.update_settings(self._common_settings, self._agent_settings)

        elif mtype == "prompt_push":
            content = payload.get("content", "")
            if content and self._planner is not None:
                self._planner.update_prompt(content)
                logger.info("Prompt push received (%d chars)", len(content))

        elif mtype == "error":
            logger.error("Orchestrator error [%s]: %s",
                         payload.get("code"), payload.get("detail"))
            original_id = payload.get("original_message_id")
            if original_id and original_id in self._pending_responses:
                fut = self._pending_responses.pop(original_id)
                if not fut.done():
                    fut.set_exception(RuntimeError(
                        f"[{payload.get('code')}] {payload.get('detail')}"
                    ))

        elif mtype == "agent_registered":
            logger.info("Peer joined: %s", payload.get("agent_id"))

        elif mtype == "agent_offline":
            logger.info("Peer left: %s", payload.get("agent_id"))

        else:
            logger.debug("Unhandled message type: %r", mtype)

    # ── Incoming task handling ─────────────────────────────────────────────────

    async def _handle_incoming_task(self, ws, msg: dict) -> None:
        req_id     = msg.get("id")
        sender_id  = msg.get("sender_id")
        payload    = msg.get("payload", {})
        capability = payload.get("capability")
        input_data = payload.get("input_data", {})

        self._active_tasks += 1
        self._status = "busy"
        t0 = time.monotonic()

        try:
            if capability == "plan_task":
                output, error = await self._cap_plan_task(input_data, sender_id, ws)
            elif capability == "get_workflow_status":
                output, error = await self._cap_get_workflow_status(input_data)
            elif capability == "list_workflows":
                output, error = await self._cap_list_workflows(input_data)
            elif capability == "format_step_output":
                output, error = await self._cap_format_step_output(input_data)
            else:
                output, error = None, f"Unknown capability: {capability!r}"

            duration_ms = (time.monotonic() - t0) * 1000

            if error:
                self._tasks_failed += 1
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": error,
                     "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))
            else:
                self._tasks_completed += 1
                self._total_duration_ms += duration_ms
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": True, "output_data": output,
                     "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._tasks_failed += 1
            logger.exception("Unhandled error in capability %r", capability)
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": str(exc),
                 "duration_ms": round(duration_ms, 1)},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))

        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
            self._status = "draining" if self._shutting_down else (
                "busy" if self._active_tasks else "available"
            )
            await self._send_status_update(ws)

    # ── Capability: plan_task ──────────────────────────────────────────────────

    async def _cap_plan_task(
        self, input_data: dict, requester_id: str, ws
    ) -> tuple[dict | None, str | None]:
        goal       = _clean_text(input_data.get("goal"))
        channel_id = _clean_text(input_data.get("channel_id"))
        user_id    = _clean_text(input_data.get("user_id"))
        thread_id  = _clean_text(input_data.get("thread_id"))
        delivery_channel = _clean_text(input_data.get("delivery_channel")).lower()
        persona = _clean_text(input_data.get("persona"))
        summary_format = _clean_text(input_data.get("summary_format"))
        source = _clean_text(input_data.get("source"))
        session_history = input_data.get("session_history") or []
        if not isinstance(session_history, list):
            session_history = []
        payload = input_data.get("payload")
        if isinstance(payload, dict):
            if not channel_id:
                channel_id = _clean_text(payload.get("channel_id"))
            if not user_id:
                user_id = _clean_text(payload.get("user_id"))
            if not thread_id:
                # Slack sends thread_ts; planner uses thread_id semantics.
                thread_id = (
                    _clean_text(payload.get("thread_id"))
                    or _clean_text(payload.get("thread_ts"))
                )
            if not delivery_channel:
                delivery_channel = _clean_text(payload.get("delivery_channel")).lower()
            if not persona:
                persona = _clean_text(payload.get("persona"))
            if not summary_format:
                summary_format = _clean_text(payload.get("summary_format"))
        if not goal:
            return None, "input_data.goal is required"
        if not self._planner:
            return None, "Planner not initialised"

        # Step -1: check if this is a reply to a pending agent follow-up question
        if thread_id and channel_id:
            pending_followup = await asyncio.to_thread(
                self._store.get_pending_followup, thread_id, channel_id
            )
            if pending_followup:
                return await self._handle_followup_reply(
                    ws=ws,
                    user_reply=goal,
                    pending=pending_followup,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    user_id=user_id,
                )

        # Step 0: check if this is a reply to a pending replan approval request
        if thread_id and channel_id:
            pending_replan = await asyncio.to_thread(
                self._store.get_pending_replan_approval, thread_id, channel_id
            )
            if pending_replan:
                return await self._handle_replan_approval_reply(
                    ws=ws,
                    user_reply=goal,
                    pending=pending_replan,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    user_id=user_id,
                )

        # Step 0.5: reply to pending sensitive-memory consent request
        if thread_id and channel_id:
            pending_consent = await asyncio.to_thread(
                self._store.get_pending_memory_consent, thread_id, channel_id
            )
            if pending_consent:
                return await self._handle_memory_consent_reply(
                    user_reply=goal,
                    pending=pending_consent,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    user_id=user_id,
                )

        # ── Fetch Cortex memory once — used for clarification check AND planning ──
        memory_context = await self._planner.fetch_memory_context(user_id=user_id)
        if memory_context:
            logger.info("Cortex memory loaded for user=%s", user_id or "—")

        # ── Clarification gate ────────────────────────────────────────────────
        effective_goal = goal
        clarification_message = ""
        clarification_answers = ""

        # Step 1: check if incoming message is a reply to a pending clarification
        if thread_id and channel_id:
            pending = await asyncio.to_thread(
                self._store.get_pending_clarification, thread_id, channel_id
            )
            if pending:
                logger.info("Clarification reply received (thread=%s)", thread_id[:12])
                # Restore the original goal; keep the user's reply as answers
                # so plan() can build a proper multi-turn conversation history.
                effective_goal = pending['goal']
                clarification_answers = goal  # the user's answers to the questions

                clarification_message = pending.get('clarification_message', '')
                if not clarification_message:
                    # Reconstruct from the stored questions list — covers records
                    # created before the clarification_message column was added.
                    try:
                        stored_qs = json.loads(pending.get('questions', '[]'))
                        if stored_qs:
                            q_lines = "\n".join(
                                f"{i+1}. {q}" for i, q in enumerate(stored_qs)
                            )
                            clarification_message = (
                                f"Before creating a plan, I asked:\n{q_lines}"
                            )
                    except (json.JSONDecodeError, TypeError):
                        pass
                if clarification_message:
                    logger.debug(
                        "Clarification context restored (len=%d)", len(clarification_message)
                    )
                else:
                    logger.warning(
                        "No clarification_message available — answers will be "
                        "injected directly into planning context"
                    )

                await asyncio.to_thread(
                    self._store.delete_pending_clarification, pending["id"]
                )

        # Step 2: if not a reply (effective_goal unchanged) and we can send messages,
        #         check if clarification is needed — memory context suppresses
        #         questions whose answers are already known from Cortex.
        if effective_goal == goal and (channel_id or user_id) and self._planner:
            try:
                agents = await self._planner.discover_capabilities()
                clarity = await self._planner.check_needs_clarification(
                    goal, agents, memory_context=memory_context,
                    session_history=session_history,
                )
                if clarity.get("needs_clarification") and clarity.get("questions"):
                    questions: list[str] = clarity["questions"][:3]
                    understood_as: str = clarity.get("understood_as", "")

                    q_lines = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
                    if understood_as:
                        msg = (
                            f"I understand you'd like: _{understood_as}_\n\n"
                            f"Before I create a plan, I have a few questions:\n{q_lines}\n\n"
                            "_Please reply in this thread with your answers._"
                        )
                    else:
                        msg = (
                            f"Before I create a plan, I need a few details:\n{q_lines}\n\n"
                            "_Please reply in this thread with your answers._"
                        )

                    clarification_id = str(uuid.uuid4())
                    await asyncio.to_thread(
                        self._store.save_pending_clarification,
                        clarification_id, thread_id, channel_id,
                        requester_id, user_id, goal, json.dumps(questions), msg,
                    )
                    await self._send_clarification_message(
                        channel_id, msg, thread_ts=thread_id, user_id=user_id
                    )
                    logger.info(
                        "Sent %d clarification question(s) (id=%s)",
                        len(questions), clarification_id[:8],
                    )
                    return {
                        "task_id": clarification_id,
                        "status":  "awaiting_clarification",
                        "message": "Clarification questions sent to requester",
                    }, None
            except Exception as exc:
                logger.warning("Clarification check failed — proceeding directly: %s", exc)

        # ── Single LLM call ──────────────────────────────────────────────────
        try:
            plan: WorkflowPlan = await self._planner.plan(
                effective_goal, requester_id,
                channel_id=channel_id,
                thread_id=thread_id,
                user_id=user_id,
                delivery_channel=delivery_channel,
                persona=persona,
                summary_format=summary_format,
                source=source,
                memory_context=memory_context,              # reuse already-fetched context
                clarification_message=clarification_message,  # "" for fresh requests
                clarification_answers=clarification_answers,  # "" for fresh requests
                session_history=session_history,
            )
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            return None, f"Planning failed: {exc}"

        steps = [s.to_dict() for s in plan.steps]
        _inject_slack_user_id(steps, user_id)
        logger.info("Plan ready: task_id=%s  title=%r  steps=%d",
                    plan.task_id, plan.title, len(steps))

        # ── Persist user memory entries with sensitive-data consent ───────────
        memory_entries = [
            e for e in (plan.memory_entries or [])
            if isinstance(e, dict) and _clean_text(e.get("content"))
        ]
        sensitive_entries = [
            e for e in memory_entries if self._planner.is_sensitive_memory_entry(e)
        ]
        non_sensitive_entries = [
            e for e in memory_entries if not self._planner.is_sensitive_memory_entry(e)
        ]
        if non_sensitive_entries:
            asyncio.create_task(
                self._planner.write_memory_entries(
                    user_entries=non_sensitive_entries,
                    user_id=user_id,
                    planner_entries=None,
                ),
                name=f"cortex-user-memory-{plan.task_id[:8]}",
            )
            if channel_id:
                await self._send_clarification_message(
                    channel_id=channel_id,
                    thread_ts=thread_id,
                    user_id=user_id,
                    text=(
                        f"Saved {len(non_sensitive_entries)} preference/fact entr"
                        f"{'ies' if len(non_sensitive_entries) != 1 else 'y'} to memory."
                    ),
                )
        if sensitive_entries and channel_id and thread_id:
            preview = _memory_entries_preview(sensitive_entries)
            consent_prompt = (
                "I detected sensitive information that could be stored in memory.\n"
                f"{preview}\n\n"
                "Reply `yes` to store it in Cortex for future tasks, or `no` to keep it out of memory."
            )
            consent_id = str(uuid.uuid4())
            await asyncio.to_thread(
                self._store.save_pending_memory_consent,
                consent_id,
                thread_id,
                channel_id,
                requester_id,
                user_id,
                json.dumps(sensitive_entries),
                consent_prompt,
            )
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text=consent_prompt,
            )

        # ── Persist ──────────────────────────────────────────────────────────
        await asyncio.to_thread(
            self._store.create_workflow,
            plan.task_id, effective_goal, plan.title, plan.description,
            requester_id, steps,
            channel_id=channel_id,
            thread_id=thread_id,
            user_id=user_id,
            delivery_channel=delivery_channel,
            persona=persona,
            summary_format=summary_format,
            source=source,
        )
        await asyncio.to_thread(self._store.set_status, plan.task_id, "running")

        # ── Emit workflow_started ─────────────────────────────────────────────
        await self._emit_workflow_event(ws, {
            "event":           "workflow_started",
            "task_id":         plan.task_id,
            "title":           plan.title,
            "description":     plan.description,
            "goal":            effective_goal,
            "total_steps":     len(steps),
            "workflow_status": "running",
        })

        # ── Notify originating channel with plan summary ───────────────────────
        if channel_id or user_id:
            asyncio.create_task(
                self._send_plan_summary(
                    channel_id=channel_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    title=plan.title,
                    description=plan.description,
                    steps=steps,
                ),
                name=f"plan-summary-{plan.task_id[:8]}",
            )

        # ── Dispatch step 0 (non-blocking) ────────────────────────────────────
        if steps:
            asyncio.create_task(
                self._dispatch_step(ws, plan.task_id, 0),
                name=f"step-{plan.task_id[:8]}-0",
            )

        return {
            "task_id":     plan.task_id,
            "title":       plan.title,
            "description": plan.description,
            "total_steps": len(steps),
            "status":      "running" if steps else "completed",
        }, None

    # ── Step dispatch ──────────────────────────────────────────────────────────

    async def _dispatch_step(self, ws, task_id: str, step_index: int) -> None:
        """
        Look up the workflow, resolve input refs, find the best agent,
        send task_request, and save the correlation so the callback is routed here.

        When hybrid execution is enabled and the step has execution_mode="emergent",
        delegates to _dispatch_emergent_step instead.
        """
        workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
        if not workflow:
            logger.error("dispatch_step: workflow %s not found", task_id)
            return

        steps   = workflow["steps"]
        outputs = workflow["outputs"]

        if step_index >= len(steps):
            logger.error("dispatch_step: step %d out of range for %s", step_index, task_id)
            return

        step       = steps[step_index]

        # Emergent-only mode: route every step through the tool-loop runner
        if self._emergent_only:
            asyncio.create_task(
                self._dispatch_emergent_step(ws, task_id, step_index),
                name=f"emergent-only-{task_id[:8]}-{step_index}",
            )
            return

        # Route emergent steps to the tool-loop runner when hybrid mode is active
        if self._hybrid_execution and step.get("execution_mode") == "emergent":
            logger.info(
                "Workflow %s step %d/%d: routing to emergent runner (confidence=%.2f)",
                task_id[:8], step_index + 1, len(steps),
                float(step.get("confidence", 0.0)),
            )
            asyncio.create_task(
                self._dispatch_emergent_step(ws, task_id, step_index),
                name=f"emergent-{task_id[:8]}-{step_index}",
            )
            return
        step_id    = step["step_id"]
        step_name  = step["name"]
        step_goal  = step.get("goal", step.get("description", ""))
        capability = step["capability"]
        total      = len(steps)

        # Resolve {{steps[N].output.field}} references
        input_data = _resolve_step_refs(step.get("input_data", {}), outputs)
        if isinstance(input_data, dict):
            input_data = _normalise_step_input(capability, input_data)

        # Propagate reply context so downstream agents can send status updates
        wf_channel_id = _clean_text(workflow.get("channel_id"))
        if wf_channel_id or _clean_text(workflow.get("user_id")):
            if isinstance(input_data, dict):
                input_data["_reply_context"] = {
                    "channel_type": _clean_text(workflow.get("source")) or "slack",
                    "channel_id":   wf_channel_id,
                    "thread_id":    _clean_text(workflow.get("thread_id")),
                    "user_id":      _clean_text(workflow.get("user_id")),
                }

        # Pre-flight: catch the common mistake where the planner emits an agent name
        # (e.g. "code-execution-agent") instead of a capability name (e.g. "execute_code").
        # Agent names are kebab-case strings ending in "-agent" or "-service"; capability
        # names use snake_case.  Warn early so the replan context is explicit.
        if re.search(r'(?:^|-)agent$|-service$', capability) and '_' not in capability:
            logger.warning(
                "Workflow %s step %d: capability '%s' looks like an agent name, not a "
                "capability identifier. The planner should use the snake_case capability "
                "name (e.g. 'execute_code'), not the agent name (e.g. 'code-execution-agent').",
                task_id[:8], step_index + 1, capability,
            )

        # Discover target agent (cached)
        target_agent_id = step.get("target_agent_id") or await self._discover_best(capability)
        if not target_agent_id:
            # Provide a more actionable error when the capability looks like an agent name
            if re.search(r'(?:^|-)agent$|-service$', capability) and '_' not in capability:
                err = (
                    f"No capability named '{capability}' is registered. "
                    f"'{capability}' is an agent name, not a capability name — "
                    f"use the snake_case capability identifier instead "
                    f"(e.g. 'execute_code' not 'code-execution-agent')."
                )
            else:
                err = f"No available agent for capability '{capability}'"
            logger.warning("Workflow %s step %d: %s", task_id[:8], step_index + 1, err)

            # If hybrid execution is enabled, let the emergent runner discover a
            # suitable capability dynamically instead of immediately failing.
            if self._hybrid_execution and not step.get("_emergent_fallback"):
                logger.info(
                    "Workflow %s step %d/%d: no agent for '%s' — "
                    "falling back to emergent runner",
                    task_id[:8], step_index + 1, total, capability,
                )
                await asyncio.to_thread(
                    self._store.patch_step, task_id, step_index,
                    {"_emergent_fallback": True, "execution_mode": "emergent"},
                )
                asyncio.create_task(
                    self._dispatch_emergent_step(ws, task_id, step_index),
                    name=f"emergent-fallback-{task_id[:8]}-{step_index}",
                )
                return

            await asyncio.to_thread(self._store.advance_step, task_id, step_index, None)
            await self._handle_workflow_failure(
                ws=ws,
                task_id=task_id,
                workflow=workflow,
                step_index=step_index,
                step=step,
                err_msg=err,
                duration_ms=0,
            )
            return

        # Emit step_started
        await self._emit_workflow_event(ws, {
            "event":           "step_started",
            "task_id":         task_id,
            "step_id":         step_id,
            "step_order":      step_index + 1,
            "step_name":       step_name,
            "step_desc":       step.get("description", ""),
            "step_goal":       step_goal,
            "capability":      capability,
            "total_steps":     total,
            "workflow_status": "running",
            "execution_mode":  step.get("execution_mode", "strict"),
            "confidence":      step.get("confidence", 1.0),
        })

        # Build correlation and save before sending (avoids race if response arrives fast)
        req_id = str(uuid.uuid4())
        await asyncio.to_thread(self._store.save_correlation, req_id, task_id, step_index)

        ws_ref = self._current_ws
        if ws_ref is None:
            logger.warning("WS dropped before dispatching step %d of %s — "
                           "correlation saved, will retry on reconnect",
                           step_index + 1, task_id[:8])
            return

        await self._ws_send(ws_ref, _envelope(
            sender_id=self._agent_id,
            msg_type="task_request",
            payload={
                "capability": capability,
                "input_data": input_data,
                "timeout_ms": STEP_TIMEOUT_S * 1000,
            },
            recipient_id=target_agent_id,
            msg_id=req_id,
        ))

        logger.info(
            "Step %d/%d dispatched: workflow=%s  capability=%s  "
            "agent=%s  corr=%s",
            step_index + 1, total, task_id[:8],
            capability, target_agent_id[:8], req_id[:8],
        )

    # ── Step callback handler ──────────────────────────────────────────────────

    async def _on_step_response(
        self, ws, task_id: str, step_index: int, payload: dict
    ) -> None:
        """
        Called when an agent sends back a task_response for a dispatched step.
        Advances workflow state and dispatches the next step (or completes).
        """
        workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
        if not workflow:
            logger.error("on_step_response: unknown workflow %s", task_id)
            return

        steps       = workflow["steps"]
        step        = steps[step_index]
        step_id     = step["step_id"]
        step_name   = step["name"]
        capability  = step["capability"]
        total       = len(steps)
        success     = payload.get("success", False)
        output      = payload.get("output_data")
        error       = payload.get("error")
        duration_ms = payload.get("duration_ms", 0)

        if success:
            if isinstance(output, dict) and output.get("followup_request"):
                await self._handle_agent_followup_request(
                    ws=ws,
                    task_id=task_id,
                    workflow=workflow,
                    step_index=step_index,
                    step=step,
                    followup_request=output.get("followup_request"),
                )
                return

            await asyncio.to_thread(self._store.advance_step, task_id, step_index, output)
            completed = step_index + 1

            await self._emit_workflow_event(ws, {
                "event":           "step_completed",
                "task_id":         task_id,
                "step_id":         step_id,
                "step_order":      step_index + 1,
                "step_name":       step_name,
                "capability":      capability,
                "output_data":     output,
                "duration_ms":     duration_ms,
                "steps_completed": completed,
                "total_steps":     total,
                "workflow_status": "running",
            })

            # Notify user of step completion
            asyncio.create_task(
                self._send_step_update(workflow, step_index + 1, total, step_name, success=True),
                name=f"step-notify-{task_id[:8]}-{step_index}",
            )

            next_index = step_index + 1
            if next_index < total:
                logger.info("Workflow %s: step %d/%d done → dispatching step %d",
                            task_id[:8], step_index + 1, total, next_index + 1)
                asyncio.create_task(
                    self._dispatch_step(ws, task_id, next_index),
                    name=f"step-{task_id[:8]}-{next_index}",
                )
            else:
                # All steps completed
                await asyncio.to_thread(self._store.set_status, task_id, "completed")
                logger.info("Workflow %s completed (%d/%d steps)", task_id[:8], total, total)
                await self._emit_workflow_event(ws, {
                    "event":           "workflow_completed",
                    "task_id":         task_id,
                    "steps_completed": total,
                    "steps_failed":    0,
                    "total_steps":     total,
                    "workflow_status": "completed",
                })

                # Record completed workflow in planner's Cortex memory (best-effort)
                wf_title = workflow.get("title", task_id[:8])
                wf_goal  = (workflow.get("goal") or "")[:100].replace("\n", " ")
                asyncio.create_task(
                    self._write_cortex_entry(
                        "task-planner-agent",
                        "Patterns",
                        f"Completed '{wf_title}' ({total} steps) — goal: {wf_goal}",
                    ),
                    name=f"cortex-complete-{task_id[:8]}",
                )
                asyncio.create_task(
                    self._write_cortex_entry(
                        "__global__",
                        "Patterns",
                        f"Workflow '{wf_title}' succeeded ({total} steps) — goal: {wf_goal}",
                    ),
                    name=f"cortex-global-complete-{task_id[:8]}",
                )

                # Notify source channel of completion result
                updated_workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
                asyncio.create_task(
                    self._notify_source_of_completion(ws, task_id, updated_workflow or workflow),
                    name=f"notify-complete-{task_id[:8]}",
                )

        else:
            # Step failed
            err_msg = error or "Unknown error"
            # Extract replan context if the browser agent provided one (LLM retry exhausted)
            replan_context: dict | None = None
            if isinstance(output, dict) and output.get("replan_context"):
                replan_context = output["replan_context"]

            # If hybrid execution is enabled and this was a planned (strict) step that
            # hasn't already been retried via emergent, fall back to the emergent runner
            # before escalating to a full replan.
            if (
                self._hybrid_execution
                and step.get("execution_mode", "strict") != "emergent"
                and not step.get("_emergent_fallback")
            ):
                logger.info(
                    "Workflow %s step %d/%d: planned step failed (%s) — "
                    "falling back to emergent runner",
                    task_id[:8], step_index + 1, total, err_msg,
                )
                await asyncio.to_thread(
                    self._store.patch_step, task_id, step_index,
                    {"_emergent_fallback": True, "execution_mode": "emergent"},
                )
                asyncio.create_task(
                    self._dispatch_emergent_step(ws, task_id, step_index),
                    name=f"emergent-fallback-{task_id[:8]}-{step_index}",
                )
                return

            # No emergent fallback available — persist failure and replan
            await asyncio.to_thread(self._store.advance_step, task_id, step_index, None)
            await self._handle_workflow_failure(
                ws=ws,
                task_id=task_id,
                workflow=workflow,
                step_index=step_index,
                step=step,
                err_msg=err_msg,
                duration_ms=duration_ms,
                replan_context=replan_context,
            )

    async def _handle_workflow_failure(
        self,
        ws,
        task_id: str,
        workflow: dict,
        step_index: int,
        step: dict,
        err_msg: str,
        duration_ms: float,
        replan_context: dict | None = None,
    ) -> None:
        """Emit failure events, notify user, and prepare a re-plan approval request."""
        steps = workflow.get("steps", [])
        total = len(steps)
        step_name = step.get("name", f"Step {step_index + 1}")
        capability = step.get("capability", "")
        completed = step_index
        fail_detail = f"Step {step_index + 1} '{step_name}' failed: {err_msg}"

        await asyncio.to_thread(self._store.set_status, task_id, "failed", fail_detail)
        logger.warning(
            "Workflow %s: step %d/%d failed: %s",
            task_id[:8], step_index + 1, total, err_msg,
        )

        await self._emit_workflow_event(ws, {
            "event":           "step_failed",
            "task_id":         task_id,
            "step_id":         step.get("step_id"),
            "step_order":      step_index + 1,
            "step_name":       step_name,
            "capability":      capability,
            "error":           err_msg,
            "duration_ms":     duration_ms,
            "steps_failed":    1,
            "total_steps":     total,
            "workflow_status": "failed",
        })
        await self._emit_workflow_event(ws, {
            "event":           "workflow_failed",
            "task_id":         task_id,
            "steps_completed": completed,
            "steps_failed":    1,
            "total_steps":     total,
            "error":           err_msg,
            "workflow_status": "failed",
        })

        channel_id   = _clean_text(workflow.get("channel_id"))
        thread_id    = _clean_text(workflow.get("thread_id"))
        user_id      = _clean_text(workflow.get("user_id"))
        source       = _clean_text(workflow.get("source"))
        requester_id = _clean_text(workflow.get("requester_id"))
        goal         = _clean_text(workflow.get("goal"))

        failure_summary = (
            f"Step {step_index + 1}/{total} ({step_name}) failed: {err_msg}"
        )

        # Notify the source channel of the failure
        _is_llm_retry = replan_context and replan_context.get("failure_type") == "llm_unavailable"
        _agent_label  = (replan_context or {}).get("agent_name", step_name)
        if channel_id:
            if _is_llm_retry:
                notify_text = (
                    f"⚠️ LLM became unavailable for *{_agent_label}* at step {step_index + 1}. "
                    "Assessing whether to retry or find an alternative…"
                )
            else:
                notify_text = f"Workflow hit an error — {failure_summary}. Trying a revised plan…"
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text=notify_text,
            )
        elif source == "avatar" and requester_id:
            asyncio.create_task(
                self._notify_avatar(requester_id, f"Ran into a problem with that — {err_msg}. Let me try a different approach…"),
                name=f"notify-fail-avatar-{task_id[:8]}",
            )

        replan_count = int(workflow.get("replan_count") or 0)
        if replan_count >= self._max_replan_attempts or not self._planner:
            give_up_msg = (
                f"Re-plan limit reached ({replan_count}/{self._max_replan_attempts}). "
                "Unable to complete the task automatically."
            )
            logger.warning("Workflow %s: %s", task_id[:8], give_up_msg)
            if channel_id:
                await self._send_clarification_message(
                    channel_id=channel_id,
                    thread_ts=thread_id,
                    user_id=user_id,
                    text=give_up_msg,
                )
            elif source == "avatar" and requester_id:
                asyncio.create_task(
                    self._notify_avatar(requester_id, give_up_msg),
                    name=f"notify-giveup-avatar-{task_id[:8]}",
                )
            return

        # Auto-replan without user approval — include agent state context when available
        if replan_context:
            rc              = replan_context
            failure_type    = rc.get("failure_type", "unknown")
            agent_name      = rc.get("agent_name", capability)
            retry_attempts  = rc.get("retry_attempts", 0)
            last_error      = rc.get("last_error", err_msg)
            retry_possible  = rc.get("retry_possible", False)
            progress_summary = rc.get("progress_summary", "")
            steps_done      = rc.get("completed_steps", [])
            resume_ctx      = rc.get("resume_context", {})   # agent-specific resume hints

            steps_summary = (
                "\n".join(f"  - {s}" for s in steps_done[-10:])
                if steps_done else "  (none recorded)"
            )
            # Format any agent-specific resume hints (e.g. current_url for browser-agent)
            resume_lines = "\n".join(
                f"  {k}: {v}" for k, v in resume_ctx.items() if v
            )
            resume_section = (
                f"Agent resume context (include relevant fields in retry input_data):\n{resume_lines}\n"
                if resume_lines else ""
            )

            failure_context_goal = (
                f"{goal}\n\n"
                f"REPLAN CONTEXT — step {step_index + 1}/{total} "
                f"({step_name}, capability={capability}, agent={agent_name}):\n"
                f"  failure_type:    {failure_type}\n"
                f"  retry_attempts:  {retry_attempts}\n"
                f"  last_error:      {last_error}\n"
                f"  retry_possible:  {retry_possible}\n"
                + (f"  progress:        {progress_summary}\n" if progress_summary else "")
                + f"Steps completed before failure:\n{steps_summary}\n"
                + (f"{resume_section}" if resume_section else "")
                + "\nREPLAN INSTRUCTIONS: see system rules."
            )
        else:
            failure_context_goal = (
                f"{goal}\n\n"
                f"Previous attempt failed at step {step_index + 1}/{total} "
                f"({step_name}, capability={capability}).\n"
                f"Error: {err_msg}\n"
                "Create an alternative plan that avoids the failed path and still achieves the goal."
            )
        next_replan_count = replan_count + 1
        try:
            replan = await self._planner.plan(
                goal=failure_context_goal,
                requester_id=requester_id,
                channel_id=channel_id,
                thread_id=thread_id,
                user_id=user_id,
                delivery_channel=_clean_text(workflow.get("delivery_channel")),
                persona=_clean_text(workflow.get("persona")),
                summary_format=_clean_text(workflow.get("summary_format")),
                source=source,
            )
        except Exception as exc:
            logger.warning("Workflow %s re-plan generation failed: %s", task_id[:8], exc)
            err_text = f"Could not generate a revised plan: {exc}"
            if channel_id:
                await self._send_clarification_message(
                    channel_id=channel_id, thread_ts=thread_id, user_id=user_id, text=err_text,
                )
            elif source == "avatar" and requester_id:
                asyncio.create_task(
                    self._notify_avatar(requester_id, err_text),
                    name=f"notify-replan-err-avatar-{task_id[:8]}",
                )
            return

        replanned_steps = [s.to_dict() for s in replan.steps]
        _inject_slack_user_id(replanned_steps, user_id)

        await asyncio.to_thread(
            self._store.replace_workflow_plan,
            task_id,
            failure_context_goal,
            replan.title,
            replan.description,
            replanned_steps,
            next_replan_count,
        )
        await self._emit_workflow_event(ws, {
            "event":           "workflow_started",
            "task_id":         task_id,
            "title":           replan.title,
            "description":     replan.description,
            "goal":            failure_context_goal,
            "total_steps":     len(replanned_steps),
            "workflow_status": "running",
        })

        retry_msg = (
            f"Revised plan ready ({next_replan_count}/{self._max_replan_attempts}) "
            f"— {len(replanned_steps)} step(s). Continuing now…"
        )
        if channel_id:
            await self._send_clarification_message(
                channel_id=channel_id, thread_ts=thread_id, user_id=user_id, text=retry_msg,
            )
        elif source == "avatar" and requester_id:
            asyncio.create_task(
                self._notify_avatar(requester_id, retry_msg),
                name=f"notify-retry-avatar-{task_id[:8]}",
            )

        asyncio.create_task(
            self._dispatch_step(ws, task_id, 0),
            name=f"replan-step-{task_id[:8]}-0",
        )
        logger.info(
            "Workflow %s auto-replanned (attempt %d/%d): %d step(s)",
            task_id[:8], next_replan_count, self._max_replan_attempts, len(replanned_steps),
        )

    async def _handle_replan_approval_reply(
        self,
        ws,
        user_reply: str,
        pending: dict,
        channel_id: str,
        thread_id: str,
        user_id: str,
    ) -> tuple[dict | None, str | None]:
        decision = _parse_yes_no_reply(user_reply)
        task_id = pending.get("task_id", "")
        approval_id = pending.get("id", "")
        if decision is None:
            msg = (
                "Please reply with `yes` to continue with the revised plan "
                "or `no` to keep the workflow stopped."
            )
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text=msg,
            )
            return {
                "task_id": task_id,
                "status": "awaiting_replan_approval",
                "message": "Waiting for explicit yes/no confirmation",
            }, None

        await asyncio.to_thread(self._store.delete_pending_replan_approval, approval_id)

        if decision is False:
            await asyncio.to_thread(
                self._store.set_status,
                task_id,
                "failed",
                "User declined re-plan continuation",
            )
            await self._emit_workflow_event(ws, {
                "event": "workflow_failed",
                "task_id": task_id,
                "error": "User declined re-plan continuation",
                "workflow_status": "failed",
            })
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text="Understood. Workflow remains stopped. Share new instructions whenever you want to retry.",
            )
            return {
                "task_id": task_id,
                "status": "failed",
                "message": "User declined re-plan continuation",
            }, None

        try:
            steps = json.loads(pending.get("replanned_steps_json", "[]"))
            if not isinstance(steps, list) or not steps:
                raise ValueError("Stored re-plan is empty or invalid")
        except Exception as exc:
            return None, f"Failed to restore stored re-plan: {exc}"

        await asyncio.to_thread(
            self._store.replace_workflow_plan,
            task_id,
            pending.get("replanned_goal", ""),
            pending.get("replanned_title", ""),
            pending.get("replanned_description", ""),
            steps,
            int(pending.get("replan_count") or 0),
        )
        await self._emit_workflow_event(ws, {
            "event": "workflow_started",
            "task_id": task_id,
            "title": pending.get("replanned_title", ""),
            "description": pending.get("replanned_description", ""),
            "goal": pending.get("replanned_goal", ""),
            "total_steps": len(steps),
            "workflow_status": "running",
        })
        asyncio.create_task(
            self._dispatch_step(ws, task_id, 0),
            name=f"replan-step-{task_id[:8]}-0",
        )
        await self._send_clarification_message(
            channel_id=channel_id,
            thread_ts=thread_id,
            user_id=user_id,
            text="Approved. Continuing with the revised plan now.",
        )
        return {
            "task_id": task_id,
            "status": "running",
            "message": "Re-plan approved and resumed",
        }, None

    @staticmethod
    def _apply_followup_answer_to_input(
        base_input: dict[str, Any],
        field_name: str,
        question_id: str,
        answer: object,
    ) -> dict[str, Any]:
        updated = dict(base_input)
        if field_name:
            updated[field_name] = answer
        answers = dict(updated.get("followup_answers", {}) or {})
        answers[question_id] = answer
        updated["followup_answers"] = answers
        return updated

    async def _handle_agent_followup_request(
        self,
        ws,
        task_id: str,
        workflow: dict,
        step_index: int,
        step: dict,
        followup_request: object,
    ) -> None:
        """Route agent follow-up either from Cortex context or to the user."""
        if isinstance(followup_request, str):
            req = {"question": followup_request}
        elif isinstance(followup_request, dict):
            req = dict(followup_request)
        else:
            req = {}
        question = _clean_text(req.get("question"))
        if not question:
            await self._handle_workflow_failure(
                ws=ws,
                task_id=task_id,
                workflow=workflow,
                step_index=step_index,
                step=step,
                err_msg="Agent requested follow-up but did not provide a question",
                duration_ms=0,
            )
            return
        question_id = _clean_text(req.get("question_id")) or str(uuid.uuid4())
        field_name = _clean_text(req.get("field"))
        answer_format = _clean_text(req.get("answer_format")).lower() or "text"
        choices = req.get("choices") if isinstance(req.get("choices"), list) else []
        choices = [str(c) for c in choices if str(c).strip()]
        agent_capability = _clean_text(req.get("agent_capability"))
        agent_task = _clean_text(req.get("agent_task")) or question

        channel_id = _clean_text(workflow.get("channel_id"))
        thread_id = _clean_text(workflow.get("thread_id"))
        user_id = _clean_text(workflow.get("user_id"))
        requester_id = _clean_text(workflow.get("requester_id"))

        input_data = step.get("input_data", {})
        if not isinstance(input_data, dict):
            input_data = {}

        # Resolution priority:
        # 1. Cortex long-term memory (instant, no network call)
        # 2. A capable agent (e.g. gmail reader, SMS agent) — automatic, no human needed
        # 3. Ask the human via Slack (last resort)

        # ── 1. Cortex memory ──────────────────────────────────────────────
        if self._planner:
            try:
                memory = await self._planner.fetch_memory_context(user_id=user_id)
                resolved = await self._planner.answer_followup_from_memory(question, memory)
                if resolved.get("found") and resolved.get("confidence", 0.0) >= 0.65:
                    answer = resolved.get("answer")
                    patched = self._apply_followup_answer_to_input(
                        input_data, field_name, question_id, answer
                    )
                    await asyncio.to_thread(
                        self._store.update_step_input, task_id, step_index, patched
                    )
                    await asyncio.to_thread(
                        self._store.set_status, task_id, "running", None
                    )
                    await self._emit_workflow_event(ws, {
                        "event": "followup_resolved_from_context",
                        "task_id": task_id,
                        "step_id": step.get("step_id"),
                        "step_order": step_index + 1,
                        "question_id": question_id,
                        "field_name": field_name,
                        "workflow_status": "running",
                    })
                    asyncio.create_task(
                        self._dispatch_step(ws, task_id, step_index),
                        name=f"followup-retry-{task_id[:8]}-{step_index}",
                    )
                    return
            except Exception as exc:
                logger.warning("Failed to resolve follow-up from Cortex: %s", exc)

        # ── 2. Agent-assisted resolution ──────────────────────────────────
        if agent_capability:
            try:
                agent_answer = await self._resolve_followup_via_agent(
                    agent_capability, agent_task
                )
                if agent_answer:
                    patched = self._apply_followup_answer_to_input(
                        input_data, field_name, question_id, agent_answer
                    )
                    await asyncio.to_thread(
                        self._store.update_step_input, task_id, step_index, patched
                    )
                    await asyncio.to_thread(
                        self._store.set_status, task_id, "running", None
                    )
                    await self._emit_workflow_event(ws, {
                        "event": "followup_resolved_by_agent",
                        "task_id": task_id,
                        "step_id": step.get("step_id"),
                        "step_order": step_index + 1,
                        "question_id": question_id,
                        "field_name": field_name,
                        "agent_capability": agent_capability,
                        "workflow_status": "running",
                    })
                    # Store the resolved answer in global memory so future
                    # workflows can resolve the same question from Cortex
                    # without needing another agent or human in the loop.
                    if field_name and agent_answer:
                        _safe_answer = str(agent_answer)[:200]
                        asyncio.create_task(
                            self._write_cortex_entry(
                                "__global__",
                                "Facts",
                                f"{field_name}: {_safe_answer} (resolved via {agent_capability})",
                            ),
                            name=f"cortex-global-followup-{task_id[:8]}",
                        )
                    asyncio.create_task(
                        self._dispatch_step(ws, task_id, step_index),
                        name=f"followup-agent-retry-{task_id[:8]}-{step_index}",
                    )
                    return
                logger.info(
                    "Agent %r could not answer — falling back to user", agent_capability
                )
            except Exception as exc:
                logger.warning(
                    "Agent-assisted followup failed (%s): %s", agent_capability, exc
                )

        # ── 3. Ask the human ──────────────────────────────────────────────
        if not channel_id:
            await self._handle_workflow_failure(
                ws=ws,
                task_id=task_id,
                workflow=workflow,
                step_index=step_index,
                step=step,
                err_msg=f"Follow-up needed but no user channel available: {question}",
                duration_ms=0,
            )
            return

        choices_hint = f"\nOptions: {', '.join(choices)}" if choices else ""
        prompt = (
            f"I need one detail to continue workflow step {step_index + 1} "
            f"({step.get('name') or step.get('capability')}).\n"
            f"Question: {question}"
            f"{choices_hint}\n"
            "Please reply in this thread."
        )
        pending_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self._store.save_pending_followup,
            pending_id,
            task_id,
            step_index,
            _clean_text(step.get("step_id")),
            _clean_text(step.get("capability")),
            question_id,
            question,
            field_name,
            answer_format,
            json.dumps(choices),
            thread_id,
            channel_id,
            requester_id,
            user_id,
        )
        await asyncio.to_thread(
            self._store.set_status,
            task_id,
            "awaiting_followup",
            f"Awaiting follow-up answer: {question}",
        )
        await self._emit_workflow_event(ws, {
            "event": "followup_requested",
            "task_id": task_id,
            "step_id": step.get("step_id"),
            "step_order": step_index + 1,
            "question_id": question_id,
            "question": question,
            "field_name": field_name,
            "workflow_status": "awaiting_followup",
        })
        await self._send_clarification_message(
            channel_id=channel_id,
            thread_ts=thread_id,
            user_id=user_id,
            text=prompt,
        )

    async def _handle_followup_reply(
        self,
        ws,
        user_reply: str,
        pending: dict,
        channel_id: str,
        thread_id: str,
        user_id: str,
    ) -> tuple[dict | None, str | None]:
        task_id = _clean_text(pending.get("task_id"))
        step_index = int(pending.get("step_index") or 0)
        question_id = _clean_text(pending.get("question_id"))
        field_name = _clean_text(pending.get("field_name"))
        answer_format = _clean_text(pending.get("answer_format")).lower() or "text"
        try:
            choices = json.loads(pending.get("choices_json") or "[]")
            if not isinstance(choices, list):
                choices = []
        except Exception:
            choices = []
        answer, err = _normalise_followup_answer(
            user_reply, answer_format, [str(c) for c in choices]
        )
        if err:
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text=err,
            )
            return {
                "task_id": task_id,
                "status": "awaiting_followup",
                "message": "Waiting for valid follow-up answer",
            }, None

        workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
        if not workflow:
            await asyncio.to_thread(self._store.delete_pending_followup, pending.get("id"))
            return None, f"Workflow {task_id} not found for pending follow-up"
        steps = workflow.get("steps", [])
        if step_index < 0 or step_index >= len(steps):
            await asyncio.to_thread(self._store.delete_pending_followup, pending.get("id"))
            return None, f"Step index {step_index} out of range for follow-up"

        step = steps[step_index]
        input_data = step.get("input_data", {})
        if not isinstance(input_data, dict):
            input_data = {}
        patched = self._apply_followup_answer_to_input(
            input_data, field_name, question_id, answer
        )
        await asyncio.to_thread(
            self._store.update_step_input, task_id, step_index, patched
        )
        await asyncio.to_thread(
            self._store.delete_pending_followup, pending.get("id")
        )
        await asyncio.to_thread(
            self._store.set_status, task_id, "running", None
        )
        await self._emit_workflow_event(ws, {
            "event": "followup_answer_received",
            "task_id": task_id,
            "step_id": step.get("step_id"),
            "step_order": step_index + 1,
            "question_id": question_id,
            "field_name": field_name,
            "workflow_status": "running",
        })
        asyncio.create_task(
            self._dispatch_step(ws, task_id, step_index),
            name=f"followup-user-retry-{task_id[:8]}-{step_index}",
        )
        await self._send_clarification_message(
            channel_id=channel_id,
            thread_ts=thread_id,
            user_id=user_id,
            text="Thanks. Continuing with that answer.",
        )
        return {
            "task_id": task_id,
            "status": "running",
            "message": "Follow-up answered and step resumed",
        }, None

    async def _handle_memory_consent_reply(
        self,
        user_reply: str,
        pending: dict,
        channel_id: str,
        thread_id: str,
        user_id: str,
    ) -> tuple[dict | None, str | None]:
        """Handle yes/no reply for storing sensitive memory entries."""
        decision = _parse_yes_no_reply(user_reply)
        if decision is None:
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text="Please reply with `yes` or `no` for storing sensitive memory.",
            )
            return {
                "status": "awaiting_memory_consent",
                "message": "Waiting for explicit yes/no consent",
            }, None

        consent_id = _clean_text(pending.get("id"))
        await asyncio.to_thread(self._store.delete_pending_memory_consent, consent_id)

        if decision is False:
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text="Acknowledged. Sensitive information was not stored in Cortex.",
            )
            return {
                "status": "memory_consent_denied",
                "message": "Sensitive entries skipped",
            }, None

        try:
            entries = json.loads(pending.get("entries_json", "[]"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []

        if entries and self._planner:
            asyncio.create_task(
                self._planner.write_memory_entries(
                    user_entries=[e for e in entries if isinstance(e, dict)],
                    user_id=_clean_text(pending.get("user_id")),
                    planner_entries=None,
                ),
                name=f"cortex-sensitive-consent-{consent_id[:8]}",
            )
        await self._send_clarification_message(
            channel_id=channel_id,
            thread_ts=thread_id,
            user_id=user_id,
            text=(
                f"Acknowledged. Stored {len(entries)} sensitive entr"
                f"{'ies' if len(entries) != 1 else 'y'} in Cortex with your consent."
            ),
        )
        return {
            "status": "memory_consent_approved",
            "message": "Sensitive entries stored",
        }, None

    # ── Reconnect: resume stalled workflows ────────────────────────────────────

    async def _resume_in_progress_workflows(self, ws) -> None:
        """
        After a WS reconnect, find workflows still in 'running' state and
        re-dispatch their current step. Clears stale correlations first to
        avoid acting on responses from the previous connection.
        """
        deleted = await asyncio.to_thread(self._store.cleanup_stale_clarifications)
        if deleted:
            logger.info("Cleaned up %d stale pending clarification(s)", deleted)
        deleted_replans = await asyncio.to_thread(self._store.cleanup_stale_replan_approvals)
        if deleted_replans:
            logger.info("Cleaned up %d stale pending replan approval(s)", deleted_replans)
        deleted_followups = await asyncio.to_thread(self._store.cleanup_stale_followups)
        if deleted_followups:
            logger.info("Cleaned up %d stale pending follow-up(s)", deleted_followups)
        deleted_consents = await asyncio.to_thread(self._store.cleanup_stale_memory_consents)
        if deleted_consents:
            logger.info("Cleaned up %d stale pending memory consent(s)", deleted_consents)

        running = await asyncio.to_thread(self._store.list_running)
        if not running:
            return

        logger.info("Resuming %d in-progress workflow(s) after reconnect", len(running))
        for wf in running:
            task_id    = wf["task_id"]
            step_index = wf["current_step"]
            total      = wf["total_steps"]

            if step_index >= total:
                # Shouldn't happen, but guard
                await asyncio.to_thread(self._store.set_status, task_id, "completed")
                continue

            logger.info("Resuming workflow %s at step %d/%d",
                        task_id[:8], step_index + 1, total)
            # Clear stale correlation to avoid double-processing old callbacks
            await asyncio.to_thread(
                self._store.clear_stale_correlations, task_id, step_index
            )
            asyncio.create_task(
                self._dispatch_step(ws, task_id, step_index),
                name=f"resume-{task_id[:8]}-{step_index}",
            )

    # ── Capabilities: status + list ────────────────────────────────────────────

    async def _cap_get_workflow_status(
        self, input_data: dict
    ) -> tuple[dict | None, str | None]:
        task_id = _clean_text(input_data.get("task_id"))
        if not task_id:
            return None, "input_data.task_id is required"
        wf = await asyncio.to_thread(self._store.get_workflow, task_id)
        if wf is None:
            return None, f"Workflow {task_id} not found"
        return {"workflow": wf}, None

    async def _cap_list_workflows(
        self, input_data: dict
    ) -> tuple[dict | None, str | None]:
        limit = int(input_data.get("limit", 20))
        wfs = await asyncio.to_thread(self._store.list_workflows, limit)
        return {"workflows": wfs, "count": len(wfs)}, None

    # ── Capability: format_step_output ────────────────────────────────────────

    async def _cap_format_step_output(
        self, input_data: dict
    ) -> tuple[dict | None, str | None]:
        data = input_data.get("data")
        if data is None:
            return None, "input_data.data is required"
        capability_name = _clean_text(input_data.get("capability_name", ""))
        template_override = _clean_text(input_data.get("template", ""))
        text = await asyncio.to_thread(
            render_formatter, capability_name, data, template_override
        )
        return {"text": text}, None

    # ── Discovery (cached) ─────────────────────────────────────────────────────

    async def _discover_best(self, capability: str) -> Optional[str]:
        """
        Return agent_id of the best agent for *capability*.
        Results cached for DISCOVERY_CACHE_TTL_S seconds to minimise REST calls
        when consecutive steps share the same capability.
        """
        now = time.monotonic()
        if capability in self._discovery_cache:
            exp, agent_id = self._discovery_cache[capability]
            if now < exp:
                logger.debug("Discovery cache hit: %s → %s", capability, agent_id[:8])
                return agent_id

        try:
            resp = await self._http.get(
                f"{self._base}/api/v1/discover/best",
                params={"capability": capability},
            )
            if resp.status_code == 200:
                data = resp.json()
                agent_id = data.get("agent_id")
                if agent_id:
                    self._discovery_cache[capability] = (
                        now + DISCOVERY_CACHE_TTL_S,
                        agent_id,
                    )
                    logger.info("Discovered agent %s for capability '%s'",
                                agent_id[:8], capability)
                    return agent_id
            logger.warning("No agent for capability '%s' (status=%d)",
                           capability, resp.status_code)
        except Exception as exc:
            logger.error("Discovery request failed: %s", exc)
        return None

    # ── Hybrid execution helpers ───────────────────────────────────────────────

    def _apply_hybrid_settings(self, settings: dict) -> None:
        """Parse and apply planner_hybrid_execution / planner_emergent_max_turns."""
        if "planner_hybrid_execution" in settings:
            raw = str(settings["planner_hybrid_execution"]).lower().strip()
            self._hybrid_execution = raw in ("true", "1", "yes")
            logger.info("Hybrid execution → %s", self._hybrid_execution)
        if "planner_emergent_max_turns" in settings:
            try:
                self._emergent_max_turns = max(1, int(settings["planner_emergent_max_turns"]))
                logger.info("Emergent max turns → %d", self._emergent_max_turns)
            except (ValueError, TypeError):
                pass
        if "planner_emergent_only" in settings:
            raw = str(settings["planner_emergent_only"]).lower().strip()
            self._emergent_only = raw in ("true", "1", "yes")
            logger.info("Emergent-only execution → %s", self._emergent_only)

    def _make_emergent_runner(self) -> EmergentStepRunner:
        """Build an EmergentStepRunner wired to this client's infrastructure."""
        planner = self._planner
        return EmergentStepRunner(
            proxy_url=f"{self._base}/api/v1/llm/complete",
            agent_id=self._agent_id,
            model=planner._model if planner else "claude-haiku-4-5-20251001",
            provider=planner._provider if planner else "anthropic",
            discover_fn=self._discover_best,
            send_task_fn=self._emergent_send_task,
            list_capabilities_fn=self._list_capabilities_for_emergent,
            max_turns=self._emergent_max_turns,
        )

    async def _list_capabilities_for_emergent(self) -> list[dict]:
        """
        Fetch the live capability catalogue for the emergent runner.

        Uses the same two-step pattern as TaskPlanner.discover_capabilities():
          1. GET /api/v1/agents?active=true  — AgentSummary list (names/ids only)
          2. GET /api/v1/agents/{id}          — full AgentRecord with input schemas

        GET /api/v1/agents returns AgentSummary which only contains
        capability_names (strings), NOT the full schema objects.  Without the
        second fetch the emergent runner would see an empty catalogue and
        immediately return "no capabilities available".

        Reuses the planner's in-memory cache when it is fresh (< 60 s) to avoid
        duplicate REST calls when planning and emergent execution happen close
        together.
        """
        _HIDDEN = {"task-planner-agent", "task-executor-agent", "avatar-agent"}

        # ── Reuse planner's cache if fresh ────────────────────────────────────
        if self._planner is not None:
            import time as _time
            cache_age = _time.monotonic() - self._planner._caps_cache_time
            if cache_age < 60.0 and self._planner._caps_cache:
                caps: list[dict] = []
                for agent in self._planner._caps_cache:
                    if agent.get("name") in _HIDDEN:
                        continue
                    for cap in agent.get("capabilities", []):
                        caps.append(cap)
                logger.debug(
                    "EmergentRunner: reused planner capability cache "
                    "(age=%.1fs, %d cap(s))", cache_age, len(caps)
                )
                return caps

        # ── Fresh two-step fetch ──────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                list_resp = await http.get(
                    f"{self._base}/api/v1/agents",
                    params={"active": "true"},
                )
                if list_resp.status_code != 200:
                    logger.warning(
                        "list_capabilities_for_emergent: agents list returned %d",
                        list_resp.status_code,
                    )
                    return []

                agent_summaries = list_resp.json()
                caps = []
                for summary in agent_summaries:
                    if summary.get("name") in _HIDDEN:
                        continue
                    agent_id = summary.get("id")
                    if not agent_id:
                        continue
                    detail_resp = await http.get(f"{self._base}/api/v1/agents/{agent_id}")
                    if detail_resp.status_code == 200:
                        agent = detail_resp.json()
                        for cap in agent.get("capabilities", []):
                            caps.append(cap)

                logger.info(
                    "EmergentRunner: fetched capability catalogue — %d cap(s) from %d agent(s)",
                    len(caps), len(agent_summaries),
                )
                return caps
        except Exception as exc:
            logger.warning("list_capabilities_for_emergent failed: %s", exc)
        return []

    async def _emergent_send_task(
        self,
        target_agent_id: str,
        capability: str,
        input_data: dict,
    ) -> dict:
        """
        Send a task_request to *target_agent_id* and await its task_response.
        Used exclusively by EmergentStepRunner tool loop calls.
        """
        ws = self._current_ws
        if ws is None:
            raise RuntimeError("WebSocket not connected")

        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._emergent_pending[req_id] = fut

        try:
            await self._ws_send(ws, _envelope(
                sender_id=self._agent_id,
                msg_type="task_request",
                payload={
                    "capability": capability,
                    "input_data": input_data,
                    "timeout_ms": 120_000,
                },
                recipient_id=target_agent_id,
                msg_id=req_id,
            ))
            result_payload = await asyncio.wait_for(fut, timeout=130.0)
            output = result_payload.get("output_data") or {}
            if not result_payload.get("success"):
                output["error"] = result_payload.get("error", "unknown error")
            return output
        finally:
            self._emergent_pending.pop(req_id, None)

    async def _dispatch_emergent_step(
        self,
        ws,
        task_id: str,
        step_index: int,
    ) -> None:
        """
        Run a single workflow step using the LLM tool loop (EmergentStepRunner).
        On success, advance workflow and dispatch next step.
        On failure/bad output, fall back to _handle_workflow_failure for replan.
        """
        workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
        if not workflow:
            logger.error("dispatch_emergent_step: workflow %s not found", task_id)
            return

        steps   = workflow["steps"]
        outputs = workflow["outputs"]

        if step_index >= len(steps):
            logger.error("dispatch_emergent_step: step %d out of range for %s", step_index, task_id)
            return

        step       = steps[step_index]
        step_id    = step["step_id"]
        step_name  = step["name"]
        step_goal  = step.get("goal", step.get("description", ""))
        capability = step["capability"]
        total      = len(steps)

        hint_input = _resolve_step_refs(step.get("input_data", {}), outputs)
        if not isinstance(hint_input, dict):
            hint_input = {}

        # Emit step_started
        await self._emit_workflow_event(ws, {
            "event":           "step_started",
            "task_id":         task_id,
            "step_id":         step_id,
            "step_order":      step_index + 1,
            "step_name":       step_name,
            "step_desc":       step.get("description", ""),
            "step_goal":       step_goal,
            "capability":      capability,
            "total_steps":     total,
            "workflow_status": "running",
            "execution_mode":  "emergent",
        })

        t0 = asyncio.get_event_loop().time()
        runner = self._make_emergent_runner()
        output: Optional[dict] = None
        err_msg: str = ""

        try:
            output = await runner.run(
                step_name=step_name,
                step_goal=step_goal,
                hint_input=hint_input,
                prior_outputs=outputs,
            )
        except Exception as exc:
            err_msg = str(exc)
            logger.error(
                "EmergentRunner failed for workflow %s step %d: %s",
                task_id[:8], step_index + 1, exc,
            )

        duration_ms = (asyncio.get_event_loop().time() - t0) * 1000

        # Validate output with PlanTracker
        decision, reason = self._plan_tracker.validate(step, output)
        logger.info(
            "PlanTracker decision for workflow %s step %d: %s — %s",
            task_id[:8], step_index + 1, decision, reason,
        )

        if decision == "continue":
            await asyncio.to_thread(self._store.advance_step, task_id, step_index, output)
            completed = step_index + 1

            await self._emit_workflow_event(ws, {
                "event":           "step_completed",
                "task_id":         task_id,
                "step_id":         step_id,
                "step_order":      step_index + 1,
                "step_name":       step_name,
                "capability":      capability,
                "output_data":     output,
                "duration_ms":     duration_ms,
                "steps_completed": completed,
                "total_steps":     total,
                "workflow_status": "running",
                "execution_mode":  "emergent",
            })

            asyncio.create_task(
                self._send_step_update(workflow, step_index + 1, total, step_name, success=True),
                name=f"step-notify-{task_id[:8]}-{step_index}",
            )

            next_index = step_index + 1
            if next_index < total:
                logger.info("Workflow %s: emergent step %d/%d done → dispatching step %d",
                            task_id[:8], step_index + 1, total, next_index + 1)
                asyncio.create_task(
                    self._dispatch_step(ws, task_id, next_index),
                    name=f"step-{task_id[:8]}-{next_index}",
                )
            else:
                await asyncio.to_thread(self._store.set_status, task_id, "completed")
                logger.info("Workflow %s completed (emergent final step)", task_id[:8])
                await self._emit_workflow_event(ws, {
                    "event":           "workflow_completed",
                    "task_id":         task_id,
                    "steps_completed": total,
                    "steps_failed":    0,
                    "total_steps":     total,
                    "workflow_status": "completed",
                })
                updated_workflow = await asyncio.to_thread(self._store.get_workflow, task_id)
                asyncio.create_task(
                    self._notify_source_of_completion(ws, task_id, updated_workflow or workflow),
                    name=f"notify-complete-{task_id[:8]}",
                )

        elif decision in ("local_replan", "escalate"):
            replan_context = PlanTracker.replan_context(step, step_index, output, reason)
            await self._handle_workflow_failure(
                ws=ws,
                task_id=task_id,
                workflow=workflow,
                step_index=step_index,
                step=step,
                err_msg=err_msg or reason,
                duration_ms=duration_ms,
                replan_context=replan_context,
            )

    # ── Source completion notification ─────────────────────────────────────────

    async def _notify_source_of_completion(
        self, ws, task_id: str, workflow: dict
    ) -> None:
        """
        After all steps succeed, push the result back to the originating source.
        For avatar: sends talk_to_avatar to the requester agent.
        For Slack: the plan's final send_slack_message step already handles delivery.
        """
        source       = _clean_text(workflow.get("source"))
        requester_id = _clean_text(workflow.get("requester_id"))
        channel_id   = _clean_text(workflow.get("channel_id"))

        if not source and not requester_id:
            return

        # Build result text from the last successful step output
        outputs: list = workflow.get("outputs") or []

        result_text = ""
        for output in reversed(outputs):
            if not isinstance(output, dict):
                continue
            for key in ("result", "response", "message", "summary", "text", "content"):
                val = output.get(key)
                if isinstance(val, str) and val.strip():
                    result_text = val.strip()
                    break
            if result_text:
                break

        wf_title = workflow.get("title") or "Your task"
        if not result_text:
            result_text = f"'{wf_title}' completed successfully."

        if source == "avatar" and requester_id and not channel_id:
            await self._notify_avatar(requester_id, result_text)

    async def _notify_avatar(self, avatar_agent_id: str, message: str) -> None:
        """Send a talk_to_avatar task_request directly to the avatar agent."""
        ws_ref = self._current_ws
        if not ws_ref:
            logger.warning("Cannot notify avatar — WS not connected")
            return
        req_id = str(uuid.uuid4())
        try:
            await self._ws_send(ws_ref, _envelope(
                sender_id=self._agent_id,
                msg_type="task_request",
                payload={
                    "capability": "talk_to_avatar",
                    "input_data": {"message": message},
                    "timeout_ms": 30_000,
                },
                recipient_id=avatar_agent_id,
                msg_id=req_id,
            ))
            logger.info("Sent talk_to_avatar result to agent %s", avatar_agent_id[:8])
        except Exception as exc:
            logger.warning("Failed to notify avatar agent: %s", exc)

    # ── Plan summary + step status notifications ───────────────────────────────

    async def _send_plan_summary(
        self,
        channel_id: str,
        thread_id: str,
        user_id: str,
        title: str,
        description: str,
        steps: list[dict],
    ) -> None:
        """Send a formatted plan summary back to the originating channel."""
        if not channel_id and not user_id:
            return
        number_emoji = ["1️⃣","2️⃣","3️⃣","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
        step_lines = []
        for i, s in enumerate(steps):
            icon = number_emoji[i] if i < len(number_emoji) else f"{i+1}."
            name = s.get("name", f"Step {i+1}")
            desc = s.get("description", "")
            line = f"  {icon} *{name}*" + (f" — {desc}" if desc else "")
            step_lines.append(line)
        text = f"📋 *{title}*\n_{description}_\n\n" + "\n".join(step_lines)
        await self._send_clarification_message(
            channel_id=channel_id,
            thread_ts=thread_id,
            user_id=user_id,
            text=text,
        )

    async def _send_step_update(
        self,
        workflow: dict,
        step_num: int,
        total: int,
        step_name: str,
        success: bool,
    ) -> None:
        """Send a per-step status notification to the originating channel."""
        channel_id = _clean_text(workflow.get("channel_id"))
        thread_id  = _clean_text(workflow.get("thread_id"))
        user_id    = _clean_text(workflow.get("user_id"))
        source     = _clean_text(workflow.get("source"))
        requester_id = _clean_text(workflow.get("requester_id"))

        icon = "✅" if success else "❌"
        text = f"{icon} Step {step_num}/{total}: *{step_name}*"

        if channel_id:
            await self._send_clarification_message(
                channel_id=channel_id,
                thread_ts=thread_id,
                user_id=user_id,
                text=text,
            )
        elif source == "avatar" and requester_id:
            asyncio.create_task(
                self._notify_avatar(requester_id, text),
                name=f"step-notify-avatar-{step_num}",
            )

    # ── Clarification message ──────────────────────────────────────────────────

    async def _send_clarification_message(
        self,
        channel_id: str,
        text: str,
        thread_ts: str = "",
        user_id: str = "",
    ) -> Optional[str]:
        """
        Send a message directly (outside any workflow step).
        Routes to the built-in chat store when channel_id starts with "chat:",
        otherwise forwards to the slack-connector via WS.
        Returns message ts/id, or None on failure.
        """
        # ── Built-in chat channel ──────────────────────────────────────────────
        if channel_id.startswith("chat:"):
            reply_ctx = {
                "channel_type": "chat",
                "channel_id":   channel_id[5:] or "general",  # strip "chat:" prefix
                "thread_id":    thread_ts or "",
                "user_id":      user_id or "",
            }
            await self._notify_user(reply_ctx, text)
            return thread_ts or channel_id

        # ── Slack connector ────────────────────────────────────────────────────
        agent_id = await self._discover_best("send_slack_message")
        if not agent_id:
            logger.warning("No send_slack_message agent available for clarification")
            return None

        req_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_responses[req_id] = fut

        input_data: dict = {"channel": channel_id, "text": text}
        if thread_ts:
            input_data["thread_ts"] = thread_ts
        if user_id and not channel_id:
            input_data["user_id"] = user_id

        ws_ref = self._current_ws
        if ws_ref is None:
            self._pending_responses.pop(req_id, None)
            return None

        await self._ws_send(ws_ref, _envelope(
            sender_id=self._agent_id,
            msg_type="task_request",
            payload={"capability": "send_slack_message", "input_data": input_data},
            recipient_id=agent_id,
            msg_id=req_id,
        ))
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=30.0)
            return (result.get("output_data") or {}).get("ts")
        except (asyncio.TimeoutError, Exception) as exc:
            self._pending_responses.pop(req_id, None)
            logger.warning("Clarification message timed out / failed: %s", exc)
            return None

    async def _resolve_followup_via_agent(
        self,
        agent_capability: str,
        agent_task: str,
        timeout_s: float = 60.0,
    ) -> str | None:
        """
        Discover an agent that has *agent_capability*, dispatch *agent_task* to it,
        and return the text answer.  Returns None if no agent is available, the
        agent fails, or the call times out.

        This is called before falling back to asking the human via Slack, allowing
        agents like gmail/SMS readers to satisfy follow-up requests automatically.
        """
        agent_id = await self._discover_best(agent_capability)
        if not agent_id:
            logger.info(
                "followup: no agent for capability %r — will ask user", agent_capability
            )
            return None

        ws_ref = self._current_ws
        if ws_ref is None:
            return None

        req_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_responses[req_id] = fut

        await self._ws_send(ws_ref, _envelope(
            sender_id=self._agent_id,
            msg_type="task_request",
            payload={
                "capability": agent_capability,
                "input_data": {"task": agent_task},
                "timeout_ms": int(timeout_s * 1000),
            },
            recipient_id=agent_id,
            msg_id=req_id,
        ))
        logger.info(
            "followup: dispatched to agent %s (cap=%s) corr=%s",
            agent_id[:8], agent_capability, req_id[:8],
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(fut), timeout=timeout_s)
            if not result.get("success"):
                logger.info(
                    "followup agent %s returned failure: %s",
                    agent_capability, result.get("error"),
                )
                return None
            output = result.get("output_data") or {}
            # Accept any common text-output field name
            answer = (
                output.get("result")
                or output.get("answer")
                or output.get("code")
                or output.get("text")
                or output.get("content")
                or output.get("summary")
            )
            if answer is None and output:
                answer = str(output)
            if answer:
                logger.info(
                    "followup resolved by agent %s: %r", agent_capability, str(answer)[:80]
                )
            return str(answer) if answer else None
        except (asyncio.TimeoutError, Exception) as exc:
            self._pending_responses.pop(req_id, None)
            logger.warning(
                "followup agent %s timed out / failed: %s", agent_capability, exc
            )
            return None

    # ── Cortex memory helpers ──────────────────────────────────────────────────

    async def _write_cortex_entry(
        self, agent_namespace: str, category: str, content: str
    ) -> None:
        """Write a single entry to a Cortex memory namespace via REST (best-effort)."""
        try:
            resp = await self._http.post(
                f"{self._base}/api/v1/cortex/agents/{agent_namespace}/entries",
                json={"category": category, "content": content},
            )
            if resp.status_code not in (200, 201, 204):
                logger.debug(
                    "Cortex write returned %d for %s", resp.status_code, agent_namespace
                )
        except Exception as exc:
            logger.debug("Cortex write failed (%s): %s", agent_namespace, exc)

    # ── workflow_event emission ────────────────────────────────────────────────

    async def _emit_workflow_event(self, ws, payload: dict) -> None:
        try:
            await self._ws_send(ws, self._msg("workflow_event", payload))
        except Exception as exc:
            logger.warning("Failed to emit workflow_event: %s", exc)

    # ── User notification helper (HTTP) ────────────────────────────────────────

    async def _notify_user(self, reply_context: dict, message: str) -> None:
        """Send a status message back to the originating user channel via orchestrator notify API."""
        if not reply_context or not message:
            return
        try:
            await self._http.post(
                f"{self._base}/api/v1/notify",
                json={**reply_context, "message": message, "sender_agent_id": self._agent_id},
                timeout=10.0,
            )
        except Exception as exc:
            logger.warning("_notify_user failed: %s", exc)

    # ── Status update ──────────────────────────────────────────────────────────

    async def _send_status_update(self, ws) -> None:
        await self._ws_send(ws, self._msg(
            "status_update",
            {
                "status":       self._status,
                "current_load": min(self._active_tasks / 5.0, 1.0),
                "active_tasks": self._active_tasks,
                "metrics":      self._metrics(),
            },
        ))

    # ── Graceful shutdown ──────────────────────────────────────────────────────

    async def _graceful_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown signal — draining …")
        self._status = "draining"

        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while self._active_tasks > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.5)

        if self._agent_id:
            try:
                await self._http.delete(f"{self._base}/api/v1/agents/{self._agent_id}")
                logger.info("Deregistered from orchestrator.")
            except Exception as exc:
                logger.warning("Deregister failed: %s", exc)

        self._store.close()
        await self._http.aclose()
        logger.info("Shutdown complete.")

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _ws_send(self, ws, msg_str: str) -> None:
        msg   = json.loads(msg_str)
        mtype = msg.get("type", "?")
        noisy = mtype in ("heartbeat", "status_update")
        (logger.debug if noisy else logger.info)(
            "→ [%s] to=%s", mtype, msg.get("recipient_id") or "orchestrator"
        )
        try:
            await ws.send(msg_str)
        except websockets.exceptions.ConnectionClosed:
            raise  # propagate → heartbeat loop exits → asyncio.gather raises → reconnect
        except Exception as exc:
            logger.warning("WS send failed: %s", exc)

    def _msg(
        self,
        msg_type: str,
        payload: dict,
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        return _envelope(self._agent_id, msg_type, payload, recipient_id, correlation_id)

    def _metrics(self) -> dict:
        n = self._tasks_completed + self._tasks_failed
        return {
            "tasks_completed":      self._tasks_completed,
            "tasks_failed":         self._tasks_failed,
            "avg_response_time_ms": round(self._total_duration_ms / n, 1) if n else 0.0,
            "uptime_seconds":       round(time.monotonic() - self._start_time, 1),
        }
