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

LLM optimisations
─────────────────
- Capability list cached 60 s — multiple plans share one REST fetch
- Discovery (best agent per capability) cached 30 s per capability key
- Exactly ONE anthropic.messages.create call per plan_task request
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

from models import WorkflowPlan
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

_REF_RE = re.compile(r"\{\{steps\[(\d+)\]\.output\.([^}]+)\}\}")


def _resolve_step_refs(value: Any, outputs: list[Optional[dict]]) -> Any:
    """Recursively substitute ``{{steps[N].output.field}}`` in *value*."""

    def _resolve_str(s: str) -> Any:
        full = _REF_RE.fullmatch(s)
        if full:
            idx, path = int(full.group(1)), full.group(2).split(".")
            node: Any = (outputs[idx] or {}) if idx < len(outputs) else {}
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            return node

        def _sub(m: re.Match) -> str:
            idx, path = int(m.group(1)), m.group(2).split(".")
            node: Any = (outputs[idx] or {}) if idx < len(outputs) else {}
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            return str(node) if node is not None else m.group(0)

        return _REF_RE.sub(_sub, s)

    if isinstance(value, str):
        return _resolve_str(value)
    if isinstance(value, dict):
        return {k: _resolve_step_refs(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_step_refs(v, outputs) for v in value]
    return value


# ── Agent identity ─────────────────────────────────────────────────────────────

AGENT_NAME        = "task-planner-agent"
AGENT_VERSION     = "2.0.0"
AGENT_DESCRIPTION = (
    "Accepts a natural-language goal, discovers available agent capabilities, "
    "generates a structured workflow plan with one LLM call, persists state in "
    "SQLite, and drives step-by-step execution directly — dispatching each step "
    "to the appropriate agent and resuming on callback."
)

REGISTRATION_PAYLOAD: dict = {
    "name":        AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "version":     AGENT_VERSION,
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
            "key": "anthropic_api_key",
            "label": "Anthropic API Key",
            "type": "secret",
            "required": False,
            "description": "API key for Claude. Falls back to ANTHROPIC_API_KEY env var.",
        }
    ],
}

# ── Constants ──────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_S:   int   = 15
MAX_BACKOFF_S:          int   = 60
DRAIN_TIMEOUT_S:        int   = 30
STEP_TIMEOUT_S:         float = 300.0   # 5 min per step
DISCOVERY_CACHE_TTL_S:  float = 30.0   # cache best-agent per capability


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

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._graceful_shutdown()))

        self._store.open()
        await self._register()
        self._planner = TaskPlanner(
            orchestrator_base_url=self._base,
            api_key=self._common_settings.get("anthropic_api_key"),
        )
        await self._connect_loop()

    # ── Registration ───────────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": _stable_agent_id()}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id      = data["agent_id"]
        self._ws_url        = data["ws_url"]
        self._common_settings = data.get("common_settings", {})
        logger.info("Registered — agent_id=%s", self._agent_id)
        api_key = self._common_settings.get("anthropic_api_key")
        if api_key and self._planner:
            self._planner._api_key = api_key

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

            # ── Normal pending-response path ───────────────────────────────
            if corr in self._pending_responses:
                fut = self._pending_responses.pop(corr)
                if not fut.done():
                    fut.set_result(payload)

        elif mtype == "settings_push":
            logger.info("Settings pushed: %d key(s)", len(payload))
            self._common_settings.update(payload)
            api_key = payload.get("anthropic_api_key")
            if api_key and self._planner:
                self._planner._api_key = api_key

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
        goal = input_data.get("goal", "").strip()
        if not goal:
            return None, "input_data.goal is required"
        if not self._planner:
            return None, "Planner not initialised"

        # ── Single LLM call ──────────────────────────────────────────────────
        try:
            plan: WorkflowPlan = await self._planner.plan(goal, requester_id)
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            return None, f"Planning failed: {exc}"

        steps = [s.to_dict() for s in plan.steps]
        logger.info("Plan ready: task_id=%s  title=%r  steps=%d",
                    plan.task_id, plan.title, len(steps))

        # ── Persist ──────────────────────────────────────────────────────────
        await asyncio.to_thread(
            self._store.create_workflow,
            plan.task_id, goal, plan.title, plan.description,
            requester_id, steps,
        )
        await asyncio.to_thread(self._store.set_status, plan.task_id, "running")

        # ── Emit workflow_started ─────────────────────────────────────────────
        await self._emit_workflow_event(ws, {
            "event":           "workflow_started",
            "task_id":         plan.task_id,
            "title":           plan.title,
            "description":     plan.description,
            "goal":            goal,
            "total_steps":     len(steps),
            "workflow_status": "running",
        })

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
        step_id    = step["step_id"]
        step_name  = step["name"]
        step_goal  = step.get("goal", step.get("description", ""))
        capability = step["capability"]
        total      = len(steps)

        # Resolve {{steps[N].output.field}} references
        input_data = _resolve_step_refs(step.get("input_data", {}), outputs)

        # Discover target agent (cached)
        target_agent_id = step.get("target_agent_id") or await self._discover_best(capability)
        if not target_agent_id:
            err = f"No available agent for capability '{capability}'"
            logger.warning("Workflow %s step %d: %s", task_id[:8], step_index + 1, err)
            await asyncio.to_thread(self._store.advance_step, task_id, step_index, None)
            await asyncio.to_thread(self._store.set_status, task_id, "failed", err)
            await self._emit_workflow_event(ws, {
                "event":          "step_failed",
                "task_id":        task_id,
                "step_id":        step_id,
                "step_order":     step_index + 1,
                "step_name":      step_name,
                "capability":     capability,
                "error":          err,
                "total_steps":    total,
                "workflow_status": "failed",
            })
            await self._emit_workflow_event(ws, {
                "event":           "workflow_failed",
                "task_id":         task_id,
                "steps_completed": step_index,
                "steps_failed":    1,
                "total_steps":     total,
                "error":           err,
                "workflow_status": "failed",
            })
            return

        # Emit step_started
        await self._emit_workflow_event(ws, {
            "event":          "step_started",
            "task_id":        task_id,
            "step_id":        step_id,
            "step_order":     step_index + 1,
            "step_name":      step_name,
            "step_desc":      step.get("description", ""),
            "step_goal":      step_goal,
            "capability":     capability,
            "total_steps":    total,
            "workflow_status": "running",
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

        else:
            # Step failed — persist and abort
            await asyncio.to_thread(self._store.advance_step, task_id, step_index, None)
            err_msg = error or "Unknown error"
            await asyncio.to_thread(
                self._store.set_status, task_id, "failed",
                f"Step {step_index + 1} '{step_name}' failed: {err_msg}",
            )
            logger.warning("Workflow %s: step %d/%d failed: %s",
                           task_id[:8], step_index + 1, total, err_msg)

            await self._emit_workflow_event(ws, {
                "event":          "step_failed",
                "task_id":        task_id,
                "step_id":        step_id,
                "step_order":     step_index + 1,
                "step_name":      step_name,
                "capability":     capability,
                "error":          err_msg,
                "duration_ms":    duration_ms,
                "steps_failed":   1,
                "total_steps":    total,
                "workflow_status": "failed",
            })
            await self._emit_workflow_event(ws, {
                "event":           "workflow_failed",
                "task_id":         task_id,
                "steps_completed": step_index,
                "steps_failed":    1,
                "total_steps":     total,
                "error":           err_msg,
                "workflow_status": "failed",
            })

    # ── Reconnect: resume stalled workflows ────────────────────────────────────

    async def _resume_in_progress_workflows(self, ws) -> None:
        """
        After a WS reconnect, find workflows still in 'running' state and
        re-dispatch their current step. Clears stale correlations first to
        avoid acting on responses from the previous connection.
        """
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
        task_id = input_data.get("task_id", "").strip()
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

    # ── workflow_event emission ────────────────────────────────────────────────

    async def _emit_workflow_event(self, ws, payload: dict) -> None:
        try:
            await self._ws_send(ws, self._msg("workflow_event", payload))
        except Exception as exc:
            logger.warning("Failed to emit workflow_event: %s", exc)

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
