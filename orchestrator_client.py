"""
orchestrator_client.py — WebSocket + HTTP client for the task-planner-agent.

Registers the agent with the orchestrator, handles incoming plan_task
requests, and optionally forwards the generated plan to a task-executor-agent.
"""
from __future__ import annotations

import asyncio
import json
import logging
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

logger = logging.getLogger(__name__)

# ── Stable agent identity ──────────────────────────────────────────────────

_AGENT_ID_FILE = Path(".agent_id")


def _stable_agent_id() -> str:
    if _AGENT_ID_FILE.exists():
        return _AGENT_ID_FILE.read_text().strip()
    new_id = str(uuid.uuid4())
    _AGENT_ID_FILE.write_text(new_id)
    logger.info("Generated new stable agent ID: %s → %s", new_id, _AGENT_ID_FILE)
    return new_id


# ── Agent identity ─────────────────────────────────────────────────────────

AGENT_NAME = "task-planner-agent"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = (
    "Accepts a natural-language task description, discovers available agent "
    "capabilities, and uses an LLM to generate a structured workflow execution "
    "plan. Forwards the plan to the task-executor-agent automatically."
)

REGISTRATION_PAYLOAD: dict = {
    "name": AGENT_NAME,
    "description": AGENT_DESCRIPTION,
    "version": AGENT_VERSION,
    "capabilities": [
        {
            "name": "plan_task",
            "description": (
                "Accept a natural-language goal, discover available agent "
                "capabilities, and produce a structured workflow plan using an LLM. "
                "The plan is automatically forwarded to the task-executor-agent."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Natural-language description of the task to plan.",
                    },
                    "auto_execute": {
                        "type": "boolean",
                        "description": (
                            "If true (default), the plan is forwarded to the "
                            "task-executor-agent immediately after planning."
                        ),
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
                    "steps":       {"type": "array"},
                    "status":      {"type": "string"},
                },
            },
            "tags": ["planning", "llm", "workflow"],
        }
    ],
    "tags": ["planner", "llm", "workflow"],
    "metadata": {
        "language": "python",
        "llm_model": "claude-sonnet-4-6",
    },
    "required_settings": [
        {
            "key": "anthropic_api_key",
            "label": "Anthropic API Key",
            "type": "secret",
            "required": False,
            "description": "API key for Claude models used in planning. Falls back to ANTHROPIC_API_KEY env var.",
        }
    ],
}

# ── Constants ──────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL_S: int = 15
MAX_BACKOFF_S: int = 60
DRAIN_TIMEOUT_S: int = 30
DISPATCH_TIMEOUT_S: float = 120.0


# ── Helpers ────────────────────────────────────────────────────────────────

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


# ── Main client ────────────────────────────────────────────────────────────

class OrchestratorClient:
    """
    Registers the task-planner-agent with the orchestrator and handles
    incoming plan_task requests.
    """

    def __init__(self, orchestrator_url: str = "http://localhost:8000") -> None:
        self._base = orchestrator_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=30)

        self._agent_id: str = ""
        self._ws_url: str = ""

        self._status: str = "starting"
        self._active_tasks: int = 0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._total_duration_ms: float = 0.0
        self._start_time: float = time.monotonic()

        self._shutting_down: bool = False
        self._current_ws: Any = None
        self._pending_responses: dict[str, asyncio.Future] = {}

        self._planner: Optional[TaskPlanner] = None
        self._common_settings: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self._graceful_shutdown()))

        await self._register()
        self._planner = TaskPlanner(
            orchestrator_base_url=self._base,
            api_key=self._common_settings.get("anthropic_api_key"),
        )
        await self._connect_loop()

    # ── Registration ───────────────────────────────────────────────────────

    async def _register(self) -> None:
        url = f"{self._base}/api/v1/agents/register"
        logger.info("Registering with orchestrator at %s …", url)
        payload = {**REGISTRATION_PAYLOAD, "id": _stable_agent_id()}
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._agent_id = data["agent_id"]
        self._ws_url = data["ws_url"]
        self._common_settings = data.get("common_settings", {})
        logger.info("Registered — agent_id=%s  ws=%s", self._agent_id, self._ws_url)

        # Re-initialise planner with fresh API key from common settings
        api_key = self._common_settings.get("anthropic_api_key")
        if api_key and self._planner:
            self._planner._api_key = api_key

    # ── WebSocket loop ─────────────────────────────────────────────────────

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
        logger.info("WebSocket session active")
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

    # ── Heartbeat ─────────────────────────────────────────────────────────

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

    # ── Receive loop ───────────────────────────────────────────────────────

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
        mtype = msg.get("type", "")
        payload = msg.get("payload", {})

        if mtype == "task_request":
            asyncio.create_task(self._handle_incoming_task(ws, msg))

        elif mtype == "task_response":
            corr = msg.get("correlation_id")
            if corr and corr in self._pending_responses:
                fut = self._pending_responses.pop(corr)
                if not fut.done():
                    fut.set_result(payload)

        elif mtype == "settings_push":
            logger.info("Settings pushed from orchestrator: %d keys", len(payload))
            self._common_settings.update(payload)
            api_key = payload.get("anthropic_api_key")
            if api_key and self._planner:
                self._planner._api_key = api_key

        elif mtype == "error":
            logger.error(
                "Orchestrator error [%s]: %s",
                payload.get("code"),
                payload.get("detail"),
            )
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

    # ── Incoming task handling ─────────────────────────────────────────────

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
            else:
                output, error = None, f"Unknown capability: {capability!r}"

            duration_ms = (time.monotonic() - t0) * 1000

            if error:
                self._tasks_failed += 1
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": False, "error": error, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))
            else:
                self._tasks_completed += 1
                self._total_duration_ms += duration_ms
                await self._ws_send(ws, self._msg(
                    "task_response",
                    {"success": True, "output_data": output, "duration_ms": round(duration_ms, 1)},
                    recipient_id=sender_id,
                    correlation_id=req_id,
                ))

        except Exception as exc:
            duration_ms = (time.monotonic() - t0) * 1000
            self._tasks_failed += 1
            logger.exception("Unhandled error in capability %r", capability)
            await self._ws_send(ws, self._msg(
                "task_response",
                {"success": False, "error": str(exc), "duration_ms": round(duration_ms, 1)},
                recipient_id=sender_id,
                correlation_id=req_id,
            ))

        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
            self._status = "draining" if self._shutting_down else (
                "busy" if self._active_tasks else "available"
            )
            await self._send_status_update(ws)

    async def _cap_plan_task(
        self, input_data: dict, requester_id: str, ws
    ) -> tuple[dict | None, str | None]:
        goal = input_data.get("goal", "").strip()
        if not goal:
            return None, "input_data.goal is required"

        auto_execute = input_data.get("auto_execute", True)

        if not self._planner:
            return None, "Planner not initialised"

        try:
            plan: WorkflowPlan = await self._planner.plan(goal, requester_id)
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            return None, f"Planning failed: {exc}"

        logger.info(
            "Workflow plan ready: task_id=%s  steps=%d  title=%r",
            plan.task_id,
            len(plan.steps),
            plan.title,
        )

        plan_dict = plan.to_dict()

        if auto_execute and plan.steps:
            # Discover the task-executor-agent
            executor_id = await self._discover_executor()
            if executor_id:
                logger.info(
                    "Forwarding plan %s to executor %s …", plan.task_id, executor_id
                )
                asyncio.create_task(
                    self._forward_plan(plan_dict, executor_id, ws),
                    name=f"fwd-plan-{plan.task_id[:8]}",
                )
                plan_dict["forwarded_to"] = executor_id
                plan_dict["status"] = "forwarded"
            else:
                logger.warning(
                    "No task-executor-agent available — plan %s not forwarded",
                    plan.task_id,
                )
                plan_dict["status"] = "pending_executor"
        else:
            plan_dict["status"] = "planned"

        return plan_dict, None

    async def _discover_executor(self) -> Optional[str]:
        """Return the agent_id of the best task-executor-agent."""
        try:
            resp = await self._http.get(
                f"{self._base}/api/v1/discover/best",
                params={"capability": "execute_workflow"},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("agent_id")
        except Exception as exc:
            logger.warning("Discovery failed: %s", exc)
        return None

    async def _forward_plan(self, plan_dict: dict, executor_id: str, ws) -> None:
        """Send the plan to the task-executor-agent as a task_request."""
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending_responses[req_id] = fut

        try:
            await self._ws_send(ws, _envelope(
                sender_id=self._agent_id,
                msg_type="task_request",
                payload={
                    "capability": "execute_workflow",
                    "input_data": {"plan": plan_dict},
                    "timeout_ms": DISPATCH_TIMEOUT_S * 1000,
                },
                recipient_id=executor_id,
                msg_id=req_id,
            ))

            resp_payload = await asyncio.wait_for(
                asyncio.shield(fut), timeout=DISPATCH_TIMEOUT_S
            )
            if resp_payload.get("success"):
                logger.info(
                    "Executor accepted plan %s", plan_dict.get("task_id")
                )
            else:
                logger.warning(
                    "Executor rejected plan %s: %s",
                    plan_dict.get("task_id"),
                    resp_payload.get("error"),
                )
        except asyncio.TimeoutError:
            logger.warning(
                "Executor did not respond for plan %s (timeout)", plan_dict.get("task_id")
            )
        except Exception as exc:
            logger.error("Failed to forward plan: %s", exc)
        finally:
            self._pending_responses.pop(req_id, None)

    # ── Status update ──────────────────────────────────────────────────────

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

    # ── Graceful shutdown ──────────────────────────────────────────────────

    async def _graceful_shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown signal received — draining …")
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

        await self._http.aclose()
        logger.info("Shutdown complete.")

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _ws_send(self, ws, msg_str: str) -> None:
        msg   = json.loads(msg_str)
        mtype = msg.get("type", "?")
        noisy = mtype in ("heartbeat", "status_update")
        log   = logger.debug if noisy else logger.info
        log("→ [%s] to=%s", mtype, msg.get("recipient_id") or "orchestrator")
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
            "avg_response_time_ms": (
                round(self._total_duration_ms / n, 1) if n else 0.0
            ),
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }
