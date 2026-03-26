"""
emergent_runner.py — LLM tool-loop runner for low-confidence workflow steps.

When the planner marks a step as execution_mode="emergent", this runner takes
over instead of the normal strict dispatch.  It gives the LLM:

  • The step goal and hint input_data
  • The prior step outputs (for chaining context)
  • The full live capability catalogue (filtered to available agents)

The LLM then iteratively calls capabilities (tool loop) until it either:
  • Produces a final answer  →  {"done": true, "output": {...}}
  • Exhausts max_turns       →  raises RuntimeError
  • Hits a hard error        →  raises RuntimeError

The runner communicates with agents through the orchestrator exactly like the
strict path does (task_request / task_response), so existing agent protocol is
unchanged.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Load system prompt from prompts/ directory ────────────────────────────────

_PROMPT_PATH = Path(__file__).parent / "prompts" / "emergent_prompt.md"

def _load_system_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not load emergent_prompt.md: %s — using built-in fallback", exc)
        return (
            "You are an adaptive task executor. Call capabilities to achieve the goal. "
            'Output {"action":"tool_call",...} or {"action":"done",...} or {"action":"ask",...}. '
            "JSON only — no extra text."
        )

_MAX_TURNS_DEFAULT = 6
_STEP_TIMEOUT_S    = 120.0   # per-capability call inside the tool loop


class EmergentStepRunner:
    """
    Runs a single workflow step as an LLM-driven tool loop.

    Parameters
    ----------
    proxy_url:
        URL of the orchestrator LLM proxy  (``/api/v1/llm/complete``).
    agent_id:
        This planner's agent ID (sent as ``X-Agent-Id``).
    model / provider:
        LLM to use for the tool loop.
    discover_fn:
        Async callable ``(capability: str) -> Optional[str]`` that returns the
        best agent_id for a capability.
    send_task_fn:
        Async callable that sends a ``task_request`` and awaits the response.
        Signature: ``(agent_id, capability, input_data, timeout_ms) -> dict``
        The dict is the raw ``task_response`` payload.
    list_capabilities_fn:
        Async callable ``() -> list[dict]`` returning the current live
        capability catalogue from the orchestrator.
    max_turns:
        Maximum LLM+tool iterations before giving up.
    """

    def __init__(
        self,
        proxy_url: str,
        agent_id: str,
        model: str,
        provider: str,
        discover_fn: Callable,
        send_task_fn: Callable,
        list_capabilities_fn: Callable,
        max_turns: int = _MAX_TURNS_DEFAULT,
    ) -> None:
        self._proxy_url = proxy_url
        self._agent_id  = agent_id
        self._model     = model
        self._provider  = provider
        self._discover  = discover_fn
        self._send_task = send_task_fn
        self._list_caps = list_capabilities_fn
        self._max_turns = max(1, max_turns)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        step_name: str,
        step_goal: str,
        hint_input: dict,
        prior_outputs: list[Optional[dict]],
    ) -> dict[str, Any]:
        """
        Execute one emergent step.  Returns the final output dict on success.
        Raises ``RuntimeError`` if the goal cannot be achieved within max_turns.
        """
        caps = await self._list_caps()
        cap_catalogue = _format_catalogue(caps)

        context_parts: list[str] = []
        if prior_outputs:
            context_parts.append(_format_prior_outputs(prior_outputs))
        if hint_input:
            context_parts.append(f"Hint input from planner:\n{json.dumps(hint_input, indent=2)}")

        context_block = "\n\n".join(context_parts)

        user_prompt = (
            f"Goal: {step_goal}\n\n"
            f"{context_block}\n\n"
            f"Available capabilities:\n{cap_catalogue}"
        ).strip()

        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        tool_results: list[dict] = []

        logger.info("EmergentRunner: starting step %r (max_turns=%d)", step_name, self._max_turns)

        system_prompt = _load_system_prompt()

        for turn in range(self._max_turns):
            response_text = await self._llm_call(messages, system_prompt)
            logger.info("EmergentRunner LLM request: %r ", messages)

            parsed = _parse_json(response_text)
            if parsed is None:
                logger.warning("EmergentRunner turn %d: LLM returned unparseable JSON — retrying", turn + 1)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "Your response was not valid JSON. Please output only a JSON object."})
                continue

            action = parsed.get("action")

            if action == "done":
                output = parsed.get("output") or {}
                summary = parsed.get("summary", "")
                logger.info("EmergentRunner: step %r done after %d turn(s) — %s", step_name, turn + 1, summary)
                if not isinstance(output, dict):
                    output = {"result": output}
                output["_emergent_summary"] = summary
                output["_emergent_turns"]   = turn + 1
                return output

            if action == "ask":
                questions = parsed.get("questions") or []
                reason    = parsed.get("reason", "")
                logger.info(
                    "EmergentRunner turn %d: LLM asking follow-up (%d question(s)) — %s",
                    turn + 1, len(questions), reason,
                )
                # Surface questions via the ask_user capability if available,
                # otherwise encode them as a structured output so PlanTracker
                # can decide whether to escalate or local_replan.
                ask_result = await self._call_capability(
                    "ask_user",
                    {"questions": questions, "reason": reason},
                )
                messages.append({"role": "assistant", "content": response_text})
                if "error" in ask_result:
                    # ask_user not available — treat unanswered questions as blocker
                    messages.append({
                        "role": "user",
                        "content": (
                            f"ask_user capability is not available. "
                            f"Try to infer reasonable defaults for: {questions}. "
                            "Proceed with your best guess and note any assumptions in the output."
                        ),
                    })
                else:
                    answers_json = json.dumps(ask_result, ensure_ascii=False, default=str)
                    messages.append({
                        "role": "user",
                        "content": f"User answered your questions:\n{answers_json}\n\nNow continue working toward the goal.",
                    })
                continue

            if action == "tool_call":
                capability = parsed.get("capability", "")
                input_data = parsed.get("input_data") or {}
                reason     = parsed.get("reason", "")

                logger.info(
                    "EmergentRunner turn %d: calling capability=%r reason=%r",
                    turn + 1, capability, reason,
                )

                # Append LLM's tool-call decision to message history
                messages.append({"role": "assistant", "content": response_text})

                tool_result = await self._call_capability(capability, input_data)
                tool_results.append({"turn": turn + 1, "capability": capability, "result": tool_result})

                # Feed result back as user message for next turn
                result_json = json.dumps(tool_result, ensure_ascii=False, default=str)
                messages.append({
                    "role": "user",
                    "content": f"Capability {capability!r} returned:\n{result_json}\n\nContinue working toward the goal.",
                })
                continue

            # Unrecognised action — ask LLM to fix
            logger.warning("EmergentRunner turn %d: unknown action %r", turn + 1, action)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": 'Invalid response. Output {"action":"tool_call",...}, {"action":"ask",...}, or {"action":"done",...}.',
            })

        raise RuntimeError(
            f"EmergentRunner: step {step_name!r} did not complete within {self._max_turns} turns"
        )

    # ── LLM call ──────────────────────────────────────────────────────────────

    async def _llm_call(self, messages: list[dict], system_prompt: str) -> str:
        payload = {
            "provider":   self._provider,
            "model":      self._model,
            "messages":   messages,
            "system":     system_prompt,
            "max_tokens": 1024,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                self._proxy_url,
                headers={"X-Agent-Id": self._agent_id},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            return next(b["text"] for b in data["content"] if b["type"] == "text")

    # ── Capability call ───────────────────────────────────────────────────────

    async def _call_capability(self, capability: str, input_data: dict) -> dict:
        agent_id = await self._discover(capability)
        if not agent_id:
            return {"error": f"No agent available for capability '{capability}'"}
        try:
            result = await asyncio.wait_for(
                self._send_task(agent_id, capability, input_data),
                timeout=_STEP_TIMEOUT_S,
            )
            return result
        except asyncio.TimeoutError:
            return {"error": f"Capability '{capability}' timed out after {_STEP_TIMEOUT_S:.0f}s"}
        except Exception as exc:
            return {"error": str(exc)}


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_catalogue(caps: list[dict]) -> str:
    """Format capability list into a compact text catalogue for the LLM."""
    if not caps:
        return "(no capabilities available)"
    lines: list[str] = []
    for cap in caps:
        name = cap.get("name", "")
        desc = cap.get("description", "")
        schema = cap.get("input_schema") or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        param_parts: list[str] = []
        for k, v in props.items():
            ptype = v.get("type", "any")
            req_marker = "*" if k in required else ""
            param_parts.append(f"{k}{req_marker}:{ptype}")
        params = ", ".join(param_parts) if param_parts else "none"
        lines.append(f"- {name}: {desc} | params: {params}")
    return "\n".join(lines)


def _format_prior_outputs(outputs: list[Optional[dict]]) -> str:
    """Format prior step outputs as context for the LLM."""
    parts: list[str] = ["Prior step outputs (for reference):"]
    for i, out in enumerate(outputs):
        if out is not None:
            parts.append(f"  steps[{i}].output = {json.dumps(out, ensure_ascii=False, default=str)[:400]}")
    return "\n".join(parts)


def _parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON parse; strips markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None
