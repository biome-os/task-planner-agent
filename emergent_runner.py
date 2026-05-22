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

from privacy import AnthropicPrivacyClient, PrivacyContext, PrivacyProxy

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
_STEP_TIMEOUT_S    = 310.0   # per-capability call inside the tool loop (matches strict STEP_TIMEOUT_S=300s + 10s buffer)


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
        on_event: Optional[Callable] = None,
        privacy_proxy: Optional[PrivacyProxy] = None,
        search_fn: Optional[Callable] = None,
    ) -> None:
        self._proxy_url = proxy_url
        self._agent_id  = agent_id
        self._model     = model
        self._provider  = provider
        self._discover  = discover_fn
        self._send_task = send_task_fn
        self._list_caps = list_capabilities_fn
        self._max_turns = max(1, max_turns)
        # Optional async search fn: (query: str, limit: int) -> list[dict]
        # Falls back to keyword-ranked full-catalogue filtering when absent.
        self._search_fn = search_fn
        # Populated at the start of each run() call for keyword-fallback search.
        self._caps_cache: list[dict] = []
        # Optional async callback: on_event(event_dict) — fired for each turn
        self._on_event: Optional[Callable] = on_event
        # Privacy client — wraps the LLM proxy call with redaction/restoration
        _pproxy = privacy_proxy or PrivacyProxy()
        self._llm_client = AnthropicPrivacyClient(
            proxy_url=proxy_url,
            agent_id=agent_id,
            privacy_proxy=_pproxy,
        )

    # ── Event helper ──────────────────────────────────────────────────────────

    async def _emit(self, event: dict) -> None:
        if self._on_event is not None:
            try:
                await self._on_event(event)
            except Exception as exc:
                logger.debug("EmergentRunner on_event callback failed: %s", exc)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        step_name: str,
        step_goal: str,
        hint_input: dict,
        prior_outputs: list[Optional[dict]],
        privacy_ctx: Optional[PrivacyContext] = None,
    ) -> dict[str, Any]:
        """
        Execute one emergent step.  Returns the final output dict on success.
        Raises ``RuntimeError`` if the goal cannot be achieved within max_turns.
        """
        # Pre-fetch catalogue for keyword-fallback search; NOT injected into the
        # initial prompt — the LLM must search explicitly via search_capabilities.
        self._caps_cache: list[dict] = await self._list_caps()

        context_parts: list[str] = []
        if prior_outputs:
            context_parts.append(_format_prior_outputs(prior_outputs))
        if hint_input:
            context_parts.append(f"Hint input from planner:\n{json.dumps(hint_input, indent=2)}")

        context_block = "\n\n".join(context_parts)

        user_prompt = (
            f"Goal: {step_goal}\n\n"
            f"{context_block}\n\n"
            "Use the search_capabilities action to discover what capabilities are available "
            "before calling any capability."
        ).strip()

        messages: list[dict] = [{"role": "user", "content": user_prompt}]
        tool_results: list[dict] = []

        logger.info("EmergentRunner: starting step %r (max_turns=%d)", step_name, self._max_turns)

        system_prompt = _load_system_prompt()

        for turn in range(self._max_turns):
            response_text = await self._llm_call(messages, system_prompt, privacy_ctx=privacy_ctx)
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
                await self._emit({
                    "type":    "done",
                    "turn":    turn + 1,
                    "summary": summary,
                    "output":  output,
                })
                return output

            if action == "ask":
                questions = parsed.get("questions") or []
                reason    = parsed.get("reason", "")
                logger.info(
                    "EmergentRunner turn %d: LLM asking follow-up (%d question(s)) — %s",
                    turn + 1, len(questions), reason,
                )
                await self._emit({
                    "type":      "ask",
                    "turn":      turn + 1,
                    "questions": questions,
                    "reason":    reason,
                })
                ask_result = await self._call_capability(
                    "ask_user",
                    {"questions": questions, "reason": reason},
                )
                messages.append({"role": "assistant", "content": response_text})
                if "error" in ask_result:
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

            if action == "search_capabilities":
                keywords = parsed.get("keywords") or []
                limit    = int(parsed.get("limit") or 6)
                reason   = parsed.get("reason", "")

                logger.info(
                    "EmergentRunner turn %d: search_capabilities keywords=%r reason=%r",
                    turn + 1, keywords, reason,
                )
                await self._emit({
                    "type":     "search_capabilities",
                    "turn":     turn + 1,
                    "keywords": keywords,
                    "reason":   reason,
                })

                search_result = await self._do_search_capabilities(keywords, limit)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"search_capabilities results:\n{search_result}\n\n"
                        "Now continue working toward the goal."
                    ),
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

                await self._emit({
                    "type":       "tool_call",
                    "turn":       turn + 1,
                    "capability": capability,
                    "input_data": input_data,
                    "reason":     reason,
                })

                # Append LLM's tool-call decision to message history
                messages.append({"role": "assistant", "content": response_text})

                tool_result = await self._call_capability(capability, input_data)
                tool_results.append({"turn": turn + 1, "capability": capability, "result": tool_result})

                await self._emit({
                    "type":       "tool_result",
                    "turn":       turn + 1,
                    "capability": capability,
                    "result":     tool_result,
                    "error":      tool_result.get("error") if isinstance(tool_result, dict) else None,
                })

                # If the sub-agent needs user input, stop the loop and surface
                # the clarification request to the caller unchanged.
                if isinstance(tool_result, dict) and tool_result.get("followup_request"):
                    logger.info(
                        "EmergentRunner turn %d: capability %r returned followup_request"
                        " — pausing for user input",
                        turn + 1, capability,
                    )
                    await self._emit({
                        "type":             "clarification_required",
                        "turn":             turn + 1,
                        "capability":       capability,
                        "followup_request": tool_result["followup_request"],
                    })
                    return {
                        "_clarification_required": True,
                        "followup_request":        tool_result["followup_request"],
                    }

                # Gap 1: browser-agent LlmRetryExhausted — provider unavailable,
                # replan needed.  Surface to caller so it can trigger _handle_workflow_failure.
                if isinstance(tool_result, dict) and tool_result.get("replan_context"):
                    logger.info(
                        "EmergentRunner turn %d: capability %r returned replan_context"
                        " — escalating to workflow failure handler",
                        turn + 1, capability,
                    )
                    return {
                        "_replan_required": True,
                        "replan_context":   tool_result["replan_context"],
                        "error":            tool_result.get("error"),
                    }

                # Gap 2: document-agent consent gate — synthesise a followup_request
                # so the existing _clarification_required path handles it uniformly.
                if isinstance(tool_result, dict) and tool_result.get("consent_required"):
                    logger.info(
                        "EmergentRunner turn %d: capability %r returned consent_required"
                        " — converting to followup_request",
                        turn + 1, capability,
                    )
                    synthetic_followup = {
                        "question":        tool_result.get("message", "Consent is required to proceed."),
                        "data_disclosure": tool_result.get("data_disclosure", ""),
                        "answer_format":   "yes_no",
                        "intent":          "consent_required",
                    }
                    return {"_clarification_required": True, "followup_request": synthetic_followup}

                # Feed result back as user message for next turn.
                # Cap at 2000 chars to limit history growth; LLM only needs
                # key fields (stdout, error, status) not the full payload.
                result_json = json.dumps(tool_result, ensure_ascii=False, default=str)
                if len(result_json) > 2000:
                    result_json = result_json[:2000] + "…[truncated]"
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

    async def _llm_call(
        self,
        messages: list[dict],
        system_prompt: str,
        *,
        privacy_ctx: Optional[PrivacyContext] = None,
    ) -> str:
        # Anthropic supports assistant-turn prefill: if the last message has
        # role "assistant", the model completes from there.  We use this to
        # force a JSON-only response that always starts with `{`.
        _PREFILL = "{"
        call_messages = messages + [{"role": "assistant", "content": _PREFILL}]

        payload = {
            "provider":   self._provider,
            "model":      self._model,
            "messages":   call_messages,
            "system":     system_prompt,
            "max_tokens": 4096,
        }
        response_text = await self._llm_client.complete(
            payload, privacy_ctx=privacy_ctx, timeout=120.0
        )
        # Reconstruct the full JSON object (prepend the prefill the model
        # continued from).  If the provider ignored the prefill (e.g. OpenAI)
        # and the model already started with `{`, avoid double-adding it.
        if response_text.lstrip().startswith("{"):
            return response_text
        return _PREFILL + response_text

    # ── Capability call ───────────────────────────────────────────────────────

    async def _do_search_capabilities(self, keywords: list[str], limit: int) -> str:
        """
        Search the capability catalogue by keyword phrases.

        Delegates to ``self._search_fn(query, limit)`` when provided (e.g. the
        planner's vector-search index).  Falls back to keyword-ranked scoring
        over the pre-fetched ``_caps_cache`` so the mode works without any
        external search service.
        """
        limit = max(1, min(limit, 12))
        query = " ".join(keywords).lower()

        if not query.strip():
            return "(no keywords provided — supply at least one keyword phrase)"

        if self._search_fn is not None:
            try:
                results: list[dict] = await self._search_fn(query, limit)
                if results:
                    return _format_catalogue(results)
            except Exception as exc:
                logger.debug("EmergentRunner search_fn failed, falling back: %s", exc)

        # Keyword-ranked fallback: score each capability by term overlap
        caps = self._caps_cache
        if not caps:
            return "(no capabilities available)"

        terms = [t for t in query.split() if len(t) > 2]
        scored: list[tuple[int, dict]] = []
        for cap in caps:
            text = f"{cap.get('name', '')} {cap.get('description', '')}".lower()
            score = sum(1 for t in terms if t in text)
            if score > 0:
                scored.append((score, cap))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:limit]]

        if not top:
            # Nothing matched — return a broad hint instead of empty
            return (
                "(no capabilities matched those keywords)\n"
                f"Try broader terms. {len(caps)} capabilities are registered in total."
            )
        return _format_catalogue(top)

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
    """Format capability list into a compact text catalogue for the LLM.

    Format: name(req_param*, req2*[+N opt]) — 70-char description
    Types are omitted; only required params are listed by name; optional param
    count is appended in brackets to keep the LLM aware without listing them all.
    """
    if not caps:
        return "(no capabilities available)"
    lines: list[str] = []
    for cap in caps:
        name = cap.get("name", "")
        desc = cap.get("description", "")
        # Truncate description to first sentence or 70 chars, whichever is shorter
        first_sentence = desc.split(".")[0].strip() if desc else ""
        short_desc = first_sentence[:70] if first_sentence else desc[:70]

        schema = cap.get("input_schema") or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        req_params = [k for k in props if k in required]
        opt_count  = len(props) - len(req_params)

        if req_params:
            param_str = ", ".join(f"{k}*" for k in req_params)
            if opt_count:
                param_str += f" [+{opt_count} opt]"
        elif opt_count:
            param_str = f"[{opt_count} opt]"
        else:
            param_str = ""

        sig = f"{name}({param_str})" if param_str else name
        lines.append(f"{sig} — {short_desc}")
    return "\n".join(lines)


def _format_prior_outputs(outputs: list[Optional[dict]]) -> str:
    """Format prior step outputs as context for the LLM."""
    parts: list[str] = ["Prior step outputs (for reference):"]
    for i, out in enumerate(outputs):
        if out is not None:
            parts.append(f"  steps[{i}].output = {json.dumps(out, ensure_ascii=False, default=str)[:400]}")
    return "\n".join(parts)


def _parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON parse; handles markdown fences, text preambles, and
    XML <function_calls> wrappers that some models emit despite JSON-only
    instructions."""
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    # Direct parse (fast path)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # If it's a list (e.g. model wrapped in array), return first dict element
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    return item
    except json.JSONDecodeError:
        pass

    # Extract the outermost {...} block — handles text preambles and
    # <function_calls>[{...}] wrappers
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None
