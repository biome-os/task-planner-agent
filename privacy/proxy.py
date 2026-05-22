"""
privacy/proxy.py — PrivacyProxy facade.

Central coordinator for pre-request redaction and post-response restoration.
A single PrivacyProxy instance is shared across TaskPlanner and
EmergentStepRunner for a given agent session; the ``enabled`` flag can be
toggled at runtime via a settings_push without restarting the agent.

When disabled (default) all methods are identity pass-throughs with zero
overhead — no regex scanning, no allocation.
"""
from __future__ import annotations

import copy
import json
import logging

from .context import PrivacyContext
from .redactor import SensitiveDataRedactor
from .restorer import PlaceholderRestorer

logger = logging.getLogger(__name__)

# Injected into the system prompt when the proxy is active so the model
# knows to preserve placeholder tokens as-is.
_PRIVACY_NOTE = (
    "\n\n[Privacy proxy active] Some values in this conversation have been "
    "replaced with privacy placeholder tokens such as <<EMAIL_a1b2c3d4>>, "
    "<<PHONE_e5f6a7b8>>, <<SK_KEY_12345678>>, etc.  "
    "Use these tokens exactly as shown in your response — do NOT expand, "
    "paraphrase, guess, or omit them."
)


class PrivacyProxy:
    """
    Wraps outbound LLM payloads with redaction and inbound responses with
    restoration.

    Usage
    -----
    proxy = PrivacyProxy()
    proxy.enabled = True                    # toggled via settings_push

    ctx = PrivacyContext()                  # one per workflow run

    # Before sending to the LLM:
    clean_payload = proxy.apply_to_payload(payload, ctx)

    # After receiving the response text:
    original_text = proxy.unwrap_response(llm_text, ctx)
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._redactor = SensitiveDataRedactor()
        self._restorer = PlaceholderRestorer()

    # ── Public API ─────────────────────────────────────────────────────────────

    def apply_to_payload(self, payload: dict, context: PrivacyContext) -> dict:
        """
        Return a copy of *payload* with sensitive data redacted in all message
        content and segments.  Also injects the privacy note into the system
        prompt so the model knows to treat placeholder tokens as opaque.

        When ``enabled`` is False returns *payload* unchanged (no copy).
        """
        if not self.enabled:
            return payload

        payload = copy.deepcopy(payload)

        if "prompt_segments" in payload:
            payload["prompt_segments"] = self._process_segments(
                payload["prompt_segments"], context, inject_note=True
            )
        else:
            # Flat messages + system format
            if payload.get("system"):
                payload["system"] = payload["system"] + _PRIVACY_NOTE
            elif _PRIVACY_NOTE:
                payload["system"] = _PRIVACY_NOTE.strip()

            if "messages" in payload:
                payload["messages"] = self.wrap_messages(payload["messages"], context)

        logger.debug(
            "PrivacyProxy: redacted payload for context %s (%d mappings so far)",
            context.conversation_id[:8], len(context.mapping),
        )
        return payload

    def wrap_messages(self, messages: list, context: PrivacyContext) -> list:
        """
        Return a new list of messages with all string content redacted.
        Handles both plain string content and structured content-block lists.

        When ``enabled`` is False returns *messages* unchanged.
        """
        if not self.enabled:
            return messages

        result: list[dict] = []
        for msg in messages:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = self._redactor.redact(content, context)
            elif isinstance(content, list):
                msg["content"] = self._redact_content_blocks(content, context)
            result.append(msg)
        return result

    def unwrap_response(self, response_text: str, context: PrivacyContext) -> str:
        """
        Restore all placeholder tokens in the LLM response text.

        When ``enabled`` is False returns *response_text* unchanged.
        """
        if not self.enabled:
            return response_text

        restored = self._restorer.restore(response_text, context)
        if restored != response_text:
            logger.debug(
                "PrivacyProxy: restored %d placeholder(s) in response for context %s",
                response_text.count("<<"), context.conversation_id[:8],
            )
        return restored

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _process_segments(
        self,
        segments: list[dict],
        context: PrivacyContext,
        inject_note: bool = True,
    ) -> list[dict]:
        """Redact a prompt_segments list and optionally inject the privacy note."""
        note_injected = False
        result: list[dict] = []

        for seg in segments:
            seg = dict(seg)
            seg_type = seg.get("type", "")

            if seg_type == "system":
                content = seg.get("content", "")
                if inject_note and not note_injected:
                    content = content + _PRIVACY_NOTE
                    note_injected = True
                seg["content"] = self._redactor.redact(content, context)

            elif seg_type == "messages":
                msgs = seg.get("content") or []
                seg["content"] = self.wrap_messages(msgs, context)

            elif seg_type == "context":
                # Context segments carry a plain string payload
                content = seg.get("content", "")
                if isinstance(content, str):
                    seg["content"] = self._redactor.redact(content, context)

            result.append(seg)

        # If no system segment was found, prepend one carrying only the note
        if inject_note and not note_injected:
            result.insert(0, {
                "name":      "_privacy_note",
                "type":      "system",
                "content":   _PRIVACY_NOTE.strip(),
                "cacheable": True,
            })

        return result

    def _redact_content_blocks(self, blocks: list, context: PrivacyContext) -> list:
        """Redact structured content blocks (Anthropic tool-use / tool-result format)."""
        result: list = []
        for block in blocks:
            if not isinstance(block, dict):
                result.append(block)
                continue
            block = dict(block)
            btype = block.get("type", "")

            if btype == "text":
                block["text"] = self._redactor.redact(block.get("text", ""), context)

            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, str):
                    block["content"] = self._redactor.redact(inner, context)
                elif isinstance(inner, list):
                    block["content"] = self._redact_content_blocks(inner, context)

            elif btype == "tool_use":
                # Redact tool input (dict → serialise → redact → deserialise)
                inp = block.get("input")
                if isinstance(inp, dict):
                    raw = json.dumps(inp, ensure_ascii=False)
                    redacted_raw = self._redactor.redact(raw, context)
                    try:
                        block["input"] = json.loads(redacted_raw)
                    except json.JSONDecodeError:
                        block["input"] = inp  # leave unchanged on parse error

            result.append(block)
        return result
