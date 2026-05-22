"""
privacy — LLM privacy proxy for task-planner-agent.

Intercepts all outbound LLM API calls, redacts sensitive data (emails, phone
numbers, API keys, IPs, credit-card numbers, SSNs) before they reach the
model, and restores the original values in the response before returning to
the caller.

Public API
----------
PrivacyProxy
    The main coordinator.  Toggle ``proxy.enabled`` at runtime.

PrivacyContext
    Per-workflow mapping of placeholder↔original pairs.  Create one context
    per plan_task invocation and reuse it across the planning phase and all
    emergent tool-loop steps so placeholders are stable end-to-end.

get_proxy() -> PrivacyProxy
    Returns the module-level singleton.  OrchestratorClient calls this once
    and passes the result to TaskPlanner and EmergentStepRunner.
"""
from __future__ import annotations

from .clients import AnthropicPrivacyClient
from .context import PrivacyContext
from .proxy import PrivacyProxy

__all__ = [
    "PrivacyProxy",
    "PrivacyContext",
    "AnthropicPrivacyClient",
    "get_proxy",
]

_proxy_instance: PrivacyProxy | None = None


def get_proxy() -> PrivacyProxy:
    """Return the module-level PrivacyProxy singleton (created on first call)."""
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = PrivacyProxy()
    return _proxy_instance
