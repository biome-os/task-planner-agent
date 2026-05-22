"""
privacy/context.py — Per-workflow privacy context.

Holds the placeholder→original mapping for one planning+execution session.
The same PrivacyContext object is reused across all LLM turns in a workflow
(planning phase + any emergent tool-loop steps) so that a value redacted during
planning keeps the same placeholder token throughout the entire run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class PrivacyContext:
    """
    Tracks the bidirectional mapping between redacted placeholder tokens and
    the original sensitive values they replaced within one workflow session.

    Attributes
    ----------
    conversation_id:
        Stable identifier for the workflow this context belongs to.
        Used only for logging / diagnostics.
    mapping:
        ``{placeholder: original}``  e.g.
        ``{"<<EMAIL_a1b2c3d4>>": "alice@example.com"}``
    """

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mapping: dict[str, str] = field(default_factory=dict)

    def placeholder_for(self, original: str, category: str) -> str:
        """
        Return the stable placeholder for *original*, creating one if this
        value has not been seen before.
        """
        # Re-use an existing placeholder for the exact same value to keep
        # token count stable and avoid ambiguity in multi-turn conversations.
        for ph, val in self.mapping.items():
            if val == original:
                return ph
        token_id = uuid.uuid4().hex[:8]
        placeholder = f"<<{category}_{token_id}>>"
        self.mapping[placeholder] = original
        return placeholder
