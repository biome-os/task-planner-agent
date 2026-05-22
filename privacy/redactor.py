"""
privacy/redactor.py — Regex-based PII / secret redactor.

Scans a string for known sensitive-data patterns, replaces each match with a
stable ``<<CATEGORY_xxxxxxxx>>`` placeholder, and records the mapping in the
supplied PrivacyContext so the restorer can reverse the substitution later.

The redactor operates on the *full serialised string* of whatever is passed to
it — including values deeply nested inside JSON — so callers should serialise
structured data to a string before redacting and parse it again afterwards
(or use PrivacyProxy.apply_to_payload which handles this automatically).
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .context import PrivacyContext


class _Pattern(NamedTuple):
    category: str   # used as the placeholder prefix, e.g. "EMAIL"
    regex: str


# Patterns are evaluated in order; more specific patterns come first to avoid
# partial matches being hidden by broader ones.
_PATTERNS: list[_Pattern] = [
    # SSN — must precede generic digit sequences
    _Pattern("SSN",         r"\b\d{3}-\d{2}-\d{4}\b"),

    # Email addresses
    _Pattern("EMAIL",       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),

    # Anthropic / OpenAI API keys  (sk-…)
    _Pattern("SK_KEY",      r"\bsk-[A-Za-z0-9]{20,}\b"),

    # AWS access key IDs
    _Pattern("AWS_KEY",     r"\bAKIA[A-Z0-9]{16}\b"),

    # HTTP Bearer tokens (the token part only — stops at whitespace/quote/})
    _Pattern("BEARER",      r"(?<=Bearer\s)[A-Za-z0-9._\-/+=]{20,}"),

    # key= / api_key= / token= / secret= / password= value patterns
    # We redact only the value (group 2); the key prefix (group 1) is kept intact.
    # Uses a non-capturing prefix instead of a variable-width lookbehind (which
    # Python's re module does not support).
    _Pattern(
        "SECRET_VALUE",
        r"""(?:[\"']?(?:api_key|api-key|token|secret|password|passwd|credential)[\"']?\s*[=:]\s*[\"']?)([A-Za-z0-9+/=_\-]{16,})""",
    ),

    # Long hex strings that look like secrets (≥32 hex chars) appearing after common key names
    _Pattern(
        "HEX_SECRET",
        r"\b[0-9a-fA-F]{32,}\b",
    ),

    # Long base64-ish strings (≥40 chars containing letters+digits+/+=) in key= context
    _Pattern(
        "B64_SECRET",
        r"\b[A-Za-z0-9+/]{40,}={0,2}\b",
    ),

    # IPv4 addresses (after less-specific patterns that might include them)
    _Pattern(
        "IP_ADDRESS",
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    ),

    # Credit card numbers: 13-19 digits optionally separated by spaces or dashes
    # Use a word-boundary / lookahead to avoid matching version numbers etc.
    _Pattern(
        "CREDIT_CARD",
        r"\b(?:\d[ \-]?){13,19}\b(?![\./])",
    ),

    # Phone numbers (North-American + international formats)
    _Pattern(
        "PHONE",
        r"\b(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
    ),
]


class SensitiveDataRedactor:
    """
    Finds sensitive values in *text* and replaces each with a placeholder token.

    Placeholders have the form ``<<CATEGORY_xxxxxxxx>>`` where *xxxxxxxx* is an
    8-character hex UUID fragment that is unique per redacted value per context.
    """

    def __init__(self) -> None:
        self._compiled: list[tuple[str, re.Pattern[str]]] = [
            (p.category, re.compile(p.regex)) for p in _PATTERNS
        ]

    def redact(self, text: str, context: PrivacyContext) -> str:
        """
        Return *text* with all detected sensitive values replaced by placeholders.
        Mapping is accumulated in *context*.

        When a pattern contains a capture group (group 1), only that group is
        replaced — the surrounding match prefix/suffix is kept intact.  This
        lets patterns use a non-capturing prefix instead of a lookbehind.
        """
        for category, pattern in self._compiled:
            has_group = pattern.groups > 0

            def _replace(
                m: re.Match,
                cat: str = category,
                ctx: PrivacyContext = context,
                grp: bool = has_group,
            ) -> str:
                if grp:
                    # Preserve the prefix, replace only the captured value
                    full = m.group(0)
                    value = m.group(1)
                    placeholder = ctx.placeholder_for(value, cat)
                    return full[: m.start(1) - m.start(0)] + placeholder
                return ctx.placeholder_for(m.group(0), cat)

            text = pattern.sub(_replace, text)
        return text
