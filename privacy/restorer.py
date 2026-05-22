"""
privacy/restorer.py — Placeholder restorer.

Scans text for ``<<CATEGORY_xxxxxxxx>>`` tokens produced by the redactor and
replaces each with the original value stored in the PrivacyContext mapping.

Unknown placeholders (not in the context) are left intact so the caller can
detect them easily.
"""
from __future__ import annotations

import json
import re

from .context import PrivacyContext

# Matches any placeholder emitted by SensitiveDataRedactor.
_PLACEHOLDER_RE = re.compile(r"<<([A-Z0-9_]+)_([0-9a-f]{8})>>")


class PlaceholderRestorer:
    """
    Reverses the redaction performed by SensitiveDataRedactor.

    Two entry points are provided:

    restore(text, context)
        Plain-text restore: replaces all placeholder tokens in *text* with
        their originals from *context.mapping*.

    restore_structure(obj, context)
        Recursive restore: walks a dict / list / str structure (e.g. a
        tool-use / tool-result JSON block) and restores all strings in place,
        returning a new structure with originals substituted.
    """

    def restore(self, text: str, context: PrivacyContext) -> str:
        """Replace all placeholder tokens in *text* with their original values."""
        if not context.mapping:
            return text

        def _replace(m: re.Match) -> str:
            placeholder = m.group(0)
            return context.mapping.get(placeholder, placeholder)

        return _PLACEHOLDER_RE.sub(_replace, text)

    def restore_structure(self, obj: object, context: PrivacyContext) -> object:
        """
        Recursively restore placeholders inside a nested dict / list / str
        structure (e.g. a parsed tool-use arguments dict or a content block list).
        Returns a new object — the original is never mutated.
        """
        if not context.mapping:
            return obj

        if isinstance(obj, str):
            return self.restore(obj, context)

        if isinstance(obj, list):
            return [self.restore_structure(item, context) for item in obj]

        if isinstance(obj, dict):
            return {k: self.restore_structure(v, context) for k, v in obj.items()}

        # int, float, bool, None — pass through
        return obj

    def restore_json_string(self, json_text: str, context: PrivacyContext) -> str:
        """
        Parse *json_text* as JSON, restore all string values recursively, then
        re-serialise.  Falls back to plain-text restore if JSON parsing fails.
        """
        if not context.mapping:
            return json_text
        try:
            parsed = json.loads(json_text)
            restored = self.restore_structure(parsed, context)
            return json.dumps(restored, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return self.restore(json_text, context)
