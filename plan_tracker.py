"""
plan_tracker.py — Validates emergent step outputs and decides how to continue.

After an emergent step completes, PlanTracker compares the output against the
step's expected output schema (if the planner supplied one) and returns one of
three decisions:

  continue       — output looks good; proceed to the next step
  local_replan   — output is incomplete / wrong shape; re-plan remaining steps
                   only (no full restart) with an explanation of what went wrong
  escalate       — output is so far off that a full replan / human escalation is
                   warranted

Design notes:
- Schema validation is intentionally lenient: presence of required keys is
  checked but value types are not strictly enforced (LLM output is inherently
  fuzzy).
- The LLM validator is only invoked when a structured output_schema is present.
  Plain text outputs ("summary", "text", "result" keys) are accepted as-is.
- All validation is best-effort; unexpected errors default to "continue" to
  avoid blocking workflows unnecessarily.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

Decision = Literal["continue", "local_replan", "escalate"]


class PlanTracker:
    """
    Stateless validator for emergent step outputs.

    Usage
    -----
    tracker = PlanTracker()
    decision, reason = tracker.validate(step, output)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        step: dict,
        output: Optional[dict],
    ) -> tuple[Decision, str]:
        """
        Validate *output* against the step definition and return a (decision, reason) tuple.

        Parameters
        ----------
        step:    The step dict from the workflow plan (as stored in WorkflowStore).
        output:  The dict returned by EmergentStepRunner.run() — may be None on error.
        """
        if output is None:
            return "local_replan", "Emergent step returned no output"

        # If the output contains an error key, trigger local replan
        if isinstance(output.get("error"), str) and output["error"].strip():
            return "local_replan", f"Emergent step error: {output['error']}"

        # Check output_schema if the step has one
        expected_schema = step.get("output_schema")
        if expected_schema and isinstance(expected_schema, dict):
            decision, reason = self._validate_schema(output, expected_schema)
            if decision != "continue":
                return decision, reason

        # Plain-text outputs: accept if any text-like key is present and non-empty
        text_keys = ("text", "result", "summary", "response", "content", "message")
        if any(isinstance(output.get(k), str) and output[k].strip() for k in text_keys):
            return "continue", "Output contains text result"

        # Dict output with meaningful keys (not just _emergent_ metadata)
        user_keys = [k for k in output if not k.startswith("_emergent_")]
        if user_keys:
            return "continue", "Output contains structured data"

        return "local_replan", "Emergent step output is empty or metadata-only"

    # ── Schema validation ─────────────────────────────────────────────────────

    def _validate_schema(
        self,
        output: dict,
        schema: dict,
    ) -> tuple[Decision, str]:
        """
        Check that *output* satisfies the required properties in *schema*.
        Returns ("continue", reason) on pass, ("local_replan", reason) on soft miss,
        or ("escalate", reason) on a complete mismatch.
        """
        required: list[str] = schema.get("required", [])
        properties: dict = schema.get("properties", {})

        if not required and not properties:
            return "continue", "No schema constraints to validate"

        missing = [k for k in required if k not in output or output[k] in (None, "", [], {})]
        if not missing:
            return "continue", "All required output fields present"

        present_optional = [
            k for k in properties
            if k not in required and k in output and output[k] not in (None, "", [], {})
        ]

        if present_optional:
            # Some useful keys present even if required ones are missing → local replan
            reason = f"Missing required output fields: {missing}; optional fields present: {present_optional}"
            logger.warning("PlanTracker: %s", reason)
            return "local_replan", reason

        # Nothing useful in the output at all
        reason = f"Missing required output fields: {missing} and no optional fields present"
        logger.warning("PlanTracker: escalating — %s", reason)
        return "escalate", reason

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def replan_context(
        step: dict,
        step_index: int,
        output: Optional[dict],
        reason: str,
    ) -> dict[str, Any]:
        """
        Build the replan_context dict that gets passed into _partial_replan.
        Captures what we know so far so the planner can repair the plan.
        """
        return {
            "failure_type":   "emergent_output_invalid",
            "failed_step":    step_index + 1,
            "step_name":      step.get("name", f"Step {step_index + 1}"),
            "capability":     step.get("capability", ""),
            "goal":           step.get("goal", ""),
            "reason":         reason,
            "partial_output": output,
            "retry_possible": False,
        }
