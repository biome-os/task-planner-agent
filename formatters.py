"""
formatters.py — Jinja2 Slack formatters for each known capability.

Each entry maps a capability name to a Jinja2 template string rendered with
the capability's output dict as the top-level context.  The whole output is
also available as the variable ``data`` in every template.

Add a new entry to ``FORMATTERS`` whenever you want a capability's results to
render with a custom layout.  ``_default`` is used as a fallback.
"""
from __future__ import annotations

import logging
from typing import Any

from jinja2 import Environment, Undefined

logger = logging.getLogger(__name__)

# ── Templates ──────────────────────────────────────────────────────────────────

FORMATTERS: dict[str, str] = {

    # ── serper_search ──────────────────────────────────────────────────────────
    # Output keys: query, search_type, count, results[{title,link,snippet,source}], result
    "serper_search": """\
*{{ query }}* — {{ count }} result(s)

{% for item in results[:5] %}
*{{ loop.index }}. {{ item.title }}*
{% if item.snippet %}{{ item.snippet }}
{% endif %}
<{{ item.link }}|:link: View>{% if item.source %} · _{{ item.source }}_{% endif %}

{% endfor %}\
""",

    # ── browse_web ─────────────────────────────────────────────────────────────
    # Output keys: summary
    "browse_web": """\
{{ summary }}\
""",

    # ── read_emails ────────────────────────────────────────────────────────────
    # Output keys: emails[{subject, sender/from_, snippet/body, date}]
    "read_emails": """\
{% if emails %}
*{{ emails | length }} email(s) found:*

{% for email in emails[:5] %}
*{{ loop.index }}. {{ email.subject or "(no subject)" }}*
From: {{ email.sender or email.get("from_", "unknown") }}{% if email.date %} · {{ email.date }}{% endif %}
{% if email.snippet or email.body %}{{ (email.snippet or email.body)[:200] }}{% endif %}

{% endfor %}
{% else %}
No emails found.
{% endif %}\
""",

    # ── analyze_portfolio ──────────────────────────────────────────────────────
    # Output keys: total_positions, tickers, estimated_cash_balance,
    #              top_covered_calls, top_cash_secured_puts, recommended_combos
    "analyze_portfolio": """\
*Portfolio Analysis*
Positions: {{ total_positions }} · Cash: ${{ "%.2f" | format(estimated_cash_balance or 0) }}
Tickers: {{ tickers | join(", ") }}

{% if top_covered_calls %}
*Top Covered Calls:*
{% for c in top_covered_calls[:3] %}
• {{ c.ticker }} {{ c.expiry }} ${{ c.strike }} call — premium ${{ c.premium }}
{% endfor %}
{% endif %}
{% if top_cash_secured_puts %}
*Top Cash-Secured Puts:*
{% for p in top_cash_secured_puts[:3] %}
• {{ p.ticker }} {{ p.expiry }} ${{ p.strike }} put — premium ${{ p.premium }}
{% endfor %}
{% endif %}
{% if recommended_combos %}
*Recommended Combos:*
{% for combo in recommended_combos[:3] %}
• {{ combo.ticker }}: {{ combo.strategy }} — ${{ combo.total_premium }}
{% endfor %}
{% endif %}\
""",

    # ── get_options_recommendations ────────────────────────────────────────────
    # Output keys: tickers_analyzed, covered_calls, cash_secured_puts, target_combos
    "get_options_recommendations": """\
*Options Recommendations* ({{ tickers_analyzed | join(", ") }})

{% if covered_calls %}
*Covered Calls:*
{% for c in covered_calls[:5] %}
• {{ c.ticker }} {{ c.expiry }} ${{ c.strike }} — ${{ c.premium }}
{% endfor %}
{% endif %}
{% if cash_secured_puts %}
*Cash-Secured Puts:*
{% for p in cash_secured_puts[:5] %}
• {{ p.ticker }} {{ p.expiry }} ${{ p.strike }} — ${{ p.premium }}
{% endfor %}
{% endif %}\
""",

    # ── list_scheduled_tasks ───────────────────────────────────────────────────
    # Output keys: tasks[{task_id, capability, scheduled_at, status}], count
    "list_scheduled_tasks": """\
*Scheduled Tasks* ({{ count }} total)

{% for t in tasks[:10] %}
• *{{ t.capability }}* — {{ t.scheduled_at }} · _{{ t.status }}_
{% endfor %}\
""",

    # ── _default ───────────────────────────────────────────────────────────────
    # Generic fallback: renders string and simple-list fields from the output.
    "_default": """\
{% for key, value in data.items() %}
{% if value is string and value and key != "raw" %}
*{{ key }}:* {{ value }}
{% elif value is iterable and value is not mapping and value is not string %}
*{{ key }}:* {{ value | join(", ") }}
{% endif %}
{% endfor %}\
""",
}

# ── Jinja2 environment ─────────────────────────────────────────────────────────

_ENV = Environment(
    undefined=Undefined,   # silently ignore missing variables (don't raise)
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_formatter(
    capability_name: str,
    data: Any,
    template_override: str = "",
) -> str:
    """
    Render the Jinja2 formatter for *capability_name* against *data*.

    Priority:
      1. *template_override* — caller-supplied ad-hoc template
      2. ``FORMATTERS[capability_name]`` — registered per-capability template
      3. ``FORMATTERS["_default"]`` — generic fallback

    The template context is the output dict spread as top-level variables, with
    the whole dict also accessible as ``data``.  Fails open to ``str(data)`` on
    any rendering error.
    """
    template_str = (
        template_override.strip()
        or FORMATTERS.get(capability_name)
        or FORMATTERS["_default"]
    )

    ctx: dict = {}
    if isinstance(data, dict):
        ctx = dict(data)
        ctx.setdefault("data", data)
    else:
        ctx = {"data": data, "value": data}

    try:
        rendered = _ENV.from_string(template_str).render(**ctx).strip()
        # Collapse runs of 3+ blank lines down to 2
        import re
        rendered = re.sub(r"\n{3,}", "\n\n", rendered)
        return rendered
    except Exception as exc:
        logger.warning(
            "Formatter render failed (capability=%r): %s", capability_name, exc
        )
        return str(data)
