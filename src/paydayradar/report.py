"""Markdown report used by the weekly GitHub Actions run."""

from __future__ import annotations

from .pipeline import Workspace


def render_report(ws: Workspace) -> str:
    scored = ws.score_open_invoices()
    show = scored.drop(columns=["due_date", "expected_pay_date"]).head(10)
    fc = ws.forecast(3).assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))
    an = ws.find_anomalies()
    return "\n".join([
        f"# Payday Radar report (as of {ws.as_of.date()}, backend: {ws.backend})",
        "", "## Invoices most at risk of paying late", "```", show.to_string(index=False), "```",
        "", "## 90-day cash-flow forecast (NOK)", "```", fc.to_string(index=False), "```",
        "", "## Unusual expenses", "```",
        an["expenses"][["date", "vendor", "category", "amount", "expected_amount", "robust_z"]]
        .to_string(index=False), "```",
        "", "## Possible duplicate invoices", "```", an["duplicates"].to_string(index=False), "```",
        "",
    ])
