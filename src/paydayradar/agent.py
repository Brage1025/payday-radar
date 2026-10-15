"""A small cash-flow agent: tools over the Workspace, driven by a local LLM (Ollama
tool calling) with a keyword-router fallback so it also works with no model running."""

from __future__ import annotations

import json
import math
import re

import pandas as pd

from . import llm
from .pipeline import Workspace
from .reminders import draft_reminder

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]

SYSTEM = (
    "You are Payday Radar, a cash-flow assistant for a freelancer. Always use the tools "
    "for numbers; never invent figures. Amounts are NOK. Be brief and concrete."
)

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "risky_invoices", "description": "Open invoices most likely to be paid late.",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "cashflow_forecast", "description": "Monthly income/expense/balance forecast.",
        "parameters": {"type": "object", "properties": {
            "months": {"type": "integer"}, "intervals": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "can_afford",
        "description": "Check affordability using a pessimistic P10 balance scenario.",
        "parameters": {"type": "object", "required": ["amount"], "properties": {
            "amount": {"type": "number"}, "within_days": {"type": "integer"},
            "use_intervals": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "draft_reminder", "description": "Draft a payment reminder for an invoice id.",
        "parameters": {"type": "object", "required": ["invoice_id"], "properties": {
            "invoice_id": {"type": "string"}, "lang": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "anomalies", "description": "Unusual expenses and duplicate invoices.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Agent:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.tools = {
            "risky_invoices": self.risky_invoices,
            "cashflow_forecast": self.cashflow_forecast,
            "can_afford": self.can_afford,
            "draft_reminder": self.draft_reminder,
            "anomalies": self.anomalies,
        }

    # -------------------------------------------------------------------- tools
    def risky_invoices(self, top_n: int = 5) -> list[dict]:
        df = self.ws.score_open_invoices().head(int(top_n))
        return json.loads(df.assign(
            due_date=df["due_date"].dt.strftime("%Y-%m-%d"),
            expected_pay_date=df["expected_pay_date"].dt.strftime("%Y-%m-%d"),
        ).to_json(orient="records"))

    def cashflow_forecast(self, months: int = 3, intervals: bool = False) -> list[dict]:
        df = self.ws.forecast(int(months), intervals=intervals)
        return json.loads(df.assign(month=df["month"].dt.strftime("%Y-%m")).to_json(orient="records"))

    def can_afford(
        self, amount: float, within_days: int = 90, use_intervals: bool = True
    ) -> dict:
        months = min(max(math.ceil(int(within_days) / 30), 1), 6)
        fc = self.ws.forecast(months, intervals=use_intervals)
        cushion = float(fc["forecast_expenses"].mean())  # one month of costs
        if use_intervals:
            lowest_p10 = float(fc["balance_p10"].min()) - float(amount)
            lowest_p50 = float(fc["balance_p50"].min()) - float(amount)
            verdict = "yes" if lowest_p10 >= cushion else "tight" if lowest_p10 >= 0 else "no"
            return {
                "verdict": verdict,
                "basis": "P10: a 1-in-10 pessimistic scenario",
                "lowest_p10_balance_after_purchase": round(lowest_p10),
                "lowest_p50_balance_after_purchase": round(lowest_p50),
                "one_month_cushion": round(cushion),
                "horizon_months": months,
            }
        lowest = float(fc["balance"].min()) - float(amount)
        verdict = "yes" if lowest >= cushion else "tight" if lowest >= 0 else "no"
        return {"verdict": verdict, "lowest_projected_balance_after_purchase": round(lowest),
                "one_month_cushion": round(cushion), "horizon_months": months}

    def draft_reminder(self, invoice_id: str, lang: str = "en") -> dict:
        scored = self.ws.score_open_invoices()
        row = scored[scored["invoice_id"] == invoice_id]
        if row.empty:
            return {"error": f"{invoice_id} is not an open invoice"}
        return draft_reminder(row.iloc[0].to_dict(), lang=lang)

    def anomalies(self) -> dict:
        res = self.ws.find_anomalies()
        flagged = res["expenses"].drop(columns=["is_anomaly"], errors="ignore")
        return {"unusual_expenses": json.loads(flagged.assign(
                    date=flagged["date"].dt.strftime("%Y-%m-%d")).to_json(orient="records")),
                "duplicate_invoices": json.loads(res["duplicates"].assign(
                    issue_date=res["duplicates"]["issue_date"].dt.strftime("%Y-%m-%d")
                ).to_json(orient="records"))}

    # ------------------------------------------------------------------- routing
    def _days_until_end_of(self, month_name: str) -> int:
        target = _MONTHS.index(month_name) + 1
        year = self.ws.as_of.year + (1 if target < self.ws.as_of.month else 0)
        end = pd.Timestamp(year=year, month=target, day=1) + pd.offsets.MonthEnd(0)
        return max((end - self.ws.as_of).days, 30)

    def route(self, question: str) -> str:
        q = question.lower()
        if "afford" in q or "can i buy" in q or "can i spend" in q:
            m = re.search(r"(\d[\d\s.,]*)\s*(k\b)?", q)
            amount = 0.0
            if m:
                amount = float(re.sub(r"[^\d]", "", m.group(1)) or 0) * (1000 if m.group(2) else 1)
            days = 90
            for name in _MONTHS:
                if name in q:
                    days = self._days_until_end_of(name)
            return json.dumps(self.can_afford(amount, days), indent=2)
        if "remind" in q:
            top = self.risky_invoices(3)
            return "\n\n".join(
                f"To: {t['client']}\n{(d := self.draft_reminder(t['invoice_id']))['subject']}\n{d['body']}"
                for t in top)
        if any(w in q for w in ("anomal", "duplicate", "suspicious", "unusual")):
            return json.dumps(self.anomalies(), indent=2)
        if any(w in q for w in ("forecast", "cash", "balance", "runway")):
            return json.dumps(self.cashflow_forecast(3), indent=2)
        if any(w in q for w in ("late", "risk", "overdue", "unpaid", "invoice")):
            return json.dumps(self.risky_invoices(5), indent=2)
        return ("Try: 'Can I afford a 20000 laptop in November?', 'Which invoices are at risk?', "
                "'Draft reminders', 'Show the cash-flow forecast', 'Any anomalies?'")

    # ------------------------------------------------------------------ main loop
    def ask(self, question: str, max_steps: int = 5) -> str:
        if not llm.is_available():
            return self.route(question)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        try:
            for _ in range(max_steps):
                msg = llm.chat(llm.text_model(), messages, tools=TOOL_SPECS)["message"]
                messages.append(msg)
                calls = msg.get("tool_calls") or []
                if not calls:
                    return msg.get("content", "").strip() or self.route(question)
                for call in calls:
                    fn = call["function"]
                    fn_name, args = fn["name"], fn.get("arguments") or {}
                    try:
                        result = self.tools[fn_name](**args)
                    except Exception as e:
                        result = {"error": str(e)}
                    messages.append({"role": "tool", "tool_name": fn_name,
                                     "content": json.dumps(result, default=str)})
        except Exception:
            pass
        return self.route(question)
