"""Expose Payday Radar agent tools over the Model Context Protocol."""

from __future__ import annotations

import json
import re
from typing import Any

from mcp.server.mcpserver import MCPServer

from .agent import Agent
from .pipeline import Workspace


def _letter(index: int) -> str:
    letters = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        letters = chr(ord("A") + remainder) + letters
        index -= 1
    return letters


class _Redactor:
    def __init__(self, ws: Workspace):
        self.aliases: dict[str, str] = {}
        counts = {"Client": 0, "Vendor": 0}
        names = (
            ("Client", ws.invoices["client"]),
            ("Vendor", ws.expenses["vendor"]),
        )
        for kind, values in names:
            for name in sorted({str(value) for value in values if value}):
                if name not in self.aliases:
                    self.aliases[name] = f"{kind} {_letter(counts[kind])}"
                    counts[kind] += 1
        self.pattern = re.compile(
            "|".join(re.escape(name) for name in sorted(self.aliases, key=len, reverse=True))
        ) if self.aliases else None

    def json(self, result: Any) -> str:
        text = json.dumps(result, ensure_ascii=False, default=str)
        if self.pattern is not None:
            text = self.pattern.sub(lambda match: self.aliases[match.group()], text)
        return text

def _clamp(value: int, low: int, high: int) -> int:
    """Keep client-supplied sizes sane (each forecast month is a model call)."""
    return max(low, min(high, int(value)))

def build_server(ws: Workspace, redact: bool = False) -> MCPServer:
    """Build an MCP server exposing the Payday Radar agent tools."""
    agent = Agent(ws)
    redactor = _Redactor(ws) if redact else None
    server = MCPServer(name="Payday Radar")

    def serialize(result: Any) -> str:
        if redactor is not None:
            return redactor.json(result)
        return json.dumps(result, ensure_ascii=False, default=str)

    @server.tool()
    def risky_invoices(top_n: int = 5) -> str:
        """List open invoices most likely to be paid late, with amounts and expected payment dates."""
        return serialize(agent.risky_invoices(top_n=_clamp(top_n, 1, 50)))

    @server.tool()
    def cashflow_forecast(months: int = 3, intervals: bool = False) -> str:
        """Forecast monthly cash flow; P10 is a pessimistic scenario, P50 median, P90 optimistic."""
        return serialize(
            agent.cashflow_forecast(months=_clamp(months, 1, 12), intervals=intervals)
        )

    @server.tool()
    def can_afford(amount: float, within_days: int = 90) -> str:
        """Check affordability using a pessimistic P10 balance scenario and cash cushion."""
        return serialize(agent.can_afford(amount=amount, within_days=within_days))

    @server.tool()
    def draft_reminder(invoice_id: str, lang: str = "en") -> str:
        """Draft a payment reminder for an open invoice ID in English or Norwegian."""
        return serialize(agent.draft_reminder(invoice_id=invoice_id, lang=lang))

    @server.tool()
    def anomalies() -> str:
        """Find unusual expenses and possible duplicate invoices."""
        return serialize(agent.anomalies())

    return server


__all__ = ["MCPServer", "build_server"]
