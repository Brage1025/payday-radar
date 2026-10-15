"""End-to-end MCP check without any desktop app.

Launches the Payday Radar MCP server as a separate process, speaks the real MCP protocol
over stdio (the same way Claude Desktop or any MCP client would), lists the tools, calls a
few, and checks that no real client or vendor name appears in the output.

Run from the repo root:
    python scripts/mcp_client_demo.py                     # sklearn backend, --redact
    python scripts/mcp_client_demo.py --backend tabpfn-api
    python scripts/mcp_client_demo.py --no-redact         # expect the leak check to FAIL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

# The MCP client launches the server with a tiny default environment (HOME/PATH/TERM only),
# so anything the server needs must be passed through explicitly.
PASS_THROUGH = ("TABPFN_TOKEN", "OLLAMA_HOST", "PAYDAY_TEXT_MODEL", "PAYDAY_VISION_MODEL")


class ToolError(RuntimeError):
    pass


def pretty(text: str, limit: int = 700) -> str:
    """Pretty-print a JSON text block, truncated for readable terminal output."""
    try:
        text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        pass
    return text if len(text) <= limit else text[:limit] + "\n  ..."


def real_names(data_dir: Path) -> set[str]:
    names: set[str] = set()
    inv, exp = data_dir / "invoices.csv", data_dir / "expenses.csv"
    if inv.exists():
        names |= set(pd.read_csv(inv)["client"].dropna())
    if exp.exists():
        names |= set(pd.read_csv(exp)["vendor"].dropna())
    return names


async def call(session: ClientSession, name: str, arguments: dict) -> list[str]:
    """Call a tool and return its text blocks; raise ToolError if the server reports an error."""
    result = await session.call_tool(name, arguments)
    texts = [block.text for block in result.content if hasattr(block, "text")]
    if result.is_error:
        raise ToolError(f"{name} failed: {' '.join(texts) or 'no details'}")
    return texts


async def run_calls(session: ClientSession, outputs: list[str]) -> None:
    tools = (await session.list_tools()).tools
    print("Tools exposed:", ", ".join(t.name for t in tools), "\n")

    risky = await call(session, "risky_invoices", {"top_n": 3})
    outputs += risky
    print("risky_invoices(top_n=3):\n" + pretty(risky[0]), "\n")

    forecast = await call(session, "cashflow_forecast", {"months": 3})
    outputs += forecast
    print("cashflow_forecast(months=3):\n" + pretty(forecast[0]), "\n")

    afford = await call(session, "can_afford", {"amount": 20000, "within_days": 60})
    outputs += afford
    print("can_afford(20000, 60 days):\n" + pretty(afford[0]), "\n")

    # Tools return one JSON text block; the invoice list is a JSON array inside it.
    rows = json.loads(risky[0]) if risky else []
    first = rows[0] if isinstance(rows, list) and rows else (rows if isinstance(rows, dict) else {})
    if "invoice_id" in first:
        reminder = await call(session, "draft_reminder", {"invoice_id": first["invoice_id"]})
        outputs += reminder
        print(f"draft_reminder({first['invoice_id']}):\n" + pretty(reminder[0]), "\n")

    anomalies = await call(session, "anomalies", {})
    outputs += anomalies
    print("anomalies():\n" + pretty(anomalies[0], 500), "\n")


def leak_check(data_dir: str, outputs: list[str]) -> int:
    names = real_names(Path(data_dir))
    if not names:
        print(f"(leak check skipped: no data files found in {data_dir})")
        return 0
    blob = "\n".join(outputs)
    leaked = sorted(n for n in names if n in blob)
    if leaked:
        print("❌ Real names found in tool output:", leaked)
        return 1
    print(f"✅ Leak check passed: none of {len(names)} real client/vendor names appear in any tool output.")
    return 0


async def main(args: argparse.Namespace) -> int:
    cmd = ["-m", "paydayradar", "--data", args.data, "--backend", args.backend, "mcp"]
    if args.redact:
        cmd.append("--redact")
    env = {**get_default_environment(), **{k: os.environ[k] for k in PASS_THROUGH if k in os.environ}}
    params = StdioServerParameters(command=sys.executable, args=cmd, env=env)
    outputs: list[str] = []
    failure: ToolError | None = None

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                await run_calls(session, outputs)
            except ToolError as exc:
                failure = exc

    if failure:
        print(f"\n❌ {failure}\n")
        if args.backend.startswith("tabpfn") and "TABPFN_TOKEN" not in os.environ:
            print("   TABPFN_TOKEN is not set in this terminal. Set it, then rerun.\n")
        print("   Tool errors hide the details. To see the real traceback, run the same work directly:")
        print(f"   python -m paydayradar --backend {args.backend} score")
        return 1
    return leak_check(args.data, outputs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--backend", default="sklearn")
    p.add_argument("--no-redact", dest="redact", action="store_false")
    raise SystemExit(asyncio.run(main(p.parse_args())))