import asyncio
import json
import re

import pytest

mcp = pytest.importorskip("mcp")

from paydayradar import synthetic  # noqa: E402
from paydayradar.mcp_server import build_server  # noqa: E402
from paydayradar.pipeline import Workspace  # noqa: E402


@pytest.fixture
def mcp_workspace():
    invoices, expenses = synthetic.generate(seed=42)
    return Workspace(
        invoices,
        expenses,
        as_of=synthetic.DEFAULT_AS_OF,
        opening_balance=synthetic.DEFAULT_OPENING_BALANCE,
        backend="sklearn",
    )


def _tool_outputs(server, ws):
    arguments = {
        "risky_invoices": {"top_n": 5},
        "cashflow_forecast": {"months": 3},
        "can_afford": {"amount": 20000, "within_days": 90},
        "draft_reminder": {
            "invoice_id": ws.score_open_invoices().iloc[0]["invoice_id"],
            "lang": "en",
        },
        "anomalies": {},
    }
    outputs = {}
    for name, args in arguments.items():
        result = asyncio.run(server.call_tool(name, args))
        assert result.content and result.content[0].text
        outputs[name] = "\n".join(block.text for block in result.content)
        json.loads(outputs[name])
    return outputs


def test_mcp_tools(mcp_workspace):
    server = build_server(mcp_workspace)
    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == {
        "risky_invoices",
        "cashflow_forecast",
        "can_afford",
        "draft_reminder",
        "anomalies",
    }
    assert set(_tool_outputs(server, mcp_workspace)) == {tool.name for tool in tools}


def test_mcp_redacts_client_and_vendor_names(mcp_workspace):
    real_names = set(mcp_workspace.invoices["client"]) | set(mcp_workspace.expenses["vendor"])
    plain = "\n".join(_tool_outputs(build_server(mcp_workspace), mcp_workspace).values())
    redacted = "\n".join(
        _tool_outputs(build_server(mcp_workspace, redact=True), mcp_workspace).values()
    )
    # Positive controls: the leak check must not pass because outputs were empty.
    assert any(name in plain for name in real_names)
    assert re.search(r"Client [A-Z]\b", redacted) and re.search(r"Vendor [A-Z]\b", redacted)
    assert all(name not in redacted for name in real_names)


def test_mcp_cli_options_after_subcommand(monkeypatch, mcp_workspace):
    from paydayradar import cli, mcp_server

    options = {}

    def workspace(args, *, announce=True):
        options["data"] = args.data
        options["backend"] = args.backend
        options["announce"] = announce
        return mcp_workspace

    class StubServer:
        def run(self, *, transport):
            options["transport"] = transport

    def build_server_stub(ws, *, redact):
        options["workspace"] = ws
        options["redact"] = redact
        return StubServer()

    monkeypatch.setattr(cli, "_workspace", workspace)
    monkeypatch.setattr(mcp_server, "build_server", build_server_stub)

    assert cli.main(
        ["mcp", "--data", "private-data", "--backend", "sklearn", "--redact"]
    ) == 0
    assert options == {
        "data": "private-data",
        "backend": "sklearn",
        "announce": False,
        "workspace": mcp_workspace,
        "redact": True,
        "transport": "stdio",
    }

def test_mcp_clamps_unreasonable_sizes(mcp_workspace):
    server = build_server(mcp_workspace)
    months = asyncio.run(server.call_tool("cashflow_forecast", {"months": 60}))
    assert len(json.loads(months.content[0].text)) == 12
    top = asyncio.run(server.call_tool("risky_invoices", {"top_n": 10_000}))
    assert len(json.loads(top.content[0].text)) <= 50