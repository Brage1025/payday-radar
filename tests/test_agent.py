from paydayradar import llm
from paydayradar.agent import Agent


def test_router_afford(ws, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda timeout=0.5: False)
    out = Agent(ws).ask("Can I afford a 20000 laptop in November?")
    assert "verdict" in out


def test_router_risk_and_help(ws, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda timeout=0.5: False)
    agent = Agent(ws)
    assert "p_late" in agent.ask("Which invoices are at risk?")
    assert "Try:" in agent.ask("what is the weather")


def test_reminder_tool(ws):
    top = ws.score_open_invoices().iloc[0]["invoice_id"]
    d = Agent(ws).draft_reminder(top, lang="no")
    assert "faktura" in d["subject"].lower()
