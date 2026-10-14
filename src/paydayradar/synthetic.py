"""Synthetic freelancer data, so the demo never needs real client data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_AS_OF = "2026-10-01"
DEFAULT_OPENING_BALANCE = 85_000.0

# name, typical invoice (NOK), invoices/month, payment terms (days), lateness mean/sd (days)
_CLIENTS = [
    ("Fjordly AS", 24000, 0.9, 14, -2, 2.0),
    ("Bryggen Bryggeri", 18000, 0.8, 30, 6, 4.0),
    ("Nordlys Studio", 9500, 1.0, 30, 1, 3.0),
    ("Vestland Vekst", 42000, 0.5, 30, 22, 9.0),
    ("Salmon & Co", 31000, 0.6, 45, 12, 6.0),
    ("Fløyen Digital", 15000, 0.9, 14, 0, 2.0),
    ("Hordaland Hydro Services", 56000, 0.4, 45, 28, 12.0),
    ("Kaffebrenneriet Nord", 6200, 1.0, 14, -1, 1.5),
    ("Havn Logistikk", 27500, 0.6, 30, 9, 5.0),
    ("Troll Tech", 38000, 0.5, 30, 3, 3.0),
]

_VENDORS = {
    "software": ["Notion", "GitHub", "Figma", "Adobe"],
    "travel": ["Widerøe", "Vy", "Scandic"],
    "hardware": ["Elkjøp", "Apple"],
    "marketing": ["Meta Ads", "Google Ads"],
}


def generate(
    seed: int = 42, months: int = 30, as_of: str = DEFAULT_AS_OF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (invoices, expenses). Open invoices have a NaT paid_date."""
    rng = np.random.default_rng(seed)
    as_of_ts = pd.Timestamp(as_of)
    starts = pd.date_range(end=as_of_ts - pd.offsets.MonthBegin(1), periods=months, freq="MS")

    inv_rows = []
    n = 0
    for start in starts:
        for name, base, rate, terms, mu, sigma in _CLIENTS:
            for _ in range(rng.poisson(rate)):
                n += 1
                issue = start + pd.Timedelta(days=rng.integers(0, 28))
                amount = round(float(base * rng.lognormal(0, 0.35)), -1)
                season = 5 if issue.month in (7, 12) else 0  # holiday slowdowns
                late = max(int(round(rng.normal(mu, sigma))) + season, -7)
                due = issue + pd.Timedelta(days=terms)
                paid = due + pd.Timedelta(days=late)
                inv_rows.append(
                    {
                        "invoice_id": f"INV-{n:04d}",
                        "client": name,
                        "issue_date": issue,
                        "due_date": due,
                        "paid_date": paid if paid <= as_of_ts else pd.NaT,
                        "amount": amount,
                        "terms_days": terms,
                        "injected_duplicate": False,
                    }
                )
    invoices = pd.DataFrame(inv_rows)

    # Inject two duplicate invoices among the most recent ones (ground truth for the demo).
    recent = invoices[invoices["issue_date"] >= as_of_ts - pd.Timedelta(days=60)]
    dups = []
    for idx in rng.choice(recent.index, size=min(2, len(recent)), replace=False):
        n += 1
        dup = invoices.loc[idx].copy()
        dup["invoice_id"] = f"INV-{n:04d}"
        dup["issue_date"] = dup["issue_date"] + pd.Timedelta(days=2)
        dup["due_date"] = dup["due_date"] + pd.Timedelta(days=2)
        dup["paid_date"] = pd.NaT
        dup["injected_duplicate"] = True
        dups.append(dup)
    if dups:
        invoices = pd.concat([invoices, pd.DataFrame(dups)], ignore_index=True)
    invoices = invoices.sort_values("issue_date").reset_index(drop=True)

    exp_rows = []
    for start in starts:
        def add(day, vendor, category, amount, start=start):
            exp_rows.append(
                {
                    "date": start + pd.Timedelta(days=int(day)),
                    "vendor": vendor,
                    "category": category,
                    "amount": round(float(amount), 0),
                    "is_anomaly": False,
                }
            )

        add(1, "Bergen Cowork", "coworking", 3200)
        add(24, "Owner", "owner_draw", rng.normal(85000, 3000))  # the freelancer's own salary
        add(4, "Regnskap Partner", "accounting", rng.normal(1900, 80))
        if start.month % 2 == 1:
            add(14, "Skatteetaten", "tax", rng.normal(14000, 1200))
        for _ in range(int(rng.integers(2, 5))):
            add(rng.integers(0, 28), rng.choice(_VENDORS["software"]), "software", rng.uniform(90, 600))
        for cat, rate, lo, hi in [
            ("travel", 1.0, 400, 4500),
            ("hardware", 0.3, 3000, 14000),
            ("marketing", 0.5, 500, 3500),
        ]:
            for _ in range(rng.poisson(rate)):
                add(rng.integers(0, 28), rng.choice(_VENDORS[cat]), cat, rng.uniform(lo, hi))
    expenses = pd.DataFrame(exp_rows).sort_values("date").reset_index(drop=True)

    # Inject unusual expenses (ground truth for anomaly detection).
    candidates = expenses[expenses["category"].isin(["software", "marketing", "travel"])].index
    for idx in rng.choice(candidates, size=5, replace=False):
        expenses.loc[idx, "amount"] = round(float(expenses.loc[idx, "amount"]) * 10, 0)
        expenses.loc[idx, "is_anomaly"] = True
    return invoices, expenses


def save(
    invoices: pd.DataFrame,
    expenses: pd.DataFrame,
    out_dir: str | Path,
    as_of: str = DEFAULT_AS_OF,
    opening_balance: float = DEFAULT_OPENING_BALANCE,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    invoices.to_csv(out / "invoices.csv", index=False)
    expenses.to_csv(out / "expenses.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps({"as_of": as_of, "opening_balance": opening_balance, "currency": "NOK"})
    )
    return out
