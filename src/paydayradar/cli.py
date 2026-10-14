from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import benchmark, synthetic
from .agent import Agent
from .pipeline import Workspace
from .reminders import draft_reminder
from .report import render_report

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def _workspace(args) -> Workspace:
    data = Path(args.data)
    if not (data / "invoices.csv").exists():
        inv, exp = synthetic.generate(seed=args.seed)
        synthetic.save(inv, exp, data)
        print(f"[generated synthetic data in {data}/]")
    return Workspace.from_dir(data, backend=args.backend)


def cmd_generate(args):
    inv, exp = synthetic.generate(seed=args.seed)
    out = synthetic.save(inv, exp, args.data)
    print(f"Wrote {len(inv)} invoices and {len(exp)} expenses to {out}/")


def cmd_score(args):
    ws = _workspace(args)
    print(f"Backend: {ws.backend}\n")
    print(_fmt(ws.score_open_invoices().drop(columns=["due_date", "expected_pay_date"])))


def cmd_forecast(args):
    ws = _workspace(args)
    print(f"Backend: {ws.backend}\n")
    print(_fmt(ws.forecast(args.months).assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))))


def cmd_anomalies(args):
    ws = _workspace(args)
    res = ws.find_anomalies()
    print("Unusual expenses:\n" + _fmt(res["expenses"][
        ["date", "vendor", "category", "amount", "expected_amount", "robust_z"]]))
    print("\nPossible duplicate invoices:\n" + _fmt(res["duplicates"]))


def cmd_ask(args):
    print(Agent(_workspace(args)).ask(" ".join(args.question)))


def cmd_demo(args):
    ws = _workspace(args)
    print(f"=== Payday Radar demo | as of {ws.as_of.date()} | backend: {ws.backend} ===\n")
    scored = ws.score_open_invoices()
    print("Open invoices by amount at risk:")
    print(_fmt(scored.drop(columns=["due_date", "expected_pay_date"]).head(8)))
    print("\n90-day forecast (NOK):")
    print(_fmt(ws.forecast(3).assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))))
    top = scored.iloc[0].to_dict()
    d = draft_reminder(top)
    print(f"\nDraft reminder for {top['client']}:\n{d['subject']}\n\n{d['body']}")
    q = "Can I afford a 20000 laptop in November?"
    print(f"\nQ: {q}\nA: {Agent(ws).ask(q)}")
    if args.out:
        Path(args.out).write_text(render_report(ws))
        print(f"\n[report written to {args.out}]")


def cmd_benchmark(args):
    table = benchmark.run_benchmark(seeds=args.seeds, backends=args.backends)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n[benchmark written to {out}]")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="payday-radar", description=__doc__)
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "tabpfn-api", "tabpfn-local", "sklearn"])
    p.add_argument("--seed", type=int, default=42)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate").set_defaults(fn=cmd_generate)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    f = sub.add_parser("forecast"); f.add_argument("--months", type=int, default=3)
    f.set_defaults(fn=cmd_forecast)
    sub.add_parser("anomalies").set_defaults(fn=cmd_anomalies)
    a = sub.add_parser("ask"); a.add_argument("question", nargs="+"); a.set_defaults(fn=cmd_ask)
    d = sub.add_parser("demo"); d.add_argument("--out"); d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("benchmark")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--backends", default="sklearn,tabpfn-api")
    b.add_argument("--out", default="docs/RESULTS.md")
    b.set_defaults(fn=cmd_benchmark)
    args = p.parse_args(argv)
    args.fn(args)
    return 0
