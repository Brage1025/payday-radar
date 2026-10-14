import math
import re

from paydayradar.benchmark import run_benchmark


def test_sklearn_benchmark_is_offline_and_reports_finite_metrics(monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)

    table = run_benchmark(seeds=2)

    assert "| Backend | ROC-AUC | Log-loss | Brier score | Days-late MAE |" in table
    assert "| sklearn |" in table
    assert "tabpfn-api" not in table
    assert "| Naive: train late rate |" in table
    assert "| Naive: train median days late |" in table
    values = re.findall(r"(-?\d+\.\d+) ± (-?\d+\.\d+)", table)
    assert values
    assert all(math.isfinite(float(value)) for pair in values for value in pair)
