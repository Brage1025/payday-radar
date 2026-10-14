"""Time-ordered benchmark for late-payment predictions."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

from . import features, models, synthetic

METRICS = ("ROC-AUC", "Log-loss", "Brier score", "Days-late MAE")
DEFAULT_BACKENDS = ("sklearn", "tabpfn-api")
RESOLUTION_DAYS = 120  # longest terms + longest lateness: older invoices have resolved
_CLASSIFICATION_METRICS = METRICS[:3]


def _selected_backends(backends: str | Iterable[str] | None) -> list[str]:
    if backends is None:
        selected = list(DEFAULT_BACKENDS)
    elif isinstance(backends, str):
        selected = [backend.strip() for backend in backends.split(",") if backend.strip()]
    else:
        selected = list(backends)

    if not os.environ.get("TABPFN_TOKEN"):
        selected = [backend for backend in selected if backend != "tabpfn-api"]
    if not selected:
        raise ValueError("At least one available backend must be selected.")
    return list(dict.fromkeys(selected))


def _format_metric(values: list[float | None]) -> str:
    measured = [value for value in values if value is not None]
    if not measured:
        return "N/A"
    if not np.isfinite(measured).all():
        raise ValueError("Benchmark metrics must be finite.")
    return f"{np.mean(measured):.3f} ± {np.std(measured):.3f}"


def _render_table(results: dict[str, dict[str, list[float | None]]]) -> str:
    headers = ("Backend", *METRICS)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for backend, metrics in results.items():
        values = [backend, *(_format_metric(metrics[metric]) for metric in METRICS)]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_benchmark(
    seeds: int = 3,
    backends: str | Iterable[str] | None = None,
) -> str:
    """Run an offline-friendly benchmark and return its Markdown table."""
    if seeds < 1:
        raise ValueError("seeds must be at least 1.")
    selected = _selected_backends(backends)
    metrics_by_backend: dict[str, dict[str, list[float | None]]] = {}

    for backend in selected:
        metrics_by_backend[backend] = {metric: [] for metric in METRICS}
    metrics_by_backend["Naive: train late rate"] = {
        metric: [] for metric in METRICS
    }
    metrics_by_backend["Naive: train median days late"] = {
        metric: [] for metric in METRICS
    }

    unavailable: set[str] = set()
    for seed in range(seeds):
        invoices, _ = synthetic.generate(seed=seed)
        all_features = features.build_features(invoices, synthetic.DEFAULT_AS_OF)
        as_of = pd.Timestamp(synthetic.DEFAULT_AS_OF)
        paid = invoices[invoices["paid_date"].notna()].copy()
        paid["issue_date"] = pd.to_datetime(paid["issue_date"])
        paid["paid_date"] = pd.to_datetime(paid["paid_date"])
        paid["days_late"] = (paid["paid_date"] - pd.to_datetime(paid["due_date"])).dt.days
        # Keep only invoices old enough for their outcome to have resolved; otherwise recent
        # slow payers are missing from the data (right-censoring).
        paid = paid[paid["issue_date"] <= as_of - pd.Timedelta(days=RESOLUTION_DAYS)]
        paid = paid.sort_values("issue_date", kind="stable")

        split = int(len(paid) * 0.7)
        if split < 1 or split >= len(paid):
            raise ValueError("The paid-invoice dataset is too small for a 70/30 split.")
        train, test = paid.iloc[:split], paid.iloc[split:]
        # A training label is only known once the invoice is paid. Drop training invoices that
        # were still unpaid when the first test invoice was issued (they would leak the future).
        train = train[train["paid_date"] <= test["issue_date"].iloc[0]]

        X_train = models.ensure_numeric(all_features.loc[train["invoice_id"]])
        X_test = models.ensure_numeric(all_features.loc[test["invoice_id"]])
        y_late_train = (train["days_late"] > 0).astype(int)
        y_late_test = (test["days_late"] > 0).astype(int)
        y_days_train = train["days_late"].astype(float)
        y_days_test = test["days_late"].astype(float)
        late_rate = float(y_late_train.mean())
        median_days_late = float(y_days_train.median())

        baseline_probability = np.full(len(y_late_test), late_rate)
        baseline_probability = np.clip(baseline_probability, 1e-15, 1 - 1e-15)
        baseline_late = metrics_by_backend["Naive: train late rate"]
        baseline_late["ROC-AUC"].append(
            float(roc_auc_score(y_late_test, baseline_probability))
            if y_late_test.nunique() > 1
            else None
        )
        baseline_late["Log-loss"].append(
            float(log_loss(y_late_test, baseline_probability, labels=[0, 1]))
        )
        baseline_late["Brier score"].append(
            float(brier_score_loss(y_late_test, baseline_probability))
        )
        baseline_late["Days-late MAE"].append(None)
        baseline_regression = metrics_by_backend["Naive: train median days late"]
        for metric in _CLASSIFICATION_METRICS:
            baseline_regression[metric].append(None)
        baseline_regression["Days-late MAE"].append(
            float(mean_absolute_error(y_days_test, np.full(len(y_days_test), median_days_late)))
        )

        for backend in selected:
            if backend in unavailable:
                continue
            try:
                classifier = models.make_classifier(backend)
                regressor = models.make_regressor(backend)
                classifier.fit(X_train, y_late_train)
                predicted_proba = classifier.predict_proba(X_test)
                classes = list(classifier.classes_)
                probabilities = (
                    predicted_proba[:, classes.index(1)]
                    if 1 in classes
                    else np.zeros(len(y_late_test))
                )
                probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
                regressor.fit(X_train, y_days_train)
                predictions = regressor.predict(X_test)
            except (ImportError, RuntimeError) as exc:
                unavailable.add(backend)
                metrics_by_backend.pop(backend, None)
                warnings.warn(
                    f"Skipping unavailable benchmark backend {backend!r}: {exc}",
                    stacklevel=2,
                )
                continue

            metrics = metrics_by_backend[backend]
            metrics["ROC-AUC"].append(
                float(roc_auc_score(y_late_test, probabilities))
                if y_late_test.nunique() > 1
                else None
            )
            metrics["Log-loss"].append(
                float(log_loss(y_late_test, probabilities, labels=[0, 1]))
            )
            metrics["Brier score"].append(float(brier_score_loss(y_late_test, probabilities)))
            metrics["Days-late MAE"].append(
                float(mean_absolute_error(y_days_test, predictions))
            )

    return _render_table(metrics_by_backend)
