"""Workspace: scoring, forecasting and anomaly detection on top of TabPFN."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .features import FEATURE_COLUMNS, build_features, prepare_invoices
from .models import ensure_numeric, make_classifier, make_regressor, resolve_backend


@dataclass
class Workspace:
    invoices: pd.DataFrame
    expenses: pd.DataFrame
    as_of: pd.Timestamp
    opening_balance: float = 0.0
    backend: str = "auto"
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.as_of = pd.Timestamp(self.as_of)
        self.invoices = prepare_invoices(self.invoices)
        self.expenses = self.expenses.copy()
        self.expenses["date"] = pd.to_datetime(self.expenses["date"])
        self.backend = resolve_backend(self.backend)

    @classmethod
    def from_dir(cls, path: str | Path, backend: str = "auto") -> Workspace:
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        return cls(
            invoices=pd.read_csv(p / "invoices.csv"),
            expenses=pd.read_csv(p / "expenses.csv"),
            as_of=meta["as_of"],
            opening_balance=float(meta.get("opening_balance", 0.0)),
            backend=backend,
        )

    # ------------------------------------------------------------------ scoring
    def score_open_invoices(self) -> pd.DataFrame:
        """Predict P(late) and days late for every unpaid invoice."""
        if "scored" in self._cache:
            return self._cache["scored"]
        inv = self.invoices
        feats = ensure_numeric(build_features(inv, self.as_of))
        closed_ids = inv.loc[inv["paid_date"].notna(), "invoice_id"]
        open_inv = inv[inv["paid_date"].isna()].copy()
        if open_inv.empty:
            return open_inv
        X_train = feats.loc[closed_ids]
        days = inv.set_index("invoice_id").loc[closed_ids, "days_late"]
        y_late = (days > 0).astype(int)
        X_open = feats.loc[open_inv["invoice_id"]]

        if y_late.nunique() < 2:
            p_late = np.full(len(open_inv), float(y_late.iloc[0]))
        else:
            clf = make_classifier(self.backend)
            clf.fit(X_train, y_late)
            col = list(clf.classes_).index(1)
            p_late = np.asarray(clf.predict_proba(X_open))[:, col]
        reg = make_regressor(self.backend)
        reg.fit(X_train, days)
        exp_days = np.asarray(reg.predict(X_open), dtype=float)

        open_inv["days_overdue"] = (self.as_of - open_inv["due_date"]).dt.days.clip(lower=0)
        overdue = open_inv["days_overdue"] > 0
        open_inv["p_late"] = np.where(overdue, 1.0, p_late).round(3)
        open_inv["expected_days_late"] = np.maximum(exp_days, open_inv["days_overdue"]).round(0)
        open_inv["expected_pay_date"] = open_inv["due_date"] + pd.to_timedelta(
            open_inv["expected_days_late"], unit="D"
        )
        open_inv["risk"] = pd.cut(
            open_inv["p_late"], [-0.01, 0.35, 0.65, 1.01], labels=["low", "medium", "high"]
        ).astype(str)
        open_inv["amount_at_risk"] = (open_inv["amount"] * open_inv["p_late"]).round(0)
        cols = [
            "invoice_id", "client", "amount", "due_date", "days_overdue", "p_late",
            "expected_days_late", "expected_pay_date", "risk", "amount_at_risk",
        ]
        scored = open_inv[cols].sort_values("amount_at_risk", ascending=False).reset_index(drop=True)
        self._cache["scored"] = scored
        return scored

    # ----------------------------------------------------------------- forecast
    def monthly_series(self) -> pd.DataFrame:
        """Complete calendar months before `as_of`: income (by paid date) and expenses."""
        last = self.as_of.to_period("M") - 1
        paid = self.invoices[self.invoices["paid_date"].notna()]
        inc = paid.groupby(paid["paid_date"].dt.to_period("M"))["amount"].sum()
        exp = self.expenses.groupby(self.expenses["date"].dt.to_period("M"))["amount"].sum()
        first = min(inc.index.min(), exp.index.min())
        idx = pd.period_range(first, last, freq="M")
        df = pd.DataFrame({"income": inc.reindex(idx, fill_value=0.0),
                           "expenses": exp.reindex(idx, fill_value=0.0)})
        df.index = idx.to_timestamp()
        return df

    @staticmethod
    def _lag_row(hist: list[float], month: int, t: int) -> list[float]:
        return [month, t, hist[-1], hist[-2], hist[-3], float(np.mean(hist[-3:]))]

    def _forecast_target(self, values: pd.Series, months: pd.DatetimeIndex, horizon: int):
        hist = list(values.astype(float))
        if len(hist) < 8:
            return [float(np.mean(hist))] * horizon
        X, y = [], []
        for i in range(3, len(hist)):
            X.append(self._lag_row(hist[:i], values.index[i].month, i))
            y.append(hist[i])
        cols = ["month", "t", "lag1", "lag2", "lag3", "roll3"]
        X = ensure_numeric(pd.DataFrame(X, columns=cols))
        model = make_regressor(self.backend)
        model.fit(X, pd.Series(y))
        out = []
        for step in range(horizon):
            row = pd.DataFrame(
                [self._lag_row(hist, months[step].month, len(hist))], columns=cols
            )
            pred = float(np.asarray(model.predict(row))[0])
            pred = max(pred, 0.0)
            out.append(pred)
            hist.append(pred)
        return out

    def forecast(self, months: int = 3) -> pd.DataFrame:
        """Recursive monthly income/expense forecast with a running balance."""
        key = ("forecast", months)
        if key in self._cache:
            return self._cache[key]
        series = self.monthly_series()
        future = pd.date_range(self.as_of.to_period("M").to_timestamp(), periods=months, freq="MS")
        inc = self._forecast_target(series["income"], future, months)
        exp = self._forecast_target(series["expenses"], future, months)
        df = pd.DataFrame({"month": future, "forecast_income": inc, "forecast_expenses": exp})
        df["net"] = df["forecast_income"] - df["forecast_expenses"]
        df["balance"] = self.opening_balance + df["net"].cumsum()
        num = df.columns.drop("month")
        df[num] = df[num].round(0)
        self._cache[key] = df
        return df

    # ---------------------------------------------------------------- anomalies
    def find_anomalies(self, z_threshold: float = 3.5) -> dict[str, pd.DataFrame]:
        """Unusual expenses (TabPFN predict-and-compare) and likely duplicate invoices."""
        exp = self.expenses.reset_index(drop=True).copy()
        X = pd.DataFrame(
            {
                "category_code": exp["category"].astype("category").cat.codes,
                "vendor_code": exp["vendor"].astype("category").cat.codes,
                "month": exp["date"].dt.month,
                "dow": exp["date"].dt.dayofweek,
            }
        ).astype(float)
        ensure_numeric(X)
        y = np.log1p(exp["amount"].astype(float))
        pred = np.zeros(len(exp))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            model = make_regressor(self.backend)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = np.asarray(model.predict(X.iloc[te]), dtype=float)
        resid = y.to_numpy() - pred
        med = np.median(resid)
        mad = max(np.median(np.abs(resid - med)), 0.05)
        exp["robust_z"] = (0.6745 * (resid - med) / mad).round(2)
        exp["expected_amount"] = np.expm1(pred).round(0)
        flagged = exp[exp["robust_z"].abs() > z_threshold]
        flagged = flagged.sort_values("robust_z", key=abs, ascending=False)

        inv = self.invoices.sort_values("issue_date")
        dups = []
        for (_, _), g in inv.groupby(["client", "amount"]):
            gaps = g["issue_date"].diff().dt.days
            dups.append(g[gaps <= 7])
        dup_df = pd.concat(dups) if dups else inv.iloc[0:0]
        dup_df = dup_df[["invoice_id", "client", "amount", "issue_date"]].reset_index(drop=True)
        return {"expenses": flagged.reset_index(drop=True), "duplicates": dup_df}


__all__ = ["FEATURE_COLUMNS", "Workspace"]
