"""Leak-free, numeric-only features for invoice payment prediction.

PRIVACY: client names are used only to look up history locally. The returned
frame contains numbers only, which is all that is ever sent to a remote model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "terms_days",
    "issue_month",
    "issue_dow",
    "client_prior_n",
    "client_prior_mean_late",
    "client_prior_max_late",
    "client_prior_late_rate",
    "client_prior_last_late",
    "amount_vs_client_median",
]


def prepare_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
    """Ensure date dtypes and a days_late column (NaN for open invoices)."""
    inv = invoices.copy()
    for col in ("issue_date", "due_date", "paid_date"):
        inv[col] = pd.to_datetime(inv[col])
    inv["days_late"] = (inv["paid_date"] - inv["due_date"]).dt.days.astype(float)
    return inv


def build_features(invoices: pd.DataFrame, as_of) -> pd.DataFrame:
    """One row per invoice (indexed by invoice_id).

    Client history only includes invoices paid *before* the reference date:
    the issue date for closed invoices, `as_of` for open ones.
    """
    as_of = pd.Timestamp(as_of)
    inv = prepare_invoices(invoices).sort_values("issue_date").reset_index(drop=True)
    closed = inv[inv["paid_date"].notna()]

    rows = []
    for r in inv.itertuples(index=False):
        ref = r.issue_date if pd.notna(r.paid_date) else as_of
        h = closed[
            (closed["client"] == r.client)
            & (closed["paid_date"] < ref)
            & (closed["invoice_id"] != r.invoice_id)
        ].sort_values("paid_date")
        late = h["days_late"]
        has_hist = len(h) > 0
        rows.append(
            {
                "amount": r.amount,
                "log_amount": float(np.log1p(r.amount)),
                "terms_days": r.terms_days,
                "issue_month": r.issue_date.month,
                "issue_dow": r.issue_date.dayofweek,
                "client_prior_n": len(h),
                "client_prior_mean_late": late.mean() if has_hist else np.nan,
                "client_prior_max_late": late.max() if has_hist else np.nan,
                "client_prior_late_rate": (late > 0).mean() if has_hist else np.nan,
                "client_prior_last_late": late.iloc[-1] if has_hist else np.nan,
                "amount_vs_client_median": r.amount / h["amount"].median() if has_hist else np.nan,
            }
        )
    out = pd.DataFrame(rows, index=inv["invoice_id"])[FEATURE_COLUMNS].astype(float)
    return out
