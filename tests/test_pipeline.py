import pandas as pd
import pytest

from paydayradar.features import FEATURE_COLUMNS, build_features
from paydayradar.models import PrivacyError, ensure_numeric


def test_synthetic_has_open_and_paid(data):
    inv, exp = data
    assert inv["paid_date"].isna().any() and inv["paid_date"].notna().any()
    assert exp["is_anomaly"].sum() == 5


def test_features_are_numeric_only_and_leak_free(data):
    inv, _ = data
    feats = build_features(inv, "2026-10-01")
    assert list(feats.columns) == FEATURE_COLUMNS
    ensure_numeric(feats)  # must not raise
    first = inv.sort_values("issue_date").iloc[0]
    assert feats.loc[first["invoice_id"], "client_prior_n"] == 0  # no history yet


def test_privacy_guard_blocks_names():
    with pytest.raises(PrivacyError):
        ensure_numeric(pd.DataFrame({"client": ["Fjordly AS"], "amount": [1.0]}))


def test_scoring(ws):
    scored = ws.score_open_invoices()
    assert len(scored) == ws.invoices["paid_date"].isna().sum()
    assert scored["p_late"].between(0, 1).all()
    assert (scored["amount_at_risk"].diff().dropna() <= 0).all()  # sorted desc


def test_forecast(ws):
    fc = ws.forecast(3)
    assert len(fc) == 3 and (fc["forecast_income"] >= 0).all()
    assert fc["balance"].iloc[0] == ws.opening_balance + fc["net"].iloc[0]


def test_anomalies_recall_and_duplicates(ws):
    res = ws.find_anomalies()
    truth = ws.expenses[ws.expenses["is_anomaly"]]
    key = lambda d: set(zip(d["date"], d["vendor"], d["amount"]))
    recall = len(key(res["expenses"]) & key(truth)) / len(truth)
    assert recall >= 0.6, recall
    injected = set(ws.invoices.loc[ws.invoices["injected_duplicate"], "invoice_id"])
    assert injected <= set(res["duplicates"]["invoice_id"])
