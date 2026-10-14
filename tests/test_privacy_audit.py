"""Privacy audit: nothing identifying may ever reach a model backend.

Every frame passed to fit/predict is audited the moment it arrives, so a leak fails with a
clear message instead of depending on the estimator crashing. (TabPFN-3.5 accepts text
columns, so a string leak would NOT crash it; this test is the safety net.)
"""

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from paydayradar import pipeline, synthetic
from paydayradar.models import make_classifier as real_classifier
from paydayradar.models import make_regressor as real_regressor


def _audit(frame, identifiers, where):
    assert isinstance(frame, pd.DataFrame), f"{where}: expected a DataFrame"
    bad = [str(c) for c, dt in frame.dtypes.items() if not is_numeric_dtype(dt)]
    assert not bad, f"{where}: non-numeric columns reached the model: {bad}"
    leaked_cols = identifiers & {str(c) for c in frame.columns}
    assert not leaked_cols, f"{where}: identifier used as a column name: {leaked_cols}"
    leaked_vals = {
        v for v in frame.to_numpy(dtype=object).ravel() if isinstance(v, str) and v in identifiers
    }
    assert not leaked_vals, f"{where}: identifier found in cell values: {leaked_vals}"


class RecordingEstimator:
    def __init__(self, kind, estimator, identifiers, log):
        self.kind, self.estimator, self.identifiers, self.log = kind, estimator, identifiers, log

    def _check(self, X, method):
        _audit(X, self.identifiers, f"{self.kind}.{method}")
        self.log.append((self.kind, method))

    def fit(self, X, y):
        self._check(X, "fit")
        self.estimator.fit(X, y)
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        return self

    def predict(self, X):
        self._check(X, "predict")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        self._check(X, "predict_proba")
        return self.estimator.predict_proba(X)


def test_pipeline_model_inputs_contain_no_client_or_vendor_names(monkeypatch, data):
    invoices, expenses = data
    identifiers = set(invoices["client"]) | set(expenses["vendor"])
    log = []

    monkeypatch.setattr(
        pipeline, "make_classifier",
        lambda backend: RecordingEstimator("classifier", real_classifier("sklearn"), identifiers, log),
    )
    monkeypatch.setattr(
        pipeline, "make_regressor",
        lambda backend: RecordingEstimator("regressor", real_regressor("sklearn"), identifiers, log),
    )

    ws = pipeline.Workspace(
        invoices, expenses, as_of=synthetic.DEFAULT_AS_OF,
        opening_balance=synthetic.DEFAULT_OPENING_BALANCE, backend="sklearn",
    )
    ws.score_open_invoices()
    ws.forecast(3)
    ws.find_anomalies()

    assert ("classifier", "fit") in log and ("classifier", "predict_proba") in log
    assert ("regressor", "predict") in log
    # 1 (late days) + 2 (income, expenses forecast) + 5 (anomaly CV folds) = all three stages ran
    assert log.count(("regressor", "fit")) >= 8


def test_audit_helper_catches_leaks():
    ids = {"Fjordly AS"}
    with pytest.raises(AssertionError, match="non-numeric"):
        _audit(pd.DataFrame({"client": ["Fjordly AS"], "x": [1.0]}), ids, "t")
    with pytest.raises(AssertionError, match="column name"):
        _audit(pd.DataFrame({"Fjordly AS": [1.0]}), ids, "t")
    _audit(pd.DataFrame({"x": [1.0, 2.0]}), ids, "t")  # clean frame passes