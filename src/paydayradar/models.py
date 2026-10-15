"""Model backends. TabPFN is the default; scikit-learn is an offline/CI fallback.

Backends:
  tabpfn-api    hosted TabPFN-3.5 via `tabpfn-client` (needs TABPFN_TOKEN)
  tabpfn-local  open weights via the `tabpfn` package (can run fully offline)
  sklearn       random forest fallback so tests and CI need no token or GPU
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

BACKENDS = ("auto", "tabpfn-api", "tabpfn-local", "sklearn")


class PrivacyError(ValueError):
    """Raised when non-numeric (possibly identifying) data would reach a model."""


def ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Guard: only numeric columns may ever be handed to a model backend."""
    bad = list(X.select_dtypes(exclude="number").columns)
    if bad:
        raise PrivacyError(f"Refusing to send non-numeric columns to a model: {bad}")
    return X


def predict_quantiles(model, X: pd.DataFrame, quantiles: Sequence[float]) -> np.ndarray:
    """Return sorted, non-crossing predictions with shape (rows, quantiles)."""
    X = ensure_numeric(X)
    quantile_levels = list(quantiles)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    if (
        isinstance(model, Pipeline)
        and model.steps
        and isinstance(model.steps[-1][1], RandomForestRegressor)
    ):
        forest = model.steps[-1][1]
        transformed = model[:-1].transform(X) if len(model.steps) > 1 else X
        tree_predictions = np.asarray(
            [tree.predict(transformed) for tree in forest.estimators_], dtype=float
        )
        predictions = np.quantile(tree_predictions, quantile_levels, axis=0).T
    else:
        predictions = np.column_stack(
            model.predict(X, output_type="quantiles", quantiles=quantile_levels)
        )

    expected_shape = (len(X), len(quantile_levels))
    if predictions.shape != expected_shape:
        raise ValueError(
            f"Quantile predictions must have shape {expected_shape}, got {predictions.shape}"
        )
    return np.sort(predictions, axis=1)


def resolve_backend(backend: str = "auto") -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if backend != "auto":
        return backend
    if os.environ.get("TABPFN_TOKEN"):
        try:
            import tabpfn_client  # noqa: F401

            return "tabpfn-api"
        except ImportError:
            pass
    try:
        import tabpfn  # noqa: F401

        return "tabpfn-local"
    except ImportError:
        pass
    warnings.warn("TabPFN not installed; using scikit-learn fallback.", stacklevel=2)
    return "sklearn"


def _tabpfn_api(kind: str):
    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
    cls = tabpfn_client.TabPFNClassifier if kind == "classifier" else tabpfn_client.TabPFNRegressor
    try:
        return cls.create_default_for_version("v3.5")
    except AttributeError:  # older client
        return cls()


def _tabpfn_local(kind: str):
    import tabpfn

    cls = tabpfn.TabPFNClassifier if kind == "classifier" else tabpfn.TabPFNRegressor
    try:
        from tabpfn.constants import ModelVersion

        return cls.create_default_for_version(ModelVersion.V3_5)
    except (ImportError, AttributeError):  # older package
        return cls()


def _sklearn(kind: str):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    est = (
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
        if kind == "classifier"
        else RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)
    )
    return make_pipeline(SimpleImputer(strategy="median"), est)


def _make(kind: str, backend: str):
    backend = resolve_backend(backend)
    factory = {"tabpfn-api": _tabpfn_api, "tabpfn-local": _tabpfn_local, "sklearn": _sklearn}
    return factory[backend](kind)


def make_classifier(backend: str = "auto"):
    return _make("classifier", backend)


def make_regressor(backend: str = "auto"):
    return _make("regressor", backend)
