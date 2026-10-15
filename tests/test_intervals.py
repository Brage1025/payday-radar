import numpy as np
import pandas as pd

from paydayradar.models import make_regressor, predict_quantiles


def test_predict_quantiles_sklearn_shape_sorted_and_finite():
    X = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float) ** 2})
    model = make_regressor("sklearn").fit(X, np.arange(20, dtype=float))

    predictions = predict_quantiles(model, X.iloc[:4], [0.1, 0.5, 0.9])

    assert predictions.shape == (4, 3)
    assert np.isfinite(predictions).all()
    assert (np.diff(predictions, axis=1) >= 0).all()


def test_predict_quantiles_tabpfn_output_sorted_when_quantiles_cross():
    class FakeTabPFN:
        def predict(self, X, output_type=None, quantiles=None):
            assert output_type == "quantiles"
            assert quantiles == [0.1, 0.5, 0.9]
            return [
                np.array([3.0, 4.0]),
                np.array([2.0, 2.0]),
                np.array([1.0, 5.0]),
            ]

    predictions = predict_quantiles(
        FakeTabPFN(), pd.DataFrame({"feature": [0.0, 1.0]}), [0.1, 0.5, 0.9]
    )

    assert predictions.shape == (2, 3)
    np.testing.assert_array_equal(predictions, [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]])


def test_predict_quantiles_rejects_wrong_shape():
    class WrongShape:
        def predict(self, X, output_type=None, quantiles=None):
            return [np.array([1.0])]

    try:
        predict_quantiles(WrongShape(), pd.DataFrame({"feature": [0.0, 1.0]}), [0.1, 0.9])
    except ValueError as exc:
        assert "must have shape" in str(exc)
    else:
        raise AssertionError("wrong-shaped quantile output was accepted")


def test_short_history_uses_empirical_quantiles_each_step(ws):
    history = pd.Series(
        np.arange(7, dtype=float),
        index=pd.date_range("2026-01-01", periods=7, freq="MS"),
    )
    future = pd.date_range("2026-08-01", periods=2, freq="MS")

    point, quantiles = ws._forecast_target(history, future, 2, intervals=True)

    assert point == [3.0, 3.0]
    np.testing.assert_array_equal(quantiles[0], quantiles[1])
    np.testing.assert_array_equal(quantiles[0], np.quantile(history, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))


def test_forecast_intervals_are_ordered_unchanged_and_deterministic(ws):
    forecast = ws.forecast(3)

    with_intervals = ws.forecast(3, intervals=True, seed=17)
    repeated = ws.forecast(3, intervals=True, seed=17)

    interval_columns = ["balance_p10", "balance_p50", "balance_p90"]
    assert interval_columns == [
        column for column in interval_columns if column in with_intervals.columns
    ]
    assert (with_intervals["balance_p10"] <= with_intervals["balance_p50"]).all()
    assert (with_intervals["balance_p50"] <= with_intervals["balance_p90"]).all()
    pd.testing.assert_frame_equal(with_intervals[forecast.columns], forecast)
    pd.testing.assert_frame_equal(with_intervals, repeated)
