# Uncertainty: low/likely/high cash-flow scenarios

Use TabPFN's distributional output (quantiles) for the regressors to produce P10/P50/P90 balance paths. Fall back to bootstrap or RandomForest tree spread for the sklearn backend.

## Acceptance criteria
- `forecast(..., intervals=True)` adds balance_p10/p50/p90
- `can_afford` uses P10 for its verdict and says so
- Tests on the sklearn backend only
