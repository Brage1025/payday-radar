# Benchmark: TabPFN vs scikit-learn baseline

Add `scripts/benchmark.py` comparing the RandomForest fallback, TabPFN-3.5 (when a token or local weights are available) on late-payment classification (ROC-AUC, log-loss) and days-late regression (MAE) using a time-based split on synthetic data. Output a markdown table for the README.

## Acceptance criteria
- Time-ordered split, no leakage
- Gracefully skips TabPFN rows if unavailable
- README 'Results' section updated by the script output
