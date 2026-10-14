# Privacy audit test: nothing identifying reaches a model

Add a test that wraps `make_classifier`/`make_regressor` with a recording fake and runs the full pipeline, asserting every frame passed to fit/predict is numeric-only and contains no value equal to a client or vendor name.

## Acceptance criteria
- Fails if someone adds a string column to any model input
- Runs in CI
