# Extraction evaluation harness

Generate a golden set of synthetic invoice images with Pillow (varied layouts, Norwegian and English) plus ground-truth JSON. Add `scripts/eval_extraction.py` that reports per-field accuracy for any Ollama vision model so users can compare models.

## Acceptance criteria
- Deterministic generation (seeded)
- Prints a table: model, field, accuracy, and overall exact-match rate
- Documented in README under 'Swap the model'
