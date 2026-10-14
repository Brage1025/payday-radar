# Copilot instructions for Payday Radar

Payday Radar is a **local-first cash-flow radar for freelancers**. It reads invoices and
expenses, predicts which invoices will be paid late, forecasts cash flow, flags anomalies,
and answers questions through a small agent. Open-weight models (via Ollama) do extraction
and chat; **TabPFN-3.5** (Prior Labs' tabular foundation model) does all prediction.

## Hard rules (never break these)
1. **Privacy:** client names, invoice text and any free-form strings must never be sent to a
   remote service. Only numeric feature frames go to a model backend, and every model call must
   pass through `paydayradar.models.ensure_numeric`. Do not weaken or bypass it.
2. **No secrets in code.** TabPFN uses the `TABPFN_TOKEN` env var. Never hardcode or log it.
3. **Backends stay swappable.** All model construction goes through `models.make_classifier`
   and `models.make_regressor`. Do not import `tabpfn` / `tabpfn_client` anywhere else.
4. **Tests and CI must work offline** using `backend="sklearn"`. Never require a token, GPU,
   network or a running Ollama in tests. Mock `paydayradar.llm` instead.
5. **No new runtime dependencies** without a note in the PR explaining why. Core deps are
   pandas, numpy and scikit-learn only. Put optional ones in `[project.optional-dependencies]`.

## Layout
- `src/paydayradar/synthetic.py`: fake freelancer data (the demo never needs real data)
- `features.py`: leak-free numeric features (client history uses only invoices paid *before* the reference date)
- `models.py`: TabPFN API / TabPFN local / sklearn fallback
- `pipeline.py`: `Workspace` with `score_open_invoices`, `forecast`, `find_anomalies`
- `extract.py`: invoice image -> structured fields with a local vision model; Norwegian/EU parsing
- `llm.py`: stdlib Ollama client; `agent.py`: tool-calling agent plus offline keyword router
- `reminders.py`: reminder drafts (English and Norwegian); `cli.py`: `payday-radar` entry point

## Commands
```bash
pip install -e ".[dev]"
ruff check .
pytest -q
python -m paydayradar --backend sklearn demo
```

## Conventions
- Python 3.10+, type hints on public functions, short docstrings, ruff-clean (line length 100).
- Money is NOK by default; dates are ISO strings in files, `pd.Timestamp` in memory.
- Add or update a test for every behavior change. Keep PRs small and focused on one issue.
- Be honest in PR descriptions: say what you ran and what you could not verify
  (for example, hosted TabPFN calls cannot run in CI).
- Update `docs/BUILD_LOG.md` only if the maintainer asks.
