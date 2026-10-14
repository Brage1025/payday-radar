# Streamlit dashboard: risk table and forecast chart

Add `app.py` (optional extra `ui`) with: a sidebar to pick a data folder or upload invoices.csv/expenses.csv, a sortable table of open invoices coloured by risk, a forecast line chart, an anomalies tab, and a chat box wired to `Agent.ask`.

## Acceptance criteria
- Runs with `streamlit run app.py` on the synthetic data
- Backend selector (auto / tabpfn-api / tabpfn-local / sklearn) shown in the UI
- No client names leave the process; reuse `Workspace`, do not call models directly
- A smoke test imports the app's helper functions (no Streamlit server in tests)
