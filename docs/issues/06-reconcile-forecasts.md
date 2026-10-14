# Reconcile bottom-up collections with the top-down forecast

Today the monthly forecast is time-series based and ignores expected pay dates of open invoices. Blend them: income for the next 30/60/90 days = expected collections from open invoices (using `expected_pay_date`) + forecast of not-yet-issued work, without double counting.

## Acceptance criteria
- New `Workspace.cashflow_projection()` with a column showing each source
- Unit test showing a late invoice moves income into a later month
- Explain the method in docs/
