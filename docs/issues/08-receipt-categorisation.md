# Receipt categorisation with a local LLM

Add `extract_receipt` for expense receipts (vendor, date, total, suggested category from the fixed category list). Feed results into `expenses.csv`. Let users correct categories and remember vendor-to-category mappings locally.

## Acceptance criteria
- Reuses `llm.chat` structured output; unknown categories map to 'other'
- Mock-based tests
