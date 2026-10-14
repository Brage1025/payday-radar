# Harden extraction for Norwegian/EU invoice layouts

Handle KID numbers, MVA/VAT lines, `Forfallsdato`/`Fakturadato` labels, `kr`/`NOK`/`EUR`, and two-digit years. Extend `parse_amount`/`parse_date` and the prompt. Distinguish total incl. MVA from subtotal.

## Acceptance criteria
- New parametrized tests with at least 15 realistic strings
- Ambiguous inputs raise `ValueError` rather than guessing
