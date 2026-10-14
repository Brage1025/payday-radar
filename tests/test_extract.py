import pytest

from paydayradar.extract import parse_amount, parse_date, validate_invoice


@pytest.mark.parametrize("raw,expected", [
    ("12 345,67", 12345.67), ("kr 1.234,56", 1234.56), ("1,234.56", 1234.56),
    ("NOK 4 500", 4500.0), ("1.234", 1234.0), ("99,5", 99.5),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw", ["2026-10-31", "31.10.2026", "31/10/2026", "31.10.26"])
def test_parse_date(raw):
    assert parse_date(raw) == "2026-10-31"


def test_validate_invoice():
    out = validate_invoice({"client": " Fjordly AS ", "issue_date": "01.10.2026",
                            "due_date": "15.10.2026", "amount": "24 000,00"})
    assert out["amount"] == 24000.0 and out["currency"] == "NOK" and out["client"] == "Fjordly AS"


def test_validate_requires_client():
    with pytest.raises(ValueError):
        validate_invoice({"client": "", "issue_date": "2026-10-01",
                          "due_date": "2026-10-15", "amount": "1"})
