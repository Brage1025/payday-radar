"""Invoice extraction with a local open-weight vision model (via Ollama).

Parsing helpers handle Norwegian/EU formats: "12 345,67", "1.234,56", "31.10.2026".
"""

from __future__ import annotations

import base64
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from . import llm

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "client": {"type": "string"},
        "invoice_number": {"type": "string"},
        "issue_date": {"type": "string"},
        "due_date": {"type": "string"},
        "amount": {"type": "string"},
        "currency": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": ["client", "issue_date", "due_date", "amount"],
}

PROMPT = (
    "Extract the invoice fields as JSON. 'client' is the party being billed. "
    "Copy dates and the total amount exactly as printed (including VAT/MVA). "
    "Use an empty string for anything not present."
)

CSV_FIELDS = ["invoice_number", "client", "issue_date", "due_date", "amount", "currency", "category"]


def parse_amount(text: str) -> float:
    s = re.sub(r"[^\d,.\-]", "", str(text).replace("\u00a0", " ").replace(" ", ""))
    if not re.search(r"\d", s):
        raise ValueError(f"No number in {text!r}")
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s:
        tail = s.rsplit(",", 1)[1]
        s = s.replace(",", ".") if len(tail) in (1, 2) and s.count(",") == 1 else s.replace(",", "")
    elif "." in s:
        tail = s.rsplit(".", 1)[1]
        if not (len(tail) in (1, 2) and s.count(".") == 1):
            s = s.replace(".", "")
    return float(s)


def parse_date(text: str) -> str:
    t = str(text).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d.%m.%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {text!r}")


def validate_invoice(data: dict) -> dict:
    """Normalise raw model output; raises ValueError if required fields are unusable."""
    out = {
        "invoice_number": str(data.get("invoice_number", "")).strip(),
        "client": str(data.get("client", "")).strip(),
        "issue_date": parse_date(data["issue_date"]),
        "due_date": parse_date(data["due_date"]),
        "amount": parse_amount(data["amount"]),
        "currency": (str(data.get("currency", "")).strip().upper() or "NOK"),
        "category": str(data.get("category", "")).strip().lower(),
    }
    if not out["client"]:
        raise ValueError("Missing client")
    return out


def extract_invoice(image_path: str | Path, model: str | None = None) -> dict:
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    resp = llm.chat(
        model or llm.vision_model(),
        [{"role": "user", "content": PROMPT, "images": [img]}],
        fmt=INVOICE_SCHEMA,
    )
    return validate_invoice(json.loads(resp["message"]["content"]))


def ingest_folder(folder: str | Path, out_csv: str | Path) -> tuple[int, list[str]]:
    """Extract every image in a folder into a CSV. Returns (ok_count, errors)."""
    ok, errors = 0, []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                w.writerow(extract_invoice(p))
                ok += 1
            except Exception as e:  # keep going; report per-file failures
                errors.append(f"{p.name}: {e}")
    return ok, errors
