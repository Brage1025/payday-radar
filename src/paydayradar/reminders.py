"""Payment-reminder drafts. Template first; local LLM polish if Ollama is running."""

from __future__ import annotations

from . import llm

_TEMPLATES = {
    "en": {
        "subject": "Friendly reminder: invoice {invoice_id}",
        "body": (
            "Hi {client},\n\nA quick reminder that invoice {invoice_id} for NOK {amount:,.0f} "
            "{status}. Could you let me know when I can expect payment?\n\n"
            "Thanks a lot,\n{sender}"
        ),
    },
    "no": {
        "subject": "Vennlig påminnelse: faktura {invoice_id}",
        "body": (
            "Hei {client},\n\nEn vennlig påminnelse om at faktura {invoice_id} på "
            "kr {amount:,.0f} {status}. Kan du gi meg beskjed om når jeg kan forvente betaling?\n\n"
            "Mvh,\n{sender}"
        ),
    },
}


def _status(row: dict, lang: str) -> str:
    overdue = int(row.get("days_overdue", 0) or 0)
    due = str(row["due_date"])[:10]
    if lang == "no":
        return f"forfalt for {overdue} dager siden ({due})" if overdue > 0 else f"forfaller {due}"
    return f"was due {overdue} days ago ({due})" if overdue > 0 else f"is due on {due}"


def draft_reminder(row: dict, sender: str = "Your name", lang: str = "en",
                   use_llm: bool = False) -> dict:
    t = _TEMPLATES[lang]
    fields = {**row, "sender": sender, "status": _status(row, lang)}
    draft = {"subject": t["subject"].format(**fields), "body": t["body"].format(**fields)}
    if use_llm and llm.is_available():
        try:
            resp = llm.chat(
                llm.text_model(),
                [{"role": "user", "content": "Polish this payment reminder. Keep it short, "
                  f"polite, and in the same language. Return only the email body.\n\n{draft['body']}"}],
            )
            draft["body"] = resp["message"]["content"].strip()
        except Exception:
            pass  # fall back to the template
    return draft
