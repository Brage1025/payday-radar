# Ingest PDFs and folders of scans

`extract.ingest_folder` only handles images. Add PDF support (render pages to images with PyMuPDF as an optional extra), a `payday-radar ingest <folder>` CLI command, and merge new rows into `invoices.csv` without creating duplicates.

## Acceptance criteria
- Multi-page PDFs: one invoice per file, using page 1 unless total not found
- Failed files are listed with the reason, never silently skipped
- Tests mock `llm.chat`; no Ollama needed
