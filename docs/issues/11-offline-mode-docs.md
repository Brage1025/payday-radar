# Fully offline mode and docs

Document and test the air-gapped path: `pip install .[tabpfn-local]`, pre-downloading TabPFN weights, Ollama models pulled ahead of time, `--backend tabpfn-local`. Add a `payday-radar doctor` command that reports which backends and models are available.

## Acceptance criteria
- `doctor` prints backend availability, Ollama reachability and installed model names
- README offline section with exact commands
- Unit tests mock imports and Ollama
