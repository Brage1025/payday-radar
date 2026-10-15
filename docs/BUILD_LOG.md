# Build log: who built what

An honest record of how this project was built, for the "Best Use of GitHub Copilot" track.

**Plan note:** I'm on GitHub Copilot **Free**. The Copilot coding agent (assigning issues to
Copilot) and automatic Copilot code review on PRs need a paid plan, so I did **not** use them
and don't claim them. What I did use: **Copilot CLI** (drafting features on branches) and
**GitHub Actions** (CI, weekly report, agent setup file), with every change going through a PR.

## Who built what

| Area                                                                         | First draft by                     | Copilot surface | What review found / what I fixed                                                                                                                                                                                                                                                                                                                                                                               | Issue / PR           |
| ---------------------------------------------------------------------------- | ---------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| Initial scaffold (pipeline, TabPFN backends, synthetic data, agent, CLI, CI) | Claude (Anthropic), reviewed by me | none            | TabPFN API paths confirmed on a real run; added `.gitattributes`; silenced a test warning                                                                                                                                                                                                                                                                                                                      | initial commits      |
| Privacy audit test                                                           | Copilot CLI                        | Copilot CLI     | Copilot couldn't run pytest/ruff (not on PATH), so I ran them. Review found its assertions ran after the pipeline finished, so on TabPFN (which accepts text columns) a string leak wouldn't crash and wouldn't be caught. Moved the audit to the moment frames reach fit/predict and added a self-test of the audit helper                                                                                    | #13 / PR #14         |
| TabPFN vs scikit-learn benchmark                                             | Copilot CLI                        | Copilot CLI     | Its shell used a Python without the package installed (`No module named paydayradar`); fixed with an editable install. Review found right-censoring and future-label leakage in the time split; fixing it lowered scikit-learn ROC-AUC by roughly 0.03 (3 seeds, measured by Claude in a sandbox). First TabPFN run silently skipped the backend because the token wasn't set                                  | #11 / PR #[fill in]  |
| MCP server exposing the TabPFN-backed tools                                  | Copilot CLI                        | Copilot CLI     | Copilot checked the installed SDK itself and used the mcp 2.x API (`MCPServer`, not `FastMCP`). Review added input clamping (one forecast month = one hosted call), positive controls in the redaction test, and removal of the synthetic ground-truth label from anomaly output. End-to-end test on the `tabpfn-api` backend found that MCP clients don't pass `TABPFN_TOKEN` to the server unless configured | #6 / PR #[fill in]   |
| stdio MCP client demo script                                                 | Claude (not Copilot)               | none            | Verified against Copilot's server on sklearn and TabPFN                                                                                                                                                                                                                                                                                                                                                        | PR #[fill in]        |
| GitHub Actions: `ci.yml`, `copilot-setup-steps.yml`, `weekly-radar.yml`      | Claude, reviewed by me             | Actions         | CI runs ruff + pytest on Python 3.10 and 3.12 for every PR                                                                                                                                                                                                                                                                                                                                                     | `.github/workflows/` |

Copilot CLI prompts used: [fill in]. Sessions that produced nothing: [fill in].

## What I learned

- Copilot CLI wrote working code from tightly scoped single-paragraph prompts that pointed at an issue file and `.github/copilot-instructions.md`.
- It verified an API it didn't know by reading the installed SDK source instead of guessing.
- Weak spots were all verification problems, not syntax: it couldn't run its own tests on my machine, and two of its drafts were "correct but too optimistic" (a privacy test that only checked afterwards, a benchmark split that leaked future labels). Reviewing diffs and mutation-testing the tests mattered.
- Free-tier quota made me batch work into a few precise prompts.

## Built with GitHub Copilot

Built on the Copilot **Free** plan, so I used **Copilot CLI** and **GitHub Actions**; the paid
coding agent and automatic PR review were not available.

- `.github/copilot-instructions.md` steers Copilot CLI (privacy rules, offline tests).
- Features were drafted with Copilot CLI on feature branches, then reviewed and verified by hand:
  the privacy audit test, the TabPFN benchmark and the MCP server.
- Actions: CI (`ci.yml`), agent setup (`copilot-setup-steps.yml`) and a weekly report (`weekly-radar.yml`).
- Full attribution, including what Copilot got wrong: `docs/BUILD_LOG.md`.

## Screenshots (docs/screenshots/)

| #   | File                               | Shows                                  |
| --- | ---------------------------------- | -------------------------------------- |
| 01  | 01-copilot-code-review-ruleset.png | Branch ruleset requiring PRs           |
| 02  | 02-issues-list.png                 | The 12 scoped issues                   |
| 03  | 03-copilot-cli-session.png         | Copilot CLI on the privacy audit test  |
| 04  | 04-copilot-cli-pr.png              | The resulting PR                       |
| 05  | 05-copilot-cli-benchmark.png       | Copilot CLI on the benchmark           |
| 06  | 06-benchmark-results.png           | TabPFN vs scikit-learn table           |
| 07  | 07-copilot-cli-mcp-server.png      | Copilot CLI on the MCP server          |
| 08  | 08-mcp-client-demo.png             | Separate MCP client calling the server |
| 09  | 09-mcp-tabpfn-run.png              | Same, on the tabpfn-api backend        |
| 10  | 10-actions-all-green.png           | CI and workflows passing               |
