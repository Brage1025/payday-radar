# Expose Payday Radar tools as an MCP server

Add `payday-radar mcp` exposing `risky_invoices`, `cashflow_forecast`, `can_afford`, `draft_reminder`, `anomalies` over MCP (stdio) so any MCP client can use them. Document how to run it alongside the TabPFN MCP server.

## Acceptance criteria
- Optional extra `mcp`; core install unaffected
- Tools reuse the `Agent` methods (no duplicated logic)
- README snippet with a client config example
