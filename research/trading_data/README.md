# Trading Data Integrity & Research Foundation

**DRAFT — NOT APPROVED. Offline candidate only; no production integration.**

Branch `research/trading-data-foundation-v1` derives from main
`2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`, not Finance. No production imports,
trading calls, live DB connections, migrations or startup hooks are included.

Run from repository root:
`python -m unittest discover -s research/trading_data/tests -v`.

- evidence.py: immutable retained variants and separate delivery receipts.
- lifecycle.py: account/environment-scoped signal/order/fill/trade/outcome graph.
- candles.py: explicit-calendar candle quality; absent evidence is not zero.
- replay.py: hypothetical R-only midpoint/spread replay; ambiguity is explicit.
- explorer.py: deny-by-default bounded read reference, not an authenticated server.
- retention_candidate.sql: local research schema proposal; runtime NOT executed.
- daily_report.schema.json: bounded contract; Finance manifest referenced, no competing accounting.
- explorer.html: offline synthetic demonstration; no real account data or network.
- field_mapping.csv: all637legacy headers, with candidate mappings clearly labelled.
- evidence/: fixed-window redacted observations and executed synthetic test results.

Current evidence is insufficient for complete broker reconciliation, official
profit factor or reliable rejected-signal replay. See HANDOVER.md for findings.
Raw exports are not published to GitHub; hashes and source windows are supplied.
An authenticated local service identity/ingestion integration requires separate
review and authorization. Never expose these helpers directly as public APIs.
