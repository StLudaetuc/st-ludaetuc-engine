# Historical Market Recovery v1.1 — DRAFT, NOT APPROVED

Isolated offline proposal. No production imports, HTTP reader, ledger, migration,
indicator computation or trading authority. Parent is production baseline
`2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`, not an unapproved candidate.

## Components

- `normalizer.py`: strict ISO / integer epoch-millisecond parsing, Decimal OHLCV,
  original payload/feature preservation, digest provenance, explicit calendar and
  record-level research eligibility.
- `auditor.py`: bounded stream diagnostics, duplicate/conflict handling, dataset
  completeness and hash-bound read-only pages. Not an authentication boundary.
- `extract.py`: operator-only scoped PostgreSQL SELECTs in one read-only repeatable
  transaction, stable ID pages, counts, redaction, private exports and audit hashes.
- `project.py`: offline manifest verification and non-destructive recovery reports.
- `quality_report.schema.json`: deliberately strict contract for the current
  **unvalidated CEO-reported** population summary, not a schema for all future audit
  results. `reader_contract.schema.json` bounds the safe dataset-page product.
- `tests/`: synthetic positive/negative controls. No credentials or live source reads.

Run from repository root, Python 3.12+:

```
python -m pip install -r research/historical_recovery/requirements-test.txt
python -m unittest discover -s research/historical_recovery/tests -v
```

Schema tests use jsonschema 4.26.0 (test only). Extraction uses
psycopg2-binary 2.9.11 on an operator-authorised internal host. Production
requirements are not changed. No DB package is imported by normalization/tests.

## Source contracts and limitations

Legacy: aware ISO `payload.time`; optional aware ISO `time_close`. V2: integer
Unix milliseconds `time_open` and `time_close`. Numeric strings are supported;
float milliseconds and fractional milliseconds are rejected. Simultaneous forms
must agree exactly; even agreeing dual schemas require review. Dates must be
2000–2100 under this proposal. Seconds/milliseconds are never guessed/corrected.

Exact canonical envelope instruments: GBPUSD, EURUSD, AUDUSD, USDCAD, USDJPY,
XAUUSD, XAGUSD. Timeframes: 1m/2m/3m/5m/10m. `ticker` or `symbol` must agree with
the envelope; `interval` or `timeframe` accepts canonical strings or 1/2/3/5/10
strings. Provider-prefixed symbols, integer interval values, unspecified source
versions and other dialects deliberately fail/hold until actual logger samples
establish a reviewed adapter. These field names are supported interfaces, not a
claim that the actual legacy/v2 scripts have been inspected. No real payloads
were supplied. Unknown fields stay in original_event.payload.

Original features are copied, never recomputed: all `indicators` contents plus
recognized root EMA/RSI/ATR/ADX/MACD/Bollinger aliases. Availability means an
original non-null field is present; **not** that its numeric value, Pine formula,
lookback, bar confirmation or usefulness has been validated. Nested/unknown
indicator structures remain preserved even if the alias registry cannot classify
them. Real field-name/version mapping remains an evidence dependency.

The original event, stored event_time and received_at stay unchanged in private
projections. canonical_json_value_v2_utc6 hashes typed JSON values (including Decimal
numbers), not original HTTP bytes: JSONB cannot reproduce wire lexical order or
whitespace. Hashes detect changes; they do not establish authenticity. Independent
operator acquisition review must bind exact event digests to trusted provenance.
Source `verified=true` never grants trust.

## Calendar and eligibility

No real approved calendar or source registry is bundled. `Calendar` contains
versioned, explicit aware UTC slots, bounded to one stream/window, and an external
approval flag. Fixture calendars are conspicuously SYNTHETIC. Exact calendar slots
can allow shortened/shifted session bars; nominal UTC grid disagreement alone is
not a missing-bar assertion. DST offsets are converted to UTC before comparison.
Empty schedules need reviewed_empty. Runtime callers must not promote source
payload fields into trusted Context/Calendar approval.

Timestamp recovery state is separate from full research eligibility:

| State | Criterion |
|---|---|
| NOT_YET_ASSESSED | Exact event evidence not independently bound to trusted Context |
| UNRECOVERABLE | No usable source-open time, or mutually conflicting forms, **from this record alone** after evidence binding; later external evidence may support a new projection |
| RECOVERABLE_WITH_SOURCE_EVIDENCE | Usable original time exists, but calendar/version/confirmation/metadata checks remain unresolved |
| ORIGINAL_TIMESTAMP_VERIFIED | Bound evidence and required timestamp checks pass; OHLCV can still independently fail |
| ESTIMATED_NOT_AUTHORITATIVE | Reserved for explicitly labelled external estimates; this normaliser never fabricates estimates |

For a record's descriptive eligibility, source provenance, known script/schema
version, confirmed candle, valid OHLCV, receipt chronology, canonical stream,
source time and verified close/calendar must all pass. Valid delayed arrivals do
not invalidate a candle, but cannot be treated as available before receipt for
causal decision linkage. `received_at` can be changed by production upserts, so
latency is current retained receipt latency, not guaranteed first delivery.

Indicator research additionally requires source feature availability, and remains
SOURCE_FEATURES_ONLY until formulas/semantics are independently established.
Signal linkage needs original signal/decision evidence; rejected replay needs
validated complete windows and replay logic; backtests additionally need execution
assumptions; training needs label provenance, temporal splits and leakage checks.
These are never enabled merely by a valid-looking timestamp.

## Read-only research interface

`reader_page(report, manifest_hash=..., after_event_id=..., limit=...)` verifies
summary/population/row hashes, rejects unknown cursors and repeated event IDs,
returns <=1000 allowlisted records plus feature availability (no raw payload),
and carries both captured-window and research-completeness status. The entire
report is one scope; caller authentication/authorisation must happen before the
private report is loaded. This is a local library contract, **not** a deployed
or authenticated Reader service. It does not modify the frozen security candidate.
A future authorised service must bind service identity to environment/source/
instrument/timeframe and dataset digest, audit reads/denials, enforce egress and
redact responses. Do not mount the Python function as a public endpoint.

## v1.1 correction boundary

Read FINDINGS_V11.md and ANCHOR_SPEC.md first. The complete independent D1–D9
report is unavailable: D5/D6/D8/D9 cannot be asserted resolved under those IDs.
Known D1/D2/D3/D4/D7 and separately listed contract issues are corrected/tested.
PostgreSQL 18 execution remains UNPROVED. No actual historical records processed.

The old writer-bearing Render Web Shell is **not** a permitted acquisition route.
Only a future CEO-approved isolated host and dedicated SELECT-only login may run
the proposed CLI. No resources/credentials are provisioned here.

Internal checksums are distinguished from independently anchored expectations.
project_export verifies export_audit.json and can require a separate TrustedAnchor;
without it, authenticity and row provenance stay UNPROVED regardless of supplied
Context. Fixed UTC six-digit timestamps make record hashing session-timezone
independent in Python; true PostgreSQL execution is still a review dependency.

Calendar presence is separate from valid-slot coverage. Invalid/conflicting/
unproved slots never satisfy COMPLETE_EXPECTED_SLOTS. Unresolved duplicates are
demoted in aggregate recovery reporting while original timestamps, OHLCV and
indicator values remain inspectable. Rich readiness_summary and slot sets are
shadow-only offline products; training remains blocked pending label/split policy.
