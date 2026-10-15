# ST LUDAETUC CLOUD V1.0 — HISTORICAL MARKET DATA RECOVERY FOUNDATION

DRAFT — NOT APPROVED. Isolated implementation for independent review.

## A. Executive summary

Implemented a version-aware offline recovery foundation beneath
`research/historical_recovery/`, based directly on production SHA
`2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`. Branch:
`research/historical-market-recovery-v1`. Exact frozen new SHA is supplied in the
final handover rather than embedded recursively in its own commit.

The implementation preserves both legacy aware ISO source time and v2 integer
Unix-millisecond bar open/close fields. It keeps stored event_time, received_at,
original event/payload, schema/version metadata and original source features.
It validates source times without receipt fallback or rounding; detects duplicates,
conflicts and scope mismatches; produces bounded diagnostics and hash-bound reader
pages; and provides an operator-only read-only PostgreSQL extractor and offline
projection CLI. No indicator is recomputed and no trading calculation is changed.

The new CEO summary reports 164,209 stored market-data events, including 121,309
ISO time and 42,900 time_open records, with zero lacking both across 35 streams.
This establishes reported source-field presence, not validity or historical
completeness. No actual raw production rows were acquired or marked recovered in
this task. The engineering candidate is ready to review; real data recovery and
research use remain blocked on row-level source evidence and approved contexts.

All 99 tests pass: positive/negative synthetic, adversarial, acquisition/projection
lineage and JSON contract validation. PostgreSQL 18 runtime is UNPROVED; no safe
local postgres/initdb executable exists in the root-only mapped environment.
No shared/cloud database connection, migration or writer was used.

## B. Technical evidence

### Source inspection and authoritative boundaries

Immutable production source confirms /market-data reads `body.time`, stores the
original payload, and uses a forgiving timestamp parser that can substitute now.
log_event upserts payload/status/receipt time on an existing event ID; old versions
are not retained there. GitHub baseline engine blob reported by the connector is
`99ea93e1006cba840672cbdb1682395aa1ec78fa`. Fetch text is a representation, not a
claim to preserve original wire bytes. Source field presence cannot prove which
historical records used fallback. That requires exact original record comparison.

Only read-only baseline/branch metadata was inspected remotely. No unauthenticated
analytics API was used, no database extraction was executed and no active alerts
or source accounts were accessed. The CEO-demonstrated internal Web Shell is an
operator route described in the runbook; no networking/credentials were changed.

Frozen candidates are reference-only and not imported/merged/advanced:

| Candidate | Frozen SHA |
|---|---|
| Trading Data Foundation | 960cd8cdedfc5aa17d0827560f1dcc7924d2d53d |
| Finance | 832fc31e02a494d88af3676a4e3e685a3f5f7692 |
| Analytics security | c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44 |

No Chief/recovery-controller code, main, governance, Render, Make, TradingView,
OANDA or authoritative financial record was modified. Market Intelligence v3.2
is design context only. No architecture/calendar/source-registry approval is inferred.

### Component behaviour

| Component | Implemented behaviour | Important bound |
|---|---|---|
| normalizer.py | ISO/ms parsing, separate times, original payload/feature copies, Decimal OHLCV, source identity/version, explicit calendar, research classification | 256 KiB payload, known seven symbols/five timeframes, dates 2000–2100 |
| auditor.py | Unique bars, event/bar duplicates, full-payload conflict checks, features, missing slots, late/out-of-order receipts, scope/window completeness | 50,000 rows per stream/window, <=31 days, <=35 explicit streams |
| reader_page | Manifest/population/row integrity, unique event-ID cursor, allowlisted metadata/OHLCV, feature availability | <=1000 rows, private offline library only |
| extract.py | Parameterised SELECT-only acquisition, count + ID pages in one read-only repeatable snapshot, redaction/private files/audit | <=7 days receipts, <=1000/page, <=50,000 rows, <=64 MiB total |
| project.py | Verify export manifest/row chains, original event recovery projection, fail-closed trust defaults | <=64 MiB input, no overwrite, no DB writes |

Late arrivals can be valid descriptive candles, but causal linkage is held. Original
receipt sequence is unproved where upserts have replaced received_at. Explicit
calendar slots override nominal grid assumptions for legitimate shifted/shortened
bars. Without an approved calendar, missing slots are null/unproved rather than
24-hour inferred gaps. Field aliases are strict: no provider/symbol/timeframe
coercion is invented. Multiple version declarations require a reviewed mapping.

Provenance comes from trusted Context with independently reviewed event digests,
version registry and calendar, not a source-supplied verified flag. This is an offline
trust input, not a cryptographic identity/approval service. An operator can supply
bad trust inputs; governance must review them. Hashes prove internal integrity,
not authenticity. JSONB value hashing cannot recover original HTTP bytes.

## C. Data recovery assessment

See POPULATION_REPORT.md and evidence/population_summary.json. Raw-validity,
missing-bar, duplicate, indicator-value and completeness counts remain unknown.
All 35 reported streams are NOT_YET_ASSESSED at record level, not proven lost.
Legacy time and v2 time_open provide potential recovery sources; conflicting forms,
incorrect units and invalid times are held. UNRECOVERABLE means no usable/consistent
time from the supplied retained record alone, not proof external recovery is impossible.
ESTIMATED_NOT_AUTHORITATIVE is reserved; the implementation never manufactures time
or OHLCV estimates. Originals stay intact in private projections and PostgreSQL.

Timestamp trust and OHLCV eligibility are independent. A source timestamp may be
verified while the price record is invalid. A captured receipt snapshot can be
complete while the required source-bar window is unproved. CLI projections never
promote receipt snapshots into complete source windows. Delayed/retried records,
known collection start/outages and independently expected calendar slots all matter.

## D. Research readiness

| Activity | Real historical dataset state | Required next evidence |
|---|---|---|
| Counts/source-field inventory | PROVISIONAL — CEO summary only | Exact SQL, capture timestamp, per-stream/format counts |
| Descriptive candle analysis | BLOCKED_ON_DATA | Bound source records + provenance + valid OHLCV/times/calendar |
| Source-indicator research | BLOCKED_ON_DATA | Original legacy fields and formula/script/version/confirmation mapping |
| Signal/candle linkage | BLOCKED_ON_DATA | Valid candle times plus signal/decision records and receipt availability |
| Rejected-trade replay | BLOCKED_ON_VALIDATION | Complete verified windows + independently validated replay assumptions |
| Backtesting | BLOCKED_ON_VALIDATION | Above plus execution/cost assumptions, no lookahead leakage |
| Training datasets | BLOCKED_ON_GOVERNANCE | Valid data/labels, temporal splits, approved training purpose and access |

Synthetic fixtures demonstrate code behaviour only. No new Python feature can
influence production. Existing Pine features are retained with provenance, not
promoted to authoritative training labels. Actual OANDA P&L and Finance remain separate.

## E. Implementation changes and tests

All changes are new files under the dedicated recovery namespace. Files comprise
README, OPERATOR_RUNBOOK, POPULATION_REPORT, this HANDOVER, normalizer/auditor/extract/
project modules, __init__, demo, two JSON schemas, test requirements, two test modules
and evidence files for reported population/source inventory, runtime probe, synthetic
demonstration and complete test output. No production source file is changed.

99 executed tests pass, 0 failures/errors/skips. Full names/results:
`evidence/test_results.txt`. Python 3.12; jsonschema 4.26.0 test-only. Standard-library
normalisation/audit tests never import engine or psycopg2. Fake-connection extraction
tests validate SQL/rollback/cursor/snapshot control flow, not PostgreSQL semantics.
No real PG acceptance, database projection, production integration, privileges or
performance test is claimed. The lack of local binaries/root mapping is recorded
in evidence/runtime_probe.json. A disposable non-root PostgreSQL 18 runner remains
a named validation dependency.

Tests include both valid formats, bad/missing/naive/extreme dates, seconds/ms,
conflicting dual times, timeframe/symbol/source/environment mismatch, off-grid and
explicit calendar exceptions, spring/fall DST, invalid OHLCV, missing/preserved
features, input immutability, stale provenance, payload self-verification rejection,
late/out-of-order delivery, event/bar/full-feature conflicts, partial/empty snapshots,
missing slots, read-only mode checks, rollback, pagination/caps/resume semantics,
private file permissions/no overwrite, redaction, sanitized failures, Decimal JSON
round-trips, manifest/row tampering, multistream contamination and schema validation.

Reproduce from repository root:

```
python -m pip install -r research/historical_recovery/requirements-test.txt
python -m unittest discover -s research/historical_recovery/tests -v
python -m research.historical_recovery.demo
```

Extraction needs an existing operator-only psycopg2 environment. Do not install
or run it inside production trading as part of this assignment.

## F. Independent review requirements

Review the exact new immutable SHA reported with this handover. Compare against
production baseline, confirm every changed path starts research/historical_recovery/,
parent is the baseline, and frozen candidate refs/main are unchanged. Read the
runbook, contracts and all code before running synthetic tests. No live DSN required.

Challenge timestamp units, timezone/DST/calendar assumptions, strict-version holds,
record/classifier vs dataset-completeness semantics, provenance binding, full-payload
conflicts, delayed data causality, redaction limitations and input bounds. Specifically
verify that source fields/verified flags cannot self-ratify trust and that identical
OHLCV does not erase conflicting original indicators. Check export pages share a
real repeatable snapshot, receipt vs source-window distinctions, and failure/rollback.

On an authorised genuinely disposable local PostgreSQL 18 instance, validate SELECT
syntax, real set_session/transaction_read_only, C-collation pagination, exact Decimal
JSONB/timestamptz round trips, counts at page boundaries, concurrent source updates,
rollback after failure and least-privilege denial. Use synthetic public.events copies
only. Confirm no cloud/shared DB is targeted. None of these runtime checks has been
performed here, so review must not treat Python mocks as database acceptance.

Request actual redacted samples and the independently checked source/calendar
registry before judging historical validity. Coordinate version/alert evidence with
Grok's independent work; this implementation does not overwrite its design documents.

## G. Next authorised action

First independently review this isolated engineering SHA. Then the CEO/data owner
may authorise a named internal operator to run one bounded stream/day export using
an existing read-only identity and approved private destination. The runbook gives
exact commands and fields. Request:

1. Original population query/capture time, bounds, format overlap and all 35 stream
   counts; no credentials or unrestricted extraction.
2. Original event_id/instrument/timeframe, payload.time/time_open/time_close/OHLCV/
   indicators, event_time/received_at/retry_count, source/schema/script/provider and
   confirmation fields across session/format boundaries.
3. Exact legacy/v2 logger definitions and active-alert version-effective history.
4. Versioned approved market calendars, capture-start/outage history, receipt catch-up
   policy, trusted acquisition attestations and source-window coverage criteria.
5. Disposable local PostgreSQL 18 execution evidence before any integration claim.

Named residuals: R1 no real raw payloads; R2 real script/alias/provider registry
unverified; R3 no ratified calendars; R4 source-window completeness unproved; R5 PG18
runtime unproved; R6 offline reader authentication/service wrapper not implemented;
R7 arbitrary-text secret redaction needs human review; R8 first delivery/revisions
may already be overwritten in production and cannot be reconstructed from current
rows alone; R9 feature value/formula validity unproved. No additional production
access or deployment is inferred from engineering readiness.

### Deliverable coverage

| Requested deliverable | Location / status |
|---|---|
| 1 Architecture/source inventory | README + evidence/source_inventory.json |
| 2 Version-aware normaliser | normalizer.py |
| 3 Timestamp validation | normalizer.py iso_time/epoch_ms/Calendar |
| 4 Feature preservation | normalizer.py features |
| 5 Integrity auditor | auditor.py audit/audit_streams |
| 6 Duplicate/conflict detector | auditor.py full-payload/event groups |
| 7 Recovery projection | normalize + project.py; no DB writer |
| 8 Operator extraction | extract.py + OPERATOR_RUNBOOK.md |
| 9 Quality contract | quality_report.schema.json + reader_contract.schema.json |
| 10 Eligibility | normalizer.py + README criteria |
| 11 Synthetic fixtures | tests/test_recovery.py + evidence/synthetic_demo.json |
| 12 Adversarial suite | tests/ + evidence/test_results.txt |
| 13 PG runtime evidence | evidence/runtime_probe.json — UNPROVED |
| 14 Full-population proposed report | POPULATION_REPORT.md + evidence/population_summary.json |
| 15 Exact evidence requirements | runbook + section G |
| 16 Security/privacy | runbook + README + named residuals |
| 17 Engineering handover | this document |
| 18 Independent review instructions | section F |
| 19 Dependencies/decisions | section G |
| 20 Exact frozen commit | final handover and branch ref; never self-referential SHA |

HISTORICAL_MARKET_RECOVERY_READY_FOR_INDEPENDENT_REVIEW
