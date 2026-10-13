# ST LUDAETUC CLOUD V1.0 — FINANCE FOUNDATION INDEPENDENT-REVIEW CORRECTION

Old reviewed SHA: `189587ce3e211b22782dc221af9afdf6a8cc123c`.
Old independent verdict: `FINANCE_FOUNDATION_INDEPENDENT_REVIEW_REQUIRES_CORRECTION`.
New SHA: the exact commit containing this corrected source; the delivered copy
records the literal SHA once Git assigns it. This source does not self-approve.

Repository `StLudaetuc/st-ludaetuc-engine`; writable branch only
`finance/foundation-v1`; sole existing [draft PR #2](https://github.com/StLudaetuc/st-ludaetuc-engine/pull/2).
No merge, deployment, cloud validation DB, shared database connection, engine
execution change, Make/TradingView/OANDA action or Chief/recovery-controller edit.
Original v1 source/evidence remains retrievable at the old exact SHA.

## Correction evidence — HIGH

| Finding | Corrected implementation | Regression evidence |
| --- | --- | --- |
| H1 Authoritative ordering | All SQL authority-bearing rows receive a forced server sequence identity and applicable server receipt clock via trigger. Supplied server_order is overwritten. Latest coverage, verification, runs and exception decisions use server_order, never source timestamps. Python journals allocate under server locks. Source clocks are separate. | Future assertion followed by withdrawal; verification withdrawal; concurrent journal IDs; later reconciliation supersedes future-dated source; stale resolution rejected. SQL acceptance includes injected IDs and future clocks. |
| H2 Empty evidence | Coverage requires count, bounds, manifest, first/last IDs, review, currency/cost completeness and unresolved state. VERIFIED cannot be empty. Zero needs REVIEWED_EMPTY and count zero; unknown/withdrawn/count mismatch yields unavailable. Broker SQL also checks complete supported cash components and reconciliation evidence counts/results. | Unknown broker/expenses/inventory suppress economic and break-even output; explicit reviewed empty yields genuine zero; count mismatches fail; incomplete legacy P&L remains unavailable. |
| H3 Project economics | Eligible account net trading is summed first; verified project cash spend or recognised economic expense is subtracted once. Account totals never deduct project costs. Duplicate accounts rejected and incomplete eligible account invalidates project result. | Two accounts 95+45 less project 12 = 128; Practice 9,999 excluded; duplicate/incomplete account tests. Same SQL acceptance vector supplied. |
| H4 Financial reality | Scope has five required independent dimensions with no defaults. Lowercase legacy production is not a broker environment. Simulated requires explicit simulation context and cannot be REALISED. Live project eligibility requires OANDA_LIVE+PRODUCTION+ACTUAL+REALISED+no simulation. | Missing defaults/legacy production rejected; simulated realised rejected; Practice/SHADOW excluded from actual gains. |
| H5 Semantic serializer | One Reader/weekly egress validates JSON Schema, trusted producer scope and complete expected metric manifest, current server-ordered coverage/reconciliation, counts, classifications, precision, policy and denominators. Missing dependencies fail closed. | All six products and weekly input; Practice live-profit rejection; simulated/realised rejection; withdrawal, incomplete, unresolved, estimated/unknown FACT, forged values/counts/state, wrong denominator, missing recognition and extra fields rejected. Positive Practice output is hypothetical only. |

Important trust boundary: journal/context/expected-metric manifests are trusted
server objects, not LLM arguments or user-supplied assertions. A future publisher
must construct them from verified deterministic source outputs and reviewed
coverage. Copying model JSON into this context would violate the design. No
endpoint, credential or role enabling that access exists in this candidate.

## Correction evidence — MEDIUM

| Finding | Corrected implementation | Evidence |
| --- | --- | --- |
| M1 Precision | `money_scale12_half_away_v1`: bounded finite decimal input; scale 12, ties away from zero. FX quantises each cash component once, then sums. Python uses local high precision; SQL numeric round matches. Recognition uses cumulative rounded endpoint differences. | Positive/negative ties, FX vector, hostile global Decimal context, run-rate division, recognition additivity. SQL/PL grammar PASS; runtime parity NOT EXECUTED. |
| M2 Lifecycle | Canonical MATCHED, BROKER_ONLY, INTERNAL_ONLY, VALUE_MISMATCH, PENDING_EOD_SYNC, INVESTIGATION_REQUIRED. Malformed rows produce exceptions and processing continues. Exact replays dedupe; conflicting events quarantine. Partial REDUCE+CLOSE slices sum once. | Partial-close rollup, EOD boundary, late matching, missing records, malformed money/identity, conflicting/receipt-only replay cases. |
| M3 Attribution | Parent transaction is authoritative cash. Nested trade/position amounts are attribution only. Where supplied, nested sums must reconcile to parent; missing/mismatched attribution quarantines the transaction. Batch processing continues. SQL cash profiles enforce field/kind/fee-sign consistency. | Existing nested no-double-count tests; mismatch/missing attribution with a valid adjacent transaction; fee/financing sign and transfers retained. |
| M4 CSV | 1 MiB/1,000 rows/512-character field limits, exact header allowlist, candidate currency registry, opaque evidence namespaces, formula-neutralised text, safe source IDs. CSV cannot declare verification or verifier. Trusted verification is a separate append-only revision-specific decision. | Self-verification, unsafe reference/scheme, unknown currency, oversized/late-invalid row, neutralisation/retry and concurrent duplicate import tests. |
| M5 Source time | Expense revision sequence and all current authority pointers use server acceptance order, not incurred/source/as_of timestamps. Paid/service/event/receipt times remain separately queryable. | Future incurred timestamp corrected by next revision; future decisions withdrawn; later run chosen regardless of source clock. |
| M6 Practice | Practice unit economics are HYPOTHETICAL_NON_CASH; actual_net_profit absent. They do not deduct actual project spend from simulated gain as actual profit and never reduce actual break-even. Denominator basis is explicit. | Practice positive/non-cash output, live project exclusion and attempted-order denominator rejection. |
| M7 Run rate | Output distinguishes verified/reviewed-empty zero, excluded estimates, unverified recurring records and absent coverage. Unknown data is null, not zero. Period/as_of coverage and overlap checks apply. | Genuine reviewed zero, missing evidence, separate estimate/unknown counts, overlap/expiry cases. |
| M8 Recognition | paid_at controls cash; explicit per-expense service recognition controls economic expense; schedules control forecast run rate. Prepaid charges recognise over billing/service period. Missing policy/payment evidence yields unavailable for that measure. Cumulative quantisation preserves adjacent-period totals. | 365 prepaid recognised 31 over October, missing policy, separate payment date, refunds and three-period exact allocation. |
| M9 Exception resolution | Append-only reviewed resolution links a prior exception to later server-accepted matched evidence in the same account/scope. Withdrawal supersedes resolution, original result retained. SQL enforces same-account, same-trade, later-run match. | Resolve/withdraw, stale matched evidence, cross-account and unmatched-resolution rejection; source history unchanged. |

## Operational semantics preserved

The previous bounded audit contained 155 order/open records: 112 FILLED and
43 CANCELLED. HTTP 200 means logging succeeded; it does not mean execution.
Attempted orders, FILLED orders, matched fully closed trades, partial-close
slices and transactions are separate populations. Cancelled or conflicted
records cannot enter the matched closed-trade profitability denominator.
The serializer checks denominator values against deterministic run populations.

EOD closure sync is expected. A healthy bridge with a closed broker trade and
missing outcome is PENDING_EOD_SYNC before the explicit EOD deadline; at/after
that boundary it is a reconciliation gap, not an automatic trade/accounting
loss. Later matching/replayed evidence can produce a new MATCHED run and a
reviewed resolution. Original evidence is never overwritten. Event time and
receipt time remain different fields.

Legacy realised P&L remains source evidence. No scope/currency/cost completeness
means no authoritative Finance conclusion; absent financing is not zero. No
current real financial data has been imported or claimed complete.

Market status is **RESOLVED_PENDING_FULL_COVERAGE_VERIFICATION**, using the newer
CEO-supplied read-only verification: latest seven hours 6,338 HTTP 200, zero 422
and zero observed literal ticker failures; preceding seven hours 1,049 HTTP 422.
Exact window bounds were not supplied and are not invented. Full symbol/timeframe
coverage is not proven. `market_data_v2` proposal and placeholder rejection remain
unchanged; no market infrastructure or alert changes were made.

## Files changed/added

* Core: `core.py`, `policy.py` (new), `reconciliation.py` (new), `serializer.py`
  (new), `oanda.py`.
* SQL: `migrations/001_foundation.sql`, `tests/postgres_acceptance.sql`,
  `tests/postgres_local_validation.py` (new local-only opt-in runner).
* Contracts/template: `contracts/reader_products.schema.json`,
  `contracts/weekly_report.schema.json`, `templates/historical_expenses.csv`.
* Dependencies: `requirements.txt`, `requirements-test.txt` (new, Finance only).
* Tests: `tests/test_core.py` (same 56 original case names with explicit fixture
  scope/evidence and changed exception/replay expectations),
  `tests/test_corrections.py` (new), `tests/test_sql_candidate.py` (new).
* Documentation: `README.md`, `docs/HANDOVER.md`, `docs/REVIEW_RECORD.md`,
  `docs/TEST_RESULTS.json`, `docs/operational_evidence.json`.

Unchanged: engine, app, shadow scorer, existing requirements, all Pine scripts,
Finance market-data validator, market_data_v2 schema/example, 11 original
market-data cases and all non-Finance branches/resources.

## Complete test result

Original baseline before correction: **67/67 PASS**.
Corrected final suite: **152 tests; 0 failures; 0 errors; 0 skipped**.
Python 3.12.14; jsonschema 4.26.0; pglast 8.5 / PostgreSQL grammar 18.6.

* Original Finance case names: 56 PASS (fixtures adapted to required explicit
  scope/coverage and changed quarantine/replay semantics; no case deleted).
* Original market-data cases: 11 PASS, unchanged.
* Correction and semantic regressions: 81 PASS.
* PostgreSQL 18 SQL/PL grammar tests: 4 PASS.

`docs/TEST_RESULTS.json` records every case name, group totals and test time.

**SQL/Python runtime parity: NOT EXECUTED.** No psql/postgres/initdb binaries are
available locally; package setup encountered an OS user-identity restriction.
No shared/cloud database was used and no cloud PostgreSQL was created. PG18 SQL
and PL/pgSQL parser checks passed; parsing does not validate actual catalog,
trigger, privilege, execution or concurrency behaviour.

A guarded local-only runner is supplied for a fresh existing
`finance_validation_isolated` on a scratch Unix socket. It checks PG18, scrubs
libpq environment configuration, has no TCP/URI/shared fallback, applies the
migration and acceptance, compares four arithmetic vectors, and runs concurrent
revision and forced-order checks. **These runtime checks have not passed yet.**
The replacement 001 is appropriate only because the previous 001 was an unapplied
review candidate; it is not a deployed-schema upgrade migration.

## Unresolved policy choices

No default was invented for economic recognition. `recognition_policy=None`
produces unavailable economic expense. Candidate supported choices:
straight-line service seconds (with cumulative quantised allocation), or
point-in-time recognition. Each row must explicitly select an approved method.
Paid-at evidence must be supplied separately for cash spend.

For ratification before real ingestion: allowed recognition methods and expenses
requiring exceptions; prepaid/API-credit treatment and refunds; historical FX
sources/rates/dates; shared/personal subscription allocation; tax treatment;
closed-cohort cost allocation; cash versus accrual break-even reporting policy;
and the versioned currency/evidence-namespace registry. Amounts and assumptions
were not invented. Numeric quantisation is a visible engineering policy proposal,
not an assertion of statutory/CEO accounting-policy approval.

## Named residuals

* **R-SQL-01:** PG18 runtime migration/acceptance, SQL/Python parity, actual database
  concurrency and privilege denial remain NOT_YET_DEMONSTRATED. Syntax checks are
  insufficient to authorise a migration.
* **R-WRITER-02:** Future durable staging/verification/reconciliation writers,
  composite coverage producer and scoped read publisher/role grants are not
  commissioned. Python journals are a locked deterministic reference, not a
  durable production database substitute. No authority is granted to Chief.
* **R-BROKER-03:** Complete OANDA type/division profiles, independent balance/NAV
  roll-forward and full-history coverage still need redacted real exports.
  Unknown types/fees/conversion embedding fail closed. Partial-close bridge
  reconciliation is implemented; complete broker economics is not yet proven.
* **R-POLICY-04:** The accounting/FX/allocation/registry choices above require
  recorded ratification before actual evidence is elevated to authoritative
  Finance truth. Missing choices remain unavailable.
* **R-EVIDENCE-05:** Historic invoices, scoped broker evidence, Make EOD schedule/
  pagination/retry blueprint and complete alert coverage remain outstanding.
  This correction does not fill operational gaps with estimates or credentials.

No named residual is concealed as a passed gate. Source is ready for another
independent review; it is not ready for deployment, production data ingestion,
agent activation or a profitability declaration.

## Exact next review instruction

Claude Code must review the **new exact Finance SHA in the delivery message**,
not old SHA `189587ce3e211b22782dc221af9afdf6a8cc123c` and not floating branch head.
Verify H1–H5 and M1–M9 regressions, SQL/catalog correctness, serializer trust
boundary, explicit unknowns/policies and all residuals. Keep PR #2 draft/unmerged.
No deploy or cloud validation DB is authorised. Any subsequent changes require a
new SHA and another independent assessment.

FINANCE_CORRECTED_CANDIDATE_READY_FOR_INDEPENDENT_REVIEW
