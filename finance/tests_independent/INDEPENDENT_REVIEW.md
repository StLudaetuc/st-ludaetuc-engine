# ST LUDAETUC CLOUD V1.0 — INDEPENDENT FINANCE FOUNDATION ENGINEERING REVIEW

| | |
|---|---|
| **Reviewer** | Claude, independent cross-provider reviewer. The implementation is by OpenAI GPT Work. |
| **Reviewed SHA** | `189587ce3e211b22782dc221af9afdf6a8cc123c`. This equals PR #2 head and `finance/foundation-v1` at review time. Base `2ca3fa49…` = `main`. |
| **Branch for this review** | `claude/finance-foundation-independent-review-v1`, based directly on the reviewed SHA. It adds files under `finance/tests_independent/` only. |
| **PostgreSQL** | **18.6** (conda-forge build, `x86_64-conda-linux-gnu`). Disposable, socket-only cluster; no network listener. Each test creates and drops its own database. No St Ludaetuc database was touched. |
| **Python** | 3.13.16; psycopg 3.2.9 (PostgreSQL tests only) |
| **Requirements basis** | `finance/docs/HANDOVER.md`, `finance/README.md` and the review brief. **The Finance Governance/Evidence pack was not available in this session** (not uploaded, not in the tree), so no comparison against it was possible. |

**Untouched:** PR #2, `finance/foundation-v1`, `main`, OANDA, Make, TradingView, Render, and the Chief/recovery-controller workstream.

## Tests

| Suite | Result |
|---|---|
| Existing author tests (`finance/tests`, 67) | **67/67 pass** |
| Author PostgreSQL acceptance (`postgres_acceptance.sql`) on PG 18.6 | **Pass.** The migration applies cleanly and a re-run is refused (`42P06`). |
| Independent Python (`test_independent_python.py`, 41) | 24 pass; **17 `test_finding_*` fail by design** |
| Independent PostgreSQL 18 (`test_independent_postgres.py`, 17) | 10 pass; **7 `test_finding_*` fail by design** |
| Secret scan | gitleaks 8.21.2 (checksum verified): 0 findings across the 35 fetched commits reachable from the SHA. This was a shallow fetch, so older history was not scanned. |

**Run commands:**
- `python -m unittest discover -s finance/tests_independent -t .`
- Add `FIN_REVIEW_PG_ADMIN_DSN=<disposable cluster>` to run the PostgreSQL suite.

## SQL/Python parity (PostgreSQL 18.6, randomised fixtures)

| Formula | Fixture | Result |
|---|---|---|
| Spend (period) and lifetime verified spend | 300 expenses with random sign, scale 0–12 dp, GBP/USD/EUR FX, all four verification states, 25 random periods | **Identical** |
| Broker gross, financing, other costs and net trading | 200 transactions, six component kinds including transfers, scale 0–8 dp, three FX rates | **Identical** |
| Run rate month and year | — | **Identical** |
| Run rate day | — | **DIFFERENT.** SQL `0.32854209445585215606` (20 dp) versus Python 80 significant digits. No rounding policy exists. |
| Money domain bound | — | **DIFFERENT at the edge.** `core.money()` takes `abs()` under the 28-digit context, so values within about 1e-10 of ±1e18 are rejected that SQL accepts. |
| Unit economics, reconciliation, break-even | — | **Single implementation each.** Python only for unit economics and reconciliation; break-even is gated differently in SQL (`period_summary`) and Python (`economics`). Parity is not applicable, but see H3, H4 and M6. |

**No binary float** appears in `core.py` or `oanda.py`: none in the AST, and `float()` is never called. `money()` rejects float, NaN, Inf and over-precision.

## Findings

### CRITICAL
None.

### HIGH

| ID | Finding | Evidence |
|---|---|---|
| **H1** | **A coverage withdrawal can be defeated.** `period_summary` selects the "latest" coverage assertion by the caller-writable `recorded_at`, and the latest reconciliation run by the caller-writable `as_of`. An assertion inserted with `recorded_at = 2099-…` stays "latest" forever, so a later withdrawal (complete = false) is ignored and the period still reports complete with a break-even conclusion. | `test_finding_future_dated_assertion_cannot_pin_latest` |
| **H2** | **Empty evidence can conclude break-even.** With no expenses, no broker transactions, and a reconciliation run whose first/last transaction IDs are NULL, one reviewed assertion yields `net_economic 0, remaining_break_even 0`, which reads as "paid for itself". Nothing requires an evidence manifest, counts or a transaction range. Python `economics()` behaves the same way. | `test_finding_empty_evidence_cannot_show_break_even` (PG), `test_finding_empty_evidence_cannot_conclude_break_even` (Py) |
| **H3** | **Project expenditure is subtracted once per account.** `period_summary(account)` subtracts *all* project spend from each account's trading. Two live accounts each report −100 for a project spend of 100, so any consumer summing accounts double-counts. There is no project-level break-even function. | `test_finding_project_spend_subtracted_once_across_accounts` |
| **H4** | **The Python actual-money scope is fail-open.** `economics()` defaults `system_environment='PRODUCTION'`, `financial_reality='ACTUAL'` and `financial_state='REALISED'`, so passing only `environment='live'` makes figures actual. The handover records legacy `environment="production"` on what is currently OANDA **Practice** trading, so mapping that label to `'live'` would make Practice P&L count toward actual break-even. SQL is safe here: it requires an explicit `broker_environment`. | `test_finding_live_scope_must_be_explicit_not_defaulted` |
| **H5** | **The weekly-report and Reader contracts cannot stop Practice, ESTIMATED or incomplete figures being labelled verified actual.** The schemas contain no `if`/`then`/`not`: `metric.state = "verified"` is valid on `broker_environment = OANDA_PRACTICE`, `financial_reality = SIMULATED` or `coverage_complete = false`; `state = "unavailable"` may carry a value; `verified` may have a null value. No serializer enforces these rules (the REVIEW_RECORD says so). | `test_finding_weekly_schema_forbids_verified_actual_from_practice_or_incomplete` |

### MEDIUM

| ID | Finding |
|---|---|
| **M1** | **No declared rounding or quantisation policy for reported GBP.** Run-rate day parity fails, and reported values carry up to 24 dp (SQL) or 80 digits (Python). |
| **M2** | **Reconciliation produces false failures:** (a) a partial close (several broker records for one trade) is reported as `duplicate_identity` plus a false `pnl_mismatch`; (b) broker-export lag *before* the EOD deadline is reported as `broker_record_missing`, while only the outcome side has `pending_eod`; (c) one malformed (float) `realised_pnl` aborts the whole run instead of becoming an exception. |
| **M3** | **No consistency check between OANDA nested and top-level P&L.** The adapter correctly treats top-level `pl` as cash and nested `tradesClosed`/`tradeReduced` as attribution, so there is **no double counting**. But nothing checks that the nested sum equals top-level `pl`, so attribution drift goes undetected. |
| **M4** | **Historical CSV import accepts unsafe and untrusted values:** spreadsheet formulas (`=`, `+`, `-`, `@`) in text fields; any `evidence_ref` (`javascript:`, `file://`, `../`, external URLs); self-declared `invoice_verified` (verification is spoofable in bulk); any three-letter code as a currency (`XXX`, `ABC`); and unbounded input size. |
| **M5** | **`recorded_at`/`created_at` are caller-supplied on every evidence table** (backdating). This also underlies H1. |
| **M6** | **Practice unit economics mix in actual spend.** `net_profit_per_trade_gbp` subtracts real project spend from **Practice** P&L. |
| **M7** | **Run rate (SQL and Python) silently excludes estimate/unverified schedules**, so 0 is indistinguishable from "no subscriptions". `lifetime_spend` does expose an uncertain count. |
| **M8** | **No accounting basis is declared for multi-period charges.** Annual subscriptions and prepaid periods are lumped into the month they were incurred, and `billing_start`/`billing_end` are never used. Period and operating profitability depend on this choice. |
| **M9** | **The reconciliation lifecycle is incomplete.** The Python reconciler is not wired to `finance.reconciliation_exception`. Exceptions have no resolution or status lifecycle, so the only remedy is a new run. The brief's states (MATCHED, BROKER_ONLY, INTERNAL_ONLY, VALUE_MISMATCH, INVESTIGATION_REQUIRED) have no mapping. `engine_event_evidence` allows conflicting duplicates (the SHA is in the primary key), and no code detects them. |

### LOW

| ID | Finding |
|---|---|
| **L1** | `money()` rejects values within about 1e-10 of ±1e18 that SQL accepts. |
| **L2** | Views lack `security_invoker`, so a future Reader grant on a view would bypass table ACLs. |
| **L3** | An owner or superuser can `DISABLE TRIGGER` and mutate history. This is documented, and demonstrated here. |
| **L4** | SQL does not enforce component sign by kind (a positive commission is accepted). Only the Python adapter enforces it. |
| **L5** | market-data-v2 does not validate the `timeframe` vocabulary. Floats are acceptable for raw price evidence. |
| **L6** | The governance pack was unavailable, so policy conformance is unproved. |

## Properties confirmed

**Immutability and corrections:**
- UPDATE, DELETE and TRUNCATE are rejected on every evidence table.
- Corrections append revisions; the original evidence and history are retained, and only the latest revision counts.
- Revision gaps are rejected.
- Concurrent corrections serialize, with exactly one winner.
- A duplicate source line, or a source-identity change, is rejected.
- Idempotent retries are no-ops; conflicting retries are rejected.

**Batch import:** the historical import is atomic.

**Inputs:** missing dates, timezone-naive timestamps, missing currency, missing or invalid FX, and GBP with FX ≠ 1 are all rejected.

**Arithmetic:** refunds net correctly; FX products are exact; there is no binary float.

**Broker:**
- Commission and guaranteed fee become negative costs; financing keeps its sign.
- Transfers and deposits never count as profit.
- Unknown types, missing fields, float fields, negative fees, wrong accounts and naive times are quarantined.
- Duplicate ingestion is idempotent; a conflicting duplicate is rejected.

**Practice versus actual:** in SQL, Practice accounts never produce an economic conclusion, even when labelled PRODUCTION.

**What blocks a conclusion:** estimates, unverified expenses, unresolved exceptions, simulation and unrealised states.

**Break-even arithmetic:** exact at the boundary (equal means not profitable; remaining is exact to the penny).

**Reconciliation:** matched chain, pending EOD → overdue at the exact deadline, value mismatch, currency mismatch, duplicate outcome, false closure, and scope mismatch all behave correctly.

**Contracts:**
- The Reader contracts have no arrays, no free filters or SQL, no account numbers or invoices, and bounded strings.
- market-data-v2 rejects literal `{{…}}` placeholders and carries no strategy-derived features.

## Corrections required before isolated PostgreSQL validation

1. **H1/M5:** make `recorded_at`/`created_at` server-assigned (for example a BEFORE INSERT trigger forcing `now()`/`clock_timestamp()`, or revoke the column from writers). Choose the "latest" assertion and run by a server-assigned monotonic sequence.
2. **H2:** coverage must bind evidence. Require a manifest: an expense count/digest, a broker transaction range that is non-null when transactions exist, and an explicit, reviewed `empty_period` flag. Without evidence, return NULL conclusions rather than 0.
3. **H3:** add a project-level break-even function. Subtract spend **once** against the sum of eligible (`OANDA_LIVE`/`PRODUCTION`/`ACTUAL`/`REALISED`) accounts. Per-account summaries should expose trading only.
4. **H4:** make `system_environment`, `financial_reality`, `financial_state` and `simulation_type` required keyword arguments with no defaults. Add an explicit mapping table for the legacy `environment="production"` value that resolves to Practice unless evidenced otherwise.
5. **H5:** add `if`/`then`/`not` constraints:
   - `state: verified` forbidden unless `broker_environment = OANDA_LIVE`, `financial_reality = ACTUAL`, `coverage_complete = true` and `simulation_type = null`;
   - `unavailable` ⇒ `value: null`;
   - `verified` ⇒ `value` non-null.
   Also provide a serializer with tests.
6. **M1:** declare one rounding policy (for example quantise reported GBP to 2 dp `ROUND_HALF_EVEN`, and the run-rate day to a fixed scale) applied identically in SQL and Python, and add parity tests.
7. **M2/M9:**
   - support partial closes, aggregating by trade over closing transactions;
   - add a pending state for broker-export lag before EOD;
   - convert per-row parse failures into exception codes;
   - map codes to the brief's status taxonomy;
   - wire reconciler output to SQL and define an exception resolution lifecycle.
8. **M3:** fail closed when nested `realizedPL` does not sum to top-level `pl`.
9. **M4:**
   - neutralise or reject formula-leading text;
   - allowlist `evidence_ref` schemes and paths;
   - do not accept `*_verified` from bulk CSV without bound evidence (import as `unverified` plus a separate verification step);
   - use an ISO 4217 allowlist;
   - set row and byte limits.
10. **M6/M7/M8:** null unit economics for Practice, or use a clearly separate field; have the run rate return uncertain-schedule counts; declare and implement the accounting basis (cash or accrual/amortised) for multi-period charges.
11. Lows: remediate or record as named residuals (security_invoker views; DB constraints on component sign; money bound).

**Re-review** should re-run both independent suites. Each `test_finding_*` test must pass, and the 67 author tests and the PostgreSQL acceptance script must still pass.

FINANCE_FOUNDATION_INDEPENDENT_REVIEW_REQUIRES_CORRECTION
