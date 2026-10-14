# ST LUDAETUC CLOUD V1.0 — FINANCE FOUNDATION INDEPENDENT REVIEW OF 4da760ba

| | |
|---|---|
| **Reviewer** | Claude, independent cross-provider reviewer. The implementation is by GPT Work. |
| **Reviewed SHA** | `4da760ba716c23afe140421ac06e19574525f5a8`. This was `finance/foundation-v1` head and PR #2 head at review time. |
| **Previous reviewed SHA** | `189587ce3e211b22782dc221af9afdf6a8cc123c` (not re-reviewed). |
| **Regression baseline** | `220d50b05cd2f1f6ef33b0159621c5f4e3b42bc2` (independent tests v1). It is unchanged and retained in this branch's history. |
| **Review branch** | `claude/finance-foundation-independent-review-v1`. The reviewed SHA was merged in so the tests run against it in place. Reviewer files are under `finance/tests_independent/v2/` only. |
| **PostgreSQL** | **18.6 runtime, really executed.** Local disposable cluster (conda-forge build), Unix socket only (`listen_addresses=''`), run as an unprivileged OS user. Every test creates and drops its own `finance_rev_*_isolated` database. No shared, Render or cloud database was touched. |
| **Python** | 3.13.16; psycopg 3.2.9, jsonschema 4.26.0, pglast 8.5 |
| **Governance basis** | Three Finance governance documents, all **DRAFT — NOT APPROVED**. They are used as context only; nothing in them is treated as ratified. |

**Untouched:** `finance/foundation-v1`, `main`, PR #2, Make, TradingView, OANDA, Render, Chief/Reader/recovery-controller, and all governance documents. Nothing was merged or deployed.

## Changed files (189587ce → 4da760ba)

21 files changed, +2,781 / −903:
- **Code:** `core.py`, `policy.py` (new), `reconciliation.py` (new), `serializer.py` (new), `oanda.py`.
- **SQL:** `migrations/001_foundation.sql`.
- **Contracts:** `reader_products.schema.json`, `weekly_report.schema.json`.
- **Tests:**
  - `tests/test_core.py` (fixtures adapted)
  - `tests/test_corrections.py` (new)
  - `tests/test_sql_candidate.py` (new)
  - `tests/postgres_acceptance.sql`
  - `tests/postgres_local_validation.py` (new)
- **Docs:** `README.md`, `HANDOVER.md`, `REVIEW_RECORD.md`, `TEST_RESULTS.json`, `operational_evidence.json`.
- **Other:** requirements files and the CSV template.

## Test evidence

Logs are in `finance/tests_independent/v2/evidence/`.

| Suite | Result |
|---|---|
| Author suite `finance/tests` (152, as claimed) | **152/152 pass.** All 152 executed case names match `TEST_RESULTS.json` exactly; none is skipped. |
| ↳ Original 67 cases (56 core + 11 market-data) | All 67 names are still present and pass. 11 market-data cases are unchanged. Core cases now run through test-local compatibility shims that supply explicit scope and coverage. Expectations changed in 6 places; I judged each change to be stricter or design-consistent: estimate run rate 0→None, counterfactual→raises, replay deduplicated, Practice/shadow −spend instead of None. |
| Author `postgres_acceptance.sql` on PG 18.6 **as shipped** | **FAILS** at the TRUNCATE probe (lines 29–34): `cannot truncate a table referenced in a foreign key constraint`. PG raises `0A000` before the trigger's `raise_exception`, and the probe accepts only the latter. |
| Author runner `postgres_local_validation.py` | Fails because of the acceptance failure above. With **only** that probe changed to also accept `feature_not_supported`, the runner **PASSES**: migration, acceptance, 4 parity vectors, revision race and forced-order checks. |
| Independent v2 Python (`test_v2_python.py`, 56) | **49 pass; 7 `test_finding_*` fail by design** |
| Independent v2 PostgreSQL 18 (`test_v2_postgres.py`, 32) | **22 pass; 10 `test_finding_*` fail by design** |
| Baseline v1 independent tests (58) run unchanged | Python: 16 pass, 4 fail, 21 errors. PostgreSQL: 1 pass, 1 fail, 15 errors. The errors are API/schema drift: functions renamed, explicit Scope/Coverage now required, columns renamed. Every v1 finding was re-expressed against the new API in v2. The v1 assertion failures are re-evaluated in the matrices below. |
| Secret scan, gitleaks 8.21.2 | 0 findings across the 35 commits reachable from the SHA, and 0 in the `finance/` tree. |

**Run commands:**
- `python -m unittest discover -s finance/tests_independent/v2 -t .`
- PostgreSQL suite: also set `FIN_REVIEW_PG_ADMIN_DSN='host=<local socket dir> user=<u> dbname=postgres'`.

## PostgreSQL 18 runtime outcome

| Check | Outcome |
|---|---|
| Migration on a clean database | **PASS** |
| Re-application | Refused (`42P06`); object count unchanged |
| Failed migration (injected error) | Full rollback; no `finance` schema remains |
| UPDATE / DELETE / TRUNCATE … CASCADE on all 15 evidence tables | **Blocked** (tables populated so that row triggers fire) |
| Forced `server_order` / `server_received_at` | **PASS.** A caller-supplied 999999999 and a 2199 timestamp are overwritten. |
| Cash profile triggers | **PASS.** Positive commission, `pl` mislabelled as commission, and an unreviewed component key (`halfSpreadCost`) are rejected. |
| Unknown transaction type | Blocks account completeness |
| Importer self-verification | Rejected |
| `VERIFIED` coverage with count 0 | Rejected |
| Unsafe `evidence_ref` | Rejected |
| Privileges | **PASS for denial.** PUBLIC has no schema or function access, and a role with no grants is denied. A narrowly granted writer cannot write either: invoker triggers need undocumented SELECT/UPDATE and sequence grants (see L-6). |
| Concurrency: revision race | **PASS.** One winner. |
| Concurrency: authority ordering | **FAIL**, see H1 |
| Exception lifecycle (SQL) | **FAIL.** No resolution row can ever be inserted; see N-2. |

### SQL/Python runtime parity

| Area | Fixture | Result |
|---|---|---|
| `finance.q()` vs `quantise()` | 3,005 values (±, 0–24 dp, explicit ties, ±1e18 edge) | **Identical** |
| Run-rate day `q(m*12/365.25)` | 20,000 random monthly amounts; `finance.run_rate()` checked end-to-end | **Identical** |
| `expense_totals` cash and economic | 60 random USD/EUR expenses × 150 random windows | **Identical** |
| Broker `account_totals` vs `broker_totals` | 120 random ORDER_FILLs (pl, financing, commission, fee) | **Identical** |
| Project totals (`project_totals` vs `project_economics`) | Two live accounts plus Practice | **Identical** (140 / 128 / 0) |
| Straight-line recognition, constructed vector (L-1) | `1234.559980108419` GBP over a 365-day + 1 s service period, October cut | **DIFFERENT by 1e-12.** SQL `104.853036081601`, Python `104.853036081600`. SQL numeric division rounds at its own result scale, then `q()` rounds again (double rounding). |

## H1–H5 result matrix

| ID | Claimed correction | Independent result | Classification |
|---|---|---|---|
| **H1** Authoritative ordering | Server sequence forced by trigger; latest-by-`server_order`; source clocks not used. | **Serial:** a future-dated source cannot beat a later withdrawal (SQL and Python), and caller orders are overwritten.<br>**Concurrent: defeated.** The `server_order` is taken when the row is inserted (`a_server_order` BEFORE INSERT, `001_foundation.sql:279`), not when it commits, and nothing serializes decisions for the same key.<br>Reproduced on PG 18 for verifications and for coverage:<br>1. T1 inserts a **withdrawal** and gets order N.<br>2. T2 inserts a **verification** and gets N+1, then commits.<br>3. T1 commits last.<br>`latest_verifications` / `latest_coverage` (`:45`, `:179`) still select the verification. `expense_totals` returns cash 10.000000000000 although the last-committed decision withdrew the evidence. SERIALIZABLE does not prevent this, because neither insert reads what the other writes. The same mechanism applies to `latest_runs` and `latest_resolutions`. The author's runner only checks that orders are distinct. | **PARTIALLY_RESOLVED** |
| **H2** Empty evidence | Coverage needs count, range, manifest and review; REVIEWED_EMPTY for zero. | SQL and Python return NULL with no, UNKNOWN, INCOMPLETE or WITHDRAWN coverage, and on count mismatch. `VERIFIED` cannot be empty. Explicit REVIEWED_EMPTY yields a genuine 0. Related completeness gaps are in N-3 and N-4. | **RESOLVED** |
| **H3** Spend once | Live accounts summed first; spend subtracted once. | SQL `project_totals` = Python: 95+45−12 = 128. Practice 9,999 is excluded. An incomplete or non-boolean eligible account blocks the result. Duplicate accounts are rejected. | **RESOLVED** |
| **H4** Explicit scope | Five required dimensions, no defaults. | `Scope` has no defaults. `production` / `live` / lowercase labels are rejected, as are SIMULATED+REALISED and ACTUAL+simulation. Only `OANDA_LIVE/PRODUCTION/ACTUAL/REALISED/None` is cash. The adapter requires a Scope object. In SQL, Practice labelled PRODUCTION still contributes 0 and leaves the full spend to break even. | **RESOLVED** |
| **H5** Contracts / serializer | Single semantic egress. | **Strong:** forged values, rewritten reconciliation state, withdrawn coverage, excess precision, unavailable-with-value, Practice economic keys, a missing `jsonschema` (fails closed) and self-declared completeness are all rejected.<br>**Defects:**<br>(a) Weekly `realised_pnl`, `broker_costs` and `other_trading_costs` and Reader `observed_expectancy` / `estimated_spend` are in no guard set. A **Practice** report publishes `realised_pnl` as `verified`/`FACT` (`serializer.py:15`, `:105`).<br>(b) `verified`/`FACT` with `value: null` is accepted (`:91`).<br>(c) See N-5: lifetime metrics are judged only against the week's coverage.<br>Provenance authenticity is caller-trusted by design (R-WRITER-02); a test documents it. | **PARTIALLY_RESOLVED** |

## M1–M9 result matrix

| ID | Independent result | Classification |
|---|---|---|
| **M1** Rounding | Policy declared (`money_scale12_half_away_v1`). Ties are away from zero, and negative zero is normalised. Random parity holds. One constructed recognition vector shows 1e-12 SQL double rounding (L-1). Presentation rounding (pence) is not defined (policy item). | **PARTIALLY_RESOLVED** |
| **M2** False failures | (a) Partial close REDUCE+CLOSE rolls up: **fixed**. (c) Malformed money is quarantined per row and the run continues: **fixed**. (b) **Still open:**<br>– Before the deadline, trade_opened arrived but no broker slice yet ⇒ `INTERNAL_ONLY broker_record_missing` (`reconciliation.py:111`).<br>– Outcome posted before the broker CLOSE slice ⇒ `INVESTIGATION_REQUIRED outcome_without_broker_closure` (`:127`).<br>Neither respects `as_of < outcome_due_at`. | **PARTIALLY_RESOLVED** |
| **M3** Nested P&L | Nested `realizedPL` / `positionFinancings` must sum to the parent, otherwise the transaction is quarantined; the batch continues. | **RESOLVED** (see N-6 for routine non-cash types) |
| **M4** CSV | Formula text neutralised. Bulk self-verification is impossible. Evidence namespaces are allowlisted. Currency registry enforced. 1 MiB / 1,000-row / 512-char bounds. Atomic on late failure. (Identities may still start with `-`: L-7.) | **RESOLVED** |
| **M5** Caller timestamps | `server_received_at` and `server_order` are forced. Source times are kept separately and not used for authority. | **RESOLVED** (ordering under concurrency: H1) |
| **M6** Practice unit economics | `HYPOTHETICAL_NON_CASH`, `actual_net_profit=None`, closed-trade denominator only. | **RESOLVED** |
| **M7** Run rate | Unknown or unverified ⇒ NULL with `excluded_estimates` / `unknown_records` counts. A reviewed-empty zero is distinguishable. | **RESOLVED** |
| **M8** Accounting basis | Cash by `paid_at`; economic by explicit per-expense policy (straight-line seconds or point-in-time); a missing policy ⇒ NULL. Adjacent windows (including across the DST change) sum exactly to the converted amount. Ratification is still open. | **RESOLVED** (engineering) |
| **M9** Lifecycle | Six canonical states; the Python `ReconciliationJournal` resolve/withdraw lifecycle works. **SQL `exception_resolution` cannot accept any row:** RESOLVED and WITHDRAWN inserts both fail at runtime (N-2). SQL still accepts conflicting `engine_event_evidence` duplicates, because the SHA is in the primary key (`:123`). | **PARTIALLY_RESOLVED** |

## Newly discovered findings

### HIGH
None beyond the open H1 concurrency defect above. It is not repeated here.

### MEDIUM

| ID | Finding | Evidence |
|---|---|---|
| **N-1** | **The candidate's PostgreSQL acceptance script fails as shipped.** Its TRUNCATE probe expects `raise_exception`, but PG 18 raises `0A000` from the FK check first. All later assertions never ran in the delivered form. The protection itself holds. **This blocks the "acceptance passes" requirement.** | `tests/postgres_acceptance.sql:29-34`; `test_finding_author_acceptance_script_passes_on_pg18` |
| **N-2** | **SQL exception resolution cannot run.** `resolution_evidence()` aliases a table `old`, which collides with PL/pgSQL `OLD`. Every RESOLVED **and** WITHDRAWN insert fails with `column reference "old.run_id" is ambiguous`. Grammar-only checks cannot catch this. | `001_foundation.sql:150-160` (`:154`); `test_finding_sql_exception_resolution_insertable` |
| **N-3** | **"Latest" coverage and runs ignore the period.** `latest_coverage` keys on (domain, scope) and `latest_runs` on account (`:179-180`). Asserting November makes a previously complete October revert to UNKNOWN. Broker coverage also requires `evidence_count` to equal the queried window's count. So a lifetime window and a week cannot both be complete, which defeats the weekly report's lifetime + week structure. | `test_finding_coverage_of_one_period_survives_assertion_of_the_next` |
| **N-4** | **Reconciliation results are not bound to the evidence population.** `coverage_ok` needs only *one* result row to exist (`:187`). Three broker transactions with one MATCHED result count as complete. An exception can disappear by being omitted from a newer run, without a resolution. | `test_finding_reconciliation_results_must_cover_population` |
| **N-5** | **Weekly lifetime metrics are validated against the week only.** The serializer derives `complete` from one coverage key and window (`serializer.py:43`). `lifetime.remaining_break_even` is accepted as FACT with coverage of only the final week. | `test_finding_lifetime_metrics_bound_to_lifetime_coverage` |
| **N-6** | **Routine OANDA non-cash types cannot be represented.** MARKET_ORDER, ORDER_CANCEL, STOP_LOSS_ORDER, TAKE_PROFIT_ORDER and others are `INVESTIGATION_REQUIRED` in the adapter (`oanda.py:32`), and in SQL any such transaction makes the account incomplete. A real export can therefore never be complete. Pre-filtering it would break the first/last-ID manifest. | `test_finding_routine_non_cash_oanda_types_are_not_investigations`; SQL `test_unclassified_transaction_blocks_account_completeness` |
| **N-7** | **The revised Closed Trade Data Sync schedule is not modelled.** `outcome_due_at` is free caller input. No component derives the due time from 00:30 Europe/London Tue–Sat, DST, or the no-Monday-run gap. A missed scheduled run is indistinguishable from a missing outcome (both become `BROKER_ONLY outcome_overdue`). The state machine is correct *when given the true due time*: a reference schedule, tested across the 25 Oct 2026 DST change, yields correct PENDING / overdue. | `test_finding_due_time_is_derived_not_caller_asserted`; `test_reference_schedule_across_dst_end_2026` |

### LOW

| ID | Finding |
|---|---|
| **L-1** | SQL/Python double rounding: `q(a/b)` after PG's own division rounding, e.g. recognition `104.853036081601` vs `104.853036081600`. It affects reproducibility claims, not material amounts. Fix: compute with explicit scale (`div(a*10^k, b)` / `round(a/b, 40)`), or quantise only once from an exact rational. |
| **L-2** | Views are still not `security_invoker` (previously L2). |
| **L-3** | The owner can `setval()` the authority sequence. `server_order` has no UNIQUE constraint, so duplicate orders make `DISTINCT ON` arbitrary. The owner can also disable triggers (previously L3). |
| **L-4** | Conflicting `engine_event_evidence` duplicates are accepted in SQL (part of M9). |
| **L-5** | `weekly_cash_spend` / `monthly_cash_spend` use UTC weeks. A Monday 00:15 BST payment falls in the previous week. The reporting timezone is a policy item. |
| **L-6** | Least-privilege writer roles are undesigned. Invoker triggers require SELECT/UPDATE on identity rows and USAGE on the sequence (R-WRITER-02). |
| **L-7** | CSV identities may start with `-`, e.g. `-A1`. They are not executable as written. |
| **L-8** | The author runner's "forced-order" check verifies only distinctness, not commit-order semantics. It does not check that the target database is empty. Its isolation is otherwise sound: it refuses TCP, URI and non-repository sockets and scrubs `PG*` variables (tested). |

**Previously LOW, now resolved:**
- L1: the money bound now matches between SQL and Python.
- L4: component sign is enforced in SQL.

**Unchanged residual:**
- L5: market-data timeframe vocabulary.

## Financial-reality safeguards (Assignment A/D summary)

| Safeguard | Result |
|---|---|
| Practice vs Live | Practice never contributes to project P&L in SQL or Python. In SQL, 1,000,000 of Practice profit leaves a full remaining break-even of 10. Exception: serializer labelling, H5(a). |
| Simulated vs realised | SIMULATED cannot be REALISED (Scope, SQL check, weekly schema const). |
| Absent evidence vs zero | NULL unless coverage is VERIFIED or REVIEWED_EMPTY. A FACT may still carry null (H5(b)). |
| Project expenditure deducted once | Yes (H3) |
| Provenance authenticity | The manifest ID must match the current journal coverage. The journal itself is caller-trusted (R-WRITER-02). |
| Incomplete reconciliation | Unresolved results in the latest run block. Gaps remain: N-3, N-4 and H1. |
| HTTP 200 `/trade-opened` | Not treated as execution. A missing `fill_status` gives INVESTIGATION and is not a closed trade. |
| CANCELLED orders | Counted in attempted orders only; never in filled or closed trades. CANCELLED with broker evidence gives INVESTIGATION (Python). SQL `order_denominators` counts filled and cancelled separately. |
| Late outcomes | A later MATCHED run supersedes. The Python resolution path works; the SQL path is broken (N-2). |

## Open CEO policy choices (DRAFT governance; nothing is treated as ratified)

1. **`money_scale12_half_away_v1` versus the earlier half-even proposal.**
   - The two differ only on exact ties at the 13th decimal place. These are rare: they need a product or quotient whose digits after 1e-12 are exactly `5000…`.
   - Half-away has a tiny upward-magnitude bias on ties. Half-even is unbiased.
   - Half-away matches PostgreSQL `round(numeric)` natively, which is why parity holds without custom SQL. Half-even would need a custom SQL function and parity tests.
   - Neither choice decides **presentation** rounding to pence, which is undefined. Reports carry 12 dp.
   - For statutory or bank reconciliation, the CEO should decide (i) the internal scale and tie rule and (ii) the reported precision and where rounding occurs (per line or per total).
   - This is a policy choice, not an engineering failure.
2. **Reporting currency.** GBP is hard-coded (`const GBP`); the draft decision on GBP is unratified. **FX source rule:** `fx_source` is free text with no approved-source registry.
3. **Estimates.**
   - The candidate makes *all* official expense totals NULL while any ESTIMATE exists.
   - That is stricter than the draft's "exclude from official break-even, separate estimated view".
   - No separate estimated view exists.
4. **Break-even basis.** The draft defines lifetime break-even on full-project economic P&L and operating P&L excluding one-off historical cost.
   - The candidate's `remaining_cash_break_even` is cash-basis.
   - `operationally_profitable` uses *all* recognised expense, with no one-off/recurring split. The `recurring` flag is stored but unused.
   - Code depending on this decision: `core.project_economics`, `finance.project_totals`.
5. **Evidence-class taxonomy.**
   - Draft classes: VERIFIED_INVOICE, VERIFIED_PROVIDER_BILLING, VERIFIED_PAYMENT, CEO_CONFIRMED_LEGACY, ESTIMATED, UNKNOWN.
   - Candidate classes: invoice_verified, manual_verified, estimate, unverified.
6. **Report labels.**
   - Draft labels: FACT, CALCULATED, ALLOCATED, FORECAST, ESTIMATED, UNKNOWN.
   - Candidate labels: FACT, ESTIMATED, UNKNOWN, HYPOTHETICAL (+MIXED).
   - Allocated cost per trade is labelled FACT.
7. **OANDA cadence and cut-off.** PENDING_EOD_SYNC semantics are implemented. The due-time derivation and the "last completed sync" report cut-off are not (N-7).
8. **Other draft rules not implemented:** the materiality rule (5% / £100), the 30-trade forecast minimum, and the reporting timezone (UTC vs Europe/London).
9. **Candidate proposals needing ratification:** the 8-currency registry, recognition methods and the evidence namespaces.

## Residuals requiring future evidence

- **R-WRITER-02:** writer, verifier and publisher roles and grants; an authenticated journal producer.
- **R-BROKER-03:** non-cash type profiles and real redacted exports. N-6 must be fixed first.
- **R-POLICY-04:** all items above.
- **R-EVIDENCE-05:** invoices, Make EOD blueprint evidence, and full alert coverage.

## Correction instructions for GPT Work

1. **H1 (concurrency).**
   - Serialize per authority key *before* allocating `server_order`: add a BEFORE INSERT trigger that fires first, e.g. `"0_authority_lock"`, taking `pg_advisory_xact_lock(hashtextextended(<table>||key,0))`, or `SELECT … FOR UPDATE` on a per-key head row.
   - Allocate `nextval` only after the lock, so per-key order equals commit order.
   - Add compare-and-set: each decision carries `supersedes_server_order`, which must equal the current latest, otherwise reject.
   - Add `UNIQUE(server_order)`.
   - Add a two-connection PG test, mirroring `H1Concurrency`, to the runner.
2. **N-1:** fix the TRUNCATE probe to accept `feature_not_supported`, or probe a non-referenced table. Then run the runner on real PG18 and record its output.
3. **N-2:** rename the alias (`old` → `src`) in `resolution_evidence()`. Add runtime tests for RESOLVED, WITHDRAWN, cross-account and earlier-run rejection.
4. **N-3:** key `latest_coverage` / `latest_runs` by `(domain, scope, period_start, period_end)`, or evaluate the latest assertion *covering* the queried window. Support lifetime and week windows simultaneously, e.g. compose a window from contiguous verified sub-periods.
5. **N-4:** require one reconciliation result per broker trade and transaction in the run window, with result count and identity coverage checked in `coverage_ok`. Make omission of a previously excepted trade an exception unless it is resolved.
6. **H5 / N-5:**
   - Replace key-name allowlists with an exhaustive metric registry. Every schema key must map to a class (`ECONOMIC`, `CASH_ECONOMIC`, `TRADING`, `COST`, `ESTIMATE`, `FORECAST`, `ALLOCATED`), and unknown keys must be rejected.
   - Add `realised_pnl`, `broker_costs`, `other_trading_costs` and `observed_expectancy` to TRADING.
   - Require a non-null value for `verified`.
   - Validate each metric group against its own coverage key and window (lifetime vs this_week vs run_rate).
7. **M2(b):** pre-deadline broker lag (missing broker slice for a FILLED open, or a missing CLOSE with an outcome present) ⇒ `PENDING_EOD_SYNC` while `as_of < outcome_due_at`.
8. **N-6:** add reviewed zero-cash profiles for routine non-cash OANDA types. They should be recorded with no cash components and count toward ID-range completeness. Truly unknown types should remain exceptions.
9. **N-7:**
   - Add a deterministic `next_sync_due(closed_at)` for 00:30 Europe/London Tue–Sat, using zoneinfo and DST-tested, versioned as policy.
   - Add sync-run evidence (scheduled, started, completed, failed) so a missed run ⇒ INVESTIGATION or `PENDING_SYNC_MISSED`, not `outcome_overdue`.
10. **L-1 to L-8:** fix or record as named residuals. In particular: single-rounding SQL division, `security_invoker` views, `UNIQUE(server_order)`, an SQL conflicting-event constraint, and documented least-privilege grants.
11. **Do not change** policy items 1–9 without recorded CEO ratification. Keep unknown choices returning NULL.

**Runtime proof:** obtained on a disposable local PostgreSQL 18.6 cluster. The candidate's own acceptance evidence fails as delivered (N-1), and runtime defects N-2, N-3, N-4 and H1 were found. Grammar parsing alone did not and could not detect them.

FINANCE_FOUNDATION_INDEPENDENT_REVIEW_REQUIRES_CORRECTION
