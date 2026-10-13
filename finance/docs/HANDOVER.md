# ST LUDAETUC CLOUD V1.0 — FINANCE & BROKER RECONCILIATION FOUNDATION HANDOVER

Prepared 13 October 2026. Isolated, undeployed review candidate. This handover
does not approve a gate, activate an agent, grant authority, or certify current
project profitability. No historical amounts or broker totals were invented.

## 1. Current trading/P&L data-path map

TradingView CandidateGenerator → per-instrument Make orchestration → Render
engine `/evaluate` → Make approval branch → OANDA Practice order → Make
`/trade-opened` → PostgreSQL `events`. At closure: OANDA Practice → Make Closed
Trade Data Sync (intended end-of-day reconciliation) → `/trade-outcome` →
`events` → `master_trading_log` view, bridged to `trade_opened` and `prediction`.
The engine does not place OANDA orders. Market candles use a separate direct
TradingView → Render `/market-data` → `events` path, without Make.

The repository source inspected was commit
`2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`, engine version 3.11.0. The Finance
branch started at the same commit. Existing engine, app, requirements, scorer
and Pine files are unchanged. Render service metadata identifies the engine as
`srv-d71v0v7pm1nc73f60520`, Frankfurt, tracking `main`, auto-deploy on main
commit, PR previews disabled. No deployment or service configuration changed.

Read-only sources:

* [Engine source](https://github.com/StLudaetuc/st-ludaetuc-engine/blob/2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c/engine.py).
* [Reconciled Charter](https://docs.google.com/document/d/1k2Lu4LU7InOF00ZYwWY4G-S1FPK6Q1_7UtC04pKKgho/edit), Finance role and net economics.
* [Current reconciled Document 2](https://docs.google.com/document/d/1s8OpywE_RnajywpFQaEAQTfR3ljLrOqhnpLF9XQONAM/edit), read only; no evidence edits.
* [Reconciled Data Contract](https://docs.google.com/document/d/1m61jVC9oRT_HilQP39u0tVyBIBhUzV5dsIoEANzAsRk/edit), §§100–109 and 146–148.
* [Reconciled PostgreSQL Architecture](https://docs.google.com/document/d/1gwQfy9G1T-GNrXuFFJ2VYOV9-NpBnF_pVLzhhGKzlI0/edit), §§135–140.

Older canonical-folder copies were also read during discovery; their approval
records differed. The reconciled Charter is headed APPROVED. Document 2 contains
contradictory retained draft/approval sentences. This implementation follows the
supplied isolation instruction and reconciled financial semantics, and does not
resolve document approval drift or change any document or PG-P# status.

## 2. Current captured fields and data gaps

This table describes what the inspected code can capture, **not** a claim that
every live event contains those fields. Private DB access was unavailable.

| Path | Explicitly captured / derived | Missing guarantee or gap |
| --- | --- | --- |
| prediction | Pine signal ID, instrument/timeframe, time, strategy/model versions, approval/rejection, predicted probability/net R, raw signal, account snapshot, trade plans | No authoritative accounting or expense records |
| trade_opened | Pine ID; raw account snapshot; complete forwarded order response; final submitted plan; derived OANDA trade ID, create/cancel/reject transaction IDs, last transaction ID, batch/request/related IDs, created/filled/cancelled/rejected times, units, SL/TP, fill status, rejection reason | No explicit account/currency economic scope validation; no separately derived fill transaction ID; extraction handles `tradeOpened`, not all reductions/closures/netting |
| trade_outcome | broker trade ID, instrument/timeframe, closed_at, outcome, realised R, `realized_pnl` as float, exit price/reason; extra fields allowed in payload | Financing, commission, conversion costs, adjustments, broker transaction ID, account currency, account alias, balance/NAV, units/partial close sequence and raw closure transaction not required or normalised |
| master_trading_log | closed trade ID, bridged signal ID, instrument, closed/open/predicted times, outcome, exit reason/price, realised P&L, fill/approval status, strategy/model versions and expected metrics | Missing joins become NULL; no exception register; unscoped joins; no financing/net economics; fallback signal_id can misleadingly be broker ID |
| account snapshot | Arbitrary forwarded object, stored at decision/open time | Not an authoritative daily balance/NAV journal, no demonstrated currency/completeness constraints |

Raw open order responses may contain commission, financing, conversion factors
or balance, but this is not evidence of a complete authoritative ledger. Closure
payload contents could not be inspected. `TradeOutcome.realized_pnl` uses binary
float, contrary to the new deterministic decimal accounting boundary; no engine
change was made. Finance must consume decimal broker exports, not treat this
float event value as financial truth.

`log_event` uses `ON CONFLICT DO UPDATE`, overwriting payload/status and updating
received_at. Therefore legacy received_at is the latest receipt, not guaranteed
first receipt; legacy evidence is not immutable. Finance stores separate source
hashes and snapshots. No attempt was made to repair existing engine DDL or data.

## 3. Closed Trade Data Sync assessment

The CEO's intended closure behaviour is explicitly preserved: closed trades are
reconciled at end of day through Make Closed Trade Data Sync, not necessarily
immediately at broker closure. Eight successful `/trade-outcome` HTTP 200 calls
were observed in a burst on 12 October 2026, **22:05:04.375–22:05:04.956 UTC**
(23:05 Guernsey). Read-only log window: 20:00–00:00 UTC, no further pages in that
window. This supports a batch ingestion path, but cannot prove which Make
scenario sent them or verify its configured cadence/timezone.

The endpoint's docstring says “not yet wired” and incorrectly suggests the Pine
ID is the outcome ID; the view/source comments and observed calls indicate that
this comment is stale. Exact closure mappings, schedule/timezone, pagination,
watermark, lookback, retries, partial closures, financing collection and currency
mapping remain **UNVERIFIED**. No Make/OANDA connector was available; no account
credentials requested and neither system was changed.

Reconciliation accepts an explicit `outcome_due_at` from the reviewed EOD policy.
A closed broker trade without outcome is `pending_eod` before that deadline and
`outcome_overdue` afterwards. Pending does not mean reconciled: economic
conclusions remain incomplete. Broker closure time, transaction time, sync time
and report time must remain separate. Configure grace period and DST-safe zone
from actual scenario evidence; do not invent a default deadline.

Next evidence needed: a redacted Make scenario blueprint/schedule and a scoped
broker transaction/trade export covering a complete day with closing balance.
Compare transaction IDs, broker trade IDs and signal bridges one-to-one; verify
complete pagination and a successful replay without extra economic cash.

## 4. Market-data webhook diagnosis

Read-only engine logs showed paired parse failures and 422s from
23:11:12 through 23:14:04 UTC on 12 October in the sampled page (more pages
existed; this is not the full affected window/count). Exact logged body:

```text
{"ticker":"{{ticker}}","interval":"{{interval}}","time":"{{time}}","open":{{open}},"high":{{high}},"low":{{low}},"close":{{close}},"volume":{{volume}}}
```

`alert()` sends its dynamically constructed string without placeholder
substitution. Unquoted numeric placeholders make this invalid JSON. The engine
attempts JSON parsing, substitutes bare NaN/Infinity tokens with null, then
parses again; braces are not NaN/Infinity, so it returns 422. This failure is
inside the route's raw-body parser, not model binding. Quoting all placeholders
would make JSON parse, but would preserve false strings and corrupt lineage;
that is not a fix.

Symbols/timeframes and exact failing alert IDs are **not observable** from these
failed payloads: those values are literal placeholders too. Do not attribute
them to GBPUSD or a 1m chart by guess. Recent sampled `/market-data` 200s also
exist, so some ingestion continues; this alone cannot prove data quality.
No live TradingView alert or script was changed. Logger v1.1 attribution is
CEO-supplied; the logger itself is absent from the inspected source tree.

TradingView reference: [Alerts FAQ](https://www.tradingview.com/pine-script-docs/faq/alerts/)
and [Alerts concepts](https://www.tradingview.com/pine-script-docs/concepts/alerts/).
Alert instances retain a snapshot of script/settings/chart context; changing
the chart script alone does not replace the existing alert.

## 5. Proposed market_data_v2 contract and migration

See `contracts/market_data_v2.example.json` and
`contracts/market_data_v2.schema.json`. Example values are synthetic fixtures.

| Field | Pine/raw meaning |
| --- | --- |
| schema_version/source | `market_data_v2` / `tradingview` |
| ticker/ticker_id/provider | `syminfo.ticker`, `syminfo.tickerid`, `syminfo.prefix`; provider here means TradingView quote feed |
| asset_type/base_currency/quote_currency | `syminfo.type`, `syminfo.basecurrency`, `syminfo.currency`; explicit unavailable currency null |
| timeframe | `timeframe.period`, preserve original TradingView units |
| bar_open_time/bar_close_time | `time` / `time_close`, UTC epoch milliseconds; close time may be unavailable on non-time charts |
| open/high/low/close/volume | Raw bar OHLCV; finite prices, null unavailable volume; volume unit/provider semantics not invented |
| minimum_tick/is_confirmed | `syminfo.mintick` / `barstate.isconfirmed` |
| interval/time | Transitional aliases of timeframe/bar_open_time for existing endpoint |

Build strings from Pine values using `str.tostring()` and safe JSON escaping,
not `{{...}}`. No RSI, ATR, Bollinger, conviction, regime or strategy indicators
belong in this raw payload. Prices remain raw candle evidence; Finance money
uses decimal strings separately. Server owns received_at and
`received_at - bar_close_time` latency (can be negative for provisional bars).
Unavailable bar-close timestamp produces unavailable latency, not fabricated 0.

Compatibility: current `/market-data` reads `ticker|symbol`, `interval`, and
`time`; arbitrary extra fields are retained. Thus v2 with the aliases is
syntactically accepted, while v2 without aliases silently loses timeframe and
falls back to server time. The legacy event envelope remains schema `v1` even
when payload says `market_data_v2`. Current endpoint does not enforce finite
OHLC, minimum tick, placeholder exclusion, candle ordering, confirmation or
schema compatibility. Existing analytics expect top-level OHLC keys, retained
here. Millisecond epoch timestamps are accepted by `_parse_event_time`.

Future alert recreation checklist, to perform only under separate authority:

1. Inventory alert ID, feed-qualified symbol, timeframe, logger version, inputs,
   frequency, expiry and webhook target; export/save previous configuration.
2. Review/version the logger and JSON schema; stage fixtures including null
   volume, session close, non-time bars and special-character metadata.
3. Validate v2 aliases locally and against an isolated ingestion receiver.
   Verify event times, feed identity, OHLC invariants and confirmation.
4. Specify bar-close-only cadence; retain provisional bars only if explicitly
   required, with separate deterministic IDs/revisions.
5. Disable/recreate each old alert from the new script snapshot using its
   recorded symbol/timeframe. Avoid overlapping duplicate alerts.
6. Observe a real bar, reject literal placeholders, confirm persistence and
   latency, inspect each expected symbol/timeframe and gap cadence.
7. Record deployed script digest/alert inventory and rollback instructions.
   Roll back by restoring recorded alerts, never by guessing old settings.
8. Govern strict server validation and envelope-v2 support separately; do not
   silently modify legacy ingestion or reuse Chief resources.

## 6. Finance data model

`migrations/001_foundation.sql` creates only a new `finance` schema, explicitly
and once. It has no dependency on current events views, no startup integration,
and no grants/roles that could grant Chief authority.

| Entity | Purpose |
| --- | --- |
| expense_identity + expense_revision | Unique source line, immutable revisions, vendor/category, date, original money/currency, GBP rate/source/date, generated GBP amount, billing period, service/environment, recurrence, evidence, verifier/state/reason |
| subscription_schedule | Evidence-backed recurring forecast, effective window; correction points to superseded schedule; overlaps fail run-rate reads |
| broker_account | Opaque account alias, broker and system environments, currency, ACTUAL broker action semantics |
| broker_transaction | Authoritative transaction ID/time/type/order/batch/request, source digest/reference/parser, state and simulation dimension |
| broker_component | Unique account/transaction/component signed economic effect, trade attribution, evidenced FX to GBP |
| broker_trade_action | Broker open/reduce/close slices, units/price/reason; attribution evidence without adding nested amounts to cash totals |
| broker_snapshot | Balance/NAV with explicit missing-balance/missing-NAV state |
| engine_event_evidence | Scoped immutable snapshot of prediction/opened/outcome and legacy source hash; missing links remain evidence gaps |
| reconciliation_run/exception | Bounded period, EOD deadline, export completeness and transaction cursors; explicit exception codes |
| coverage_assertion | Reviewed period/account completeness, independent from empty exception counts; latest assertion/run required |
| report_snapshot | Immutable bounded issued result with formula/source manifest |

NUMERIC and Python Decimal are authoritative; money floats rejected. Currency
codes are uppercase three-letter syntactic values; supported currency registry
and approved FX source whitelist must be reviewed before real import. GBP rate
must equal 1. Original values/rates retained; report rounding is presentation
only. Ledger basis is incurred spending, not accrual amortisation or capital
funding; balance/NAV/deposits are distinct from realised P&L/project expenditure.

Canonical semantics: OANDA Practice transactions are **ACTUAL broker actions**,
not counterfactual simulations. `OANDA_PRACTICE` separately prevents their gains
from funding actual lifetime break-even. The broker ledger accepts no simulated
transactions. A future simulated Finance domain must keep simulation_type,
financial_reality and financial_state independent. Live economic conclusions
require OANDA_LIVE + PRODUCTION + ACTUAL + REALISED, complete evidence and no
simulation. Existing legacy environment=production must not imply live money.

Lowercase expense reporting categories are a proposed local presentation
classification, not a change to Document 4. Before integration, map them to the
controlled category registry: infrastructure→RENDER_COMPUTE/POSTGRESQL;
api_ai→AI_INFERENCE; subscriptions→MAKE/TRADINGVIEW/SOFTWARE_SUBSCRIPTION;
rd_testing→RESEARCH; data→MARKET_DATA; other→OTHER. Domains/tooling need reviewed
registry mapping. No controlled contract was rewritten.

## 7. Historical-expense import design

`templates/historical_expenses.csv` is a header-only CEO template; there are no
pre-filled guesses. Supports TradingView, Make, Render, OpenAI/ChatGPT/API,
Anthropic/Claude/API, xAI/Grok/API, GitHub, domains, market data, development tools
and arbitrary vendors. Preserve invoice links or manually attested evidence
references; do not embed credentials/payment details.

Populate timezone-aware incurred/billing/FX dates, decimal strings, original
currency, GBP-per-original-unit rate, source and evidence. Dates are not guessed
from a card charge description. `source_key` is a stable vendor/invoice/line
identity (or manual attestation ID), reused across retries and corrections.
Separate invoice lines are separate identities. Duplicate representations of
the same purchase must share the same identity; review cannot be replaced by a
description hash.

States: `invoice_verified`, `manual_verified`, `estimate`, `unverified`. Verified
rows need `verified_by`; estimates/unverified never enter verified spending.
Manual verification is visible separately from invoice verification. Historical
figures recalled from chat are not imported as verified. Refunds use negative
actual expense entries with their own source evidence; correcting the original
amount appends the next revision with a reason. No destructive update/delete.

`import_expenses()` validates an entire CSV against a candidate ledger and swaps
only after every row validates. Exact retries are idempotent; conflicting retry,
source duplication, revision gaps and invalid money/FX reject the batch. SQL
serialises expense revision inserts by locking identity. SQL adapter must use a
single DB transaction, compare exact retries before attempting insert, insert
identity once, and explicitly map CSV to migration fields; that persistence
adapter is pending. Source material and import manifest require hash, reviewer,
range and row count. Verified import is not confirmation that all historical
spending has been supplied; require a CEO scope/completeness attestation.

Run-rate schedules are separate forecast inputs, not expense transactions.
Changing a forecast supersedes a schedule with a new evidence-backed row;
Python `run_rate()` consumes the effective schedule projection. Never add
annualised subscription costs to actual invoices again.

## 8. Broker reconciliation model

Use only scoped, already-authorised exported OANDA evidence. No credentials,
network calls or order capability exist in the adapter. Scope account alias,
broker environment, account currency and system environment before ingestion.
The raw private account ID is checked internally, not exposed in Reader output.

Transaction uniqueness: account + transaction ID; component uniqueness adds
source field/path. Import same hash once, reject conflicting same-ID evidence.
Do not sum trade snapshot realised totals with closing transaction cash. Do not
sum nested `tradesClosed`/`tradeReduced` and top-level `pl`; nested amounts are
attribution only. Partial-close transaction slices are retained individually;
fully closed unique trades, not fills, define the report trade denominator.

`oanda.py` supports conservative ORDER_FILL, DAILY_FINANCING and TRANSFER_FUNDS
profiles. Commission is positive charge magnitude in OANDA and becomes negative
cash; financing stays signed; guaranteed fees become negative cash. Missing
required fields and unknown transaction types quarantine the whole record.
AccountBalance is mandatory/validated for these profiles. Unknown monetary
types require parser review, not blind zero assumptions. Order-create/cancel/
reject and other neutral transaction profiles are pending, so this adapter is
not a complete account-history importer yet.

Conversion factors, half-spread cost and nested attribution are diagnostic;
broker home-currency P&L incorporates execution prices. Do not subtract spread
or slippage again. Separate conversion charges may be added only with broker
evidence proving a distinct cash effect and a reviewed division-specific parser.
The currently inspected public OANDA schema does not establish a universal
separate conversion-cost field. Commission/financing/gross P&L are distinct;
cash balance roll-forward must prove the chosen fee sign/embedding profile.

Reference: [OANDA Transaction definitions](https://developer.oanda.com/rest-live-v20/transaction-df/).
The implementation requires real exported samples and balance verification
before this profile can be certified for St Ludaetuc's account division.

Reconciler checks missing IDs/scope, duplicate identities, absent broker/opened/
prediction/outcome records, P&L missing/mismatch, false closure and pending EOD.
Unknown currency cannot be treated as GBP; legacy event scope must be evidenced
from a bounded single-account export or emitted as `scope_missing`. Never infer
account from instrument alone. Full trade rollup and automatic balance/NAV
roll-forward are designed but not implemented. Persist their failures as
`component_missing`, `balance_mismatch` or `unclassified_transaction` before
any coverage assertion. Exact numeric comparisons use account-currency decimals;
reviewed broker quantisation is needed for legacy float comparisons.

## 9. Deterministic P&L/break-even outputs

Implemented SQL: effective/verified expenses; lifetime observed spend;
vendor/category spend; weekly/monthly spend; effective subscription schedules;
`run_rate(as_of)`; scoped broker economics; weekly/monthly broker totals;
`period_summary(account,start,end)` with coverage gating. Python implements
same core decimal spend/run-rate/broker/unit-economics arithmetic and explicit
trade reconciliation. SQL persistence and Python are candidates pending parity
testing against PostgreSQL; do not claim that test has passed.

| Output | Deterministic formula / limit |
| --- | --- |
| lifetime actual spend | Sum effective verified incurred expense GBP, full inception window; estimates shown separately |
| current-period actual spend | Same sum, half-open `[start,end)` UTC window |
| recurring run rate | Verified active monthly schedules; annual = month×12; day = annual÷365.25; a forecast convention, not actual billing |
| gross realised trading P&L | Sum `realised_pnl` component, realised transactions in explicit account scope; broker gross of separately posted costs, not hypothetical mid-price profit |
| financing | Signed sum of financing components, daily and closure-specific transaction cash exactly once |
| other trading costs | Signed commissions + distinct evidenced conversion charges + other adjustments; separately itemise costs versus credits |
| net trading P&L | Gross realised + signed financing + signed other broker components; deposits/transfers excluded |
| net economic P&L | Eligible actual live net trading − verified project spending, only with complete coverage |
| lifetime break-even remaining | `max(spend − eligible net trading,0)`, inception window; Practice gains excluded |
| operating gap/profitability | Same net result for declared operating period; profitable iff net>0 with complete coverage |
| allocated cost/trade | Period spending ÷ unique fully closed reconciled trades; null when denominator zero/unknown |
| gross profit/trade | Broker gross realised P&L for the closed cohort ÷ eligible trade count |
| financing/trade | Closed-cohort attributable financing ÷ eligible trade count; unallocated financing disclosed |
| net profit/trade | (Closed-cohort net trading − allocated operating spend) ÷ eligible trade count |
| realised expectancy | Mean realised net result for complete closed cohort; R expectancy only with evidenced initial risk-cash denominator for every eligible trade |

Do not combine broker costs charged this week on open trades with closed-cohort
expectancy silently. Period cash economics and closed-cohort trade economics are
different products. `core.economics` is the aggregate reference; caller must
provide the correctly scoped closed cohort and completeness result. SQL
trade-cohort views are pending, not deployed or represented as active.

Partial coverage returns observed subtotals with coverage false and unavailable
economic/break-even conclusions. SQL excludes uncertain expense rows and refuses
complete conclusions when such rows exist in the period. Reviewed export and
expense coverage remain mandatory even if no exceptions exist. Empty fixtures
are not proof of zero spending or zero costs. Multi-account consolidation must
sum eligible broker accounts before subtracting project spend once; do not sum
per-account net-economic outputs, which would repeat project cost.

## 10. Finance Reader products

`contracts/reader_products.schema.json` defines six strict bounded products:
finance-summary, finance-costs, finance-trading-pnl, finance-break-even,
finance-reconciliation-health, finance-forecast-inputs. These are schemas/designs,
not deployed endpoints or additions to the existing Chief Reader allowlist.

Every output declares period/as_of, account alias, broker/system environment,
financial reality/state/simulation type, coverage, trade/exception/pending-EOD
counts, formula version and source manifest. Money is a decimal string or null,
with GBP currency, evidence ID, state and unavailable reason. Objects reject
additional properties; text/count/amount sizes are bounded. No raw DB queries,
credentials, private account IDs, invoice/payment details or raw broker JSON.
No arbitrary SQL, filter expressions, URL dereferences or user-picked table names.

Future interface policy: fixed product allowlist, scoped service identity,
approved account/window, max 366-day detail window, max 120 periods for historical
summary (separate inception aggregate), max 100 vendor/category groups, capped
response bytes and statement timeout, auditable request/outcome, no writes.
Strict serializers must enforce semantic rules in addition to JSON schema:
unknown/estimate/unavailable values cannot acquire verified labels, Practice
cannot acquire live economic conclusions, and incompatible dimension combinations
must fail. No role, grant, credential or issuer was created in this workstream.

## 11. Weekly CEO report contract

`contracts/weekly_report.schema.json` gives complete bounded machine inputs for
the future agent. The deterministic producer owns all numbers; the model may
explain them and quote unavailable inputs, not manufacture missing fields.

Lifetime: verified project spend, broker gross P&L, signed financing/other costs,
net economic result and remaining break-even. This week: infrastructure, API/AI,
subscriptions, R&D/testing, other operating cost, broker costs, realised P&L and
net result. Run rate: day/month/year under explicit annualisation convention.
Unit economics: denominator, allocated cost/trade, gross profit/trade,
financing/trade, net profit/trade and realised expectancy.

Forecast inputs: evidenced balance/NAV capital snapshot, realised closed-cohort
statistics and sample/window, operating schedule, financing/cost assumptions,
drawdown assumption, assumption-set ID and evidence. Missing inputs remain null
with reason. A drawdown assumption is an approved assumption, not a realised
fact. Its schema uses a separate `fraction_of_capital` unit (0–1), explicit
approved-assumption state and evidence. Cash expectancy is GBP/trade; R
expectancy needs a future dimensioned extension and must not be put into GBP
money fields. Unapproved drawdown assumptions remain unavailable. No forecasting engine or forecast financial figures were fabricated.

Weekly periods: Monday 00:00 UTC to next Monday 00:00 UTC, half-open; label the
timezone and convert presentation separately. Lifetime begins at CEO-attested
inception date. Persist report snapshot and source/coverage manifest. Later
corrections create a restatement report; they never rewrite an issued report.

## 12. Files/code created

All repository changes are confined to `finance/`:

* `__init__.py`, `core.py`, `oanda.py`, `market_data.py`, `README.md`.
* `migrations/001_foundation.sql`.
* `tests/test_core.py`, `tests/test_market_data.py`, `tests/postgres_acceptance.sql`.
* `templates/historical_expenses.csv`.
* `contracts/market_data_v2.example.json`, `market_data_v2.schema.json`.
* `contracts/reader_products.schema.json`, `weekly_report.schema.json`.
* `docs/HANDOVER.md`, `docs/REVIEW_RECORD.md`.

Scratch-only engine source used for inspection is not part of the commit. No
Chief candidate/recovery-controller files were fetched for implementation or
modified. Shared service inventory was read only; Chief services and validation/
audit DBs were not queried or changed. PR 3/4/5 were not edited or merged.

## 13. Branch/PR/commit details

Repository `StLudaetuc/st-ludaetuc-engine`; only writable branch
`finance/foundation-v1`; base `2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c`.
Exact published commit and optional single draft PR are recorded in
`docs/REVIEW_RECORD.md` and the delivery message after publication. Main remains
read-only, nothing merged or deployed, no changes outside Finance branch.

## 14. Tests and results

67 pure Python unittest cases passed locally on 13 October 2026: decimal
arithmetic, FX/currency/time validation, duplicate protection/idempotent retries,
frozen history and corrections, estimate/manual verification, refunds, atomic
CSV rejection, half-open windows, run rates/overlaps/expiry, financing/commission,
broker duplicates/scope/missing money, deposits excluded, break-even and
profitability, zero denominators/incomplete data, Practice/SHADOW/counterfactual
exclusion, EOD pending/overdue, missing/mismatched P&L/prediction, adapter fee
signs/unknown fields/types and nested financing/P&L no-double-counting; missing
currency/state and currency mismatch; raw-v2 validation, literal JSON failures,
placeholder/non-finite rejection, aliases, OHLC ordering and bounded fields.

PostgreSQL acceptance script is supplied but **NOT EXECUTED**: no PostgreSQL
runtime/client exists here; the existing private trading database cannot be
queried via the connector because its allowlist is empty. The allowlist was not
changed. Thus SQL syntax/runtime, trigger enforcement, role denial, concurrent
inserts and SQL/Python parity are not claimed tested. No cloud test database
was created and no infrastructure costs incurred. The draft remains a review
candidate, not migration-ready or approved for live ingestion.

## 15. Remaining blockers

* Complete inception spending and independent verification/coverage attestation.
* Make Closed Trade Data Sync blueprint, schedule/timezone, closure IDs/mappings,
  pagination, retry/partial-close and financing behaviour evidence.
* OANDA scoped decimal transaction export, actual account currency/division,
  opening/closing balance/NAV and full-history completeness; no credentials needed.
* Fresh isolated Finance PostgreSQL 18 test environment for acceptance/concurrency/
  role tests and deterministic SQL/Python parity; explicit separate authority.
* Complete broker type classifier, balance roll-forward, trade-cohort rollups,
  provenance-aware persistence importer and exact legacy float comparison policy.
* Bounded serializer/API implementation, semantic validation, scoped grants and
  independent security review before any Finance Reader access.
* Reviewed forecast/drawdown assumptions, R-expectancy extension and canonical category mapping.
* Alert inventory and separately authorised logger replacement/recreation;
  existing ingestion source cannot identify failing symbol/timeframe.
* Controlled-document approval wording drift, to owning workstream; not changed.

## 16. Exact next governance/implementation decision

Review this single Finance draft for scope and design acceptance. If accepted,
authorise a **separate isolated Finance validation stage only**: create/use a
fresh non-production Finance PostgreSQL 18 database; run migration, acceptance,
concurrency, privilege and parity tests; ingest a redacted EOD OANDA export and
Make blueprint plus CEO expense CSV into staging; prove balance reconciliation
and completeness. Keep Chief Stage 1 inactive and untouched, no production
migration, no deployment, no live broker credential and no live trading action.

Merge, deployment, changes to Make/TradingView, Finance Reader allowlist/grants
and any PG-P# progression are separate later decisions. No such permission is
implied by reviewing this foundation. Source is ready for review with explicit
unexecuted SQL/integration checks; operational financial truth is not yet ready.

FINANCE_FOUNDATION_READY_FOR_REVIEW
