# Update prompts for The Chief and The Oracle — 31 October 2026

Written by Claude Code. Everything below is **REPORTED by Claude Code** unless The Chief or The Oracle verifies it
independently (repo: `StLudaetuc/st-ludaetuc-engine`, branch `research/gbpusd-15m-liquidity-strategy`, draft PR #16,
research only, not merged). Paste Prompt 1 into The Chief's chat and Prompt 2 into The Oracle's chat.

---

## PROMPT 1 — THE CHIEF

CHIEF UPDATE — 31 Oct 2026 — research results since 27 Oct, IG demo trial, and why we are moving the research venue from OANDA to IG

From Andy (CEO). The report comes from Claude Code and covers 27–31 Oct. Under your standing rules, record in the
Snapshot only what is confirmed. Label everything here REPORTED unless you verify it yourself (GitHub branch
`research/gbpusd-15m-liquidity-strategy`, commits listed below). Flag to me any conflict with the approved specification
or a recorded decision.

### A. What has NOT changed (production)
- Claude Code made no Production, Render, database, live Pine or OANDA engine changes. It placed no trades on any live
  account and handled no credentials. The Render engine (v4.2.0 at your last update) and the OANDA Practice set-up were
  not touched. Claude Code did not re-verify their state, so keep your existing labels.
- Nothing is approved for live money. No SHADOW trial has started.

### B. Decision: research and demo venue moves from OANDA to IG (CEO decision, 29–30 Oct 2026)
This is a venue change for research and demo trading, not a live deployment. Why:
1. **Holding costs.** OANDA charges a daily financing markup on top of the interest differential, about 1%/yr per side
   on FX and about 2.5%/yr on CFDs and metals (inferred from OANDA's live long/short rates). Any strategy that holds for
   days or weeks pays that every day, and in our backtests it destroyed most of the edge. IG's futures-style
   (forward-dated) markets have no daily funding charge; funding is in the price, and you pay a spread when the contract
   rolls.
   - Same pre-registered trend rules on gold, silver, copper, WTI and the US 10-year, 2007–2026, Sharpe:
     | Cost model | Sharpe |
     |---|---|
     | OANDA costs | 0.31 |
     | IG measured demo costs | 0.50 |
     | No costs | 0.58 |
   - Caveat: the five markets were chosen after earlier diagnostics, so this is partly in-sample. Source:
     `research/sim_trial/IG_DEMO_CHECK_20261030.md`, `research/daily_portfolio/ig_cost_check.py`.
2. **Breadth.** IG offers futures-style metals, energy, grains, bonds and indices, plus FX, under one account. OANDA's
   non-FX range is narrower and dearer to hold.
3. **Spread betting (£ per point).** IG's futures-style CFD contracts are far too large for £10k: gold ≈ $130k per
   contract (minimum 0.5), the others $21k–$52k (minimum 1). Spread bets size in £ per point, so £100 risk with a
   3×ATR stop is about £0.47/pt on gold, £0.19 silver, £0.03 copper, £0.09 crude and £0.49 T-note. Spread bets also
   settle P&L in GBP. Tax treatment for a Guernsey resident is not assumed; Andy will confirm with an adviser.
4. **API.** IG has a full REST API, comparable to OANDA's: login, market data, positions, orders, deal confirmations and
   client sentiment. Claude Code tested it on the IG demo through Make.
5. **Alternatives looked at and set aside for now:** Interactive Brokers, tastytrade UK, NinjaTrader (UK launch
   27 Oct 2026, not FCA-regulated), Capital.com (≈4% overnight fee, which makes holding strategies worse).
6. **Live capital will be £10,000 maximum** (Andy). The demo is kept realistic at £10k; no virtual top-ups.

### C. IG demo status (demo only)
- **Account.** IG CFD demo Z5QIAR, GBP, £10,000. A spread-bet demo is not yet available. Andy is chasing IG; the live
  spread-bet application (RTQOK) is pending documents.
- **Make scenarios** (all demo-only, hard-coded to demo-api.ig.com; credentials stored in Make data store 190220, never
  in chat):
  | Make item | ID | Status |
  |---|---|---|
  | Execute Signal | 10075127 | active, webhook 4478291 |
  | Send Test Signal | 10075131 | on demand |
  | Weekly Trend Signals | 10075136 | DEACTIVATED |
  | Close All 18 Nov | 10075347 | DEACTIVATED |
  | Close All Mon 2 Nov 08:30 UTC | 10075362 | ACTIVE, one-off |
  | Trade log data store `stl_ig_demo_trades` | 203924 | in use |
  - The scheduled close shuts the minimum-size test positions, which are oversized for £10k: gold short 0.5, copper long
    1, crude long 1, T-note short 1. The silver order was rejected for insufficient funds.
- **Andy's decisions:**
  - Stops are mandatory.
  - Chosen rule for the demo: weekly trend signal, 3×ATR stop, no take profit, 1% risk per trade.
  - Spread bet preferred over CFD.
  - Up to 10+ markets once running.
- **The weekly pipeline code** (`research/sim_trial/pipeline/`) is NOT deployed. The Oracle's review of it was CHANGES
  REQUIRED: H1–H11 and M1–M7 are open.
- **New, 31 Oct:** IG client sentiment recorder. Make scenario 10075822 runs every 12 hours and writes to data store
  `stl_ig_client_sentiment` (203972), covering 29 markets. It collects data only and never trades. The first snapshot
  (03:04 UTC) was complete. See `research/positioning/IG_SENTIMENT_RECORDER.md`.

### D. Research results 27–31 Oct (all pre-registered before testing; every one FAILED)
| Line | Result | Commit |
|---|---|---|
| Short-timeframe 10s/30s/1m | FAIL; gross edge ≈ 0 | e201eb9 |
| Daily trend + carry, 123 OANDA markets | FAIL; OANDA financing markup diagnosed | b66d2db |
| Flow edges (intraday momentum, turn of month, month-end FX) | FAIL | d138393 |
| Old TradingView gold BB+RSI | High profit factors were a window effect | 0688115 |
| Trend with SL/TP, 5 markets, 1% risk | All 5 variants FAIL. Best is 3×ATR stop, no target: PF 1.40 / 1.66 / 1.14, ≈3.5%/yr, max DD ≈25%. Take-profits hurt. | 3ac31be |
| Same rule on 45 spread-bet-style markets | FAIL. PF 1.16 / 1.14 / **0.66** in 2023–26. Picking markets by past results did not work. | 8617fa7 |
| 7 novel ideas (panic reversal, pairs, lead-lag, overnight drift, vol-managed holding, gold session drift, commodity spreads) | All FAIL. Overnight index drift is real before costs (+3.2bp/night, t 2.7) but −2bp after the spread. | d932f14 |
| CFTC COT hedging pressure, 22 markets 2007–26 | FAIL: −1.1bp/week, t −0.7 | 93d4d91 |
| OANDA retail position book contrarian, 16 markets 2017–26 | FAIL: −9.5bp/week. Loss is almost all from shorting gold/silver in a bull market; FX alone is flat. | 57003e2 |

Implication to record: **the rule chosen for the demo (3×ATR stop, no take profit) has no validated edge.** It came
closest of the variants tested but failed its own pre-registered pass test, and it lost money in 2023–26 across 45
markets. The demo is for testing plumbing, costs and operations, not for proving profit. Claude Code's recommendation:
no live trading on this evidence.

### E. Requested Snapshot actions
1. Record the CEO decision in B: IG is the research/demo venue. OANDA Practice and the engine are unchanged. No
   live-money change.
2. Add the IG demo items in C to the system inventory, each with its status.
3. Add or keep these open actions:
   - Oracle H1–H11 / M1–M7 on the IG pipeline.
   - Spread-bet demo and live account set-up.
   - Contract-roll handling before DEC-26 expiry (~19–25 Nov).
   - Export of the sentiment data store before the 1 MB cap (~2 years).
   - Independent review of the 27–31 Oct research (Oracle).
4. Mark as superseded any statement that a validated strategy exists for the IG trial.

---

## PROMPT 2 — THE ORACLE

You are The Oracle, St Ludaetuc's independent reviewer. Review the following adversarially. Claude Code wrote it, and the
author does not approve its own work. Report findings only: do not change any code, Make scenario, broker account or
deployment, and do not trade.

### Where
Repository `StLudaetuc/st-ludaetuc-engine`, branch `research/gbpusd-15m-liquidity-strategy` (draft PR #16, research
only, must not be merged). Folders:
- `research/sim_trial/`: IG demo plan, IG API check, Make set-up, pipeline (your earlier CHANGES REQUIRED review)
- `research/daily_portfolio/`: cost and venue diagnostics (`broker_markup.py`, `ig_cost_check.py`, `calendar_check.py`)
- `research/sltp_trend/`, `research/sltp_universe/`: trend with stop/target; 45-market scan
- `research/novel_ideas/`: ideas A–G
- `research/positioning/`: COT and OANDA position-book tests, IG sentiment recorder

### Context since your last review
- **Live capital will be £10,000 maximum.** Andy closed the oversized demo positions (scheduled for Mon 2 Nov 08:30
  UTC) rather than adding virtual funds.
- **Venue decision (CEO):** research and demo move from OANDA to IG. Reasons:
  - OANDA's daily financing markup (≈1%/yr FX, ≈2.5%/yr CFDs and metals) versus IG futures-style markets with no daily
    funding.
  - Breadth of markets.
  - Spread-bet £/pt sizing; CFD contracts are too large for £10k.
  - A full REST API.
  - Alternatives set aside: IBKR, tastytrade UK, NinjaTrader, Capital.com.
  - Claimed effect on the same trend rules (5 markets): Sharpe 0.31 with OANDA costs, 0.50 with IG costs.
- **Rule chosen for the demo:** 3×ATR stop, no take profit, 1% risk. Claude Code reports that this rule failed its
  pre-registered test, and that the same rule on 45 markets lost in 2023–26.
- **Positioning work, 31 Oct:**
  - CFTC COT hedging pressure: pre-registered PRIMARY rule FAILED.
  - OANDA position-book contrarian: pre-registered PRIMARY rule FAILED.
  - IG client sentiment: collection only, started 31 Oct; no test for about 12 months.

### Please answer with evidence (file and line, or a reproduced number)
1. **Venue rationale.**
   - Is the OANDA-vs-IG cost comparison fair: financing markup inference, IG roll cost assumption, one snapshot of demo
     spreads, the double-markup fix in `ig_cost_check.py`?
   - Is the 0.31 → 0.50 Sharpe difference robust, or mostly an artefact of in-sample market selection?
   - Is moving venue justified when no strategy has passed?
2. **Spread bet vs CFD.** Are the reasons sound (sizing granularity, GBP P&L, futures-style pricing)? Is there any cost or
   risk we have missed, e.g. spread-bet markups versus CFD, guaranteed-stop costs, or IG's margin and close-out rules
   at £10k?
3. **Research integrity, 27–31 Oct.**
   - Were the pre-registrations genuinely written before the data was looked at? Check the commit order: each
     PREREGISTRATION commit comes before its results commit.
   - Was any rule changed after seeing results?
   - Are the PASS bars appropriate (t ≥ 2.5, profit factor at 1.5× and 3× costs, positive in each third)?
4. **Positioning code** (`research/positioning/positioning_tests.py`). Look for look-ahead and sign errors:
   - COT: a Tuesday report used from the following Monday; JPY/CAD/CHF sign flips; percentile over the prior 156 weeks.
   - Position book: Friday 20:00 UTC snapshot used from the Monday.
   - Costs: half spread × multiplier per position change, plus rolls.
   - Two bugs were fixed before the reported results: a pandas index realignment that turned the continuous variant
     into NaN, and 0/0 GBP_CHF snapshots treated as missing. Do you agree both fixes are neutral to the verdict?
5. **Overfitting risk in what's left.**
   - Several post-hoc patterns have been noticed: metals and US indices held up in 2023–26, Brent/WTI and corn/wheat
     pairs, and overnight index drift before costs. Are any of these worth a fresh pre-registered test?
   - Or is the honest conclusion that no retail-cost edge has been found?
6. **Demo safety.**
   - Make scenario 10075127 (Execute Signal) and scenario 10075362 (scheduled close): can either reach a live account,
     double-trade, or leave positions open without stops?
   - Is the sentiment recorder (10075822) strictly read-only?
7. **Open items.** Which of your H1–H11 / M1–M7 must be fixed before any automated weekly demo run? What is the minimum
   evidence you would need before recommending a SHADOW trial?

End with an overall verdict: APPROVE / CHANGES REQUIRED / REJECT, a severity-ranked list of findings, and anything you
could not verify.
