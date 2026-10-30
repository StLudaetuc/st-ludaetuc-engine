# Weekly simulation pipeline — IG DEMO only

Implements `../PLAN.md`: every Monday it computes the pre-registered trend targets and rebalances five IG
futures-style markets on the **demo** account. Not deployed; nothing runs until Andy and The Oracle approve.

## Flow
1. **Signals** (`signals.py`) — OANDA practice daily mid closes (09:00 New York alignment, complete candles only,
   exactly the data the backtest used) → trend weights. A test proves the live code reproduces the backtest's weekly
   position path exactly over 2007–2026 (`test_parity_with_backtest_positions`).
2. **Contracts** (`run_weekly.py`, `rebalance.choose_expiry`) — for each market, find IG's futures-style contracts by
   exact name, read their specs live (value per point, minimum size, last rollover time) and pick the earliest expiry
   that is more than 5 days from IG's last rollover time.
3. **Orders** (`rebalance.plan_orders`) — target contracts = weight × equity ÷ contract notional, rounded to 0.01;
   below IG's minimum size → flat. Trade only outside the 25% buffer. Positions on an old expiry are closed and
   re-opened on the new one (roll). Orders are marketable limits, fill-or-kill, at most 2 spreads from the touch.
4. **Log** — one JSON record per run (signals, equity, contracts and spreads, orders, fills, notes) printed to stdout
   and optionally POSTed to `SIM_LOG_WEBHOOK` (e.g. a Make data store).

## Safety guards
- Client refuses any IG base URL except `https://demo-api.ig.com/gateway/deal`, and stops if the logged-in account
  is not the configured demo account.
- Dry run by default; orders are sent only with `SIM_EXECUTE=1`.
- Refuses to trade if IG's price is >5% from the OANDA price (wrong contract or scaling) or if total notional
  would exceed 3× equity.
- Circuit breaker: equity below 85% of `START_EQUITY_GBP` → flatten everything (every run until reset).
- Credentials only from environment variables; never written to the log.

## Run
```
pip install requests pandas numpy
export IG_API_KEY=... IG_USERNAME=... IG_PASSWORD=... IG_ACCOUNT_ID=Z5QIAR OANDA_TOKEN=... START_EQUITY_GBP=...
FORCE_RUN=1 python3 research/sim_trial/pipeline/run_weekly.py     # dry run, prints the plan
python3 -m pytest research/sim_trial/pipeline -q                  # 11 tests
```

## Open decision before the first run: account size (VERIFIED from IG contract specs, 30 Oct 2026)
IG's futures-style **CFD** contracts are large (1 contract ≈ $130k gold, $49k silver, $32k copper, $52k crude,
$21k T-note; minimum 0.5–1 contract). With the current signals:
| Demo equity | Markets that can hold a position |
|---|---|
| £10,000 | none |
| £100,000 | copper, T-note |
| £250,000 | gold, copper, T-note |
| £500,000 | all five |
Options: (a) raise the CFD demo's virtual funds to about £500k (tests costs and plumbing at a size Andy would not trade
live); or (b) add a **spread-bet** demo account, where stakes are £ per point with small minimums, so a realistic
£10k–£50k account can hold all five. The code reads contract specs at run time, so either works; (b) needs the
instrument names in `config.py` updated to IG's spread-bet market names.

## Deployment (not done; needs Andy's approval after review)
Recommended: a Render **cron job** running `python3 research/sim_trial/pipeline/run_weekly.py` on Mondays 14:00 UK,
from this branch, with the variables above set in Render (separate from the production engine service).
