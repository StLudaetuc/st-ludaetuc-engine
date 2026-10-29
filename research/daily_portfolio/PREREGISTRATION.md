# Daily-bar trend-following and carry portfolio: pre-registration

Written 2026-10-29 ~00:20Z, before any daily-bar backtest was run. Research only; no live changes.

**Request (Andy, 2026-10-28):** "explore the daily-bar trend-following and carry portfolio across OANDA markets ... find the highest profit factor you can find."

## Data
- **Prices:** OANDA daily (D) bid/ask candles for all 123 instruments on the practice account, 2007-01 to 2026-10, aligned to 17:00 New York. Fetched read-only through a temporary Make scenario.
- **Interest rates:** OECD monthly 3-month interbank rates (`IR3TIB`), 2006-01 onwards, for USA, GBR, JPN, AUS, CAN, CHE, NZL, NOR, SWE, EA20, DEU, MEX, ZAF, POL, HUN, CZE and DNK. Public data, fetched without credentials.
  - Each month's rate is used from the first trading day of the following month, as a publication lag.
  - EA20 is used for EUR. Currencies with no OECD series (HKD, SGD, CNH, THB, TRY) are left out of every test that needs a rate.

## Costs (always on)
- **Spread:** every position change pays half the bid/ask spread on the units traded, taken from that day's open quote.
- **Slippage:** an extra 10% of the spread.
- **FX financing:** long base / short quote pays or earns (r_base − r_quote) − 1.0% a year; short pays or earns (r_quote − r_base) − 1.0%.
  - The 1.0% markup per side is inferred from OANDA's current long and short rates, which sum to about −2%.
- **CFDs and metals:** long pays r_quote + 2.5%; short earns r_quote − 2.5%. The 2.5% markup is inferred the same way.
  - Commodity CFDs (oil, gas, grains, sugar) are also included, but reported both with and without them, because OANDA's historical treatment of their rolls is unknown.
- **Stress:** every result is repeated with costs doubled.

## Systems (fixed now, no parameter search)
1. **TREND (time-series momentum).**
   - Signal = average of sign(return over 63, 126 and 252 trading days), in [−1, +1].
   - Position size targets equal risk: 10% annual volatility per instrument from a 60-day EWMA, then the whole portfolio is scaled by 1/√N (N = instruments with data that day).
   - Rebalanced weekly (Mondays) at the open, using the previous close's signal.
   - A position only changes when the target moves by more than 25% of itself, which limits turnover.
2. **CARRY (FX only, pairs where both currencies have OECD rates).**
   - Signal = sign(r_base − r_quote), used only if the gap exceeds 0.5 percentage points.
   - Same volatility targeting, weekly rebalance and buffer as TREND. The carry return is the financing actually received or paid.
3. **COMBINED:** 50% TREND + 50% CARRY, each at the same risk budget.

## Universe groups (each reported)
- FX (all pairs with rate data)
- Metals
- Equity indices
- Bonds
- Commodities
- ALL

## Periods
- **Development:** 2007–2018
- **Validation:** 2019–2022
- **Holdout:** 2023–2026-10

Nothing is tuned, so all three are reported. The holdout is still read as the final check.

## Metrics
- **Profit factor** on daily portfolio P&L, and on monthly P&L.
- **Trade profit factor:** a "trade" is the span between sign changes of an instrument's position.
- Annual return, volatility and Sharpe ratio, maximum drawdown, and returns by year.

## Pass criteria for "promising, worth a SHADOW trial"
1. Monthly PF ≥ 1.2 in the holdout at 1× costs.
2. Sharpe ratio ≥ 0.4 over the full period at 2× costs.
3. Sharpe ratio > 0 in each of development, validation and holdout at 1× costs.

## After this
Any variant explored after these first results (different lookbacks, universe or sizing) will be labelled EXPLORATORY. It is selected on development data only and checked on validation; the holdout is not re-used for selection.
