# Very short timeframes (10s, 30s, 1m): pre-registration

Written 2026-10-29 ~00:35Z, before any short-timeframe backtest was run. Research only.

**Request (Andy):** "explore very short timeframes such as 10 second, 30 second and 1 minute ... if we scope out a pattern early enough, we can predict movement and enter to make a profit."

## Data (OANDA bid/ask, read-only via temporary Make scenarios)
- **S5 (5-second) candles:** EUR/USD, GBP/USD and XAU/USD, 2026-10-12 to 2026-10-28. 10s and 30s bars are built from these. The 5s resolution lets the backtest model execution delay.
- **M1 candles:**
  - EUR/USD and XAU/USD: one year, 2025-10-27 to 2026-10-28.
  - GBP/USD: two years (already held).

## Execution model (always on)
- **Delay:** the TradingView → Make → Manus engine → OANDA chain takes seconds. Every entry and exit is filled **5 seconds after the signal bar closes**, at the bid/ask of the 5s bar then open.
  - On M1-only data (no 5s bars), fills are at the next 1-minute bar's open bid/ask. That understates the real delay, so it favours the strategy.
- **Prices:** all fills on real bid/ask; no mid-price fills.
- **Stops and targets:** if both are hit inside one bar, the stop is assumed first.
- **Stress:** every strategy is also run with an extra 1× spread per round trip.

## Strategies (fixed now; each run at 10s, 30s and 1m)
| # | Name | Rule |
|---|---|---|
| S1 | Burst momentum | The last 6 bars moved more than 3× the current spread in one direction and tick volume over those 6 bars is above its trailing 90th percentile → enter in that direction. Exit after 6 bars. |
| S2 | Burst fade | Same trigger as S1 → enter against the move. Exit after 6 bars. |
| S3 | Session opening-range breakout | Range of the first 5 minutes after 07:00 London and 09:30 New York → enter on the first close outside the range. Stop at the opposite side; exit after 30 minutes. One trade per session. |
| S4 | US-data spike continuation | At 08:30 New York on weekdays, take the direction of the first 30 seconds' move (if larger than 2× spread) → enter, hold 5 minutes. |
| S5 | Round-number fade | First touch of a round level (EUR/USD and GBP/USD: x.xx00 and x.xx50; gold: whole $10) in a bar → fade it. Target and stop both 1.5× the average 1-minute range. Max hold 15 minutes. |
| S6 | %B extreme reversal | %B(20, 2) below 0 or above 100, then a bar closing back inside → enter the reversal. Stop beyond the extreme; target the middle band. |

No parameter is tuned. Parameter sensitivity may be shown later, clearly labelled EXPLORATORY.

## Splits and metrics
- **M1 year data:** chronological 60% development / 20% validation / 20% holdout.
- **S5 data:** only about 12 trading days, so it is reported as a single block and is mainly used to measure the cost of the 5-second delay and the spread.
- **Metrics:** profit factor, trades, average P&L per trade in spread units and in R (where a stop exists), win rate, and t-statistic.

## Pass criteria ("worth a SHADOW trial")
1. PF ≥ 1.2 at 1× costs in both validation and holdout.
2. PF ≥ 1.0 with the extra spread.
3. At least 100 holdout trades.
4. Cleared on at least 2 of the 3 instruments.
