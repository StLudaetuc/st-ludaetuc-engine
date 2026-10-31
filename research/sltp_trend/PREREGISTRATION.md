# Trend with stop loss and take profit, sized for a £10k account — pre-registration
Written 30 Oct 2026 before any SL/TP backtest. Research only.

**Why:** Andy will trade live with at most £10,000 and wants every position to have a stop loss and a realistic
take profit. The earlier trend results had no stops, so this is a new strategy and must be tested as one.

**Markets:** gold, silver, copper, WTI crude, US 10-year T-note (OANDA daily mid OHLC, 2007–2026, **weekday rows only**,
09:00 New York alignment).

**Rules (fixed now)**
- Every Monday, using the previous weekday close: score = sign(c0−c63) + sign(c0−c126) + sign(c0−c252) (weekday bars).
  Direction = sign(score) (score is never 0).
- Flat → enter at Monday's open in that direction. Holding and the direction flips → exit at Monday's open and reverse.
  Holding and the direction is unchanged → keep the position and its original stop/target.
- Stop = entry ∓ ks × ATR(20); take profit = entry ± kt × ATR(20) (ATR from the previous day's weekday bars).
  Checked every day from the entry day using the day's high/low; if both could be hit in one day, the **stop** is assumed.
  A gap through the stop or target fills at the open. After an exit the market stays flat until the next Monday.
- Size: risk **1% of equity** per trade (equity compounds). Results in R (1R = stop distance).
- Costs: IG futures-style spreads measured on 30 Oct 2026 (gold 0.020%, silver 0.20%, copper 0.061%, crude 0.058%,
  T-note 0.037% of price), one full spread per round trip, plus one spread per contract roll while held
  (6, 4, 4, 12, 4 rolls a year). No overnight funding (futures-style). Stress test: costs ×2.

**Variants (all reported):** (ks, kt) = (2, 3), (2, 4), (3, 4.5), (3, 6), (3, no target).
**Periods:** development 2007–2018, validation 2019–2022, holdout 2023–2026.

**PASS (per variant, at 1× costs):** profit factor ≥ 1.15 and average R > 0 in **each** period, and profit factor ≥ 1.05
at 2× costs over the full period. Because five variants are tested, a single pass is weak evidence; the variant chosen for
the demo must be the one with the best **development** result among those that pass, decided before reading the others' holdout.
