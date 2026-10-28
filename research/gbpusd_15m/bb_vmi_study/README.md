# Bollinger %B grid + St Ludaetuc Value Migration Index: GBP/USD 15m (research, negative result)

**Question (Andy, 2026-10-28):** can a 15-minute GBP/USD strategy reach profit factor ≥ 1 after costs? The test uses Bollinger %B zones 34–40 / 77–79 (21 combinations) plus a new custom "St Ludaetuc" indicator.

**Answer: not with these components (VERIFIED).** No combination reached PF ≥ 1 on development or validation data. The holdout (final 20%) was **not** touched, because no candidate passed development.

## Protocol
- **Data:** `../data/GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz` (SHA-256 in `../DATA_SHA256.txt`).
- **Splits:** chronological, the same as `../backtest.py`. Development is the first 60%; validation is the next 20%.
- **Signals and fills:** signals use mid prices (%B on a 20-period, 2-standard-deviation band). Fills use real OANDA bid/ask plus 0.2 pip slippage. When a stop and target fall in one bar, the stop is assumed first. Positions are flat by 20:45 London.
- **System (`bbsys.py`):**
  - Arm when %B reaches a zone.
  - Enter on a close beyond the previous 2 bars' high or low, at the next bar's open.
  - Stop beyond the setup extreme, plus 0.25 ATR (minimum 0.75 ATR). Target 1.5R.

## The St Ludaetuc indicator: Value Migration Index (`vmi.py`)
- **Formula:** VMI = (London-session tick-volume VWAP − Asian-range midpoint) ÷ (half the Asian range).
- **Meaning:** |VMI| > 1 means London has accepted value outside the overnight range.
- **Used two ways:** with the migration (buy %B-low pullbacks in up-migrations) and against it (fade %B extremes after migration).

## Results (VERIFIED, net of costs)

| System | Development PF (21 settings) | Validation PF (21 settings) |
|---|---|---|
| %B reversion | 0.83–0.84 | 0.83–0.86 |
| %B breakout (zones reversed) | 0.85–0.88 | 0.94–0.97 |
| VMI with-flow pullback | 0.76–0.80 | not run (failed development) |
| VMI fade | 0.86 (37/78) | 0.85–0.89 |

**Configurations with PF ≥ 1: 0 of 63 on validation.**

## Why
- **Before costs, the setups are a coin flip.** Gross PF is 0.98 for %B reversion and 1.02 for the VMI fade.
- **Costs decide the result.** The median spread alone is 0.16 ATR on 15-minute bars, about 0.09R per trade at these stops.
- **The band levels barely matter.** Moving the low zone from 34 to 40, or the high from 77 to 79, changes PF by about 0.01.
- **No time-of-day pattern survives.** In a scan of 18 London time slots × session-move conditions (`scan.py`), none reached |t| ≥ 2. The largest was 1.4, which is what noise produces across 18 tests.
- **Context filters didn't help.** Seasonality-adjusted range, tick-volume ratios, distance from the daily open and ATR percentile made no reliable difference to outcomes (`why.py`). The one standout was %B reversion failing at 12:00–13:59 London, around the US data releases.

## Files
- `bbsys.py`, `vmi.py`: the system and the indicator.
- `grid.py`, `vmi_test.py`, `vmi_test2.py`, `why.py`, `scan.py`, `val.py`: the experiments, each reproducible.
- `grid_dev_baseline.csv`, `grid_validation.csv`: the grid results.
