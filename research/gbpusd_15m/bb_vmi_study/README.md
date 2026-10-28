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

## Follow-up: building on the breakout base to try to reach PF ≥ 1 (2026-10-28)
Andy asked to build on the closest system, the %B breakout. Its validation PF was 0.94–0.97, but its development PF was 0.85–0.88.

**Protocol:**
- %B fixed at 36/78, the middle of the grid, so no lucky setting could be picked.
- Eight ideas, each with a market reason, declared before testing.
- Develop on development data only. Validation once. Holdout once.

**Development results (`exp.py`).** Base: avgR −0.063, PF 0.878.

| Idea | Result |
|---|---|
| A: wider stop, 2R | PF 0.892 |
| **B: hold to session end, no target** | **PF 0.960** (the only change beyond noise) |
| C: 2-ATR trailing stop | PF 0.762 |
| D: 4H trend only | PF 0.841 |
| E: squeeze only | PF 0.827 |
| F: core liquid hours | PF 0.898 |
| G: skip 12–14 London | PF 0.886 |
| H: daily trend only | PF 0.887 |

**Combinations (`exp2.py`):** the best was B+F+H, with development PF 0.990 (avgR −0.006 ± 0.046). It did **not** clear 1 in development, but as the only near-miss candidate it went to validation and holdout once each.

**B+F+H out of sample:**

| Period | Trades | PF at 1.0× costs | PF at 1.5× | PF at 2.0× |
|---|---:|---:|---:|---:|
| Validation (`val2.py`) | 430 | 1.058 (avgR +0.032 ± 0.076) | 0.987 | 0.922 |
| **Holdout (`hold2.py`)** | 447 | **0.852** (avgR −0.089 ± 0.073) | 0.787 | 0.729 |

- **Full 5 years:** PF 0.976, 2,262 trades. Average R by year: 2021 +0.08, 2022 +0.06, 2023 0.00, 2024 −0.11, 2025 +0.03, 2026 −0.07. The trades are in `bfh_trades_1x.csv`.

**Verdict (VERIFIED):** no edge. The validation PF of 1.06 was within noise, and the holdout failed. Any system tuned further on this family would be fitting noise.
