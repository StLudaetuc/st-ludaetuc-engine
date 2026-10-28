# GBP/USD 15m liquidity-sweep strategy: research results

**Status: RESEARCH ONLY. Not for live use. Not reviewed. Do not deploy.**
Independent review by The Oracle is required. Nothing goes live without Andy's approval and a SHADOW trial first.

Labels: **VERIFIED** means reproduced by `backtest.py` from the committed files. **REPORTED** means taken from another document. **UNPROVED** means not established.

## Data (VERIFIED)
- **File:** `data/GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz`, SHA-256 `b6388884808783c5f750fb7ff5d2fe13c6d46047421fa08934fb7fb744d82b7c`. Details in `data/MANIFEST.json`.
- **Source:** OANDA v20 practice API, GBP_USD, M15, bid and ask OHLC, complete candles only.
  - It was downloaded read-only on 2026-10-28 through temporary Make scenarios, which were deleted after use, as authorized by Andy.
  - This is a deviation from the original brief's "no OANDA/Make" rule. Andy approved it explicitly because no 15-minute data existed anywhere else.
- **Coverage:** 122,134 bars, from 2021-10-25 00:00 to 2026-10-28 15:15 UTC.
  - Timestamps are bar opens, in UTC.
  - Gaps: weekends, Christmas and New Year, and one 2h45m outage (2022-05-12). Smaller 30–45 minute gaps sit around the daily rollover.
- **Prices:** signals use mid = (bid+ask)/2. Fills use the stored bid and ask.

## Method (VERIFIED)
- **Splits, chronological by bar:**
  - Development: 2021-10-25 00:00 to 2024-11-08 18:15
  - Validation: to 2025-11-04 05:30
  - Holdout: to 2026-10-28 15:15
- **One continuous run:** the strategy runs once over the whole series, and each trade is assigned to the split its entry falls in.
- **Costs:** the real OANDA bid/ask spread on every fill (median 1.2 pips in London and New York hours), plus 0.2 pip slippage per market fill. Stress tests multiply the half-spread and slippage by 1.5 and 2.0. Targets are limit orders and get no slippage.
- **Conservative fills:** when a stop and target fall in the same bar, the stop is assumed first. Gaps through the stop fill at the open.
- **No repainting:** signals use confirmed bars only, and entries are at the next bar's open. Higher-timeframe structure is built in-script from completed 1-hour and 4-hour buckets, with no `request.security`. A swing is only known two higher-timeframe bars after it forms.

## Results after costs
| Split / case | Trades | Win rate | Avg R | Profit factor | Max DD (R) | Longest losing streak |
|---|---:|---:|---:|---:|---:|---:|
| development @ 1.0x | 20 | 25.0% | -0.27 | 0.62 | 7.6 | 5 |
| development @ 1.5x | 20 | 25.0% | -0.29 | 0.61 | 8.0 | 5 |
| development @ 2.0x | 20 | 25.0% | -0.31 | 0.59 | 8.3 | 5 |
| development @ limit_mid_1.0x_SENSITIVITY | 25 | 28.0% | -0.19 | 0.74 | 7.1 | 4 |
| validation @ 1.0x | 9 | 33.3% | -0.28 | 0.59 | 3.4 | 2 |
| validation @ 1.5x | 9 | 33.3% | -0.30 | 0.57 | 3.5 | 2 |
| validation @ 2.0x | 9 | 33.3% | -0.32 | 0.55 | 3.7 | 2 |
| validation @ limit_mid_1.0x_SENSITIVITY | 9 | 33.3% | -0.08 | 0.88 | 3.1 | 3 |
| holdout @ 1.0x | 5 | 20.0% | -0.41 | 0.5 | 4.0 | 4 |
| holdout @ 1.5x | 5 | 20.0% | -0.42 | 0.49 | 4.1 | 4 |
| holdout @ 2.0x | 5 | 20.0% | -0.44 | 0.48 | 4.2 | 4 |
| holdout @ limit_mid_1.0x_SENSITIVITY | 10 | 30.0% | -0.23 | 0.67 | 4.3 | 4 |

The holdout was run **once**, at the end (`python3 backtest.py --holdout`). Every trade is in `results/trades_costs_1x.csv`, and the full breakdown is in `results/results_with_holdout.json`.

**Development (1.0x costs)**

| Breakdown | Trades | Win rate | Avg R |
|---|---:|---:|---:|
| session: london | 17 | 23.5% | -0.29 |
| session: new_york | 3 | 33.3% | -0.17 |
| direction: short | 5 | 20.0% | -0.41 |
| direction: long | 15 | 26.7% | -0.22 |
| level: asia_high | 4 | 25.0% | -0.26 |
| level: asia_low | 12 | 16.7% | -0.49 |
| level: pdh+asia_high | 1 | 0.0% | -1.00 |
| level: pdl | 3 | 66.7% | +0.84 |

Monthly total R (months with trades): 2021-10 -1.0, 2021-11 +2.0, 2022-02 -1.0, 2022-03 -1.0, 2022-05 +2.0, 2022-06 -1.4, 2022-11 -1.0, 2022-12 -1.1, 2023-01 -1.0, 2023-03 +2.0, 2024-02 -2.0, 2024-06 -1.0, 2024-09 -2.0, 2024-10 +1.2

**Validation (1.0x costs)**

| Breakdown | Trades | Win rate | Avg R |
|---|---:|---:|---:|
| session: london | 8 | 25.0% | -0.40 |
| session: new_york | 1 | 100.0% | +0.66 |
| direction: short | 3 | 33.3% | -0.01 |
| direction: long | 6 | 33.3% | -0.41 |
| level: asia_high | 3 | 33.3% | -0.01 |
| level: asia_low | 2 | 0.0% | -1.01 |
| level: pdl | 3 | 66.7% | +0.18 |
| level: pdl+asia_low | 1 | 0.0% | -1.00 |

Monthly total R (months with trades): 2024-12 +0.9, 2025-03 -1.0, 2025-06 -0.3, 2025-08 -1.0, 2025-09 +1.0, 2025-10 -2.0

**Holdout (1.0x costs)**

| Breakdown | Trades | Win rate | Avg R |
|---|---:|---:|---:|
| session: london | 5 | 20.0% | -0.41 |
| direction: short | 1 | 0.0% | -1.01 |
| direction: long | 4 | 25.0% | -0.26 |
| level: asia_high | 1 | 0.0% | -1.01 |
| level: asia_low | 4 | 25.0% | -0.26 |

Monthly total R (months with trades): 2025-12 -1.0, 2026-02 -1.0, 2026-03 -2.0, 2026-05 +2.0


## What happened in development (disclosed)
1. **First rule set:** cancelling a setup whenever price went past the sweep extreme killed 208 of 314 setups.
2. **Development-only change:** I replaced it with: a wick extends the stop run, and a close beyond the swept level cancels (a breakout).
3. **A slip:** the first run had also printed validation results (7 trades, −0.31R per trade) for the old rule. So validation was seen once before the rule change.
4. **Variant table:** a small, pre-declared variant table was run on development data only. All variants were negative:
   - displacement 0.5 ATR: −0.17R
   - no higher-timeframe filter: −0.32R
   - both of those: −0.33R
   - 1.5R target: −0.24R
5. **No selection:** none was selected, and the pre-declared baseline was frozen.
6. **Sensitivity case:** the resting limit order at the gap midpoint was declared before validation and holdout. It is shown for information and is also negative in every split.

## Findings
- **No edge (VERIFIED).** The strategy loses money after costs in development, validation and holdout, at every cost level tested.
- **Tiny samples:** 20, 9 and 5 trades. They are too small to prove *no* edge either. But nothing here supports trading it.
- **Why so few trades (VERIFIED, development funnel):**
  - About 610 level sweeps closed back inside in 3 years.
  - 61% of them closed back beyond the level before any displacement, so they were breakouts, not stop runs.
  - Of the rest, 48 showed qualifying displacement and 20 retraced to an entry.
  - On this data, "sweep then reverse" at the Asian or prior-day levels is the exception, not the rule.
- **Costs are not the main problem (VERIFIED).** The median stop is 21 pips. Round-trip costs of about 1.6 pips (1.2 spread + 0.4 slippage) are therefore about 0.08R per trade, and doubling them costs only about another 0.04R. The losses come from the signal itself.

## Compared with the 2-minute models (REPORTED)
- **No like-for-like comparison is possible.** Clean 2-minute GBP/USD history covers only 13 Apr – 30 Sep 2026.
- **The frozen GBP/USD 2-minute research replay** on that clean data (StLudaetuc_Backtest_Evidence_2026-10-27, HISTORICAL_GBP_RESULT.md) produced **zero trades**.
- **An earlier engine study** reported the v5.2 2-minute trend-pullback model at about −0.27R per trade in replay and about −0.25R live.
- **Conclusion:** neither the 2-minute models nor this 15-minute model has demonstrated a positive edge on clean data. The 15-minute timeframe does cut the cost drag sharply: about 0.08R here, against 0.2–0.5R at 2 minutes (REPORTED, engine v4 cost estimates). But lower costs alone don't make this signal profitable.

## Pine vs Python parity (UNPROVED)
`pine/research/STL_GBPUSD_15m_LiquiditySweep_RESEARCH_v1.pine` implements the same rules. It has **not** been compiled or run in TradingView from this environment. Known differences:
- **Costs:** Pine models them as commission plus slippage on chart prices, where Python uses bid/ask fills.
- **Entry bar:** on the entry bar itself, Pine's target is based on the signal close. From the next bar it is re-anchored on the fill.
- **Chart prices:** TradingView's OANDA chart prices may differ from OANDA API mid prices.

A trade-by-trade comparison with TradingView's list of trades is still needed before any parity claim.

## Reproduce
```
cd research/gbpusd_15m
sha256sum data/GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz   # must match MANIFEST.json
python3 backtest.py --holdout
```


## 2026-10-28 update: data correction and timeframe study

The M15 data file is missing 2,680 bars (two holes in Aug–Sep and Dec 2023). See `data/MANIFEST.json` (CORRECTION_2026_10_28) and `timeframe_study/README.md`, which reruns the B+F+H base on complete M5/M1 data across 3m, 5m, 10m, 15m, 30m, 45m and 1h. No timeframe shows a robust edge.
