# Very short timeframes (10s, 30s, 1m): **all pre-registered strategies FAIL** (VERIFIED)

The rules were pre-registered in `PREREGISTRATION.md` (commit 42b7e0c) before any test. Every fill is on real OANDA bid/ask.
- **5-second data:** fills come 5 seconds after the signal bar closes, which is a realistic TradingView → Make → engine → OANDA delay.
- **1-minute data:** fills are at the next minute's open.

## The key finding: before costs, price at these horizons is a coin flip
For every strategy that trades often enough to measure (S1 burst momentum, S2 burst fade, S6 %B reversal), the **mid-price (before-cost) profit factor is 0.87–1.12**. Gross P&L is within ±0.3 spreads per trade and no t-statistic exceeds 1.4. Every round trip then pays about one spread. The result is a loss of about one spread per trade, with very high confidence.

### 5-second sample (2026-10-12 to 10-28, about 12 trading days), net at 1× costs, profit factor
| Strategy | EUR/USD 10s / 30s / 1m | GBP/USD 10s / 30s / 1m | Gold 10s / 30s / 1m |
|---|---|---|---|
| S1 burst momentum | 0.34 / 0.50 / 0.46 | 0.21 / 0.42 / 0.49 | 0.47 / 0.77 / 0.94 |
| S2 burst fade | 0.29 / 0.41 / 0.74 | 0.25 / 0.30 / 0.48 | 0.56 / 0.63 / 0.62 |
| S3 session opening-range breakout (24 trades) | 0.38 / 0.37 / 0.39 | 0.61 / 0.52 / 0.85 | **2.18 / 2.47 / 2.19** |
| S4 08:30 New York data spike (2–10 trades) | 7.7* | 5.9* | 3.2* |
| S5 round-number fade | 0.19 / 0.07 / 0.09 | 0.08 / 0.07 / 0.17 | 0.53 / 0.55 / 0.62 |
| S6 %B reversal | 0.20 / 0.33 / 0.47 | 0.13 / 0.25 / 0.34 | 0.40 / 0.54 / 0.60 |

\* Too few trades to mean anything (2–10).

### One year of 1-minute data (GBP/USD two years): PF by development / validation / holdout, net
| Strategy | EUR/USD | GBP/USD | Gold |
|---|---|---|---|
| S1 | 0.57 / 0.66 / 0.54 | 0.55 / 0.55 / 0.49 | 0.75 / 0.80 / 0.70 |
| S2 | 0.67 / 0.52 / 0.63 | 0.64 / 0.62 / 0.55 | 0.75 / 0.74 / 0.87 |
| S3 opening-range breakout | 0.64 / 0.85 / 0.63 | 0.88 / 0.73 / 0.61 | 0.82 / **1.15** / 0.98 |
| S4 data spike | 0.33 / 0.37 / 2.20 (18 trades) | 0.84 / 0.17 / 0.63 | 0.41 / 0.68 / 1.31 (40 trades) |
| S5 round-number fade | 0.29 / 0.11 / 0.18 | 0.18 / 0.21 / 0.19 | 0.40 / 0.57 / 0.58 |
| S6 %B reversal | 0.50 / 0.45 / 0.43 | 0.43 / 0.43 / 0.36 | 0.64 / 0.62 / 0.65 |

## Verdict
- **No strategy passes.** The pass mark was PF ≥ 1.2 in both validation and holdout on 2 of 3 instruments.
- **Gold's opening-range breakout** looked strong on the 12-day sample (24 trades). On a full year of 1-minute data it scores 0.82 / 1.15 / 0.98. That is the same small-sample mirage seen earlier on GBP/USD 15m.
- **The 08:30 data spike** has too few events to judge, and its sign flips between periods.
- **Why speed can't fix this:** the edge before costs is about zero at 10s–1m, so faster execution cannot help. Only lower costs could. A retail OANDA account pays the spread on every trade; the firms that profit at these horizons are the ones *earning* the spread (market makers), with exchange-level access.

Files: `stf.py` (engine and strategies), `run_s5.py`, `run_m1.py`, `results_*.json`. The data (OANDA S5 and M1 bid/ask) sits in the session scratchpad and is not committed, because it is large; it can be re-fetched the same way.
