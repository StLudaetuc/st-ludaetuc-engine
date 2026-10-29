# Daily-bar trend-following and carry portfolio across OANDA markets: **FAIL** (VERIFIED)

The rules and pass criteria were pre-registered in `PREREGISTRATION.md` (commit bbfe462) before any backtest.

**Data**
- OANDA daily bid/ask candles for all 123 instruments, 2007-01 to 2026-10. They are aligned to **09:00 New York**, so every trade fills in liquid hours at real bid/ask.
- OECD 3-month interbank rates for 17 currency areas, used for carry and financing.
- The first run, on 17:00-aligned data, was discarded because of a data bug; see `NOTE_FIRST_RUN_INVALID.md`.

**Costs**
- Half the spread plus 10% slippage on every position change.
- Daily financing: the interest-rate differential plus OANDA's markup, about 1.0% a year per side on forex and 2.5% on CFDs and metals. The markups are inferred from OANDA's live long and short rates.

## Pre-registered results at 1× costs (Sharpe; monthly profit factor in brackets)
| System | Development 2007–18 | Validation 2019–22 | Holdout 2023–26 | Full |
|---|---|---|---|---|
| Trend, forex (47 pairs) | −0.39 (0.70) | −0.79 (0.52) | −1.09 (0.36) | −0.59 |
| Trend, metals | 0.16 (1.15) | −0.15 (0.88) | **0.45 (1.48)** | 0.16 |
| Trend, indices | −0.16 (0.86) | −0.46 (0.65) | −0.32 (0.76) | −0.25 |
| Trend, bonds | −1.10 (0.39) | 0.20 (1.17) | −0.54 (0.64) | −0.64 |
| Trend, commodities | −0.05 (0.95) | 0.09 (1.08) | −0.11 (0.90) | −0.03 |
| Trend, all markets | −0.53 (0.59) | −0.72 (0.54) | −0.95 (0.39) | −0.65 |
| Carry, forex | −0.34 (0.74) | −0.26 (0.80) | **0.49 (1.66)** | −0.20 |
| Combined 50/50 | −0.64 (0.54) | −0.80 (0.46) | −0.37 (0.72) | −0.62 |

**Pass criteria:**
1. Holdout monthly PF ≥ 1.2.
2. Full-period Sharpe ≥ 0.4 at 2× costs.
3. Sharpe > 0 in every period.

**No system meets all three.** At 2× costs everything is worse (see `results/` logs).

## Why: the diagnostic (`diag.py`, frictionless: no spread and no markup, with the pure interest-rate differential kept)
| System | Frictionless Sharpe (dev / val / holdout / full) | Average gross notional ÷ equity |
|---|---|---|
| Trend, forex | 0.12 / −0.27 / −0.38 / −0.05 | 11.9 |
| Trend, metals | 0.45 / 0.06 / 0.59 / 0.41 | 0.75 |
| Trend, indices | 0.04 / −0.29 / −0.07 / −0.06 | 1.9 |
| Trend, bonds | 0.02 / 1.21 / 0.10 / 0.34 | 7.1 |
| Trend, commodities | 0.14 / 0.32 / 0.15 / 0.18 | 0.69 |
| Carry, forex | −0.12 / −0.02 / 0.82 / 0.04 | 7.2 |

1. **Before costs, simple trend-following earned little in 2007–2026** on these markets, except metals (Sharpe 0.41) and, mainly through 2022, bonds. This matches the well-documented weak decade for trend after 2009.
2. **OANDA's overnight financing markup is charged on notional every day.** Low-volatility markets (forex, and above all bond CFDs) need large notional for the same risk, so a 1–2.5% markup per side becomes a drag of 0.15–1.5 Sharpe. It destroys them.
3. **Carry earned before costs only in 2023–26**, when rate gaps were wide. Over the whole period it was flat, with deep carry-crash drawdowns.

## Exploratory check (`explore.py`; labelled, selected on development only)
- **Setup:** 4 trend variants (a lookback blend, 252-day, 126- and 252-day, EWMAC) × 5 groups (metals, commodities, indices and their combinations).
- **Best development pick:** metals with the 126/252-day lookbacks, development Sharpe 0.18. It **failed validation (−0.48)**.
- **The rest:** no variant is positive in both development and validation by a meaningful margin. Full table: `results/explore_trend_variants.csv`.

## Conclusion
- On a retail OANDA account, daily trend-following and carry do **not** produce a robust edge after costs over 2007–2026.
- The one market group with a real before-cost trend edge is **metals** (gold, silver, platinum, palladium, copper), but after OANDA's financing markup it is too weak and inconsistent (Sharpe 0.16; validation negative).

Your original reason for closing trades before the overnight roll is supported: on OANDA the overnight markup is a significant cost.

The data sits in the session scratchpad and is not committed, because of its size. Code: `dp.py`, `run_prereg.py`, `diag.py`, `explore.py`.
