# AUD/USD, USD/CAD and USD/JPY test of B+F+H(+F2): **all three FAIL** (VERIFIED)

The rules and criteria were pre-registered in `PREREGISTRATION.md` (commit b8123f1) before any data for these pairs was downloaded. Each pair was run once with the rules unchanged (`run_test.py`, output in `result.json`).

**Data:** OANDA M15 bid/ask, 2021-10-25 to 2026-10-28, about 124,800 bars per pair. The only gaps are holiday closures plus short OANDA feed gaps in May 2022. Manifests and SHA-256 values are in `../data` and `../DATA_SHA256.txt`.

## B+F+H+F2 (primary system)

| Pair | spread/ATR | Trades | PF 1× | PF 1.5× | PF 2× | avgR 1× (t) | Years positive |
|---|---|---|---|---|---|---|---|
| AUD/USD | 0.245 | 1434 | 0.779 | 0.698 | 0.628 | −0.134 (t −3.64) | 0 of 6 |
| USD/CAD | 0.307 | 1961 | 0.724 | 0.644 | 0.576 | −0.184 (t −5.35) | 0 of 6 |
| USD/JPY | 0.170 | 1420 | 1.052 | 0.967 | 0.892 | +0.031 (t +0.63) | 2 of 6 |

All three pairs fail all three criteria. The B+F+H base without F2 is similar or worse; the full table is in `result.json`.

## All seven markets pooled (B+F+H+F2, equal weight per trade)
Markets: GBP/USD, EUR/USD, AUD/USD, USD/CAD, USD/JPY, gold and silver.

| Costs | Trades | avgR ± SE | t | PF |
|---|---|---|---|---|
| 1× | 11,528 | −0.048 ± 0.016 | **−2.97** | 0.923 |
| 1.5× | 11,528 | −0.099 ± 0.016 | **−6.37** | 0.846 |

## Conclusion (VERIFIED)
- With 11,528 trades across seven markets, the strategy **reliably loses money**. The pooled result is significantly negative, not just unproven.
- AUD/USD and USD/CAD lose in every single year. Their spread is a large share of each move (spread/ATR 0.25–0.31), and they have no breakout follow-through to pay for it.
- USD/JPY and gold are slightly positive. Out of seven markets, that is what chance produces when the true edge is zero or negative.
- **Recommendation:** close the B+F+H family for good. The gold result alone does not justify a SHADOW trial, because pooled across seven markets it reads as noise. Any future work should start from a different market hypothesis.
