# Bespoke per-instrument scripts, 2026-10-26

These scripts produce `pine/STL_{EURUSD,AUDUSD,USDCAD,USDJPY}_2m_CandidateGenerator_v6.pine` and `pine/STL_{XAUUSD,XAGUSD}_2m_CandidateGenerator_v2.pine`. Each one is generated from that instrument's own v1 file, so its pair constants, session timezone and history notes carry over.

| file | what it does |
|---|---|
| `pairprof.py` | volatility relative to GBP/USD, active hours, and stretched-vs-chasing win rates for each pair, from its 5m sheet |
| `pairmodel.py` | compares setups per pair: trend-aligned pullback, chasing, breakout, counter-trend fade |
| `pairhours.py` | the best setup per pair, split into UTC hour blocks, at two stop sizes |
| `make_pine_bespoke.py` | builds the six scripts: `python make_pine_bespoke.py pine` |

The data comes from the "<PAIR> Market Data" sheets, tab `5m`, columns A (candle time), D:G (ATR, RSI, BB upper, BB lower) and U (previous candle close). The last ~25,000 rows cover Jun–Sep 2026. These sheets have no high/low data, so every result is a close-path estimate with no spread cost, which makes it weaker evidence than the GBP/USD 2m backtest in `../overnight_2026_10_24/`. No gold or silver price data was available, so the metal settings are based on cost only.
