# Overnight study, 2026-10-23 → 24

Scripts behind the engine v4.0.0 and Pine v6.0.0 changes. They're kept so the numbers can be reproduced and re-run as more data arrives.

## Data (not committed, pulled from Google Drive)

- `data/mtl.pkl`: "St Ludaetuc Master Trading Log" sheet (1,505 rows × 637 columns), exported as CSV.
- `data/gbp2m.pkl`: "St Ludaetuc GBP/USD 2m Market Data" sheet. Only columns `market_bar_time_utc` (AA) and `price_open..price_close` (AJ:AM) are needed: 87,216 bars from 2026-04-13 to 2026-09-30.
- `data/feat.pkl`: built by `bt.features()` from `gbp2m.pkl`.

## Scripts

| file | what it does |
|---|---|
| `bt.py` | indicators, an exact Python replica of the v5.2 candidate logic (`candidates_v52`), and a conservative SL/TP walker (stop wins ambiguous bars) |
| `research.py` | `run()`: entry modes (close / stop / closebreak / confirm / limit), stop modes, the min-stop floor, one-position-at-a-time, cost in pips; `split_stats()` for in-sample (Apr–Jul) vs out-of-sample (Aug–Sep) |
| `fstudy2.py` | conditional win rates of trend alignment, Bollinger position, VWAP distance and RSI across every bar × direction |
| `manus_diag.py` | replicates engine `signal_score()` on every bar and shows it is inversely related to outcomes |
| `v6.py` | setup generator for the pullback-reclaim / fade families, plus the grid |
| `v6d.py` | the bar-close-confirmed v6 entry (what Pine v6 actually does) with the edge-model floor |
| `edge_model.py` | fits the logistic edge model (IS → OOS AUC and calibration, then refits on everything); writes `edge_model.json`, whose coefficients are copied into `engine.py` |
| `edge_filter.py` | applies the edge model as a filter to v5.2 and v6 candidates |

Cost assumption everywhere: 1.4 pips round trip (spread plus slippage) for GBP/USD, also tested at 1.0 and 1.8.
