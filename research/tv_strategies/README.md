# Old TradingView gold strategies (S1 BB+RSI stop-and-reverse, S2 BB+RSI 2-candle): **no edge outside the TradingView window** (VERIFIED)

Andy's TradingView Strategy Tester screenshots (29 Oct 2026, OANDA:XAUUSD): S1 10m PF 2.201 (34 trades); S2 30s PF 3.662 (44), 45s PF 2.075 (35), 5m PF 2.175 (45).

**Method.** `tvs.py` replicates both Pine scripts line by line (SMA21 ± 2 population stdev, Wilder RSI14, ATR14; signals at bar close, fills at next bar open, quantity 1 oz). Data: OANDA bid/ask — M1 Oct 2025–Oct 2026 (→1m/5m/10m/15m), S5 12–28 Oct 2026 (→30s/45s), M15 Oct 2021–Oct 2026. Mode `tv` = mid fills, no costs (what TradingView shows); `real` = ask/bid fills + overnight financing.

**Replication check (TradingView window, `tv` mode).** S2 5m: 45 trades, PF 2.175, win 64.4% — an exact match. S1 10m: 35 vs 34 trades, PF 2.31 vs 2.20. S2 45s: 35 vs 35, PF 1.85 vs 2.08. S2 30s: 39 vs 44, PF 3.98 vs 3.66 (seconds-bar alignment differs slightly). The match also shows that **S2's stop-loss/take-profit never applied**: `strategy.exit("Take Profit", "Stop Loss", ...)` names a non-existent entry "Stop Loss" as `from_entry`, so both scripts are effectively always-in stop-and-reverse with **no stop loss**.

## Results with real costs (`real`), profit factor (trades, t-stat of mean trade)
| | 30s (16 days) | 45s (16 days) | 1m (1 yr) | 5m (1 yr) | 10m (1 yr) | 15m (1 yr) | 15m (5 yr) |
|---|---|---|---|---|---|---|---|
| S1 | 2.16 (55, t 1.9) | 0.97 (38) | 0.78 (535, t −1.6) | 0.64 (120) | 1.43 (54, t 0.9) | 0.64 (38) | 0.73 (232, t −1.2) |
| S2 | 2.40 (65, t 2.4) | 1.78 (36, t 1.1) | 0.79 (498, t −1.6) | 0.72 (144) | 0.72 (54) | 0.82 (46) | 0.88 (248, t −0.6) |

Same rules immediately **before** the TradingView window: S2 5m Oct 2025–Jul 2026 PF 0.56 (99 trades); S1 10m Oct 2025–Apr 2026 PF 0.96 (19 trades).

## Conclusion
1. The high TradingView profit factors are a **window effect**: TradingView shows only recent history (days on seconds charts, months on minutes), and gold was choppy in that period. On every timeframe with a year or more of data, both strategies lose after costs (and roughly break even before costs).
2. The only positive results are 30s/45s on **16 days** of data — far too short to judge; the 1m version over a full year (≈500 trades) loses with PF 0.78–0.79.
3. The high win rates (65–80%) come from having **no stop loss**: losers are held until the opposite signal, so occasional losses are 10–30× the average win (worst trade −$600/oz vs average win ≈ +$20). On a live account this is the pattern that produces sudden large drawdowns.
4. Not recommended for live use or a SHADOW trial. If seconds-timeframe gold is to be pursued, the honest next step is to collect several months of 5-second data first and re-test the frozen rules on it.
