# EUR/USD out-of-sample test of B+F+H+F2 (pre-registered before any EUR/USD data was downloaded)

Written 2026-10-28, ~22:17Z, before the EUR/USD candles were fetched. Committed separately, ahead of the results.

**Hypothesis:** the GBP/USD 15m B+F+H strategy with filter F2 ("not extended") has a positive expectancy after costs that transfers to EUR/USD.

**Frozen rules: no changes allowed.**
- `bbsys2.run` with:
  - `%B` 36/78, arm 8 bars, swing 2, stop buffer 0.25 ATR, minimum stop 0.75 ATR;
  - session exits, flat at 20:45 London;
  - core windows 07:00–11:00 and 12:30–16:00 UTC;
  - daily-trend gate: long only above the prior daily close, short only below it.
- F2: skip a trade when (signal-bar close − prior daily close) × direction ÷ ATR > **3.544**. That value comes from GBP/USD development data; it is not re-estimated on EUR/USD.
- Slippage: 0.2 pip in price units, the same as GBP/USD (EUR/USD pip = 0.0001).

**Data:** OANDA EUR_USD M15 bid/ask, 2021-10-25 to 2026-10-28, complete candles only. The data is checked for intraweek holes before any result is computed.

**Evaluation:** one run on the **full 5 years**. No splits, because nothing is tuned on EUR/USD.

**Pass criteria (all must hold):**
1. PF ≥ 1.0 at 1.5× costs for B+F+H+F2.
2. avgR > 0 at 1× costs, with t = avgR / SE ≥ 2.0.
3. F2 improves on B+F+H without F2 (avgR higher at 1×).

**Also reported, not judged:** B+F+H without F2, PF at 1×, 1.5× and 2× costs, and results by year.

**If it fails:** F2 is treated as a GBP/USD artefact, and the B+F+H family is closed.
