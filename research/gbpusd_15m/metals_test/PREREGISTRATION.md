# Gold (XAU/USD) and silver (XAG/USD) test of B+F+H and B+F+H+F2 (pre-registered before any metals data was downloaded)

Written 2026-10-28 ~22:40Z, at Andy's request ("Can we test this theory on gold and silver?").

**Context:** B+F+H failed its pre-registered EUR/USD test, and the family was closed (`../eurusd_test`). This is extra out-of-sample evidence, not a reopening. Nothing was tuned on metals data.

**Frozen rules: identical to `../eurusd_test/PREREGISTRATION.md`.**
- `bbsys2.run` with:
  - `%B` 36/78, arm 8 bars, swing 2, stop buffer 0.25 ATR, minimum stop 0.75 ATR;
  - session exits, flat at 20:45 London;
  - core windows 07:00–11:00 and 12:30–16:00 UTC;
  - daily-trend gate on the prior 17:00 New York close.
- F2 threshold: 3.544 ATR, taken from GBP/USD development data.

**The only instrument-specific input:**
- Slippage = 0.15 × the instrument's median close spread over the whole file.
- That matches the GBP/USD assumption: 0.2 pip against a median spread of about 1.3 pips.
- It uses quote data only, not trade outcomes.

**Data:** OANDA M15 bid/ask, complete candles, 2021-10-25 to 2026-10-28. Checked for intraweek holes before any result is computed.

**Evaluation:** one run per metal on the full 5 years, at 1×, 1.5× and 2× costs.

**Pass criteria, per metal.** Two markets are tested, so the t threshold is raised for multiple testing. Primary system: B+F+H+F2.
1. PF ≥ 1.0 at 1.5× costs.
2. avgR > 0 at 1× costs, with t ≥ 2.3 (Bonferroni across 2 markets).
3. Same sign of avgR in at least 4 of the 6 calendar years.

**If a metal passes:** that is a hypothesis for that metal only. It would go to a SHADOW trial; it is not a reason to reopen GBP/USD.

**If both metals fail:** the conclusion stands, and no further markets will be added for this family.
