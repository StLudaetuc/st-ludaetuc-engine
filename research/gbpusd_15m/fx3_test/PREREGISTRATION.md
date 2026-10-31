# AUD/USD, USD/CAD and USD/JPY test of B+F+H(+F2) (pre-registered before any data for these pairs was downloaded)

Written 2026-10-28 ~23:05Z, at Andy's request ("Now let's do the 3 forex pairs we haven't done yet"). These are the three remaining pairs in the house set, the ones with Pine CandidateGenerators in `pine/`.

**Context so far:**
- GBP/USD and EUR/USD fail.
- Gold is positive but not significant (t 1.37).
- Silver fails.

This adds independent evidence only. Nothing is tuned on these pairs.

**Frozen rules: identical to `../eurusd_test` and `../metals_test`.**
- B+F+H with `%B` 36/78, arm 8 bars, swing 2, stop buffer 0.25 ATR, minimum stop 0.75 ATR.
- Session exits, flat at 20:45 London.
- Core windows 07:00–11:00 and 12:30–16:00 UTC.
- Daily-trend gate on the prior 17:00 New York close.
- F2 threshold: 3.544 ATR.

**Slippage:** 0.2 pip, the same assumption as GBP/USD and EUR/USD:
- 0.00002 for AUD/USD and USD/CAD;
- 0.002 for USD/JPY.

The JPY pip is 0.01, but R-units make results comparable across pairs.

**Data:** OANDA M15 bid/ask, complete candles, 2021-10-25 to 2026-10-28. Checked for intraweek holes before any result is computed.

**Evaluation:** one run per pair on the full 5 years, at 1×, 1.5× and 2× costs.

**Pass criteria, per pair.** Three markets are tested, so the t threshold is raised for multiple testing. Primary system: B+F+H+F2.
1. PF ≥ 1.0 at 1.5× costs.
2. avgR > 0 at 1× costs, with t ≥ 2.4 (Bonferroni across 3 pairs).
3. avgR > 0 in at least 4 of the 6 calendar years.

**Also reported:** the pooled result across all 7 markets tested so far (equal weight per trade), as a summary of the evidence. It is not a pass/fail test.
