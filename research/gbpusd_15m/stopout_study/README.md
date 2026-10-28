# Can we filter out stop-outs from 15m B+F+H? RESEARCH ONLY

**Not for live use.** The Oracle reviews this. Nothing goes live without Andy's approval and a SHADOW trial.

**Question (Andy):** "Is there anything we can add into the most successful strategy you backtested that would remove some of the trades stopped out to improve the profit factor?"

**Setup:**
- Data: complete 15m bars built from the M5 file (see `../timeframe_study/README.md`). Frozen B+F+H rules.
- Every feature is measured on the **signal bar**, the bar before entry, so nothing used would be unknown at the time of the trade.
- Filters are chosen on development data only, then checked once on validation. The holdout was already used (`../timeframe_study`), so it can't serve as a fresh test.

## 1. Stop-outs can't be identified at entry (VERIFIED, `study.py`)
- 56% of development trades are full stop-outs (average −1.01R). The rest average +1.30R, because session exits let winners run.
- Several features *do* predict the stop-out rate:
  - stop width in ATR: 74% stopped in the narrowest fifth vs 38% in the widest;
  - band width;
  - how much of the day's typical range is already used: 73% vs 38%;
  - ATR level;
  - time of day.
- **None of these improves average R.** When they cut stop-outs, the winners shrink by the same amount. A wider stop is hit less often, but each 1R is worth fewer pips. Fewer stop-outs does not by itself mean a better PF.
- 14 features were scanned in 5 buckets each. That many looks would produce a best |t| of about 3 from noise alone. Results that peak in a *middle* bucket (8-bar momentum, bar range) were treated as noise and not used.

## 2. Two filters, chosen on development (monotone, with a market reason)
- **F1 `trend_al`:** trade only in the direction of the 4H structure.
- **F2 `not_ext`:** skip a trade when price is already more than **3.544 ATR** beyond the prior daily close in the trade's direction, i.e. don't chase an extended day. The 3.544 is the 80th percentile on development data.

## 3. Validation, run once (VERIFIED, `validate.py`, `validation_result.json`)

| Filter | Development PF (1×) | Validation PF (1×) | Development / validation PF at 1.5× costs | Validation trades | Stop-out rate (validation) |
|---|---|---|---|---|---|
| Base | 1.008 | 1.060 | 0.943 / 0.989 | 439 | 52% |
| F1 trend aligned | 1.203 | **0.984** | n/a | 230 | 51% |
| **F2 not extended** | 1.081 | **1.123** | **1.010 / 1.045** | 327 | 54% |
| F1 + F2 | 1.344 | 0.983 | n/a | 156 | 56% |

- **F1 fails:** it looked like the best idea on development and collapsed on validation. That is the usual overfitting pattern.
- **F2 holds up.** It is the first change in this whole study that stays above PF 1 at 1.5× costs on both development and validation.
- **F2 is UNPROVED.**
  - The validation gain (avgR +0.034 → +0.071) is well within one standard error (0.094).
  - At 2× costs it is below 1 again (0.946 / 0.975).
  - It works by removing a group of trades that lose on average. It does **not** lower the stop-out rate.

## 4. What it would take to prove F2
The GBP/USD holdout is used up, so F2 needs data it has never seen. Two options:
- **EUR/USD 15m** (or another major) over the same 5 years, with B+F+H+F2 frozen exactly as above and evaluated once. This requires another OANDA download.
- **Forward SHADOW** on GBP/USD: paper signals only. Roughly 6–9 months would give about 150 trades, still a wide confidence interval.

Until one of these passes, treat F2 as a hypothesis, not an edge.
