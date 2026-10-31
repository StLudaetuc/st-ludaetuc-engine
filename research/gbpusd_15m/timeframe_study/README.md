# GBP/USD breakout base (B+F+H) across timeframes: RESEARCH ONLY

**Not for live use.** Nothing here touches production, Pine, Make, OANDA or Render, the engine or any database. The Oracle reviews this independently. Nothing goes live without Andy's approval and a SHADOW trial.

## Question

Andy asked: "Can we explore the idea across 3-minute, 5-minute, 10-minute, 30-minute, 45-minute and 1hr?"

The idea is the frozen B+F+H breakout base from `bb_vmi_study/`:
- session exits;
- core windows 07:00–11:00 and 12:30–16:00 UTC;
- daily-trend gate: long only above the prior daily close, short only below it.

## Method

- Every timeframe uses **the same rules with no retuning**. Bar-count parameters (BB length, ATR length and so on) are left as bar counts, so they span different clock time on each timeframe. That is the honest "same idea" test. Retuning per timeframe would be a new search, with a new multiple-comparison cost.
- Fills, costs and splits are the same as before:
  - signals on mid prices;
  - fills on bid/ask plus 0.2 pip slippage;
  - stop assumed hit first when stop and target fall in the same bar;
  - chronological 60/20/20 development / validation / holdout split;
  - costs at 1×, 1.5× and 2×.
- Holdout gate: open the holdout only if a timeframe clears PF 1 in **both** development and validation.
- Data, all OANDA bid/ask candles (VERIFIED: SHA-256 values in `../DATA_SHA256.txt` and per-file `*.MANIFEST.json`):
  - 5m, 10m, 15m, 30m, 45m and 1h are built from **M5, 5 years** (374,217 bars).
  - 3m is built from **M1, 2 years** (742,751 bars). OANDA has no native M3. Five years of M1 is about 1.85M bars, so I pulled 2 years, the brief's minimum. For a like-for-like comparison, 5m and 15m are also shown on the same 2 years.

## Data correction (VERIFIED)

The M15 file committed earlier (`GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz`) is **missing 2,680 bars** in two holes:
- 2023-08-15 to 2023-09-04
- 2023-12-12 to 2024-01-03

Its manifest wrongly said there were no holes; a correction is now recorded in `MANIFEST.json`. Every bar that is present matches the new M5 data exactly. All earlier 15m, 30m, 45m and 1h numbers ran on the holed file. Both holes fall in the development split.

**Every number below uses the complete M5/M1 data.**

## Results: development / validation, profit factor (avgR ± SE) (VERIFIED, from `alltfs.json`)

### 5 years (from M5)

| TF | spread/ATR | Dev 1× | Dev 1.5× | Val 1× | Val 1.5× | Trades dev/val |
|---|---|---|---|---|---|---|
| 5m  | 0.298 | 0.937 (−0.046 ± 0.052) | 0.859 | 0.947 (−0.037 ± 0.080) | 0.870 | 2026 / 623 |
| 10m | 0.206 | 0.915 (−0.056 ± 0.050) | 0.853 | 1.053 (+0.032 ± 0.078) | 0.980 | 1616 / 492 |
| 15m | 0.167 | **1.008** (+0.005 ± 0.046) | 0.943 | **1.060** (+0.034 ± 0.075) | 0.989 | 1417 / 439 |
| 30m | 0.115 | 0.939 (−0.033 ± 0.041) | 0.880 | 1.030 (+0.014 ± 0.066) | 0.963 | 1214 / 370 |
| 45m | 0.092 | 0.791 (−0.105 ± 0.037) | 0.743 | 0.899 (−0.047 ± 0.065) | 0.842 | 1068 / 342 |
| 1h  | 0.079 | 0.704 (−0.139 ± 0.034) | 0.663 | 0.961 (−0.016 ± 0.067) | 0.898 | 928 / 285 |

### 2 years (from M1)

| TF | spread/ATR | Dev 1× | Dev 1.5× | Val 1× | Val 1.5× | Trades dev/val |
|---|---|---|---|---|---|---|
| 3m  | 0.417 | 0.865 (−0.104 ± 0.075) | 0.781 | 0.899 (−0.076 ± 0.130) | 0.808 | 926 / 300 |
| 5m  | 0.317 | 0.916 | 0.840 | 0.993 | 0.899 | 753 / 260 |
| 15m | 0.177 | 1.014 | 0.944 | 0.831 | 0.772 | 528 / 194 |

At 2× costs everything is below 0.93 (see `alltfs.json`).

### 15m holdout (VERIFIED, `m15_complete_holdout.json`)

On complete data, 15m is the only timeframe that clears 1 at 1× in both development and validation, so the protocol opens its holdout.

**Caveat:** this rule set's holdout was already looked at once, on the holed file, with PF 0.852. The holdout period (2025-10-27 onward) was not affected by the holes. This is therefore a re-reading of a burned holdout, not a fresh test.

| Holdout | Trades | avgR ± SE | PF |
|---|---|---|---|
| 1×   | 456 | −0.093 ± 0.072 | 0.845 |
| 1.5× | 456 | −0.136 ± 0.070 | 0.781 |
| 2×   | 456 | −0.177 ± 0.069 | 0.723 |

## Conclusions

1. **No timeframe produces a robust edge (VERIFIED for these rules and this data).**
   - Nothing clears PF 1 at 1.5× costs in any split.
   - The best case, 15m, has a development avgR of +0.005R with an SE of 0.046R: indistinguishable from zero.
   - The 15m holdout is 0.845.
2. **Cost drag explains the shape (VERIFIED as a pattern, REPORTED as the mechanism).**
   - Spread relative to ATR falls from 0.42 (3m) to 0.08 (1h).
   - The shortest timeframes lose most to costs. 3m and 5m are the worst at 1.5× and 2×.
   - Faster entries do not recover what they pay in spread.
3. **The signal itself weakens above 30m (VERIFIED).** At 45m and 1h, development PF is low even with costs near their smallest share of ATR. The structure that 15m partly captures does not carry to slower bars with these session windows.
4. **15m stays the best of the six, but its edge is UNPROVED and probably zero.**
   - The 0.99 → 1.008 change in development PF comes only from filling the data holes. A shift that size from about 2% more data shows how close to noise the result is.
5. **Recommendation:** stop iterating on B+F+H. On every timeframe and both data versions it sits within about 1 SE of breakeven before realistic cost slack, and below breakeven after it. More variants would mostly fit noise.

## Reproduce

```
python3 research/gbpusd_15m/timeframe_study/run_alltfs.py
```
