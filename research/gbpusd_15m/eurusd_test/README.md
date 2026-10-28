# EUR/USD out-of-sample test of B+F+H+F2: **FAIL** (VERIFIED)

The rules and pass criteria were pre-registered in `PREREGISTRATION.md` (commit d87449a) **before** the EUR/USD data was downloaded. They were run once, unchanged (`run_test.py`, output in `result.json`).

**Data:** OANDA EUR_USD M15 bid/ask, 2021-10-25 to 2026-10-28, 124,850 bars. The only gaps are holiday closures. The SHA-256 is in `../DATA_SHA256.txt`. Median spread/ATR is about 0.17, essentially the same cost burden as GBP/USD (0.167).

| System | Costs | Trades | avgR ± SE | t | PF |
|---|---|---|---|---|---|
| B+F+H | 1× | 2354 | −0.037 ± 0.033 | −1.12 | 0.937 |
| **B+F+H+F2** | **1×** | 1789 | −0.027 ± 0.040 | −0.69 | **0.956** |
| B+F+H | 1.5× | 2354 | −0.073 ± 0.033 | −2.24 | 0.881 |
| **B+F+H+F2** | **1.5×** | 1789 | −0.065 ± 0.039 | −1.68 | **0.896** |
| B+F+H | 2× | 2354 | −0.107 ± 0.032 | −3.35 | 0.830 |
| B+F+H+F2 | 2× | 1789 | −0.101 ± 0.038 | −2.68 | 0.843 |

**Criteria:**

| Criterion | Result |
|---|---|
| 1. PF ≥ 1 at 1.5× costs | **failed** (0.896) |
| 2. avgR > 0 with t ≥ 2 | **failed** (−0.027, t −0.69) |
| 3. F2 beats B+F+H | passed, but the gain (+0.010R) is far inside the noise |

**Average R by year (B+F+H+F2, 1× costs):**

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| −0.22 | −0.09 | +0.04 | −0.08 | +0.04 | +0.03 |

## Conclusion
- The "not extended" filter (F2) does not carry over to an independent market, and neither does the B+F+H base. What looked like a small edge on GBP/USD 15m is consistent with noise plus favourable split timing.
- As pre-registered, **F2 is treated as a GBP/USD artefact, and the B+F+H family is closed.**
