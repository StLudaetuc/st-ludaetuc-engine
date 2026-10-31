# Gold and silver test of B+F+H(+F2): **both FAIL** the pre-registered criteria (VERIFIED)

The rules and criteria were pre-registered in `PREREGISTRATION.md` (commit 022ebb7) before any metals data was downloaded. Each metal was run once with the rules unchanged (`run_test.py`, output in `result.json`).

**Data:** OANDA M15 bid/ask, 2021-10-25 to 2026-10-28. Gold has 118,578 bars and silver 118,538. The only gaps are the daily 1-hour maintenance break and US holiday or early closes, with an identical pattern in both metals. SHA-256 values are in `../DATA_SHA256.txt`.

**Slippage:** 0.15 × median spread. Gold: 0.0555 USD, spread/ATR 0.125. Silver: 0.00272 USD, spread/ATR 0.275.

## Gold (XAU/USD)

| System | Costs | Trades | avgR ± SE | t | PF |
|---|---|---|---|---|---|
| B+F+H | 1× | 2150 | +0.045 ± 0.040 | +1.11 | 1.076 |
| **B+F+H+F2** | **1×** | 1510 | **+0.071 ± 0.052** | **+1.37** | **1.118** |
| B+F+H | 1.5× | 2150 | +0.009 ± 0.039 | +0.22 | 1.014 |
| **B+F+H+F2** | **1.5×** | 1510 | +0.031 ± 0.050 | +0.62 | **1.051** |
| B+F+H+F2 | 2× | 1510 | −0.007 ± 0.049 | −0.13 | 0.990 |

**Criteria:**

| Criterion | Result |
|---|---|
| 1. PF ≥ 1 at 1.5× costs | passed (1.051) |
| 2. t ≥ 2.3 | **failed** (1.37) |
| 3. Same sign in ≥ 4 of 6 years | passed (4 of 6) |

**Average R by year (B+F+H+F2, 1×):**

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| −0.03 | +0.19 | +0.04 | +0.10 | −0.04 | +0.07 |

## Silver (XAG/USD)

| System | Costs | Trades | avgR ± SE | t | PF |
|---|---|---|---|---|---|
| B+F+H | 1× | 2198 | −0.094 ± 0.035 | −2.67 | 0.845 |
| **B+F+H+F2** | **1×** | 1605 | −0.090 ± 0.043 | −2.07 | **0.857** |
| B+F+H+F2 | 1.5× | 1605 | −0.156 ± 0.042 | −3.74 | 0.763 |

Silver fails all three criteria. It loses more than its costs explain: spread/ATR is high (0.275), and gross returns are negative too.

## Interpretation
1. **Gold is the best result of this whole study, but it is not proven.**
   - It is the first market where the base system, with no filter, is above PF 1 at 1× costs on its own data. F2 again improves it, as on GBP/USD.
   - But t = 1.37 is roughly a 1-in-12 chance result for one market. Across the five independent tests so far (GBP/USD, EUR/USD, gold, silver, and the GBP/USD holdout), one result this good is about what luck alone would produce.
   - The edge disappears at 2× costs.
2. **Silver is clearly negative** and should not be traded with this system.
3. **Overall, the evidence still does not support an edge.**
   - Two of three new markets lose and one is slightly positive. That spread is what you'd expect from a system that is breakeven before costs.
4. **The only reasonable next step for gold** is a forward SHADOW trial, which needs Andy's approval. Rules and the F2 threshold stay frozen exactly as here, with paper signals only. **Correction (2026-10-29):** this originally said about 150 trades (6 months) would confirm or reject avgR ≈ +0.07. That was wrong. Trades vary by about 2R each, so confirming +0.07R at t ≥ 2 needs about 3,300 trades (about 11 years). A short trial can only catch a clear failure. See also `../fx3_test/README.md`, which recommends against a trial. Tuning anything further on these 5 years would invalidate this test.
