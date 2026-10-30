# Universe scan: stop-loss trend strategy on 45 IG-style spread-bet markets — results (pre-registered in PREREGISTRATION.md)

Strategy: weekly 63/126/252-day trend signal, 3×ATR stop, no take profit, 1% risk per trade. OANDA daily data 2007–2026
(weekdays), costs 1.5× OANDA spread per round trip and per roll, no funding.

| Portfolio | Dev 2007–18 PF / avgR | Val 2019–22 | Holdout 2023–26 | Full at 3× costs |
|---|---|---|---|---|
| All 45 markets | 1.16 / +0.07 | 1.14 / +0.07 | **0.66 / −0.15** | 0.96 |
| Top 10 by development | — | 0.96 / −0.02 | 1.38 / +0.14 | — |
| Bottom 10 by development | — | 0.73 / −0.12 | 0.44 / −0.26 | — |
| Best 2 per asset class (dev) | — | 1.19 / +0.09 | 0.71 / −0.12 | — |

By asset class, holdout 2023–26 PF: metals 1.31, energy 0.59, grains 0.52, indices 0.84, bonds 0.29, FX 0.41.
At 0.5% risk per trade the all-market portfolio made ≈ +5.9%/yr (dev), +4.0%/yr (val) and **−19%/yr (holdout)**,
max drawdown up to 69% (trade-sequence approximation; ignores open-trade marking).

Picking test (pre-registered rule: picking helps only if the dev-top-10 beats all-markets in both validation and
holdout): **FAIL** — worse in validation (0.96 vs 1.14), better in holdout (1.38 vs 0.66). Ranking on past results did
not reliably pick future winners, although the dev-bottom-10 stayed bad.

Conclusion: the strategy's results depend heavily on period and market; 2023–26 lost money in most asset classes. Metals
(above all gold) and US equity indices were the exceptions, which is exactly the after-the-fact pattern that should not
be used to choose markets. Per-market tables: results.json.
