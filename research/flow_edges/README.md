# Flow-based edges (US intraday momentum, turn of month, month-end FX rebalancing): **all FAIL** (VERIFIED)

Hypotheses, data, costs and PASS criteria were pre-registered in `PREREGISTRATION.md` (commit 9f4aaaf, pushed before any intraday data was fetched). Research only; nothing touches production.

**Data:** OANDA M30 bid/ask, 2015-01-01 to 2026-10-29, 12 instruments, no unexplained gaps (`results/DATA_MANIFEST.txt`). **Costs (1×):** buy at ask / sell at bid plus 10% of the spread per leg; H2 also pays financing (USD 3-month rate + 2.5%) per night.

## Results (net, basis points per trade; t-statistic clustered by date)
| | Trades | Gross mean | Net mean 1× | t (1×) | PF 1× | PF 2× | Thirds (1×) | Verdict |
|---|---|---|---|---|---|---|---|---|
| **H1** US intraday momentum (last 30 min) | 11,706 | +0.5 (t 1.0) | −1.2 | −2.3 | 0.88 | 0.73 | −1.8 / +0.5 / −2.4 | **FAIL** |
| **H2** Turn of month (4 US indices) | 564 | +21.0 | +19.3 | 1.1 | 1.27 | 1.24 | −8.6 / +58.5 / +8.0 | **FAIL** |
| **H3** Month-end FX hedge rebalancing (4 pairs) | 492 | +0.6 | −0.9 | −0.4 | 0.94 | 0.85 | −0.3 / −4.1 / +1.5 | **FAIL** |

- **H1:** The effect published for 1993–2013 (Gao et al.) and through ~2020 (Baltussen et al.) is not present on OANDA index CFDs since 2015: gross +0.5 bp, not significant, smaller than the ~1.7 bp round-trip cost. Post-2021: −1.5 bp net (t −2.7).
- **H2:** Turn-of-month windows earned +19.3 bp net versus +7.5 bp for every other 4-day long window (the ordinary equity drift). The 12 bp difference is not significant (t 1.1) and comes almost entirely from 2019–22; the first third is negative. Fails criteria 1 and 3.
- **H3:** No sign of the Melvin & Prins month-end effect in 2015–2026 (gross +0.6 bp).

## What this adds
1. Published, mechanism-backed intraday flow effects are **gone or below OANDA's costs** on these instruments. This is consistent with everything else tested: before-cost edges on liquid OANDA markets are about zero.
2. The only reliably positive number in this study is the **baseline**: simply being long US indices earned +7.5 bp per 4-day window after OANDA spread and financing (t 2.1). That is the equity risk premium, not a trading edge — and on a CFD it costs about 2.5% a year more than holding a low-cost index fund.

## Related venue diagnostic (`../daily_portfolio/venue_diag.py`, not a pre-registered test)
Daily trend-following with OANDA spreads but **no financing markup** (≈ a futures venue): metals + commodities Sharpe 0.20 / 0.12 / 0.45 (dev / val / holdout), full 0.23, monthly PF 1.23. Positive in every period but modest; not tradable as such on OANDA.

Reproduce: `FLOW_STORE=<parsed M30 dir> RATES=<OECD csv> python3 fe.py`
