# IG demo API check — 30 Oct 2026 ~12:20Z (read-only; no orders)

Temporary Make scenario (deleted after use) logged in to demo-api.ig.com with the credentials in Make data store
`stl_oanda_credentials` (records `ig_demo_*`). Login method: IG REST v3 (token in response body; v2 header tokens
could not be read by Make's HTTP module). Account: CFD demo Z5QIAR, GBP, balance £10,000, dealing enabled. No
spread-bet demo account exists yet.

## Futures-style markets (no overnight funding; funding is in the price) — live demo quotes
| Market | Epic (smaller contract) | Expiry | Spread | Spread % | Rolls/yr |
|---|---|---|---|---|---|
| Gold ($33.20/pt; $100/pt also) | MT.D.GC.FWM3.IP | DEC-26, then FEB-27 | 0.8 | 0.020% | ~6 |
| Silver ($10/pt; $50/pt also) | MT.D.SI.FWM1.IP | DEC-26, then MAR-27 | 10 | 0.20% | ~4 |
| High Grade Copper ($0.50/pt; $2.50/pt also) | MT.D.HG.FWM2.IP | DEC-26, then MAR-27 | 40 | 0.061% | ~4 |
| US Crude ($5/pt; $10, £1 also) | EN.D.CL.FWM5.IP | DEC-26, then JAN-27 | 6 | 0.058% | 12 |
| US 10-Year T-Note ($2/pt; $10/pt also) | IR.D.10YEAR100.FWM2.IP | DEC-26 | 4 | 0.037% | ~4 |
Spot/undated versions (with overnight funding) for comparison: Spot Gold 0.3, Spot Silver 4, US Crude undated 2.8.

## Backtest with these costs (`../daily_portfolio/ig_cost_check.py`, diagnostic, not pre-registered)
Same pre-registered TREND rules on these 5 markets, 2007–2026, Sharpe (dev / val / holdout / full):
- OANDA CFD costs: 0.21 / 0.61 / 0.30 / 0.31 (≈3.4%/yr at a 10% vol target)
- IG futures-style costs: 0.38 / 0.83 / 0.50 / **0.50** (≈5.4%/yr; costs ≈0.84%/yr)
- No costs at all: 0.47 / 0.90 / 0.58 / 0.58
Caveats: one snapshot of demo spreads (live spreads can differ, widen out of hours); the 5-market selection was made
after earlier diagnostics showed metals/commodities/bonds trended, so the full-period number is partly in-sample;
roll cost assumes one full spread per roll (IG may discount rolls).
