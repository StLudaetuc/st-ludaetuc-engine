# Simulation trial plan: metals/commodities/bond trend on IG demo and NinjaTrader simulator

**Status: DRAFT for Andy and The Oracle. Rules frozen at this commit; no changes once the trial starts. Demo/simulated money only.**

## Purpose (be clear what this can and cannot show)
- **Can show:** real spreads, commissions, overnight/roll costs, fill quality, and whether the Make → engine → broker chain runs reliably each week on each venue.
- **Cannot show:** that the strategy is profitable. With a Sharpe of about 0.2–0.45, even 6–12 months of results are mostly noise. The edge evidence is the 2007–2026 backtest and decades of published research; the trial checks that the real costs match the backtest's assumptions.

## System (identical to the pre-registered TREND in `../daily_portfolio/PREREGISTRATION.md`)
- Signal: average of sign(return over 63, 126, 252 trading days), in [−1, +1], from daily closes.
- Size: 10% annual volatility target per instrument (60-day EWMA), portfolio scaled by 1/√N.
- Rebalance weekly, Monday 14:00 UK (liquid hours), on the previous close's signal; trade only if the target moved >25%.
- No stop losses (the system exits by signal); a portfolio-level circuit breaker is the only override (see below).

## Instruments
| | IG demo (spread bet, futures/forward market) | NinjaTrader sim (CME micro) |
|---|---|---|
| Gold | Gold futures | MGC (10 oz) |
| Silver | Silver futures | SIL (1,000 oz) |
| Copper | Copper futures | MHG (2,500 lb) |
| Crude oil | US Crude (WTI) futures | MCL (100 bbl) |
| US 10-year bond | US 10-Year T-Note futures | 10-year Treasury (micro or standard; contract to be confirmed) |

Notional per instrument = signal × 10% / vol × equity / √5. IG stakes are £/point and can be fractional; CME contracts are whole numbers, rounded to nearest (a target below 0.5 contracts = flat). Simulated equity: IG demo default balance; NinjaTrader sim set to **$100,000** if the platform allows (whole contracts make smaller balances unrepresentative).

## Measurements (logged every rebalance)
Signal, target, order, fill price, mid at order time, spread paid, commission, overnight/roll charges, and any API error. Weekly comparison against the backtest's expected cost and P&L for the same week.

## Pass / fail for each venue at 6 months
- **Operations:** ≤ 1 missed or failed rebalance per month; zero orders ever sent to a live address.
- **Costs:** measured all-in cost ≤ 1.5× the backtest's futures-venue assumption. Above 2× = venue fails.
- **Tracking:** realised position path matches the backtest's targets (after rounding) on ≥ 95% of rebalances.
- P&L is reported but is **not** a pass criterion (too short to be meaningful).

## Safety
- Demo/simulation addresses only; the adapter refuses any live base URL in code.
- Circuit breaker: if simulated equity falls 15% from its peak, flatten and notify Andy (rehearses the live safeguard).
- Separate from the production OANDA engine and Make scenarios; nothing in production changes.

## After 6 months
A written report → The Oracle's independent review → Andy decides whether a small live allocation is justified, on which venue.
