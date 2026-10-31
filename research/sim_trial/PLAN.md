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

Notional per instrument = signal × 10% / vol × equity / √N (N = instruments with data). Signals use OANDA practice daily mid closes (09:00 New York alignment), the same series as the backtest; execution is on the venue.

**IG (amended 30 Oct 2026 after reading live contract specs):** futures-style CFD contracts are large (gold ≈ $130k per contract, minimum 0.5; others $21k–$52k, minimum 1). Size = weight × equity ÷ contract notional, rounded to 0.01; below the minimum = flat. At £10k demo equity no market can hold a position, so the trial needs either the CFD demo raised to ≈ £500k virtual funds or a spread-bet demo (£ per point). Andy chooses before the first run. NinjaTrader: whole CME micro contracts, simulator balance to be confirmed (trial status pending NinjaTrader's answers on API access and the 14-day simulator limit).

## Measurements (logged every rebalance)
Signal, target, order, fill price, mid at order time, spread paid, commission, overnight/roll charges, and any API error. Weekly comparison against the backtest's expected cost and P&L for the same week.

## Pass / fail for each venue at 6 months
- **Operations:** ≤ 1 missed or failed rebalance per month; zero orders ever sent to a live address.
- **Costs:** measured all-in cost ≤ 1.5× the backtest's futures-venue assumption. Above 2× = venue fails.
- **Tracking:** realised position path matches the backtest's targets (after rounding) on ≥ 95% of rebalances.
- P&L is reported but is **not** a pass criterion (too short to be meaningful).

## Safety
- Demo/simulation addresses only; the adapter refuses any live base URL in code and checks the logged-in account ID. Dry run unless explicitly enabled.
- Circuit breaker: if equity falls below 85% of the trial's starting equity, flatten and keep flat until Andy resets it (stateless, so it cannot be lost between runs).
- Orders: marketable limit, fill-or-kill, at most 2 spreads from the current price. Contract rolls: positions move to the next expiry 5 days before IG's last rollover time.
- Refuses to trade if the venue price is more than 5% from the OANDA price, or if total notional would exceed 3× equity.
- Separate from the production OANDA engine and Make scenarios; nothing in production changes.

## After 6 months
A written report → The Oracle's independent review → Andy decides whether a small live allocation is justified, on which venue.
