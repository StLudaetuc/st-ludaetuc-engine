# Novel ideas — batch 2 (pre-registered 31 Oct 2026, after batch 1 results, before any batch 2 test)

Batch 1 (A–D) all FAILED. One fact survived: stock indices drift up overnight before costs (D gross +3.2 bp/night,
positive in every third of 2015–26), but paying the spread every day kills it. Batch 2 asks how to collect such premia
cheaply, plus one new intraday pattern. Same PASS bar as batch 1 unless stated.

## E. Volatility-managed index holding (collect the equity premium without daily spreads)
Why: index returns don't rise when volatility rises, but losses cluster in high-volatility spells (Moreira & Muir 2017).
Holding the index continuously collects the overnight and equity premium; cutting exposure when recent volatility is
high should keep most of the return with smaller drawdowns.
Rule: long only. Weekly (Monday open) weight = min(2, 0.15 / annualised 20-day realised vol) in each index; trade only
if the weight changes by more than 25%. Indices: SPX500, NAS100, US30, DE30, UK100, JP225, AU200, EU50 (equal risk,
divided by √N). Futures-style: costs 1.5× OANDA spread on traded notional + rolls (4/yr), **plus** the financing a
futures price embeds (USD/other cash rate, OECD) which holding-period returns must earn above.
Compare with buy-and-hold of the same indices (weight fixed so average exposure matches).
PASS: Sharpe of E > Sharpe of buy-and-hold in **each** of dev/val/holdout, E's excess return over cash > 0 in each
period, and max drawdown of E < max drawdown of buy-and-hold over the full period.

## F. Gold session drift (who trades gold, when)
Why: physical demand from Asia and Western futures selling occur in different hours; studies (e.g. 2000s–2010s) found
gold tends to rise in Asian/European hours and fall in New York hours.
Data: OANDA XAU_USD M15 bid/ask 2021-10 → 2026-10 (the bb15 file). Sessions (UTC): Asia 23:00–07:00, London 07:00–13:00,
New York 13:00–21:00. Rule: long gold for the Asia session only (enter 23:00, exit 07:00 UTC), every weekday.
Costs: buy at ask, sell at bid, plus 10% of spread per leg (spread × 1.5 stress also reported). Overnight carry 0.6 bp
per night for a spot-gold stand-in. Periods: thirds of the sample. PASS: mean > 0 in each third, t ≥ 2.5, PF ≥ 1.2,
PF ≥ 1.05 at 3× spread. Other sessions reported as controls only (not tradable conclusions).
