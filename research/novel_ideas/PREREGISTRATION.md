# Novel strategy ideas — overnight exploration, pre-registration
Written 31 Oct 2026 before any of these were tested. Research only; no trading.

Andy asked for inventive ideas that are not the usual indicator rules. Each idea below has a reason someone is forced
or paid to trade the other side (a "why"), is tested on development (2007–18), validation (2019–22) and holdout
(2023–26), with costs = 1.5× OANDA spread per leg (IG futures-style stand-in), no funding unless stated.

**Shared PASS rule (per idea; 4 ideas, so a stricter bar):** pooled mean net return > 0 in **all three** periods,
date-clustered t-stat ≥ 2.5 on the full sample, profit factor ≥ 1.2 at 1.5× costs and ≥ 1.05 at 3× costs.
Variants inside an idea are all reported; an idea passes only through the variant named PRIMARY.

## A. Panic-reversal in equity indices (liquidity provision)
Why: after a large one-day fall, forced sellers (margin calls, risk-parity and vol-target de-leveraging) push prices
below fair value; the buyer is paid for providing liquidity. Universe: the 17 stock indices from ../sltp_universe.
Signal: daily return < −2 × (60-day stdev of daily returns). Long at next open, exit at open N days later.
PRIMARY N = 3; also N = 1 and N = 5.

## B. Relative-value pairs (mean reversion of a spread between related markets)
Why: closely linked markets (same commodity, same economy, overlapping index members) can't drift apart for long,
because arbitrageurs and hedgers pull them back. Pairs (fixed now): gold/silver, Brent/WTI, S&P 500/Nasdaq 100,
S&P 500/Dow, DAX/Euro Stoxx 50, US 10y/Bund, US 5y/US 10y, AUD/USD vs NZD/USD, corn/wheat, platinum/palladium.
Spread = log price ratio. z = (spread − 60-day mean) / 60-day stdev. Enter equal-notional, opposite legs at next open
when |z| > 2 (betting on reversion); exit when |z| < 0.5 or after 20 days. Return per trade on one leg's notional,
costs on both legs. PRIMARY = all 10 pairs pooled.

## C. Cross-market lead–lag (information diffuses slowly)
Why: news that hits one market first (e.g. copper, AUD/JPY, oil) reaches related markets with a delay because
investors specialise. Universe: 45 markets from ../sltp_universe; daily close-to-close returns (09:00 New York).
In development only, for every ordered pair (A → B), estimate corr(return_A today, return_B tomorrow); keep the
20 pairs with the largest |t|. Trade B the next day in direction sign(corr) × sign(return_A today), validation and
holdout only. PRIMARY = these 20 pairs, equal weight. (Development is used only for selection, so the test is on
validation + holdout; PASS requires both > 0 and pooled t ≥ 2.5 over validation + holdout.)

## D. Overnight drift in stock indices (who holds risk overnight is paid)
Why: market makers and intraday traders flatten before the close, so whoever carries index risk overnight earns a
premium (documented: Cliff, Cooper & Gulen 2008; Lou, Polk & Skouras 2019). Data: OANDA M30 bid/ask 2015–2026 (the
flow-edges data). US indices (SPX500, NAS100, US30, US2000): long at 16:00 ET, exit at 09:30 ET next trading day.
DE30 17:30→09:00 Frankfurt; UK100 16:30→08:00 London; JP225 15:00→09:00 Tokyo. Fills at bid/ask + 10% of spread per
leg. Carry cost 1 bp per night (futures-style price includes financing). Periods for this idea: thirds of 2015–2026.
PRIMARY = all 7 indices pooled. Control reported: the opposite (intraday) session.
