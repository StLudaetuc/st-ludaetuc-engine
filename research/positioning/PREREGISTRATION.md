# Positioning data as a trading signal — pre-registration
Written 31 Oct 2026 before any positioning data was fetched or looked at. Research only.

Shared rules: weekly decisions, trades at Monday's open, held one week (re-decided every Monday); prices from OANDA daily
mid data (weekday rows, 09:00 New York), costs 1.5× OANDA spread per round trip (3× stress), no funding (futures-style).
Result unit: net return per trade on notional. PASS (per test, PRIMARY variant only): mean > 0 in each of three equal
time thirds of the test's sample, date-clustered t ≥ 2.5, profit factor ≥ 1.2 at 1.5× costs and ≥ 1.05 at 3×.
Two tests are run, so each needs t ≥ 2.5 rather than 2.0.

## Test 2 — OANDA retail position book (contrarian)
Why: retail traders are, on average, on the losing side and crowd into positions late; when most of OANDA's clients are
on one side, the other side should be favoured.
Data: OANDA v20 `positionBook` snapshots (practice API, read-only), one per week at the latest snapshot at or before
Friday 21:00 UTC, for every instrument where OANDA provides the history (expected: major FX pairs, XAU_USD, XAG_USD).
Long share = Σ longCountPercent / (Σ longCountPercent + Σ shortCountPercent) over all buckets.
PRIMARY: long share > 60% → short next week; < 40% → long; otherwise flat.
Variant (reported): continuous — position = −sign(long share − 50%).

## Test 3 — CFTC Commitments of Traders (hedging pressure)
Why: commercial hedgers pay a risk premium to offload risk; when hedgers are unusually net short, longs should earn more
(Bessembinder 1992; de Roon, Nijman & Veld 2000).
Data: CFTC legacy futures-only reports (public), 2007 onwards, for: gold, silver, copper, WTI crude, natural gas, corn,
wheat, soybeans, sugar, S&P 500 e-mini, Nasdaq-100 e-mini, 2y/5y/10y/30y Treasuries, EUR, GBP, JPY, AUD, CAD, CHF, NZD
(mapped to the matching OANDA instrument; currency futures map to the USD pair with the correct sign).
Report date = Tuesday; usable from the following Monday open (published Friday afternoon).
Hedging pressure HP = commercial net position / open interest. Percentile of HP within the previous 156 weeks.
PRIMARY: HP percentile < 20% (hedgers unusually net short) → long; > 80% → short; otherwise flat.
Variant (reported): speculator-crowding contrarian — non-commercial net / OI percentile > 90% → short, < 10% → long.

Interpretation: a PASS is a candidate for the IG demo only after an Oracle review; a FAIL ends the line.

## IG client sentiment (data collection only, no test yet)
A Make scenario records IG client sentiment (long % / short %) twice a day from 31 Oct 2026 for a fixed list of markets.
A test will be pre-registered once there are at least 12 months of data.
