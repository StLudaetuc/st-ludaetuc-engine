# Flow-based edges: pre-registration (written before any intraday data was fetched)

**Research only.** No live, Pine, Make-production, OANDA-order, Render or database changes. The Oracle reviews independently; nothing goes live without Andy's approval and a SHADOW trial.

## Why these hypotheses
Every indicator-rule test so far (GBP/USD 15m, 3m–1h, 10s/30s/1m, daily trend/carry) found a before-cost edge of about zero. The remaining candidates that a small account can plausibly use are **structural flows**: someone trades for a reason other than price prediction (hedging, rebalancing, month-end cash flows) and the timing is predictable. All three below come from peer-reviewed literature, were chosen **before** looking at OANDA intraday data, and all trades except H2 close the same day, so OANDA's overnight financing markup does not apply.

## Data
OANDA M30 bid/ask candles from 2015-01-01 to the latest complete bar: SPX500_USD, NAS100_USD, US30_USD, US2000_USD, DE30_EUR, UK100_GBP, JP225_USD, AU200_AUD, EUR_USD, GBP_USD, USD_JPY, AUD_USD. Prices at a time T are the **open** of the M30 bar starting at T (bid/ask). Missing bars → that trade is skipped (no fill-forward).

## Costs (1×)
Buy at ask, sell at bid on both legs, plus slippage of 10% of the prevailing spread per leg. Stress: 2× (spread and slippage doubled). H2 also pays overnight financing: (OECD 3-month rate of the index currency + 2.5% OANDA markup) / 365 per night held, on notional.

## Hypotheses
**H1 US intraday momentum** (Gao, Han, Li & Zhou 2018, JFE; Baltussen, Da, Lammers & Martens 2021, JFE — gamma-hedging flows into the close).
Instruments SPX500_USD, NAS100_USD, US30_USD, US2000_USD. Each US trading day: r = mid(15:30 ET) / mid(16:00 ET previous trading day) − 1. Enter sign(r) at 15:30 ET, exit 16:00 ET. Every day traded.

**H2 Turn of the month** (Lakonishok & Smidt 1988; McConnell & Xu 2008).
Instruments SPX500_USD, NAS100_USD, US30_USD, US2000_USD. Long at 15:30 ET on the last trading day of each month, exit 15:30 ET on the third trading day of the next month. "Trading day" = a weekday on which the 15:30 ET bar exists. Additional requirement for PASS: mean net return must exceed the mean net return of all other same-length long windows (to show a calendar effect, not just equity drift).

**H3 Month-end FX hedge rebalancing** (Melvin & Prins 2015, J. Banking & Finance).
Pairs and foreign index: EUR_USD–DE30_EUR, GBP_USD–UK100_GBP, USD_JPY–JP225_USD, AUD_USD–AU200_AUD. On the last trading day of the month: m = (foreign index return in USD) − (SPX500 return), both from 10:00 London on the previous month's last trading day to 10:00 London today. If m > 0, buy USD against the foreign currency at 10:00 London and exit at 16:00 London (the WM/R fix); if m < 0, sell USD. Index returns are converted to USD with the pair's mid at the same times (JP225_USD is already USD).

## Statistics and PASS criteria (each hypothesis pooled over its instruments)
Unit = net % return per trade (notional). Standard errors clustered by date.
1. t-statistic of mean net return at 1× ≥ 2.5 (≈ Bonferroni for 3 hypotheses at 5% two-sided, rounded up).
2. Profit factor ≥ 1.15 at 1× and ≥ 1.05 at 2×.
3. Mean net return > 0 in each chronological third of the sample.
4. Mean net return > 0 for at least half the instruments.
5. (H2 only) beats the all-other-windows baseline.
All must hold. Post-publication periods are reported separately (H1 from 2021, H3 entire sample is post-publication). No parameter will be tuned; any variant tried after seeing results is labelled EXPLORATORY and cannot PASS.

## Also reported (diagnostic, not a test)
Venue diagnostic for the daily metals trend (from `../daily_portfolio`): same rules, OANDA spread kept, financing markup removed — an estimate of what a futures venue could keep. Not an OANDA-tradable result.
