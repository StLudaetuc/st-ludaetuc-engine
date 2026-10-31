# Positioning tests: results (31 Oct 2026)

Pre-registered in `PREREGISTRATION.md` (commit 0b91c6b) before any data was fetched. Code: `positioning_tests.py`.
Full numbers: `results.json`. Both PRIMARY tests **FAIL**, so both lines are closed.

| Test | Trades | Mean/trade (1.5× costs) | t | PF 1.5× / 3× | Thirds (bp) | Verdict |
|---|---|---|---|---|---|---|
| 3. COT hedging pressure (PRIMARY) | 10,550 | −1.1 bp | −0.69 | 0.99 / 0.97 | +7.1 / −11.9 / +0.9 | **FAIL** |
| 3. COT speculator contrarian (variant) | 6,588 | −1.4 bp | −0.02 | 0.99 / 0.96 | −6.8 / +9.4 / −6.3 | fail |
| 2. OANDA position book >60%/<40% (PRIMARY) | 4,177 | −9.5 bp | −2.49 | 0.85 / 0.84 | −6.8 / −6.3 / −12.8 | **FAIL** |
| 2. OANDA position book continuous (variant) | 7,987 | −6.6 bp | −2.47 | 0.88 / 0.87 | −2.3 / −7.9 / −9.6 | fail |

## What it means
- **COT (22 futures markets, 2007–2026):** in the CFTC data, hedger positioning carries no usable weekly risk premium
  after costs. Per-market results go both ways, roughly at random (corn +24bp, wheat −44bp), which is what you'd expect
  from noise.
- **OANDA position book (14 FX pairs plus gold and silver, 2017–2026):** fading the retail crowd lost money. Diagnostic,
  not pre-registered: nearly all of the loss is gold and silver. OANDA clients were net long those in essentially every
  week (the rule was short gold 100% of weeks, silver 95%), so the "contrarian" rule was really a standing short in a bull
  market. With the metals removed, FX alone is flat: −1.8bp, t 0.34, PF 0.96. So there's no edge in either direction.
  Following the crowd would only have been a disguised long-metals bet, and it wasn't pre-registered, so it isn't
  claimed.

## Data notes (nothing filled in or invented)
- **COT:** CFTC legacy futures-only, public Socrata API (`publicreporting.cftc.gov`, dataset 6dca-aqww), fetched through
  a temporary Make HTTP scenario because this container can't reach the host. 22 codes, 2004–2026; the 156-week warm-up
  means signals start in 2007. Tuesday report → position from the following Monday open.
- **Position book:** OANDA v20 `positionBook` (practice API, read-only), Friday 20:00 UTC snapshot. That is the latest one
  at or before 21:00, since 21:00 itself doesn't exist. Coverage is 2017-01-20 to 2026-10-30, about 502 weeks per
  instrument.
  - The same 11 weeks returned 404 for every instrument and were skipped.
  - GBP_CHF had 14 early-2017 snapshots with long = short = 0; these were treated as missing.
  - Three weeks in Jul–Aug 2026 repeat identical long/short values (probably a stale OANDA snapshot). They were left as
    delivered.
- **Prices and costs:** OANDA daily bid/ask (as in earlier research), Monday open to next Monday open. Costs are half the
  spread × multiplier per unit of position change, plus rolls (energy 12/yr, grains 5/yr, others 4/yr) pro rata while
  held.

## Still running
IG client sentiment is being recorded twice a day from 31 Oct 2026; see `IG_SENTIMENT_RECORDER.md`. It can't be tested
until about 12 months of data exist, and that test will be pre-registered first. Don't expect much: OANDA's equivalent
retail data just failed here.
