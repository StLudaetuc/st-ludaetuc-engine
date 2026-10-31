# First daily run discarded (data bug)
The first run, at 00:30Z on 2026-10-29, used 17:00 New York-aligned daily candles. It was invalid for two reasons:
1. The 5,000-candle request ended in April 2024, because pre-2014 data includes Saturday candles. The "holdout" therefore held only four months.
2. Every fill was at the 17:00 rollover open, when spreads are widest; EUR/USD's median was 3.6 pips.

The data is being re-fetched with `dailyAlignment=9` (09:00 New York) in two chunks to reach October 2026. The pre-registered rules are unchanged.
