# Which spread-bet markets suit the stop-loss trend strategy? — pre-registration
Written 31 Oct 2026, before any run on markets beyond the original five. Research only.

**Strategy (frozen, from ../sltp_trend):** weekly Monday signal = sign(c0−c63)+sign(c0−c126)+sign(c0−c252) on weekday daily
closes; enter/reverse at Monday open; stop 3×ATR(20); no take profit; exit on stop or weekly flip; risk 1% of equity per trade.

**Universe (45 markets, chosen now by what IG offers as futures/forward spread bets, not by past results).**
OANDA daily mid OHLC 2007–2026 used as stand-ins:
- Metals (5): XAU_USD, XAG_USD, XPT_USD, XPD_USD, XCU_USD
- Energy (3): BCO_USD, WTICO_USD, NATGAS_USD
- Grains/softs (4): CORN_USD, WHEAT_USD, SOYBN_USD, SUGAR_USD
- Stock indices (17): SPX500, NAS100, US30, US2000, UK100, DE30, FR40, EU50, NL25, CH20, ESPIX, JP225_USD, AU200, HK33,
  CHINAH, CN50, SG30
- Bonds (6): DE10YB, UK10YB, USB02Y, USB05Y, USB10Y, USB30Y
- FX (10): EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, USD_CHF, NZD_USD, EUR_GBP, EUR_JPY, GBP_JPY
Excluded: metal crosses (duplicates of gold/silver), JP225Y (duplicate), other FX crosses and EM currencies.

**Costs:** IG futures-style spread bets have wider spreads than OANDA spot, so base cost = **1.5×** the OANDA spread at
entry per round trip, plus 1.5× spread per contract roll while held (energy 12/yr, grains 5/yr, everything else 4/yr).
No overnight funding. Stress: 3×.

**Questions and tests**
1. Per market: profit factor and average R in development (2007–18), validation (2019–22) and holdout (2023–26).
2. Whole universe as one portfolio (all 45, 0.5% and 1% risk per trade): profit factor per period, CAGR, max drawdown.
3. **Does picking markets help?** Rank markets on **development average R only**; take the top 10 and the bottom 10;
   compare their validation+holdout results with the all-market portfolio. Also a diversified pick: the best 2 per asset
   class on development. Picking "works" only if the top-10 beats all-markets in **both** validation and holdout.
4. £10k fit: the £/point stake each market needs for £100 risk at a 3×ATR stop (to compare with IG minimum stakes once
   the spread-bet demo exists).

**Interpretation rule:** a market is only recommended if it is (a) positive in development and validation, and (b) fits
the £10k stake limits; holdout results are reported but cannot be used to add markets.
