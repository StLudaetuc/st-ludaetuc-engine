# Novel strategy ideas — overnight exploration 31 Oct 2026 (all pre-registered before testing)

| # | Idea (why someone else pays us) | Result | Key numbers (net of IG-style costs) |
|---|---|---|---|
| A | Buy stock indices after a panic day (liquidity provision) | FAIL | 3-day hold: +33 bp/trade 2007–18, −26 bp 2019–22, −4 bp 2023–26 |
| B | Relative-value pairs revert (10 linked pairs) | FAIL | pooled PF 0.70 / 0.83 / 1.11 |
| C | Cross-market lead–lag (slow information) | FAIL | out-of-sample PF 0.76 / 0.74; "leads" were mostly asynchronous-close artefacts |
| D | Overnight drift in stock indices | FAIL | real before costs (+3.2 bp/night, t 2.7, every third of 2015–26) but −2.0 bp after the daily spread |
| E | Vol-managed index holding (collect the premium cheaply) | FAIL | Sharpe below plain buy-and-hold in 2019–22 and 2023–26 |
| F | Gold session drift (Asia vs New York) | FAIL | Asia session −0.2 bp/night, t −0.1 |
| G | Same-complex commodity spreads, unseen pairs (out-of-sample check of B's two good pairs) | FAIL | pooled PF 0.79 / 0.83 / 1.23; B's Brent/WTI and corn/wheat treated as chance |

What survived: risk premia exist **before costs** (index overnight drift; long-run index and gold drift), but any strategy
that pays the retail spread often gives them away. Nothing here is a trading edge.

Files: PREREGISTRATION*.md, ideas_daily.py, ideas_overnight.py, ideas_batch2.py, results_*.json.
