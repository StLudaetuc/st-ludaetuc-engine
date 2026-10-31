# IG client sentiment recorder (data collection only)

Started 31 Oct 2026. Nothing trades from this; it only records data so a test can be pre-registered in about 12 months.

| Item | Value |
|---|---|
| Make scenario | 10075822 "IG client sentiment recorder (twice daily, research only)", active |
| Schedule | every 12 hours (first run 2026-10-31 03:04 UTC, so roughly 03:00 and 15:00 UTC) |
| Data store | `stl_ig_client_sentiment` (id 203972, structure 652716), 1 MB cap (Make plan limit) |
| Source | IG **demo** REST API, `GET /clientsentiment/{marketId}` (v1), logged in with the demo credentials in data store 190220 |
| API use | 1 login + 29 calls per run, 3 s apart (well inside the demo allowance) |

Record key = snapshot time `YYYY-MM-DDTHH:mm` (UTC). Fields: `ts_utc`, `source`, and `rows`, which holds
`marketId:httpStatus:long%:short%` for every market, joined by `|`. Failed calls are kept, with their status and blank
percentages, and are never filled in.

Markets (29): EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURGBP EURJPY GBPJPY · GC (gold) SI (silver) PL (platinum)
HG (copper) · CL (US crude) LCO (Brent) NG (natural gas) · C (corn) W (wheat) S (soybeans) SB (sugar) · US500 USTECH WALL
FT100 DE30 JP225 AU200 · 10YRTND (10-year T-note). The marketIds came from IG's own market details (Make discovery run,
31 Oct 2026). Bitcoin was left out because the search didn't return a clean Bitcoin market.

First snapshot (2026-10-31 03:04 UTC): all 29 returned 200. Retail is net long almost everywhere, e.g. gold 77/23,
platinum 84/16, FTSE 75/25.

Capacity: each record is about 0.6 KB, so about 1.2 KB a day. 1 MB lasts about 2 years. Export it to the repo or a sheet
before then (or raise the cap if the plan allows).

To stop it: deactivate scenario 10075822. The data store stays.
