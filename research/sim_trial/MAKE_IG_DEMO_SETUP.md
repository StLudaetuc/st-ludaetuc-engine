# IG demo automation in Make (built 30 Oct 2026, demo account only)

| Item | ID | Purpose |
|---|---|---|
| Scenario "IG Demo – Execute Signal (St Ludaetuc)" | 10075127 | Webhook in → IG demo login (v3) → front-month futures contract → open position → order (limit, fill-or-kill, ≤2 spreads) → confirmation → log. Sequential, de-duplicated by `signal_id`. |
| Webhook "IG Demo – Execute Signal (St Ludaetuc)" | 4478291 | `https://hook.eu2.make.com/quabxagsofrwcs2y0p5ctgj7nq6jyt0p` |
| Scenario "IG Demo – Weekly Trend Signals (St Ludaetuc)" | 10075136 | Mondays 14:00 (Make account time zone). OANDA daily closes → score = sign(63) + sign(126) + sign(252) → `long` / `short` per market → Execute webhook. |
| Scenario "IG Demo – Send Test Signal (St Ludaetuc)" | 10075131 | Run on demand with inputs `instrument`, `action`, optional `size`, `signal_id`. |
| Data store `stl_ig_demo_trades` | 203924 (structure 652646) | One record per signal: prices, spread, order, IG deal status/reason, fill level, deal ID. |
| Credentials | data store 190220 | `ig_demo_api_key`, `ig_demo_username`, `ig_demo_password` (value in `bearer_token`). |

Signal payload: `{"signal_id": "...", "instrument": "GOLD|SILVER|COPPER|CRUDE|TNOTE", "action": "long|short|close|buy|sell", "size": optional, "source": "..."}`.
`long`/`short` set the target position (± minimum deal size unless `size` given); `close` sets it to zero; `buy`/`sell` add one unit.

Known limits (plumbing test, not the reviewed pipeline): minimum-size positions (not volatility-weighted); no contract-roll handling
(DEC-26 contracts roll ~19–25 Nov); trend lookbacks count OANDA weekend rows; the Oracle's findings (research/sim_trial, CHANGES REQUIRED)
remain open. URLs are hard-coded to demo-api.ig.com.
