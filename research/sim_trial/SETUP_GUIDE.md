# Demo account set-up: IG and NinjaTrader (for Andy)

Account opening needs your identity and your own login details, so these steps are yours. Nothing here involves real money.

## 1. IG — spread betting demo + API key
1. Go to ig.com/uk and open a **spread betting demo account**. IG says the API demo must use the **same email address as a live IG account**, so you may also need to create (not fund) a live account to generate an API key.
2. Log in to the live account → My IG → **Settings → API keys** → create a key for the **demo** environment.
3. Note four things: API key, demo username, demo password, demo account ID (shown in My IG).
4. In the demo platform, check the **"futures"/"forward"** markets for: Spot Gold futures, Silver, High Grade Copper, US Crude (WTI), US 10-Year T-Note. Note the market names; I will find their API codes ("epics").

## 2. NinjaTrader — simulation + API
1. Go to ninjatrader.com (UK site, launched 27 Oct 2026) and create a free account with the **simulator**.
2. Before going further, ask NinjaTrader support (copy-paste):
   > "I'm a UK resident. Can I use the Tradovate REST API with a simulation (demo) account? What does API access cost for UK clients, and which plan and commission rates apply to CME micro futures (MGC, SIL, MHG, MCL, 10-year Treasury)?"
3. If API access is available: in the Tradovate/NinjaTrader web app, open the **API Access** add-on and generate API credentials (it will give a **cid** and **secret**). Note your demo username and password.

## 3. Storing credentials (please do not paste them into chat)
Add them to the existing Make data store (190220, where `oanda_practice_token` lives) as new records:
- `ig_demo_api_key`, `ig_demo_username`, `ig_demo_password`, `ig_demo_account_id`
- `tradovate_demo_username`, `tradovate_demo_password`, `tradovate_demo_cid`, `tradovate_demo_secret`

Then tell me which ones are in place. I will only ever point them at the **demo** addresses (demo-api.ig.com, demo.tradovateapi.com).
