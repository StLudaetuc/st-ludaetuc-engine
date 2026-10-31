"""Frozen settings for the IG demo simulation trial (see ../PLAN.md). Changing anything here after the trial
starts is a protocol change and must be recorded."""

# --- Safety: demo only. The IG client refuses any other base URL. ---
IG_DEMO_BASE = "https://demo-api.ig.com/gateway/deal"
OANDA_PRACTICE_BASE = "https://api-fxpractice.oanda.com/v3"

# --- Trend rules: identical to TREND in research/daily_portfolio/PREREGISTRATION.md ---
LOOKBACKS = (63, 126, 252)      # trading days; signal = mean(sign(return over each))
VOL_SPAN = 60                   # EWMA span for daily-return volatility
VOL_MIN_PERIODS = 40
VOL_TARGET = 0.10               # annual, per instrument, before the 1/sqrt(N) portfolio scaling
BUFFER = 0.25                   # trade only if |target - current| > 25% of |target|
REBALANCE_WEEKDAY = 0           # Monday (scheduler runs ~14:00 UK)

# --- Circuit breaker (stateless): flatten everything if equity < 85% of the trial's starting equity ---
CIRCUIT_BREAKER = 0.85

# --- Roll: move to the next expiry this many days before IG's lastRolloverTime ---
ROLL_DAYS_BEFORE = 5

# --- Execution: marketable limit, fill-or-kill, bounded slippage ---
LIMIT_SLIPPAGE_SPREADS = 2.0    # limit = offer + 2 spreads (buy) / bid - 2 spreads (sell)
MAX_GROSS_LEVERAGE = 3.0        # refuse to place orders if total notional > 3x equity
PRICE_SANITY_TOL = 0.05         # IG futures price must be within 5% of the OANDA signal price

# --- Instruments: OANDA signal series -> IG futures-style market (smallest contract) ---
INSTRUMENTS = {
    "GOLD":   {"oanda": "XAU_USD",    "ig_search": "Gold",     "ig_name": "Gold ($33.20)"},
    "SILVER": {"oanda": "XAG_USD",    "ig_search": "Silver",   "ig_name": "Silver ($10)"},
    "COPPER": {"oanda": "XCU_USD",    "ig_search": "Copper",   "ig_name": "High Grade Copper ($0.50)"},
    "CRUDE":  {"oanda": "WTICO_USD",  "ig_search": "US Crude", "ig_name": "Oil - US Crude ($5)"},
    "TNOTE":  {"oanda": "USB10Y_USD", "ig_search": "T-Note",   "ig_name": "10-Year T-Note Decimalised ($2)"},
}
