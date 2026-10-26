# MANUS ENGINE v4 - FULL SOURCE (PROPOSAL, research/SHADOW/DEMO only, LIVE refused in code). 36/36 tests pass.

# ===== FILE: README.md =====
# Manus Engine v4.0.0-research (SHADOW/RESEARCH/DEMO only)
Stdlib Python. LIVE mode raises LiveModeRefused. No network, no DB, nothing sends orders.
- Package: manus_v4/ (instruments, precision, fx, sizing, analysis, plan, portfolio, safety, oanda_io, ledger, legacy_v3, engine)
- Tests: `python3 -m unittest tests.test_v4` -> 36/36 OK (23 Oct 2026)
- Backtest: `python3 backtest_gbpusd.py` -> backtest_results.json
- Instrument values are REPORTED_UNVERIFIED; verify against the OANDA instruments endpoint before DEMO.
- Coverage map, weaknesses and backtest are in the CEO report Page (section "Manus v4 build").
Backtest GBP/USD 2m, 1.5 pip cost, SL-first: baseline -0.293R (n2036); v4 gates -0.202R (n475); v4+EV -0.162R (n74). No profitable variant.

# ===== FILE: manus_v4/__init__.py =====
"""St Ludaetuc Manus Engine v4 - RESEARCH / DEMO / SHADOW build.

Pure-Python (standard library only). Never places orders, never calls OANDA,
Postgres or any network service. LIVE mode is refused by design.
"""
ENGINE_VERSION = "4.0.0-research"
RULESET_VERSION = "manus_ruleset_2026_10_v4_0_proposal"
CALIBRATION_VERSION = "heuristic_uncalibrated_v4"  # nothing is calibrated yet

# ===== FILE: manus_v4/instruments.py =====
"""Instrument registry: every OANDA FX pair + XAU_USD / XAG_USD.

PROVENANCE: every value here is REPORTED (from public OANDA documentation /
the conventions OANDA's v20 `GET /v3/accounts/{id}/instruments` endpoint is
documented to return) or ESTIMATED (typical spreads, session hours). NONE of
it has been read from a live account. Before any DEMO run, regenerate this
table from the live instruments endpoint (pipLocation, displayPrecision,
tradeUnitsPrecision, minimumTradeSize, maximumOrderUnits, marginRate,
minimumTrailingStopDistance) and diff it against these values; the live
values always win. Availability differs by OANDA division (e.g. OANDA
Europe / UK may not offer every pair below, notably TRY / CNH / some
exotics), so `tradable` must also be taken from that endpoint.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

PROVENANCE = "REPORTED_PUBLIC_DOCS_UNVERIFIED_AGAINST_LIVE_INSTRUMENTS_ENDPOINT"

# UK/ESMA retail leverage caps (REPORTED): majors 30:1, other FX 20:1,
# gold 20:1, other commodities incl. silver 10:1. OANDA's per-account
# marginRate may be higher - the live value wins.
MAJOR_CCYS = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD"}

# Quote currencies whose pip is the 2nd decimal (REPORTED OANDA convention).
_TWO_DP_PIP_QUOTES = {"JPY", "HUF", "THB"}

_FX_PAIRS = [
    "AUD_CAD","AUD_CHF","AUD_HKD","AUD_JPY","AUD_NZD","AUD_SGD","AUD_USD",
    "CAD_CHF","CAD_HKD","CAD_JPY","CAD_SGD","CHF_HKD","CHF_JPY","CHF_ZAR",
    "EUR_AUD","EUR_CAD","EUR_CHF","EUR_CZK","EUR_DKK","EUR_GBP","EUR_HKD",
    "EUR_HUF","EUR_JPY","EUR_NOK","EUR_NZD","EUR_PLN","EUR_SEK","EUR_SGD",
    "EUR_TRY","EUR_USD","EUR_ZAR","GBP_AUD","GBP_CAD","GBP_CHF","GBP_HKD",
    "GBP_JPY","GBP_NZD","GBP_PLN","GBP_SGD","GBP_USD","GBP_ZAR","HKD_JPY",
    "NZD_CAD","NZD_CHF","NZD_HKD","NZD_JPY","NZD_SGD","NZD_USD","SGD_CHF",
    "SGD_JPY","TRY_JPY","USD_CAD","USD_CHF","USD_CNH","USD_CZK","USD_DKK",
    "USD_HKD","USD_HUF","USD_JPY","USD_MXN","USD_NOK","USD_PLN","USD_SEK",
    "USD_SGD","USD_THB","USD_TRY","USD_ZAR","ZAR_JPY",
]

# ESTIMATED typical OANDA spreads in pips (standard account, liquid hours).
_TYPICAL_SPREAD_PIPS = {
    "EUR_USD":1.0,"GBP_USD":1.4,"USD_JPY":1.2,"AUD_USD":1.3,"USD_CAD":1.8,
    "USD_CHF":1.6,"NZD_USD":1.8,"EUR_GBP":1.4,"EUR_JPY":1.8,"GBP_JPY":2.8,
    "EUR_CHF":1.8,"AUD_JPY":1.9,"EUR_AUD":2.2,"GBP_CHF":3.0,"GBP_AUD":3.0,
    "EUR_CAD":2.5,"GBP_CAD":3.5,"CAD_JPY":2.2,"CHF_JPY":2.5,"NZD_JPY":2.4,
    "AUD_NZD":2.6,"AUD_CAD":2.2,"AUD_CHF":2.2,"NZD_CAD":2.8,"NZD_CHF":2.8,
    "CAD_CHF":2.4,"GBP_NZD":4.5,"EUR_NZD":3.8,
}
_DEFAULT_MINOR_SPREAD = 4.0
_DEFAULT_EXOTIC_SPREAD = 30.0
_EXOTIC_CCYS = {"TRY","ZAR","MXN","HUF","PLN","CZK","SEK","NOK","DKK","HKD","SGD","CNH","THB"}

# Session windows in UTC hours [start, end). ESTIMATED liquidity windows.
SESSIONS_UTC = {
    "asia": (0, 7), "london": (7, 12), "overlap_london_new_york": (12, 16),
    "new_york": (16, 21), "rollover": (21, 22), "overnight": (22, 24),
}
_CCY_HOME_SESSIONS = {
    "JPY": ("asia","london"), "AUD": ("asia","london"), "NZD": ("asia",),
    "SGD": ("asia",), "HKD": ("asia",), "CNH": ("asia",), "THB": ("asia",),
    "EUR": ("london","overlap_london_new_york"), "GBP": ("london","overlap_london_new_york"),
    "CHF": ("london","overlap_london_new_york"), "SEK": ("london",), "NOK": ("london",),
    "DKK": ("london",), "PLN": ("london",), "CZK": ("london",), "HUF": ("london",),
    "TRY": ("london",), "ZAR": ("london","overlap_london_new_york"),
    "USD": ("overlap_london_new_york","new_york"), "CAD": ("overlap_london_new_york","new_york"),
    "MXN": ("overlap_london_new_york","new_york"), "XAU": ("london","overlap_london_new_york","new_york"),
    "XAG": ("london","overlap_london_new_york","new_york"),
}

# Correlation / exposure groups used by the portfolio layer (heuristic).
CORRELATION_GROUPS = {
    "usd_bloc_short_usd": {"EUR_USD","GBP_USD","AUD_USD","NZD_USD","XAU_USD","XAG_USD"},
    "commodity_fx": {"AUD_USD","NZD_USD","USD_CAD","AUD_NZD","AUD_CAD","NZD_CAD"},
    "jpy_crosses": {p for p in _FX_PAIRS if p.endswith("_JPY")},
    "precious_metals": {"XAU_USD","XAG_USD"},
    "europe": {"EUR_USD","GBP_USD","EUR_GBP","EUR_CHF","GBP_CHF","USD_CHF"},
}


@dataclass(frozen=True)
class Instrument:
    name: str                 # OANDA code, e.g. GBP_USD
    base: str
    quote: str
    kind: str                 # CURRENCY | METAL
    asset_class: str          # fx_major | fx_minor | fx_exotic | metal
    pip_location: int         # pip = 10**pip_location
    display_precision: int    # max price decimals OANDA accepts
    trade_units_precision: int
    minimum_trade_size: float
    maximum_order_units: float
    margin_rate: float
    min_trailing_stop_distance: float   # price units
    typical_spread_pips: float
    max_spread_pips: float              # hard gate
    min_stop_pips: float                # stop floor (fixes the tight-stop problem)
    atr_stop_mult: float
    default_rr: float
    max_risk_percent: float
    engine_max_units: float             # engine cap (<= maximum_order_units)
    structure_buffer_pips: float
    preferred_sessions: Tuple[str, ...]
    blocked_sessions: Tuple[str, ...] = ("unknown",)
    tradable: bool = True               # must be confirmed from the live endpoint
    provenance: str = PROVENANCE

    @property
    def pip(self) -> float:
        return 10.0 ** self.pip_location

    @property
    def tick(self) -> float:
        return 10.0 ** (-self.display_precision)

    @property
    def symbol(self) -> str:
        return self.name.replace("_", "")

    @property
    def tradingview_code(self) -> str:
        return self.symbol


def _fx(name: str) -> Instrument:
    base, quote = name.split("_")
    fca_major = base in MAJOR_CCYS and quote in MAJOR_CCYS
    if base in _EXOTIC_CCYS or quote in _EXOTIC_CCYS:
        cls = "fx_exotic"
    elif fca_major or name in ("AUD_USD", "NZD_USD"):
        cls = "fx_major"
    else:
        cls = "fx_minor"
    margin = 1 / 30 if fca_major else 0.05
    pl = -2 if quote in _TWO_DP_PIP_QUOTES else -4
    dp = 3 if pl == -2 else 5
    spread = _TYPICAL_SPREAD_PIPS.get(name, _DEFAULT_EXOTIC_SPREAD if cls == "fx_exotic" else _DEFAULT_MINOR_SPREAD)
    min_stop = max(8.0, round(spread * 7.5, 1))  # spread must be <= ~13% of the stop
    prefs = tuple(dict.fromkeys(_CCY_HOME_SESSIONS.get(base, ()) + _CCY_HOME_SESSIONS.get(quote, ())))
    prefs = tuple(s for s in prefs if s != "overnight") or ("london", "overlap_london_new_york")
    return Instrument(
        name=name, base=base, quote=quote, kind="CURRENCY", asset_class=cls,
        pip_location=pl, display_precision=dp, trade_units_precision=0,
        minimum_trade_size=1, maximum_order_units=100_000_000, margin_rate=round(margin, 4),
        min_trailing_stop_distance=5 * (10.0 ** pl),
        typical_spread_pips=spread, max_spread_pips=round(spread * 2.5, 1),
        min_stop_pips=min_stop, atr_stop_mult=1.5, default_rr=1.6,
        max_risk_percent=0.50 if cls != "fx_exotic" else 0.25,
        engine_max_units=200_000 if cls != "fx_exotic" else 50_000,
        structure_buffer_pips=0.6 if cls != "fx_exotic" else 5.0,
        preferred_sessions=prefs, tradable=cls != "fx_exotic" or name in ("USD_ZAR","USD_MXN","EUR_PLN","USD_SEK","USD_NOK"),
    )


def _build() -> Dict[str, Instrument]:
    reg = {n: _fx(n) for n in _FX_PAIRS}
    # Metals (REPORTED): XAU_USD pipLocation -2, displayPrecision 3; XAG_USD
    # pipLocation -4, displayPrecision 5; units = troy ounces, whole units.
    # NOTE: old engine v3.11 rounded XAU to 0.01 and XAG to 0.001 (coarser -
    # still valid prices); its pip_size 0.1 (XAU) / 0.01 (XAG) is a "trader
    # pip", not OANDA's pipLocation. v4 keeps OANDA's pip and expresses gold
    # stop floors in OANDA pips (1 pip = 0.01 USD) to avoid that ambiguity.
    reg["XAU_USD"] = Instrument(
        name="XAU_USD", base="XAU", quote="USD", kind="METAL", asset_class="metal",
        pip_location=-2, display_precision=3, trade_units_precision=0,
        minimum_trade_size=1, maximum_order_units=100_000, margin_rate=0.05,
        min_trailing_stop_distance=0.05, typical_spread_pips=35.0, max_spread_pips=80.0,
        min_stop_pips=250.0,  # 2.50 USD
        atr_stop_mult=1.6, default_rr=1.7, max_risk_percent=0.35, engine_max_units=500,
        structure_buffer_pips=10.0, preferred_sessions=("london","overlap_london_new_york","new_york"))
    reg["XAG_USD"] = Instrument(
        name="XAG_USD", base="XAG", quote="USD", kind="METAL", asset_class="metal",
        pip_location=-4, display_precision=5, trade_units_precision=0,
        minimum_trade_size=1, maximum_order_units=1_000_000, margin_rate=0.10,
        min_trailing_stop_distance=0.005, typical_spread_pips=250.0, max_spread_pips=600.0,
        min_stop_pips=1500.0,  # 0.15 USD
        atr_stop_mult=1.6, default_rr=1.7, max_risk_percent=0.35, engine_max_units=5000,
        structure_buffer_pips=50.0, preferred_sessions=("london","overlap_london_new_york","new_york"))
    return reg


REGISTRY: Dict[str, Instrument] = _build()

# Old-engine per-pair overrides carried forward verbatim where they differ
# (spread/slippage limits were in price units in v3.11).
LEGACY_V3_LIMITS = {
    "GBP_USD": dict(max_spread_price=0.00025, max_slippage_price=0.00012, sl_atr_mult=1.50, default_rr=1.60),
    "EUR_USD": dict(max_spread_price=0.00020, max_slippage_price=0.00010, sl_atr_mult=1.40, default_rr=1.55),
    "AUD_USD": dict(max_spread_price=0.00025, max_slippage_price=0.00012, sl_atr_mult=1.45, default_rr=1.55),
    "USD_CAD": dict(max_spread_price=0.00030, max_slippage_price=0.00015, sl_atr_mult=1.45, default_rr=1.55),
    "USD_JPY": dict(max_spread_price=0.030, max_slippage_price=0.015, sl_atr_mult=1.45, default_rr=1.55),
    "XAU_USD": dict(max_spread_price=0.60, max_slippage_price=0.30, sl_atr_mult=1.60, default_rr=1.70),
    "XAG_USD": dict(max_spread_price=0.030, max_slippage_price=0.015, sl_atr_mult=1.60, default_rr=1.70),
}


def expected_slippage_price(inst: Instrument) -> float:
    """ESTIMATED expected slippage for cost-in-R (25% of typical spread)."""
    return inst.typical_spread_pips * 0.25 * inst.pip


def max_slippage_price(inst: Instrument) -> float:
    leg = LEGACY_V3_LIMITS.get(inst.name)
    if leg:
        return leg["max_slippage_price"]
    return inst.typical_spread_pips * 0.5 * inst.pip


def normalize_symbol(raw) -> str | None:
    """Accepts GBPUSD, GBP/USD, GBP_USD, OANDA:GBPUSD, gbp-usd, 'GBP USD'."""
    if raw is None:
        return None
    s = str(raw).upper().strip()
    if ":" in s:
        s = s.split(":")[-1]
    s = s.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
    if len(s) == 6:
        cand = f"{s[:3]}_{s[3:]}"
        if cand in REGISTRY:
            return cand
    return None


def session_of(hour_utc: int) -> str:
    for name, (a, b) in SESSIONS_UTC.items():
        if a <= hour_utc < b:
            return name
    return "unknown"

# ===== FILE: manus_v4/precision.py =====
"""Exact price/unit rounding with Decimal so OANDA never sees float noise
(e.g. 1.2345000000000002) or too many decimals."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_FLOOR, ROUND_CEILING, ROUND_DOWN
from .instruments import Instrument


def _q(dp: int) -> Decimal:
    return Decimal(1).scaleb(-dp)


def round_price(inst: Instrument, price: float, mode: str = "nearest") -> Decimal:
    d = Decimal(repr(float(price)))
    rnd = {"nearest": ROUND_HALF_EVEN, "down": ROUND_FLOOR, "up": ROUND_CEILING}[mode]
    return d.quantize(_q(inst.display_precision), rounding=rnd)


def round_stop(inst: Instrument, price: float, direction: str) -> Decimal:
    """Stops round AWAY from entry (never tighter than computed risk)."""
    return round_price(inst, price, "down" if direction == "long" else "up")


def round_target(inst: Instrument, price: float, direction: str) -> Decimal:
    """Targets round TOWARD entry (never promise more than computed)."""
    return round_price(inst, price, "down" if direction == "long" else "up")


def format_price(inst: Instrument, price) -> str:
    return f"{Decimal(str(price)).quantize(_q(inst.display_precision)):f}"


def round_units(inst: Instrument, units: float) -> Decimal:
    """Units always truncate toward zero so risk is never exceeded."""
    return Decimal(repr(float(units))).quantize(_q(inst.trade_units_precision), rounding=ROUND_DOWN)


def format_units(inst: Instrument, units) -> str:
    return f"{Decimal(str(units)).quantize(_q(inst.trade_units_precision), rounding=ROUND_DOWN):f}"


def price_to_pips(inst: Instrument, distance: float) -> float:
    return distance / inst.pip


def pips_to_price(inst: Instrument, pips: float) -> float:
    return pips * inst.pip

# ===== FILE: manus_v4/fx.py =====
"""Currency conversion for any account currency (default GBP)."""
from __future__ import annotations
from typing import Dict, Optional

Prices = Dict[str, float]   # OANDA code -> mid price, e.g. {"GBP_USD": 1.27}


def rate(from_ccy: str, to_ccy: str, prices: Prices) -> Optional[float]:
    """Units of `to_ccy` per 1 `from_ccy`, using direct, inverse or a single
    USD/EUR/GBP cross. Returns None when no path exists (caller must reject,
    never guess)."""
    if from_ccy == to_ccy:
        return 1.0
    d = prices.get(f"{from_ccy}_{to_ccy}")
    if d:
        return d
    inv = prices.get(f"{to_ccy}_{from_ccy}")
    if inv:
        return 1.0 / inv
    for via in ("USD", "EUR", "GBP"):
        if via in (from_ccy, to_ccy):
            continue
        a = rate_direct(from_ccy, via, prices)
        b = rate_direct(via, to_ccy, prices)
        if a and b:
            return a * b
    return None


def rate_direct(a: str, b: str, prices: Prices) -> Optional[float]:
    if a == b:
        return 1.0
    if prices.get(f"{a}_{b}"):
        return prices[f"{a}_{b}"]
    if prices.get(f"{b}_{a}"):
        return 1.0 / prices[f"{b}_{a}"]
    return None

# ===== FILE: manus_v4/sizing.py =====
"""Risk-% position sizing with account-currency conversion + margin check."""
from __future__ import annotations
from decimal import Decimal
from typing import Dict, Optional
from .instruments import Instrument
from .precision import round_units
from . import fx


def size_position(inst: Instrument, *, account_ccy: str, equity: float, risk_percent: float,
                  entry: float, stop: float, prices: fx.Prices, margin_available: Optional[float] = None,
                  margin_use_cap: float = 0.5, risk_multiplier: float = 1.0) -> Dict:
    """units = risk_cash_acct / (stop_distance * quote->account rate).

    Fixes the v3.5 class of bug for ANY pair and ANY account currency (v3
    only handled quote==USD or base==USD with a USD account)."""
    out = {"ok": False, "reason": None}
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or equity <= 0:
        out["reason"] = "INVALID_STOP_OR_EQUITY"
        return out
    risk_pct = min(max(risk_percent, 0.0), inst.max_risk_percent) * max(0.0, min(risk_multiplier, 1.0))
    risk_cash = equity * risk_pct / 100.0
    q2a = fx.rate(inst.quote, account_ccy, prices)
    if q2a is None:
        out["reason"] = "NO_CONVERSION_RATE"
        return out
    loss_per_unit_acct = stop_dist * q2a
    raw_units = round(risk_cash / loss_per_unit_acct, 6)  # kill float noise before truncation
    cap = min(inst.engine_max_units, inst.maximum_order_units)
    units = round_units(inst, min(raw_units, cap))
    capped_by = "engine_max_units" if raw_units > cap else None
    # Margin: notional in account ccy * margin rate
    b2a = fx.rate(inst.base, account_ccy, prices)
    if b2a is None:
        b2a = entry * q2a
    if margin_available is not None and units > 0:
        max_by_margin = round_units(inst, (margin_available * margin_use_cap) / (b2a * inst.margin_rate))
        if units > max_by_margin:
            units, capped_by = max_by_margin, "margin"
    if units < Decimal(str(inst.minimum_trade_size)):
        out.update(reason="BELOW_MINIMUM_TRADE_SIZE", raw_units=raw_units, units=units)
        return out
    actual_risk = float(units) * loss_per_unit_acct
    out.update(ok=True, units=units, raw_units=raw_units, risk_percent=risk_pct, risk_cash=risk_cash,
               actual_risk_cash=actual_risk, quote_to_account_rate=q2a, base_to_account_rate=b2a,
               margin_required=float(units) * b2a * inst.margin_rate, notional_account=float(units) * b2a,
               capped_by=capped_by, lots=float(units) / (100000 if inst.kind == "CURRENCY" else 1))
    return out

# ===== FILE: manus_v4/analysis.py =====
"""Regime detection, multi-timeframe confirmation, entry timing and
confidence scoring. Every function returns (value, explanations) so the
decision can always say WHY."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import exp
from typing import Dict, List, Optional, Tuple


@dataclass
class Snapshot:
    instrument: str
    direction: str
    entry_model: str
    bar_time_utc: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    atr: float
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    atr_avg: Optional[float] = None          # e.g. 50-bar mean of ATR
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    rsi: Optional[float] = None
    rsi_prev: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_hist_prev: Optional[float] = None
    ema_fast: Optional[float] = None         # 20
    ema_slow: Optional[float] = None         # 50
    ema_trend: Optional[float] = None        # 200
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    # [{"tf": "10m", "bias": "bullish", "completed": True}, ...]
    htf: List[Dict] = field(default_factory=list)
    bid: Optional[float] = None
    ask: Optional[float] = None
    quote_is_real: bool = False               # bid/ask from broker at decision time
    quote_age_seconds: Optional[float] = None
    # entry confirmation: the signal bar arms; a later bar must confirm
    confirmed: bool = False                   # close beyond armed bar extreme
    bars_since_arm: int = 0
    pullback_touched_ema: Optional[bool] = None
    high_impact_event_minutes: Optional[float] = None   # minutes to/since nearest event (abs)
    context_status: str = "unchecked"
    session_name: Optional[str] = None
    conviction_score: Optional[float] = None
    candidate_strength: Optional[float] = None
    signal_id: Optional[str] = None
    spread_price: Optional[float] = None
    expected_slippage_price: Optional[float] = None


# ---------------- regime ----------------
def detect_regime(s: Snapshot) -> Tuple[str, str, List[str]]:
    """Returns (market_regime, volatility_regime, why)."""
    why = []
    atr_ratio = (s.atr / s.atr_avg) if s.atr_avg else None
    if atr_ratio is None:
        vol = "unknown"
    elif atr_ratio > 2.2:
        vol = "extreme"
    elif atr_ratio > 1.5:
        vol = "high"
    elif atr_ratio < 0.7:
        vol = "low"
    else:
        vol = "normal"
    why.append(f"volatility={vol} (ATR/ATRavg={atr_ratio:.2f})" if atr_ratio else "volatility=unknown (no ATR average)")
    sep = abs(s.ema_fast - s.ema_slow) / s.atr if (s.ema_fast and s.ema_slow and s.atr) else None
    adx = s.adx
    if vol == "extreme":
        reg = "extreme"
    elif adx is None or sep is None:
        reg = "unknown"
    elif adx >= 25 and sep >= 0.6:
        reg = "trend"
    elif adx >= 20 and atr_ratio and atr_ratio > 1.3:
        reg = "breakout"
    elif adx < 19 and sep < 0.40:
        reg = "range"
    else:
        reg = "transition"
    why.append(f"regime={reg} (ADX={adx if adx is None else round(adx,1)}, EMA sep/ATR={sep if sep is None else round(sep,2)})")
    return reg, vol, why


# ---------------- multi-timeframe ----------------
def mtf_alignment(s: Snapshot) -> Tuple[float, bool, List[str]]:
    """Score in [-1, 1]; second value False if any HTF input may repaint."""
    want = "bullish" if s.direction == "long" else "bearish"
    against = "bearish" if s.direction == "long" else "bullish"
    if not s.htf:
        return 0.0, False, ["no higher-timeframe input"]
    weights = {"5m": 0.5, "10m": 1.0, "15m": 1.0, "30m": 1.2, "60m": 1.5, "240m": 1.5, "d": 1.0}
    tot = sc = 0.0
    all_done = True
    why = []
    for h in s.htf:
        w = weights.get(h.get("tf"), 1.0)
        b = str(h.get("bias", "unknown")).lower()
        tot += w
        sc += w if b == want else (-w if b == against else 0)
        if not h.get("completed", False):
            all_done = False
        why.append(f"HTF {h.get('tf')}={b}{'' if h.get('completed') else ' (DEVELOPING BAR - may repaint)'}")
    return (sc / tot if tot else 0.0), all_done, why


# ---------------- entry timing ----------------
def entry_timing(s: Snapshot, max_chase_atr: float = 1.0, max_wait_bars: int = 3) -> Tuple[str, List[str], Dict]:
    """READY | WAIT | STALE | CHASE. Stops early entries: the signal bar only
    ARMS a setup; Manus needs a later bar to close beyond the armed bar's
    extreme (or price to trade through the trigger) before READY."""
    why, info = [], {}
    trig = s.prev_high if s.direction == "long" else s.prev_low
    info["trigger_price"] = trig
    ext = abs(s.close - s.ema_fast) / s.atr if (s.ema_fast and s.atr) else None
    info["distance_from_ema_atr"] = ext
    if ext is not None and ext > max_chase_atr:
        why.append(f"price {ext:.2f} ATR from EMA20 > {max_chase_atr} - chasing")
        return "CHASE", why, info
    if s.rsi is not None and ((s.direction == "long" and s.rsi >= 70) or (s.direction == "short" and s.rsi <= 30)):
        why.append(f"RSI {s.rsi:.1f} already stretched in trade direction")
        return "CHASE", why, info
    if not s.confirmed:
        if s.bars_since_arm > max_wait_bars:
            why.append(f"no confirmation within {max_wait_bars} bars - setup expired")
            return "STALE", why, info
        why.append("setup ARMED, waiting for a close beyond the trigger bar (anti-early-entry)")
        return "WAIT", why, info
    if s.entry_model == "trend_pullback" and s.pullback_touched_ema is False:
        why.append("trend_pullback without an actual EMA20 touch - not a pullback")
        return "WAIT", why, info
    why.append("entry confirmed by close beyond trigger bar")
    return "READY", why, info


# ---------------- momentum / signal quality ----------------
def momentum_score(s: Snapshot) -> Tuple[float, List[str]]:
    sgn = 1 if s.direction == "long" else -1
    pts, why = 0.0, []
    if s.ema_fast and s.ema_slow and s.ema_trend:
        stack = (s.ema_fast > s.ema_slow and s.close > s.ema_trend) if sgn > 0 else (s.ema_fast < s.ema_slow and s.close < s.ema_trend)
        pts += 1 if stack else -1
        why.append("EMA stack aligned" if stack else "EMA stack NOT aligned")
    if s.macd_hist is not None and s.macd_hist_prev is not None:
        rising = (s.macd_hist - s.macd_hist_prev) * sgn > 0
        pts += 0.5 if rising else -0.5
        why.append("MACD histogram turning with trade" if rising else "MACD histogram against trade")
    if s.plus_di is not None and s.minus_di is not None:
        ok = (s.plus_di - s.minus_di) * sgn > 0
        pts += 0.5 if ok else -0.5
        why.append("DI aligned" if ok else "DI against")
    if s.rsi is not None and s.rsi_prev is not None:
        ok = (s.rsi - s.rsi_prev) * sgn > 0
        pts += 0.25 if ok else -0.25
    return pts / 2.25, why   # roughly [-1, 1]


@dataclass(frozen=True)
class ConfidenceWeights:
    intercept: float = -0.35
    mtf: float = 0.45
    momentum: float = 0.35
    regime_fit: float = 0.30
    session: float = 0.15
    vol_normal: float = 0.10
    cost_penalty: float = 1.2     # per 1R of cost


def logistic(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def shrunk_bucket_rate(wins: int, losses: int, prior_a: float = 8.0, prior_b: float = 12.0) -> float:
    """Beta-binomial shrinkage toward a sceptical 40% prior."""
    return (wins + prior_a) / (wins + losses + prior_a + prior_b)

# ===== FILE: manus_v4/plan.py =====
"""Stop / target / trade-management planning with ATR + structure +
per-instrument stop floor (fixes the 6.2-pip median stop problem)."""
from __future__ import annotations
from typing import Dict, List, Optional
from .instruments import Instrument
from .precision import round_stop, round_target, round_price, format_price


def plan_levels(inst: Instrument, direction: str, entry: float, atr: float, *,
                nearest_support: Optional[float] = None, nearest_resistance: Optional[float] = None,
                spread_price: Optional[float] = None, rr: Optional[float] = None,
                min_rr: float = 1.35, atr_mult: Optional[float] = None,
                partial_at_r: float = 1.0, partial_fraction: float = 0.5,
                trail_atr_mult: float = 1.0, time_stop_bars: int = 30) -> Dict:
    why: List[str] = []
    m = atr_mult or inst.atr_stop_mult
    rr = max(rr or inst.default_rr, min_rr)
    buf = inst.structure_buffer_pips * inst.pip
    atr_dist = atr * m if atr > 0 else entry * 0.001
    floor_dist = inst.min_stop_pips * inst.pip
    spread_floor = (spread_price or 0) * 7.5
    if direction == "long":
        struct_dist = (entry - (nearest_support - buf)) if nearest_support is not None and nearest_support < entry else 0
    else:
        struct_dist = ((nearest_resistance + buf) - entry) if nearest_resistance is not None and nearest_resistance > entry else 0
    cands = {"atr": atr_dist, "structure": struct_dist, "instrument_floor": floor_dist, "spread_x7_5": spread_floor}
    basis = max(cands, key=cands.get)
    dist = cands[basis]
    why.append(f"stop basis={basis}: " + ", ".join(f"{k}={v / inst.pip:.1f}p" for k, v in cands.items()))
    sgn = 1 if direction == "long" else -1
    e = round_price(inst, entry)
    sl = round_stop(inst, float(e) - sgn * dist, direction)
    risk = abs(float(e) - float(sl))
    tp = round_target(inst, float(e) + sgn * risk * rr, direction)
    real_rr = abs(float(tp) - float(e)) / risk if risk else None
    partial = round_target(inst, float(e) + sgn * risk * partial_at_r, direction)
    trail = max(atr * trail_atr_mult, inst.min_trailing_stop_distance)
    trail = float(round_price(inst, trail, "up"))
    return {
        "entry": e, "stop_loss": sl, "take_profit": tp, "rr": real_rr,
        "stop_distance": risk, "stop_pips": risk / inst.pip, "stop_basis": basis,
        "entry_str": format_price(inst, e), "stop_loss_str": format_price(inst, sl),
        "take_profit_str": format_price(inst, tp),
        "management": {
            "partial_take_profit": {"price": format_price(inst, partial), "at_r": partial_at_r, "fraction": partial_fraction},
            "move_stop_to_breakeven_at_r": partial_at_r,
            "trailing_stop_distance": format_price(inst, trail),
            "trailing_activates_at_r": partial_at_r,
            "time_stop_bars": time_stop_bars,
            "note": "Management plan is SHADOW guidance; v3 was fixed_entry_fixed_exit. Must be simulated before use.",
        },
        "why": why,
    }

# ===== FILE: manus_v4/portfolio.py =====
"""Portfolio exposure, per-currency risk and correlation limits."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .instruments import REGISTRY, CORRELATION_GROUPS


@dataclass(frozen=True)
class PortfolioLimits:
    max_open_trades: int = 4
    max_total_risk_percent: float = 2.0        # sum of open risk
    max_currency_risk_percent: float = 1.0     # net risk on any one currency leg
    max_group_trades: int = 2                  # per correlation group, same direction
    max_trades_per_instrument: int = 1


def currency_legs(instrument: str, direction: str, risk_pct: float) -> Dict[str, float]:
    i = REGISTRY[instrument]
    s = 1 if direction == "long" else -1
    return {i.base: s * risk_pct, i.quote: -s * risk_pct}


def check(instrument: str, direction: str, risk_pct: float, open_positions: List[Dict], lim: PortfolioLimits) -> Tuple[List[str], Dict]:
    """open_positions: [{"instrument": "EUR_USD", "direction": "long", "risk_percent": 0.5}]"""
    veto, why = [], {}
    if len(open_positions) >= lim.max_open_trades:
        veto.append("MAX_OPEN_TRADES")
    if sum(p["risk_percent"] for p in open_positions) + risk_pct > lim.max_total_risk_percent:
        veto.append("PORTFOLIO_HEAT_LIMIT")
    if sum(1 for p in open_positions if p["instrument"] == instrument) >= lim.max_trades_per_instrument:
        veto.append("INSTRUMENT_ALREADY_OPEN")
    legs: Dict[str, float] = {}
    for p in open_positions + [{"instrument": instrument, "direction": direction, "risk_percent": risk_pct}]:
        for c, v in currency_legs(p["instrument"], p["direction"], p["risk_percent"]).items():
            legs[c] = legs.get(c, 0) + v
    why["net_currency_risk"] = legs
    worst = max(legs.items(), key=lambda kv: abs(kv[1]))
    if abs(worst[1]) > lim.max_currency_risk_percent + 1e-9:
        veto.append(f"CURRENCY_CONCENTRATION_{worst[0]}")
    for g, members in CORRELATION_GROUPS.items():
        if instrument in members:
            same = sum(1 for p in open_positions if p["instrument"] in members and p["direction"] == direction)
            if same >= lim.max_group_trades:
                veto.append(f"CORRELATION_GROUP_{g.upper()}")
    return veto, why

# ===== FILE: manus_v4/safety.py =====
"""Kill switches, mode guard, session / rollover / weekend / news filters."""
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

ALLOWED_MODES = ("RESEARCH", "SHADOW", "DEMO")
NY = ZoneInfo("America/New_York")


class LiveModeRefused(RuntimeError):
    pass


def assert_mode(mode: str) -> str:
    m = (mode or "").upper()
    if m not in ALLOWED_MODES:
        raise LiveModeRefused(f"mode {mode!r} refused: this build is RESEARCH/SHADOW/DEMO only")
    return m


@dataclass(frozen=True)
class KillSwitchLimits:
    max_daily_loss_r: float = 3.0
    max_daily_loss_percent: float = 1.5
    max_weekly_drawdown_percent: float = 4.0
    max_consecutive_losses: int = 4
    max_trades_per_day: int = 12
    max_quote_age_seconds: float = 10.0
    kill_file: str = "/workspace/stage0/manus_v4/KILL"   # presence halts all approvals


def kill_switches(account: Dict, lim: KillSwitchLimits) -> List[str]:
    v = []
    if os.path.exists(lim.kill_file) or str(os.environ.get("MANUS_KILL", "")).lower() in ("1", "true"):
        v.append("MANUAL_KILL_SWITCH")
    if account.get("realised_r_today", 0) <= -lim.max_daily_loss_r:
        v.append("DAILY_LOSS_R_STOP")
    if account.get("realised_pct_today", 0) <= -lim.max_daily_loss_percent:
        v.append("DAILY_LOSS_PCT_STOP")
    if account.get("drawdown_pct_week", 0) >= lim.max_weekly_drawdown_percent:
        v.append("WEEKLY_DRAWDOWN_STOP")
    if account.get("consecutive_losses", 0) >= lim.max_consecutive_losses:
        v.append("CONSECUTIVE_LOSS_STOP")
    if account.get("trades_today", 0) >= lim.max_trades_per_day:
        v.append("MAX_TRADES_PER_DAY")
    return v


def market_time_vetoes(ts: datetime, high_impact_event_minutes: Optional[float], news_window_min: float = 30.0) -> List[str]:
    v = []
    ny = ts.astimezone(NY)
    # FX week: Sun 17:00 NY -> Fri 17:00 NY. Block last hour Fri and first hour Sun.
    wd, hm = ny.weekday(), ny.hour * 60 + ny.minute
    if wd == 5 or (wd == 4 and hm >= 16 * 60) or (wd == 6 and hm < 18 * 60):
        v.append("WEEKEND_OR_WEEKLY_EDGE")
    if 16 * 60 + 55 <= hm < 17 * 60 + 15:
        v.append("DAILY_ROLLOVER_WINDOW")
    if high_impact_event_minutes is not None and abs(high_impact_event_minutes) <= news_window_min:
        v.append("NEWS_WINDOW")
    return v


def parse_time(raw) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        x = float(s)
        return datetime.fromtimestamp(x / 1000 if x > 1e12 else x, tz=timezone.utc)

# ===== FILE: manus_v4/oanda_io.py =====
"""OANDA v20 shapes - BUILD ONLY. Nothing here sends a request. The order
body is a dict for review / DEMO wiring later, with every price and unit
pre-formatted as a string at the instrument's exact precision."""
from __future__ import annotations
import json
from typing import Any, Dict
from .instruments import Instrument
from .precision import format_price, format_units


def build_market_order(inst: Instrument, units, direction: str, plan: Dict, client_id: str, use_trailing: bool = False) -> Dict:
    u = format_units(inst, abs(units))
    body = {"order": {
        "type": "MARKET", "instrument": inst.name,
        "units": u if direction == "long" else f"-{u}",
        "timeInForce": "FOK", "positionFill": "DEFAULT",
        "stopLossOnFill": {"price": format_price(inst, plan["stop_loss"]), "timeInForce": "GTC"},
        "takeProfitOnFill": {"price": format_price(inst, plan["take_profit"])},
        "clientExtensions": {"id": client_id[:128], "tag": "manus_v4_shadow"},
    }}
    if use_trailing:
        body["order"]["trailingStopLossOnFill"] = {"distance": plan["management"]["trailing_stop_distance"]}
    return body


def build_stop_entry_order(inst: Instrument, units, direction: str, trigger: float, plan: Dict, client_id: str, gtd_iso: str) -> Dict:
    """Stop-entry at the confirmation trigger - the anti-early-entry order type."""
    u = format_units(inst, abs(units))
    return {"order": {
        "type": "STOP", "instrument": inst.name, "units": u if direction == "long" else f"-{u}",
        "price": format_price(inst, trigger), "timeInForce": "GTD", "gtdTime": gtd_iso,
        "stopLossOnFill": {"price": format_price(inst, plan["stop_loss"])},
        "takeProfitOnFill": {"price": format_price(inst, plan["take_profit"])},
        "clientExtensions": {"id": client_id[:128], "tag": "manus_v4_shadow"},
    }}


def parse_json_field(value: Any) -> Dict[str, Any]:
    """v3.6.1 carry-forward: accept a dict or JSON text, never raise."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            p = json.loads(value)
        except json.JSONDecodeError:
            return {"_unparsed": value[:2000]}
        return p if isinstance(p, dict) else {"_unparsed": value[:2000]}
    return {}


def derive_oanda_fill(resp: Dict[str, Any]) -> Dict[str, Any]:
    """v3.6.0 carry-forward of _derive_oanda_fill (unchanged semantics)."""
    resp = resp or {}
    c, f = resp.get("orderCreateTransaction") or {}, resp.get("orderFillTransaction") or {}
    x, r = resp.get("orderCancelTransaction") or {}, resp.get("orderRejectTransaction") or {}
    status = "FILLED" if f.get("id") else "CANCELLED" if x.get("id") else "REJECTED" if r.get("id") else "UNKNOWN"
    return {
        "fill_status": status, "oanda_trade_id": (f.get("tradeOpened") or {}).get("tradeID"),
        "order_create_transaction_id": c.get("id"), "order_cancel_transaction_id": x.get("id"),
        "order_reject_transaction_id": r.get("id"), "last_transaction_id": resp.get("lastTransactionID"),
        "order_batch_id": c.get("batchID"), "order_request_id": c.get("requestID"),
        "related_transaction_ids": resp.get("relatedTransactionIDs"),
        "cancel_or_reject_reason": x.get("reason") or r.get("rejectReason"),
        "order_created_at": c.get("time"), "filled_at": f.get("time"), "cancelled_at": x.get("time"),
        "rejected_at": r.get("time"), "submitted_units": c.get("units"),
        "submitted_stop_loss": (c.get("stopLossOnFill") or {}).get("price"),
        "submitted_take_profit": (c.get("takeProfitOnFill") or {}).get("price"),
        "fill_price": f.get("price"),
    }

# ===== FILE: manus_v4/ledger.py =====
"""Master data log record + offline analytics (carry-forward of
/analytics/prediction-accuracy logic, without any DB)."""
from __future__ import annotations
import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

LOG_SCHEMA_VERSION = "manus_decision_v4.0"


def _j(v: Any) -> Any:
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dict):
        return {k: _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    return v


def event_id(signal_id: str) -> str:
    """Same key scheme as v3 log_event for predictions."""
    return hashlib.sha256(f"prediction:{signal_id}".encode()).hexdigest()


def decision_record(decision: Dict, raw_payload: Optional[Dict], account_snapshot: Optional[Dict]) -> Dict:
    """Superset of the v3 'prediction' event payload. Every v3 key used by
    master_decision_log / master_trading_log / prediction-accuracy is
    present with the same name, so existing views keep working."""
    sid = decision.get("signal_id") or "unknown"
    rec = {
        "event_id": event_id(sid), "event_type": "prediction", "origin": "manus_engine_v4",
        "environment": decision.get("mode"), "schema_version": LOG_SCHEMA_VERSION,
        "signal_id": sid, "instrument": decision.get("instrument"), "timeframe": decision.get("timeframe"),
        "bar_time_utc": decision.get("bar_time_utc"), "entry_model": decision.get("entry_model"),
        "direction": decision.get("direction"),
        "approval_status": decision.get("approval_status"), "action": decision.get("action"),
        "rejection_stage": decision.get("rejection_stage"),
        "rejection_reason_code": decision.get("rejection_reason_code"),
        "all_veto_codes": decision.get("vetoes"),
        "predicted_win_probability": decision.get("probability"),
        "predicted_tp_before_sl_probability": decision.get("probability"),
        "probability_is_calibrated": False,
        "confidence_score": decision.get("confidence_score"),
        "expected_gross_r": decision.get("expected_gross_r"), "expected_cost_r": decision.get("cost_r"),
        "expected_net_r": decision.get("expected_net_r"),
        "regime": decision.get("regime"), "volatility_regime": decision.get("volatility_regime"),
        "mtf_score": decision.get("mtf_score"), "entry_timing": decision.get("entry_timing"),
        "legacy_v3_scores": decision.get("legacy_v3"),
        "final_trade_plan": decision.get("final_trade_plan"),
        "hypothetical_trade_plan": decision.get("hypothetical_trade_plan"),
        "oanda_order_preview": decision.get("oanda_order_preview"),
        "explain": decision.get("explain"),
        "engine_version": decision.get("engine_version"), "ruleset_version": decision.get("ruleset_version"),
        "calibration_version": decision.get("calibration_version"),
        "instrument_provenance": decision.get("instrument_provenance"),
        "shadow_score_v0": decision.get("shadow_score_v0"),
        "raw_signal_payload": raw_payload, "account_snapshot": account_snapshot,
    }
    return json.loads(json.dumps(_j(rec), default=str))


def score_against_candles(direction: str, stop_loss: float, take_profit: float, candles: List[Dict]) -> Tuple[str, Optional[str]]:
    """v3.10 carry-forward: SL-first on ambiguous bars (conservative)."""
    for c in candles:
        sl = c["low"] <= stop_loss if direction == "long" else c["high"] >= stop_loss
        tp = c["high"] >= take_profit if direction == "long" else c["low"] <= take_profit
        if sl:
            return "loss", c.get("_event_time")
        if tp:
            return "win", c.get("_event_time")
    return "undetermined", None


def bucket_stats(rows: List[Dict]) -> Dict:
    n = len(rows)
    w = sum(1 for r in rows if r["actual_outcome"] == "win")
    ps = [r["predicted_win_probability"] for r in rows if r.get("predicted_win_probability") is not None]
    return {"n": n, "wins": w, "actual_win_rate": round(w / n, 4) if n else None,
            "avg_predicted_win_probability": round(sum(ps) / len(ps), 4) if ps else None}


def calibration_deciles(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, List] = {}
    for r in rows:
        p = r.get("predicted_win_probability")
        if p is None:
            continue
        lo = min(int(p * 10) * 10, 90)
        out.setdefault(f"{lo}-{lo + 10}%", []).append(r)
    return {k: bucket_stats(v) for k, v in sorted(out.items())}


def brier(pairs: List[Tuple[float, bool]]) -> float:
    return sum((p - (1.0 if w else 0.0)) ** 2 for p, w in pairs) / max(1, len(pairs))

# ===== FILE: manus_v4/legacy_v3.py =====
"""Carried forward from engine.py v3.11.0 so nothing is lost: payload
helpers, model policies, data-quality validation and the v3 scoring
functions. v4 logs these as `legacy_v3_*` scores for side-by-side
comparison; they no longer decide on their own."""
from __future__ import annotations
import re
from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Optional, Tuple

JSON_NAN_INF_RE = re.compile(r'(?<![\w"])(-?Infinity|NaN)(?![\w"])')


def sanitize_json_text(raw: str) -> str:
    return JSON_NAN_INF_RE.sub("null", raw)


def get_path(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(d: Dict[str, Any], path: str, value: Any) -> None:
    cur = d
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.replace("£", "").replace("$", "").replace(",", "").strip()
            if x.lower() in ("", "none", "null", "nan"):
                return default
        v = float(x)
        return v if isfinite(v) else default
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    return default if x is None else str(x)


def as_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off", ""):
        return False
    return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class ModelPolicy:
    enabled: bool
    research_only: bool
    min_probability: float
    min_expected_net_r: float
    min_rr: float
    min_pre_gate_score: float
    preferred_regimes: Tuple[str, ...]
    blocked_regimes: Tuple[str, ...]


MODEL_POLICIES: Dict[str, ModelPolicy] = {
    "trend_pullback": ModelPolicy(True, False, 0.53, 0.05, 1.35, 72.0, ("trend", "trend_pullback"), ("extreme",)),
    "momentum_continuation": ModelPolicy(True, False, 0.55, 0.06, 1.40, 74.0, ("trend", "breakout"), ("range", "extreme")),
    "range_reversal": ModelPolicy(True, True, 0.58, 0.10, 1.50, 80.0, ("range", "range_to_reversal"), ("trend", "breakout", "extreme")),
    "bb_rsi_immediate_rejection": ModelPolicy(True, True, 0.60, 0.10, 1.50, 82.0, ("range", "range_to_reversal"), ("trend", "breakout", "extreme")),
    "bb_rsi_armed_reversal": ModelPolicy(True, True, 0.58, 0.08, 1.45, 80.0, ("range", "range_to_reversal"), ("breakout", "extreme")),
    "mean_reversion_with_trend_filter": ModelPolicy(False, True, 0.65, 0.15, 1.60, 90.0, ("range",), ("trend", "breakout", "extreme")),
    "unknown": ModelPolicy(False, True, 0.65, 0.15, 1.60, 90.0, tuple(), ("unknown", "extreme")),
}
MODEL_ALIASES = {
    "trend pullback": "trend_pullback", "momentum continuation": "momentum_continuation",
    "range reversal": "range_reversal", "bb_rsi_exhaustion_reversal": "bb_rsi_armed_reversal",
    "bb_rsi_reversal": "bb_rsi_armed_reversal",
}
REVERSAL_MODELS = ("range_reversal", "bb_rsi_immediate_rejection", "bb_rsi_armed_reversal")

REQUIRED_FIELDS = ("meta.signal_id", "system.strategy_id", "system.strategy_version", "instrument.symbol",
                   "market.timeframe", "market.bar_time_utc", "signal.direction", "signal.entry_model",
                   "price.close", "indicators.atr")


def normalize_model(payload: Dict[str, Any]) -> str:
    raw = as_str(get_path(payload, "signal.entry_model"), "unknown").strip().lower()
    m = MODEL_ALIASES.get(raw, raw)
    return m if m in MODEL_POLICIES else "unknown"


def validate_data(payload: Dict[str, Any]):
    missing = [p for p in REQUIRED_FIELDS if get_path(payload, p) in (None, "")]
    invalid = []
    if as_str(get_path(payload, "signal.direction"), "").lower() not in ("long", "short"):
        invalid.append("signal.direction")
    if (as_float(get_path(payload, "price.close"), 0.0) or 0.0) <= 0:
        invalid.append("price.close")
    if (as_float(get_path(payload, "indicators.atr"), 0.0) or 0.0) <= 0:
        invalid.append("indicators.atr")
    bid, ask = as_float(get_path(payload, "price.bid")), as_float(get_path(payload, "price.ask"))
    spread = as_float(get_path(payload, "price.spread_price"))
    exec_ok = bid is not None and ask is not None and ask >= bid and spread is not None
    cs = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
    score = 100.0 - 8.0 * len(missing) - 15.0 * len(invalid) - (0 if exec_ok else 20.0) - (10.0 if cs in ("unchecked", "unknown", "") else 0)
    return (not missing and not invalid), clamp(score, 0, 100), {
        "missing_fields": missing, "invalid_fields": invalid, "execution_data_complete": exec_ok, "context_status": cs}


def signal_score(payload: Dict[str, Any], model: str) -> float:
    g = lambda p: get_path(payload, p)
    direction = as_str(g("signal.direction"), "neutral").lower()
    rsi, ef, es, macd, adx = (as_float(g(k)) for k in ("indicators.rsi", "indicators.ema_fast", "indicators.ema_slow", "indicators.macd_histogram", "indicators.adx"))
    htf = as_str(g("structure.higher_timeframe_bias"), "unknown").lower()
    trend = as_str(g("structure.trend_bias"), "unknown").lower()
    vol = as_str(g("structure.volatility_regime"), "unknown").lower()
    cstr = as_float(g("extensions.candidate_strength"), 50.0) or 50.0
    conv = as_float(g("signal.conviction_score"), 50.0) or 50.0
    s = 45.0
    if direction == "long":
        if ef is not None and es is not None: s += 8 if ef >= es else -8
        s += 8 if trend == "bullish" else (-12 if trend == "bearish" else 0)
        s += 10 if htf == "bullish" else (-12 if htf == "bearish" else -3)
        s += 6 if rsi is not None and 35 <= rsi <= 62 else 0
        s += 4 if macd is not None and macd >= 0 else 0
    else:
        if ef is not None and es is not None: s += 8 if ef <= es else -8
        s += 8 if trend == "bearish" else (-12 if trend == "bullish" else 0)
        s += 10 if htf == "bearish" else (-12 if htf == "bullish" else -3)
        s += 6 if rsi is not None and 38 <= rsi <= 65 else 0
        s += 4 if macd is not None and macd <= 0 else 0
    s += {"trend_pullback": 6, "momentum_continuation": 3}.get(model, -5 if model in REVERSAL_MODELS else 0)
    s += 6 if vol == "normal" else (1 if vol in ("low", "high") else -12)
    if adx is not None:
        if model in REVERSAL_MODELS: s += 6 if adx <= 25 else (-10 if adx >= 35 else 0)
        elif 15 <= adx <= 40: s += 5
    s += clamp((cstr - 50) * 0.20, -8, 8) + clamp((conv - 50) * 0.12, -6, 6)
    return clamp(s, 0, 100)


def context_score(payload, preferred, blocked) -> float:
    session = as_str(get_path(payload, "market.session_name"), "unknown").lower()
    status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
    ev = as_bool(get_path(payload, "context.high_impact_event_nearby"), False)
    s = 50.0 + (15 if session in preferred else (-25 if session in blocked else -5))
    s += 20 if status == "clear" else (-15 if status in ("warning", "unchecked", "unknown", "") else -50)
    return clamp(s - (35 if ev else 0), 0, 100)


def fit_score(payload, policy: ModelPolicy) -> float:
    if not policy.enabled:
        return 0.0
    reg = as_str(get_path(payload, "structure.market_regime"), "unknown").lower()
    vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()
    s = 50.0 + (20 if reg in policy.preferred_regimes else (-30 if reg in policy.blocked_regimes else 0))
    s += 10 if vol == "normal" else (2 if vol in ("low", "high") else -15)
    return clamp(s - (8 if policy.research_only else 0), 0, 100)


def estimate_probability(sig, ctx, fit, dq, policy: ModelPolicy) -> float:
    p = 0.20 + ((sig * .35 + ctx * .20 + fit * .30 + dq * .15) / 100) * 0.45
    if policy.research_only: p -= 0.05
    if not policy.enabled: p = 0.0
    return clamp(p, 0.05, 0.75)

# ===== FILE: manus_v4/engine.py =====
"""Manus v4 decision orchestrator. Order of evaluation:
 1 mode guard  2 instrument + data quality  3 kill switches  4 market-time
 (weekend/rollover/news)  5 session  6 quote/spread  7 regime + model fit
 8 MTF (completed bars only)  9 entry timing  10 stop/target plan with floors
 11 cost/R gate  12 sizing + margin  13 portfolio/correlation  14 confidence,
 probability, EV  15 record + explain.
All vetoes are collected (not just the first) so the log shows every reason."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from . import ENGINE_VERSION, RULESET_VERSION, CALIBRATION_VERSION
from .instruments import REGISTRY, normalize_symbol, session_of, max_slippage_price, expected_slippage_price, PROVENANCE
from .analysis import Snapshot, detect_regime, mtf_alignment, entry_timing, momentum_score, ConfidenceWeights, logistic, shrunk_bucket_rate
from .plan import plan_levels
from .sizing import size_position
from .portfolio import PortfolioLimits, check as portfolio_check
from .safety import assert_mode, KillSwitchLimits, kill_switches, market_time_vetoes, parse_time
from .oanda_io import build_market_order, build_stop_entry_order
from . import legacy_v3 as L

# Hard-veto codes that end evaluation stages (order = rejection_stage priority)
STAGES = [
    ("mode", ()), ("instrument", ("UNSUPPORTED_INSTRUMENT", "INSTRUMENT_NOT_TRADABLE")),
    ("data_quality", ("DATA_QUALITY_FAIL", "INVALID_DIRECTION")),
    ("kill_switch", ("MANUAL_KILL_SWITCH", "DAILY_LOSS_R_STOP", "DAILY_LOSS_PCT_STOP", "WEEKLY_DRAWDOWN_STOP", "CONSECUTIVE_LOSS_STOP", "MAX_TRADES_PER_DAY")),
    ("context", ("WEEKEND_OR_WEEKLY_EDGE", "DAILY_ROLLOVER_WINDOW", "NEWS_WINDOW", "BLOCKED_SESSION", "CONTEXT_BLOCKED")),
    ("execution_cost", ("NO_REAL_QUOTE", "STALE_QUOTE", "SPREAD_TOO_WIDE", "SLIPPAGE_TOO_HIGH", "COST_R_EXCESSIVE")),
    ("strategy_fit", ("MODEL_DISABLED", "EXTREME_VOLATILITY", "REGIME_BLOCKED_FOR_MODEL")),
    ("mtf", ("HTF_BIAS_MAY_REPAINT", "HTF_AGAINST_TRADE")),
    ("entry_timing", ("ENTRY_NOT_CONFIRMED", "ENTRY_CHASING", "SETUP_EXPIRED")),
    ("execution_planning", ("INVALID_TRADE_PLAN", "RR_BELOW_MODEL_MINIMUM", "NO_CONVERSION_RATE", "BELOW_MINIMUM_TRADE_SIZE")),
    ("portfolio", ("MAX_OPEN_TRADES", "PORTFOLIO_HEAT_LIMIT", "INSTRUMENT_ALREADY_OPEN")),
    ("expected_value", ("INSUFFICIENT_POST_GATE_EDGE", "BUCKET_EVIDENCE_LOW")),
]


@dataclass
class Config:
    mode: str = "SHADOW"
    account_currency: str = "GBP"
    max_cost_r: float = 0.20
    require_real_quote: bool = True
    max_quote_age_seconds: float = 10.0
    require_completed_htf: bool = True
    min_mtf_score: float = 0.34
    max_chase_atr: float = 1.0
    max_wait_bars: int = 3
    min_bucket_n: int = 30
    min_ev_r: float = 0.05
    min_probability_floor: float = 0.0     # per-model policy minimum also applies
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    portfolio: PortfolioLimits = field(default_factory=PortfolioLimits)
    kill: KillSwitchLimits = field(default_factory=KillSwitchLimits)
    news_window_min: float = 30.0


def _stage_of(code: str) -> str:
    for st, codes in STAGES:
        if any(code == c or code.startswith(c) for c in codes):
            return st
    if code.startswith(("CURRENCY_CONCENTRATION", "CORRELATION_GROUP")):
        return "portfolio"
    return "other"


def decide(s: Snapshot, account: Dict[str, Any], prices: Dict[str, float], cfg: Config = Config(),
           bucket_stats: Optional[Dict] = None, open_positions: Optional[List[Dict]] = None,
           raw_payload: Optional[Dict] = None) -> Dict[str, Any]:
    mode = assert_mode(cfg.mode)
    explain: List[str] = []
    vetoes: List[str] = []
    out: Dict[str, Any] = {"engine_version": ENGINE_VERSION, "ruleset_version": RULESET_VERSION,
                           "calibration_version": CALIBRATION_VERSION, "mode": mode,
                           "signal_id": s.signal_id, "timeframe": s.timeframe, "bar_time_utc": s.bar_time_utc,
                           "direction": s.direction, "entry_model": s.entry_model,
                           "instrument_provenance": PROVENANCE, "probability_is_calibrated": False}
    name = normalize_symbol(s.instrument)
    if not name:
        return _finish(out, ["UNSUPPORTED_INSTRUMENT"], [f"instrument {s.instrument!r} not in registry"])
    inst = REGISTRY[name]
    out["instrument"] = name
    if not inst.tradable:
        vetoes.append("INSTRUMENT_NOT_TRADABLE"); explain.append("instrument flagged not-tradable pending live endpoint check")
    if s.direction not in ("long", "short"):
        return _finish(out, vetoes + ["INVALID_DIRECTION"], explain)
    if not (s.close > 0 and s.atr > 0):
        return _finish(out, vetoes + ["DATA_QUALITY_FAIL"], explain + ["close/ATR missing or non-positive"])
    model = L.MODEL_ALIASES.get(s.entry_model, s.entry_model)
    model = model if model in L.MODEL_POLICIES else "unknown"
    policy = L.MODEL_POLICIES[model]
    out["entry_model"] = model
    if not policy.enabled:
        vetoes.append("MODEL_DISABLED")

    # 3 kill switches
    ks = kill_switches(account, cfg.kill)
    vetoes += ks
    # 4 market time
    ts = parse_time(s.bar_time_utc)
    vetoes += market_time_vetoes(ts, s.high_impact_event_minutes, cfg.news_window_min)
    session = (s.session_name or session_of(ts.hour)).lower()
    out["session"] = session
    if session in inst.blocked_sessions:
        vetoes.append("BLOCKED_SESSION")
    if s.context_status.lower() == "blocked":
        vetoes.append("CONTEXT_BLOCKED")

    # 6 quote / spread
    spread = s.spread_price if s.spread_price is not None else ((s.ask - s.bid) if (s.ask and s.bid and s.ask >= s.bid) else None)
    spread_est = spread is None
    if spread_est:
        spread = inst.typical_spread_pips * inst.pip
    if cfg.require_real_quote and not s.quote_is_real:
        vetoes.append("NO_REAL_QUOTE"); explain.append("no broker bid/ask at decision time (v3 legacy rows had none)")
    if s.quote_age_seconds is not None and s.quote_age_seconds > cfg.max_quote_age_seconds:
        vetoes.append("STALE_QUOTE")
    if spread / inst.pip > inst.max_spread_pips:
        vetoes.append("SPREAD_TOO_WIDE")
    slip = expected_slippage_price(inst)
    if s.expected_slippage_price is not None and s.expected_slippage_price > max_slippage_price(inst):
        vetoes.append("SLIPPAGE_TOO_HIGH")

    # 7 regime
    regime, vol, w = detect_regime(s); explain += w
    out.update(regime=regime, volatility_regime=vol)
    if vol == "extreme":
        vetoes.append("EXTREME_VOLATILITY")
    if regime in policy.blocked_regimes:
        vetoes.append("REGIME_BLOCKED_FOR_MODEL")
    regime_fit = 1.0 if regime in policy.preferred_regimes or (regime == "trend" and "trend_pullback" in policy.preferred_regimes) else (0.0 if regime in ("transition", "unknown") else -0.5)

    # 8 MTF
    mtf, htf_done, w = mtf_alignment(s); explain += w
    out["mtf_score"] = round(mtf, 3)
    if cfg.require_completed_htf and not htf_done:
        vetoes.append("HTF_BIAS_MAY_REPAINT")
    if mtf < cfg.min_mtf_score and model not in L.REVERSAL_MODELS:
        vetoes.append("HTF_AGAINST_TRADE"); explain.append(f"MTF score {mtf:.2f} < {cfg.min_mtf_score}")

    # 9 entry timing
    timing, w, tinfo = entry_timing(s, cfg.max_chase_atr, cfg.max_wait_bars); explain += w
    out["entry_timing"] = timing
    vetoes += {"WAIT": ["ENTRY_NOT_CONFIRMED"], "CHASE": ["ENTRY_CHASING"], "STALE": ["SETUP_EXPIRED"]}.get(timing, [])

    # 10 plan
    entry = (s.ask if s.direction == "long" else s.bid) if s.quote_is_real and s.ask and s.bid else s.close
    lv = plan_levels(inst, s.direction, entry, s.atr, nearest_support=s.nearest_support,
                     nearest_resistance=s.nearest_resistance, spread_price=spread, min_rr=policy.min_rr)
    explain += lv.pop("why")
    if lv["rr"] is None or lv["rr"] < policy.min_rr - 1e-9:
        vetoes.append("RR_BELOW_MODEL_MINIMUM")
    cost_r = (spread + slip) / lv["stop_distance"] if lv["stop_distance"] else 1.0
    out["cost_r"] = round(cost_r, 4)
    explain.append(f"cost {cost_r:.3f}R (spread{' est.' if spread_est else ''} {spread / inst.pip:.1f}p + slip {slip / inst.pip:.1f}p over {lv['stop_pips']:.1f}p stop)")
    if cost_r > cfg.max_cost_r:
        vetoes.append("COST_R_EXCESSIVE")

    # 12 sizing
    equity = L.as_float(account.get("balance")) or L.as_float(account.get("nav")) or 0.0
    sz = size_position(inst, account_ccy=account.get("currency", cfg.account_currency), equity=equity,
                       risk_percent=L.as_float(account.get("risk_percent"), inst.max_risk_percent),
                       entry=float(lv["entry"]), stop=float(lv["stop_loss"]), prices=prices,
                       margin_available=L.as_float(account.get("margin_available")),
                       risk_multiplier=1.0)
    if not sz["ok"]:
        vetoes.append(sz["reason"])

    # 13 portfolio
    pv, pinfo = portfolio_check(name, s.direction, sz.get("risk_percent", inst.max_risk_percent), open_positions or [], cfg.portfolio)
    vetoes += pv

    # 14 confidence
    mom, w = momentum_score(s); explain += w
    W = cfg.weights
    z = (W.intercept + W.mtf * mtf + W.momentum * mom + W.regime_fit * regime_fit
         + W.session * (1 if session in inst.preferred_sessions else -0.5)
         + W.vol_normal * (1 if vol == "normal" else 0) - W.cost_penalty * cost_r)
    p_model = logistic(z)
    key = f"{name}|{model}|{session}|{regime}|{s.direction}"
    wins, losses = (bucket_stats or {}).get(key, (0, 0))
    n = wins + losses
    p_bucket = shrunk_bucket_rate(wins, losses)
    wgt = min(1.0, n / 200.0)
    prob = wgt * p_model + (1 - wgt) * p_bucket if n else 0.5 * p_model + 0.5 * p_bucket
    rr = lv["rr"] or 0
    gross = prob * rr - (1 - prob)
    net = gross - cost_r
    out.update(probability=round(prob, 4), confidence_score=round(prob * 100, 2),
               expected_gross_r=round(gross, 4), expected_net_r=round(net, 4),
               bucket={"key": key, "n": n, "p_bucket": round(p_bucket, 4), "p_model": round(p_model, 4)})
    explain.append(f"probability {prob:.3f} (model {p_model:.3f}, bucket {p_bucket:.3f} n={n}); EV {net:+.3f}R")
    if n < cfg.min_bucket_n:
        vetoes.append("BUCKET_EVIDENCE_LOW")
    if net < max(cfg.min_ev_r, policy.min_expected_net_r) or prob < max(policy.min_probability, cfg.min_probability_floor):
        vetoes.append("INSUFFICIENT_POST_GATE_EDGE")

    # legacy v3 scores for side-by-side logging
    if raw_payload:
        try:
            dq_ok, dq, _ = L.validate_data(raw_payload)
            sig = L.signal_score(raw_payload, model)
            ctx = L.context_score(raw_payload, inst.preferred_sessions, inst.blocked_sessions)
            fit = L.fit_score(raw_payload, policy)
            out["legacy_v3"] = {"data_quality_score": dq, "signal_quality_score": sig, "context_score": ctx,
                                "strategy_fit_score": fit, "predicted_win_probability": L.estimate_probability(sig, ctx, fit, dq, policy)}
        except Exception as e:  # never break a decision on legacy scoring
            out["legacy_v3"] = {"error": str(e)}

    plan = {"instrument": name, "broker_symbol": name, "direction": s.direction,
            "entry": lv["entry_str"], "stop_loss": lv["stop_loss_str"], "take_profit": lv["take_profit_str"],
            "rr_ratio": round(rr, 3), "stop_pips": round(lv["stop_pips"], 1), "stop_basis": lv["stop_basis"],
            "position_size_units": str(sz.get("units")) if sz.get("ok") else None,
            "position_size_lots": sz.get("lots"), "risk_percent": sz.get("risk_percent"),
            "risk_cash": sz.get("risk_cash"), "actual_risk_cash": sz.get("actual_risk_cash"),
            "account_currency": account.get("currency", cfg.account_currency),
            "quote_to_account_rate": sz.get("quote_to_account_rate"), "margin_required": sz.get("margin_required"),
            "capped_by": sz.get("capped_by"), "management": lv["management"],
            "planning_source": "manus_v4_atr_structure_floor", "execution_model": "confirmed_entry_managed_exit_SHADOW"}
    out["hypothetical_trade_plan"] = plan
    out["portfolio"] = pinfo
    if sz.get("ok"):
        cid = f"mv4-{s.signal_id or 'x'}"
        out["oanda_order_preview"] = (build_market_order(inst, sz["units"], s.direction, {**lv}, cid) if timing == "READY"
                                      else build_stop_entry_order(inst, sz["units"], s.direction, tinfo.get("trigger_price") or s.close, lv, cid, "PT6M"))
        out["oanda_order_preview"]["_note"] = "PREVIEW ONLY - never sent by this build"
    return _finish(out, vetoes, explain, plan)


def _finish(out: Dict, vetoes: List[str], explain: List[str], plan: Optional[Dict] = None) -> Dict:
    vetoes = list(dict.fromkeys(vetoes))
    out["vetoes"] = vetoes
    out["explain"] = explain
    if not vetoes:
        out.update(approval_status="approved", action="APPROVE_SHADOW", rejection_stage=None, rejection_reason_code=None,
                   final_trade_plan=plan)
    else:
        order = [st for st, _ in STAGES]
        first = sorted(vetoes, key=lambda c: order.index(_stage_of(c)) if _stage_of(c) in order else 99)[0]
        armed_only = set(vetoes) <= {"ENTRY_NOT_CONFIRMED", "BUCKET_EVIDENCE_LOW"} and "ENTRY_NOT_CONFIRMED" in vetoes
        out.update(approval_status="rejected", action="ARMED_WAIT" if armed_only else ("SHADOW_ONLY" if vetoes == ["BUCKET_EVIDENCE_LOW"] else "REJECT"),
                   rejection_stage=_stage_of(first), rejection_reason_code=first, final_trade_plan=None)
    return out


def snapshot_from_v3_payload(p: Dict[str, Any], authoritative_instrument: Optional[str] = None) -> Snapshot:
    """Adapter: v3 Pine/Make payload -> Snapshot (keeps the authoritative
    instrument override from v3 normalize_symbol)."""
    g = lambda k, d=None: L.get_path(p, k, d)
    f = lambda k: L.as_float(g(k))
    inst = None
    for raw in (authoritative_instrument, g("instrument.symbol"), g("instrument.broker_symbol"), g("instrument.display_name")):
        if normalize_symbol(raw):
            inst = normalize_symbol(raw); break
    htf = []
    if g("structure.higher_timeframe_bias"):
        htf.append({"tf": g("structure.higher_timeframe", "10m"), "bias": g("structure.higher_timeframe_bias"),
                    "completed": L.as_bool(g("structure.htf_completed_bar"), False)})
    bid, ask = f("price.bid"), f("price.ask")
    return Snapshot(
        instrument=inst or str(g("instrument.symbol")), direction=L.as_str(g("signal.direction"), "").lower(),
        entry_model=L.as_str(g("signal.entry_model"), "unknown").strip().lower(), bar_time_utc=g("market.bar_time_utc"),
        timeframe=L.as_str(g("market.timeframe"), ""), open=f("price.open") or 0, high=f("price.high") or 0,
        low=f("price.low") or 0, close=f("price.close") or 0, atr=f("indicators.atr") or 0,
        prev_high=f("price.prev_high"), prev_low=f("price.prev_low"), atr_avg=f("indicators.atr_avg"),
        adx=f("indicators.adx"), plus_di=f("indicators.plus_di"), minus_di=f("indicators.minus_di"),
        rsi=f("indicators.rsi"), rsi_prev=f("indicators.rsi_prev"), macd_hist=f("indicators.macd_histogram"),
        macd_hist_prev=f("indicators.macd_histogram_prev"), ema_fast=f("indicators.ema_fast"),
        ema_slow=f("indicators.ema_slow"), ema_trend=f("indicators.ema_trend"), bb_upper=f("indicators.bb_upper"),
        bb_lower=f("indicators.bb_lower"), nearest_support=f("structure.nearest_support"),
        nearest_resistance=f("structure.nearest_resistance"), htf=htf, bid=bid, ask=ask,
        quote_is_real=L.as_bool(g("price.quote_is_real"), False) and bid is not None and ask is not None,
        confirmed=L.as_bool(g("signal.entry_confirmed"), False), bars_since_arm=int(f("signal.bars_since_arm") or 0),
        high_impact_event_minutes=f("context.high_impact_event_minutes") if f("context.high_impact_event_minutes") is not None else (0.0 if L.as_bool(g("context.high_impact_event_nearby")) else None),
        context_status=L.as_str(g("context.context_status"), "unchecked"), session_name=g("market.session_name"),
        conviction_score=f("signal.conviction_score"), candidate_strength=f("extensions.candidate_strength"),
        signal_id=g("meta.signal_id"), spread_price=f("price.spread_price"))

# ===== FILE: tests/test_v4.py =====
import os, sys, unittest, json
from decimal import Decimal
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from manus_v4.instruments import REGISTRY, normalize_symbol
from manus_v4.precision import round_price, round_stop, round_target, round_units, format_price, format_units
from manus_v4.sizing import size_position
from manus_v4 import fx
from manus_v4.plan import plan_levels
from manus_v4.analysis import Snapshot
from manus_v4.engine import decide, Config, snapshot_from_v3_payload
from manus_v4.safety import assert_mode, LiveModeRefused
from manus_v4.portfolio import check, PortfolioLimits
from manus_v4.oanda_io import derive_oanda_fill, parse_json_field
from manus_v4.ledger import decision_record, score_against_candles, calibration_deciles
from manus_v4 import legacy_v3 as L

PRICES = {"GBP_USD": 1.2700, "USD_JPY": 150.00, "EUR_USD": 1.0800, "USD_CAD": 1.3600,
          "XAU_USD": 2400.00, "XAG_USD": 30.000, "EUR_GBP": 0.8504, "GBP_JPY": 190.50, "USD_CHF": 0.9000}


class Registry(unittest.TestCase):
    def test_counts_and_metals(self):
        self.assertGreaterEqual(len([k for k, v in REGISTRY.items() if v.kind == "CURRENCY"]), 68)
        self.assertIn("XAU_USD", REGISTRY); self.assertIn("XAG_USD", REGISTRY)

    def test_precision_conventions(self):
        for n, i in REGISTRY.items():
            if i.kind == "CURRENCY":
                self.assertEqual(i.display_precision, -i.pip_location + 1, n)
        self.assertEqual(REGISTRY["USD_JPY"].pip_location, -2)
        self.assertEqual(REGISTRY["EUR_HUF"].pip_location, -2)
        self.assertEqual(REGISTRY["GBP_USD"].display_precision, 5)
        self.assertEqual(REGISTRY["XAU_USD"].display_precision, 3)
        self.assertEqual(REGISTRY["XAG_USD"].display_precision, 5)

    def test_normalize(self):
        for raw in ("GBPUSD", "GBP/USD", "OANDA:GBPUSD", "gbp_usd", "GBP-USD"):
            self.assertEqual(normalize_symbol(raw), "GBP_USD")
        self.assertEqual(normalize_symbol("XAUUSD"), "XAU_USD")
        self.assertIsNone(normalize_symbol("BTCUSD"))


class Rounding(unittest.TestCase):
    def test_major(self):
        i = REGISTRY["GBP_USD"]
        self.assertEqual(format_price(i, round_price(i, 1.2345678)), "1.23457")
        self.assertEqual(format_price(i, round_price(i, 0.1 + 0.2)), "0.30000")
        self.assertEqual(round_stop(i, 1.234567, "long"), Decimal("1.23456"))
        self.assertEqual(round_stop(i, 1.234561, "short"), Decimal("1.23457"))

    def test_jpy(self):
        i = REGISTRY["USD_JPY"]
        self.assertEqual(format_price(i, round_price(i, 150.12345)), "150.123")
        self.assertEqual(round_target(i, 150.1239, "long"), Decimal("150.123"))
        self.assertEqual(round_target(i, 150.1231, "short"), Decimal("150.124"))

    def test_xau(self):
        i = REGISTRY["XAU_USD"]
        self.assertEqual(format_price(i, round_price(i, 2401.23456)), "2401.235")
        self.assertEqual(format_units(i, round_units(i, 3.99)), "3")

    def test_xag(self):
        i = REGISTRY["XAG_USD"]
        self.assertEqual(format_price(i, round_price(i, 30.1234567)), "30.12346")
        self.assertEqual(format_units(i, round_units(i, 157.9)), "157")

    def test_units_truncate(self):
        i = REGISTRY["EUR_USD"]
        self.assertEqual(round_units(i, 12345.999), Decimal("12345"))
        self.assertEqual(round_units(i, -12345.999), Decimal("-12345"))


class Sizing(unittest.TestCase):
    def _risk(self, name, entry, stop, eq=10000.0, ccy="GBP"):
        i = REGISTRY[name]
        r = size_position(i, account_ccy=ccy, equity=eq, risk_percent=0.5, entry=entry, stop=stop, prices=PRICES)
        self.assertTrue(r["ok"], r)
        self.assertLessEqual(r["actual_risk_cash"], r["risk_cash"] + 1e-9)       # never exceed risk
        if i.kind == "CURRENCY":
            self.assertGreater(r["actual_risk_cash"], r["risk_cash"] * 0.99)  # close to target
        return r

    def test_gbpusd_gbp_account(self):
        r = self._risk("GBP_USD", 1.27000, 1.26900)   # 10 pips, £50 risk
        # loss per unit = 0.001 USD = 0.001/1.27 GBP -> 63,500 units
        self.assertEqual(int(r["units"]), 63500)

    def test_usdjpy_gbp_account(self):
        r = self._risk("USD_JPY", 150.000, 149.850)   # 15 pips
        # JPY->GBP = 1/(150*... ) via USD: 1/150 USD per JPY, /1.27 GBP
        exp = 50 / (0.15 * (1 / 150) / 1.27)
        self.assertEqual(int(r["units"]), min(int(exp), 200000))

    def test_eurgbp_quote_is_account(self):
        r = self._risk("EUR_GBP", 0.85040, 0.84940)
        self.assertAlmostEqual(r["quote_to_account_rate"], 1.0)
        self.assertEqual(int(r["units"]), 50000)

    def test_xau(self):
        r = self._risk("XAU_USD", 2400.0, 2395.0)     # metals capped 0.35%: £35 -> $44.45 / $5 = 8.89 -> 8 oz
        self.assertEqual(int(r["units"]), 8)

    def test_xag(self):
        r = self._risk("XAG_USD", 30.000, 29.800)     # £35 -> $44.45 / $0.20 = 222.25 -> 222 oz
        self.assertEqual(int(r["units"]), 222)

    def test_usd_account_matches_v3_fix(self):
        r = self._risk("USD_JPY", 150.0, 149.85, ccy="USD")
        self.assertEqual(int(r["units"]), 50000)       # 50 USD / (0.15/150)

    def test_engine_cap_and_min_size(self):
        i = REGISTRY["GBP_USD"]
        r = size_position(i, account_ccy="GBP", equity=1e7, risk_percent=0.5, entry=1.27, stop=1.2699, prices=PRICES)
        self.assertEqual(r["capped_by"], "engine_max_units")
        i = REGISTRY["XAU_USD"]
        r = size_position(i, account_ccy="GBP", equity=100, risk_percent=0.35, entry=2400, stop=2390, prices=PRICES)
        self.assertFalse(r["ok"]); self.assertEqual(r["reason"], "BELOW_MINIMUM_TRADE_SIZE")

    def test_margin_cap(self):
        i = REGISTRY["GBP_USD"]
        r = size_position(i, account_ccy="GBP", equity=10000, risk_percent=0.5, entry=1.27, stop=1.2690, prices=PRICES, margin_available=500)
        self.assertEqual(r["capped_by"], "margin")
        self.assertLessEqual(r["margin_required"], 250 + 1e-6)

    def test_risk_pct_capped(self):
        i = REGISTRY["XAU_USD"]
        r = size_position(i, account_ccy="GBP", equity=10000, risk_percent=5, entry=2400, stop=2395, prices=PRICES)
        self.assertEqual(r["risk_percent"], 0.35)

    def test_no_conversion_path(self):
        i = REGISTRY["USD_THB"]
        r = size_position(i, account_ccy="GBP", equity=10000, risk_percent=0.25, entry=36.0, stop=35.9, prices=PRICES)
        self.assertEqual(r["reason"], "NO_CONVERSION_RATE")


class Planning(unittest.TestCase):
    def test_stop_floor(self):
        i = REGISTRY["GBP_USD"]
        p = plan_levels(i, "long", 1.27, atr=0.0003)   # 4.5 pip ATR stop -> floor 10.5
        self.assertGreaterEqual(p["stop_pips"], i.min_stop_pips - 1e-6)
        self.assertEqual(p["stop_basis"], "instrument_floor")
        self.assertEqual(len(p["stop_loss_str"].split(".")[1]), 5)

    def test_metal_strings(self):
        p = plan_levels(REGISTRY["XAU_USD"], "short", 2400.1234, atr=2.0)
        self.assertEqual(len(p["take_profit_str"].split(".")[1]), 3)
        self.assertGreater(float(p["stop_loss"]), 2400.1234)


def snap(**kw):
    base = dict(instrument="GBPUSD", direction="long", entry_model="trend_pullback", bar_time_utc="2026-10-21T13:30:00Z",
                timeframe="2m", open=1.2700, high=1.2706, low=1.2698, close=1.2705, atr=0.0006, prev_high=1.2703, prev_low=1.2695,
                atr_avg=0.0006, adx=28, plus_di=25, minus_di=15, rsi=55, rsi_prev=50, macd_hist=0.00002, macd_hist_prev=0.00001,
                ema_fast=1.2702, ema_slow=1.2697, ema_trend=1.2680, htf=[{"tf": "10m", "bias": "bullish", "completed": True}, {"tf": "60m", "bias": "bullish", "completed": True}],
                bid=1.27045, ask=1.27055, quote_is_real=True, quote_age_seconds=1, confirmed=True, pullback_touched_ema=True,
                context_status="clear", signal_id="t1", nearest_support=1.2690)
    base.update(kw)
    return Snapshot(**base)


ACCT = {"balance": 10000, "currency": "GBP", "margin_available": 9000}


class Decisions(unittest.TestCase):
    def setUp(self):
        self.bs = {"GBP_USD|trend_pullback|overlap_london_new_york|trend|long": (60, 40)}

    def test_approves_clean_setup(self):
        d = decide(snap(), ACCT, PRICES, Config(), bucket_stats=self.bs)
        self.assertEqual(d["action"], "APPROVE_SHADOW", d["vetoes"])
        self.assertIsNotNone(d["final_trade_plan"]["position_size_units"])
        self.assertTrue(d["oanda_order_preview"]["order"]["units"].isdigit())

    def test_early_entry_waits(self):
        d = decide(snap(confirmed=False), ACCT, PRICES, Config(), bucket_stats=self.bs)
        self.assertEqual(d["action"], "ARMED_WAIT"); self.assertEqual(d["oanda_order_preview"]["order"]["type"], "STOP")

    def test_repaint_veto(self):
        d = decide(snap(htf=[{"tf": "10m", "bias": "bullish", "completed": False}]), ACCT, PRICES, Config(), bucket_stats=self.bs)
        self.assertIn("HTF_BIAS_MAY_REPAINT", d["vetoes"])

    def test_no_quote_and_news_and_kill(self):
        d = decide(snap(quote_is_real=False, high_impact_event_minutes=10), dict(ACCT, realised_r_today=-3.5), PRICES, Config(), bucket_stats=self.bs)
        for c in ("NO_REAL_QUOTE", "NEWS_WINDOW", "DAILY_LOSS_R_STOP"):
            self.assertIn(c, d["vetoes"])
        self.assertEqual(d["rejection_stage"], "kill_switch")

    def test_weekend(self):
        d = decide(snap(bar_time_utc="2026-10-24T12:00:00Z"), ACCT, PRICES, Config(), bucket_stats=self.bs)
        self.assertIn("WEEKEND_OR_WEEKLY_EDGE", d["vetoes"])

    def test_chase(self):
        d = decide(snap(close=1.2715, ema_fast=1.2700), ACCT, PRICES, Config(), bucket_stats=self.bs)
        self.assertIn("ENTRY_CHASING", d["vetoes"])

    def test_bucket_low_is_shadow_only(self):
        d = decide(snap(), ACCT, PRICES, Config(), bucket_stats={})
        self.assertIn("BUCKET_EVIDENCE_LOW", d["vetoes"])

    def test_jpy_and_metal_decisions(self):
        d = decide(snap(instrument="USD_JPY", open=150.0, high=150.06, low=149.98, close=150.05, atr=0.06, prev_high=150.03, prev_low=149.95,
                        atr_avg=0.06, ema_fast=150.02, ema_slow=149.97, ema_trend=149.8, bid=150.045, ask=150.055, nearest_support=149.9), ACCT, PRICES, Config())
        self.assertEqual(len(d["hypothetical_trade_plan"]["stop_loss"].split(".")[1]), 3)
        d = decide(snap(instrument="XAGUSD", open=30.0, high=30.06, low=29.98, close=30.05, atr=0.06, prev_high=30.03, prev_low=29.95,
                        atr_avg=0.06, ema_fast=30.02, ema_slow=29.97, ema_trend=29.8, bid=30.045, ask=30.055, nearest_support=29.9), ACCT, PRICES, Config())
        self.assertEqual(len(d["hypothetical_trade_plan"]["take_profit"].split(".")[1]), 5)

    def test_live_refused(self):
        with self.assertRaises(LiveModeRefused):
            assert_mode("LIVE")
        with self.assertRaises(LiveModeRefused):
            decide(snap(), ACCT, PRICES, Config(mode="PRODUCTION"))

    def test_log_record_compat(self):
        d = decide(snap(), ACCT, PRICES, Config(), bucket_stats=self.bs)
        rec = decision_record(d, {"meta": {"signal_id": "t1"}}, ACCT)
        for k in ("signal_id", "approval_status", "rejection_reason_code", "predicted_win_probability", "expected_net_r",
                  "final_trade_plan", "hypothetical_trade_plan", "raw_signal_payload", "account_snapshot", "shadow_score_v0"):
            self.assertIn(k, rec)
        json.dumps(rec)


class Portfolio(unittest.TestCase):
    def test_currency_concentration(self):
        opens = [{"instrument": "EUR_USD", "direction": "long", "risk_percent": 0.5}, {"instrument": "AUD_USD", "direction": "long", "risk_percent": 0.5}]
        v, _ = check("GBP_USD", "long", 0.5, opens, PortfolioLimits())
        self.assertTrue(any(x.startswith("CURRENCY_CONCENTRATION_USD") for x in v))
        self.assertTrue(any(x.startswith("CORRELATION_GROUP") for x in v))


class CarryForward(unittest.TestCase):
    def test_fill(self):
        f = derive_oanda_fill({"orderFillTransaction": {"id": "5", "tradeOpened": {"tradeID": "6"}}})
        self.assertEqual((f["fill_status"], f["oanda_trade_id"]), ("FILLED", "6"))
        self.assertEqual(parse_json_field("{bad")["_unparsed"], "{bad")

    def test_candles_sl_first(self):
        self.assertEqual(score_against_candles("long", 1.0, 2.0, [{"high": 2.1, "low": 0.9}])[0], "loss")

    def test_nan_sanitize_and_bool(self):
        self.assertEqual(json.loads(L.sanitize_json_text('{"a": NaN, "b": -Infinity}')), {"a": None, "b": None})
        self.assertTrue(L.as_bool("true")); self.assertEqual(L.as_float("£1,234.5"), 1234.5)

    def test_v3_adapter(self):
        p = {"meta": {"signal_id": "s"}, "instrument": {"symbol": "EURUSD"}, "signal": {"direction": "long", "entry_model": "Trend Pullback"},
             "market": {"bar_time_utc": "2026-10-21T13:30:00Z", "timeframe": "2"}, "price": {"close": 1.08}, "indicators": {"atr": 0.0005}}
        s = snapshot_from_v3_payload(p, authoritative_instrument="GBPUSD")
        self.assertEqual(s.instrument, "GBP_USD")   # authoritative override kept
        self.assertEqual(L.normalize_model(p), "trend_pullback")

    def test_calibration(self):
        rows = [{"predicted_win_probability": 0.55, "actual_outcome": "win"}, {"predicted_win_probability": 0.57, "actual_outcome": "loss"}]
        self.assertEqual(calibration_deciles(rows)["50-60%"]["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ===== FILE: backtest_gbpusd.py =====
"""GBP/USD 2m research backtest: v3-style baseline vs Manus v4 decide().
Data: Claude archive snapshot (same file as the overnight study). Costs:
1.5 pip round-trip charged on every trade. SL-first on ambiguous bars.
Simulated quotes (close +/- 0.7 pip) - NOT real broker quotes."""
import sys, json, math
sys.path.insert(0, '/workspace/stage0/manus_v4')
src = open('/workspace/stage0/t63/work/study.py').read().split("res={'bars'")[0]
exec(src)
from manus_v4.analysis import Snapshot
from manus_v4.engine import decide, Config
from manus_v4.instruments import REGISTRY
INST = REGISTRY['GBP_USD']
# 60m completed-bar EMA50 bias
bk60 = [int(t.timestamp() // 3600) for t in T]
h60c = [None]*N; h60e = [None]*N
a = 2/51; le = None; pb = None; blc = None; dc = de = None
for i in range(N):
    if pb is not None and bk60[i] != pb:
        le = blc if le is None else a*blc + (1-a)*le; dc, de = blc, le
    blc = C[i]; pb = bk60[i]; h60c[i] = dc; h60e[i] = de

def sim(i, d, sl, tp, cost_pips=1.5, hold=30):
    e = C[i]; risk = abs(e - sl)
    for j in range(i+1, min(i+1+hold, N)):
        if (T[j]-T[j-1]) > timedelta(minutes=10): return None
        if d == 'long':
            if L[j] <= sl: return -1 - cost_pips*PIP/risk, risk
            if H[j] >= tp: return (tp-e)/risk - cost_pips*PIP/risk, risk
        else:
            if H[j] >= sl: return -1 - cost_pips*PIP/risk, risk
            if L[j] <= tp: return (e-tp)/risk - cost_pips*PIP/risk, risk
    j = min(i+hold, N-1)
    return ((C[j]-e) if d == 'long' else (e-C[j]))/risk - cost_pips*PIP/risk, risk

def bias(c, e):
    if c is None or e is None: return 'unknown'
    return 'bullish' if c > e else 'bearish'

def run_v4(cfg, hold=30, stop_mode='v4'):
    out = []; last = -999; day = None; cnt = 0
    for i in range(250, N-hold-5):
        dd = T[i].date()
        if dd != day: day = dd; cnt = 0
        if i - last < 5 or cnt >= 18: continue
        c = cand(i, 'done')
        if not c: continue
        d, mdl = c
        model = {'A': 'trend_pullback', 'B': 'momentum_continuation', 'C': 'range_reversal'}[mdl]
        # look for confirmation within 3 bars
        for k in range(1, 4):
            j = i + k
            conf = C[j] > H[i] if d == 'long' else C[j] < L[i]
            if conf: break
        else:
            continue
        if (T[j]-T[i]) > timedelta(minutes=10): continue
        lb = 12
        s = Snapshot(instrument='GBP_USD', direction=d, entry_model=model, bar_time_utc=T[j].isoformat(), timeframe='2m',
                     open=O[j], high=H[j], low=L[j], close=C[j], atr=ATR[j], prev_high=H[i], prev_low=L[i], atr_avg=ATR50[j],
                     adx=ADX[j], plus_di=PDI[j], minus_di=MDI[j], rsi=RSI[j], rsi_prev=RSI[j-1], macd_hist=HIST[j], macd_hist_prev=HIST[j-1],
                     ema_fast=e20[j], ema_slow=e50[j], ema_trend=e200[j], bb_upper=BBU[j], bb_lower=BBL[j],
                     nearest_support=min(L[j-lb+1:j+1]), nearest_resistance=max(H[j-lb+1:j+1]),
                     htf=[{'tf': '10m', 'bias': bias(htf_close_done[j], htf_ema_done[j]), 'completed': True},
                          {'tf': '60m', 'bias': bias(h60c[j], h60e[j]), 'completed': True}],
                     bid=C[j]-0.7*PIP, ask=C[j]+0.7*PIP, quote_is_real=True, quote_age_seconds=1, confirmed=True, bars_since_arm=k,
                     pullback_touched_ema=(min(L[i-2:i+1]) <= e20[i]) if d == 'long' else (max(H[i-2:i+1]) >= e20[i]),
                     context_status='clear', signal_id=f'bt{j}')
        dec = decide(s, {'balance': 10000, 'currency': 'GBP', 'margin_available': 9000}, {'GBP_USD': C[j]}, cfg)
        if dec['action'] != 'APPROVE_SHADOW': continue
        plan = dec['final_trade_plan']
        r = sim(j, d, float(plan['stop_loss']), float(plan['take_profit']), hold=hold)
        if r is None: continue
        out.append({'R': r[0], 'riskp': r[1]/PIP, 'month': T[j].strftime('%Y-%m'), 'm': mdl, 'p': dec['probability'], 'gross': r[0] + 1.5*PIP/r[1]})
        last = j; cnt += 1
    return out

def st(x):
    n = len(x)
    if not n: return {'n': 0}
    R = [t['R'] for t in x]; mu = sum(R)/n; sd = math.sqrt(sum((r-mu)**2 for r in R)/(n-1)) if n > 1 else 0
    return {'n': n, 'win': round(sum(1 for t in x if t['R'] > 0)/n, 3), 'expR_net': round(mu, 3), 'se': round(sd/math.sqrt(n), 3),
            'expR_gross': round(sum(t['gross'] for t in x)/n, 3), 'median_stop_pips': round(sorted(t['riskp'] for t in x)[n//2], 1),
            'totalR_net': round(sum(R), 1)}

def sp(x):
    g = {'IS Apr-Jun': [], 'VAL Jul-Aug': [], 'OOS Sep+': []}
    for t in x:
        g['IS Apr-Jun' if t['month'] <= '2026-06' else 'VAL Jul-Aug' if t['month'] <= '2026-08' else 'OOS Sep+'].append(t)
    return {k: st(v) for k, v in g.items()}

res = {'bars': N, 'first': ts[0], 'last': ts[-1], 'cost_pips': 1.5}
b = run('dev'); 
for t in b: t['gross'] = t['gross']
res['baseline_v3_rules'] = st([dict(R=t['R'], riskp=t['riskp'], month=t['month'], gross=t['gross']) for t in b])
res['baseline_v3_rules_split'] = sp([dict(R=t['R'], riskp=t['riskp'], month=t['month'], gross=t['gross']) for t in b])
gates = Config(min_bucket_n=0, min_ev_r=-99.0)
gates.__dict__['_label'] = 'structural'
# disable EV/prob gate: patch policy floor via min_probability through a huge negative EV and zero prob floor
import manus_v4.legacy_v3 as Lg
from dataclasses import replace
saved = dict(Lg.MODEL_POLICIES)
for k, v in saved.items(): Lg.MODEL_POLICIES[k] = replace(v, min_probability=0.0, min_expected_net_r=-99.0)
x = run_v4(gates); res['v4_structural_gates'] = st(x); res['v4_structural_gates_split'] = sp(x)
res['v4_structural_by_model'] = {m: st([t for t in x if t['m'] == m]) for m in 'ABC'}
x60 = run_v4(gates, hold=60); res['v4_structural_hold60'] = st(x60); res['v4_structural_hold60_split'] = sp(x60)
for k, v in saved.items(): Lg.MODEL_POLICIES[k] = v
ev = Config(min_bucket_n=0)
y = run_v4(ev); res['v4_with_ev_gate'] = st(y); res['v4_with_ev_gate_split'] = sp(y)
json.dump(res, open('/workspace/stage0/manus_v4/backtest_results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
