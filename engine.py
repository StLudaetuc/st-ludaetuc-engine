
"""
St Ludaetuc Manus Engine v3.0.0
Multi-instrument cognitive decision engine for:
GBP/USD, EUR/USD, AUD/USD, USD/CAD, USD/JPY, XAG/USD, XAU/USD.

TradingView remains a candidate generator.
Manus remains the final decision authority.

Run:
    uvicorn manus_engine_multi_asset_v3:app --host 0.0.0.0 --port 10000

Endpoints:
    GET  /health
    POST /evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, isfinite
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field


ENGINE_VERSION = "3.0.0"
RULESET_VERSION = "manus_ruleset_multi_asset_2026_07_08_v3"


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    """
    Accepts either:
      1. direct canonical payload body
      2. {"payload": {...}, "account": {...}}
    """
    payload: Optional[Dict[str, Any]] = None
    account: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    display_name: str
    asset_class: str
    broker_symbol: str
    base_currency: str
    quote_currency: str
    pip_size: float
    tick_size: float
    contract_size: float
    unit_step: float
    min_units: float
    max_units: float

    # Empirical / operating thresholds.
    normal_atr_min: float
    normal_atr_max: float
    high_atr_max: float
    max_spread: float
    max_slippage: float

    min_rr: float
    target_rr_floor: float
    default_risk_percent: float
    max_risk_percent: float

    stop_atr_min_mult: float
    stop_atr_max_mult: float
    default_stop_atr_mult: float

    preferred_sessions: Tuple[str, ...]
    blocked_sessions: Tuple[str, ...]


# Thresholds for FX pairs are derived from the uploaded 5m market-data CSV distributions:
# median ATR / upper quartile ATR / p90 ATR / common spread and slippage.
# XAU/XAG use conservative defaults because no metal CSV was attached in this request.
INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "GBP_USD": InstrumentConfig(
        symbol="GBPUSD", display_name="GBP/USD", asset_class="forex", broker_symbol="GBP_USD",
        base_currency="GBP", quote_currency="USD", pip_size=0.0001, tick_size=0.00001,
        contract_size=100000, unit_step=1, min_units=1, max_units=200000,
        normal_atr_min=0.00025, normal_atr_max=0.00070, high_atr_max=0.00105,
        max_spread=0.00030, max_slippage=0.00020,
        min_rr=1.25, target_rr_floor=1.45, default_risk_percent=0.35, max_risk_percent=0.50,
        stop_atr_min_mult=1.00, stop_atr_max_mult=2.25, default_stop_atr_mult=1.35,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("asia", "overnight", "unknown", "OffHours"),
    ),
    "EUR_USD": InstrumentConfig(
        symbol="EURUSD", display_name="EUR/USD", asset_class="forex", broker_symbol="EUR_USD",
        base_currency="EUR", quote_currency="USD", pip_size=0.0001, tick_size=0.00001,
        contract_size=100000, unit_step=1, min_units=1, max_units=200000,
        normal_atr_min=0.00018, normal_atr_max=0.00060, high_atr_max=0.00090,
        max_spread=0.00030, max_slippage=0.00020,
        min_rr=1.20, target_rr_floor=1.45, default_risk_percent=0.35, max_risk_percent=0.50,
        stop_atr_min_mult=1.00, stop_atr_max_mult=2.20, default_stop_atr_mult=1.35,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("asia", "overnight", "unknown", "OffHours"),
    ),
    "AUD_USD": InstrumentConfig(
        symbol="AUDUSD", display_name="AUD/USD", asset_class="forex", broker_symbol="AUD_USD",
        base_currency="AUD", quote_currency="USD", pip_size=0.0001, tick_size=0.00001,
        contract_size=100000, unit_step=1, min_units=1, max_units=200000,
        normal_atr_min=0.00016, normal_atr_max=0.00050, high_atr_max=0.00080,
        max_spread=0.00035, max_slippage=0.00020,
        min_rr=1.20, target_rr_floor=1.45, default_risk_percent=0.30, max_risk_percent=0.45,
        stop_atr_min_mult=1.05, stop_atr_max_mult=2.30, default_stop_atr_mult=1.40,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("overnight", "unknown", "OffHours"),
    ),
    "USD_CAD": InstrumentConfig(
        symbol="USDCAD", display_name="USD/CAD", asset_class="forex", broker_symbol="USD_CAD",
        base_currency="USD", quote_currency="CAD", pip_size=0.0001, tick_size=0.00001,
        contract_size=100000, unit_step=1, min_units=1, max_units=200000,
        normal_atr_min=0.00018, normal_atr_max=0.00055, high_atr_max=0.00085,
        max_spread=0.00035, max_slippage=0.00020,
        min_rr=1.25, target_rr_floor=1.50, default_risk_percent=0.30, max_risk_percent=0.45,
        stop_atr_min_mult=1.05, stop_atr_max_mult=2.35, default_stop_atr_mult=1.40,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("asia", "overnight", "unknown", "OffHours"),
    ),
    "USD_JPY": InstrumentConfig(
        symbol="USDJPY", display_name="USD/JPY", asset_class="forex", broker_symbol="USD_JPY",
        base_currency="USD", quote_currency="JPY", pip_size=0.01, tick_size=0.001,
        contract_size=100000, unit_step=1, min_units=1, max_units=200000,
        normal_atr_min=0.035, normal_atr_max=0.105, high_atr_max=0.160,
        max_spread=0.030, max_slippage=0.020,
        min_rr=1.25, target_rr_floor=1.50, default_risk_percent=0.30, max_risk_percent=0.45,
        stop_atr_min_mult=1.05, stop_atr_max_mult=2.40, default_stop_atr_mult=1.45,
        preferred_sessions=("asia", "london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("overnight", "unknown", "OffHours"),
    ),
    "XAU_USD": InstrumentConfig(
        symbol="XAUUSD", display_name="XAU/USD", asset_class="metals", broker_symbol="XAU_USD",
        base_currency="XAU", quote_currency="USD", pip_size=0.01, tick_size=0.01,
        contract_size=1, unit_step=1, min_units=1, max_units=100,
        normal_atr_min=1.50, normal_atr_max=8.00, high_atr_max=15.00,
        max_spread=0.60, max_slippage=0.30,
        min_rr=1.35, target_rr_floor=1.60, default_risk_percent=0.20, max_risk_percent=0.35,
        stop_atr_min_mult=1.20, stop_atr_max_mult=2.80, default_stop_atr_mult=1.65,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("asia", "overnight", "unknown", "OffHours"),
    ),
    "XAG_USD": InstrumentConfig(
        symbol="XAGUSD", display_name="XAG/USD", asset_class="metals", broker_symbol="XAG_USD",
        base_currency="XAG", quote_currency="USD", pip_size=0.001, tick_size=0.001,
        contract_size=1, unit_step=1, min_units=1, max_units=5000,
        normal_atr_min=0.020, normal_atr_max=0.120, high_atr_max=0.220,
        max_spread=0.050, max_slippage=0.025,
        min_rr=1.35, target_rr_floor=1.60, default_risk_percent=0.20, max_risk_percent=0.35,
        stop_atr_min_mult=1.20, stop_atr_max_mult=2.80, default_stop_atr_mult=1.70,
        preferred_sessions=("london", "overlap_london_new_york", "new_york", "London"),
        blocked_sessions=("asia", "overnight", "unknown", "OffHours"),
    ),
}

ALIASES = {
    "GBPUSD": "GBP_USD", "GBP/USD": "GBP_USD", "GBP_USD": "GBP_USD",
    "EURUSD": "EUR_USD", "EUR/USD": "EUR_USD", "EUR_USD": "EUR_USD",
    "AUDUSD": "AUD_USD", "AUD/USD": "AUD_USD", "AUD_USD": "AUD_USD",
    "USDCAD": "USD_CAD", "USD/CAD": "USD_CAD", "USD_CAD": "USD_CAD",
    "USDJPY": "USD_JPY", "USD/JPY": "USD_JPY", "USD_JPY": "USD_JPY",
    "XAUUSD": "XAU_USD", "XAU/USD": "XAU_USD", "XAU_USD": "XAU_USD",
    "XAGUSD": "XAG_USD", "XAG/USD": "XAG_USD", "XAG_USD": "XAG_USD",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deep_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(obj: Dict[str, Any], path: str, value: Any) -> None:
    cur = obj
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
        v = float(x)
        return v if isfinite(v) else default
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    return str(x)


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 10)


def normalize_symbol(payload: Dict[str, Any]) -> Optional[str]:
    candidates = [
        deep_get(payload, "instrument.broker_symbol"),
        deep_get(payload, "instrument.symbol"),
        deep_get(payload, "instrument.display_name"),
        deep_get(payload, "asset_specific.fx.broker_symbol"),
    ]
    for raw in candidates:
        if not raw:
            continue
        key = str(raw).upper().replace(" ", "")
        if key in ALIASES:
            return ALIASES[key]
        compact = key.replace("_", "").replace("/", "")
        if compact in ALIASES:
            return ALIASES[compact]
    return None


def canonical_payload(req: EvaluateRequest) -> Dict[str, Any]:
    body = req.model_dump(exclude_none=True)
    if isinstance(req.payload, dict):
        return req.payload
    # If no wrapper was used, treat entire request body as the payload.
    body.pop("payload", None)
    body.pop("account", None)
    return body


def get_account_equity(account: Optional[Dict[str, Any]], payload: Dict[str, Any]) -> float:
    account = account or {}
    for key in ("equity", "balance", "NAV", "nav", "marginAvailable", "account_equity"):
        v = as_float(account.get(key))
        if v and v > 0:
            return v
    for path in ("account.equity", "account.balance", "risk.account_equity", "extensions.account.equity"):
        v = as_float(deep_get(payload, path))
        if v and v > 0:
            return v
    return 10_000.0


# -----------------------------------------------------------------------------
# Cognitive scoring
# -----------------------------------------------------------------------------
def score_signal_quality(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[int, Dict[str, Any]]:
    direction = as_str(deep_get(payload, "signal.direction"), "neutral").lower()
    rsi = as_float(deep_get(payload, "indicators.rsi"))
    ema_fast = as_float(deep_get(payload, "indicators.ema_fast"))
    ema_slow = as_float(deep_get(payload, "indicators.ema_slow"))
    macd_hist = as_float(deep_get(payload, "indicators.macd_histogram"))
    trend_bias = as_str(deep_get(payload, "structure.trend_bias"), "unknown").lower()
    htf_bias = as_str(deep_get(payload, "structure.higher_timeframe_bias"), "unknown").lower()
    entry_model = as_str(deep_get(payload, "signal.entry_model"), "unknown").lower()
    strength = as_str(deep_get(payload, "signal.strength_label"), "unknown").lower()

    score = 50
    reasons = []

    if direction not in ("long", "short"):
        return 0, {"fail": True, "reason": "Signal direction is not long or short."}

    aligned_ema = (
        direction == "long" and ema_fast is not None and ema_slow is not None and ema_fast >= ema_slow
    ) or (
        direction == "short" and ema_fast is not None and ema_slow is not None and ema_fast <= ema_slow
    )
    if aligned_ema:
        score += 12; reasons.append("EMA alignment supports direction.")
    else:
        score -= 12; reasons.append("EMA alignment does not support direction.")

    aligned_htf = (
        direction == "long" and htf_bias == "bullish"
    ) or (
        direction == "short" and htf_bias == "bearish"
    )
    mixed_htf = htf_bias in ("mixed", "sideways", "unknown")
    if aligned_htf:
        score += 18; reasons.append("Higher timeframe bias supports direction.")
    elif mixed_htf:
        score -= 5; reasons.append("Higher timeframe bias is mixed/unknown.")
    else:
        score -= 22; reasons.append("Higher timeframe bias conflicts with direction.")

    aligned_trend = (
        direction == "long" and trend_bias == "bullish"
    ) or (
        direction == "short" and trend_bias == "bearish"
    )
    if aligned_trend:
        score += 10; reasons.append("Trend bias supports direction.")
    elif trend_bias == "mixed":
        score -= 3; reasons.append("Trend bias is mixed.")
    else:
        score -= 12; reasons.append("Trend bias conflicts with direction.")

    if rsi is not None:
        if direction == "long":
            if 38 <= rsi <= 62:
                score += 8; reasons.append("RSI is in constructive long range.")
            elif rsi > 72:
                score -= 10; reasons.append("RSI is extended for long.")
        if direction == "short":
            if 38 <= rsi <= 62:
                score += 8; reasons.append("RSI is in constructive short range.")
            elif rsi < 28:
                score -= 10; reasons.append("RSI is extended for short.")

    if macd_hist is not None:
        if (direction == "long" and macd_hist >= 0) or (direction == "short" and macd_hist <= 0):
            score += 7; reasons.append("MACD histogram supports direction.")
        else:
            score -= 5; reasons.append("MACD histogram is adverse/early.")

    if "pullback" in entry_model:
        score += 5
    elif "mean_reversion" in entry_model:
        score += 3
    elif "momentum" in entry_model:
        score += 1

    if strength in ("strong", "very_strong"):
        score += 5
    elif strength in ("weak", "very_weak"):
        score -= 8

    return max(0, min(100, score)), {"reasons": reasons}


def score_risk(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[int, Dict[str, Any]]:
    atr = as_float(deep_get(payload, "indicators.atr"))
    rr = as_float(deep_get(payload, "risk.rr_ratio"))
    proposed_entry = as_float(deep_get(payload, "risk.proposed_entry")) or as_float(deep_get(payload, "price.close"))
    stop = as_float(deep_get(payload, "risk.proposed_stop_loss"))
    take_profit = as_float(deep_get(payload, "risk.proposed_take_profit"))
    spread = as_float(deep_get(payload, "price.spread_price")) or as_float(deep_get(payload, "risk.estimated_spread_cost")) or as_float(deep_get(payload, "extensions.assumed_spread_price"))
    slippage = as_float(deep_get(payload, "risk.estimated_slippage")) or as_float(deep_get(payload, "extensions.assumed_slippage_price"))
    risk_percent = as_float(deep_get(payload, "risk.risk_percent"), cfg.default_risk_percent)

    score = 50
    reasons = []

    if proposed_entry is None or stop is None or take_profit is None:
        return 0, {"fail": True, "reason": "Missing entry, stop loss, or take profit proposal."}

    stop_distance = abs(proposed_entry - stop)
    target_distance = abs(take_profit - proposed_entry)
    if stop_distance <= 0 or target_distance <= 0:
        return 0, {"fail": True, "reason": "Invalid stop/target geometry."}

    if rr is None:
        rr = target_distance / stop_distance

    if rr >= cfg.target_rr_floor:
        score += 18; reasons.append("RR is above preferred floor.")
    elif rr >= cfg.min_rr:
        score += 8; reasons.append("RR is acceptable but not ideal.")
    else:
        score -= 30; reasons.append("RR is below instrument minimum.")

    if atr is not None and atr > 0:
        stop_atr = stop_distance / atr
        if cfg.stop_atr_min_mult <= stop_atr <= cfg.stop_atr_max_mult:
            score += 12; reasons.append("Stop distance is ATR-consistent.")
        elif stop_atr < cfg.stop_atr_min_mult:
            score -= 16; reasons.append("Stop is too tight relative to ATR.")
        else:
            score -= 8; reasons.append("Stop is wide relative to ATR.")
        if cfg.normal_atr_min <= atr <= cfg.normal_atr_max:
            score += 10; reasons.append("ATR is normal for instrument.")
        elif atr <= cfg.high_atr_max:
            score += 2; reasons.append("ATR is elevated but tradable.")
        else:
            score -= 18; reasons.append("ATR is extreme for instrument.")

    if spread is not None:
        if spread <= cfg.max_spread:
            score += 5; reasons.append("Spread is acceptable.")
        else:
            score -= 20; reasons.append("Spread exceeds instrument limit.")

    if slippage is not None:
        if slippage <= cfg.max_slippage:
            score += 3; reasons.append("Slippage allowance is acceptable.")
        else:
            score -= 12; reasons.append("Slippage allowance exceeds instrument limit.")

    if risk_percent is not None:
        if risk_percent <= cfg.max_risk_percent:
            score += 5; reasons.append("Risk percent is within limit.")
        else:
            score -= 30; reasons.append("Risk percent exceeds Manus max.")

    return max(0, min(100, score)), {
        "reasons": reasons,
        "rr": rr,
        "stop_distance": stop_distance,
        "target_distance": target_distance,
        "risk_percent": risk_percent,
    }


def score_context(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[int, Dict[str, Any]]:
    context_status = as_str(deep_get(payload, "context.context_status"), "unchecked").lower()
    high_impact = deep_get(payload, "context.high_impact_event_nearby")
    mins_to_event = as_float(deep_get(payload, "context.minutes_to_next_high_impact_event"))
    session = as_str(deep_get(payload, "market.session_name"), "unknown")
    rollover = bool(deep_get(payload, "market.rollover_window_flag", False))

    score = 55
    reasons = []

    if rollover:
        return 0, {"fail": True, "reason": "Rollover window is blocked."}

    if context_status == "blocked":
        return 0, {"fail": True, "reason": "Context status is blocked."}
    if context_status == "clear":
        score += 15; reasons.append("External context is clear.")
    elif context_status in ("warning",):
        score -= 18; reasons.append("External context warning.")
    else:
        score -= 3; reasons.append("External context unchecked.")

    if high_impact is True and (mins_to_event is None or mins_to_event <= 30):
        return 0, {"fail": True, "reason": "High-impact event is too close."}
    if high_impact is True and mins_to_event is not None and mins_to_event <= 60:
        score -= 20; reasons.append("High-impact event within 60 minutes.")

    if session in cfg.preferred_sessions:
        score += 15; reasons.append("Session is preferred for instrument.")
    elif session in cfg.blocked_sessions:
        score -= 30; reasons.append("Session is blocked or historically weak.")
    else:
        score -= 5; reasons.append("Session is neutral/unknown for instrument.")

    return max(0, min(100, score)), {"reasons": reasons}


def score_strategy_fit(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[int, Dict[str, Any]]:
    vol_regime = as_str(deep_get(payload, "structure.volatility_regime"), "unknown").lower()
    market_regime = as_str(deep_get(payload, "structure.market_regime"), "unknown").lower()
    direction = as_str(deep_get(payload, "signal.direction"), "neutral").lower()
    htf = as_str(deep_get(payload, "structure.higher_timeframe_bias"), "unknown").lower()

    score = 50
    reasons = []

    if vol_regime == "normal":
        score += 20; reasons.append("Normal volatility regime.")
    elif vol_regime in ("low", "high"):
        score += 5; reasons.append("Tradable but non-ideal volatility.")
    elif vol_regime in ("very_low", "extreme", "unknown"):
        score -= 20; reasons.append("Weak volatility regime for execution.")

    if market_regime in ("trend_pullback", "range_to_reversal", "trend"):
        score += 10; reasons.append("Market regime matches supported models.")
    elif market_regime in ("breakout", "range_to_breakout"):
        score += 2; reasons.append("Breakout regime is allowed with caution.")
    else:
        score -= 5; reasons.append("Market regime is unclear.")

    if (direction == "long" and htf == "bullish") or (direction == "short" and htf == "bearish"):
        score += 15; reasons.append("Direction matches HTF.")
    elif htf == "mixed":
        score -= 5; reasons.append("HTF is mixed.")
    else:
        score -= 20; reasons.append("Counter-HTF trade.")

    return max(0, min(100, score)), {"reasons": reasons}


def expected_value_score(signal: int, risk: int, context: int, fit: int) -> int:
    return max(0, min(100, round(signal * 0.30 + risk * 0.30 + context * 0.20 + fit * 0.20)))


# -----------------------------------------------------------------------------
# Execution planning
# -----------------------------------------------------------------------------
def plan_trade(payload: Dict[str, Any], cfg: InstrumentConfig, account: Optional[Dict[str, Any]], risk_diag: Dict[str, Any]) -> Dict[str, Any]:
    direction = as_str(deep_get(payload, "signal.direction"), "neutral").lower()
    entry = as_float(deep_get(payload, "risk.proposed_entry")) or as_float(deep_get(payload, "price.close"))
    atr = as_float(deep_get(payload, "indicators.atr"))
    proposed_stop = as_float(deep_get(payload, "risk.proposed_stop_loss"))
    proposed_tp = as_float(deep_get(payload, "risk.proposed_take_profit"))
    risk_percent = as_float(deep_get(payload, "risk.risk_percent"), cfg.default_risk_percent) or cfg.default_risk_percent
    risk_percent = min(risk_percent, cfg.max_risk_percent)

    if entry is None:
        raise ValueError("Cannot plan trade without entry/close price.")

    # Use Pine proposal if valid; otherwise create ATR-consistent fixed SL/TP.
    stop = proposed_stop
    take_profit = proposed_tp

    if stop is None or take_profit is None:
        if atr is None or atr <= 0:
            raise ValueError("Cannot create SL/TP without ATR or valid Pine proposal.")
        stop_distance = atr * cfg.default_stop_atr_mult
        if direction == "long":
            stop = entry - stop_distance
            take_profit = entry + stop_distance * cfg.target_rr_floor
        elif direction == "short":
            stop = entry + stop_distance
            take_profit = entry - stop_distance * cfg.target_rr_floor
        else:
            raise ValueError("Cannot plan trade for neutral direction.")

    stop = round_to_tick(stop, cfg.tick_size)
    take_profit = round_to_tick(take_profit, cfg.tick_size)
    entry = round_to_tick(entry, cfg.tick_size)

    stop_distance = abs(entry - stop)
    target_distance = abs(take_profit - entry)
    rr = target_distance / stop_distance if stop_distance > 0 else 0

    # If the proposed RR is below the target floor, adjust TP only.
    if rr < cfg.target_rr_floor and stop_distance > 0:
        if direction == "long":
            take_profit = round_to_tick(entry + stop_distance * cfg.target_rr_floor, cfg.tick_size)
        else:
            take_profit = round_to_tick(entry - stop_distance * cfg.target_rr_floor, cfg.tick_size)
        target_distance = abs(take_profit - entry)
        rr = target_distance / stop_distance

    equity = get_account_equity(account, payload)
    risk_cash = equity * risk_percent / 100.0
    raw_units = risk_cash / stop_distance if stop_distance > 0 else 0.0

    # Conservative sizing by instrument contract/unit model.
    units = floor(raw_units / cfg.unit_step) * cfg.unit_step
    units = max(cfg.min_units, min(cfg.max_units, units))
    signed_units = -units if direction == "short" else units
    lots = signed_units / cfg.contract_size if cfg.contract_size else signed_units

    return {
        "final_entry": entry,
        "final_stop_loss": stop,
        "final_take_profit": take_profit,
        "final_position_size": signed_units,
        "final_position_size_lots": lots,
        "final_rr_ratio": rr,
        "risk_percent": risk_percent,
        "risk_cash": risk_cash,
        "equity_used": equity,
        "fixed_exit_model": True,
    }


def hard_rejection(payload: Dict[str, Any], cfg: InstrumentConfig, scores: Dict[str, int], diagnostics: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    direction = as_str(deep_get(payload, "signal.direction"), "neutral").lower()
    event_type = as_str(deep_get(payload, "event_type"), "signal").lower()
    if event_type not in ("signal", "decision"):
        return "schema", f"Unsupported event_type for evaluation: {event_type}"
    if direction not in ("long", "short"):
        return "signal", "No executable long/short direction."

    # Propagate fail from individual agents.
    for stage in ("signal", "risk", "context"):
        if diagnostics.get(stage, {}).get("fail"):
            return stage, diagnostics[stage].get("reason", f"{stage} agent failed.")

    if scores["signal_quality"] < 45:
        return "signal", "Signal quality score below minimum."
    if scores["risk"] < 45:
        return "risk", "Risk score below minimum."
    if scores["context"] < 40:
        return "context", "Context score below minimum."
    if scores["strategy_fit"] < 40:
        return "strategy_fit", "Strategy fit score below minimum."
    if scores["expected_value"] < 55:
        return "expected_value", "Expected value score below approval threshold."

    # Additional geometry check.
    rr = diagnostics.get("risk", {}).get("rr")
    if rr is not None and rr < cfg.min_rr:
        return "risk", f"RR {rr:.2f} below {cfg.symbol} minimum {cfg.min_rr:.2f}."

    return None


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="St Ludaetuc Manus Engine",
    version=ENGINE_VERSION,
    description="Multi-instrument Manus decision engine for St Ludaetuc.",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "supported_instruments": sorted(INSTRUMENTS.keys()),
        "timestamp_utc": now_utc(),
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    payload = canonical_payload(req)
    if not isinstance(payload, dict):
        return {
            "approval_status": "rejected",
            "rejection_stage": "schema",
            "rejection_reason": "Request body is not a valid payload object.",
            "engine_version": ENGINE_VERSION,
        }

    symbol_key = normalize_symbol(payload)
    if symbol_key is None or symbol_key not in INSTRUMENTS:
        return {
            "approval_status": "rejected",
            "rejection_stage": "instrument",
            "rejection_reason": "Unsupported or missing instrument symbol.",
            "engine_version": ENGINE_VERSION,
            "supported_instruments": sorted(INSTRUMENTS.keys()),
        }

    cfg = INSTRUMENTS[symbol_key]

    signal_score, signal_diag = score_signal_quality(payload, cfg)
    risk_score, risk_diag = score_risk(payload, cfg)
    context_score, context_diag = score_context(payload, cfg)
    fit_score, fit_diag = score_strategy_fit(payload, cfg)
    ev_score = expected_value_score(signal_score, risk_score, context_score, fit_score)

    scores = {
        "signal_quality": signal_score,
        "risk": risk_score,
        "context": context_score,
        "strategy_fit": fit_score,
        "expected_value": ev_score,
    }
    diagnostics = {
        "signal": signal_diag,
        "risk": risk_diag,
        "context": context_diag,
        "strategy_fit": fit_diag,
        "instrument_config": {
            "symbol": cfg.symbol,
            "broker_symbol": cfg.broker_symbol,
            "asset_class": cfg.asset_class,
            "min_rr": cfg.min_rr,
            "target_rr_floor": cfg.target_rr_floor,
            "max_spread": cfg.max_spread,
            "max_slippage": cfg.max_slippage,
            "normal_atr_min": cfg.normal_atr_min,
            "normal_atr_max": cfg.normal_atr_max,
            "high_atr_max": cfg.high_atr_max,
        },
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "evaluated_at_utc": now_utc(),
    }

    rejection = hard_rejection(payload, cfg, scores, diagnostics)

    result_payload = dict(payload)
    if rejection:
        stage, reason = rejection
        deep_set(result_payload, "manus.signal_quality_score", signal_score)
        deep_set(result_payload, "manus.risk_score", risk_score)
        deep_set(result_payload, "manus.context_score", context_score)
        deep_set(result_payload, "manus.strategy_fit_score", fit_score)
        deep_set(result_payload, "manus.expected_value_score", ev_score)
        deep_set(result_payload, "manus.confidence_score", ev_score)
        deep_set(result_payload, "manus.approval_status", "rejected")
        deep_set(result_payload, "manus.approval_reason", None)
        deep_set(result_payload, "manus.rejection_stage", stage)
        deep_set(result_payload, "manus.rejection_reason", reason)
        deep_set(result_payload, "manus.diagnostics", diagnostics)

        return {
            "approval_status": "rejected",
            "approval_reason": None,
            "rejection_stage": stage,
            "rejection_reason": reason,
            "signal_quality_score": signal_score,
            "risk_score": risk_score,
            "context_score": context_score,
            "strategy_fit_score": fit_score,
            "expected_value_score": ev_score,
            "confidence_score": ev_score,
            "final_entry": None,
            "final_stop_loss": None,
            "final_take_profit": None,
            "final_position_size": None,
            "final_position_size_lots": None,
            "final_rr_ratio": None,
            "final_trade_plan": None,
            "diagnostics": diagnostics,
            "payload": result_payload,
        }

    try:
        plan = plan_trade(payload, cfg, req.account, risk_diag)
    except Exception as exc:
        stage, reason = "execution_planning", str(exc)
        deep_set(result_payload, "manus.approval_status", "rejected")
        deep_set(result_payload, "manus.rejection_stage", stage)
        deep_set(result_payload, "manus.rejection_reason", reason)
        deep_set(result_payload, "manus.diagnostics", diagnostics)
        return {
            "approval_status": "rejected",
            "rejection_stage": stage,
            "rejection_reason": reason,
            "diagnostics": diagnostics,
            "payload": result_payload,
        }

    approval_reason = (
        f"Approved by Manus {ENGINE_VERSION}: {cfg.display_name} signal passed "
        f"signal/risk/context/strategy checks with confidence {ev_score}."
    )

    final_trade_plan = {
        "model": "fixed_entry_fixed_exit",
        "instrument": cfg.broker_symbol,
        "entry": plan["final_entry"],
        "stop_loss": plan["final_stop_loss"],
        "take_profit": plan["final_take_profit"],
        "position_size": plan["final_position_size"],
        "position_size_lots": plan["final_position_size_lots"],
        "rr_ratio": plan["final_rr_ratio"],
        "risk_percent": plan["risk_percent"],
        "risk_cash": plan["risk_cash"],
        "equity_used": plan["equity_used"],
        "no_trailing_stop": True,
        "no_breakeven_move": True,
        "no_partial_exit": True,
    }

    deep_set(result_payload, "manus.signal_quality_score", signal_score)
    deep_set(result_payload, "manus.risk_score", risk_score)
    deep_set(result_payload, "manus.context_score", context_score)
    deep_set(result_payload, "manus.strategy_fit_score", fit_score)
    deep_set(result_payload, "manus.execution_quality_score", risk_score)
    deep_set(result_payload, "manus.expected_value_score", ev_score)
    deep_set(result_payload, "manus.confidence_score", ev_score)
    deep_set(result_payload, "manus.approval_status", "approved")
    deep_set(result_payload, "manus.approval_reason", approval_reason)
    deep_set(result_payload, "manus.rejection_stage", None)
    deep_set(result_payload, "manus.rejection_reason", None)
    deep_set(result_payload, "manus.final_entry", plan["final_entry"])
    deep_set(result_payload, "manus.final_stop_loss", plan["final_stop_loss"])
    deep_set(result_payload, "manus.final_take_profit", plan["final_take_profit"])
    deep_set(result_payload, "manus.final_position_size", plan["final_position_size"])
    deep_set(result_payload, "manus.final_position_size_lots", plan["final_position_size_lots"])
    deep_set(result_payload, "manus.final_rr_ratio", plan["final_rr_ratio"])
    deep_set(result_payload, "manus.final_trade_plan", final_trade_plan)
    deep_set(result_payload, "manus.diagnostics", diagnostics)

    return {
        "approval_status": "approved",
        "approval_reason": approval_reason,
        "rejection_stage": None,
        "rejection_reason": None,
        "signal_quality_score": signal_score,
        "risk_score": risk_score,
        "context_score": context_score,
        "strategy_fit_score": fit_score,
        "execution_quality_score": risk_score,
        "expected_value_score": ev_score,
        "confidence_score": ev_score,
        "final_entry": plan["final_entry"],
        "final_stop_loss": plan["final_stop_loss"],
        "final_take_profit": plan["final_take_profit"],
        "final_position_size": plan["final_position_size"],
        "final_position_size_lots": plan["final_position_size_lots"],
        "final_rr_ratio": plan["final_rr_ratio"],
        "final_trade_plan": final_trade_plan,
        "diagnostics": diagnostics,
        "payload": result_payload,
    }
