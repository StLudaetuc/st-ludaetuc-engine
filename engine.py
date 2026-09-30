from __future__ import annotations

import hashlib
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor, isfinite
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from shadow_scorer import shadow_score_0_100

try:
    import psycopg2
    from psycopg2 import pool as _pg_pool
    from psycopg2.extras import Json as _PgJson
except Exception:  # pragma: no cover - psycopg2 should always be installed, but
    # the events pipeline is best-effort and must never take trading down.
    psycopg2 = None
    _pg_pool = None
    _PgJson = None


ENGINE_VERSION = "3.4.0"
RULESET_VERSION = "manus_ruleset_2026_09_v3_3"
CALIBRATION_VERSION = "heuristic_uncalibrated_v1"

# --- St Ludaetuc Market Data Centre / canonical events ledger -------------
#
# One generic `events` table (Cloud V1 charter, Canonical Event Ledger,
# section 6) backs three event types so far:
#   - "market_data"   : raw OHLCV ticks pushed directly from TradingView,
#                        never routed through Make (by explicit decision).
#   - "prediction"    : every /evaluate call's confidence/expected-value
#                        output, keyed by the signal's trade_id, so it can
#                        later be joined against...
#   - "trade_outcome" : ...the actual closed-trade result, to finally make
#                        `probability_is_calibrated` mean something.
#
# All DB access is best-effort from the trading path's point of view: a
# Postgres hiccup must never block or fail a /evaluate call. It's only
# allowed to be loud (500) on /market-data and /trade-outcome, since there
# logging the event *is* the whole point of the request.
EVENTS_SCHEMA_VERSION = "v1"
DATABASE_URL = os.environ.get("DATABASE_URL", "")

_db_pool = None


def _get_pool():
    global _db_pool
    if _db_pool is None and DATABASE_URL and _pg_pool is not None:
        try:
            _db_pool = _pg_pool.SimpleConnectionPool(1, 5, DATABASE_URL, connect_timeout=5)
        except Exception:
            logging.exception("Failed to create Postgres connection pool")
            _db_pool = None
    return _db_pool


_EVENTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    origin TEXT NOT NULL,
    destination TEXT,
    trade_id TEXT,
    instrument TEXT,
    timeframe TEXT,
    event_time TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    schema_version TEXT NOT NULL DEFAULT 'v1',
    strategy_version TEXT,
    model_version TEXT,
    environment TEXT NOT NULL DEFAULT 'production',
    status TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    related_experiment_id TEXT,
    related_deployment_id TEXT,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events (event_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_instrument_tf_time ON events (instrument, timeframe, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_trade_id ON events (trade_id);
"""


def _ensure_events_schema() -> None:
    pool = _get_pool()
    if pool is None:
        return
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_EVENTS_SCHEMA_SQL)
    finally:
        pool.putconn(conn)


def _parse_event_time(raw: Any) -> datetime:
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    s = str(raw).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        ms = float(s)
        ts = ms / 1000.0 if ms > 1e12 else ms
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def log_event(
    event_type: str,
    origin: str,
    event_time: datetime,
    payload: Dict[str, Any],
    *,
    event_id: Optional[str] = None,
    trade_id: Optional[str] = None,
    instrument: Optional[str] = None,
    timeframe: Optional[str] = None,
    destination: Optional[str] = None,
    strategy_version: Optional[str] = None,
    model_version: Optional[str] = None,
    status: Optional[str] = None,
    environment: str = "production",
    related_experiment_id: Optional[str] = None,
    related_deployment_id: Optional[str] = None,
) -> str:
    pool = _get_pool()
    if pool is None:
        raise RuntimeError("DATABASE_URL not configured or Postgres pool unavailable")

    if event_id is None:
        basis = f"{event_type}:{origin}:{instrument}:{timeframe}:{trade_id}:{event_time.isoformat()}"
        event_id = hashlib.sha256(basis.encode("utf-8")).hexdigest()

    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO events (
                    event_id, event_type, origin, destination, trade_id,
                    instrument, timeframe, event_time, schema_version,
                    strategy_version, model_version, environment, status,
                    related_experiment_id, related_deployment_id, payload
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    status = EXCLUDED.status,
                    received_at = now(),
                    retry_count = events.retry_count + 1
                """,
                (
                    event_id, event_type, origin, destination, trade_id,
                    instrument, timeframe, event_time, EVENTS_SCHEMA_VERSION,
                    strategy_version, model_version, environment, status,
                    related_experiment_id, related_deployment_id, _PgJson(payload),
                ),
            )
    finally:
        pool.putconn(conn)
    return event_id


def _normalize_tv_ticker(raw: Any) -> str:
    s = str(raw or "").upper().strip()
    if ":" in s:
        s = s.split(":")[-1]
    return s.replace("/", "").replace("_", "").replace("-", "").replace(" ", "")


def _normalize_tv_interval(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return f"{s}m" if s.isdigit() else s


class EvaluateRequest(BaseModel):
    payload: Optional[Dict[str, Any]] = None
    account: Optional[Dict[str, Any]] = None
    # Ground-truth pair symbol from the calling Make scenario (e.g. "AUDUSD"),
    # independent of whatever the Pine Script payload's instrument.* fields
    # say. Optional and backward compatible: callers that don't send it get
    # the old payload-self-reported behavior unchanged. See normalize_symbol().
    authoritative_instrument: Optional[str] = None
    model_config = {"extra": "allow"}


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    broker_symbol: str
    pip_size: float
    tick_size: float
    contract_size: float
    unit_step: float
    min_units: float
    max_units: float
    max_risk_percent: float
    max_spread_price: float
    max_slippage_price: float
    sl_atr_mult: float
    default_rr: float
    structure_buffer_pips: float
    preferred_sessions: Tuple[str, ...]
    blocked_sessions: Tuple[str, ...]


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


# structure_buffer_pips mirrors each Pine script's "Structure buffer, pips"
# input (candidate generator, section "12. PROPOSED STOP AND TARGET").
# Confirmed at 0.6 pips / 12-bar lookback for GBPUSD in
# STL_GBPUSD_2m_CandidateGenerator_v5.pine. The other pairs still run their
# pre-rebuild Pine scripts, so 0.6 is used here as the same base-template
# default for now - re-verify this value against each pair's script once it
# is rebuilt from the GBPUSD v5.0.0 template.
INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "GBPUSD": InstrumentConfig("GBPUSD", "GBP_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00025, 0.00012, 1.50, 1.60, 0.6, ("london", "overlap_london_new_york", "new_york"), ("overnight", "unknown")),
    "EURUSD": InstrumentConfig("EURUSD", "EUR_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00020, 0.00010, 1.40, 1.55, 0.6, ("london", "overlap_london_new_york", "new_york"), ("overnight", "unknown")),
    "AUDUSD": InstrumentConfig("AUDUSD", "AUD_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00025, 0.00012, 1.45, 1.55, 0.6, ("asia", "london", "overlap_london_new_york"), ("overnight", "unknown")),
    "USDCAD": InstrumentConfig("USDCAD", "USD_CAD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00030, 0.00015, 1.45, 1.55, 0.6, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "USDJPY": InstrumentConfig("USDJPY", "USD_JPY", 0.01, 0.001, 100000, 1, 1, 200000, 0.50, 0.030, 0.015, 1.45, 1.55, 0.6, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "XAUUSD": InstrumentConfig("XAUUSD", "XAU_USD", 0.1, 0.01, 1, 1, 1, 500, 0.35, 0.60, 0.30, 1.60, 1.70, 0.6, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "XAGUSD": InstrumentConfig("XAGUSD", "XAG_USD", 0.01, 0.001, 1, 1, 1, 5000, 0.35, 0.030, 0.015, 1.60, 1.70, 0.6, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
}

MODEL_POLICIES: Dict[str, ModelPolicy] = {
    "trend_pullback": ModelPolicy(True, False, 0.53, 0.05, 1.35, 72.0, ("trend", "trend_pullback"), ("extreme",)),
    "momentum_continuation": ModelPolicy(True, False, 0.55, 0.06, 1.40, 74.0, ("trend", "breakout"), ("range", "extreme")),
    "range_reversal": ModelPolicy(True, True, 0.58, 0.10, 1.50, 80.0, ("range", "range_to_reversal"), ("trend", "breakout", "extreme")),
    "bb_rsi_immediate_rejection": ModelPolicy(True, True, 0.60, 0.10, 1.50, 82.0, ("range", "range_to_reversal"), ("trend", "breakout", "extreme")),
    "bb_rsi_armed_reversal": ModelPolicy(True, True, 0.58, 0.08, 1.45, 80.0, ("range", "range_to_reversal"), ("breakout", "extreme")),
    "mean_reversion_with_trend_filter": ModelPolicy(False, True, 0.65, 0.15, 1.60, 90.0, ("range",), ("trend", "breakout", "extreme")),
    "unknown": ModelPolicy(False, True, 0.65, 0.15, 1.60, 90.0, tuple(), ("unknown", "extreme")),
}

REQUIRED_FIELDS = (
    "meta.signal_id",
    "system.strategy_id",
    "system.strategy_version",
    "instrument.symbol",
    "market.timeframe",
    "market.bar_time_utc",
    "signal.direction",
    "signal.entry_model",
    "price.close",
    "indicators.atr",
)


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
        value = float(x)
        return value if isfinite(value) else default
    except Exception:
        return default


def as_str(x: Any, default: str = "") -> str:
    return default if x is None else str(x)


def as_bool(x: Any, default: bool = False) -> bool:
    # Make.com's HTTP templates send every field as a JSON string (e.g. "true"),
    # not a native JSON boolean, so a strict `x is True` check silently never
    # fires. Coerce both real booleans and their string/number spellings.
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10) if tick > 0 else price


def round_to_step(units: float, step: float) -> float:
    return floor(units / step) * step if step > 0 else units


def normalize_symbol(payload: Dict[str, Any], authoritative: Optional[str] = None) -> Tuple[Optional[str], str]:
    # `authoritative` is a pair symbol supplied by the CALLER (Make), not by
    # the Pine Script alert payload. Each Make scenario already hardcodes its
    # own correct OANDA instrument for order placement (see modules
    # 101/102), so it can supply that same value here as ground truth. This
    # takes priority over the payload's self-reported instrument.* fields,
    # which come straight from the TradingView alert and have been found to
    # be wrong for pairs still running a pre-rebuild, cloned-from-EURUSD
    # Pine script (their instrument.symbol/broker_symbol/display_name say
    # "EURUSD" regardless of the pair actually being traded). Without this,
    # normalize_symbol() silently picks the wrong InstrumentConfig - wrong
    # spread/slippage limits, ATR stop multiplier, and session gating - for
    # every one of those mislabeled pairs. This is a mitigation, not a fix:
    # the payload's own instrument.* fields should still be corrected at the
    # Pine Script source; this just stops the engine from being fooled by
    # them in the meantime.
    if authoritative:
        key = str(authoritative).upper().replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
        if key in INSTRUMENTS:
            return key, "authoritative_override"

    for raw in (
        get_path(payload, "instrument.symbol"),
        get_path(payload, "instrument.broker_symbol"),
        get_path(payload, "instrument.display_name"),
    ):
        if raw:
            key = str(raw).upper().replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
            if key in INSTRUMENTS:
                return key, "payload_self_reported"
    return None, "unresolved"


def normalize_model(payload: Dict[str, Any]) -> str:
    raw = as_str(get_path(payload, "signal.entry_model"), "unknown").strip().lower()
    aliases = {
        "trend pullback": "trend_pullback",
        "momentum continuation": "momentum_continuation",
        "range reversal": "range_reversal",
        "bb_rsi_exhaustion_reversal": "bb_rsi_armed_reversal",
        "bb_rsi_reversal": "bb_rsi_armed_reversal",
    }
    model = aliases.get(raw, raw)
    return model if model in MODEL_POLICIES else "unknown"


def validate_data(payload: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
    missing = [p for p in REQUIRED_FIELDS if get_path(payload, p) in (None, "")]
    invalid = []
    if as_str(get_path(payload, "signal.direction"), "").lower() not in ("long", "short"):
        invalid.append("signal.direction")
    if (as_float(get_path(payload, "price.close"), 0.0) or 0.0) <= 0:
        invalid.append("price.close")
    if (as_float(get_path(payload, "indicators.atr"), 0.0) or 0.0) <= 0:
        invalid.append("indicators.atr")

    bid = as_float(get_path(payload, "price.bid"))
    ask = as_float(get_path(payload, "price.ask"))
    spread = as_float(get_path(payload, "price.spread_price"))
    execution_data_complete = bid is not None and ask is not None and ask >= bid and spread is not None

    context_status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
    score = 100.0 - 8.0 * len(missing) - 15.0 * len(invalid)
    if not execution_data_complete:
        score -= 20.0
    if context_status in ("unchecked", "unknown", ""):
        score -= 10.0

    return not missing and not invalid, clamp(score, 0.0, 100.0), {
        "missing_fields": missing,
        "invalid_fields": invalid,
        "execution_data_complete": execution_data_complete,
        "context_status": context_status,
    }


def signal_score(payload: Dict[str, Any], model: str) -> float:
    direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
    rsi = as_float(get_path(payload, "indicators.rsi"))
    ema_fast = as_float(get_path(payload, "indicators.ema_fast"))
    ema_slow = as_float(get_path(payload, "indicators.ema_slow"))
    macd = as_float(get_path(payload, "indicators.macd_histogram"))
    htf = as_str(get_path(payload, "structure.higher_timeframe_bias"), "unknown").lower()
    trend = as_str(get_path(payload, "structure.trend_bias"), "unknown").lower()
    vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()
    adx = as_float(get_path(payload, "indicators.adx"))
    candidate_strength = as_float(get_path(payload, "extensions.candidate_strength"), 50.0) or 50.0

    score = 45.0
    if direction == "long":
        if ema_fast is not None and ema_slow is not None:
            score += 8 if ema_fast >= ema_slow else -8
        score += 8 if trend == "bullish" else (-12 if trend == "bearish" else 0)
        score += 10 if htf == "bullish" else (-12 if htf == "bearish" else -3)
        score += 6 if rsi is not None and 35 <= rsi <= 62 else 0
        score += 4 if macd is not None and macd >= 0 else 0
    else:
        if ema_fast is not None and ema_slow is not None:
            score += 8 if ema_fast <= ema_slow else -8
        score += 8 if trend == "bearish" else (-12 if trend == "bullish" else 0)
        score += 10 if htf == "bearish" else (-12 if htf == "bullish" else -3)
        score += 6 if rsi is not None and 38 <= rsi <= 65 else 0
        score += 4 if macd is not None and macd <= 0 else 0

    if model == "trend_pullback":
        score += 6
    elif model == "momentum_continuation":
        score += 3
    elif model in ("range_reversal", "bb_rsi_immediate_rejection", "bb_rsi_armed_reversal"):
        score -= 5

    score += 6 if vol == "normal" else (1 if vol in ("low", "high") else -12)
    if adx is not None:
        if model in ("range_reversal", "bb_rsi_immediate_rejection", "bb_rsi_armed_reversal"):
            score += 6 if adx <= 25 else (-10 if adx >= 35 else 0)
        elif 15 <= adx <= 40:
            score += 5

    score += clamp((candidate_strength - 50.0) * 0.20, -8.0, 8.0)
    return clamp(score, 0.0, 100.0)


def context_score(payload: Dict[str, Any], cfg: InstrumentConfig) -> float:
    session = as_str(get_path(payload, "market.session_name"), "unknown").lower()
    status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
    event = as_bool(get_path(payload, "context.high_impact_event_nearby"), False)

    score = 50.0
    score += 15 if session in cfg.preferred_sessions else (-25 if session in cfg.blocked_sessions else -5)
    score += 20 if status == "clear" else (-15 if status in ("warning", "unchecked", "unknown", "") else -50)
    if event:
        score -= 35
    return clamp(score, 0.0, 100.0)


def fit_score(payload: Dict[str, Any], policy: ModelPolicy) -> float:
    market_regime = as_str(get_path(payload, "structure.market_regime"), "unknown").lower()
    vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()

    if not policy.enabled:
        return 0.0

    score = 50.0
    score += 20 if market_regime in policy.preferred_regimes else (-30 if market_regime in policy.blocked_regimes else 0)
    score += 10 if vol == "normal" else (2 if vol in ("low", "high") else -15)
    if policy.research_only:
        score -= 8
    return clamp(score, 0.0, 100.0)


def plan_trade(payload: Dict[str, Any], account: Dict[str, Any], cfg: InstrumentConfig, policy: ModelPolicy) -> Dict[str, Any]:
    direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
    entry = as_float(get_path(payload, "risk.proposed_entry")) or as_float(get_path(payload, "price.close")) or 0.0
    atr = as_float(get_path(payload, "indicators.atr"), 0.0) or 0.0

    # pine_proposal_only / allow_manus_override_sl_tp: when the candidate
    # generator marks its own proposed_stop_loss/proposed_take_profit as a
    # non-authoritative proposal (rather than a value to execute verbatim),
    # and explicitly grants the engine permission to override it, the engine
    # ignores Pine's numbers and computes its own stop/target below - the
    # same math already used as the missing-data fallback, just now applied
    # unconditionally rather than only when Pine sent nothing. This is what
    # makes Manus the actual execution authority for risk placement, instead
    # of a pass-through of Pine's suggestion.
    pine_proposal_only = as_bool(get_path(payload, "extensions.pine_proposal_only"), False)
    allow_override = as_bool(get_path(payload, "extensions.allow_manus_override_sl_tp"), False)
    engine_authoritative_sl_tp = pine_proposal_only and allow_override

    sl = None if engine_authoritative_sl_tp else as_float(get_path(payload, "risk.proposed_stop_loss"))
    tp = None if engine_authoritative_sl_tp else as_float(get_path(payload, "risk.proposed_take_profit"))
    source = "payload"
    stop_basis = None

    if sl is None or tp is None:
        # Mirrors Pine's own stop formula exactly (candidate generator
        # v5.0.0, section "12. PROPOSED STOP AND TARGET"): the wider of a
        # plain ATR stop and a structure stop set just beyond the recent
        # N-bar swing high/low plus a small buffer. structure.nearest_support
        # / structure.nearest_resistance are Pine's own rolling
        # ta.lowest(low, structureLookback) / ta.highest(high,
        # structureLookback) values (confirmed forwarded by Make module 98),
        # so this reproduces Pine's proposal rather than a cruder ATR-only
        # approximation - the engine's fallback/authoritative stop now
        # matches what Pine itself would have proposed.
        buffer_price = cfg.structure_buffer_pips * cfg.pip_size
        nearest_support = as_float(get_path(payload, "structure.nearest_support"))
        nearest_resistance = as_float(get_path(payload, "structure.nearest_resistance"))
        atr_dist = atr * cfg.sl_atr_mult if atr > 0 else entry * 0.001
        rr = max(cfg.default_rr, policy.min_rr)

        if direction == "long":
            atr_stop = entry - atr_dist
            if nearest_support is not None:
                structure_stop = nearest_support - buffer_price
                sl = min(atr_stop, structure_stop)
                stop_basis = "structure" if structure_stop < atr_stop else "atr"
            else:
                sl = atr_stop
                stop_basis = "atr"
            raw_stop_dist = entry - sl
            tp = entry + raw_stop_dist * rr
        elif direction == "short":
            atr_stop = entry + atr_dist
            if nearest_resistance is not None:
                structure_stop = nearest_resistance + buffer_price
                sl = max(atr_stop, structure_stop)
                stop_basis = "structure" if structure_stop > atr_stop else "atr"
            else:
                sl = atr_stop
                stop_basis = "atr"
            raw_stop_dist = sl - entry
            tp = entry - raw_stop_dist * rr
        else:
            sl, tp = None, None
        source = "manus_authoritative_atr_structure" if engine_authoritative_sl_tp else "manus_atr_structure_fallback"

    if sl is None or tp is None or entry <= 0:
        return {"entry": entry, "sl": None, "tp": None, "rr": None, "units": None, "lots": None, "source": source, "stop_basis": stop_basis}

    entry, sl, tp = round_to_tick(entry, cfg.tick_size), round_to_tick(sl, cfg.tick_size), round_to_tick(tp, cfg.tick_size)
    stop_dist = abs(entry - sl)
    target_dist = abs(tp - entry)
    rr = target_dist / stop_dist if stop_dist > 0 else None

    equity = as_float(account.get("balance")) or as_float(account.get("margin_available")) or 10000.0
    risk_pct = min(as_float(get_path(payload, "risk.risk_percent"), cfg.max_risk_percent) or cfg.max_risk_percent, cfg.max_risk_percent)
    risk_cash = equity * risk_pct / 100.0
    units = round_to_step(risk_cash / stop_dist, cfg.unit_step) if stop_dist > 0 else 0.0
    units = clamp(units, cfg.min_units, cfg.max_units)
    units = -abs(units) if direction == "short" else abs(units)

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "units": units,
        "lots": abs(units) / cfg.contract_size,
        "stop_distance": stop_dist,
        "risk_percent": risk_pct,
        "risk_cash": risk_cash,
        "source": source,
        "stop_basis": stop_basis,
    }


def cost_r(payload: Dict[str, Any], cfg: InstrumentConfig, plan: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    spread = as_float(get_path(payload, "price.spread_price"))
    if spread is None:
        bid = as_float(get_path(payload, "price.bid"))
        ask = as_float(get_path(payload, "price.ask"))
        if bid is not None and ask is not None and ask >= bid:
            spread = ask - bid

    spread_used = spread if spread is not None else cfg.max_spread_price
    slippage = as_float(get_path(payload, "execution.expected_slippage_price")) or as_float(get_path(payload, "risk.max_slippage_allowed")) or cfg.max_slippage_price
    commission = as_float(get_path(payload, "costs.commission_price_equivalent"), 0.0) or 0.0
    financing = as_float(get_path(payload, "costs.estimated_financing_price_equivalent"), 0.0) or 0.0
    stop_dist = as_float(plan.get("stop_distance"), 0.0) or 0.0

    total_price_cost = max(spread_used, 0.0) + max(slippage, 0.0) + max(commission, 0.0) + max(financing, 0.0)
    result = total_price_cost / stop_dist if stop_dist > 0 else 1.0

    return result, {
        "spread": spread,
        "spread_used": spread_used,
        "spread_estimated": spread is None,
        "slippage": slippage,
        "commission_price_equivalent": commission,
        "financing_price_equivalent": financing,
        "total_price_cost": total_price_cost,
        "cost_r": result,
    }


def risk_score(payload: Dict[str, Any], account: Dict[str, Any], cfg: InstrumentConfig, policy: ModelPolicy, plan: Dict[str, Any], costs: Dict[str, Any]) -> float:
    rr = as_float(plan.get("rr"), 0.0) or 0.0
    risk_pct = as_float(get_path(payload, "risk.risk_percent"), cfg.max_risk_percent) or cfg.max_risk_percent
    spread = as_float(costs.get("spread"))
    slippage = as_float(costs.get("slippage"), 0.0) or 0.0
    cr = as_float(costs.get("cost_r"), 1.0) or 1.0
    heat = as_float(account.get("portfolio_heat_percent"), 0.0) or 0.0
    open_trades = as_float(account.get("open_trade_count"), 0.0) or 0.0

    score = 60.0
    score += 15 if rr >= policy.min_rr else -30
    score += 8 if risk_pct <= cfg.max_risk_percent else -30
    score += -15 if spread is None else (8 if spread <= cfg.max_spread_price else -30)
    score += -12 if slippage > cfg.max_slippage_price else 0
    score += 8 if cr <= 0.15 else (2 if cr <= 0.30 else -20)
    score += -25 if heat > 3.0 else 0
    score += -15 if open_trades >= 4 else 0
    return clamp(score, 0.0, 100.0)


def estimate_probability(signal: float, context: float, fit: float, data_quality: float, policy: ModelPolicy) -> float:
    composite = signal * 0.35 + context * 0.20 + fit * 0.30 + data_quality * 0.15
    probability = 0.20 + (composite / 100.0) * 0.45
    if policy.research_only:
        probability -= 0.05
    if not policy.enabled:
        probability = 0.0
    return clamp(probability, 0.05, 0.75)


class TradingSignalEvaluationEngine:
    def evaluate(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        req = EvaluateRequest(**request_body)
        payload = req.payload if isinstance(req.payload, dict) else req.model_dump()
        account = req.account if isinstance(req.account, dict) else {}

        if not payload:
            return self._reject({}, "Invalid or empty payload.", "schema", "INVALID_PAYLOAD")

        symbol, instrument_resolved_via = normalize_symbol(payload, req.authoritative_instrument)
        if symbol is None:
            return self._reject(payload, "Unsupported or missing instrument.", "instrument", "UNSUPPORTED_INSTRUMENT")

        cfg = INSTRUMENTS[symbol]
        model = normalize_model(payload)
        policy = MODEL_POLICIES[model]

        dq_pass, dq_score, dq_diag = validate_data(payload)
        sig = signal_score(payload, model)
        ctx = context_score(payload, cfg)
        fit = fit_score(payload, policy)
        plan = plan_trade(payload, account, cfg, policy)
        cr, cost_diag = cost_r(payload, cfg, plan)
        risk = risk_score(payload, account, cfg, policy, plan, cost_diag)
        prob = estimate_probability(sig, ctx, fit, dq_score, policy)

        rr = as_float(plan.get("rr"), 0.0) or 0.0
        gross_ev_r = prob * rr - (1.0 - prob)
        net_ev_r = gross_ev_r - cr
        pre_gate_score = dq_score * 0.15 + sig * 0.25 + risk * 0.25 + ctx * 0.15 + fit * 0.20

        diagnostics = {
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "probability_is_calibrated": False,
            "instrument_resolved_via": instrument_resolved_via,
            "data_quality": dq_diag,
            "costs": cost_diag,
            "trade_plan_preview": plan,
            "policy": policy.__dict__,
        }

        rejection = self._hard_rejection(payload, cfg, model, policy, dq_pass, sig, risk, ctx, fit, plan, cost_diag)
        if rejection:
            stage, reason, code = rejection
            return self._result(payload, cfg, model, "rejected", stage, reason, code, dq_score, sig, risk, ctx, fit, pre_gate_score, prob, gross_ev_r, cr, net_ev_r, diagnostics, None)

        passed = (
            pre_gate_score >= policy.min_pre_gate_score
            and prob >= policy.min_probability
            and net_ev_r >= policy.min_expected_net_r
        )

        if passed:
            return self._result(payload, cfg, model, "approved", None, None, None, dq_score, sig, risk, ctx, fit, pre_gate_score, prob, gross_ev_r, cr, net_ev_r, diagnostics, plan)

        reasons = []
        if pre_gate_score < policy.min_pre_gate_score:
            reasons.append(f"pre_gate_score={pre_gate_score:.2f}<{policy.min_pre_gate_score:.2f}")
        if prob < policy.min_probability:
            reasons.append(f"probability={prob:.3f}<{policy.min_probability:.3f}")
        if net_ev_r < policy.min_expected_net_r:
            reasons.append(f"expected_net_r={net_ev_r:.3f}<{policy.min_expected_net_r:.3f}")

        return self._result(payload, cfg, model, "rejected", "expected_value", "; ".join(reasons), "INSUFFICIENT_POST_GATE_EDGE", dq_score, sig, risk, ctx, fit, pre_gate_score, prob, gross_ev_r, cr, net_ev_r, diagnostics, None)

    def _hard_rejection(self, payload: Dict[str, Any], cfg: InstrumentConfig, model: str, policy: ModelPolicy, dq_pass: bool, sig: float, risk: float, ctx: float, fit: float, plan: Dict[str, Any], costs: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
        direction = as_str(get_path(payload, "signal.direction"), "").lower()
        session = as_str(get_path(payload, "market.session_name"), "unknown").lower()
        vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()
        context_status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()

        if not dq_pass:
            return "data_quality", "Required fields missing or invalid.", "DATA_QUALITY_FAIL"
        if not policy.enabled:
            return "strategy_fit", f"Model disabled or unsupported: {model}.", "MODEL_DISABLED"
        if direction not in ("long", "short"):
            return "signal", "Direction must be long or short.", "INVALID_DIRECTION"
        if session in cfg.blocked_sessions:
            return "context", f"Blocked session: {session}.", "BLOCKED_SESSION"
        if context_status == "blocked":
            return "context", "Context explicitly blocked.", "CONTEXT_BLOCKED"
        if vol == "extreme":
            return "strategy_fit", "Extreme volatility blocked.", "EXTREME_VOLATILITY"
        if plan.get("sl") is None or plan.get("tp") is None or plan.get("units") is None:
            return "execution_planning", "Invalid trade plan.", "INVALID_TRADE_PLAN"
        if (as_float(plan.get("rr"), 0.0) or 0.0) < policy.min_rr:
            return "risk", "Reward-to-risk below model minimum.", "RR_BELOW_MODEL_MINIMUM"
        spread = as_float(costs.get("spread"))
        if spread is not None and spread > cfg.max_spread_price:
            return "execution_cost", "Observed spread exceeds limit.", "SPREAD_TOO_WIDE"
        if (as_float(costs.get("slippage"), 0.0) or 0.0) > cfg.max_slippage_price:
            return "execution_cost", "Expected slippage exceeds limit.", "SLIPPAGE_TOO_HIGH"
        if sig < 45:
            return "signal", "Signal quality below hard minimum.", "SIGNAL_SCORE_TOO_LOW"
        if risk < 45:
            return "risk", "Risk score below hard minimum.", "RISK_SCORE_TOO_LOW"
        if ctx < 35:
            return "context", "Context score below hard minimum.", "CONTEXT_SCORE_TOO_LOW"
        if fit < 40:
            return "strategy_fit", "Strategy-fit score below hard minimum.", "FIT_SCORE_TOO_LOW"
        if (as_float(costs.get("cost_r"), 1.0) or 1.0) >= 0.50:
            return "execution_cost", "Estimated costs consume at least 0.50R.", "COST_R_EXCESSIVE"
        return None

    def _result(self, payload: Dict[str, Any], cfg: InstrumentConfig, model: str, status: str, stage: Optional[str], reason: Optional[str], code: Optional[str], dq: float, sig: float, risk: float, ctx: float, fit: float, pre: float, prob: float, gross_r: float, cost_r_value: float, net_r: float, diagnostics: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = deepcopy(payload)
        manus = {
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "entry_model_normalized": model,
            "data_quality_score": round(dq, 2),
            "signal_quality_score": round(sig, 2),
            "risk_score": round(risk, 2),
            "context_score": round(ctx, 2),
            "strategy_fit_score": round(fit, 2),
            "pre_gate_score": round(pre, 2),
            "confidence_score": round(prob * 100.0, 2),
            "predicted_win_probability": round(prob, 4),
            "predicted_tp_before_sl_probability": round(prob, 4),
            "probability_is_calibrated": False,
            "expected_gross_r": round(gross_r, 4),
            "expected_cost_r": round(cost_r_value, 4),
            "expected_net_r": round(net_r, 4),
            "expected_value_score": round(net_r, 4),
            "approval_status": status,
            "approval_reason": "Approved by Manus v3.3 deterministic ruleset." if status == "approved" else None,
            "rejection_stage": stage,
            "rejection_reason": reason,
            "rejection_reason_code": code,
            "diagnostics": diagnostics,
        }

        if status == "approved" and plan:
            manus.update({
                "final_entry": plan["entry"],
                "final_stop_loss": plan["sl"],
                "final_take_profit": plan["tp"],
                "final_position_size": plan["units"],
                "final_position_size_lots": plan["lots"],
                "final_rr_ratio": plan["rr"],
                "final_trade_plan": {
                    "execution_model": "fixed_entry_fixed_exit",
                    "instrument": cfg.symbol,
                    "broker_symbol": cfg.broker_symbol,
                    "entry": plan["entry"],
                    "stop_loss": plan["sl"],
                    "take_profit": plan["tp"],
                    "position_size_units": plan["units"],
                    "position_size_lots": plan["lots"],
                    "rr_ratio": plan["rr"],
                    "risk_percent": plan["risk_percent"],
                    "risk_cash": plan["risk_cash"],
                    "planning_source": plan["source"],
                    "stop_basis": plan.get("stop_basis"),
                    "no_mid_trade_adjustment": True,
                },
            })
        else:
            manus.update({
                "final_entry": None,
                "final_stop_loss": None,
                "final_take_profit": None,
                "final_position_size": None,
                "final_position_size_lots": None,
                "final_rr_ratio": None,
                "final_trade_plan": None,
            })

        set_path(out, "manus", manus)
        return {
            "approval_status": status,
            "approval_reason": manus["approval_reason"],
            "rejection_stage": stage,
            "rejection_reason": reason,
            "rejection_reason_code": code,
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "instrument": cfg.symbol,
            "entry_model": model,
            "payload": out,
            "manus": manus,
        }

    def _reject(self, payload: Dict[str, Any], reason: str, stage: str, code: str) -> Dict[str, Any]:
        return {
            "approval_status": "rejected",
            "rejection_stage": stage,
            "rejection_reason": reason,
            "rejection_reason_code": code,
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "payload": payload,
            "manus": {
                "approval_status": "rejected",
                "rejection_stage": stage,
                "rejection_reason": reason,
                "rejection_reason_code": code,
                "probability_is_calibrated": False,
            },
        }


def _build_shadow_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Builds the flat feature dict shadow_scorer.py expects, from engine.py's
    nested payload. Shadow-mode only — never touches approval logic."""
    display_name = get_path(payload, "instrument.display_name")
    if not display_name:
        raw_symbol = get_path(payload, "instrument.symbol") or get_path(payload, "instrument.broker_symbol") or ""
        clean = str(raw_symbol).upper().replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
        display_name = f"{clean[:3]}/{clean[3:]}" if len(clean) == 6 else clean
    return {
        "indicators_rsi": get_path(payload, "indicators.rsi"),
        "indicators_atr": get_path(payload, "indicators.atr"),
        "indicators_atr_percent_of_price": get_path(payload, "indicators.atr_percent_of_price"),
        "indicators_ema_fast": get_path(payload, "indicators.ema_fast"),
        "indicators_ema_slow": get_path(payload, "indicators.ema_slow"),
        "indicators_macd_line": get_path(payload, "indicators.macd_line"),
        "indicators_macd_signal": get_path(payload, "indicators.macd_signal"),
        "indicators_macd_histogram": get_path(payload, "indicators.macd_histogram"),
        "indicators_bb_basis": get_path(payload, "indicators.bb_basis"),
        "indicators_bb_upper": get_path(payload, "indicators.bb_upper"),
        "indicators_bb_lower": get_path(payload, "indicators.bb_lower"),
        "indicators_bb_width": get_path(payload, "indicators.bb_width"),
        "structure_distance_to_recent_high": get_path(payload, "structure.distance_to_recent_high"),
        "structure_distance_to_recent_low": get_path(payload, "structure.distance_to_recent_low"),
        "structure_nearest_support": get_path(payload, "structure.nearest_support"),
        "structure_nearest_resistance": get_path(payload, "structure.nearest_resistance"),
        "instrument_norm": display_name,
        "market_session_name": get_path(payload, "market.session_name"),
        "market_session_phase": get_path(payload, "market.session_phase"),
        "market_market_state": get_path(payload, "market.market_state"),
        "signal_type": get_path(payload, "signal.type"),
        "signal_strength_label": get_path(payload, "signal.strength_label"),
        "signal_entry_model": get_path(payload, "signal.entry_model"),
        "structure_trend_bias": get_path(payload, "structure.trend_bias"),
        "structure_market_regime": get_path(payload, "structure.market_regime"),
        "structure_higher_timeframe_bias": get_path(payload, "structure.higher_timeframe_bias"),
        "context_context_status": get_path(payload, "context.context_status"),
    }


engine = TradingSignalEvaluationEngine()
app = FastAPI(title="St Ludaetuc Manus Engine", version=ENGINE_VERSION)


class MarketDataTick(BaseModel):
    # TradingView's own placeholders ({{ticker}}, {{interval}}, {{time}},
    # {{open}}/{{high}}/{{low}}/{{close}}/{{volume}}) populate these - see
    # the shared alert message template. `extra: allow` so the same endpoint
    # tolerates minor template tweaks without a deploy.
    ticker: Optional[str] = None
    symbol: Optional[str] = None
    interval: Optional[str] = None
    time: Optional[Any] = None
    open: Optional[Any] = None
    high: Optional[Any] = None
    low: Optional[Any] = None
    close: Optional[Any] = None
    volume: Optional[Any] = None
    model_config = {"extra": "allow"}


class TradeOutcome(BaseModel):
    trade_id: Optional[str] = None
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    closed_at: Optional[Any] = None
    outcome: Optional[str] = None  # "win" | "loss" | "breakeven" | "cancelled"
    realized_r: Optional[float] = None
    realized_pnl: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    model_config = {"extra": "allow"}


@app.on_event("startup")
def _on_startup() -> None:
    try:
        _ensure_events_schema()
    except Exception:
        logging.exception("Failed to ensure events schema on startup (non-fatal)")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "supported_instruments": sorted(INSTRUMENTS.keys()),
        "supported_entry_models": sorted(MODEL_POLICIES.keys()),
        "probability_is_calibrated": False,
        "events_db_configured": bool(DATABASE_URL),
    }


@app.post("/market-data")
def market_data(tick: MarketDataTick):
    """Direct TradingView -> Render ingestion for the Market Data Centre.
    Deliberately never touches Make - see the explicit 'I don't want the
    market data going through make' decision. Must respond well inside
    TradingView's 3s webhook timeout, so this does exactly one insert."""
    body = tick.model_dump()
    ticker_raw = body.get("ticker") or body.get("symbol")
    instrument = _normalize_tv_ticker(ticker_raw) or None
    timeframe = _normalize_tv_interval(body.get("interval")) or None
    event_time = _parse_event_time(body.get("time"))
    try:
        event_id = log_event(
            event_type="market_data",
            origin="tradingview",
            event_time=event_time,
            payload=body,
            instrument=instrument,
            timeframe=timeframe,
            status="recorded",
        )
    except Exception as exc:
        logging.exception("market-data insert failed")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})
    return {"status": "ok", "event_id": event_id, "instrument": instrument, "timeframe": timeframe}


@app.post("/trade-outcome")
def trade_outcome(outcome: TradeOutcome):
    """Actual closed-trade result, keyed by trade_id (the same
    meta.signal_id a 'prediction' event was logged under at evaluation
    time), so the two can be joined to check calibration. Not yet wired to
    the Closed Trade Data Sync Make scenario - that's a separate, deliberate
    follow-up step, since it means touching a live trading scenario."""
    body = outcome.model_dump()
    trade_id = body.get("trade_id")
    if not trade_id:
        return JSONResponse(status_code=400, content={"status": "error", "error": "trade_id is required"})
    event_time = _parse_event_time(body.get("closed_at"))
    try:
        event_id = log_event(
            event_type="trade_outcome",
            origin="oanda_via_make",
            event_time=event_time,
            payload=body,
            event_id=hashlib.sha256(f"trade_outcome:{trade_id}".encode("utf-8")).hexdigest(),
            trade_id=str(trade_id),
            instrument=body.get("instrument"),
            timeframe=body.get("timeframe"),
            status=body.get("outcome"),
        )
    except Exception as exc:
        logging.exception("trade-outcome insert failed")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})
    return {"status": "ok", "event_id": event_id}


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    result = engine.evaluate(req.model_dump())
    try:
        shadow_payload = req.payload if isinstance(req.payload, dict) else req.model_dump()
        shadow_features = _build_shadow_features(shadow_payload)
        shadow_score = shadow_score_0_100(shadow_features)
    except Exception:
        shadow_score = None
    result["shadow_score_v0"] = shadow_score
    if isinstance(result.get("manus"), dict):
        result["manus"]["shadow_score_v0"] = shadow_score

    # Best-effort prediction logging for calibration tracking. Must NEVER
    # block or break a trade evaluation - a Postgres hiccup should be
    # invisible to the trading path, just an unlogged data point.
    try:
        payload_in = req.payload if isinstance(req.payload, dict) else {}
        trade_id = get_path(payload_in, "meta.signal_id")
        if trade_id:
            manus = result.get("manus") if isinstance(result.get("manus"), dict) else {}
            event_time = _parse_event_time(get_path(payload_in, "market.bar_time_utc"))
            log_event(
                event_type="prediction",
                origin="manus_engine",
                event_time=event_time,
                payload={
                    "signal_id": trade_id,
                    "instrument": result.get("instrument"),
                    "entry_model": result.get("entry_model"),
                    "approval_status": result.get("approval_status"),
                    "rejection_stage": result.get("rejection_stage"),
                    "rejection_reason_code": result.get("rejection_reason_code"),
                    "predicted_win_probability": manus.get("predicted_win_probability"),
                    "predicted_tp_before_sl_probability": manus.get("predicted_tp_before_sl_probability"),
                    "expected_gross_r": manus.get("expected_gross_r"),
                    "expected_cost_r": manus.get("expected_cost_r"),
                    "expected_net_r": manus.get("expected_net_r"),
                    "final_trade_plan": manus.get("final_trade_plan"),
                    "shadow_score_v0": shadow_score,
                },
                event_id=hashlib.sha256(f"prediction:{trade_id}".encode("utf-8")).hexdigest(),
                trade_id=str(trade_id),
                instrument=result.get("instrument"),
                timeframe=get_path(payload_in, "market.timeframe"),
                strategy_version=result.get("ruleset_version"),
                model_version=result.get("engine_version"),
                status=result.get("approval_status"),
            )
    except Exception:
        logging.exception("prediction event logging failed (non-fatal, trade unaffected)")

    return result
