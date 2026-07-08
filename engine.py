from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import floor, isfinite
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel


ENGINE_VERSION = "3.1.0"
RULESET_VERSION = "manus_ruleset_multi_asset_2026_07_08_v3_1"


class EvaluateRequest(BaseModel):
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
    min_rr: float
    max_risk_percent: float
    max_spread_price: float
    max_slippage_price: float
    min_atr_pct: float
    max_atr_pct: float
    preferred_sessions: Tuple[str, ...]
    blocked_sessions: Tuple[str, ...]
    sl_atr_mult: float
    tp_rr: float


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "GBPUSD": InstrumentConfig("GBPUSD", "GBP/USD", "forex", "GBP_USD", "GBP", "USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 1.35, 0.50, 0.00025, 0.00012, 0.015, 0.180, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.35, 1.60),
    "EURUSD": InstrumentConfig("EURUSD", "EUR/USD", "forex", "EUR_USD", "EUR", "USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 1.25, 0.50, 0.00020, 0.00010, 0.012, 0.160, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.30, 1.55),
    "AUDUSD": InstrumentConfig("AUDUSD", "AUD/USD", "forex", "AUD_USD", "AUD", "USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 1.25, 0.50, 0.00025, 0.00012, 0.012, 0.180, ("asia", "london", "overlap_london_new_york"), ("overnight", "unknown"), 1.35, 1.55),
    "USDCAD": InstrumentConfig("USDCAD", "USD/CAD", "forex", "USD_CAD", "USD", "CAD", 0.0001, 0.00001, 100000, 1, 1, 200000, 1.25, 0.50, 0.00030, 0.00015, 0.012, 0.180, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.35, 1.55),
    "USDJPY": InstrumentConfig("USDJPY", "USD/JPY", "forex", "USD_JPY", "USD", "JPY", 0.01, 0.001, 100000, 1, 1, 200000, 1.25, 0.50, 0.030, 0.015, 0.012, 0.200, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.35, 1.55),
    "XAUUSD": InstrumentConfig("XAUUSD", "XAU/USD", "metals", "XAU_USD", "XAU", "USD", 0.1, 0.01, 1, 1, 1, 500, 1.40, 0.35, 0.60, 0.30, 0.020, 0.450, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.50, 1.70),
    "XAGUSD": InstrumentConfig("XAGUSD", "XAG/USD", "metals", "XAG_USD", "XAG", "USD", 0.01, 0.001, 1, 1, 1, 5000, 1.40, 0.35, 0.030, 0.015, 0.020, 0.500, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown"), 1.50, 1.70),
}


def get_path(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
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
    if x is None:
        return default
    return str(x)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_symbol(payload: Dict[str, Any]) -> Optional[str]:
    candidates = [
        get_path(payload, "instrument.symbol"),
        get_path(payload, "instrument.broker_symbol"),
        get_path(payload, "instrument.display_name"),
    ]
    for raw in candidates:
        if raw:
            s = str(raw).upper().replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
            if s in INSTRUMENTS:
                return s
    return None


def round_to_tick(price: float, tick: float) -> float:
    if not tick:
        return price
    return round(round(price / tick) * tick, 10)


def round_to_step(units: float, step: float) -> float:
    if step <= 0:
        return units
    return floor(units / step) * step


def extract_payload(req: EvaluateRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw = req.model_dump()
    payload = req.payload if isinstance(req.payload, dict) else None
    account = req.account if isinstance(req.account, dict) else {}

    if payload is None:
        if isinstance(raw.get("payload"), dict):
            payload = raw["payload"]
        else:
            payload = raw

    return payload or {}, account or {}


def score_signal_quality(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[float, Dict[str, Any]]:
    direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
    entry_model = as_str(get_path(payload, "signal.entry_model"), "").lower()
    rsi = as_float(get_path(payload, "indicators.rsi"))
    ema_fast = as_float(get_path(payload, "indicators.ema_fast"))
    ema_slow = as_float(get_path(payload, "indicators.ema_slow"))
    macd_hist = as_float(get_path(payload, "indicators.macd_histogram"))
    htf_bias = as_str(get_path(payload, "structure.higher_timeframe_bias"), "unknown").lower()
    trend_bias = as_str(get_path(payload, "structure.trend_bias"), "unknown").lower()
    vol_regime = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()

    score = 50.0
    notes = []

    if direction not in ("long", "short"):
        return 0.0, {"fail": "invalid_direction"}

    if direction == "long":
        if ema_fast is not None and ema_slow is not None and ema_fast >= ema_slow:
            score += 12
            notes.append("EMA long alignment")
        if htf_bias == "bullish":
            score += 16
            notes.append("HTF bullish")
        elif htf_bias == "mixed":
            score -= 6
        else:
            score -= 18
        if trend_bias == "bullish":
            score += 8
        if rsi is not None and 35 <= rsi <= 62:
            score += 8
        if macd_hist is not None and macd_hist >= 0:
            score += 6

    if direction == "short":
        if ema_fast is not None and ema_slow is not None and ema_fast <= ema_slow:
            score += 12
            notes.append("EMA short alignment")
        if htf_bias == "bearish":
            score += 16
            notes.append("HTF bearish")
        elif htf_bias == "mixed":
            score -= 6
        else:
            score -= 18
        if trend_bias == "bearish":
            score += 8
        if rsi is not None and 38 <= rsi <= 65:
            score += 8
        if macd_hist is not None and macd_hist <= 0:
            score += 6

    if "pullback" in entry_model:
        score += 5
    if "momentum" in entry_model:
        score += 2
    if vol_regime == "normal":
        score += 8
    elif vol_regime in ("low", "high"):
        score += 1
    elif vol_regime in ("very_low", "extreme", "unknown"):
        score -= 15

    return clamp(score, 0, 100), {"notes": notes, "entry_model": entry_model}


def score_risk(payload: Dict[str, Any], account: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[float, Dict[str, Any]]:
    rr = as_float(get_path(payload, "risk.rr_ratio"), 0.0) or 0.0
    risk_pct = as_float(get_path(payload, "risk.risk_percent"), cfg.max_risk_percent) or cfg.max_risk_percent
    spread = as_float(get_path(payload, "price.spread_price"))
    max_spread = as_float(get_path(payload, "risk.max_spread_allowed"), cfg.max_spread_price) or cfg.max_spread_price
    slippage = as_float(get_path(payload, "risk.max_slippage_allowed"), cfg.max_slippage_price) or cfg.max_slippage_price
    atr_pct = as_float(get_path(payload, "indicators.atr_percent_of_price"))
    portfolio_heat = as_float(account.get("portfolio_heat_percent"), 0.0) or 0.0
    open_trades = as_float(account.get("open_trade_count"), 0.0) or 0.0

    score = 70.0
    notes = []

    if rr >= cfg.min_rr:
        score += 12
    else:
        score -= 25
        notes.append("RR below instrument threshold")

    if risk_pct <= cfg.max_risk_percent:
        score += 8
    else:
        score -= 30
        notes.append("Risk percent above maximum")

    if spread is not None and spread > max_spread:
        score -= 30
        notes.append("Spread too wide")
    else:
        score += 5

    if slippage > cfg.max_slippage_price:
        score -= 10

    if atr_pct is not None:
        if cfg.min_atr_pct <= atr_pct <= cfg.max_atr_pct:
            score += 8
        else:
            score -= 15
            notes.append("ATR percent outside preferred band")

    if portfolio_heat > 3.0:
        score -= 25
        notes.append("Portfolio heat high")
    if open_trades >= 4:
        score -= 15
        notes.append("Open trade count high")

    return clamp(score, 0, 100), {"notes": notes, "rr": rr, "risk_percent": risk_pct}


def score_context(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[float, Dict[str, Any]]:
    session = as_str(get_path(payload, "market.session_name"), "unknown").lower()
    context_status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
    event_nearby = get_path(payload, "context.high_impact_event_nearby")

    score = 70.0
    notes = []

    if session in cfg.preferred_sessions:
        score += 15
    elif session in cfg.blocked_sessions:
        score -= 30
        notes.append("Blocked/weak session")
    else:
        score -= 5

    if context_status == "blocked":
        score -= 50
        notes.append("Context blocked")
    elif context_status == "warning":
        score -= 20
    elif context_status == "clear":
        score += 10
    elif context_status == "unchecked":
        score -= 5

    if event_nearby is True:
        score -= 35
        notes.append("High impact event nearby")

    return clamp(score, 0, 100), {"notes": notes, "session": session, "context_status": context_status}


def score_strategy_fit(payload: Dict[str, Any], cfg: InstrumentConfig) -> Tuple[float, Dict[str, Any]]:
    vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()
    market_regime = as_str(get_path(payload, "structure.market_regime"), "unknown").lower()
    direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
    htf = as_str(get_path(payload, "structure.higher_timeframe_bias"), "unknown").lower()

    score = 60.0

    if vol == "normal":
        score += 20
    elif vol in ("low", "high"):
        score += 5
    else:
        score -= 20

    if market_regime in ("trend_pullback", "range_to_reversal", "trend"):
        score += 10

    if direction == "long" and htf == "bullish":
        score += 10
    if direction == "short" and htf == "bearish":
        score += 10
    if htf == "mixed":
        score -= 5

    return clamp(score, 0, 100), {"volatility_regime": vol, "market_regime": market_regime}


def plan_trade(payload: Dict[str, Any], account: Dict[str, Any], cfg: InstrumentConfig) -> Dict[str, Any]:
    direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
    entry = as_float(get_path(payload, "risk.proposed_entry")) or as_float(get_path(payload, "price.close"), 0.0) or 0.0
    atr = as_float(get_path(payload, "indicators.atr"), 0.0) or 0.0

    proposed_sl = as_float(get_path(payload, "risk.proposed_stop_loss"))
    proposed_tp = as_float(get_path(payload, "risk.proposed_take_profit"))

    if proposed_sl is not None and proposed_tp is not None:
        sl = proposed_sl
        tp = proposed_tp
    else:
        stop_dist = atr * cfg.sl_atr_mult if atr > 0 else entry * 0.001
        if direction == "long":
            sl = entry - stop_dist
            tp = entry + stop_dist * cfg.tp_rr
        elif direction == "short":
            sl = entry + stop_dist
            tp = entry - stop_dist * cfg.tp_rr
        else:
            sl = None
            tp = None

    if sl is None or tp is None or entry <= 0:
        return {"entry": entry, "sl": None, "tp": None, "units": None, "lots": None, "rr": None}

    sl = round_to_tick(sl, cfg.tick_size)
    tp = round_to_tick(tp, cfg.tick_size)
    stop_dist = abs(entry - sl)
    target_dist = abs(tp - entry)
    rr = target_dist / stop_dist if stop_dist > 0 else None

    balance = as_float(account.get("balance"), None)
    margin_available = as_float(account.get("margin_available"), None)
    equity = balance or margin_available or 10000.0
    risk_pct = min(as_float(get_path(payload, "risk.risk_percent"), cfg.max_risk_percent) or cfg.max_risk_percent, cfg.max_risk_percent)
    risk_cash = equity * risk_pct / 100.0

    raw_units = risk_cash / stop_dist if stop_dist > 0 else 0.0
    units = round_to_step(raw_units, cfg.unit_step)
    units = clamp(units, cfg.min_units, cfg.max_units)

    if direction == "short":
        units = -abs(units)
    else:
        units = abs(units)

    lots = abs(units) / cfg.contract_size if cfg.contract_size else None

    return {
        "entry": round_to_tick(entry, cfg.tick_size),
        "sl": sl,
        "tp": tp,
        "units": units,
        "lots": lots,
        "rr": rr,
    }


class TradingSignalEvaluationEngine:
    def evaluate(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        req = EvaluateRequest(**request_body)
        payload, account = extract_payload(req)

        if not isinstance(payload, dict) or not payload:
            return self._reject({}, "Request body is not a valid payload object.", "schema")

        symbol_key = normalize_symbol(payload)
        if symbol_key is None or symbol_key not in INSTRUMENTS:
            return self._reject(
                payload,
                "Unsupported or missing instrument symbol.",
                "instrument",
                extra={"supported_instruments": sorted(INSTRUMENTS.keys())},
            )

        cfg = INSTRUMENTS[symbol_key]

        signal_score, signal_diag = score_signal_quality(payload, cfg)
        risk_score, risk_diag = score_risk(payload, account, cfg)
        context_score, context_diag = score_context(payload, cfg)
        fit_score, fit_diag = score_strategy_fit(payload, cfg)

        ev_score = (
            signal_score * 0.30
            + risk_score * 0.25
            + context_score * 0.20
            + fit_score * 0.25
        )

        plan = plan_trade(payload, account, cfg)

        diagnostics = {
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "instrument": cfg.symbol,
            "signal": signal_diag,
            "risk": risk_diag,
            "context": context_diag,
            "strategy_fit": fit_diag,
            "trade_plan_preview": plan,
        }

        hard_rejection = self._hard_rejection(payload, cfg, signal_score, risk_score, context_score, fit_score, plan)
        if hard_rejection:
            stage, reason = hard_rejection
            return self._with_manus(
                payload, cfg, "rejected", stage, reason,
                signal_score, risk_score, context_score, fit_score, ev_score, diagnostics, plan=None
            )

        approval_threshold = 72.0
        if ev_score >= approval_threshold:
            return self._with_manus(
                payload, cfg, "approved", None, None,
                signal_score, risk_score, context_score, fit_score, ev_score, diagnostics, plan=plan
            )

        return self._with_manus(
            payload, cfg, "rejected", "expected_value",
            f"Confidence {ev_score:.2f} below threshold {approval_threshold:.2f}.",
            signal_score, risk_score, context_score, fit_score, ev_score, diagnostics, plan=None
        )

    def _hard_rejection(
        self,
        payload: Dict[str, Any],
        cfg: InstrumentConfig,
        signal_score: float,
        risk_score: float,
        context_score: float,
        fit_score: float,
        plan: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        direction = as_str(get_path(payload, "signal.direction"), "neutral").lower()
        session = as_str(get_path(payload, "market.session_name"), "unknown").lower()
        vol = as_str(get_path(payload, "structure.volatility_regime"), "unknown").lower()
        context_status = as_str(get_path(payload, "context.context_status"), "unchecked").lower()
        rr = plan.get("rr")

        if direction not in ("long", "short"):
            return "signal", "Direction is not long or short."

        if session in cfg.blocked_sessions:
            return "context", f"Session blocked for {cfg.symbol}: {session}."

        if context_status == "blocked":
            return "context", "Context status is blocked."

        if vol in ("extreme", "unknown"):
            return "strategy_fit", f"Volatility regime not acceptable: {vol}."

        if rr is None or rr < cfg.min_rr:
            return "risk", f"Final RR below threshold for {cfg.symbol}."

        if plan.get("sl") is None or plan.get("tp") is None or plan.get("units") is None:
            return "execution_planning", "Unable to produce valid fixed SL/TP/size plan."

        if signal_score < 50:
            return "signal", "Signal quality score below hard minimum."

        if risk_score < 45:
            return "risk", "Risk score below hard minimum."

        if context_score < 40:
            return "context", "Context score below hard minimum."

        if fit_score < 40:
            return "strategy_fit", "Strategy fit score below hard minimum."

        return None

    def _with_manus(
        self,
        payload: Dict[str, Any],
        cfg: InstrumentConfig,
        status: str,
        stage: Optional[str],
        reason: Optional[str],
        signal_score: float,
        risk_score: float,
        context_score: float,
        fit_score: float,
        ev_score: float,
        diagnostics: Dict[str, Any],
        plan: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result_payload = deepcopy(payload)

        set_path(result_payload, "manus.signal_quality_score", round(signal_score, 2))
        set_path(result_payload, "manus.risk_score", round(risk_score, 2))
        set_path(result_payload, "manus.context_score", round(context_score, 2))
        set_path(result_payload, "manus.strategy_fit_score", round(fit_score, 2))
        set_path(result_payload, "manus.expected_value_score", round(ev_score, 2))
        set_path(result_payload, "manus.confidence_score", round(ev_score, 2))
        set_path(result_payload, "manus.approval_status", status)
        set_path(result_payload, "manus.approval_reason", "Approved by Manus multi-asset ruleset." if status == "approved" else None)
        set_path(result_payload, "manus.rejection_stage", stage)
        set_path(result_payload, "manus.rejection_reason", reason)
        set_path(result_payload, "manus.diagnostics", diagnostics)

        if plan and status == "approved":
            set_path(result_payload, "manus.final_entry", plan["entry"])
            set_path(result_payload, "manus.final_stop_loss", plan["sl"])
            set_path(result_payload, "manus.final_take_profit", plan["tp"])
            set_path(result_payload, "manus.final_position_size", plan["units"])
            set_path(result_payload, "manus.final_position_size_lots", plan["lots"])
            set_path(result_payload, "manus.final_rr_ratio", plan["rr"])
            set_path(result_payload, "manus.final_trade_plan", {
                "execution_model": "fixed_entry_fixed_exit",
                "instrument": cfg.symbol,
                "broker_symbol": cfg.broker_symbol,
                "entry": plan["entry"],
                "stop_loss": plan["sl"],
                "take_profit": plan["tp"],
                "position_size_units": plan["units"],
                "position_size_lots": plan["lots"],
                "rr_ratio": plan["rr"],
                "no_mid_trade_adjustment": True,
            })
        else:
            set_path(result_payload, "manus.final_entry", None)
            set_path(result_payload, "manus.final_stop_loss", None)
            set_path(result_payload, "manus.final_take_profit", None)
            set_path(result_payload, "manus.final_position_size", None)
            set_path(result_payload, "manus.final_position_size_lots", None)
            set_path(result_payload, "manus.final_rr_ratio", None)
            set_path(result_payload, "manus.final_trade_plan", None)

        return {
            "approval_status": status,
            "approval_reason": get_path(result_payload, "manus.approval_reason"),
            "rejection_stage": stage,
            "rejection_reason": reason,
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "instrument": cfg.symbol,
            "payload": result_payload,
            "manus": get_path(result_payload, "manus"),
        }

    def _reject(
        self,
        payload: Dict[str, Any],
        reason: str,
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = {
            "approval_status": "rejected",
            "rejection_stage": stage,
            "rejection_reason": reason,
            "engine_version": ENGINE_VERSION,
            "ruleset_version": RULESET_VERSION,
            "payload": payload,
            "manus": {
                "approval_status": "rejected",
                "rejection_stage": stage,
                "rejection_reason": reason,
                "signal_quality_score": 0,
                "risk_score": 0,
                "context_score": 0,
                "strategy_fit_score": 0,
                "expected_value_score": 0,
                "confidence_score": 0,
            },
        }
        if extra:
            result.update(extra)
        return result


engine = TradingSignalEvaluationEngine()
app = FastAPI(title="St Ludaetuc Manus Engine", version=ENGINE_VERSION)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "ruleset_version": RULESET_VERSION,
        "supported_instruments": sorted(INSTRUMENTS.keys()),
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    return engine.evaluate(req.model_dump())
