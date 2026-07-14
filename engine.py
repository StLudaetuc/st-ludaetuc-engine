ENGINE_VERSION = "3.2.0"
RULESET_VERSION = "manus_ruleset_2026_07_v3_2"
CALIBRATION_VERSION = "heuristic_uncalibrated_v1"


class EvaluateRequest(BaseModel):
    payload: Optional[Dict[str, Any]] = None
    account: Optional[Dict[str, Any]] = None
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


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "GBPUSD": InstrumentConfig("GBPUSD", "GBP_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00025, 0.00012, 1.50, 1.60, ("london", "overlap_london_new_york", "new_york"), ("overnight", "unknown")),
    "EURUSD": InstrumentConfig("EURUSD", "EUR_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00020, 0.00010, 1.40, 1.55, ("london", "overlap_london_new_york", "new_york"), ("overnight", "unknown")),
    "AUDUSD": InstrumentConfig("AUDUSD", "AUD_USD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00025, 0.00012, 1.45, 1.55, ("asia", "london", "overlap_london_new_york"), ("overnight", "unknown")),
    "USDCAD": InstrumentConfig("USDCAD", "USD_CAD", 0.0001, 0.00001, 100000, 1, 1, 200000, 0.50, 0.00030, 0.00015, 1.45, 1.55, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "USDJPY": InstrumentConfig("USDJPY", "USD_JPY", 0.01, 0.001, 100000, 1, 1, 200000, 0.50, 0.030, 0.015, 1.45, 1.55, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "XAUUSD": InstrumentConfig("XAUUSD", "XAU_USD", 0.1, 0.01, 1, 1, 1, 500, 0.35, 0.60, 0.30, 1.60, 1.70, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
    "XAGUSD": InstrumentConfig("XAGUSD", "XAG_USD", 0.01, 0.001, 1, 1, 1, 5000, 0.35, 0.030, 0.015, 1.60, 1.70, ("london", "overlap_london_new_york", "new_york"), ("asia", "overnight", "unknown")),
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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10) if tick > 0 else price


def round_to_step(units: float, step: float) -> float:
    return floor(units / step) * step if step > 0 else units


def normalize_symbol(payload: Dict[str, Any]) -> Optional[str]:
    for raw in (
        get_path(payload, "instrument.symbol"),
        get_path(payload, "instrument.broker_symbol"),
        get_path(payload, "instrument.display_name"),
    ):
        if raw:
            key = str(raw).upper().replace("/", "").replace("_", "").replace("-", "").replace(" ", "")
            if key in INSTRUMENTS:
                return key
    return None


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
    event = get_path(payload, "context.high_impact_event_nearby")

    score = 50.0
    score += 15 if session in cfg.preferred_sessions else (-25 if session in cfg.blocked_sessions else -5)
    score += 20 if status == "clear" else (-15 if status in ("warning", "unchecked", "unknown", "") else -50)
    if event is True:
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
    sl = as_float(get_path(payload, "risk.proposed_stop_loss"))
    tp = as_float(get_path(payload, "risk.proposed_take_profit"))
    source = "payload"

    if sl is None or tp is None:
        dist = atr * cfg.sl_atr_mult if atr > 0 else entry * 0.001
        rr = max(cfg.default_rr, policy.min_rr)
        if direction == "long":
            sl, tp = entry - dist, entry + dist * rr
        elif direction == "short":
            sl, tp = entry + dist, entry - dist * rr
        else:
            sl, tp = None, None
        source = "manus_atr_fallback"

    if sl is None or tp is None or entry <= 0:
        return {"entry": entry, "sl": None, "tp": None, "rr": None, "units": None, "lots": None, "source": source}

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

        symbol = normalize_symbol(payload)
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
            "approval_reason": "Approved by Manus v3.2 deterministic ruleset." if status == "approved" else None,
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


engine = TradingSignalEvaluationEngine()
app = FastAPI(title="St Ludaetuc Manus Engine", version=ENGINE_VERSION)


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
    }


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    return engine.evaluate(req.model_dump())
