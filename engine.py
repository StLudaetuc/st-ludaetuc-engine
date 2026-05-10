import copy
import math
from typing import Any, Dict, List, Optional, Tuple


class TradingSignalEvaluationEngine:
    """
    St Ludaetuc Manus Evaluation Engine v2.0

    Purpose:
    - Preserve canonical payload structure.
    - Use Manus as the final approval authority.
    - Reject statistically weak GBP/USD 2m contexts observed in current data.
    - Set fixed-at-entry SL/TP only; no mid-trade adjustment required.
    - Return broker-usable final_entry, final_stop_loss, final_take_profit, final_position_size.

    Expected input:
    {
      "payload": { ...canonical St Ludaetuc payload... },
      "account": {
        "balance": 10000,
        "margin_available": 9000,
        "margin_rate": 0.0333,       # optional
        "open_trade_count": 0,       # optional
        "portfolio_heat": 0.0        # optional, percent
      }
    }
    """

    ENGINE_VERSION = "2.0.0"
    PROMPT_VERSION = "manus_ruleset_gbpusd_2m_2026_04_15_plus_v2"

    # GBP/USD conventions
    DEFAULT_PIP_SIZE = 0.0001
    DEFAULT_TICK_SIZE = 0.00001
    DEFAULT_UNIT_STEP = 1.0
    DEFAULT_MIN_UNITS = 1.0
    DEFAULT_MAX_UNITS = 200000.0

    # Fixed-at-entry trade design
    MIN_RR = 1.35
    DEFAULT_RISK_PERCENT_CAP = 0.50
    MAX_PORTFOLIO_HEAT_AFTER = 2.00
    MAX_OPEN_TRADES = 1

    # Hard filters derived from current forensic review.
    BLOCK_SESSIONS = {"unknown", "overnight"}
    PREFERRED_SESSIONS = {"london", "overlap_london_new_york", "new_york"}
    BLOCK_VOL_REGIMES = {"very_low", "extreme", "unknown"}
    ALLOW_VOL_REGIMES = {"low", "normal", "high"}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Safe utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
        current: Any = d
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _set(d: Dict[str, Any], path: str, value: Any) -> None:
        current: Any = d
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
                return default
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return default
            return f
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n"}:
                return False
        return default

    @staticmethod
    def _clean_str(value: Any, default: str = "unknown") -> str:
        if value is None:
            return default
        text = str(value).strip().lower()
        return text if text else default

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _round_to_tick(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 5)

    @staticmethod
    def _round_units(units: float, unit_step: float, min_units: float, max_units: float) -> float:
        if units <= 0:
            return 0.0
        step = unit_step if unit_step > 0 else 1.0
        rounded = math.floor(units / step) * step
        rounded = max(min_units, rounded)
        rounded = min(max_units, rounded)
        return float(round(rounded))

    @staticmethod
    def _direction_sign(direction: str) -> int:
        return 1 if direction == "long" else -1 if direction == "short" else 0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(self, payload: Dict[str, Any], account: Dict[str, Any]) -> Dict[str, Any]:
        pip_size = self._to_float(self._get(payload, "instrument.pip_size"), self.DEFAULT_PIP_SIZE)
        tick_size = self._to_float(self._get(payload, "instrument.tick_size"), self.DEFAULT_TICK_SIZE)
        unit_step = self._to_float(self._get(payload, "instrument.unit_step"), self.DEFAULT_UNIT_STEP)
        min_units = self._to_float(self._get(payload, "instrument.min_units"), self.DEFAULT_MIN_UNITS)
        max_units = self._to_float(self._get(payload, "instrument.max_units"), self.DEFAULT_MAX_UNITS)

        entry = self._to_float(self._get(payload, "risk.proposed_entry"), 0.0)
        close = self._to_float(self._get(payload, "price.close"), entry)
        if entry <= 0 and close > 0:
            entry = close

        atr = self._to_float(self._get(payload, "indicators.atr"), 0.0)
        rsi = self._to_float(self._get(payload, "indicators.rsi"), 50.0)
        ema_fast = self._to_float(self._get(payload, "indicators.ema_fast"), 0.0)
        ema_slow = self._to_float(self._get(payload, "indicators.ema_slow"), 0.0)
        macd_hist = self._to_float(self._get(payload, "indicators.macd_histogram"), 0.0)
        bb_width = self._to_float(self._get(payload, "indicators.bb_width"), 0.0)
        atr_ratio = self._to_float(self._get(payload, "extensions.atr_ratio"), 1.0)
        bb_width_ratio = self._to_float(self._get(payload, "extensions.bb_width_ratio"), 1.0)

        stop_distance_price = self._to_float(self._get(payload, "risk.stop_distance_price"), 0.0)
        proposed_sl = self._to_float(self._get(payload, "risk.proposed_stop_loss"), 0.0)
        proposed_tp = self._to_float(self._get(payload, "risk.proposed_take_profit"), 0.0)
        proposed_rr = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)

        if stop_distance_price <= 0 and proposed_sl > 0 and entry > 0:
            stop_distance_price = abs(entry - proposed_sl)

        return {
            "signal_id": self._get(payload, "meta.signal_id"),
            "direction": self._clean_str(self._get(payload, "signal.direction"), "neutral"),
            "signal_type": self._clean_str(self._get(payload, "signal.signal_type"), "unknown"),
            "trigger_reason": str(self._get(payload, "signal.trigger_reason", "") or ""),
            "confidence_raw": self._to_float(self._get(payload, "signal.confidence_raw"), 0.0),

            "session": self._clean_str(self._get(payload, "market.session_name"), "unknown"),
            "session_phase": self._clean_str(self._get(payload, "market.session_phase"), "unknown"),
            "context_status": self._clean_str(self._get(payload, "context.context_status"), "unchecked"),
            "news_nearby": self._to_bool(self._get(payload, "context.high_impact_event_nearby"), False),
            "minutes_to_news": self._to_float(self._get(payload, "context.minutes_to_next_high_impact_event"), 9999.0),

            "trend_bias": self._clean_str(self._get(payload, "structure.trend_bias"), "unknown"),
            "htf_bias": self._clean_str(self._get(payload, "structure.higher_timeframe_bias"), "unknown"),
            "market_regime": self._clean_str(self._get(payload, "structure.market_regime"), "unknown"),
            "volatility_regime": self._clean_str(self._get(payload, "structure.volatility_regime"), "unknown"),
            "distance_to_high": self._to_float(self._get(payload, "structure.distance_to_recent_high"), 0.0),
            "distance_to_low": self._to_float(self._get(payload, "structure.distance_to_recent_low"), 0.0),
            "nearest_support": self._to_float(self._get(payload, "structure.nearest_support"), 0.0),
            "nearest_resistance": self._to_float(self._get(payload, "structure.nearest_resistance"), 0.0),

            "entry": entry,
            "close": close,
            "proposed_sl": proposed_sl,
            "proposed_tp": proposed_tp,
            "proposed_rr": proposed_rr,
            "stop_distance_price": stop_distance_price,
            "risk_percent": self._to_float(self._get(payload, "risk.risk_percent"), self.DEFAULT_RISK_PERCENT_CAP),

            "spread_price": self._to_float(self._get(payload, "price.spread_price"), 0.0),
            "max_spread_allowed": self._to_float(self._get(payload, "risk.max_spread_allowed"), 2.0 * pip_size),

            "atr": atr,
            "rsi": rsi,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd_hist": macd_hist,
            "bb_width": bb_width,
            "atr_ratio": atr_ratio,
            "bb_width_ratio": bb_width_ratio,

            "pip_size": pip_size,
            "tick_size": tick_size,
            "unit_step": unit_step,
            "min_units": min_units,
            "max_units": max_units,

            "account_balance": self._to_float(account.get("balance"), 0.0),
            "margin_available": self._to_float(account.get("margin_available"), 0.0),
            "margin_rate": self._to_float(account.get("margin_rate"), 0.0333),
            "open_trade_count": self._to_int(account.get("open_trade_count"), 0),
            "portfolio_heat": self._to_float(account.get("portfolio_heat"), 0.0),
        }

    # ------------------------------------------------------------------
    # Decision rules
    # ------------------------------------------------------------------
    def _hard_rejects(self, f: Dict[str, Any]) -> List[Tuple[str, str]]:
        rejects: List[Tuple[str, str]] = []
        direction = f["direction"]

        if not f["signal_id"]:
            rejects.append(("schema", "Missing meta.signal_id"))
        if f["signal_type"] not in {"entry", "signal"}:
            rejects.append(("signal", f"Signal type is not an entry signal: {f['signal_type']}"))
        if direction not in {"long", "short"}:
            rejects.append(("signal", f"Invalid direction: {direction}"))
        if f["entry"] <= 0:
            rejects.append(("execution", "Missing/invalid proposed entry"))
        if f["account_balance"] <= 0:
            rejects.append(("account", "Invalid account balance"))

        # Empirical filters from current GBP/USD 2m review.
        if f["session"] in self.BLOCK_SESSIONS:
            rejects.append(("context", f"Blocked session: {f['session']}"))
        if f["session"] not in self.PREFERRED_SESSIONS:
            rejects.append(("context", f"Non-preferred session: {f['session']}"))
        if f["volatility_regime"] in self.BLOCK_VOL_REGIMES:
            rejects.append(("strategy_fit", f"Blocked volatility regime: {f['volatility_regime']}"))
        if f["context_status"] == "blocked":
            rejects.append(("context", "Context status is blocked"))
        if f["news_nearby"] and f["minutes_to_news"] <= 30:
            rejects.append(("context", "High-impact news within 30 minutes"))

        # Direction / HTF alignment.
        if direction == "short" and f["htf_bias"] == "bullish":
            # Current data showed this as a high-risk loser class.
            rejects.append(("strategy_fit", "Short blocked against bullish higher-timeframe bias"))
        if direction == "long" and f["htf_bias"] in {"bearish", "mixed", "unknown"}:
            rejects.append(("strategy_fit", f"Long requires bullish HTF bias, got {f['htf_bias']}"))
        if direction == "short" and f["htf_bias"] not in {"bearish"}:
            rejects.append(("strategy_fit", f"Short requires bearish HTF bias, got {f['htf_bias']}"))

        # Spread only if real spread was supplied.
        if f["spread_price"] > 0 and f["max_spread_allowed"] > 0 and f["spread_price"] > f["max_spread_allowed"]:
            rejects.append(("risk", "Spread above allowed maximum"))

        if f["open_trade_count"] >= self.MAX_OPEN_TRADES:
            rejects.append(("risk", "Maximum concurrent open trades reached"))

        return rejects

    def _score_signal_quality(self, f: Dict[str, Any]) -> int:
        score = 0
        direction = f["direction"]
        trigger = f["trigger_reason"].lower()

        if f["confidence_raw"] >= 90:
            score += 25
        elif f["confidence_raw"] >= 80:
            score += 20
        elif f["confidence_raw"] >= 70:
            score += 14
        elif f["confidence_raw"] >= 60:
            score += 8

        if direction == "long":
            if f["rsi"] >= 50 and f["rsi"] <= 68:
                score += 15
            if f["ema_fast"] >= f["ema_slow"] and f["ema_slow"] > 0:
                score += 15
            if f["macd_hist"] >= 0 or "macd improvement" in trigger:
                score += 12
            if f["distance_to_high"] >= max(f["atr"] * 1.0, 4 * f["pip_size"]):
                score += 13

        elif direction == "short":
            if f["rsi"] <= 50 and f["rsi"] >= 32:
                score += 15
            if f["ema_fast"] <= f["ema_slow"] and f["ema_slow"] > 0:
                score += 15
            if f["macd_hist"] <= 0 or "macd" in trigger:
                score += 12
            if f["distance_to_low"] >= max(f["atr"] * 1.0, 4 * f["pip_size"]):
                score += 13

        if "rsi" in trigger:
            score += 8
        if "volatility pass" in trigger or f["volatility_regime"] in {"normal", "high"}:
            score += 7
        if f["market_regime"] in {"trend", "breakout", "range_to_reversal", "reversal"}:
            score += 5

        return int(self._clamp(score))

    def _score_risk(self, f: Dict[str, Any], final_sl: float, final_tp: float) -> int:
        entry = f["entry"]
        direction = f["direction"]
        pip = f["pip_size"]

        stop_dist = abs(entry - final_sl)
        target_dist = abs(final_tp - entry)
        rr = target_dist / stop_dist if stop_dist > 0 else 0.0
        stop_pips = stop_dist / pip if pip > 0 else 0.0

        score = 100
        if rr < self.MIN_RR:
            score -= 35
        elif rr < 1.6:
            score -= 12

        if stop_pips < 3.0:
            score -= 25
        elif stop_pips > 18.0:
            score -= 20

        if f["spread_price"] > 0 and stop_dist > 0:
            spread_to_stop = f["spread_price"] / stop_dist
            if spread_to_stop > 0.18:
                score -= 20
            elif spread_to_stop > 0.10:
                score -= 10

        if f["margin_available"] <= 0:
            score -= 20

        # Penalize structurally illogical stops.
        if direction == "long" and final_sl >= entry:
            score -= 60
        if direction == "short" and final_sl <= entry:
            score -= 60

        return int(self._clamp(score))

    def _score_context(self, f: Dict[str, Any]) -> int:
        score = 50
        if f["session"] == "overlap_london_new_york":
            score += 30
        elif f["session"] == "london":
            score += 25
        elif f["session"] == "new_york":
            score += 12
        elif f["session"] == "asia":
            score -= 10
        elif f["session"] in self.BLOCK_SESSIONS:
            score -= 40

        if f["context_status"] == "clear":
            score += 10
        elif f["context_status"] == "warning":
            score -= 10
        elif f["context_status"] == "blocked":
            score = 0

        if f["news_nearby"] and f["minutes_to_news"] <= 60:
            score -= 25

        return int(self._clamp(score))

    def _score_strategy_fit(self, f: Dict[str, Any]) -> int:
        score = 0
        direction = f["direction"]

        # HTF alignment is the strongest discovered discriminator.
        if direction == "long" and f["htf_bias"] == "bullish":
            score += 35
        if direction == "short" and f["htf_bias"] == "bearish":
            score += 40

        if f["volatility_regime"] == "normal":
            score += 25
        elif f["volatility_regime"] == "high":
            score += 18
        elif f["volatility_regime"] == "low":
            score += 8

        if f["session"] in {"london", "overlap_london_new_york"}:
            score += 20
        elif f["session"] == "new_york" and direction == "short":
            score += 12

        if f["market_regime"] in {"trend", "breakout"} and (
            (direction == "long" and f["trend_bias"] == "bullish") or
            (direction == "short" and f["trend_bias"] == "bearish")
        ):
            score += 15
        elif f["market_regime"] in {"range_to_reversal", "reversal"}:
            score += 10

        # ATR/BB expansion ratios help avoid dead tape.
        if 0.90 <= f["atr_ratio"] <= 1.60:
            score += 8
        if 0.85 <= f["bb_width_ratio"] <= 1.80:
            score += 7

        return int(self._clamp(score))

    # ------------------------------------------------------------------
    # Fixed SL/TP selection
    # ------------------------------------------------------------------
    def _select_fixed_trade_plan(self, f: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates one immutable SL/TP plan at entry.

        Design:
        - Use ATR as the primary stop model.
        - Enforce a min/max stop band suitable for GBP/USD 2m.
        - Use structure only when it makes the stop safer, not wider without limit.
        - Use lower RR for high-probability aligned trades; higher RR only where room exists.
        """
        entry = f["entry"]
        direction = f["direction"]
        pip = f["pip_size"]
        tick = f["tick_size"]
        atr = f["atr"] if f["atr"] > 0 else max(f["stop_distance_price"], 5 * pip)

        # Stop distance rules in price.
        min_stop = 4.0 * pip
        max_stop = 14.0 * pip

        if f["volatility_regime"] == "low":
            atr_mult = 1.05
        elif f["volatility_regime"] == "normal":
            atr_mult = 1.15
        elif f["volatility_regime"] == "high":
            atr_mult = 1.30
        else:
            atr_mult = 1.20

        raw_stop_dist = atr * atr_mult
        stop_dist = max(min_stop, min(max_stop, raw_stop_dist))

        # Let original Pine stop influence the plan only if sensible.
        proposed_dist = f["stop_distance_price"]
        if proposed_dist > 0:
            proposed_dist = max(min_stop, min(max_stop, proposed_dist))
            stop_dist = (0.65 * stop_dist) + (0.35 * proposed_dist)

        # RR selection. Keep it realistic for 2m GBP/USD.
        if f["session"] == "overlap_london_new_york":
            rr = 1.65
        elif f["session"] == "london":
            rr = 1.55
        else:
            rr = 1.45

        if direction == "short" and f["htf_bias"] == "bearish":
            rr += 0.10
        if f["volatility_regime"] == "high":
            rr += 0.10
        if f["market_regime"] in {"trend", "breakout"}:
            rr += 0.10

        rr = max(self.MIN_RR, min(1.90, rr))

        # Respect room to recent swing; if insufficient, reject later via notes/scores.
        room = f["distance_to_high"] if direction == "long" else f["distance_to_low"]
        target_dist = stop_dist * rr
        room_ok = room <= 0 or room >= target_dist * 0.80

        if direction == "long":
            sl = entry - stop_dist
            tp = entry + target_dist
            if f["nearest_support"] > 0 and f["nearest_support"] < entry:
                # Place SL just below support only if it remains inside max stop.
                support_sl = f["nearest_support"] - 1.0 * pip
                support_dist = entry - support_sl
                if min_stop <= support_dist <= max_stop:
                    sl = support_sl
                    stop_dist = support_dist
                    target_dist = stop_dist * rr
                    tp = entry + target_dist
        else:
            sl = entry + stop_dist
            tp = entry - target_dist
            if f["nearest_resistance"] > 0 and f["nearest_resistance"] > entry:
                resistance_sl = f["nearest_resistance"] + 1.0 * pip
                resistance_dist = resistance_sl - entry
                if min_stop <= resistance_dist <= max_stop:
                    sl = resistance_sl
                    stop_dist = resistance_dist
                    target_dist = stop_dist * rr
                    tp = entry - target_dist

        sl = self._round_to_tick(sl, tick)
        tp = self._round_to_tick(tp, tick)
        stop_dist = abs(entry - sl)
        target_dist = abs(tp - entry)
        rr_actual = target_dist / stop_dist if stop_dist > 0 else 0.0

        return {
            "entry": self._round_to_tick(entry, tick),
            "stop_loss": sl,
            "take_profit": tp,
            "stop_distance_price": stop_dist,
            "target_distance_price": target_dist,
            "stop_distance_pips": stop_dist / pip if pip > 0 else None,
            "target_distance_pips": target_dist / pip if pip > 0 else None,
            "rr": rr_actual,
            "room_ok": room_ok,
            "model": "atr_structure_fixed_entry_v2",
            "notes": [
                f"atr_mult={atr_mult}",
                f"target_rr={round(rr, 2)}",
                f"room_ok={room_ok}",
                "fixed_sl_tp_no_mid_trade_adjustment",
            ],
        }

    def _position_size(self, f: Dict[str, Any], stop_distance_price: float) -> float:
        if f["account_balance"] <= 0 or stop_distance_price <= 0:
            return 0.0

        # Cap risk percent. Pine sends 0.50 for 0.50%.
        risk_percent = min(max(f["risk_percent"], 0.01), self.DEFAULT_RISK_PERCENT_CAP)
        risk_amount = f["account_balance"] * (risk_percent / 100.0)
        raw_units = risk_amount / stop_distance_price

        # Basic margin cap. For GBP/USD units, notional approx = units * entry.
        if f["margin_available"] > 0 and f["margin_rate"] > 0 and f["entry"] > 0:
            max_units_by_margin = (f["margin_available"] * 0.80) / (f["entry"] * f["margin_rate"])
            raw_units = min(raw_units, max_units_by_margin)

        units = self._round_units(raw_units, f["unit_step"], f["min_units"], f["max_units"])

        # Portfolio heat cap.
        estimated_heat_after = f["portfolio_heat"] + min(f["risk_percent"], self.DEFAULT_RISK_PERCENT_CAP)
        if estimated_heat_after > self.MAX_PORTFOLIO_HEAT_AFTER:
            return 0.0

        return units

    # ------------------------------------------------------------------
    # Public evaluator
    # ------------------------------------------------------------------
    def evaluate(self, full_input: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(full_input.get("payload", {}) or {})
        account = full_input.get("account", {}) or {}
        f = self._extract_features(payload, account)

        plan = self._select_fixed_trade_plan(f)

        signal_score = self._score_signal_quality(f)
        risk_score = self._score_risk(f, plan["stop_loss"], plan["take_profit"])
        context_score = self._score_context(f)
        strategy_fit_score = self._score_strategy_fit(f)
        execution_quality_score = 75

        if f["spread_price"] > 0 and plan["stop_distance_price"] > 0:
            spread_ratio = f["spread_price"] / plan["stop_distance_price"]
            if spread_ratio > 0.15:
                execution_quality_score -= 25
            elif spread_ratio > 0.10:
                execution_quality_score -= 10

        if not plan["room_ok"]:
            strategy_fit_score = max(0, strategy_fit_score - 20)

        expected_value_score = round(
            0.26 * signal_score
            + 0.24 * risk_score
            + 0.20 * context_score
            + 0.22 * strategy_fit_score
            + 0.08 * execution_quality_score
        )

        confidence_score = round(
            0.30 * signal_score
            + 0.25 * strategy_fit_score
            + 0.20 * risk_score
            + 0.15 * context_score
            + 0.10 * execution_quality_score
        )

        hard_rejects = self._hard_rejects(f)

        rejection_stage = None
        rejection_reason = None
        approval_status = "rejected"
        approval_reason = "Signal does not meet Manus v2 thresholds."

        if hard_rejects:
            rejection_stage, rejection_reason = hard_rejects[0]
            approval_reason = rejection_reason
        elif plan["rr"] < self.MIN_RR:
            rejection_stage = "risk"
            rejection_reason = f"Final RR too low: {round(plan['rr'], 2)}"
            approval_reason = rejection_reason
        elif not plan["room_ok"]:
            rejection_stage = "strategy_fit"
            rejection_reason = "Insufficient structural room to target."
            approval_reason = rejection_reason
        elif signal_score < 62:
            rejection_stage = "signal_quality"
            rejection_reason = f"Signal quality below threshold: {signal_score}"
            approval_reason = rejection_reason
        elif risk_score < 68:
            rejection_stage = "risk"
            rejection_reason = f"Risk score below threshold: {risk_score}"
            approval_reason = rejection_reason
        elif context_score < 60:
            rejection_stage = "context"
            rejection_reason = f"Context score below threshold: {context_score}"
            approval_reason = rejection_reason
        elif strategy_fit_score < 65:
            rejection_stage = "strategy_fit"
            rejection_reason = f"Strategy fit below threshold: {strategy_fit_score}"
            approval_reason = rejection_reason
        elif expected_value_score < 68 or confidence_score < 68:
            rejection_stage = "expected_value"
            rejection_reason = f"EV/confidence below threshold: EV={expected_value_score}, confidence={confidence_score}"
            approval_reason = rejection_reason
        else:
            approval_status = "approved"
            approval_reason = "Approved by Manus v2 fixed-entry SL/TP model."

        final_position_size: Optional[float] = None
        if approval_status == "approved":
            units_abs = self._position_size(f, plan["stop_distance_price"])
            if units_abs <= 0:
                approval_status = "rejected"
                rejection_stage = "risk"
                rejection_reason = "Position size resolved to zero after broker/margin/risk constraints."
                approval_reason = rejection_reason
                final_position_size = None
            else:
                sign = self._direction_sign(f["direction"])
                final_position_size = units_abs * sign

        trade_plan = {
            "model": plan["model"],
            "fixed_at_entry": True,
            "no_mid_trade_sl_tp_adjustment": True,
            "entry": plan["entry"],
            "stop_loss": plan["stop_loss"],
            "take_profit": plan["take_profit"],
            "stop_distance_pips": plan["stop_distance_pips"],
            "target_distance_pips": plan["target_distance_pips"],
            "rr": plan["rr"],
            "session": f["session"],
            "htf_bias": f["htf_bias"],
            "volatility_regime": f["volatility_regime"],
            "notes": plan["notes"],
        }

        manus_block: Dict[str, Any] = {
            "model_name": "St Ludaetuc Manus Evaluation Engine",
            "model_version": self.ENGINE_VERSION,
            "prompt_version": self.PROMPT_VERSION,

            "signal_quality_score": signal_score,
            "risk_score": risk_score,
            "context_score": context_score,
            "strategy_fit_score": strategy_fit_score,
            "expected_value_score": expected_value_score,
            "execution_quality_score": int(self._clamp(execution_quality_score)),
            "confidence_score": confidence_score,

            "approval_status": approval_status,
            "approval_reason": approval_reason,
            "rejection_reason": rejection_reason,
            "rejection_stage": rejection_stage,
            "recommendation": "execute" if approval_status == "approved" else "do_not_execute",
            "notes": "; ".join(plan["notes"]),

            "asset_specific_logic_used": "gbpusd_2m_post_2026_04_15_forensic_rules_v2",
            "regime_classification": {
                "session": f["session"],
                "htf_bias": f["htf_bias"],
                "volatility_regime": f["volatility_regime"],
                "market_regime": f["market_regime"],
            },

            "final_entry": plan["entry"] if approval_status == "approved" else None,
            "final_stop_loss": plan["stop_loss"] if approval_status == "approved" else None,
            "final_take_profit": plan["take_profit"] if approval_status == "approved" else None,
            "final_position_size": final_position_size,
            "final_position_size_lots": abs(final_position_size) / 100000.0 if final_position_size else None,
            "final_rr_ratio": round(plan["rr"], 3) if approval_status == "approved" else None,
            "final_holding_period_est": "6_to_30_minutes",
            "final_trade_plan": trade_plan if approval_status == "approved" else None,

            # Helpful for Sheets/debugging.
            "diagnostics": {
                "hard_rejects": [{"stage": s, "reason": r} for s, r in hard_rejects],
                "computed_plan_even_if_rejected": trade_plan,
                "risk_inputs": {
                    "risk_percent_used_cap": min(max(f["risk_percent"], 0.01), self.DEFAULT_RISK_PERCENT_CAP),
                    "account_balance": f["account_balance"],
                    "margin_available": f["margin_available"],
                    "spread_price": f["spread_price"],
                    "max_spread_allowed": f["max_spread_allowed"],
                },
            },
        }

        payload["manus"] = manus_block

        # Also enrich execution intent without pretending execution happened.
        if "execution" not in payload or not isinstance(payload["execution"], dict):
            payload["execution"] = {}
        payload["execution"]["execution_status"] = payload["execution"].get("execution_status", "not_sent")
        payload["execution"]["execution_venue"] = payload["execution"].get("execution_venue", "oanda")
        payload["execution"]["order_type"] = payload["execution"].get("order_type", "market")
        payload["execution"]["time_in_force"] = payload["execution"].get("time_in_force", "fok")
        payload["execution"]["order_intent"] = "open_position" if approval_status == "approved" else "none"

        return {"payload": payload}
