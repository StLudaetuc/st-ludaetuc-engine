# engine.py

import copy
import math
from typing import Any, Dict, Optional, Tuple


class TradingSignalEvaluationEngine:
    """
    St Ludaetuc Manus Evaluation Engine
    Supports:
      - GBP/USD forex ruleset
      - XAU/USD gold/metals ruleset

    Input:
      {
        "payload": {... canonical payload ...},
        "account": {
          "balance": ...,
          "margin_available": ...,
          "open_trade_count": optional,
          "portfolio_heat_percent": optional
        }
      }

    Output:
      {
        "payload": {... same payload with populated manus block ...}
      }
    """

    # ---------------------------------------------------------------------
    # Generic helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
        cur: Any = d
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, str) and value.lower() in {"null", "none", ""}:
                return default
            return float(value)
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
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _round_to_step(units: float, step: float) -> float:
        if step <= 0:
            return units
        return math.floor(units / step) * step

    @staticmethod
    def _round_price(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 10)

    def _base_manus_block(self) -> Dict[str, Any]:
        return {
            "signal_quality_score": 0,
            "risk_score": 0,
            "context_score": 0,
            "strategy_fit_score": 0,
            "execution_quality_score": 0,
            "expected_value_score": 0,
            "confidence_score": 0,
            "approval_status": "rejected",
            "approval_reason": "Initial evaluation pending",
            "rejection_category": None,
            "final_entry": None,
            "final_stop_loss": None,
            "final_take_profit": None,
            "final_position_size": None,
            "ruleset_used": None,
            "risk_notes": [],
            "strategy_notes": [],
        }

    def _reject(
        self,
        payload: Dict[str, Any],
        reason: str,
        category: str = "general",
        ruleset: str = "unknown",
        scores: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out = copy.deepcopy(payload)
        manus = self._base_manus_block()
        if scores:
            manus.update(scores)
        manus["approval_status"] = "rejected"
        manus["approval_reason"] = reason
        manus["rejection_category"] = category
        manus["ruleset_used"] = ruleset
        out["manus"] = manus
        return {"payload": out}

    # ---------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------
    def evaluate(self, full_input: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(full_input.get("payload", {}) or {})
        account = full_input.get("account", {}) or {}

        asset_class = str(self._get(payload, "instrument.asset_class", "") or "").lower()
        broker_symbol = str(self._get(payload, "instrument.broker_symbol", "") or "").upper()
        symbol = str(self._get(payload, "instrument.symbol", "") or "").upper()

        if asset_class == "forex" and broker_symbol in {"GBP_USD", "GBPUSD"}:
            return self.evaluate_gbpusd(payload, account)

        if asset_class in {"metals", "commodities"} and broker_symbol in {"XAU_USD", "XAUUSD"}:
            return self.evaluate_xauusd(payload, account)

        if symbol in {"GBPUSD", "GBP/USD"}:
            return self.evaluate_gbpusd(payload, account)

        if symbol in {"XAUUSD", "XAU/USD"}:
            return self.evaluate_xauusd(payload, account)

        return self._reject(
            payload,
            f"Unsupported instrument/ruleset: asset_class={asset_class}, broker_symbol={broker_symbol}, symbol={symbol}",
            "unsupported_instrument",
            "router",
        )

    # ---------------------------------------------------------------------
    # Shared extraction / sizing
    # ---------------------------------------------------------------------
    def _extract_common(self, payload: Dict[str, Any], account: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._to_float(self._get(payload, "risk.proposed_entry"), 0.0)
        sl = self._to_float(self._get(payload, "risk.proposed_stop_loss"), 0.0)
        tp = self._to_float(self._get(payload, "risk.proposed_take_profit"), 0.0)

        stop_dist = self._to_float(self._get(payload, "risk.stop_distance_price"), 0.0)
        if stop_dist <= 0 and entry > 0 and sl > 0:
            stop_dist = abs(entry - sl)

        target_dist = self._to_float(self._get(payload, "risk.target_distance_price"), 0.0)
        if target_dist <= 0 and entry > 0 and tp > 0:
            target_dist = abs(tp - entry)

        rr = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)
        if rr <= 0 and stop_dist > 0 and target_dist > 0:
            rr = target_dist / stop_dist

        return {
            "signal_id": self._get(payload, "meta.signal_id"),
            "direction": str(self._get(payload, "signal.direction", "unknown") or "unknown").lower(),
            "signal_type": str(self._get(payload, "signal.signal_type", "entry") or "entry").lower(),
            "confidence_raw": self._to_float(self._get(payload, "signal.confidence_raw"), 0.0),
            "entry_model": str(self._get(payload, "signal.entry_model", "") or ""),
            "trigger_reason": str(self._get(payload, "signal.trigger_reason", "") or ""),
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "stop_dist": stop_dist,
            "target_dist": target_dist,
            "rr": rr,
            "risk_percent": self._to_float(self._get(payload, "risk.risk_percent"), 0.25),
            "max_spread_allowed": self._to_float(self._get(payload, "risk.max_spread_allowed"), 0.0),
            "max_slippage_allowed": self._to_float(self._get(payload, "risk.max_slippage_allowed"), 0.0),
            "spread_price": self._get(payload, "price.spread_price", None),
            "session": str(self._get(payload, "market.session_name", "unknown") or "unknown").lower(),
            "session_phase": str(self._get(payload, "market.session_phase", "unknown") or "unknown").lower(),
            "vol_regime": str(self._get(payload, "structure.volatility_regime", "unknown") or "unknown").lower(),
            "market_regime": str(self._get(payload, "structure.market_regime", "unknown") or "unknown").lower(),
            "htf_bias": str(self._get(payload, "structure.higher_timeframe_bias", "unknown") or "unknown").lower(),
            "trend_bias": str(self._get(payload, "structure.trend_bias", "unknown") or "unknown").lower(),
            "context_status": str(self._get(payload, "context.context_status", "unchecked") or "unchecked").lower(),
            "pip_size": self._to_float(self._get(payload, "instrument.pip_size"), 0.0001),
            "tick_size": self._to_float(self._get(payload, "instrument.tick_size"), 0.00001),
            "unit_step": self._to_float(self._get(payload, "instrument.unit_step"), 1.0),
            "min_units": self._to_float(self._get(payload, "instrument.min_units"), 1.0),
            "max_units": self._to_float(self._get(payload, "instrument.max_units"), 100000.0),
            "balance": self._to_float(account.get("balance"), 0.0),
            "margin_available": self._to_float(account.get("margin_available"), 0.0),
            "open_trade_count": self._to_int(account.get("open_trade_count"), 0),
            "portfolio_heat_percent": self._to_float(account.get("portfolio_heat_percent"), 0.0),
        }

    def _position_size(
        self,
        balance: float,
        risk_percent: float,
        stop_dist: float,
        min_units: float,
        max_units: float,
        unit_step: float,
        direction: str,
        risk_cap: float,
    ) -> float:
        risk_percent = min(max(risk_percent, 0.0), risk_cap)
        risk_amount = balance * (risk_percent / 100.0)

        if balance <= 0 or stop_dist <= 0 or risk_amount <= 0:
            return 0.0

        raw_units = risk_amount / stop_dist
        units = self._round_to_step(raw_units, unit_step)
        units = self._clamp(units, min_units, max_units)

        if direction == "short":
            return -abs(units)
        return abs(units)

    def _validate_core_prices(self, c: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        if not c["signal_id"]:
            return "Missing meta.signal_id", "schema"
        if c["direction"] not in {"long", "short"}:
            return "Invalid or neutral direction", "signal"
        if c["signal_type"] not in {"entry", "signal"}:
            return "Signal type is not an entry", "signal"
        if c["entry"] <= 0 or c["sl"] <= 0 or c["tp"] <= 0:
            return "Missing proposed execution prices", "risk"
        if c["stop_dist"] <= 0:
            return "Missing or invalid stop distance", "risk"
        return None

    # ---------------------------------------------------------------------
    # GBP/USD ruleset
    # ---------------------------------------------------------------------
    def evaluate_gbpusd(self, payload: Dict[str, Any], account: Dict[str, Any]) -> Dict[str, Any]:
        ruleset = "gbpusd_forex_ruleset_v1"
        c = self._extract_common(payload, account)
        manus = self._base_manus_block()
        manus["ruleset_used"] = ruleset

        core_error = self._validate_core_prices(c)
        if core_error:
            reason, cat = core_error
            return self._reject(payload, reason, cat, ruleset)

        # GBP/USD guardrails
        GBP_RISK_CAP = 0.50
        GBP_MIN_RR = 0.55
        GBP_MAX_SPREAD = c["max_spread_allowed"] or 0.00015
        GBP_MAX_SLIPPAGE = c["max_slippage_allowed"] or 0.00008
        GBP_ALLOWED_VOL = {"low", "normal", "high"}
        GBP_BLOCK_SESSIONS = {"unknown", "overnight"}
        GBP_MAX_PORTFOLIO_HEAT = 1.50
        GBP_MAX_OPEN_TRADES = 1

        risk_notes = []
        strategy_notes = []

        spread_price = None if c["spread_price"] is None else self._to_float(c["spread_price"], 0.0)

        # Hard rejects
        if c["session"] in GBP_BLOCK_SESSIONS:
            return self._reject(payload, f"GBP/USD blocked session: {c['session']}", "context", ruleset)

        if c["vol_regime"] in {"very_low", "extreme", "unknown"}:
            return self._reject(payload, f"GBP/USD blocked volatility regime: {c['vol_regime']}", "strategy_fit", ruleset)

        if c["rr"] < GBP_MIN_RR:
            return self._reject(payload, f"GBP/USD RR below minimum: {c['rr']:.2f}", "risk", ruleset)

        if spread_price is not None and spread_price > GBP_MAX_SPREAD:
            return self._reject(payload, "GBP/USD spread above allowed maximum", "execution_cost", ruleset)

        if c["balance"] <= 0:
            return self._reject(payload, "Invalid account balance", "account", ruleset)

        if c["margin_available"] <= 0:
            return self._reject(payload, "No margin available", "account", ruleset)

        if c["open_trade_count"] >= GBP_MAX_OPEN_TRADES:
            return self._reject(payload, "GBP/USD max open trades reached", "portfolio_heat", ruleset)

        if c["portfolio_heat_percent"] > GBP_MAX_PORTFOLIO_HEAT:
            return self._reject(payload, "Portfolio heat above GBP/USD limit", "portfolio_heat", ruleset)

        # Scores
        signal_quality = 40
        if c["confidence_raw"] >= 80:
            signal_quality += 20
        elif c["confidence_raw"] >= 65:
            signal_quality += 12
        elif c["confidence_raw"] >= 50:
            signal_quality += 6

        if c["direction"] == "long" and c["htf_bias"] == "bullish":
            signal_quality += 15
        if c["direction"] == "short" and c["htf_bias"] == "bearish":
            signal_quality += 15

        if c["trend_bias"] in {"bullish", "bearish"}:
            signal_quality += 10

        if "score" in c["entry_model"] or "profit_factor" in c["entry_model"]:
            signal_quality += 5

        risk_score = 100
        if c["rr"] < 0.70:
            risk_score -= 20
            risk_notes.append("Low scalp RR")
        if c["stop_dist"] / c["pip_size"] < 2:
            risk_score -= 20
            risk_notes.append("Stop distance very tight")
        if c["risk_percent"] > GBP_RISK_CAP:
            risk_score -= 25
            risk_notes.append("Risk above GBP cap")
        if spread_price is None:
            risk_score -= 5
            risk_notes.append("Spread unavailable")

        context_score = 70
        if c["session"] in {"london", "overlap_london_new_york", "new_york"}:
            context_score += 15
        if c["context_status"] == "blocked":
            context_score = 0
        elif c["context_status"] == "warning":
            context_score -= 20

        strategy_fit = 50
        if c["vol_regime"] == "normal":
            strategy_fit += 20
        elif c["vol_regime"] == "high":
            strategy_fit += 15
        elif c["vol_regime"] == "low":
            strategy_fit += 5

        if c["market_regime"] in {"trend_pullback", "high_frequency_loss_reduced", "profit_factor_filtered_score_scalp"}:
            strategy_fit += 20
        elif c["market_regime"] in {"micro_scalp", "high_frequency_score_scalp"}:
            strategy_fit += 10

        execution_quality = 75
        if spread_price is not None and spread_price <= GBP_MAX_SPREAD * 0.65:
            execution_quality += 10

        signal_quality = int(self._clamp(signal_quality, 0, 100))
        risk_score = int(self._clamp(risk_score, 0, 100))
        context_score = int(self._clamp(context_score, 0, 100))
        strategy_fit = int(self._clamp(strategy_fit, 0, 100))
        execution_quality = int(self._clamp(execution_quality, 0, 100))

        expected_value = round(
            signal_quality * 0.30
            + risk_score * 0.25
            + context_score * 0.15
            + strategy_fit * 0.20
            + execution_quality * 0.10
        )

        confidence = expected_value

        approval = (
            signal_quality >= 60
            and risk_score >= 60
            and context_score >= 55
            and strategy_fit >= 55
            and expected_value >= 62
        )

        final_size = None
        if approval:
            final_size = self._position_size(
                c["balance"],
                c["risk_percent"],
                c["stop_dist"],
                c["min_units"],
                c["max_units"],
                c["unit_step"],
                c["direction"],
                GBP_RISK_CAP,
            )

        manus.update({
            "signal_quality_score": signal_quality,
            "risk_score": risk_score,
            "context_score": context_score,
            "strategy_fit_score": strategy_fit,
            "execution_quality_score": execution_quality,
            "expected_value_score": expected_value,
            "confidence_score": confidence,
            "approval_status": "approved" if approval and final_size and abs(final_size) > 0 else "rejected",
            "approval_reason": "GBP/USD signal approved." if approval else "GBP/USD signal did not meet Manus thresholds.",
            "rejection_category": None if approval else "threshold",
            "final_entry": self._round_price(c["entry"], c["tick_size"]),
            "final_stop_loss": self._round_price(c["sl"], c["tick_size"]),
            "final_take_profit": self._round_price(c["tp"], c["tick_size"]),
            "final_position_size": final_size,
            "risk_notes": risk_notes,
            "strategy_notes": strategy_notes,
        })

        out = copy.deepcopy(payload)
        out["manus"] = manus
        return {"payload": out}

    # ---------------------------------------------------------------------
    # XAU/USD gold ruleset
    # ---------------------------------------------------------------------
    def evaluate_xauusd(self, payload: Dict[str, Any], account: Dict[str, Any]) -> Dict[str, Any]:
        ruleset = "xauusd_gold_ruleset_v1"
        c = self._extract_common(payload, account)
        manus = self._base_manus_block()
        manus["ruleset_used"] = ruleset

        core_error = self._validate_core_prices(c)
        if core_error:
            reason, cat = core_error
            return self._reject(payload, reason, cat, ruleset)

        # XAU-specific guardrails
        XAU_RISK_CAP = 0.25
        XAU_MIN_RR = 1.20
        XAU_MAX_SPREAD = c["max_spread_allowed"] or 0.35
        XAU_MIN_STOP = 0.90
        XAU_MAX_STOP = 6.00
        XAU_MAX_OPEN_TRADES = 2
        XAU_MAX_PORTFOLIO_HEAT = 1.00
        XAU_ALLOWED_VOL = {"normal", "high"}
        XAU_BLOCK_SESSIONS = {"unknown", "overnight"}

        risk_notes = []
        strategy_notes = []

        spread_price = None if c["spread_price"] is None else self._to_float(c["spread_price"], 0.0)

        ext = payload.get("extensions", {}) or {}

        rr_pass = self._to_bool(ext.get("rr_pass"), c["rr"] >= XAU_MIN_RR)
        chop_blocked = self._to_bool(ext.get("chop_blocked"), False)
        spike_blocked = self._to_bool(ext.get("spike_cooldown_blocked"), False)
        vol_blocked = self._to_bool(ext.get("volatility_blocked"), False)
        session_blocked = self._to_bool(ext.get("session_blocked"), False)

        pyramid_layer = self._to_int(ext.get("pyramid_layer"), 1)
        current_open_layers = self._to_int(ext.get("current_open_layers"), 0)
        max_pyramid_layers = self._to_int(ext.get("max_pyramid_layers"), 2)

        pyramid_long_quality = self._to_bool(ext.get("pyramid_long_quality"), False)
        pyramid_short_quality = self._to_bool(ext.get("pyramid_short_quality"), False)

        room_long_pass = self._to_bool(ext.get("room_long_pass"), True)
        room_short_pass = self._to_bool(ext.get("room_short_pass"), True)
        quality_gate_long = self._to_bool(ext.get("quality_gate_long"), c["direction"] == "long")
        quality_gate_short = self._to_bool(ext.get("quality_gate_short"), c["direction"] == "short")

        adx = self._to_float(self._get(payload, "indicators.adx"), 0.0)

        # Hard rejects
        if session_blocked or c["session"] in XAU_BLOCK_SESSIONS:
            return self._reject(payload, f"XAU/USD blocked session: {c['session']}", "context", ruleset)

        if vol_blocked or c["vol_regime"] not in XAU_ALLOWED_VOL:
            return self._reject(payload, f"XAU/USD blocked volatility regime: {c['vol_regime']}", "strategy_fit", ruleset)

        if chop_blocked:
            return self._reject(payload, "XAU/USD chop filter blocked candidate", "strategy_fit", ruleset)

        if spike_blocked:
            return self._reject(payload, "XAU/USD post-spike cooldown active", "strategy_fit", ruleset)

        if c["rr"] < XAU_MIN_RR or not rr_pass:
            return self._reject(payload, f"XAU/USD RR below minimum: {c['rr']:.2f}", "risk", ruleset)

        if c["stop_dist"] < XAU_MIN_STOP:
            return self._reject(payload, "XAU/USD stop distance too small", "risk", ruleset)

        if c["stop_dist"] > XAU_MAX_STOP:
            return self._reject(payload, "XAU/USD stop distance too large", "risk", ruleset)

        if spread_price is not None and spread_price > XAU_MAX_SPREAD:
            return self._reject(payload, "XAU/USD spread above allowed maximum", "execution_cost", ruleset)

        if c["balance"] <= 0:
            return self._reject(payload, "Invalid account balance", "account", ruleset)

        if c["margin_available"] <= 0:
            return self._reject(payload, "No margin available", "account", ruleset)

        if current_open_layers >= max_pyramid_layers or c["open_trade_count"] >= XAU_MAX_OPEN_TRADES:
            return self._reject(payload, "XAU/USD max pyramid/open-trade limit reached", "portfolio_heat", ruleset)

        if c["portfolio_heat_percent"] > XAU_MAX_PORTFOLIO_HEAT:
            return self._reject(payload, "Portfolio heat above XAU/USD limit", "portfolio_heat", ruleset)

        if pyramid_layer > 1:
            if c["direction"] == "long" and not pyramid_long_quality:
                return self._reject(payload, "XAU/USD long layer-2 lacks pyramid quality", "pyramiding", ruleset)
            if c["direction"] == "short" and not pyramid_short_quality:
                return self._reject(payload, "XAU/USD short layer-2 lacks pyramid quality", "pyramiding", ruleset)

        if c["direction"] == "long" and not room_long_pass:
            return self._reject(payload, "XAU/USD insufficient room to recent high", "structure", ruleset)

        if c["direction"] == "short" and not room_short_pass:
            return self._reject(payload, "XAU/USD insufficient room to recent low", "structure", ruleset)

        if c["direction"] == "long" and not quality_gate_long:
            return self._reject(payload, "XAU/USD long quality gate failed", "signal_quality", ruleset)

        if c["direction"] == "short" and not quality_gate_short:
            return self._reject(payload, "XAU/USD short quality gate failed", "signal_quality", ruleset)

        # Scores
        signal_quality = 35
        if c["confidence_raw"] >= 85:
            signal_quality += 25
        elif c["confidence_raw"] >= 72:
            signal_quality += 20
        elif c["confidence_raw"] >= 60:
            signal_quality += 10

        if c["direction"] == "long" and c["htf_bias"] == "bullish":
            signal_quality += 15
        if c["direction"] == "short" and c["htf_bias"] == "bearish":
            signal_quality += 15

        if adx >= 20:
            signal_quality += 10
        elif adx >= 16:
            signal_quality += 6

        if c["market_regime"] in {"trend_pullback", "breakout", "range_to_reversal"}:
            signal_quality += 10

        risk_score = 100
        if c["risk_percent"] > XAU_RISK_CAP:
            risk_score -= 25
            risk_notes.append("Risk above XAU cap")
        if c["rr"] < 1.35:
            risk_score -= 10
            risk_notes.append("RR acceptable but modest")
        if spread_price is None:
            risk_score -= 5
            risk_notes.append("Spread unavailable")
        if pyramid_layer > 1:
            risk_score -= 8
            risk_notes.append("Additional pyramid layer risk")

        context_score = 70
        if c["session"] == "overlap_london_new_york":
            context_score += 20
        elif c["session"] == "london":
            context_score += 15
        elif c["session"] == "new_york":
            context_score += 10
        elif c["session"] == "asia":
            context_score -= 5

        if c["context_status"] == "blocked":
            context_score = 0
        elif c["context_status"] == "warning":
            context_score -= 20

        strategy_fit = 45
        if c["vol_regime"] == "normal":
            strategy_fit += 20
        elif c["vol_regime"] == "high":
            strategy_fit += 18

        if c["market_regime"] == "trend_pullback":
            strategy_fit += 20
        elif c["market_regime"] == "breakout":
            strategy_fit += 15
        elif c["market_regime"] == "range_to_reversal":
            strategy_fit += 10

        if not chop_blocked and not spike_blocked:
            strategy_fit += 10

        execution_quality = 75
        if spread_price is not None and spread_price <= XAU_MAX_SPREAD * 0.70:
            execution_quality += 10
        if c["session"] in {"london", "overlap_london_new_york", "new_york"}:
            execution_quality += 5

        signal_quality = int(self._clamp(signal_quality, 0, 100))
        risk_score = int(self._clamp(risk_score, 0, 100))
        context_score = int(self._clamp(context_score, 0, 100))
        strategy_fit = int(self._clamp(strategy_fit, 0, 100))
        execution_quality = int(self._clamp(execution_quality, 0, 100))

        expected_value = round(
            signal_quality * 0.30
            + risk_score * 0.25
            + context_score * 0.15
            + strategy_fit * 0.20
            + execution_quality * 0.10
        )

        confidence = expected_value

        approval = (
            signal_quality >= 70
            and risk_score >= 65
            and context_score >= 55
            and strategy_fit >= 65
            and expected_value >= 68
        )

        final_size = None
        if approval:
            final_size = self._position_size(
                c["balance"],
                c["risk_percent"],
                c["stop_dist"],
                c["min_units"],
                c["max_units"],
                c["unit_step"],
                c["direction"],
                XAU_RISK_CAP,
            )

        manus.update({
            "signal_quality_score": signal_quality,
            "risk_score": risk_score,
            "context_score": context_score,
            "strategy_fit_score": strategy_fit,
            "execution_quality_score": execution_quality,
            "expected_value_score": expected_value,
            "confidence_score": confidence,
            "approval_status": "approved" if approval and final_size and abs(final_size) > 0 else "rejected",
            "approval_reason": "XAU/USD signal approved." if approval else "XAU/USD signal did not meet Manus thresholds.",
            "rejection_category": None if approval else "threshold",
            "final_entry": self._round_price(c["entry"], c["tick_size"]),
            "final_stop_loss": self._round_price(c["sl"], c["tick_size"]),
            "final_take_profit": self._round_price(c["tp"], c["tick_size"]),
            "final_position_size": final_size,
            "risk_notes": risk_notes,
            "strategy_notes": strategy_notes,
        })

        out = copy.deepcopy(payload)
        out["manus"] = manus
        return {"payload": out}
