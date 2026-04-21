import copy
import uuid
from typing import Any, Dict, Optional


class TradingSignalEvaluationEngine:
    def __init__(self) -> None:
        # Strategy parameters for current GBP/USD 2m setup
        self.min_rr_long = 1.6
        self.min_rr_short = 1.8
        self.min_rr_countertrend = 2.0
        self.max_account_risk_pct = 1.0

    @staticmethod
    def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
        current: Any = d
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
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
            if v in {"true", "1", "yes"}:
                return True
            if v in {"false", "0", "no"}:
                return False
        return default

    @staticmethod
    def _round_units(units: float) -> int:
        if units == 0:
            return 0
        return int(round(units))

    def _validate_payload(self, payload: Dict[str, Any], account: Dict[str, Any]) -> None:
        schema_name = self._get(payload, "schema_name")
        schema_version = self._get(payload, "schema_version")
        event_type = self._get(payload, "event_type")
        direction = self._get(payload, "signal.direction")
        entry = self._get(payload, "risk.proposed_entry")
        stop = self._get(payload, "risk.proposed_stop_loss")
        take_profit = self._get(payload, "risk.proposed_take_profit")
        rr_ratio = self._get(payload, "risk.rr_ratio")
        symbol = self._get(payload, "instrument.symbol")
        asset_class = self._get(payload, "instrument.asset_class")
        timeframe = self._get(payload, "market.timeframe")

        if schema_name != "st_ludaetuc_canonical_payload":
            raise ValueError("Invalid schema_name")
        if schema_version != "1.0.0":
            raise ValueError("Invalid schema_version")
        if event_type != "signal":
            raise ValueError("Invalid event_type")
        if direction not in {"long", "short"}:
            raise ValueError("Invalid signal.direction")
        if symbol in (None, ""):
            raise ValueError("Missing instrument.symbol")
        if asset_class in (None, ""):
            raise ValueError("Missing instrument.asset_class")
        if timeframe in (None, ""):
            raise ValueError("Missing market.timeframe")
        if entry in (None, "") or stop in (None, "") or take_profit in (None, ""):
            raise ValueError("Missing proposed execution prices")
        if rr_ratio in (None, ""):
            raise ValueError("Missing risk.rr_ratio")

        balance = self._to_float(account.get("balance"), 0.0)
        margin_available = self._to_float(account.get("margin_available"), 0.0)
        if balance <= 0:
            raise ValueError("Invalid account.balance")
        if margin_available < 0:
            raise ValueError("Invalid account.margin_available")

    def _score_signal_quality(self, payload: Dict[str, Any]) -> int:
        direction = self._get(payload, "signal.direction", "neutral")
        strength_label = self._get(payload, "signal.strength_label", "unknown")
        trigger_reason = str(self._get(payload, "signal.trigger_reason", "") or "")
        rsi = self._to_float(self._get(payload, "indicators.rsi"), 0.0)
        ema_fast = self._to_float(self._get(payload, "indicators.ema_fast"), 0.0)
        ema_slow = self._to_float(self._get(payload, "indicators.ema_slow"), 0.0)
        macd_hist = self._to_float(self._get(payload, "indicators.macd_histogram"), 0.0)
        volatility_regime = str(self._get(payload, "structure.volatility_regime", "unknown") or "unknown")
        market_regime = str(self._get(payload, "structure.market_regime", "unknown") or "unknown")

        score = 0

        strength_map = {
            "very_strong": 25,
            "strong": 20,
            "moderate": 15,
            "weak": 8,
            "very_weak": 3,
            "unknown": 5,
        }
        score += strength_map.get(strength_label, 5)

        if trigger_reason:
            score += 10

        if direction == "long":
            if rsi > 48:
                score += 10
            if ema_fast >= ema_slow:
                score += 15
            if macd_hist >= 0:
                score += 10
        elif direction == "short":
            if rsi < 52:
                score += 10
            if ema_fast <= ema_slow:
                score += 15
            if macd_hist <= 0:
                score += 10

        if volatility_regime in {"normal", "high"}:
            score += 10
        elif volatility_regime == "very_low":
            score -= 10

        if market_regime in {"trend", "reversal", "trend_pullback", "range_to_reversal"}:
            score += 10

        return max(0, min(100, round(score)))

    def _score_risk(self, payload: Dict[str, Any], account: Dict[str, Any]) -> int:
        rr_ratio = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)
        stop_distance_price = self._to_float(self._get(payload, "risk.stop_distance_price"), 0.0)
        risk_percent = self._to_float(self._get(payload, "risk.risk_percent"), 0.0)
        position_size_units = self._to_float(self._get(payload, "risk.position_size_units"), 0.0)
        margin_available = self._to_float(account.get("margin_available"), 0.0)
        drawdown_guard_status = str(self._get(payload, "risk.drawdown_guard_status", "unknown") or "unknown")

        score = 100

        if rr_ratio < 1.2:
            score -= 40
        elif rr_ratio < 1.6:
            score -= 20

        if stop_distance_price <= 0:
            score -= 30

        if risk_percent <= 0 or risk_percent > self.max_account_risk_pct:
            score -= 20

        if position_size_units <= 0:
            score -= 20

        if margin_available <= 0:
            score -= 30

        if drawdown_guard_status == "warn":
            score -= 15
        elif drawdown_guard_status == "fail":
            score = 0

        return max(0, min(100, round(score)))

    def _score_context(self, payload: Dict[str, Any]) -> int:
        session_name = str(self._get(payload, "market.session_name", "unknown") or "unknown")
        session_phase = str(self._get(payload, "market.session_phase", "unknown") or "unknown")
        context_status = str(self._get(payload, "context.context_status", "unknown") or "unknown")

        score = 50

        if session_name == "overlap_london_new_york":
            score += 35
        elif session_name == "london":
            score += 30
        elif session_name == "new_york":
            score += 20
        elif session_name == "asia":
            score += 5
        elif session_name == "overnight":
            score -= 25
        elif session_name == "unknown":
            score -= 15

        if session_phase == "active":
            score += 10
        elif session_phase == "late":
            score -= 5
        elif session_phase == "post_close":
            score -= 10

        if context_status == "clear":
            score += 10
        elif context_status == "warning":
            score -= 15
        elif context_status == "blocked":
            score = 0
        elif context_status == "unchecked":
            score -= 5

        return max(0, min(100, round(score)))

    def _score_strategy_fit(self, payload: Dict[str, Any]) -> int:
        direction = str(self._get(payload, "signal.direction", "neutral") or "neutral")
        trend_bias = str(self._get(payload, "structure.trend_bias", "unknown") or "unknown")
        htf_bias = str(self._get(payload, "structure.higher_timeframe_bias", "unknown") or "unknown")
        volatility_regime = str(self._get(payload, "structure.volatility_regime", "unknown") or "unknown")
        market_regime = str(self._get(payload, "structure.market_regime", "unknown") or "unknown")
        countertrend_override = self._to_bool(self._get(payload, "extensions.countertrend_short_override"), False)

        score = 0

        if direction == "long":
            if htf_bias == "bullish":
                score += 35
            elif htf_bias == "mixed":
                score += 20
            elif htf_bias == "bearish":
                score -= 20

            if trend_bias == "bullish":
                score += 20

        elif direction == "short":
            if htf_bias == "bearish":
                score += 35
            elif htf_bias == "mixed":
                score += 10
            elif htf_bias == "bullish":
                score -= 25

            if trend_bias == "bearish":
                score += 20

            if countertrend_override:
                score += 10

        if volatility_regime in {"normal", "high"}:
            score += 20
        elif volatility_regime == "very_low":
            score -= 15

        if market_regime in {"trend", "reversal", "range_to_reversal"}:
            score += 15

        return max(0, min(100, round(score)))

    def _score_expected_value(
        self,
        signal_quality_score: int,
        risk_score: int,
        context_score: int,
        strategy_fit_score: int,
        payload: Dict[str, Any],
    ) -> int:
        rr_ratio = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)

        score = (
            0.25 * signal_quality_score
            + 0.25 * risk_score
            + 0.20 * context_score
            + 0.30 * strategy_fit_score
        )

        if rr_ratio >= 2.0:
            score += 5
        elif rr_ratio < 1.5:
            score -= 10

        return max(0, min(100, round(score)))

    def _approval_decision(
        self,
        payload: Dict[str, Any],
        account: Dict[str, Any],
        signal_quality_score: int,
        risk_score: int,
        context_score: int,
        strategy_fit_score: int,
        expected_value_score: int,
    ) -> Dict[str, Optional[str]]:
        direction = str(self._get(payload, "signal.direction", "neutral") or "neutral")
        rr_ratio = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)
        htf_bias = str(self._get(payload, "structure.higher_timeframe_bias", "unknown") or "unknown")
        countertrend_override = self._to_bool(self._get(payload, "extensions.countertrend_short_override"), False)
        margin_available = self._to_float(account.get("margin_available"), 0.0)
        position_size_units = self._to_float(self._get(payload, "risk.position_size_units"), 0.0)

        if direction == "long" and rr_ratio < self.min_rr_long:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "RR below long minimum threshold",
                "rejection_stage": "risk",
                "recommendation": "reject",
            }

        if direction == "short":
            min_rr = self.min_rr_countertrend if htf_bias == "bullish" else self.min_rr_short
            if rr_ratio < min_rr:
                return {
                    "approval_status": "rejected",
                    "approval_reason": None,
                    "rejection_reason": "RR below short minimum threshold",
                    "rejection_stage": "risk",
                    "recommendation": "reject",
                }

            if htf_bias == "bullish" and not countertrend_override:
                return {
                    "approval_status": "rejected",
                    "approval_reason": None,
                    "rejection_reason": "Short conflicts with bullish HTF bias without override",
                    "rejection_stage": "strategy_fit",
                    "recommendation": "reject",
                }

        if margin_available <= 0:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Insufficient available margin",
                "rejection_stage": "risk",
                "recommendation": "reject",
            }

        if position_size_units <= 0:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Invalid proposed position size",
                "rejection_stage": "risk",
                "recommendation": "reject",
            }

        if signal_quality_score < 65:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Signal quality too low",
                "rejection_stage": "signal_quality",
                "recommendation": "reject",
            }

        if risk_score < 60:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Risk score too low",
                "rejection_stage": "risk",
                "recommendation": "reject",
            }

        if context_score < 40:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Context score too low",
                "rejection_stage": "context",
                "recommendation": "reject",
            }

        if strategy_fit_score < 60:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Strategy fit too low",
                "rejection_stage": "strategy_fit",
                "recommendation": "reject",
            }

        if expected_value_score < 60:
            return {
                "approval_status": "rejected",
                "approval_reason": None,
                "rejection_reason": "Expected value too low",
                "rejection_stage": "expected_value",
                "recommendation": "reject",
            }

        return {
            "approval_status": "approved",
            "approval_reason": "Signal passed signal quality, risk, context, strategy fit, and expected value checks",
            "rejection_reason": None,
            "rejection_stage": None,
            "recommendation": "execute",
        }

    def _build_execution_plan(
        self,
        payload: Dict[str, Any],
        account: Dict[str, Any],
        approved: bool,
    ) -> Dict[str, Any]:
        direction = str(self._get(payload, "signal.direction", "neutral") or "neutral")
        entry = self._to_float(self._get(payload, "risk.proposed_entry"), 0.0)
        stop = self._to_float(self._get(payload, "risk.proposed_stop_loss"), 0.0)
        take_profit = self._to_float(self._get(payload, "risk.proposed_take_profit"), 0.0)
        rr_ratio = self._to_float(self._get(payload, "risk.rr_ratio"), 0.0)
        units = self._to_float(self._get(payload, "risk.position_size_units"), 0.0)
        stop_distance_price = self._to_float(self._get(payload, "risk.stop_distance_price"), 0.0)
        risk_percent = self._to_float(self._get(payload, "risk.risk_percent"), 0.0)
        balance = self._to_float(account.get("balance"), 0.0)

        final_units: Optional[int]
        if approved:
            risk_amount = balance * (risk_percent / 100.0)
            if stop_distance_price > 0:
                inferred_units = risk_amount / stop_distance_price
                signed_units = inferred_units if direction == "long" else -inferred_units
                final_units = self._round_units(signed_units)
            else:
                final_units = self._round_units(units if direction == "long" else -abs(units))
        else:
            final_units = None

        lots = None if final_units is None else round(final_units / 100000.0, 5)

        if approved:
            trade_plan = (
                f"{'Buy' if direction == 'long' else 'Sell'} "
                f"{self._get(payload, 'instrument.broker_symbol', 'UNKNOWN')} "
                f"at {entry} with stop {stop} and target {take_profit}"
            )
        else:
            trade_plan = None

        return {
            "final_entry": entry if approved else None,
            "final_stop_loss": stop if approved else None,
            "final_take_profit": take_profit if approved else None,
            "final_position_size": final_units,
            "final_position_size_lots": lots,
            "final_rr_ratio": rr_ratio if approved else None,
            "final_holding_period_est": "intraday_20_to_60m" if approved else None,
            "final_trade_plan": trade_plan,
        }

    def evaluate(self, full_input: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(full_input.get("payload", {}))
        account = full_input.get("account", {}) or {}

        self._validate_payload(payload, account)

        signal_quality_score = self._score_signal_quality(payload)
        risk_score = self._score_risk(payload, account)
        context_score = self._score_context(payload)
        strategy_fit_score = self._score_strategy_fit(payload)
        expected_value_score = self._score_expected_value(
            signal_quality_score,
            risk_score,
            context_score,
            strategy_fit_score,
            payload,
        )

        execution_quality_score = min(100, round((risk_score + context_score) / 2))
        confidence_score = min(
            100,
            round(
                0.25 * signal_quality_score
                + 0.25 * risk_score
                + 0.20 * context_score
                + 0.30 * strategy_fit_score
            ),
        )

        decision = self._approval_decision(
            payload,
            account,
            signal_quality_score,
            risk_score,
            context_score,
            strategy_fit_score,
            expected_value_score,
        )

        execution_plan = self._build_execution_plan(
            payload=payload,
            account=account,
            approved=decision["approval_status"] == "approved",
        )

        regime_classification = (
            f"{self._get(payload, 'structure.market_regime', 'unknown')}_"
            f"{self._get(payload, 'signal.direction', 'neutral')}_"
            f"{self._get(payload, 'structure.volatility_regime', 'unknown')}_"
            f"{self._get(payload, 'market.session_name', 'unknown')}"
        )

        notes = [
            f"Session={self._get(payload, 'market.session_name', 'unknown')}",
            f"HTF bias={self._get(payload, 'structure.higher_timeframe_bias', 'unknown')}",
            f"Direction={self._get(payload, 'signal.direction', 'neutral')}",
            f"RR={self._get(payload, 'risk.rr_ratio', None)}",
            f"Context={self._get(payload, 'context.context_status', 'unknown')}",
        ]

        manus_block = {
            "signal_quality_score": signal_quality_score,
            "risk_score": risk_score,
            "context_score": context_score,
            "strategy_fit_score": strategy_fit_score,
            "expected_value_score": expected_value_score,
            "approval_status": decision["approval_status"],
            "approval_reason": decision["approval_reason"],
            "final_entry": execution_plan["final_entry"],
            "final_stop_loss": execution_plan["final_stop_loss"],
            "final_take_profit": execution_plan["final_take_profit"],
            "final_position_size": execution_plan["final_position_size"],
        }

        payload["manus"] = manus_block

        return {
            "request_id": str(uuid.uuid4()),
            "status": "ok",
            "payload": payload,
            "decision": decision,
            "scores": {
                "signal_quality_score": signal_quality_score,
                "risk_score": risk_score,
                "context_score": context_score,
                "strategy_fit_score": strategy_fit_score,
                "expected_value_score": expected_value_score,
                "execution_quality_score": execution_quality_score,
                "confidence_score": confidence_score,
            },
            "classification": {
                "asset_specific_logic_used": "forex_major_fx_v2",
                "regime_classification": regime_classification,
            },
            "execution_plan": execution_plan,
            "notes": notes,
        }
