import copy
from typing import Any, Dict, Optional


class TradingSignalEvaluationEngine:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
        """
        Safe nested getter using dot-paths.
        Example: _get(payload, "risk.proposed_entry")
        """
        current: Any = d
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
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
    def _infer_rsi_reversal(trigger_reason: str) -> bool:
        text = (trigger_reason or "").lower()
        return "rsi reversal" in text or "rsi recovery" in text

    @staticmethod
    def _infer_macd_improving(trigger_reason: str) -> bool:
        text = (trigger_reason or "").lower()
        return "macd fade" in text or "macd improvement" in text or "macd exhaustion" in text

    @staticmethod
    def _round_position_size(units: float) -> float:
        # For FX units, integer-like sizing is usually cleaner for execution.
        if units <= 0:
            return 0.0
        return float(round(units))

    def evaluate(self, full_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
        {
          "payload": { ...canonical St Ludaetuc payload... },
          "account": {
            "balance": ...,
            "margin_available": ...,
            ...
          }
        }

        Output:
        {
          "payload": { ...same canonical payload with manus block populated... }
        }
        """
        input_payload = copy.deepcopy(full_input.get("payload", {}))
        account = full_input.get("account", {}) or {}

        # -----------------------------
        # Extract canonical fields
        # -----------------------------
        signal_id = self._get(input_payload, "meta.signal_id")
        direction = self._get(input_payload, "signal.direction", "unknown")
        trigger_reason = self._get(input_payload, "signal.trigger_reason", "")

        proposed_entry = self._to_float(self._get(input_payload, "risk.proposed_entry", None), 0.0)
        proposed_stop_loss = self._to_float(self._get(input_payload, "risk.proposed_stop_loss", None), 0.0)
        proposed_take_profit = self._to_float(self._get(input_payload, "risk.proposed_take_profit", None), 0.0)

        rr_ratio = self._to_float(self._get(input_payload, "risk.rr_ratio", None), 0.0)
        stop_distance_price = self._to_float(self._get(input_payload, "risk.stop_distance_price", None), 0.0)
        max_spread_allowed = self._to_float(self._get(input_payload, "risk.max_spread_allowed", None), 0.0)

        spread_price_raw = self._get(input_payload, "price.spread_price", None)
        spread_price = None if spread_price_raw is None else self._to_float(spread_price_raw, 0.0)

        band_touch_confirmed = self._to_bool(
            self._get(input_payload, "signal.band_touch_confirmed", False),
            False,
        )
        regime_ok = self._to_bool(self._get(input_payload, "signal.regime_ok", False), False)

        vwap_deviation_percent = self._to_float(
            self._get(input_payload, "context.vwap_deviation_percent", None),
            0.0,
        )

        market_regime = str(self._get(input_payload, "structure.market_regime", "") or "")
        volatility_regime = str(self._get(input_payload, "structure.volatility_regime", "") or "")
        session_name = str(self._get(input_payload, "market.session_name", "unknown") or "unknown")

        bb_width_percent = self._to_float(
            self._get(input_payload, "context.bb_width_percent", None),
            0.0,
        )
        min_bb_width_pct = self._to_float(
            self._get(input_payload, "extensions.strategy_params.min_bb_width_pct", None),
            0.0,
        )
        max_bb_width_pct = self._to_float(
            self._get(input_payload, "extensions.strategy_params.max_bb_width_pct", None),
            999999.0,
        )

        min_stop_pips = self._to_float(
            self._get(input_payload, "extensions.strategy_params.min_stop_pips", None),
            0.0,
        )
        max_stop_pips = self._to_float(
            self._get(input_payload, "extensions.strategy_params.max_stop_pips", None),
            999999.0,
        )

        pip_size = self._to_float(self._get(input_payload, "instrument.pip_size", None), 0.0001)
        risk_percent = self._to_float(self._get(input_payload, "risk.risk_percent", None), 0.5)
        account_balance = self._to_float(account.get("balance", None), 0.0)
        margin_available = self._to_float(account.get("margin_available", None), 0.0)

        confidence_raw = self._to_float(self._get(input_payload, "signal.confidence_raw", None), 0.0)

        # -----------------------------
        # Derived booleans
        # -----------------------------
        rsi_reversal_present = self._infer_rsi_reversal(trigger_reason)
        macd_improving = self._infer_macd_improving(trigger_reason)

        bb_width_ok = min_bb_width_pct <= bb_width_percent <= max_bb_width_pct if max_bb_width_pct >= min_bb_width_pct else False
        stop_distance_pips = stop_distance_price / pip_size if pip_size > 0 and stop_distance_price > 0 else 0.0

        # -----------------------------
        # Initialize manus block
        # -----------------------------
        manus_block: Dict[str, Any] = {
            "signal_quality_score": 0,
            "risk_score": 0,
            "context_score": 0,
            "strategy_fit_score": 0,
            "expected_value_score": 0,
            "approval_status": "rejected",
            "approval_reason": "Initial evaluation pending",
            "final_entry": None,
            "final_stop_loss": None,
            "final_take_profit": None,
            "final_position_size": None,
        }

        # -----------------------------
        # 1. Signal Quality Score
        # -----------------------------
        signal_quality_score = 0

        if band_touch_confirmed:
            signal_quality_score += 20

        if abs(vwap_deviation_percent) > 0.075:
            signal_quality_score += 20

        if rsi_reversal_present:
            signal_quality_score += 15

        if macd_improving:
            signal_quality_score += 15

        if regime_ok:
            signal_quality_score += 10

        # Optional use of Pine's own confidence score as a soft boost
        if confidence_raw >= 80:
            signal_quality_score += 10
        elif confidence_raw >= 65:
            signal_quality_score += 5

        manus_block["signal_quality_score"] = min(100, round(signal_quality_score))

        # -----------------------------
        # 2. Risk Score
        # -----------------------------
        risk_score = 100
        risk_notes = []

        if rr_ratio < 1.2:
            risk_score -= 30
            risk_notes.append("RR below 1.2")

        if stop_distance_pips > 0:
            if stop_distance_pips < min_stop_pips:
                risk_score -= 20
                risk_notes.append("Stop too small")
            elif stop_distance_pips > max_stop_pips:
                risk_score -= 20
                risk_notes.append("Stop too large")
        else:
            risk_score -= 20
            risk_notes.append("Missing stop distance")

        # Only penalize spread if spread exists.
        if spread_price is not None and max_spread_allowed > 0:
            if spread_price > max_spread_allowed:
                risk_score -= 30
                risk_notes.append("Spread above allowed max")

        # Optional basic margin sanity
        if margin_available <= 0:
            risk_score -= 20
            risk_notes.append("No margin available")

        manus_block["risk_score"] = max(0, round(risk_score))

        # -----------------------------
        # 3. Context Score
        # -----------------------------
        context_score = 70

        if session_name == "unknown":
            context_score -= 30

        context_status = str(self._get(input_payload, "context.context_status", "unchecked") or "unchecked")
        if context_status == "blocked":
            context_score = 0
        elif context_status == "unchecked":
            # Neutral for v1
            pass

        manus_block["context_score"] = max(0, round(context_score))

        # -----------------------------
        # 4. Strategy Fit Score
        # -----------------------------
        strategy_fit_score = 0

        if market_regime == "mean_reversion_friendly":
            strategy_fit_score += 40

        if volatility_regime in {"low", "normal"}:
            strategy_fit_score += 30

        if bb_width_ok:
            strategy_fit_score += 30

        manus_block["strategy_fit_score"] = min(100, round(strategy_fit_score))

        # -----------------------------
        # 5. Expected Value Score
        # -----------------------------
        expected_value_score = (
            manus_block["signal_quality_score"]
            + manus_block["risk_score"]
            + manus_block["strategy_fit_score"]
        ) / 3.0

        manus_block["expected_value_score"] = round(expected_value_score)

        # -----------------------------
        # Approval Logic
        # -----------------------------
        approval_status = "rejected"
        approval_reason = ""

        # Hard rejects
        if not signal_id:
            approval_reason = "Missing meta.signal_id."
        elif proposed_entry <= 0 or proposed_stop_loss <= 0 or proposed_take_profit <= 0:
            approval_reason = "Missing proposed execution prices."
        elif rr_ratio < 1.1:
            approval_reason = "Risk-reward ratio too low."
        elif manus_block["risk_score"] < 50:
            approval_reason = "Risk score too low."
        elif direction not in {"long", "short"}:
            approval_reason = "Invalid signal direction."
        elif account_balance <= 0:
            approval_reason = "Invalid account balance."
        # Approve
        elif (
            manus_block["signal_quality_score"] >= 70
            and manus_block["risk_score"] >= 60
            and manus_block["strategy_fit_score"] >= 60
            and manus_block["expected_value_score"] >= 55
        ):
            approval_status = "approved"
            approval_reason = "Signal meets all approval criteria."
        else:
            approval_reason = "Signal does not meet approval thresholds."

        manus_block["approval_status"] = approval_status
        manus_block["approval_reason"] = approval_reason

        # -----------------------------
        # Final execution parameters
        # -----------------------------
        final_entry = proposed_entry
        final_stop_loss = proposed_stop_loss
        final_take_profit = proposed_take_profit
        final_position_size: Optional[float] = None

        if approval_status == "approved":
            risk_fraction = risk_percent / 100.0  # Pine sends 0.5 for 0.5%
            risk_per_trade = account_balance * risk_fraction

            if stop_distance_price > 0:
                # This is a simple v1 size model.
                # Later you may replace this with broker/instrument-aware sizing.
                units = risk_per_trade / stop_distance_price
                final_position_size = self._round_position_size(units)
            else:
                final_position_size = 0.0
        else:
            final_position_size = None

        manus_block["final_entry"] = final_entry
        manus_block["final_stop_loss"] = final_stop_loss
        manus_block["final_take_profit"] = final_take_profit
        manus_block["final_position_size"] = final_position_size

        # -----------------------------
        # Write only to payload.manus
        # -----------------------------
        input_payload["manus"] = manus_block

        return {"payload": input_payload}
