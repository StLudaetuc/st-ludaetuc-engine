"""Engine v4.0.0 decision-path checks. Run with: python -m pytest tests/ (or
python tests/test_engine_v4.py). No database needed - evaluate() is called
directly on the engine object, not through the HTTP endpoint."""
import importlib
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load(mode):
    os.environ["MANUS_DECISION_MODE"] = mode
    import engine
    return importlib.reload(engine)


def _payload(direction="long", model="trend_pullback_reclaim", pb=0.31, rsi=45.0, macd=-0.00001,
             htf="bullish", trend="bullish", close=1.33000, atr=0.00025):
    return {
        "meta": {"signal_id": f"TEST-{direction}-{model}-{pb}"},
        "system": {"strategy_id": "stl_gbpusd_2m_candidate_generator_v6", "strategy_version": "6.0.0"},
        "instrument": {"symbol": "GBPUSD"},
        "market": {"timeframe": "2", "bar_time_utc": "2026-10-20T10:00:00Z", "hour_utc": 10, "session_name": "london"},
        "signal": {"direction": direction, "entry_model": model},
        "price": {"close": close},
        "indicators": {"atr": atr, "rsi": rsi, "macd_histogram": macd, "bb_percent_b": pb, "adx": 24.0,
                       "ema_fast": close, "ema_slow": close - 0.0002 if direction == "long" else close + 0.0002},
        "structure": {"higher_timeframe_bias": htf, "trend_bias": trend, "volatility_regime": "normal",
                      "market_regime": "trend", "nearest_support": close - 0.0004, "nearest_resistance": close + 0.0004},
        "context": {"context_status": "clear"},
        "risk": {"risk_percent": 0.5},
        "extensions": {"atr_pips": atr / 0.0001, "atr_ratio": 1.05, "distance_from_vwap_atr": 1.5,
                       "pine_proposal_only": True, "allow_manus_override_sl_tp": True},
    }


def _eval(eng, payload):
    return eng.engine.evaluate({"payload": deepcopy(payload), "account": {"balance": 10000}})


def test_v6_candidate_approved_with_stop_floor_and_target():
    eng = _load("v4")
    res = _eval(eng, _payload())
    m = res["manus"]
    assert res["approval_status"] == "approved", res["rejection_reason"]
    plan = m["final_trade_plan"]
    stop_pips = abs(plan["entry"] - plan["stop_loss"]) / 0.0001
    assert abs(stop_pips - 12.0) < 0.2, stop_pips          # 4-pip structure stop widened to the 12-pip floor
    assert abs(plan["rr_ratio"] - 1.5) < 0.01              # policy.target_rr
    assert plan["stop_basis"] == "min_stop_floor"
    assert 0.40 < m["predicted_win_probability"] < 0.60    # calibrated, not the old ~0.55 everywhere
    assert m["decision_mode"] == "v4"


def test_chasing_entry_rejected_as_extended():
    eng = _load("v4")
    res = _eval(eng, _payload(pb=0.95))                    # directional z = 0.9 >= 0.8
    assert res["rejection_reason_code"] == "ENTRY_EXTENDED"


def test_short_extension_is_directional():
    eng = _load("v4")
    # %B 0.05 is "stretched down": extended for a short, cheap for a long.
    short = _eval(eng, _payload(direction="short", pb=0.05, htf="bearish", trend="bearish", rsi=54.0, macd=-0.00001))
    assert short["rejection_reason_code"] == "ENTRY_EXTENDED"
    long_ = _eval(eng, _payload(direction="long", pb=0.05))
    assert long_["rejection_reason_code"] != "ENTRY_EXTENDED"


def test_legacy_trend_pullback_needs_real_edge():
    eng = _load("v4")
    # The v5.2-style trigger: trend aligned, already recovered, MACD up
    # (typical v5.2 candidate: %B ~0.75, RSI mid-50s).
    res = _eval(eng, _payload(model="trend_pullback", pb=0.75, rsi=56.0, macd=0.00008))
    assert res["approval_status"] == "rejected"


def test_legacy_mode_has_no_v4_gates():
    eng = _load("legacy")
    res = _eval(eng, _payload(pb=0.95))
    assert res["rejection_reason_code"] != "ENTRY_EXTENDED"
    plan = res["manus"]["diagnostics"]["trade_plan_preview"]
    assert not plan["stop_floor_applied"]
    assert res["manus"]["decision_mode"] == "legacy"


def test_missing_edge_features_degrade_not_crash():
    eng = _load("v4")
    p = _payload()
    del p["indicators"]["bb_percent_b"], p["extensions"]["atr_ratio"], p["extensions"]["distance_from_vwap_atr"]
    res = _eval(eng, p)
    missing = res["manus"]["diagnostics"]["edge_model"]["missing_features"]
    assert "atr_r" in missing and "vwap_d" in missing
    assert "bbz" in missing  # no %B and no bands to rebuild it from -> neutral default
    assert res["approval_status"] in ("approved", "rejected")


def _for_instrument(p, symbol, close, atr, pip):
    p["instrument"]["symbol"] = symbol
    p["price"]["close"] = close
    p["indicators"]["atr"] = atr
    p["indicators"]["ema_fast"] = close
    p["indicators"]["ema_slow"] = close - atr if p["signal"]["direction"] == "long" else close + atr
    p["structure"]["nearest_support"] = close - 2 * atr
    p["structure"]["nearest_resistance"] = close + 2 * atr
    p["extensions"]["atr_pips"] = atr / pip
    return p


def test_bespoke_stop_floors_per_instrument():
    eng = _load("v4")
    for symbol, close, atr, pip, floor in (("EURUSD", 1.13300, 0.00015, 0.0001, 9.5),
                                          ("USDCAD", 1.42300, 0.00018, 0.0001, 15.0),
                                          ("XAUUSD", 4000.00, 1.50, 0.1, 40.0),
                                          ("XAGUSD", 48.000, 0.040, 0.01, 30.0)):
        res = _eval(eng, _for_instrument(_payload(), symbol, close, atr, pip))
        plan = res["manus"]["diagnostics"]["trade_plan_preview"]
        assert abs(plan["stop_distance_pips"] - floor) < 0.2, (symbol, plan["stop_distance_pips"])


def test_usdjpy_breakout_allowed_to_enter_extended():
    eng = _load("v4")
    p = _for_instrument(_payload(model="trend_breakout_continuation", pb=0.97, rsi=63.0, macd=0.02),
                        "USDJPY", 157.400, 0.030, 0.01)
    res = _eval(eng, p)
    assert res["rejection_reason_code"] != "ENTRY_EXTENDED"
    assert res["approval_status"] == "approved", res["rejection_reason"]
    assert res["manus"]["diagnostics"]["edge_model"]["probability_source"] == "prior"
    # The same extended entry under the pullback model is still vetoed.
    p2 = _for_instrument(_payload(pb=0.97), "USDJPY", 157.400, 0.030, 0.01)
    assert _eval(eng, p2)["rejection_reason_code"] == "ENTRY_EXTENDED"


def test_bespoke_session_windows_not_vetoed_by_context():
    # Pine always sends context_status "unchecked"; each script's own arming
    # window must still clear the CONTEXT_SCORE_TOO_LOW gate (v4.1.1).
    eng = _load("v4")
    jpy = _for_instrument(_payload(model="trend_breakout_continuation", pb=0.97, rsi=63.0, macd=0.02), "USDJPY", 157.400, 0.030, 0.01)
    jpy["market"].update(session_name="asia", hour_utc=5)
    jpy["context"]["context_status"] = "unchecked"
    assert _eval(eng, jpy)["rejection_reason_code"] != "CONTEXT_SCORE_TOO_LOW"
    for symbol, close, atr, pip, session in (("AUDUSD", 0.69450, 0.00012, 0.0001, "new_york"),
                                             ("GBPUSD", 1.33000, 0.00025, 0.0001, "offpeak"),
                                             ("XAUUSD", 4000.00, 1.50, 0.1, "asia")):
        p = _for_instrument(_payload(), symbol, close, atr, pip)
        p["market"]["session_name"] = session
        p["context"]["context_status"] = "unchecked"
        assert _eval(eng, p)["rejection_reason_code"] != "CONTEXT_SCORE_TOO_LOW", symbol


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    _load("v4")
