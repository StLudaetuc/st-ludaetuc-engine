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


def test_inverted_payload_stop_or_target_never_approved_any_mode():
    for mode in ("v4", "legacy"):
        eng = _load(mode)
        for direction, sl, tp in (("long", 1.3330, 1.3350),   # stop above entry
                                  ("long", 1.3280, 1.3290),   # target below entry
                                  ("short", 1.3270, 1.3250),  # stop below entry
                                  ("short", 1.3330, 1.3310),  # target above entry
                                  ("long", 1.3300, 1.3350)):  # stop == entry
            p = _payload(direction=direction, htf="bullish" if direction == "long" else "bearish")
            p["extensions"]["pine_proposal_only"] = False
            p["risk"].update(proposed_entry=1.3300, proposed_stop_loss=sl, proposed_take_profit=tp)
            r = _eval(eng, p)
            assert r["approval_status"] == "rejected" and r["rejection_reason_code"] == "INVALID_TRADE_PLAN", (mode, direction, sl, tp, r.get("rejection_reason_code"))
    _load("v4")


def test_engine_computed_plan_sides_are_correct_both_directions():
    eng = _load("v4")
    for direction in ("long", "short"):
        p = _payload(direction=direction, htf="bullish" if direction == "long" else "bearish",
                     trend="bullish" if direction == "long" else "bearish", pb=0.31 if direction == "long" else 0.69,
                     rsi=45.0 if direction == "long" else 55.0, macd=-0.00001 if direction == "long" else 0.00001)
        plan = _eval(eng, p)["manus"]["diagnostics"]["trade_plan_preview"]
        if direction == "long":
            assert plan["sl"] < plan["entry"] < plan["tp"] and plan["units"] > 0
        else:
            assert plan["tp"] < plan["entry"] < plan["sl"] and plan["units"] < 0


def test_position_below_minimum_is_refused_not_upsized():
    eng = _load("v4")
    p = _for_instrument(_payload(), "XAUUSD", 4000.00, 1.50, 0.1)
    p["structure"]["nearest_support"] = 3950.00          # $50 structure stop
    p["risk"]["risk_percent"] = 0.35
    r = eng.engine.evaluate({"payload": p, "account": {"balance": 5000}})   # 17.50 risk < 1 oz x $50
    assert r["rejection_reason_code"] == "INVALID_TRADE_PLAN"
    assert "minimum" in r["rejection_reason"]


def test_v6_models_not_vetoed_by_legacy_extreme_band_but_legacy_models_are():
    eng = _load("v4")
    p = _payload(); p["structure"]["volatility_regime"] = "extreme"
    assert _eval(eng, p)["rejection_reason_code"] != "EXTREME_VOLATILITY"
    p2 = _payload(model="trend_pullback"); p2["structure"]["volatility_regime"] = "extreme"
    assert _eval(eng, p2)["rejection_reason_code"] == "EXTREME_VOLATILITY"
    eng = _load("legacy")
    assert _eval(eng, p)["rejection_reason_code"] == "EXTREME_VOLATILITY"
    _load("v4")


def _max_loss_gbp(plan, gbpusd, quote_to_usd=1.0):
    return abs(plan["units"]) * plan["stop_distance"] / quote_to_usd / gbpusd


def test_gbp_account_trades_risk_full_percent():
    # v4.2.0: a 0.5% trade on a GBP 10,000 balance must lose ~GBP 50 at the
    # stop (never more), not the ~GBP 37.6 the USD-as-GBP sizing gave.
    os.environ.pop("ACCOUNT_CURRENCY", None)
    eng = _load("v4")
    eng._latest_gbpusd_rate = lambda: (1.33000, None)
    gbp = _eval(eng, _payload())["manus"]["diagnostics"]["trade_plan_preview"]
    assert gbp["usd_rate_source"] == "signal_entry"
    assert 49.0 < _max_loss_gbp(gbp, 1.33) <= 50.0, _max_loss_gbp(gbp, 1.33)
    for symbol, close, atr, pip, quote_to_usd in (("EURUSD", 1.13300, 0.00015, 0.0001, 1.0),
                                                 ("USDJPY", 157.400, 0.030, 0.01, 157.400),
                                                 ("USDCAD", 1.42300, 0.00018, 0.0001, 1.42300),
                                                 ("XAGUSD", 48.000, 0.040, 0.01, 1.0)):
        plan = _eval(eng, _for_instrument(_payload(), symbol, close, atr, pip))["manus"]["diagnostics"]["trade_plan_preview"]
        assert plan["usd_rate_source"] == "market_data_gbpusd", symbol
        loss = _max_loss_gbp(plan, 1.33, quote_to_usd)
        budget = plan["risk_cash"]                         # metals are capped at 0.35%
        assert 0.94 * budget < loss <= budget + 1e-9, (symbol, loss, budget)
        assert abs(plan["max_loss_account_ccy"] - loss) < 1e-6


def test_gbp_sizing_falls_back_small_without_rate_and_usd_rollback():
    eng = _load("v4")
    eng._latest_gbpusd_rate = lambda: (None, None)
    p = _for_instrument(_payload(), "EURUSD", 1.13300, 0.00015, 0.0001)
    plan = _eval(eng, p)["manus"]["diagnostics"]["trade_plan_preview"]
    assert plan["usd_rate_source"] == "fallback_no_gbpusd_rate" and plan["usd_per_account_ccy"] == 1.0
    os.environ["ACCOUNT_CURRENCY"] = "USD"
    try:
        eng = _load("v4")
        plan_usd = _eval(eng, p)["manus"]["diagnostics"]["trade_plan_preview"]
        assert plan_usd["usd_rate_source"] == "account_ccy_usd" and plan_usd["units"] == plan["units"]
    finally:
        os.environ.pop("ACCOUNT_CURRENCY", None)
        _load("v4")


def _natgas(direction="short", timeframe="240", close=2.736, atr=0.045, open_trades=0):
    sign = 1.0 if direction == "long" else -1.0
    stop = close - sign * 0.130
    target = close + sign * 0.195
    p = {
        "meta": {"signal_id": f"STLNG-{timeframe}-1-{direction}"},
        "system": {"strategy_id": "stl_natgas_anytf_bb_absorption_v1", "strategy_version": "1.1.0"},
        "instrument": {"symbol": "NATGASUSD"},
        "market": {"timeframe": timeframe, "bar_time_utc": "2026-10-27T12:00:00Z", "hour_utc": 12, "session_name": "overlap_london_new_york"},
        "signal": {"signal_type": "entry", "direction": direction, "entry_model": "natgas_bb_absorption"},
        "price": {"close": close, "bid": close - 0.004, "ask": close + 0.004, "spread_price": 0.008},
        "indicators": {"atr": atr, "bb_percent_b": 0.80 if direction == "short" else 0.35},
        "structure": {"volatility_regime": "normal", "market_regime": "unclassified"},
        "risk": {"risk_percent": 0.35, "proposed_entry": close, "proposed_stop_loss": stop, "proposed_take_profit": target},
        "context": {"context_status": "unchecked"},
        "extensions": {"pine_proposal_only": False, "allow_manus_override_sl_tp": False},
        "costs": {"financing_long_annual": -1.442, "financing_short_annual": 1.392},
    }
    return {"payload": p, "authoritative_instrument": "NATGASUSD",
            "account": {"balance": 4649.29, "currency": "GBP", "instrument_open_trade_count": open_trades}}


def test_natgas_h4_short_approved_with_pine_stop_and_whole_units():
    eng = _load("v4")
    eng._latest_gbpusd_rate = lambda: (1.33, None)
    r = eng.engine.evaluate(_natgas())
    assert r["approval_status"] == "approved", (r["rejection_reason_code"], r["rejection_reason"])
    m = r["manus"]
    assert m["final_trade_plan"]["broker_symbol"] == "NATGAS_USD"
    assert m["final_stop_loss"] > m["final_entry"] > m["final_take_profit"]
    assert abs(m["final_stop_loss"] - 2.866) < 1e-9                 # Pine's own stop, not recomputed
    assert m["final_position_size"] < 0 and m["final_position_size"] == int(m["final_position_size"])
    assert m["diagnostics"]["carry"]["applied"] and m["diagnostics"]["carry"]["carry_r"] > 0
    assert m["final_trade_plan"]["max_loss_account_ccy"] <= 4649.29 * 0.0035 + 1e-9


def test_natgas_longs_and_short_timeframes_rejected_on_cost_and_carry():
    eng = _load("v4")
    assert eng.engine.evaluate(_natgas("long"))["rejection_reason_code"] == "INSUFFICIENT_POST_GATE_EDGE"
    m15 = _natgas("short", "15", atr=0.010)
    m15["payload"]["risk"].update(proposed_stop_loss=2.761, proposed_take_profit=2.6985)
    assert eng.engine.evaluate(m15)["rejection_reason_code"] == "COST_R_EXCESSIVE"


def test_natgas_one_position_at_a_time():
    eng = _load("v4")
    assert eng.engine.evaluate(_natgas(open_trades=1))["rejection_reason_code"] == "POSITION_ALREADY_OPEN"


def test_carry_never_touches_fx_or_metals():
    eng = _load("v4")
    for p in (_payload(), _for_instrument(_payload(), "XAUUSD", 4000.00, 1.50, 0.1)):
        carry = _eval(eng, p)["manus"]["diagnostics"]["carry"]
        assert carry == {"carry_r": 0.0, "applied": False}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    _load("v4")
