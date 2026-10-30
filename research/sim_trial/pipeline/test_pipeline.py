"""Tests for the IG demo simulation pipeline. Run: python3 -m pytest research/sim_trial/pipeline -q"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import INSTRUMENTS  # noqa: E402
from ig_client import IGDemoClient, IGError  # noqa: E402
from rebalance import (choose_expiry, contract_spec, needs_trade, plan_orders, point_value_usd,  # noqa: E402
                       target_contracts)
from signals import target_weights  # noqa: E402

DETAILS = json.load(open(os.path.join(HERE, "fixtures", "ig_market_details_20261030.json")))
NOW = datetime(2026, 10, 30, 17, 0)
SPECS = {k: contract_spec(DETAILS[e]) for k, e in {
    "GOLD": "MT.D.GC.FWM3.IP", "SILVER": "MT.D.SI.FWM1.IP", "COPPER": "MT.D.HG.FWM2.IP",
    "CRUDE": "EN.D.CL.FWM5.IP", "TNOTE": "IR.D.10YEAR100.FWM2.IP"}.items()}
OANDA_PX = {"XAU_USD": 3905.0, "XAG_USD": 49.18, "XCU_USD": 6.48, "WTICO_USD": 104.6, "USB10Y_USD": 106.6}


def test_point_values():
    assert point_value_usd("1 $ per troy ounce") == 1.0
    assert point_value_usd("1 cent per troy ounce") == 0.01
    assert point_value_usd("0.01 cents per pound") == pytest.approx(0.0001)
    assert point_value_usd("1 cent") == 0.01
    with pytest.raises(ValueError):
        point_value_usd("1 pip")


def test_contract_spec_matches_ig_contract_sizes():
    # 33.2 oz gold, 1000 oz silver, 5000 lb copper, 500 bbl crude, $200/point T-note
    assert SPECS["GOLD"]["notional_usd_per_contract"] == pytest.approx(3905.0 * 33.2)
    assert SPECS["SILVER"]["price_usd"] == pytest.approx(49.18, rel=1e-3)
    assert SPECS["COPPER"]["price_usd"] == pytest.approx(6.483, rel=1e-3)
    assert SPECS["CRUDE"]["notional_usd_per_contract"] == pytest.approx(10460.5 * 5)
    assert SPECS["TNOTE"]["price_usd"] == pytest.approx(106.6, rel=1e-3)


def test_choose_expiry_rolls_before_last_rollover():
    gold = [contract_spec(DETAILS["MT.D.GC.FWM3.IP"]), contract_spec(DETAILS["MT.D.GC.FWM2.IP"])]
    assert choose_expiry(gold, NOW)["expiry"] == "DEC-26"
    assert choose_expiry(gold, datetime(2026, 11, 20))["expiry"] == "FEB-27"   # within 5 days of 24 Nov


def test_min_size_and_rounding():
    s = SPECS["GOLD"]                     # min 0.5 contracts = ~$65k notional
    assert target_contracts(0.2, 13_400, s) == 0.0
    assert target_contracts(0.2, 1_000_000, s) == pytest.approx(round(0.2 * 1e6 / s["notional_usd_per_contract"], 2))
    assert target_contracts(float("nan"), 1e6, s) == 0.0


def test_buffer_rule():
    assert not needs_trade(1.0, 0.8)      # 20% away: inside the 25% buffer
    assert needs_trade(1.0, 0.7)
    assert needs_trade(0.0, 0.3)          # target flat: always close
    assert not needs_trade(0.0, 0.0)


def _pos(epic, name, direction, size, bid=3900.0, offer=3901.0, expiry="DEC-26"):
    return {"position": {"direction": direction, "size": size, "level": 3900.0},
            "market": {"epic": epic, "instrumentName": name, "expiry": expiry, "bid": bid, "offer": offer}}


def test_plan_rebalance_and_roll():
    big = 400_000.0   # GBP; large enough for whole contracts
    w = {"XAU_USD": 0.3, "XAG_USD": -0.1, "XCU_USD": 0.0, "WTICO_USD": 0.2, "USB10Y_USD": 0.5}
    feb = dict(SPECS); feb["GOLD"] = contract_spec(DETAILS["MT.D.GC.FWM2.IP"])
    held = [_pos("MT.D.GC.FWM3.IP", "Gold ($33.20)", "BUY", 1.2)]
    orders, notes = plan_orders(w, feb, held, INSTRUMENTS, big, big, {**OANDA_PX, "XAU_USD": 3937.0})
    gold = [o for o in orders if o["instrument"] == "GOLD"]
    assert gold[0]["reason"] == "ROLL_CLOSE" and gold[0]["direction"] == "SELL" and gold[0]["size"] == 1.2
    assert gold[1]["epic"] == "MT.D.GC.FWM2.IP" and gold[1]["direction"] == "BUY"
    assert not [o for o in orders if o["instrument"] == "COPPER"]            # zero target, nothing held
    assert [o for o in orders if o["instrument"] == "SILVER"][0]["direction"] == "SELL"


def test_circuit_breaker_flattens():
    held = [_pos("MT.D.GC.FWM3.IP", "Gold ($33.20)", "BUY", 1.0)]
    orders, notes = plan_orders({"XAU_USD": 0.3}, SPECS, held, INSTRUMENTS, 80_000, 100_000, OANDA_PX)
    assert any("CIRCUIT_BREAKER" in n for n in notes)
    assert orders == [o for o in orders if o["direction"] == "SELL"] and orders[0]["size"] == 1.0


def test_price_sanity_blocks_wrong_contract():
    with pytest.raises(ValueError, match="wrong contract"):
        plan_orders({}, SPECS, [], INSTRUMENTS, 1e5, 1e5, {**OANDA_PX, "XAG_USD": 4918.0})


def test_leverage_cap():
    w = {k: 3.0 for k in OANDA_PX}
    with pytest.raises(ValueError, match="refusing"):
        plan_orders(w, SPECS, [], INSTRUMENTS, 400_000, 400_000, OANDA_PX)


def test_client_refuses_live_url():
    with pytest.raises(IGError):
        IGDemoClient("k", "u", "p", "Z5QIAR", base="https://api.ig.com/gateway/deal")


DAILY = os.environ.get("DAILY_DIR", "/tmp/claude-0/-home-user/46736265-72c4-5060-89a6-fa160e5a55e4/scratchpad/night/daily9")


@pytest.mark.skipif(not os.path.isdir(DAILY), reason="historical daily data not available")
def test_parity_with_backtest_positions():
    """Live signal + buffer logic reproduces the backtest's weekly position path exactly."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "daily_portfolio"))
    os.environ["DAILY_DIR"] = DAILY
    import importlib
    import dp
    importlib.reload(dp)
    names = [c["oanda"] for c in INSTRUMENTS.values()]
    O, C, S = dp.load_panel(names)
    tgt = target_weights(C).shift(1)
    W = np.zeros(len(names)); path = []
    for t, day in enumerate(C.index):
        if t > 0 and day.dayofweek == 0:
            T = np.nan_to_num(tgt.iloc[t].values)
            ok = np.isfinite(O.iloc[t].values)
            chg = np.array([needs_trade(a, b) for a, b in zip(T, W)]) & ok
            W = np.where(ok, np.where(chg, T, W), 0.0)
        path.append(W.copy())
    mine = pd.DataFrame(path, index=C.index, columns=names)
    rates = None
    kind = {n: "CFD" for n in names}
    rates = dp.load_rates() if os.path.exists(os.path.join(os.path.dirname(dp.__file__), "data", "oecd_ir3m.csv.gz")) else None
    if rates is None:
        pytest.skip("rates file not present for dp.run")
    _, pos, _ = dp.run(names, kind, dp.tsmom, rates=rates)
    np.testing.assert_allclose(mine.values[1:], pos.values[1:], atol=1e-9)
