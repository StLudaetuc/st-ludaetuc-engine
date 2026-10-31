"""Turn target weights into IG demo orders. Pure functions (no I/O) so they can be unit-tested."""
import math
import re
from datetime import datetime, timedelta

from config import (BUFFER, CIRCUIT_BREAKER, LIMIT_SLIPPAGE_SPREADS, MAX_GROSS_LEVERAGE, PRICE_SANITY_TOL,
                    ROLL_DAYS_BEFORE)


def point_value_usd(one_pip_means: str) -> float:
    """USD price change represented by one IG point, from instrument.onePipMeans."""
    t = one_pip_means.lower().strip()
    m = re.match(r"^([\d.]+)\s*cents?\b", t)
    if m:
        return float(m.group(1)) / 100.0
    if t.startswith("1 $") or t.startswith("1 dollar"):
        return 1.0
    raise ValueError(f"unrecognised onePipMeans: {one_pip_means!r}")


def contract_spec(details: dict) -> dict:
    ins, snap, rules = details["instrument"], details["snapshot"], details["dealingRules"]
    bid, offer = float(snap["bid"]), float(snap["offer"])
    mid = (bid + offer) / 2
    vop = float(ins["valueOfOnePip"])          # USD per point per 1 contract
    return {
        "epic": ins["epic"], "expiry": ins["expiry"], "name": ins["name"],
        "currency": ins["currencies"][0]["code"], "usd_per_gbp": float(ins["currencies"][0]["baseExchangeRate"]),
        "bid": bid, "offer": offer, "mid": mid, "spread": offer - bid,
        "usd_per_point": vop, "price_usd": mid * point_value_usd(ins["onePipMeans"]),
        "notional_usd_per_contract": mid * vop, "min_size": float(rules["minDealSize"]["value"]),
        "last_rollover": datetime.fromisoformat(ins["rolloverDetails"]["lastRolloverTime"]),
        "status": snap["marketStatus"],
    }


def choose_expiry(candidates, now: datetime):
    """Earliest expiry whose last rollover time is more than ROLL_DAYS_BEFORE days away."""
    ok = [c for c in candidates if c["last_rollover"] - timedelta(days=ROLL_DAYS_BEFORE) > now]
    if not ok:
        raise ValueError("no eligible expiry")
    return min(ok, key=lambda c: c["last_rollover"])


def held_by_epic(positions, ig_name):
    """Signed open size per epic for one instrument (positions as returned by IG GET /positions v2)."""
    out = {}
    for p in positions:
        if p["market"]["instrumentName"] != ig_name:
            continue
        sz = float(p["position"]["size"]) * (1 if p["position"]["direction"] == "BUY" else -1)
        out[p["market"]["epic"]] = out.get(p["market"]["epic"], 0.0) + sz
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


def target_contracts(weight, equity_usd, spec):
    if weight is None or (isinstance(weight, float) and math.isnan(weight)):
        return 0.0
    raw = weight * equity_usd / spec["notional_usd_per_contract"]
    sized = round(raw, 2)
    return 0.0 if abs(sized) < spec["min_size"] else sized


def needs_trade(target, current):
    """Same 25% buffer rule as the backtest."""
    return abs(target - current) > BUFFER * abs(target) + 1e-12


def limit_level(spec, direction):
    pad = LIMIT_SLIPPAGE_SPREADS * spec["spread"]
    return round(spec["offer"] + pad, 2) if direction == "BUY" else round(spec["bid"] - pad, 2)


def plan_orders(targets, specs, positions, instruments, equity_gbp, start_equity_gbp, oanda_last_close):
    """Return (orders, notes). `targets` = weights by OANDA symbol; `specs` = chosen contract spec per instrument key."""
    notes, orders = [], []
    usd_per_gbp = next(iter(specs.values()))["usd_per_gbp"]
    equity_usd = equity_gbp * usd_per_gbp
    halted = equity_gbp < CIRCUIT_BREAKER * start_equity_gbp
    if halted:
        notes.append(f"CIRCUIT_BREAKER: equity {equity_gbp:.2f} < {CIRCUIT_BREAKER:.0%} of start {start_equity_gbp:.2f}; flattening")
    gross = 0.0
    for key, cfg in instruments.items():
        spec = specs[key]
        px_o = oanda_last_close.get(cfg["oanda"])
        if px_o and abs(spec["price_usd"] / px_o - 1) > PRICE_SANITY_TOL:
            raise ValueError(f"{key}: IG price {spec['price_usd']:.4f} vs OANDA {px_o:.4f} differs > {PRICE_SANITY_TOL:.0%}; wrong contract?")
        if spec["status"] != "TRADEABLE":
            notes.append(f"{key}: market {spec['status']}, skipped")
            continue
        T = 0.0 if halted else target_contracts(targets.get(cfg["oanda"]), equity_usd, spec)
        held = held_by_epic(positions, cfg["ig_name"])
        # roll: anything held on a different epic of the same instrument is closed and re-opened on the chosen expiry
        cur = held.pop(spec["epic"], 0.0)
        old_total = sum(held.values())
        for old_epic, sz in held.items():
            old = next(p for p in positions if p["market"]["epic"] == old_epic)["market"]
            orders.append({"instrument": key, "epic": old_epic, "expiry": old["expiry"], "direction": "SELL" if sz > 0 else "BUY",
                           "size": round(abs(sz), 2), "reason": "ROLL_CLOSE", "currency": spec["currency"],
                           "level": round(float(old["bid"]) - LIMIT_SLIPPAGE_SPREADS * spec["spread"], 2) if sz > 0
                           else round(float(old["offer"]) + LIMIT_SLIPPAGE_SPREADS * spec["spread"], 2)})
            notes.append(f"{key}: rolling {sz:+.2f} from {old_epic} to {spec['epic']}")
        if held and T != 0.0 and not needs_trade(T, cur + old_total):
            T = round(cur + old_total, 2)   # within the buffer: keep the existing exposure, just move it
        force = bool(held) and T != cur     # after a roll close, re-open even inside the buffer
        gross += abs(T) * spec["notional_usd_per_contract"]
        if force or needs_trade(T, cur):
            d = T - cur
            direction = "BUY" if d > 0 else "SELL"
            orders.append({"instrument": key, "epic": spec["epic"], "expiry": spec["expiry"], "direction": direction,
                           "size": round(abs(d), 2), "reason": "REBALANCE", "target": T, "current": cur,
                           "level": limit_level(spec, direction), "currency": spec["currency"]})
        notes.append(f"{key}: weight={targets.get(cfg['oanda'])} target={T:+.2f} current={cur:+.2f} "
                     f"contract_notional=${spec['notional_usd_per_contract']:,.0f} min={spec['min_size']}")
    if gross > MAX_GROSS_LEVERAGE * equity_usd:
        raise ValueError(f"gross notional ${gross:,.0f} > {MAX_GROSS_LEVERAGE}x equity ${equity_usd:,.0f}; refusing")
    return orders, notes
