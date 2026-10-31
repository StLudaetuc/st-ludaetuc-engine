"""Weekly rebalance for the IG DEMO simulation trial.

Environment (never logged): IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACCOUNT_ID, OANDA_TOKEN
Settings: START_EQUITY_GBP (trial starting equity, for the circuit breaker)
          SIM_EXECUTE=1 to send orders (default: dry run, plan only)
          SIM_LOG_WEBHOOK (optional URL that receives the JSON run record)
          FORCE_RUN=1 to run on a day other than Monday
Exit code 0 = run completed (including dry run / not-Monday skip); 1 = error (record still emitted)."""
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INSTRUMENTS, REBALANCE_WEEKDAY  # noqa: E402
from ig_client import IGDemoClient  # noqa: E402
from rebalance import contract_spec, choose_expiry, plan_orders  # noqa: E402
from signals import fetch_oanda_daily, latest_targets  # noqa: E402


def candidate_specs(client, cfg):
    specs = []
    for m in client.search(cfg["ig_search"]):
        if m.get("instrumentName") == cfg["ig_name"] and m.get("expiry") not in (None, "-"):
            specs.append(contract_spec(client.market(m["epic"])))
    return specs


def emit(record):
    print(json.dumps(record, default=str), flush=True)
    url = os.environ.get("SIM_LOG_WEBHOOK")
    if url:
        try:
            requests.post(url, json=record, timeout=20)
        except requests.RequestException as e:
            print(f"log webhook failed: {e}", file=sys.stderr)


def main():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    execute = os.environ.get("SIM_EXECUTE") == "1"
    record = {"run_utc": now.isoformat(timespec="seconds"), "mode": "EXECUTE" if execute else "DRY_RUN",
              "venue": "IG_DEMO", "orders": [], "fills": [], "notes": []}
    if now.weekday() != REBALANCE_WEEKDAY and os.environ.get("FORCE_RUN") != "1":
        record["notes"].append("not the rebalance weekday; nothing to do")
        emit(record)
        return 0
    try:
        closes = fetch_oanda_daily([c["oanda"] for c in INSTRUMENTS.values()], os.environ["OANDA_TOKEN"])
        record["signals"] = latest_targets(closes)
        client = IGDemoClient(os.environ["IG_API_KEY"], os.environ["IG_USERNAME"], os.environ["IG_PASSWORD"],
                              os.environ["IG_ACCOUNT_ID"])
        record["account_id"] = client.login()
        acct = client.account()
        equity = float(acct["balance"]["balance"]) + float(acct["balance"]["profitLoss"])
        record["equity_gbp"] = equity
        positions = client.positions()
        record["positions_before"] = [{"epic": p["market"]["epic"], "name": p["market"]["instrumentName"],
                                       "direction": p["position"]["direction"], "size": p["position"]["size"],
                                       "level": p["position"]["level"]} for p in positions]
        specs = {k: choose_expiry(candidate_specs(client, cfg), now) for k, cfg in INSTRUMENTS.items()}
        record["contracts"] = {k: {x: (str(v) if x == "last_rollover" else v) for x, v in s.items()} for k, s in specs.items()}
        orders, notes = plan_orders(record["signals"]["weights"], specs, positions, INSTRUMENTS, equity,
                                    float(os.environ["START_EQUITY_GBP"]), record["signals"]["last_close"])
        record["orders"], record["notes"] = orders, record["notes"] + notes
        if execute:
            for o in sorted(orders, key=lambda o: o["reason"] != "ROLL_CLOSE"):
                spec = specs[o["instrument"]]
                conf = client.deal(o["epic"], o["expiry"], o["direction"], o["size"], o["level"], o["currency"])
                record["fills"].append({"instrument": o["instrument"], "epic": o["epic"], "reason": o["reason"],
                                        "direction": o["direction"], "size": o["size"],
                                        "mid_at_order": spec["mid"], "spread_at_order": spec["spread"],
                                        "dealStatus": conf.get("dealStatus"), "reason_code": conf.get("reason"),
                                        "fill_level": conf.get("level"), "dealId": conf.get("dealId")})
        emit(record)
        return 0
    except Exception as e:  # emit what we have, then fail loudly
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()[-2000:]
        emit(record)
        return 1


if __name__ == "__main__":
    sys.exit(main())
