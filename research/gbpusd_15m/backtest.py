"""Backtest harness for the GBP/USD 15m liquidity-sweep research strategy.

    python3 backtest.py              # development + validation only
    python3 backtest.py --holdout    # ALSO the final 20% holdout - run ONCE, at the end

Splits are chronological by bar: first 60% development, next 20% validation,
final 20% holdout. The strategy runs once over the whole series (state is
continuous) and each trade is assigned to the split its ENTRY falls in.
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz")


def metrics(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t.r.to_numpy()
    eq = np.cumsum(r); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0; best = max(best, streak)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": int(len(r)), "win_rate": round(float((r > 0).mean()), 3), "avg_r": round(float(r.mean()), 3),
            "total_r": round(float(r.sum()), 1), "profit_factor": round(float(gp / gl), 2) if gl > 0 else None,
            "max_dd_r": round(float(dd), 1), "longest_losing_streak": int(best)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--holdout", action="store_true"); a = ap.parse_args()
    sha = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
    df = S.load(DATA)
    n = len(df); cut1, cut2 = df.index[int(n * 0.6)], df.index[int(n * 0.8)]
    splits = {"development": (df.index[0], cut1), "validation": (cut1, cut2), "holdout": (cut2, df.index[-1] + pd.Timedelta("1s"))}
    out = {"data_sha256": sha, "rows": n, "splits": {k: [str(v[0]), str(v[1])] for k, v in splits.items()}, "results": {}}
    runs = [(f"{m}x", S.Params(cost_mult=m)) for m in (1.0, 1.5, 2.0)] + [("limit_mid_1.0x_SENSITIVITY", S.Params(entry_mode="limit_mid"))]
    for tag, prm in runs:
        mult = tag
        t = S.to_frame(S.run(df, prm))
        t["entry_dt"] = pd.to_datetime(t.entry_time)
        for name, (s0, s1) in splits.items():
            if name == "holdout" and not a.holdout:
                continue
            sub = t[(t.entry_dt >= s0) & (t.entry_dt < s1)]
            res = {"all": metrics(sub),
                   "by_session": {k: metrics(g) for k, g in sub.groupby("session")},
                   "by_direction": {("long" if k == 1 else "short"): metrics(g) for k, g in sub.groupby("direction")},
                   "by_level": {k: metrics(g) for k, g in sub.groupby("level")},
                   "monthly_r": {str(k): round(float(v), 2) for k, v in sub.groupby(sub.entry_dt.dt.strftime("%Y-%m")).r.sum().items()}}
            out["results"][f"{name}@{mult}"] = res
        if mult == "1.0x":
            keep = t if a.holdout else t[t.entry_dt < cut2]
            keep.drop(columns="entry_dt").to_csv(os.path.join(HERE, "results", "trades_costs_1x.csv"), index=False)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fn = "results_with_holdout.json" if a.holdout else "results_dev_val.json"
    json.dump(out, open(os.path.join(HERE, "results", fn), "w"), indent=1)
    for k, v in out["results"].items():
        print(k, v["all"])


if __name__ == "__main__":
    main()
