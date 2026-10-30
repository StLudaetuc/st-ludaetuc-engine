"""SL/TP trend backtest (see PREREGISTRATION.md). Usage: DAILY_DIR=<daily9 dir> python3 sltp.py"""
import os, json
import numpy as np, pandas as pd

DAILY = os.environ['DAILY_DIR']
MK = {'XAU_USD': (0.00020, 6), 'XAG_USD': (0.0020, 4), 'XCU_USD': (0.00061, 4), 'WTICO_USD': (0.00058, 12), 'USB10Y_USD': (0.00037, 4)}
VARIANTS = [(2, 3), (2, 4), (3, 4.5), (3, 6), (3, None)]
PERIODS = [('dev', None, '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', None), ('full', None, None)]


def load(n):
    d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
    t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
    m = pd.DataFrame({k: (d['bid_' + k] + d['ask_' + k]).values / 2 for k in 'ohlc'}, index=t)
    m = m[~m.index.duplicated(keep='last')].sort_index()
    return m[m.index.dayofweek < 5]


def run(n, ks, kt, cost_mult):
    m = load(n); sp, rolls = MK[n]
    c = m.c.values; o = m.o.values; h = m.h.values; l = m.l.values; idx = m.index
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(20).mean().values
    trades = []; pos = 0
    for t in range(253, len(m)):
        if idx[t].dayofweek == 0:   # Monday decision on previous close
            p = t - 1
            score = np.sign(c[p] - c[p - 63]) + np.sign(c[p] - c[p - 126]) + np.sign(c[p] - c[p - 252])
            d = 1 if score > 0 else -1
            if pos != 0 and d != pos['d']:
                trades.append(dict(pos, exit=o[t], xdate=idx[t], why='flip')); pos = 0
            if pos == 0 and np.isfinite(atr[p]):
                e = o[t]; dist = ks * atr[p]
                pos = dict(d=d, entry=e, date=idx[t], i=t, dist=dist, stop=e - d * dist,
                           tp=(e + d * kt * atr[p]) if kt else None)
        if pos != 0:
            d = pos['d']; stop, tp = pos['stop'], pos['tp']
            gap_open = o[t] if t > pos['i'] else None
            hit = None
            if gap_open is not None and (d * (gap_open - stop) <= 0):
                hit = ('stop', gap_open)
            elif gap_open is not None and tp is not None and d * (gap_open - tp) >= 0:
                hit = ('tp', gap_open)
            elif (d > 0 and l[t] <= stop) or (d < 0 and h[t] >= stop):
                hit = ('stop', stop)
            elif tp is not None and ((d > 0 and h[t] >= tp) or (d < 0 and l[t] <= tp)):
                hit = ('tp', tp)
            if hit:
                trades.append(dict(pos, exit=hit[1], xdate=idx[t], why=hit[0])); pos = 0
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    days = (df.xdate - df.date).dt.days.clip(lower=1)
    cost_px = cost_mult * sp * df.entry * (1 + rolls * days / 365.0)
    df['R'] = (df.d * (df.exit - df.entry) - cost_px) / df.dist
    df['mkt'] = n
    return df


def stats(df):
    if df.empty: return dict(n=0)
    r = df.R.values
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float('inf')
    return dict(n=int(len(r)), avgR=round(float(r.mean()), 3), pf=round(float(pf), 2), win=round(float((r > 0).mean()), 3),
                tp_rate=round(float((df.why == 'tp').mean()), 3), stop_rate=round(float((df.why == 'stop').mean()), 3),
                hold_days=round(float((df.xdate - df.date).dt.days.mean()), 1))


def equity(df):
    """1% risk per trade, trades applied in exit-date order; returns CAGR and max drawdown on a 10,000 start."""
    e = 10000.0; peak = e; mdd = 0
    for r in df.sort_values('xdate').R:
        e *= 1 + 0.01 * r; peak = max(peak, e); mdd = max(mdd, 1 - e / peak)
    yrs = (df.xdate.max() - df.date.min()).days / 365.25
    return dict(end_equity=round(e), cagr_pct=round(((e / 10000) ** (1 / yrs) - 1) * 100, 2), max_dd_pct=round(mdd * 100, 1))


out = {}
for ks, kt in VARIANTS:
    key = f'stop {ks}ATR / target {kt}ATR' if kt else f'stop {ks}ATR / no target'
    all1 = pd.concat([run(n, ks, kt, 1.0) for n in MK]); all2 = pd.concat([run(n, ks, kt, 2.0) for n in MK])
    res = {}
    for lab, a, b in PERIODS:
        sel = all1[(all1.date >= pd.Timestamp(a or '1900')) & (all1.date <= pd.Timestamp(b or '2100'))]
        res[lab] = stats(sel)
    res['full_2x'] = stats(all2)
    res['per_market_full'] = {n: stats(all1[all1.mkt == n]) for n in MK}
    res['equity_1pct_risk'] = equity(all1)
    ok = all(res[p]['pf'] >= 1.15 and res[p]['avgR'] > 0 for p in ('dev', 'val', 'hold')) and res['full_2x']['pf'] >= 1.05
    res['verdict'] = 'PASS' if ok else 'FAIL'
    out[key] = res
    print(key, res['verdict'], {p: (res[p]['n'], res[p]['pf'], res[p]['avgR'], res[p]['win']) for p in ('dev', 'val', 'hold', 'full')},
          'x2', res['full_2x']['pf'], res['equity_1pct_risk'], 'tp/stop', res['full']['tp_rate'], res['full']['stop_rate'], 'hold', res['full']['hold_days'])
json.dump(out, open('results.json', 'w'), indent=1, default=str)
