"""Batch 2 ideas E (vol-managed index holding) and F (gold session drift). See PREREGISTRATION_BATCH2.md.
Usage: DAILY_DIR=<daily9> RATES=<oecd csv> GOLD_M15=<XAU_USD_M15_BA.csv.gz> python3 ideas_batch2.py"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DAILY = os.environ['DAILY_DIR']
IDX = {'SPX500_USD': 'USA', 'NAS100_USD': 'USA', 'US30_USD': 'USA', 'DE30_EUR': 'EA20', 'UK100_GBP': 'GBR',
       'JP225_USD': 'USA', 'AU200_AUD': 'AUS', 'EU50_EUR': 'EA20'}
PER = [('dev', '1900-01-01', '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', '2100-01-01')]


def load(n):
    d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
    t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
    m = pd.DataFrame({'c': (d.bid_c + d.ask_c).values / 2, 'sp': ((d.ask_o - d.bid_o) / ((d.ask_o + d.bid_o) / 2)).values}, index=t)
    m = m[~m.index.duplicated(keep='last')].sort_index()
    return m[m.index.dayofweek < 5]


def rates():
    r = pd.read_csv(os.environ['RATES'])
    r = r.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
    r.index = pd.to_datetime(r.index) + pd.offsets.MonthBegin(1)
    return r / 100.0


def perf(x):
    x = x.dropna()
    eq = (1 + x).cumprod(); dd = float((1 - eq / eq.cummax()).max())
    return dict(ann_ret_pct=round(float(x.mean() * 260 * 100), 2), ann_vol_pct=round(float(x.std() * np.sqrt(260) * 100), 2),
                sharpe=round(float(x.mean() / x.std() * np.sqrt(260)), 2), max_dd_pct=round(dd * 100, 1))


def idea_e():
    R = rates()
    C = pd.DataFrame({n: load(n).c for n in IDX}).sort_index()
    SP = pd.DataFrame({n: load(n).sp for n in IDX}).sort_index()
    ret = C.pct_change()
    cash = pd.DataFrame({n: R[a].reindex(C.index, method='ffill') for n, a in IDX.items()}) / 260.0
    exret = ret - cash                                       # futures-style excess return (dividends ignored: conservative)
    vol = ret.rolling(20, min_periods=15).std() * np.sqrt(260)
    raw = (0.15 / vol).clip(upper=2.0) / np.sqrt(len(IDX))
    mon = pd.Series(C.index.dayofweek == 0, index=C.index)
    W = pd.DataFrame(np.nan, index=C.index, columns=C.columns); w = pd.Series(0.0, index=C.columns)
    for i, d in enumerate(C.index):
        if mon.iloc[i] and i > 0:
            tgt = raw.iloc[i - 1].fillna(0)
            chg = (tgt - w).abs() > 0.25 * tgt.abs()
            w = w.where(~chg, tgt)
        W.iloc[i] = w.values
    Wp = W.shift(1).fillna(0)                                 # weight held over day t
    turn = (W - W.shift(1)).abs().fillna(0)
    cost = (turn * SP * 1.5).sum(axis=1) + (Wp * SP * 1.5 * 4 / 260).sum(axis=1)
    e = (Wp * exret.fillna(0)).sum(axis=1) - cost
    avg = W.replace(0, np.nan).mean()
    bh_w = avg.fillna(0)
    bh = (exret.fillna(0) * bh_w).sum(axis=1) - (bh_w * SP * 1.5 * 4 / 260).sum(axis=1)
    start = W.abs().sum(axis=1).gt(0).idxmax()
    e, bh = e[start:], bh[start:]
    out = {}
    for p, a, b in PER + [('full', '1900-01-01', '2100-01-01')]:
        out[p] = dict(vol_managed=perf(e[a:b]), buy_and_hold=perf(bh[a:b]))
    ok = all(out[p]['vol_managed']['sharpe'] > out[p]['buy_and_hold']['sharpe'] and out[p]['vol_managed']['ann_ret_pct'] > 0
             for p in ('dev', 'val', 'hold')) and out['full']['vol_managed']['max_dd_pct'] < out['full']['buy_and_hold']['max_dd_pct']
    out['verdict'] = 'PASS' if ok else 'FAIL'
    return out


def idea_f():
    d = pd.read_csv(os.environ['GOLD_M15'])
    d.index = pd.to_datetime(d.time_utc, utc=True)
    d = d[~d.index.duplicated()]
    res = {}
    for name, (h0, h1) in {'asia_23_07': (23, 7), 'london_07_13': (7, 13), 'newyork_13_21': (13, 21)}.items():
        for k in (1.0, 3.0):
            rows = []
            ent = d[(d.index.hour == h0) & (d.index.minute == 0)]
            for t, r in ent.iterrows():
                tx = t + pd.Timedelta(hours=(h1 - h0) % 24)
                if tx not in d.index or t.dayofweek == 4 and h0 == 23:   # no Friday-night entries
                    continue
                x = d.loc[tx]
                if isinstance(x, pd.DataFrame):
                    x = x.iloc[0]
                spe, spx = r.ask_o - r.bid_o, x.ask_o - x.bid_o
                pe = r.ask_o + (k - 1) * spe / 2 + 0.1 * spe; px = x.bid_o - (k - 1) * spx / 2 - 0.1 * spx
                mid = (r.ask_o + r.bid_o) / 2
                carry = 0.00006 if h0 == 23 else 0.0
                rows.append(dict(date=t, ret=(px - pe) / mid - carry))
            tr = pd.DataFrame(rows)
            r_ = tr.ret.values
            ts = tr.sort_values('date').reset_index(drop=True); n3 = len(ts) // 3
            q = [ts.iloc[:n3], ts.iloc[n3:2 * n3], ts.iloc[2 * n3:]]
            res[f'{name}_spread{k:g}x'] = dict(n=len(r_), mean_bp=round(float(r_.mean() * 1e4), 2),
                                              t=round(float(r_.mean() / (r_.std() / np.sqrt(len(r_)))), 2),
                                              pf=round(float(r_[r_ > 0].sum() / -r_[r_ < 0].sum()), 3),
                                              thirds_bp=[round(float(x.ret.mean() * 1e4), 2) for x in q])
    a1, a3 = res['asia_23_07_spread1x'], res['asia_23_07_spread3x']
    res['verdict'] = 'PASS' if (min(a1['thirds_bp']) > 0 and a1['t'] >= 2.5 and a1['pf'] >= 1.2 and a3['pf'] >= 1.05) else 'FAIL'
    return res


if __name__ == '__main__':
    E = idea_e()
    print('E vol-managed index holding:', E['verdict'])
    for p in ('dev', 'val', 'hold', 'full'):
        print('  ', p, 'VM', E[p]['vol_managed'], '| B&H', E[p]['buy_and_hold'])
    F = idea_f()
    print('F gold session drift:', F['verdict'])
    for k, v in F.items():
        if k != 'verdict':
            print('  ', k, v)
    json.dump(dict(E=E, F=F), open(os.path.join(HERE, 'results_batch2.json'), 'w'), indent=1, default=str)
