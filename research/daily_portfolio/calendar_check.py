"""Oracle H8 check (diagnostic): drop Saturday/Sunday-labelled daily rows so each row is one weekday.
Returns are close-to-close, so no price move is lost; a Sunday row's move is merged into the next weekday's return.
Re-runs the 5-market trend under OANDA costs, IG futures-style costs and no broker costs."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp, numpy as np
_lp = dp.load_panel
def load_weekdays(names):
    O, C, S = _lp(names)
    keep = O.index.dayofweek < 5
    return O[keep], C[keep], S[keep]
names = ['XAU_USD', 'XAG_USD', 'XCU_USD', 'WTICO_USD', 'USB10Y_USD']
kind = {n: 'CFD' for n in names}
rates = dp.load_rates()
IG = {'XAU_USD': (0.00020, 6), 'XAG_USD': (0.0020, 4), 'XCU_USD': (0.00061, 4), 'WTICO_USD': (0.00058, 12), 'USB10Y_USD': (0.00037, 4)}
P = [('dev', None, '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', None), ('full', None, None)]
out = {}
for label, loader in (('original rows', _lp), ('weekdays only', load_weekdays)):
    dp.load_panel = loader
    O, C, S = dp.load_panel(names)
    rows_per_year = float(C.groupby(C.index.year).size().loc[2010:2025].mean())
    df, pos, ip = dp.run(names, kind, dp.tsmom, cost_mult=1.0, rates=rates)
    df0, pos0, ip0 = dp.run(names, kind, dp.tsmom, cost_mult=0.0, rates=rates)
    sp = np.array([IG[n][0] for n in names]); rolls = np.array([IG[n][1] for n in names])
    reb = np.vstack([np.zeros(len(names)), np.abs(np.diff(pos0.values, axis=0))])
    cost = (reb * sp / 2 * 1.1).sum(axis=1) + (pos0.abs().values * sp * rolls / rows_per_year).sum(axis=1)
    df1 = df0.copy(); df1['pnl'] = df0.pnl - cost
    def sh(d, a, b):
        r = d.loc[a:b].pnl; return round(float(r.mean() / r.std() * np.sqrt(rows_per_year)), 2)
    res = {}
    for nm, d in (('OANDA', df), ('IG', df1), ('no broker costs', df0)):
        res[nm] = {p: sh(d, a, b) for p, a, b in P}
        res[nm]['ann_ret_%'] = round(float(d.pnl.mean() * rows_per_year * 100), 2)
    res['rows_per_year'] = round(rows_per_year, 1)
    out[label] = res
    print(label, json.dumps(res))
json.dump(out, open('results/calendar_check.json', 'w'), indent=1)
