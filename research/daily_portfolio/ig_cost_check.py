"""Diagnostic (not pre-registered): the 5-instrument sim-trial trend system under (a) OANDA CFD costs and
(b) IG futures-style costs measured live on the IG demo API, 30 Oct 2026 (spread % of price, roll cadence).
IG model: no financing markup (futures-style; only the pure rate, which futures prices embed), plus one full
spread per roll on the held notional, plus the measured spread on every rebalance."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp, numpy as np, pandas as pd
rates = dp.load_rates()
names = ['XAU_USD', 'XAG_USD', 'XCU_USD', 'WTICO_USD', 'USB10Y_USD']
kind = {n: 'CFD' for n in names}
IG = {'XAU_USD': (0.00020, 6), 'XAG_USD': (0.0020, 4), 'XCU_USD': (0.00061, 4), 'WTICO_USD': (0.00058, 12), 'USB10Y_USD': (0.00037, 4)}
P = [('dev', None, '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', None), ('full', None, None)]
out = {}
df, pos, ip = dp.run(names, kind, dp.tsmom, cost_mult=1.0, rates=rates)
out['OANDA CFD'] = {p: round(dp.metrics(df, pos, ip, a, b)['sharpe'], 2) for p, a, b in P}
out['OANDA CFD']['ann_ret_%'] = round(float(df.pnl.mean() * 252 * 100), 2)
df0, pos0, ip0 = dp.run(names, kind, dp.tsmom, cost_mult=0.0, rates=rates)  # cost_mult=0 removes spread and the markup (run() adds it back), leaving the pure rate
sp = np.array([IG[n][0] for n in names]); rolls = np.array([IG[n][1] for n in names])
W = pos0.abs().values
reb = np.vstack([np.zeros(len(names)), np.abs(np.diff(pos0.values, axis=0))])
cost = (reb * sp / 2 * 1.1).sum(axis=1) + (W * sp * rolls / 252).sum(axis=1)
df1 = df0.copy(); df1['pnl'] = df0.pnl - cost
out['IG futures-style'] = {p: round(dp.metrics(df1, pos0, ip0, a, b)['sharpe'], 2) for p, a, b in P}
out['IG futures-style']['ann_ret_%'] = round(float(df1.pnl.mean() * 252 * 100), 2)
out['IG futures-style']['ann_cost_%'] = round(float(cost.mean() * 252 * 100), 2)
out['frictionless'] = {p: round(dp.metrics(df0, pos0, ip0, a, b)['sharpe'], 2) for p, a, b in P}
for k, v in out.items(): print(k, v)
json.dump(out, open('results/ig_cost_check.json', 'w'), indent=1)
