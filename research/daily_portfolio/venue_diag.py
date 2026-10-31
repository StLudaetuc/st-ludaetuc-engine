"""Venue diagnostic (not a pre-registered test): daily trend with OANDA spreads kept but the CFD financing
markup removed (long pays the pure rate, short receives it) - roughly the excess return a futures venue would
deliver. Not tradable on OANDA."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp, pandas as pd
rates = dp.load_rates(); U = dp.universe()
kind = {**{n: 'FX' for n in U['FX']}, **{n: 'CFD' for g in ('METALS', 'INDICES', 'BONDS', 'COMMOD') for n in U[g]}}
_orig = dp.fin_rates
def fin_nomarkup(names, idx, rates, kind):
    L, S = _orig(names, idx, rates, kind)
    for n in names:
        m = 0.01 if kind[n] == 'FX' else 0.025
        L[n] += m; S[n] += m
    return L, S
dp.fin_rates = fin_nomarkup
P = [('dev', None, '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', None), ('full', None, None)]
out = {}
for lab, names in (('TREND_METALS', U['METALS']), ('TREND_COMMOD', U['COMMOD']), ('TREND_METALS+COMMOD', U['METALS'] + U['COMMOD']),
                   ('TREND_INDICES', U['INDICES']), ('TREND_BONDS', U['BONDS'])):
    df, pos, ip = dp.run(names, kind, dp.tsmom, cost_mult=1.0, rates=rates)
    out[lab] = {p: dp.metrics(df, pos, ip, a, b) for p, a, b in P}
    print(lab, {p: round(out[lab][p]['sharpe'], 2) for p, *_ in P}, 'full PFm', round(out[lab]['full'].get('pf_monthly', float('nan')), 2))
json.dump(out, open('results/venue_diag_nomarkup.json', 'w'), indent=1, default=str)
