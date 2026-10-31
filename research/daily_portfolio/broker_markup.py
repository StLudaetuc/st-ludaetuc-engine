"""Diagnostic (not pre-registered): daily metals+commodities trend under different broker overnight markups.
OANDA spreads kept for all rows; only the annual financing markup on CFD notional changes."""
import os, sys, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp
rates = dp.load_rates(); U = dp.universe()
names = U['METALS'] + U['COMMOD']
kind = {n: 'CFD' for n in names}
_orig = dp.fin_rates
P = [('dev', None, '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', None), ('full', None, None)]
out = {}
for lab, mk in (('futures (no markup)', 0.0), ('OANDA (~2.5%)', 0.025), ('Capital.com (4%)', 0.04)):
    def fr(nm, idx, r, k, mk=mk):
        L, S = _orig(nm, idx, r, k)
        return L + 0.025 - mk, S + 0.025 - mk
    dp.fin_rates = fr
    df, pos, ip = dp.run(names, kind, dp.tsmom, cost_mult=1.0, rates=rates)
    out[lab] = {p: round(dp.metrics(df, pos, ip, a, b)['sharpe'], 2) for p, a, b in P}
    out[lab]['ann_return_pct_full'] = round(float(df.pnl.mean() * 252 * 100), 2)
    print(lab, out[lab])
json.dump(out, open('results/broker_markup_metals_commod.json', 'w'), indent=1)
