import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stf, pandas as pd, numpy as np
path, name = sys.argv[1], sys.argv[2]
f = stf.load(path)
b = f.copy()
for k in 'ohlc': b[k] = (b['bid_'+k] + b['ask_'+k]) / 2
b['spr'] = b.ask_c - b.bid_c
n = len(b); cuts = [b.index[0], b.index[int(n*0.6)], b.index[int(n*0.8)], b.index[-1] + pd.Timedelta(minutes=1)]
m1range = float((b.h - b.l).median())
grid = 10.0 if name.startswith('XAU') else 0.005 if 'JPY' not in name else 0.5
strats = {'S1_burst_mom': lambda x: stf.s1_s2(x), 'S2_burst_fade': lambda x: stf.s1_s2(x, fade=True),
          'S3_orb': stf.s3_orb, 'S4_news': stf.s4_news, 'S5_round': lambda x: stf.s5_round(x, grid, m1range), 'S6_pctb': stf.s6_pctb}
res = {}
for sname, fn in strats.items():
    ents = fn(b)
    for xs in (0.0, 1.0):
        tr = stf.simulate(f, ents, 60, 0, extra_spread=xs)
        for lab, a, z in (('dev', cuts[0], cuts[1]), ('val', cuts[1], cuts[2]), ('hold', cuts[2], cuts[3])):
            s = stf.stats(tr[(tr.t >= a) & (tr.t < z)] if len(tr) else tr)
            res[f'{sname}|{xs}|{lab}'] = s
            print(f"{name} 1m {sname:14s} xs={xs} {lab:4s} {s}")
json.dump(res, open(f'results_m1_{name}.json', 'w'), indent=1)
