"""10s / 30s / 1m strategies on 5-second data with a 5-second execution delay. Also a 'gross' run on mid prices (no spread)."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stf, pandas as pd, numpy as np
path, name = sys.argv[1], sys.argv[2]
f = stf.load(path)
mid = f.copy()
for k in 'ohlc':
    m = (f['bid_'+k] + f['ask_'+k]) / 2; mid['bid_'+k] = m; mid['ask_'+k] = m
m1 = stf.resample(f, '1min'); m1range = float((m1.h - m1.l).median())
grid = 10.0 if name.startswith('XAU') else 0.005
res = {}
for tf in ('10s', '30s', '1min'):
    b = stf.resample(f, tf)
    strats = {'S1_burst_mom': lambda x: stf.s1_s2(x), 'S2_burst_fade': lambda x: stf.s1_s2(x, fade=True),
              'S3_orb': stf.s3_orb, 'S4_news': stf.s4_news, 'S5_round': lambda x: stf.s5_round(x, grid, m1range), 'S6_pctb': stf.s6_pctb}
    for sname, fn in strats.items():
        ents = fn(b)
        for lab, fine, xs in (('net', f, 0.0), ('net+1spr', f, 1.0), ('gross_mid', mid, 0.0)):
            tr = stf.simulate(fine, ents, 5, 5, extra_spread=xs)
            if lab == 'gross_mid' and len(tr):   # express gross P&L in spread units of the real quotes
                sp = f.ask_o - f.bid_o; tr['spr'] = sp.reindex(tr.t + pd.Timedelta(seconds=5), method='nearest').values
            s = stf.stats(tr); res[f'{tf}|{sname}|{lab}'] = s
            print(f"{name} {tf:4s} {sname:14s} {lab:9s} {s}", flush=True)
json.dump(res, open(f'results_s5_{name}.json', 'w'), indent=1)
