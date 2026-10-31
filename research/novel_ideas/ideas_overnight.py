"""Idea D (overnight drift in stock indices) from PREREGISTRATION.md, on OANDA M30 bid/ask 2015-2026.
Usage: FLOW_STORE=<parsed M30 dir> python3 ideas_overnight.py"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'flow_edges'))
import fe  # loader: bid_o/ask_o at the open of each M30 bar

SESS = {  # instrument: (timezone, close hh:mm, open hh:mm)
    'SPX500_USD': ('America/New_York', (16, 0), (9, 30)), 'NAS100_USD': ('America/New_York', (16, 0), (9, 30)),
    'US30_USD': ('America/New_York', (16, 0), (9, 30)), 'US2000_USD': ('America/New_York', (16, 0), (9, 30)),
    'DE30_EUR': ('Europe/Berlin', (17, 30), (9, 0)), 'UK100_GBP': ('Europe/London', (16, 30), (8, 0)),
    'JP225_USD': ('Asia/Tokyo', (15, 0), (9, 0)),
}
CARRY = 0.0001   # 1 bp per night (futures-style price includes financing)


def trades(inst, k=1.0, session='overnight'):
    tz, cl, op = SESS[inst]
    d = fe.load(inst)
    c = fe.at(d, tz, cl); o = fe.at(d, tz, op)
    days = sorted(set(c.index) & set(o.index))
    rows = []
    for i in range(len(days) - 1):
        a, b = days[i], days[i + 1]
        if session == 'overnight':
            if (pd.Timestamp(b) - pd.Timestamp(a)).days > 4:
                continue
            e, x = c.loc[a], o.loc[b]; nights = (pd.Timestamp(b) - pd.Timestamp(a)).days; date = a
        else:                     # intraday control: open -> close the same day
            e, x = o.loc[b], c.loc[b]; nights = 0; date = b
        pe, mid_e = fe.fill(e, 1, k); px, _ = fe.fill(x, -1, k)
        rows.append(dict(date=pd.Timestamp(date), mkt=inst, ret=(px - pe) / mid_e - CARRY * nights * (k > 0)))
    return pd.DataFrame(rows)


def stats(tr):
    r = tr.ret.values; by = tr.groupby('date').ret.mean()
    t = by.mean() / (by.std(ddof=1) / np.sqrt(len(by)))
    pf = r[r > 0].sum() / -r[r < 0].sum()
    return dict(n=int(len(r)), mean_bp=round(float(r.mean() * 1e4), 2), t=round(float(t), 2), pf=round(float(pf), 3))


def thirds(tr):
    ds = tr.date.sort_values().unique(); q = [ds[0], ds[len(ds) // 3], ds[2 * len(ds) // 3], ds[-1] + np.timedelta64(1, 'D')]
    return [stats(tr[(tr.date >= q[i]) & (tr.date < q[i + 1])]) for i in range(3)]


if __name__ == '__main__':
    res = {}
    for label, k, sess in (('overnight_1.5x', 1.5, 'overnight'), ('overnight_3x', 3.0, 'overnight'), ('overnight_gross', 0.0, 'overnight'),
                           ('intraday_control_1.5x', 1.5, 'intraday'), ('intraday_gross', 0.0, 'intraday')):
        tr = pd.concat([trades(i, k, sess) for i in SESS])
        res[label] = dict(pooled=stats(tr), thirds=thirds(tr), per_index={i: stats(g) for i, g in tr.groupby('mkt')})
        print(label, res[label]['pooled'], [(s['mean_bp'], s['pf']) for s in res[label]['thirds']])
        print('    ', {i: (v['mean_bp'], v['t']) for i, v in res[label]['per_index'].items()})
    p, th, p3 = res['overnight_1.5x']['pooled'], res['overnight_1.5x']['thirds'], res['overnight_3x']['pooled']
    res['verdict'] = 'PASS' if (all(s['mean_bp'] > 0 for s in th) and p['t'] >= 2.5 and p['pf'] >= 1.2 and p3['pf'] >= 1.05) else 'FAIL'
    print('D verdict', res['verdict'])
    json.dump(res, open(os.path.join(HERE, 'results_overnight.json'), 'w'), indent=1, default=str)
