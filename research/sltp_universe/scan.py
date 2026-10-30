"""Universe scan for the stop-loss trend strategy (see PREREGISTRATION.md).
Usage: DAILY_DIR=<daily9 dir> python3 scan.py"""
import os, json
import numpy as np, pandas as pd

DAILY = os.environ['DAILY_DIR']
GROUPS = {
    'metals': ['XAU_USD', 'XAG_USD', 'XPT_USD', 'XPD_USD', 'XCU_USD'],
    'energy': ['BCO_USD', 'WTICO_USD', 'NATGAS_USD'],
    'grains': ['CORN_USD', 'WHEAT_USD', 'SOYBN_USD', 'SUGAR_USD'],
    'indices': ['SPX500_USD', 'NAS100_USD', 'US30_USD', 'US2000_USD', 'UK100_GBP', 'DE30_EUR', 'FR40_EUR', 'EU50_EUR',
                'NL25_EUR', 'CH20_CHF', 'ESPIX_EUR', 'JP225_USD', 'AU200_AUD', 'HK33_HKD', 'CHINAH_HKD', 'CN50_USD', 'SG30_SGD'],
    'bonds': ['DE10YB_EUR', 'UK10YB_GBP', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD'],
    'fx': ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY'],
}
ROLLS = {'energy': 12, 'grains': 5}
KS = 3.0
PER = [('dev', '1900-01-01', '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', '2100-01-01')]


def load(n):
    d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
    t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
    m = pd.DataFrame({k: (d['bid_' + k] + d['ask_' + k]).values / 2 for k in 'ohlc'}, index=t)
    m['spread'] = (d.ask_o - d.bid_o).values
    m = m[~m.index.duplicated(keep='last')].sort_index()
    return m[m.index.dayofweek < 5]


def run(n, group, cost_mult):
    m = load(n); rolls = ROLLS.get(group, 4)
    c, o, h, l, sp, idx = m.c.values, m.o.values, m.h.values, m.l.values, m.spread.values, m.index
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(20).mean().values
    trades, pos = [], None
    for t in range(253, len(m)):
        if idx[t].dayofweek == 0:
            p = t - 1
            score = np.sign(c[p] - c[p - 63]) + np.sign(c[p] - c[p - 126]) + np.sign(c[p] - c[p - 252])
            d = 1 if score > 0 else -1
            if pos and d != pos['d']:
                trades.append(dict(pos, exit=o[t], xdate=idx[t], why='flip')); pos = None
            if pos is None and np.isfinite(atr[p]) and atr[p] > 0 and np.isfinite(o[t]):
                dist = KS * atr[p]
                pos = dict(d=d, entry=o[t], date=idx[t], i=t, dist=dist, stop=o[t] - d * dist, sp=sp[t])
        if pos:
            d, stop = pos['d'], pos['stop']
            if t > pos['i'] and d * (o[t] - stop) <= 0:
                trades.append(dict(pos, exit=o[t], xdate=idx[t], why='stop')); pos = None
            elif (d > 0 and l[t] <= stop) or (d < 0 and h[t] >= stop):
                trades.append(dict(pos, exit=stop, xdate=idx[t], why='stop')); pos = None
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    days = (df.xdate - df.date).dt.days.clip(lower=1)
    cost = cost_mult * df.sp * (1 + rolls * days / 365.0)
    df['R'] = (df.d * (df.exit - df.entry) - cost) / df.dist
    df['mkt'], df['group'] = n, group
    return df


def stats(df):
    if df is None or df.empty:
        return dict(n=0, avgR=None, pf=None)
    r = df.R.values
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float('inf')
    return dict(n=int(len(r)), avgR=round(float(r.mean()), 3), pf=round(float(pf), 2), win=round(float((r > 0).mean()), 3))


def period(df, a, b):
    return df[(df.date >= pd.Timestamp(a)) & (df.date <= pd.Timestamp(b))]


def equity(df, risk):
    """Trades applied in exit order at `risk` of equity each (approximation: ignores open-trade marking)."""
    e = 10000.0; peak = e; mdd = 0
    for r in df.sort_values('xdate').R:
        e *= 1 + risk * r; peak = max(peak, e); mdd = max(mdd, 1 - e / peak)
    yrs = max((df.xdate.max() - df.date.min()).days / 365.25, 0.5)
    return dict(end=round(e), cagr_pct=round(((e / 10000) ** (1 / yrs) - 1) * 100, 2), max_dd_pct=round(mdd * 100, 1))


def stake_for_100(n):
    m = load(n).tail(30)
    pc = m.c.shift()
    tr = np.maximum(m.h - m.l, np.maximum(abs(m.h - pc), abs(m.l - pc)))
    return float(KS * tr.tail(20).mean()), float(m.c.iloc[-1])


if __name__ == '__main__':
    out = {}
    T1 = {n: run(n, g, 1.5) for g, ns in GROUPS.items() for n in ns}
    T3 = {n: run(n, g, 3.0) for g, ns in GROUPS.items() for n in ns}
    grp = {n: g for g, ns in GROUPS.items() for n in ns}
    # 1. per market
    per = {}
    for n, df in T1.items():
        per[n] = {p: stats(period(df, a, b)) for p, a, b in PER}
        per[n]['full'] = stats(df); per[n]['full_3x'] = stats(T3[n]); per[n]['group'] = grp[n]
        sd, px = stake_for_100(n)
        per[n]['stop_distance_price'] = round(sd, 6); per[n]['price'] = px
    out['per_market'] = per
    # 2. whole universe
    ALL1 = pd.concat(T1.values()); ALL3 = pd.concat(T3.values())
    out['all_markets'] = {p: stats(period(ALL1, a, b)) for p, a, b in PER}
    out['all_markets']['full_3x'] = stats(ALL3)
    out['all_markets']['equity_1pct'] = equity(ALL1, 0.01); out['all_markets']['equity_0.5pct'] = equity(ALL1, 0.005)
    for p, a, b in PER:
        out['all_markets'][f'equity_0.5pct_{p}'] = equity(period(ALL1, a, b), 0.005)
    out['by_group'] = {g: {p: stats(period(ALL1[ALL1.group == g], a, b)) for p, a, b in PER} for g in GROUPS}
    # 3. does picking help? rank on development avgR only
    dev_rank = sorted(T1, key=lambda n: per[n]['dev']['avgR'] if per[n]['dev']['avgR'] is not None else -9, reverse=True)
    top10, bot10 = dev_rank[:10], dev_rank[-10:]
    div = [n for g in GROUPS for n in sorted(GROUPS[g], key=lambda n: per[n]['dev']['avgR'] or -9, reverse=True)[:2]]
    pick = {}
    for lab, sel in (('top10_by_dev', top10), ('bottom10_by_dev', bot10), ('best2_per_class_by_dev', div), ('all45', list(T1))):
        D = ALL1[ALL1.mkt.isin(sel)]
        pick[lab] = {'markets': sel, **{p: stats(period(D, a, b)) for p, a, b in PER if p != 'dev'}}
    out['picking_test'] = pick
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1, default=str)
    # console summary
    print('ALL 45:', {p: (out['all_markets'][p]['n'], out['all_markets'][p]['pf'], out['all_markets'][p]['avgR']) for p in ('dev', 'val', 'hold')},
          '3x', out['all_markets']['full_3x']['pf'], '1%', out['all_markets']['equity_1pct'], '0.5%', out['all_markets']['equity_0.5pct'])
    for p, a, b in PER:
        print('  0.5% risk', p, out['all_markets'][f'equity_0.5pct_{p}'])
    print('BY GROUP:'); [print(' ', g, {p: (v[p]['n'], v[p]['pf'], v[p]['avgR']) for p in ('dev', 'val', 'hold')}) for g, v in out['by_group'].items()]
    print('PICKING TEST (val / hold):'); [print(' ', k, (v['val']['pf'], v['val']['avgR']), (v['hold']['pf'], v['hold']['avgR']), v['markets'] if k != 'all45' else '') for k, v in pick.items()]
    print('PER MARKET (dev pf/avgR | val | hold):')
    for n in sorted(per, key=lambda n: (grp[n], n)):
        q = per[n]
        f = lambda s: f"{s['pf']}/{s['avgR']}" if s['n'] else '-'
        print(f"  {grp[n]:8} {n:11} {f(q['dev']):>13} | {f(q['val']):>13} | {f(q['hold']):>13}  n={q['full']['n']}")
