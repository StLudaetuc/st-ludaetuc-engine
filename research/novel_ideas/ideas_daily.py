"""Ideas A, B, C from PREREGISTRATION.md on OANDA daily data (weekday rows, 09:00 New York alignment).
Usage: DAILY_DIR=<daily9 dir> python3 ideas_daily.py"""
import os, json, itertools
import numpy as np, pandas as pd

DAILY = os.environ['DAILY_DIR']
HERE = os.path.dirname(os.path.abspath(__file__))
INDICES = ['SPX500_USD', 'NAS100_USD', 'US30_USD', 'US2000_USD', 'UK100_GBP', 'DE30_EUR', 'FR40_EUR', 'EU50_EUR', 'NL25_EUR',
           'CH20_CHF', 'ESPIX_EUR', 'JP225_USD', 'AU200_AUD', 'HK33_HKD', 'CHINAH_HKD', 'CN50_USD', 'SG30_SGD']
UNIVERSE = ['XAU_USD', 'XAG_USD', 'XPT_USD', 'XPD_USD', 'XCU_USD', 'BCO_USD', 'WTICO_USD', 'NATGAS_USD', 'CORN_USD', 'WHEAT_USD',
            'SOYBN_USD', 'SUGAR_USD'] + INDICES + ['DE10YB_EUR', 'UK10YB_GBP', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD',
            'EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY']
PAIRS = [('XAU_USD', 'XAG_USD'), ('BCO_USD', 'WTICO_USD'), ('SPX500_USD', 'NAS100_USD'), ('SPX500_USD', 'US30_USD'),
         ('DE30_EUR', 'EU50_EUR'), ('USB10Y_USD', 'DE10YB_EUR'), ('USB05Y_USD', 'USB10Y_USD'), ('AUD_USD', 'NZD_USD'),
         ('CORN_USD', 'WHEAT_USD'), ('XPT_USD', 'XPD_USD')]
PER = [('dev', '1900-01-01', '2018-12-31'), ('val', '2019-01-01', '2022-12-31'), ('hold', '2023-01-01', '2100-01-01')]
_cache = {}


def load(n):
    if n not in _cache:
        d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
        t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
        m = pd.DataFrame({k: (d['bid_' + k] + d['ask_' + k]).values / 2 for k in 'oc'}, index=t)
        m['sp'] = (d.ask_o - d.bid_o).values / m.o.values          # spread as a fraction of price
        m = m[~m.index.duplicated(keep='last')].sort_index()
        _cache[n] = m[m.index.dayofweek < 5]
    return _cache[n]


def stats(tr):
    """tr: DataFrame with columns date, ret (net fraction). t-stat clustered by date."""
    if tr is None or len(tr) == 0:
        return dict(n=0)
    r = tr.ret.values
    by = tr.groupby('date').ret.mean()
    t = by.mean() / (by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 2 else float('nan')
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float('inf')
    return dict(n=int(len(r)), mean_bp=round(float(r.mean() * 1e4), 1), t=round(float(t), 2), pf=round(float(pf), 2),
                win=round(float((r > 0).mean()), 3))


def by_period(tr):
    out = {p: stats(tr[(tr.date >= a) & (tr.date <= b)]) for p, a, b in PER}
    out['full'] = stats(tr)
    return out


def verdict(s15, s3, oos_only=False):
    ps = ('val', 'hold') if oos_only else ('dev', 'val', 'hold')
    ok = all(s15[p]['n'] > 0 and s15[p]['mean_bp'] > 0 for p in ps)
    if oos_only:
        both = s15['oos']
        ok = ok and both['t'] >= 2.5 and both['pf'] >= 1.2 and s3['oos']['pf'] >= 1.05
    else:
        ok = ok and s15['full']['t'] >= 2.5 and s15['full']['pf'] >= 1.2 and s3['full']['pf'] >= 1.05
    return 'PASS' if ok else 'FAIL'


# ---------------- A. panic reversal ----------------
def idea_a(N, cm):
    rows = []
    for n in INDICES:
        m = load(n)
        r = m.c.pct_change()
        sd = r.rolling(60).std().shift(1)
        sig = (r < -2 * sd).values
        o, sp, idx = m.o.values, m.sp.values, m.index
        t = 0
        while t < len(m) - N - 1:
            if sig[t]:
                e, x = o[t + 1], o[t + 1 + N]
                rows.append(dict(date=idx[t + 1], mkt=n, ret=x / e - 1 - cm * sp[t + 1]))
                t += N + 1          # no overlapping trades in the same market
            else:
                t += 1
    return pd.DataFrame(rows)


# ---------------- B. relative-value pairs ----------------
def idea_b(cm, entry=2.0, exit_=0.5, maxdays=20):
    rows = []
    for a, b in PAIRS:
        A, B = load(a), load(b)
        j = A.join(B, lsuffix='_a', rsuffix='_b', how='inner').dropna()
        s = np.log(j.c_a / j.c_b)
        z = ((s - s.rolling(60).mean()) / s.rolling(60).std()).values
        oa, ob, spa, spb, idx = j.o_a.values, j.o_b.values, j.sp_a.values, j.sp_b.values, j.index
        t = 60
        while t < len(j) - 2:
            if np.isfinite(z[t]) and abs(z[t]) > entry:
                side = -np.sign(z[t])        # spread high -> short A / long B
                e = t + 1; x = None
                for k in range(e, min(e + maxdays, len(j) - 1)):
                    if abs(z[k]) < exit_:
                        x = k + 1; break
                if x is None:
                    x = min(e + maxdays, len(j) - 1)
                ret = side * (oa[x] / oa[e] - 1) - side * (ob[x] / ob[e] - 1)
                cost = cm * (spa[e] + spb[e]) * 1.0           # one spread per leg, round trip
                rows.append(dict(date=idx[e], mkt=f'{a}/{b}', ret=ret - cost, days=x - e))
                t = x
            else:
                t += 1
    return pd.DataFrame(rows)


# ---------------- C. lead-lag ----------------
def idea_c(cm, top=20):
    C = pd.DataFrame({n: load(n).c for n in UNIVERSE}).sort_index()
    O = pd.DataFrame({n: load(n).o for n in UNIVERSE}).sort_index()
    SP = pd.DataFrame({n: load(n).sp for n in UNIVERSE}).sort_index()
    R = C.pct_change()
    nxt = O.shift(-2) / O.shift(-1) - 1                    # tradable: enter next open, exit the open after
    dev = R.index <= pd.Timestamp('2018-12-31')
    cands = []
    for a, b in itertools.permutations(UNIVERSE, 2):
        x = R[a][dev]; y = nxt[b][dev]
        ok = x.notna() & y.notna()
        if ok.sum() < 500:
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        t = c * np.sqrt((ok.sum() - 2) / (1 - c * c))
        cands.append((abs(t), a, b, np.sign(c), round(float(c), 4), round(float(t), 2)))
    cands.sort(reverse=True)
    chosen = cands[:top]
    rows = []
    for _, a, b, s, c, t in chosen:
        sig = np.sign(R[a]) * s
        g = (sig * nxt[b] - cm * SP[b].shift(-1)).dropna()
        g = g[(sig.reindex(g.index) != 0)]
        for d, v in g.items():
            rows.append(dict(date=d, mkt=f'{a}->{b}', ret=float(v)))
    return pd.DataFrame(rows), [dict(lead=a, follow=b, dev_corr=c, dev_t=t) for _, a, b, s, c, t in chosen]


if __name__ == '__main__':
    res = {}
    # A
    a = {}
    for N in (3, 1, 5):
        s15, s3 = by_period(idea_a(N, 1.5)), by_period(idea_a(N, 3.0))
        a[f'N={N}'] = dict(costs_1_5x=s15, costs_3x=s3, verdict=verdict(s15, s3) if N == 3 else 'variant')
        print('A panic reversal N=%d' % N, a[f'N={N}']['verdict'], {p: (s15[p]['n'], s15[p]['mean_bp'], s15[p]['pf']) for p in ('dev', 'val', 'hold')}, 't', s15['full']['t'], '3x pf', s3['full']['pf'])
    res['A_panic_reversal'] = a
    # B
    trB = idea_b(1.5); s15, s3 = by_period(trB), by_period(idea_b(3.0))
    perpair = {k: by_period(g) for k, g in trB.groupby('mkt')}
    res['B_pairs'] = dict(costs_1_5x=s15, costs_3x=s3, per_pair=perpair, verdict=verdict(s15, s3))
    print('B pairs', res['B_pairs']['verdict'], {p: (s15[p]['n'], s15[p]['mean_bp'], s15[p]['pf']) for p in ('dev', 'val', 'hold')}, 't', s15['full']['t'], '3x pf', s3['full']['pf'])
    for k, v in perpair.items():
        print('   ', k, {p: (v[p].get('n'), v[p].get('mean_bp'), v[p].get('pf')) for p in ('dev', 'val', 'hold')})
    # C
    trC15, chosen = idea_c(1.5); trC3, _ = idea_c(3.0)
    def oos(tr):
        d = by_period(tr); d['oos'] = stats(tr[tr.date >= '2019-01-01']); return d
    s15, s3 = oos(trC15), oos(trC3)
    res['C_lead_lag'] = dict(chosen=chosen, costs_1_5x=s15, costs_3x=s3, verdict=verdict(s15, s3, oos_only=True))
    print('C lead-lag', res['C_lead_lag']['verdict'], 'dev(in-sample)', (s15['dev']['mean_bp'], s15['dev']['pf']), 'val', (s15['val']['n'], s15['val']['mean_bp'], s15['val']['pf']),
          'hold', (s15['hold']['n'], s15['hold']['mean_bp'], s15['hold']['pf']), 'oos t', s15['oos']['t'], '3x oos pf', s3['oos']['pf'])
    print('   top pairs:', [(c['lead'], c['follow'], c['dev_t']) for c in chosen[:8]])
    json.dump(res, open(os.path.join(HERE, 'results_daily.json'), 'w'), indent=1, default=str)
