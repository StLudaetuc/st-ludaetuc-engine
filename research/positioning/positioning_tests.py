"""Tests 2 (OANDA position book) and 3 (CFTC COT hedging pressure) from PREREGISTRATION.md.
Usage: DAILY_DIR=<daily9> COT_DIR=<dir with cot_<code>.csv> PB_FILE=<positionbook csv> python3 positioning_tests.py"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DAILY = os.environ['DAILY_DIR']
COT = {  # CFTC code: (OANDA instrument, sign of the OANDA price vs the futures contract)
    '088691': ('XAU_USD', 1), '084691': ('XAG_USD', 1), '085692': ('XCU_USD', 1), '067651': ('WTICO_USD', 1),
    '023651': ('NATGAS_USD', 1), '002602': ('CORN_USD', 1), '001602': ('WHEAT_USD', 1), '005602': ('SOYBN_USD', 1),
    '080732': ('SUGAR_USD', 1), '13874A': ('SPX500_USD', 1), '209742': ('NAS100_USD', 1), '042601': ('USB02Y_USD', 1),
    '044601': ('USB05Y_USD', 1), '043602': ('USB10Y_USD', 1), '020601': ('USB30Y_USD', 1), '099741': ('EUR_USD', 1),
    '096742': ('GBP_USD', 1), '097741': ('USD_JPY', -1), '232741': ('AUD_USD', 1), '090741': ('USD_CAD', -1),
    '092741': ('USD_CHF', -1), '112741': ('NZD_USD', 1)}
ROLLS = {'WTICO_USD': 12, 'NATGAS_USD': 12, 'CORN_USD': 5, 'WHEAT_USD': 5, 'SOYBN_USD': 5}


def load(n):
    d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
    t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
    m = pd.DataFrame({'o': (d.bid_o + d.ask_o).values / 2, 'sp': ((d.ask_o - d.bid_o) / ((d.ask_o + d.bid_o) / 2)).values}, index=t)
    m = m[~m.index.duplicated(keep='last')].sort_index()
    return m[m.index.dayofweek < 5]


def weekly_trades(signal, inst, cm):
    """signal: Series indexed by the Monday the position applies from (+1 long, -1 short, 0 flat), in OANDA direction.
    Holds Monday open -> next Monday open. Costs: half spread x cm per unit of position change, plus rolls while held."""
    m = load(inst); rolls = ROLLS.get(inst, 4)
    days = m.index
    rows, prev = [], 0.0
    for mon, pos in signal.sort_index().items():
        i = days.searchsorted(mon)
        j = days.searchsorted(mon + pd.Timedelta(days=7))
        if j >= len(days) or i >= len(days):
            break
        o0, o1, sp = m.o.iloc[i], m.o.iloc[j], m.sp.iloc[i]
        cost = cm * sp * 0.5 * abs(pos - prev) + cm * sp * abs(pos) * rolls / 52
        prev = pos
        if pos != 0:
            rows.append(dict(date=days[i], mkt=inst, ret=pos * (o1 / o0 - 1) - cost))
        elif cost > 0 and rows:   # closing cost charged to the last held week
            rows[-1]['ret'] -= cost
    return pd.DataFrame(rows)


def stats(tr):
    if tr is None or len(tr) == 0:
        return dict(n=0)
    r = tr.ret.values
    by = tr.groupby('date').ret.mean()
    t = by.mean() / (by.std(ddof=1) / np.sqrt(len(by))) if len(by) > 2 else float('nan')
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float('inf')
    return dict(n=int(len(r)), mean_bp=round(float(r.mean() * 1e4), 1), t=round(float(t), 2), pf=round(float(pf), 3),
                win=round(float((r > 0).mean()), 3))


def evaluate(tr15, tr3):
    ds = np.sort(tr15.date.unique()); q = [ds[0], ds[len(ds) // 3], ds[2 * len(ds) // 3], ds[-1] + np.timedelta64(1, 'D')]
    thirds = [stats(tr15[(tr15.date >= q[i]) & (tr15.date < q[i + 1])]) for i in range(3)]
    s15, s3 = stats(tr15), stats(tr3)
    ok = all(t['n'] > 0 and t['mean_bp'] > 0 for t in thirds) and s15['t'] >= 2.5 and s15['pf'] >= 1.2 and s3['pf'] >= 1.05
    return dict(pooled_1_5x=s15, pooled_3x=s3, thirds=thirds, thirds_start=[str(pd.Timestamp(x).date()) for x in q[:3]],
                per_market={k: stats(g) for k, g in tr15.groupby('mkt')}, verdict='PASS' if ok else 'FAIL')


# ---------------- Test 3: COT ----------------
def cot_signals(variant):
    out = {}
    for code, (inst, sgn) in COT.items():
        p = os.path.join(os.environ['COT_DIR'], f'cot_{code}.csv')
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p)
        c['date'] = pd.to_datetime(c.report_date_as_yyyy_mm_dd)
        c = c.sort_values('date').drop_duplicates('date').set_index('date')
        oi = c.open_interest_all.astype(float)
        if variant == 'hedging_pressure':
            x = (c.comm_positions_long_all - c.comm_positions_short_all) / oi
            pct = x.rolling(156, min_periods=156).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
            sig = np.where(pct < 0.2, 1, np.where(pct > 0.8, -1, 0))
        else:   # speculator crowding contrarian
            x = (c.noncomm_positions_long_all - c.noncomm_positions_short_all) / oi
            pct = x.rolling(156, min_periods=156).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
            sig = np.where(pct > 0.9, -1, np.where(pct < 0.1, 1, 0))
        s = pd.Series(sig * sgn, index=c.index + pd.Timedelta(days=6))[pct.notna().values]   # Tuesday report -> Monday
        out[inst] = s[s.index >= '2007-01-01']
    return out


# ---------------- Test 2: OANDA position book ----------------
def pb_signals(variant):
    pb = pd.read_csv(os.environ['PB_FILE'], parse_dates=['date'])
    pb = pb[pb.status == 200]
    out = {}
    for inst, g in pb.groupby('inst'):
        g = g.sort_values('date').drop_duplicates('date')
        share = g.long / (g.long + g.short)
        if variant == 'threshold':
            sig = np.where(share > 0.6, -1, np.where(share < 0.4, 1, 0))
        else:
            sig = -np.sign(share - 0.5)
        out[inst] = pd.Series(sig, index=pd.DatetimeIndex(g.date) + pd.Timedelta(days=3))   # Friday snapshot -> Monday
    return out


def run(sigfn, variant):
    sigs = sigfn(variant)
    tr15 = pd.concat([weekly_trades(s, i, 1.5) for i, s in sigs.items()])
    tr3 = pd.concat([weekly_trades(s, i, 3.0) for i, s in sigs.items()])
    return evaluate(tr15, tr3)


if __name__ == '__main__':
    out = os.path.join(HERE, 'results.json')
    res = json.load(open(out)) if os.path.exists(out) else {}   # COT and position book can be run separately
    if os.environ.get('COT_DIR'):
        for v in ('hedging_pressure', 'speculator_contrarian'):
            res[f'COT_{v}'] = run(cot_signals, v)
            r = res[f'COT_{v}']
            print(f'COT {v}:', r['verdict'] if v == 'hedging_pressure' else '(variant)', r['pooled_1_5x'], '3x pf', r['pooled_3x']['pf'],
                  [(t['n'], t['mean_bp'], t['pf']) for t in r['thirds']], r['thirds_start'])
            print('   ', {k: (v2['n'], v2['mean_bp']) for k, v2 in r['per_market'].items()})
    if os.environ.get('PB_FILE'):
        for v in ('threshold', 'continuous'):
            res[f'PB_{v}'] = run(pb_signals, v)
            r = res[f'PB_{v}']
            print(f'Position book {v}:', r['verdict'] if v == 'threshold' else '(variant)', r['pooled_1_5x'], '3x pf', r['pooled_3x']['pf'],
                  [(t['n'], t['mean_bp'], t['pf']) for t in r['thirds']], r['thirds_start'])
            print('   ', {k: (v2['n'], v2['mean_bp']) for k, v2 in r['per_market'].items()})
    json.dump(res, open(out, 'w'), indent=1, default=str)
