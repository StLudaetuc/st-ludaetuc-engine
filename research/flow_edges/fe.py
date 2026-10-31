"""Flow-based edge tests (see PREREGISTRATION.md). Research only.
Data: OANDA M30 bid/ask; price at time T = open of the M30 bar starting at T. Missing bar -> trade skipped.
Usage: FLOW_STORE=<dir of parsed csv.gz> RATES=<oecd csv> python3 fe.py"""
import os, glob, json
import numpy as np, pandas as pd

STORE = os.environ.get('FLOW_STORE', 'data/store')
RATES = os.environ.get('RATES', 'data/oecd_ir3m.csv.gz')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CFD_MARKUP = 0.025


def load(inst):
    fs = glob.glob(os.path.join(STORE, f'M30__{inst}__*.csv.gz'))
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates('time_utc').sort_values('time_utc')
    d.index = pd.to_datetime(d.time_utc).dt.tz_localize('UTC')
    return d[['bid_o', 'ask_o']].astype(float)


def at(d, tz, hhmm):
    """Series indexed by local date: (bid, ask) at the open of the bar starting at local hh:mm."""
    loc = d.index.tz_convert(tz)
    m = (loc.hour == hhmm[0]) & (loc.minute == hhmm[1])
    s = d[m].copy()
    s.index = loc[m].date
    return s[~s.index.duplicated()]


def fill(row, side, k):
    """side +1 buy, -1 sell. Fill = mid +/- k*(half-spread + 10% of spread)."""
    b, a = row['bid_o'], row['ask_o']
    mid, h = (a + b) / 2, (a - b) / 2
    return mid + side * k * 1.2 * h, mid


def rates_usd():
    r = pd.read_csv(RATES)
    r = r[r.REF_AREA == 'USA'].set_index('TIME_PERIOD')['OBS_VALUE'] / 100
    r.index = pd.to_datetime(r.index) + pd.offsets.MonthBegin(1)
    return r.sort_index()


def trade(e_row, x_row, side, k):
    pe, mid = fill(e_row, side, k)
    px, _ = fill(x_row, -side, k)
    return side * (px - pe) / mid


# ---------------- H1 intraday momentum ----------------
def h1(k=1.0, insts=('SPX500_USD', 'NAS100_USD', 'US30_USD', 'US2000_USD')):
    rows = []
    for inst in insts:
        d = load(inst)
        a = at(d, 'America/New_York', (15, 30)); c = at(d, 'America/New_York', (16, 0))
        days = sorted(set(a.index) & set(c.index))
        for i in range(1, len(days)):
            p, t = days[i - 1], days[i]
            if (pd.Timestamp(t) - pd.Timestamp(p)).days > 4:
                continue
            m0 = c.loc[p].mean(); m1 = a.loc[t].mean()
            r = m1 / m0 - 1
            if r == 0:
                continue
            side = 1 if r > 0 else -1
            rows.append(dict(inst=inst, date=t, side=side, ret=trade(a.loc[t], c.loc[t], side, k)))
    return pd.DataFrame(rows)


# ---------------- H2 turn of month ----------------
def h2(k=1.0, insts=('SPX500_USD', 'NAS100_USD', 'US30_USD', 'US2000_USD')):
    ru = rates_usd()
    rows, base = [], []
    for inst in insts:
        a = at(load(inst), 'America/New_York', (15, 30))
        days = pd.to_datetime(pd.Index(sorted(a.index)))
        ym = days.to_period('M')
        last = [i for i in range(len(days) - 1) if ym[i] != ym[i + 1]]
        tom_entries = set()
        for i in last:
            j = i + 3  # third trading day of next month
            if j >= len(days) or ym[j] != ym[i + 1]:
                continue
            tom_entries.add(i)
        def one(i, j):
            e, x = a.loc[days[i].date()], a.loc[days[j].date()]
            r = trade(e, x, 1, k)
            nights = (days[j] - days[i]).days
            rate = ru.asof(days[i]) if len(ru) else 0.0
            rate = 0.0 if pd.isna(rate) else rate
            return r - nights * (rate + CFD_MARKUP) / 365.0
        for i in tom_entries:
            rows.append(dict(inst=inst, date=days[i].date(), side=1, ret=one(i, i + 3)))
        for i in range(len(days) - 3):
            if i in tom_entries:
                continue
            if (days[i + 3] - days[i]).days > 7:
                continue
            base.append(dict(inst=inst, date=days[i].date(), side=1, ret=one(i, i + 3)))
    return pd.DataFrame(rows), pd.DataFrame(base)


# ---------------- H3 month-end FX rebalancing ----------------
PAIRS = {'EUR_USD': ('DE30_EUR', 'EUR_USD', -1), 'GBP_USD': ('UK100_GBP', 'GBP_USD', -1),
         'USD_JPY': ('JP225_USD', None, +1), 'AUD_USD': ('AU200_AUD', 'AUD_USD', -1)}
# third field: direction of the pair when buying USD


def h3(k=1.0):
    L = 'Europe/London'
    spx = at(load('SPX500_USD'), L, (10, 0))
    rows = []
    for pair, (idx, conv, usd_side) in PAIRS.items():
        fx = load(pair)
        f10, f16 = at(fx, L, (10, 0)), at(fx, L, (16, 0))
        ix = at(load(idx), L, (10, 0))
        days = pd.to_datetime(pd.Index(sorted(set(f10.index) & set(f16.index))))
        days = days[days.dayofweek < 5]
        ym = days.to_period('M')
        last = [days[i] for i in range(len(days) - 1) if ym[i] != ym[i + 1]]
        for p, t in zip(last[:-1], last[1:]):
            p, t = p.date(), t.date()
            try:
                def usd(day):
                    v = ix.loc[day].mean()
                    return v * f10.loc[day].mean() if conv else v
                rf = usd(t) / usd(p) - 1
                rs = spx.loc[t].mean() / spx.loc[p].mean() - 1
            except KeyError:
                continue
            m = rf - rs
            if m == 0:
                continue
            side = usd_side if m > 0 else -usd_side
            rows.append(dict(inst=pair, date=t, side=side, m=m, ret=trade(f10.loc[t], f16.loc[t], side, k)))
    return pd.DataFrame(rows)


# ---------------- statistics ----------------
def stats(df):
    if df.empty:
        return {}
    r = df.ret.values
    by_date = df.groupby('date').ret.mean()
    t = by_date.mean() / (by_date.std(ddof=1) / np.sqrt(len(by_date)))
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float('inf')
    dates = pd.to_datetime(pd.Series(sorted(df.date.unique())))
    cuts = [dates.iloc[0], dates.quantile(1 / 3), dates.quantile(2 / 3), dates.iloc[-1] + pd.Timedelta(days=1)]
    dd = pd.to_datetime(df.date)
    thirds = [float(df.ret[(dd >= cuts[i]) & (dd < cuts[i + 1])].mean() * 1e4) for i in range(3)]
    per_inst = {k: float(v * 1e4) for k, v in df.groupby('inst').ret.mean().items()}
    return dict(n=int(len(r)), mean_bp=float(r.mean() * 1e4), t_clustered=float(t), pf=float(pf),
                win=float((r > 0).mean()), thirds_bp=thirds, per_inst_bp=per_inst,
                first=str(dates.iloc[0].date()), last=str(dates.iloc[-1].date()))


def verdict(s1, s2, extra=True):
    ok = (s1['t_clustered'] >= 2.5 and s1['pf'] >= 1.15 and s2['pf'] >= 1.05 and min(s1['thirds_bp']) > 0
          and np.mean([v > 0 for v in s1['per_inst_bp'].values()]) >= 0.5 and extra)
    return 'PASS' if ok else 'FAIL'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {}
    a1, a2 = h1(1.0), h1(2.0)
    res['H1'] = dict(x1=stats(a1), x2=stats(a2), gross=stats(h1(0.0)),
                     post2021_x1=stats(a1[pd.to_datetime(a1.date) >= '2021-01-01']))
    res['H1']['verdict'] = verdict(res['H1']['x1'], res['H1']['x2'])
    a1.to_csv(os.path.join(OUT, 'h1_trades_1x.csv.gz'), index=False)
    (b1, base1), (b2, _) = h2(1.0), h2(2.0)
    s_base = stats(base1)
    res['H2'] = dict(x1=stats(b1), x2=stats(b2), gross=stats(h2(0.0)[0]), baseline_other_windows_x1=s_base)
    res['H2']['verdict'] = verdict(res['H2']['x1'], res['H2']['x2'], res['H2']['x1']['mean_bp'] > s_base['mean_bp'])
    b1.to_csv(os.path.join(OUT, 'h2_trades_1x.csv.gz'), index=False)
    c1, c2 = h3(1.0), h3(2.0)
    res['H3'] = dict(x1=stats(c1), x2=stats(c2), gross=stats(h3(0.0)))
    res['H3']['verdict'] = verdict(res['H3']['x1'], res['H3']['x2'])
    c1.to_csv(os.path.join(OUT, 'h3_trades_1x.csv.gz'), index=False)
    json.dump(res, open(os.path.join(OUT, 'prereg_results.json'), 'w'), indent=1, default=str)
    for h, v in res.items():
        print(h, v['verdict'])
        for k2, s in v.items():
            if isinstance(s, dict) and s:
                print('  ', k2, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in s.items() if kk != 'per_inst_bp'})
                print('     per_inst_bp', {kk: round(vv, 2) for kk, vv in s['per_inst_bp'].items()})
