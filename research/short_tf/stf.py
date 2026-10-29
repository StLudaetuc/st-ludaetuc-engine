"""Short-timeframe strategy backtester (see PREREGISTRATION.md). Research only.
Fills: signal at bar close T -> fill at the open of the fine (5s or 1m) bar starting at T + delay, on bid/ask."""
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
NY, LDN = ZoneInfo('America/New_York'), ZoneInfo('Europe/London')
COLS = ['bid_o','bid_h','bid_l','bid_c','ask_o','ask_h','ask_l','ask_c']

def load(path):
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.time_utc, utc=True); d = d.drop(columns='time_utc')
    d = d[~d.index.duplicated()].sort_index()
    return d

def resample(f, rule):
    agg = {'bid_o':'first','bid_h':'max','bid_l':'min','bid_c':'last','ask_o':'first','ask_h':'max','ask_l':'min','ask_c':'last','volume':'sum'}
    r = f.resample(rule, label='left', closed='left').agg(agg).dropna(subset=['bid_o'])
    for k in 'ohlc': r[k] = (r['bid_'+k] + r['ask_'+k]) / 2
    r['spr'] = r.ask_c - r.bid_c
    return r

def simulate(fine, entries, fine_sec, delay_sec, extra_spread=0.0):
    """entries: list of dict(t_signal (bar close Timestamp), d (+1/-1), stop, target, t_exit (time-based exit signal close))
    returns DataFrame of trades with pnl (price units, after costs), spread_at_entry."""
    idx = fine.index; ti = idx.values
    bo, bh, bl, ao, ah, al = (fine[c].values for c in ('bid_o','bid_h','bid_l','ask_o','ask_h','ask_l'))
    out = []; busy_until = None
    for e in entries:
        tf = e['t_signal'] + pd.Timedelta(seconds=delay_sec)
        if busy_until is not None and tf <= busy_until: continue          # one position at a time
        i = idx.searchsorted(tf)
        if i >= len(idx) or (idx[i] - tf) > pd.Timedelta(seconds=max(60, 3*fine_sec)): continue
        d = e['d']; entry = ao[i] if d == 1 else bo[i]; spr0 = ao[i] - bo[i]
        stop, tgt = e.get('stop'), e.get('target')
        if stop is not None and ((d == 1 and entry <= stop) or (d == -1 and entry >= stop)): continue
        tx = e['t_exit'] + pd.Timedelta(seconds=delay_sec)
        exit_px = None; j = i
        # walk fine bars; stop/target checked from the entry bar onward (entry at its open)
        while j < len(idx):
            if idx[j] >= tx:
                exit_px = bo[j] if d == 1 else ao[j]; why = 'time'; break
            lo_side = bl[j] if d == 1 else al[j]; hi_side = bh[j] if d == 1 else ah[j]
            if stop is not None and ((d == 1 and lo_side <= stop) or (d == -1 and hi_side >= stop)):
                exit_px = stop; why = 'stop'; break
            if tgt is not None and ((d == 1 and hi_side >= tgt) or (d == -1 and lo_side <= tgt)):
                exit_px = tgt; why = 'target'; break
            j += 1
        if exit_px is None: continue
        pnl = (exit_px - entry) * d - extra_spread * spr0
        risk = abs(entry - stop) if stop is not None else np.nan
        out.append(dict(t=e['t_signal'], d=d, entry=entry, exit=exit_px, pnl=pnl, spr=spr0, why=why,
                        R=pnl / risk if risk and risk > 0 else np.nan, t_out=idx[j]))
        busy_until = idx[j]
    return pd.DataFrame(out)

def stats(tr):
    if tr is None or len(tr) == 0: return dict(n=0)
    p = tr.pnl.values; gp, gl = p[p > 0].sum(), -p[p < 0].sum()
    se = p.std() / np.sqrt(len(p)) if len(p) > 1 else np.nan
    return dict(n=len(p), pf=round(float(gp / gl), 3) if gl > 0 else None, win=round(float((p > 0).mean()), 3),
                avg_spreads=round(float((p / tr.spr.replace(0, np.nan)).mean()), 3), t=round(float(p.mean() / se), 2) if se else None)

# ---------------- strategies: each returns list of entries from bars b (index = bar open time) -----------------
def _close_time(b, k): return b.index[k] + (b.index[1] - b.index[0]) if len(b) > 1 else b.index[k]

def s1_s2(b, fade=False, K=6, H=6):
    step = b.index.to_series().diff().median()
    mv = b.c - b.c.shift(K); vol = b.volume.rolling(K).sum()
    vq = vol.rolling(2000, min_periods=500).quantile(0.9).shift(1)
    trig = (mv.abs() > 3 * b.spr) & (vol > vq)
    ents = []
    for t in b.index[trig.fillna(False).values]:
        d = int(np.sign(mv.loc[t])) * (-1 if fade else 1)
        if d == 0: continue
        tc = t + step
        ents.append(dict(t_signal=tc, d=d, stop=None, target=None, t_exit=tc + H * step))
    return ents

def s3_orb(b):
    step = b.index.to_series().diff().median(); ents = []
    for tz, hh, mm in ((LDN, 7, 0), (NY, 9, 30)):
        loc = b.index.tz_convert(tz)
        days = pd.Series(loc.date, index=b.index)
        for day, g in b.groupby(days.values):
            lt = g.index.tz_convert(tz)
            st = pd.Timestamp(day).tz_localize(tz) + pd.Timedelta(hours=hh, minutes=mm)
            if st.weekday() >= 5: continue
            rng = g[(g.index >= st) & (g.index < st + pd.Timedelta(minutes=5))]
            if len(rng) == 0: continue
            hi, lo = rng.h.max(), rng.l.min()
            after = g[(g.index >= st + pd.Timedelta(minutes=5)) & (g.index < st + pd.Timedelta(minutes=60))]
            for t, r in after.iterrows():
                if r.c > hi: ents.append(dict(t_signal=t + step, d=1, stop=lo, target=None, t_exit=t + step + pd.Timedelta(minutes=30))); break
                if r.c < lo: ents.append(dict(t_signal=t + step, d=-1, stop=hi, target=None, t_exit=t + step + pd.Timedelta(minutes=30))); break
    return sorted(ents, key=lambda e: e['t_signal'])

def s4_news(b):
    step = b.index.to_series().diff().median(); ents = []
    loc = b.index.tz_convert(NY)
    for day in sorted(set(loc.date)):
        st = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=8, minutes=30)
        if st.weekday() >= 5: continue
        w = b[(b.index >= st) & (b.index < st + pd.Timedelta(seconds=30))]
        pre = b[b.index < st]
        if len(w) == 0 or len(pre) == 0: continue
        mv = w.c.iloc[-1] - pre.c.iloc[-1]
        if abs(mv) <= 2 * w.spr.iloc[-1]: continue
        tc = w.index[-1] + step
        ents.append(dict(t_signal=tc, d=int(np.sign(mv)), stop=None, target=None, t_exit=tc + pd.Timedelta(minutes=5)))
    return ents

def s5_round(b, grid, m1range):
    step = b.index.to_series().diff().median(); ents = []
    lvl_prev = None
    lo_l = np.floor(b.l / grid); hi_l = np.floor(b.h / grid)
    for k in range(1, len(b)):
        if hi_l.iloc[k] > lo_l.iloc[k] or np.floor(b.c.iloc[k-1] / grid) != np.floor(b.c.iloc[k] / grid):
            lvl = (hi_l.iloc[k]) * grid
            if lvl == lvl_prev: continue
            lvl_prev = lvl
            d = -1 if b.c.iloc[k-1] < lvl else 1     # approached from below -> fade = short
            tc = b.index[k] + step; c = b.c.iloc[k]; R = 1.5 * m1range
            ents.append(dict(t_signal=tc, d=d, stop=c - d * R, target=c + d * R, t_exit=tc + pd.Timedelta(minutes=15)))
    return ents

def s6_pctb(b):
    step = b.index.to_series().diff().median()
    m = b.c.rolling(20).mean(); s = b.c.rolling(20).std(ddof=0); pb = (b.c - (m - 2*s)) / (4*s) * 100
    ents = []
    for k in range(21, len(b)):
        if pb.iloc[k-1] < 0 and pb.iloc[k] >= 0:
            ext = b.l.iloc[k-5:k+1].min(); tc = b.index[k] + step
            ents.append(dict(t_signal=tc, d=1, stop=ext - b.spr.iloc[k], target=m.iloc[k], t_exit=tc + 60 * step))
        elif pb.iloc[k-1] > 100 and pb.iloc[k] <= 100:
            ext = b.h.iloc[k-5:k+1].max(); tc = b.index[k] + step
            ents.append(dict(t_signal=tc, d=-1, stop=ext + b.spr.iloc[k], target=m.iloc[k], t_exit=tc + 60 * step))
    return ents
