import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, bt, itertools, json
f = pd.read_pickle('data/feat.pkl')
N = len(f)
H, L, C, O = f.h.values, f.l.values, f.c.values, f.o.values
ATR = f.atr.values
SPLIT = pd.Timestamp('2026-08-01', tz='UTC')
T = f.t.values


def run(sig, entry_mode='close', k=3, stop_mode='v52', stop_mult=1.5, rr=1.6, buf_pips=0.6,
        swing_lb=12, stop_atr_pad=0.0, min_stop_pips=0.0, cost=bt.COST_PIPS, filt=None):
    rows = []
    busy_until = -1
    for _, x in sig.iterrows():
        i, d = int(x.i), int(x.dir)
        if filt is not None and not filt(i, d):
            continue
        # --- entry
        if entry_mode == 'close':
            j0, e = i, C[i]
        elif entry_mode == 'stop':   # breakout of signal bar extreme within k bars
            lvl = H[i] + 0.2 * bt.PIP if d == 1 else L[i] - 0.2 * bt.PIP
            j0 = None
            for j in range(i + 1, min(N, i + 1 + k)):
                if (d == 1 and H[j] >= lvl) or (d == -1 and L[j] <= lvl):
                    j0 = j; e = max(lvl, O[j]) if d == 1 else min(lvl, O[j]); break
                # invalidation: price breaks signal bar opposite extreme first
                if (d == 1 and L[j] < L[i]) or (d == -1 and H[j] > H[i]):
                    break
            if j0 is None:
                continue
        elif entry_mode == 'closebreak':  # bar-close confirmation: a later bar CLOSES beyond trigger-bar extreme within k bars
            j0 = None
            for j in range(i + 1, min(N, i + 1 + k)):
                if (d == 1 and L[j] < L[i]) or (d == -1 and H[j] > H[i]):
                    break
                if (d == 1 and C[j] > H[i]) or (d == -1 and C[j] < L[i]):
                    j0 = j; e = C[j]; break
            if j0 is None:
                continue
        elif entry_mode == 'confirm':  # next bar must close beyond signal close, enter at its close
            j = i + 1
            if j >= N: continue
            ok = (C[j] > C[i] and C[j] > O[j]) if d == 1 else (C[j] < C[i] and C[j] < O[j])
            if not ok: continue
            j0, e = j, C[j]
        elif entry_mode == 'limit':   # retrace limit of k*0.1 ATR within 5 bars
            lvl = C[i] - d * (k * 0.1) * ATR[i]
            j0 = None
            for j in range(i + 1, min(N, i + 6)):
                if (d == 1 and L[j] <= lvl) or (d == -1 and H[j] >= lvl):
                    j0 = j; e = min(lvl, O[j]) if d == 1 else max(lvl, O[j]); break
            if j0 is None: continue
        if j0 <= busy_until:  # one position at a time (Make/OANDA effectively 1 per pair)
            continue
        # --- stop
        a = ATR[i]
        if stop_mode == 'v52':
            sw = L[max(0, j0 - swing_lb + 1):j0 + 1].min() if d == 1 else H[max(0, j0 - swing_lb + 1):j0 + 1].max()
            if d == 1:
                st = min(e - a * stop_mult, sw - buf_pips * bt.PIP - stop_atr_pad * a)
            else:
                st = max(e + a * stop_mult, sw + buf_pips * bt.PIP + stop_atr_pad * a)
        elif stop_mode == 'swing':
            sw = L[max(0, j0 - swing_lb + 1):j0 + 1].min() if d == 1 else H[max(0, j0 - swing_lb + 1):j0 + 1].max()
            st = sw - d * (buf_pips * bt.PIP + stop_atr_pad * a)
        sp = abs(e - st) / bt.PIP
        if sp < min_stop_pips:
            st = e - d * min_stop_pips * bt.PIP; sp = min_stop_pips
        tg = e + d * abs(e - st) * rr
        out, bars, mfe, mae, r = bt.simulate(f, i, d, e, st, tg, j0 + 1)
        busy_until = j0 + bars
        rows.append(dict(i=i, j0=j0, dir=d, model=x.model, outcome=out, bars=bars, mfe=mfe, mae=mae, r=r,
                         stop_pips=sp, cost_r=cost / sp, t=T[i]))
    return pd.DataFrame(rows)


def split_stats(res, label):
    if len(res) == 0:
        return {'label': label, 'n': 0}
    ins = res[pd.to_datetime(res.t, utc=True) < SPLIT]; oos = res[pd.to_datetime(res.t, utc=True) >= SPLIT]
    a = bt.stats(res, label); b = bt.stats(ins); c = bt.stats(oos)
    return {**a, 'IS_n': b.get('n'), 'IS_wr': b.get('win_rate'), 'IS_R': b.get('avg_R_net'),
            'OOS_n': c.get('n'), 'OOS_wr': c.get('win_rate'), 'OOS_R': c.get('avg_R_net'), 'OOS_PF': c.get('profit_factor')}
