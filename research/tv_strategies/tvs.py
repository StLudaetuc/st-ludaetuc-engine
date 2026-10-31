"""Faithful Python replications of two old St Ludaetuc TradingView strategies, run on OANDA bid/ask data.
S1 = "Bollinger Bands and RSI Strategy with Webhook Alerts" (stop-and-reverse, no stop loss).
S2 = "St Ludaetuc BB + RSI 2-Candlestick Strategy with RSI Reversal".
Signals on mid-price bars at bar close; orders fill at the NEXT bar open (TradingView default).
mode 'tv'  : fills at mid, no costs (what the TradingView Strategy Tester shows).
mode 'real': buys at ask, sells at bid, plus overnight financing (USD 3m rate + 2.5% markup on notional).
Quantity 1 unit (1 oz for XAU_USD), as in the TradingView default."""
import numpy as np, pandas as pd

USD_RATE = 0.039   # USD 3-month rate, Oct 2026 (OECD)
MARKUP = 0.025


def resample(df, rule):
    o = {'bid_o': 'first', 'bid_h': 'max', 'bid_l': 'min', 'bid_c': 'last',
         'ask_o': 'first', 'ask_h': 'max', 'ask_l': 'min', 'ask_c': 'last'}
    r = df.resample(rule, label='left', closed='left').agg(o).dropna()
    return r


def load(path, rule=None):
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.time_utc, utc=True)
    d = d.drop(columns=['time_utc', 'volume'], errors='ignore').astype(float)
    return resample(d, rule) if rule else d


def mid(b):
    m = pd.DataFrame({c: (b['bid_' + c] + b['ask_' + c]) / 2 for c in 'ohlc'}, index=b.index)
    return m


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(m, n=14):
    tr = pd.concat([m.h - m.l, (m.h - m.c.shift()).abs(), (m.l - m.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bands(c, n=21, k=2.0):
    b = c.rolling(n).mean(); s = c.rolling(n).std(ddof=0)
    return b + k * s, b - k * s


def signals_s1(m):
    c, h, l = m.c.values, m.h.values, m.l.values
    up, lo = bands(m.c); up, lo = up.values, lo.values
    r = rsi(m.c).values
    sig = np.zeros(len(m), int)
    prevH = prevL = np.nan; taken = False
    for i in range(1, len(m)):
        xo_up = c[i] > up[i] and c[i - 1] <= up[i - 1]
        xu_lo = c[i] < lo[i] and c[i - 1] >= lo[i - 1]
        shortC = xo_up and r[i] >= 77
        longC = xu_lo and r[i] <= 27
        se = le = False
        if shortC and not taken: prevH = h[i]
        if not np.isnan(prevH) and h[i] > prevH and (r[i] < 77 and r[i - 1] >= 77) and not taken:
            se = True; taken = True
        if longC and not taken: prevL = l[i]
        if not np.isnan(prevL) and l[i] < prevL and (r[i] > 27 and r[i - 1] <= 27) and not taken:
            le = True; taken = True
        if not se and not le: taken = False
        sig[i] = -1 if se else (1 if le else 0)
    return sig, None


def signals_s2(m):
    c, h, l = m.c.values, m.h.values, m.l.values
    up, lo = bands(m.c); up, lo = up.values, lo.values
    r = rsi(m.c).values; a = atr(m).values
    sig = np.zeros(len(m), int); prevH = prevL = prevR = np.nan
    for i in range(1, len(m)):
        fs = c[i - 1] > up[i - 1] and r[i - 1] > 77
        if fs: prevH, prevR = h[i - 1], r[i - 1]
        sc = fs and h[i] >= prevH and r[i] < prevR
        fl = c[i - 1] < lo[i - 1] and r[i - 1] < 27
        if fl: prevL, prevR = l[i - 1], r[i - 1]
        lc = fl and l[i] <= prevL and r[i] > prevR
        if sc: sig[i] = -1
        if lc: sig[i] = 1
    return sig, a


def simulate(b, sig, a=None, mode='real', exits=False, slm=8.5, tpm=3.0):
    """Always-in stop-and-reverse on signals; optional ATR stop/target (S2 'exits=True' variant)."""
    m = mid(b)
    t = b.index; trades = []; pos = 0; ent = None
    def px(i, side, which='o'):
        if mode == 'tv': return m[which].iloc[i]
        return (b['ask_' + which] if side > 0 else b['bid_' + which]).iloc[i]
    def close_at(i, price, why):
        nonlocal pos, ent
        pnl = pos * (price - ent['p'])
        if mode == 'real':
            nights = (t[i].tz_convert('America/New_York').normalize() - ent['t'].tz_convert('America/New_York').normalize()).days
            rate = (USD_RATE + MARKUP) if pos > 0 else (MARKUP - USD_RATE)
            pnl -= nights * rate / 365 * ent['p']
        trades.append(dict(entry=ent['t'], exit=t[i], side=pos, pin=ent['p'], pout=price, pnl=pnl, why=why))
        pos = 0; ent = None
    for i in range(len(b) - 1):
        # stop / target check on bar i+1 path is handled at the start of each bar below
        if exits and pos != 0 and ent['sl'] is not None:
            j = i
            hi, lo_ = m.h.iloc[j], m.l.iloc[j]
            if j > ent['i']:
                if pos > 0 and lo_ <= ent['sl']: close_at(j, ent['sl'] - (0 if mode == 'tv' else (b.ask_o.iloc[j] - b.bid_o.iloc[j]) / 2), 'sl')
                elif pos < 0 and hi >= ent['sl']: close_at(j, ent['sl'] + (0 if mode == 'tv' else (b.ask_o.iloc[j] - b.bid_o.iloc[j]) / 2), 'sl')
                elif pos > 0 and hi >= ent['tp']: close_at(j, ent['tp'] - (0 if mode == 'tv' else (b.ask_o.iloc[j] - b.bid_o.iloc[j]) / 2), 'tp')
                elif pos < 0 and lo_ <= ent['tp']: close_at(j, ent['tp'] + (0 if mode == 'tv' else (b.ask_o.iloc[j] - b.bid_o.iloc[j]) / 2), 'tp')
        s = sig[i]
        if s != 0 and s != pos:
            if pos != 0: close_at(i + 1, px(i + 1, -pos), 'reverse')
            p = px(i + 1, s)
            sl = tp = None
            if exits and a is not None:
                c = m.c.iloc[i]
                sl = c - s * a[i] * slm; tp = c + s * a[i] * slm * tpm
            pos = s; ent = dict(t=t[i + 1], p=p, i=i + 1, sl=sl, tp=tp)
    return pd.DataFrame(trades)


def summary(tr):
    if tr.empty: return dict(n=0)
    p = tr.pnl.values
    pf = p[p > 0].sum() / -p[p < 0].sum() if (p < 0).any() else float('inf')
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 else float('nan')
    eq = np.cumsum(p); dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(p), pnl=round(float(p.sum()), 1), pf=round(float(pf), 3), win=round(float((p > 0).mean()), 3),
                avg=round(float(p.mean()), 2), t=round(float(t), 2), maxdd=round(dd, 1),
                worst=round(float(p.min()), 1), avg_hold_h=round(float((tr.exit - tr.entry).dt.total_seconds().mean() / 3600), 1))
