"""Python replica of STL GBPUSD 2m Candidate Generator v5.2 + entry-timing research."""
import numpy as np, pandas as pd

PIP = 0.0001


def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rma(x, n):
    out = np.full_like(x, np.nan)
    s = np.nanmean(x[:n]); out[n - 1] = s
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def features(df):
    o, h, l, c = (df[k].values.astype(float) for k in "ohlc")
    f = pd.DataFrame(index=df.index)
    f["t"] = df.t
    for k in "ohlc":
        f[k] = df[k].values
    f["ema20"] = ema(c, 20); f["ema50"] = ema(c, 50); f["ema200"] = ema(c, 200)
    d = np.diff(c, prepend=c[0])
    up = rma(np.maximum(d, 0), 14); dn = rma(np.maximum(-d, 0), 14)
    f["rsi"] = 100 - 100 / (1 + up / np.where(dn == 0, np.nan, dn))
    f.loc[dn == 0, "rsi"] = 100
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    f["atr"] = rma(tr, 14)
    m = ema(c, 12) - ema(c, 26); s = ema(m, 9)
    f["macd"] = m; f["macdh"] = m - s
    ph = np.roll(h, 1); pl = np.roll(l, 1)
    upm = h - ph; dnm = pl - l
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0); mdm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atr14 = rma(tr, 14)
    pdi = 100 * rma(pdm, 14) / atr14; mdi = 100 * rma(mdm, 14) / atr14
    dx = 100 * abs(pdi - mdi) / np.where(pdi + mdi == 0, np.nan, pdi + mdi)
    f["pdi"] = pdi; f["mdi"] = mdi
    f["adx"] = rma(np.nan_to_num(dx, nan=0.0), 14)
    bb = pd.Series(c).rolling(20).mean().values; sd = pd.Series(c).rolling(20).std(ddof=0).values
    f["bbu"] = bb + 2 * sd; f["bbl"] = bb - 2 * sd; f["bbb"] = bb
    f["rh12"] = pd.Series(h).rolling(12).max().values
    f["rl12"] = pd.Series(l).rolling(12).min().values
    rng = h - l
    f["body_r"] = np.where(rng > 0, abs(c - o) / rng, 0)
    f["uw_r"] = np.where(rng > 0, (h - np.maximum(o, c)) / rng, 0)
    f["lw_r"] = np.where(rng > 0, (np.minimum(o, c) - l) / rng, 0)
    # session VWAP (UTC day anchor)
    hlc3 = (h + l + c) / 3
    day = df.t.dt.floor("D").values
    vw = pd.Series(hlc3).groupby(day).cumsum() / pd.Series(np.ones_like(hlc3)).groupby(day).cumsum()
    f["vwap"] = vw.values
    # HTF 10m close & EMA50, last COMPLETED 10m bar (lookahead_off semantics)
    t10 = df.t.dt.floor("10min")
    s10 = pd.Series(c).groupby(t10.values).last()
    e10 = pd.Series(ema(s10.values, 50), index=s10.index)
    end10 = ((df.t + pd.Timedelta(minutes=2)).dt.floor("10min") != t10).values
    key = np.where(end10, t10.values, (t10 - pd.Timedelta(minutes=10)).values)
    f["htf_c"] = s10.reindex(key).values
    f["htf_e"] = e10.reindex(key).values
    f["hour"] = df.t.dt.hour
    f["dow"] = df.t.dt.dayofweek
    f["atr_pips"] = f.atr / PIP
    f["sep_atr"] = abs(f.ema20 - f.ema50) / f.atr
    f["atr_ratio"] = f.atr / f.atr.rolling(50).mean()
    f["d_ema20_atr"] = (f.c - f.ema20) / f.atr
    f["d_vwap_atr"] = (f.c - f.vwap) / f.atr
    f["rsi_min6"] = f.rsi.rolling(6).min(); f["rsi_max6"] = f.rsi.rolling(6).max()
    f["low_min3"] = pd.Series(l).rolling(3).min().values
    f["high_max3"] = pd.Series(h).rolling(3).max().values
    f["adx_slope"] = f.adx - f.adx.shift(3)
    f["ema50_slope_atr"] = (f.ema50 - f.ema50.shift(10)) / f.atr
    f["gap"] = df.t.diff().dt.total_seconds().fillna(120) / 60
    return f


def candidates_v52(f, min_adx=17.0, atr_lo=2.0, atr_hi=6.0, cooldown=5, maxday=18):
    """Exact v5.2 trigger logic. Returns DataFrame of signals (bar index, dir, model)."""
    c, h, l, o = f.c, f.h, f.l, f.o
    p = lambda s: s.shift(1)
    bull = (f.ema20 > f.ema50) & (c > f.ema200)
    bear = (f.ema20 < f.ema50) & (c < f.ema200)
    htfb = f.htf_c > f.htf_e; htfs = f.htf_c < f.htf_e
    volp = (f.atr_pips >= atr_lo) & (f.atr_pips <= atr_hi)
    rangem = (f.adx < 19) & (f.sep_atr < 0.40)
    lloc = (l <= f.ema20) | (p(l) <= p(f.ema20)) | (c < f.ema20 + f.atr * 0.45)
    sloc = (h >= f.ema20) | (p(h) >= p(f.ema20)) | (c > f.ema20 - f.atr * 0.45)
    lrec = (c > f.ema20) & (f.rsi > p(f.rsi)) & (f.macdh > p(f.macdh))
    srec = (c < f.ema20) & (f.rsi < p(f.rsi)) & (f.macdh < p(f.macdh))
    ltp = bull & htfb & lloc & lrec & f.rsi.between(39, 59)
    stp = bear & htfs & sloc & srec & f.rsi.between(41, 61)
    bexp = (c > p(h)) & (c > o) & (f.body_r >= 0.5) & (f.macdh > 0) & (f.macdh > p(f.macdh)) & (f.pdi > f.mdi)
    sexp = (c < p(l)) & (c < o) & (f.body_r >= 0.5) & (f.macdh < 0) & (f.macdh < p(f.macdh)) & (f.mdi > f.pdi)
    lmc = bull & htfb & bexp & f.rsi.between(51, 68) & (f.adx >= min_adx + 4)
    smc = bear & htfs & sexp & f.rsi.between(32, 49) & (f.adx >= min_adx + 4)
    lrr = rangem & (l <= f.bbl) & (f.lw_r >= 0.4) & (c > o) & (f.rsi <= 36) & (f.rsi > p(f.rsi))
    srr = rangem & (h >= f.bbu) & (f.uw_r >= 0.4) & (c < o) & (f.rsi >= 64) & (f.rsi < p(f.rsi))
    L = (ltp | lmc | lrr) & volp; S = (stp | smc | srr) & volp
    model = np.where(ltp | stp, "trend_pullback", np.where(lmc | smc, "momentum_continuation", "range_reversal"))
    regime = np.where(rangem, "range", np.where((f.adx >= min_adx) & (f.sep_atr >= 0.12) & (bull | bear), "trend", "mixed"))
    sig = []
    last = -10**9; day = None; nday = 0
    Lv, Sv = L.values, S.values
    tday = f.t.dt.floor("D").values
    for i in range(250, len(f)):
        if tday[i] != day:
            day = tday[i]; nday = 0
        if f.dow.values[i] >= 5:
            continue
        lv, sv = bool(Lv[i]), bool(Sv[i])
        if lv == sv:
            continue
        if i - last < cooldown or nday >= maxday:
            continue
        sig.append((i, 1 if lv else -1, model[i], regime[i]))
        last = i; nday += 1
    return pd.DataFrame(sig, columns=["i", "dir", "model", "regime"])


def simulate(f, i, d, entry, stop, tgt, start, max_bars=720):
    """Walk bars from `start` (inclusive); conservative: SL first if both touched.
    Returns (outcome, bars_held, mfe_R, mae_R, r_result)."""
    risk = abs(entry - stop)
    H, Lw, C = f.h.values, f.l.values, f.c.values
    mfe = mae = 0.0
    n = len(f)
    for j in range(start, min(n, start + max_bars)):
        if d == 1:
            mae = max(mae, (entry - Lw[j]) / risk); mfe = max(mfe, (H[j] - entry) / risk)
            if Lw[j] <= stop: return "loss", j - start + 1, mfe, mae, -1.0
            if H[j] >= tgt: return "win", j - start + 1, mfe, mae, (tgt - entry) / risk
        else:
            mae = max(mae, (H[j] - entry) / risk); mfe = max(mfe, (entry - Lw[j]) / risk)
            if H[j] >= stop: return "loss", j - start + 1, mfe, mae, -1.0
            if Lw[j] <= tgt: return "win", j - start + 1, mfe, mae, (entry - tgt) / risk
    j = min(n - 1, start + max_bars - 1)
    return "timeout", max_bars, mfe, mae, (C[j] - entry) * d / risk


def v52_plan(f, i, d, stop_mult=1.5, rr=1.6, buf=0.6, entry=None, struct_lb=12):
    c = f.c.values[i] if entry is None else entry
    atr = f.atr.values[i]
    if d == 1:
        st = min(c - atr * stop_mult, f.rl12.values[i] - buf * PIP)
        return c, st, c + (c - st) * rr
    st = max(c + atr * stop_mult, f.rh12.values[i] + buf * PIP)
    return c, st, c - (st - c) * rr


COST_PIPS = 1.4  # spread + slippage round trip, OANDA GBPUSD practice approx


def stats(res, label=""):
    r = res[res.outcome != "skip"]
    n = len(r)
    if n == 0:
        return {"label": label, "n": 0}
    wins = (r.outcome == "win").sum()
    netR = r.r - r.cost_r
    gp = netR[netR > 0].sum(); gl = -netR[netR < 0].sum()
    return {"label": label, "n": n, "win_rate": round(wins / n, 3), "avg_R_gross": round(r.r.mean(), 3),
            "avg_R_net": round(netR.mean(), 3), "total_R_net": round(netR.sum(), 1),
            "profit_factor": round(gp / gl, 2) if gl > 0 else None,
            "early_stop_pct": round(((r.outcome == "loss") & (r.bars <= 5)).mean(), 3),
            "median_stop_pips": round(r.stop_pips.median(), 1)}
