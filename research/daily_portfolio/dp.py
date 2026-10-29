"""Daily-bar TREND / CARRY portfolio backtest (see PREREGISTRATION.md). Research only."""
import os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
DAILY = os.environ.get('DAILY_DIR', os.path.join(H, 'data', 'daily'))
AREA = dict(USD='USA', GBP='GBR', JPY='JPN', AUD='AUS', CAD='CAN', CHF='CHE', NZD='NZL', NOK='NOR', SEK='SWE',
            EUR='EA20', MXN='MEX', ZAR='ZAF', PLN='POL', HUF='HUN', CZK='CZE', DKK='DNK')
INDICES = ['AU200_AUD','CH20_CHF','CN50_USD','DE30_EUR','ESPIX_EUR','EU50_EUR','FR40_EUR','JP225_USD','NAS100_USD','NL25_EUR',
           'SPX500_USD','UK100_GBP','US2000_USD','US30_USD','JP225Y_JPY','SG30_SGD','HK33_HKD','CHINAH_HKD']
BONDS = ['DE10YB_EUR','UK10YB_GBP','USB02Y_USD','USB05Y_USD','USB10Y_USD','USB30Y_USD']
COMMOD = ['BCO_USD','WTICO_USD','NATGAS_USD','CORN_USD','WHEAT_USD','SOYBN_USD','SUGAR_USD']
METALS = ['XAU_USD','XAG_USD','XPT_USD','XPD_USD','XCU_USD']

def load_rates():
    r = pd.read_csv(os.path.join(H, 'data', 'oecd_ir3m.csv.gz'))
    r = r.pivot_table(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
    r.index = pd.to_datetime(r.index) + pd.offsets.MonthBegin(1)  # usable from 1st of following month
    return r / 100.0

def universe():
    ins = json.load(open(os.path.join(H, 'instruments.json')))
    fx = [i['name'] for i in ins if i['type'] == 'CURRENCY' and all(c in AREA for c in i['name'].split('_'))]
    return dict(FX=sorted(fx), METALS=METALS, INDICES=INDICES, BONDS=BONDS, COMMOD=COMMOD)

def load_panel(names):
    o, c, so = {}, {}, {}
    for n in names:
        d = pd.read_csv(os.path.join(DAILY, f'{n}.csv.gz'))
        # 09:00 New York-aligned daily candles: label = NY date of the candle open (open = 09:00 that day)
        t = pd.to_datetime(d.time_utc, utc=True).dt.tz_convert('America/New_York').dt.normalize().dt.tz_localize(None)
        d.index = t
        d = d[~d.index.duplicated(keep='last')]
        mo = (d.bid_o + d.ask_o) / 2; mc = (d.bid_c + d.ask_c) / 2
        o[n] = mo; c[n] = mc; so[n] = (d.ask_o - d.bid_o) / mo
    O, C, S = pd.DataFrame(o), pd.DataFrame(c), pd.DataFrame(so)
    return O.sort_index(), C.sort_index(), S.sort_index()

def fin_rates(names, idx, rates, kind):
    """annual financing rate for a +1 long and for a +1 short (positive = received)."""
    rr = rates.reindex(idx, method='ffill')
    L, Sh = {}, {}
    for n in names:
        b, q = n.split('_')
        if kind[n] == 'FX':
            rb, rq = rr[AREA[b]], rr[AREA[q]]
            L[n] = (rb - rq) - 0.01; Sh[n] = (rq - rb) - 0.01
        else:
            rq = rr[AREA[q]] if q in AREA else rr['USA']
            L[n] = -(rq + 0.025); Sh[n] = (rq - 0.025)
    return pd.DataFrame(L), pd.DataFrame(Sh)

def run(names, kind, signal_fn, cost_mult=1.0, vol_target=0.10, buffer=0.25, rates=None):
    O, C, S = load_panel(names)
    idx = C.index
    ret_cc = C.pct_change()
    vol = ret_cc.ewm(span=60, min_periods=40).std() * np.sqrt(252)
    sig = signal_fn(C, idx)                       # signal known at close t
    navail = sig.notna().sum(axis=1).clip(lower=1)
    tgt = (sig * vol_target / vol).div(np.sqrt(navail), axis=0)
    tgt = tgt.replace([np.inf, -np.inf], np.nan)
    monday = pd.Series(idx.dayofweek == 0, index=idx)
    finL, finS = fin_rates(names, idx, rates, kind)
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(1).values
    W = np.zeros(len(names)); pnl = np.zeros(len(idx)); cost = np.zeros(len(idx)); fin = np.zeros(len(idx))
    pos_hist = np.zeros((len(idx), len(names))); inst_pnl = np.zeros((len(idx), len(names)))
    Cv, Ov, Sv, Tv = C.values, O.values, S.values, tgt.shift(1).values   # trade at open t on signal from close t-1
    fL, fS = finL.values, finS.values
    for t in range(1, len(idx)):
        prev_c, o_t, c_t = Cv[t-1], Ov[t], Cv[t]
        # P&L close(t-1)->open(t) with old weights
        r1 = np.where(np.isfinite(prev_c) & np.isfinite(o_t), o_t / prev_c - 1, 0.0)
        ip = W * r1
        # rebalance at open on Mondays (or first day of data)
        if monday.iloc[t]:
            T = np.nan_to_num(Tv[t])
            chg = np.abs(T - W) > buffer * np.abs(T)
            chg &= np.isfinite(o_t)
            newW = np.where(chg, T, W)
            sp = np.nan_to_num(Sv[t], nan=0.0)
            c_cost = np.abs(newW - W) * sp / 2 * 1.1 * cost_mult
            cost[t] = c_cost.sum(); ip -= c_cost
            W = np.where(np.isfinite(o_t), newW, 0.0)
        r2 = np.where(np.isfinite(o_t) & np.isfinite(c_t), c_t / o_t - 1, 0.0)
        ip += W * r2
        f = np.where(W > 0, W * np.nan_to_num(fL[t]), -W * np.nan_to_num(fS[t])) * days[t] / 365.0
        if cost_mult != 1.0:  # stress: financing markup also scaled
            f = f - np.abs(W) * (cost_mult - 1) * np.where([kind[n] == 'FX' for n in names], 0.01, 0.025) * days[t] / 365.0
        ip += f; fin[t] = f.sum()
        inst_pnl[t] = ip; pnl[t] = ip.sum(); pos_hist[t] = W
    out = pd.DataFrame(dict(pnl=pnl, cost=cost, fin=fin), index=idx)
    return out, pd.DataFrame(pos_hist, index=idx, columns=names), pd.DataFrame(inst_pnl, index=idx, columns=names)

def tsmom(C, idx):
    s = sum(np.sign(C / C.shift(L) - 1) for L in (63, 126, 252)) / 3.0
    return s.where(C.shift(252).notna())

def make_carry(names, rates):
    def f(C, idx):
        rr = rates.reindex(idx, method='ffill'); s = {}
        for n in names:
            b, q = n.split('_'); d = rr[AREA[b]] - rr[AREA[q]]
            s[n] = np.where(d.abs() > 0.005, np.sign(d), 0.0)
        return pd.DataFrame(s, index=idx).where(C.notna())
    return f

def metrics(df, pos, ip, start=None, end=None):
    d = df.loc[start:end]; r = d.pnl
    m = r.resample('ME').sum()
    pf = lambda x: float(x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else float('inf')
    eq = r.cumsum(); dd = float((eq.cummax() - eq).max())
    # trade PF: spans of constant position sign per instrument
    P = pos.loc[start:end]; I = ip.loc[start:end]; trades = []
    for n in P.columns:
        sg = np.sign(P[n].values); x = I[n].values; cur = 0.0; s0 = 0
        for k in range(len(sg)):
            if k > 0 and sg[k] != sg[k-1]:
                if sg[k-1] != 0: trades.append(cur)
                cur = 0.0
            cur += x[k]
        if len(sg) and sg[-1] != 0: trades.append(cur)
    tr = np.array(trades)
    ann = r.mean() * 252; vol = r.std() * np.sqrt(252)
    return dict(ann_ret=round(ann*100, 2), ann_vol=round(vol*100, 2), sharpe=round(ann/vol, 2) if vol else None,
                pf_daily=round(pf(r), 3), pf_monthly=round(pf(m), 3), trades=int(len(tr)), pf_trades=round(pf(tr), 3) if len(tr) else None,
                win_trades=round(float((tr > 0).mean()), 3) if len(tr) else None, maxdd_pct=round(dd*100, 1),
                cost_pa=round(d.cost.sum()/max(len(d)/252,1)*100, 2), fin_pa=round(d.fin.sum()/max(len(d)/252,1)*100, 2))
