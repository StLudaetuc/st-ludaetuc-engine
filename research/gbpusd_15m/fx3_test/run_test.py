"""Pre-registered AUD/USD, USD/CAD, USD/JPY test (see PREREGISTRATION.md). Rules frozen; one run per pair. Also pools all 7 markets."""
import os,sys,json,numpy as np,pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(H)
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'timeframe_study'))
import strategy as S, tf as T, bbsys2 as B
THR=3.544
pf=lambda r: round(float(r[r>0].sum()/-r[r<0].sum()),3)
def trades(f,slip,m):
    d=B.prep(S.load(os.path.join(R,'data',f)))
    t=B.run(d,exit_mode="session",windows=T.CORE,gate_long=(d.c>d.pdc).values,gate_short=(d.c<d.pdc).values,cost_mult=m,slip=slip)
    sig=d.index.get_indexer(t.t)-1
    t['ext']=(d.c.values[sig]-d.pdc.values[sig])*t.d.values/d.atr.values[sig]
    return t,d
out={}
for name,f,slip in (('AUDUSD','AUDUSD_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-4),('USDCAD','USDCAD_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-4),('USDJPY','USDJPY_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-2)):
    res={}
    for m in (1.0,1.5,2.0):
        t,d=trades(f,slip,m)
        if m==1.0: res['median_spread_over_atr']=round(float(((d.ask_c-d.bid_c)/d.atr).median()),3); print(f"\n{name}: bars={len(d)} spread/ATR={res['median_spread_over_atr']}")
        for lab,mask in (('BFH',np.ones(len(t),bool)),('BFH+F2',(t.ext<=THR).values)):
            r=t.R[mask]; se=r.std()/np.sqrt(len(r))
            res[f'{lab}_{m}x']=dict(n=int(len(r)),avgR=round(float(r.mean()),4),se=round(float(se),4),t=round(float(r.mean()/se),2),pf=pf(r),
                by_year={int(y):round(float(v),3) for y,v in r.groupby(t.t[mask].dt.year).mean().items()})
            print(f"  {lab:7s} {m}x n={len(r):4d} avgR={r.mean():+.3f} se={se:.3f} t={r.mean()/se:+.2f} PF={pf(r)}")
    f2=res['BFH+F2_1.0x']
    crit={'1_pf_1.5x_ge_1':bool(res['BFH+F2_1.5x']['pf']>=1.0),'2_avgR_pos_t_ge_2.4':bool(f2['avgR']>0 and f2['t']>=2.4),
          '3_pos_in_4of6_years':bool(sum(v>0 for v in f2['by_year'].values())>=4)}
    res['criteria']=crit; res['PASS']=all(crit.values()); out[name]=res
    print(' ',crit,'PASS' if res['PASS'] else 'FAIL','| by year F2 1x',f2['by_year'])
# pooled across all 7 markets, B+F+H+F2 at 1x and 1.5x (summary only, not pass/fail)
ALL=[('GBPUSD(M5->15m)',None,0.2e-4),('EURUSD','EURUSD_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-4),('AUDUSD','AUDUSD_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-4),
     ('USDCAD','USDCAD_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-4),('USDJPY','USDJPY_M15_BA_OANDA_20211025_20261028.csv.gz',0.2e-2)]
pool={}
for m in (1.0,1.5):
    rs=[]
    for name,f,slip in ALL:
        if f is None:
            d=B.prep(T.resample(S.load(os.path.join(R,'data','GBPUSD_M5_BA_OANDA_20211025_20261028.csv.gz')),'15min'))
            t=B.run(d,exit_mode="session",windows=T.CORE,gate_long=(d.c>d.pdc).values,gate_short=(d.c<d.pdc).values,cost_mult=m,slip=slip)
            sig=d.index.get_indexer(t.t)-1; t['ext']=(d.c.values[sig]-d.pdc.values[sig])*t.d.values/d.atr.values[sig]
        else: t,_=trades(f,slip,m)
        rs.append(t.R[t.ext<=THR])
    for name,f in (('XAUUSD','XAUUSD_M15_BA_OANDA_20211025_20261028.csv.gz'),('XAGUSD','XAGUSD_M15_BA_OANDA_20211025_20261028.csv.gz')):
        d0=S.load(os.path.join(R,'data',f)); t,_=trades(f,0.15*float((d0.ask_c-d0.bid_c).median()),m); rs.append(t.R[t.ext<=THR])
    r=pd.concat(rs); se=r.std()/np.sqrt(len(r))
    pool[f'{m}x']=dict(n=int(len(r)),avgR=round(float(r.mean()),4),se=round(float(se),4),t=round(float(r.mean()/se),2),pf=pf(r))
    print(f"\nPOOLED 7 markets B+F+H+F2 {m}x: n={len(r)} avgR={r.mean():+.4f} se={se:.4f} t={r.mean()/se:+.2f} PF={pf(r)}")
out['pooled_7_markets_F2']=pool
json.dump(out,open(os.path.join(H,'result.json'),'w'),indent=1)
