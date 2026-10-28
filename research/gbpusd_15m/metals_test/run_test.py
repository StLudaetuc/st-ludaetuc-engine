"""Pre-registered gold/silver test (see PREREGISTRATION.md). Rules frozen; one run per metal on full data."""
import os,sys,json,numpy as np,pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(H)
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'timeframe_study'))
import strategy as S, tf as T, bbsys2 as B
THR=3.544
pf=lambda r: round(float(r[r>0].sum()/-r[r<0].sum()),3)
out={}
for name,f in (('XAUUSD','XAUUSD_M15_BA_OANDA_20211025_20261028.csv.gz'),('XAGUSD','XAGUSD_M15_BA_OANDA_20211025_20261028.csv.gz')):
    d=B.prep(S.load(os.path.join(R,'data',f)))
    slip=0.15*float((d.ask_c-d.bid_c).median())
    res={'slippage_price_units':slip,'median_spread_over_atr':round(float(((d.ask_c-d.bid_c)/d.atr).median()),3)}
    print(f"\n{name}: bars={len(d)} slip={slip:.5f} spread/ATR={res['median_spread_over_atr']}")
    for m in (1.0,1.5,2.0):
        t=B.run(d,exit_mode="session",windows=T.CORE,gate_long=(d.c>d.pdc).values,gate_short=(d.c<d.pdc).values,cost_mult=m,slip=slip)
        sig=d.index.get_indexer(t.t)-1
        ext=(d.c.values[sig]-d.pdc.values[sig])*t.d.values/d.atr.values[sig]
        for lab,mask in (('BFH',np.ones(len(t),bool)),('BFH+F2',ext<=THR)):
            r=t.R[mask]; se=r.std()/np.sqrt(len(r))
            by={int(y):round(float(v),3) for y,v in r.groupby(t.t[mask].dt.year).mean().items()}
            res[f'{lab}_{m}x']=dict(n=int(len(r)),avgR=round(float(r.mean()),4),se=round(float(se),4),t=round(float(r.mean()/se),2),pf=pf(r),by_year=by)
            print(f"  {lab:7s} {m}x n={len(r):4d} avgR={r.mean():+.3f} se={se:.3f} t={r.mean()/se:+.2f} PF={pf(r)}")
    f2=res['BFH+F2_1.0x']; sgn=np.sign(f2['avgR'])
    crit={'1_pf_1.5x_ge_1':res['BFH+F2_1.5x']['pf']>=1.0,'2_avgR_pos_t_ge_2.3':f2['avgR']>0 and f2['t']>=2.3,
          '3_same_sign_4of6_years':sum(np.sign(v)==sgn for v in f2['by_year'].values())>=4 and sgn>0}
    crit={k:bool(v) for k,v in crit.items()}; res['criteria']=crit; res['PASS']=all(crit.values()); out[name]=res
    print(' ',crit,'PASS' if res['PASS'] else 'FAIL','| by year F2 1x',f2['by_year'])
json.dump(out,open(os.path.join(H,'result.json'),'w'),indent=1)
