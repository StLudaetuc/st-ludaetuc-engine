"""Pre-registered EUR/USD test (see PREREGISTRATION.md). Rules frozen from GBP/USD; one run on the full data."""
import os,sys,json,numpy as np,pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(H)
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'timeframe_study')); sys.path.insert(0,os.path.join(R,'stopout_study'))
import strategy as S, tf as T, bbsys2 as B
THR=3.544
d=B.prep(S.load(os.path.join(R,'data','EURUSD_M15_BA_OANDA_20211025_20261028.csv.gz')))
pf=lambda r: round(float(r[r>0].sum()/-r[r<0].sum()),3)
res={}
for m in (1.0,1.5,2.0):
    t=B.run(d,exit_mode="session",windows=T.CORE,gate_long=(d.c>d.pdc).values,gate_short=(d.c<d.pdc).values,cost_mult=m)
    sig=d.index.get_indexer(t.t)-1
    ext=((d.c.values[sig]-d.pdc.values[sig])*t.d.values/d.atr.values[sig])
    for lab,mask in (('BFH',np.ones(len(t),bool)),('BFH+F2',ext<=THR)):
        r=t.R[mask]; se=r.std()/np.sqrt(len(r))
        res[f'{lab}_{m}x']=dict(n=int(len(r)),avgR=round(float(r.mean()),4),se=round(float(se),4),t=round(float(r.mean()/se),2),pf=pf(r),
            stop_rate=round(float((r<=-0.95).mean()),3),by_year={int(y):round(float(v),3) for y,v in r.groupby(t.t[mask].dt.year).mean().items()})
        print(f"{lab:7s} {m}x n={len(r):4d} avgR={r.mean():+.3f} se={se:.3f} t={r.mean()/se:+.2f} PF={pf(r)}")
f=res['BFH+F2_1.0x']; b=res['BFH_1.0x']
crit={'1_pf_1.5x_ge_1':res['BFH+F2_1.5x']['pf']>=1.0,'2_avgR_pos_t_ge_2':f['avgR']>0 and f['t']>=2.0,'3_F2_beats_BFH':f['avgR']>b['avgR']}
res['criteria']=crit; res['PASS']=all(crit.values()); print(crit,'PASS' if res['PASS'] else 'FAIL')
print('by year F2 1x',f['by_year'])
json.dump(res,open(os.path.join(H,'result.json'),'w'),indent=1)
