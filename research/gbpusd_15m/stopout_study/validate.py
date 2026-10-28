"""Two filters chosen on development only (monotone, with a market reason, no middle-bucket picks), then validation once.
  F1 trend_al : take only trades aligned with the 4H structure (trend4h * direction == +1)
  F2 not_ext  : skip trades where price is already > DEV 80th-percentile ATRs beyond the prior daily close in the trade direction
"""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import feats as F, numpy as np, pandas as pd, json
d=F.build(); n=len(d)
dev=F.trades(d,d.iloc[:int(n*.6)]); val=F.trades(d,d.iloc[int(n*.6):int(n*.8)])
thr=float(dev.pdc_dist_atr.quantile(0.8)); print('F2 threshold (dev 80th pct):',round(thr,3),'ATR')
pf=lambda r: round(float(r[r>0].sum()/-r[r<0].sum()),3)
def rep(t,name):
    out={}
    for lab,m in (('base',np.ones(len(t),bool)),('F1 trend_al',t.trend_al==1),('F2 not_ext',t.pdc_dist_atr<=thr),('F1+F2',(t.trend_al==1)&(t.pdc_dist_atr<=thr))):
        r=t.R[m]; out[lab]=dict(n=int(m.sum()),stop=round(float(t.stopped[m].mean()),3),avgR=round(float(r.mean()),3),se=round(float(r.std()/np.sqrt(len(r))),3),pf=pf(r))
        print(f'{name:10s} {lab:12s} n={m.sum():4d} stop%={t.stopped[m].mean():.3f} avgR={r.mean():+.3f} se={r.std()/np.sqrt(len(r)):.3f} PF={pf(r)}')
    return out
res={'threshold_atr':thr,'development':rep(dev,'dev'),'validation':rep(val,'val')}
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'validation_result.json'),'w'),indent=1)
