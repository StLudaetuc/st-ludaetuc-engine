import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df).iloc[:int(n*0.6)].copy()
lon=d.index.tz_convert('Europe/London'); d['date']=lon.date; d['hm']=lon.hour*100+lon.minute
lo=d[d.hm==700].set_index('date').o   # London open price
d['lopen']=d.date.map(lo)
d['mv']=(d.c-d.lopen)/d.atr           # London move so far, ATR units
H=8; d['fwd']=(d.c.shift(-H)-d.c)/d.atr
spread_atr=((d.ask_c-d.bid_c)/d.atr).median()
print("median spread in ATR units:",round(spread_atr,3), "(round-trip cost ~ spread+slippage)")
res=[]
for hm,g in d[(d.hm>=800)&(d.hm<=1600)&(d.hm%100==0)].groupby('hm'):
    g=g.dropna(subset=['mv','fwd'])
    for lab,m in (("up>2ATR",g.mv>2),("dn<-2ATR",g.mv<-2)):
        x=g[m].fwd; res.append(dict(hm=hm,cond=lab,n=len(x),mean_fwd_atr=round(x.mean(),3),t=round(x.mean()/(x.std()/np.sqrt(len(x))),2) if len(x)>5 else None))
print(pd.DataFrame(res).to_string())
