import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df); dev=d.iloc[:int(n*0.6)].copy()
c,h,l,v=dev.c,dev.h,dev.l,dev.volume.astype(float)
dev['rng']=h-l; dev['slot']=dev.index.hour*4+dev.index.minute//15
slotmed=lambda s: s.groupby(dev.slot).transform(lambda x: x.shift(1).rolling(20,min_periods=10).median())
dev['rsa']=dev.rng.rolling(4).sum()/slotmed(dev.rng).rolling(4).sum()
dev['vsa']=v.rolling(4).sum()/slotmed(v).rolling(4).sum()
# daily context: FX day open (17:00 NY)
ny=dev.index.tz_convert('America/New_York'); key=(ny+pd.Timedelta(hours=7)).date
dev['dopen']=dev.groupby(key).o.transform('first')
dev['from_open_atr']=(c-dev.dopen)/dev.atr
dev['atr_pct']=dev.atr.rolling(96*20).rank(pct=True)
dev['dow']=dev.index.dayofweek
for mode in ("rev","brk"):
    t=B.run(dev,36,78,mode); t=t.set_index('t')
    # features at the SIGNAL bar = bar before entry
    sig=dev.index.get_indexer(t.index)-1; f=dev.iloc[sig]
    t['hr']=(t.lm//60).values; t['rsa']=f.rsa.values; t['vsa']=f.vsa.values; t['fo']=(f.from_open_atr.values*t.d.values); t['atrp']=f.atr_pct.values; t['dow']=f.dow.values
    print(f"\n=== {mode}: {B.met(t.reset_index())}")
    print(" by London hour:", t.groupby('hr').R.agg(['size','mean']).round(2).T.to_string())
    for col in ('rsa','vsa','fo','atrp'):
        q=pd.qcut(t[col],5,labels=False,duplicates='drop'); g=t.groupby(q).R.agg(['size','mean'])
        print(f" {col:5s} quintiles avgR:", ' '.join(f"{x:+.2f}" for x in g['mean']), " | edges", np.round(t[col].quantile([.2,.4,.6,.8]).values,2))
    print(" by direction:", t.groupby('d').R.agg(['size','mean']).round(3).to_dict())
