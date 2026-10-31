import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys2 as B, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df); dev=d.iloc[:int(n*0.6)]
up4=(dev.trend4h>0).values; dn4=(dev.trend4h<0).values
sq=(dev.bbw_pct.shift(1)<=0.2).rolling(8).max().fillna(0).astype(bool).values   # band was in bottom 20% within last 2h
dup=(dev.c>dev.pdc).values; ddn=(dev.c<dev.pdc).values
CORE=((7*60,11*60),(12*60+30,16*60)); NONEWS=((7*60,12*60),(14*60,17*60))
E={"base (brk 36/78)":{},
   "A wider: min stop 1.5ATR, 2R":dict(min_stop_atr=1.5,rr=2.0),
   "B hold to session end":dict(exit_mode="session"),
   "C trail 2ATR":dict(exit_mode="trail",trail_atr=2.0),
   "D 4H trend only":dict(gate_long=up4,gate_short=dn4),
   "E squeeze breakouts only":dict(gate_long=sq,gate_short=sq),
   "F core liquid hours":dict(windows=CORE),
   "G skip 12-14 London":dict(windows=NONEWS),
   "H daily trend only":dict(gate_long=dup,gate_short=ddn)}
rows=[]
for k,kw in E.items():
    rows.append(dict(exp=k,**B.met(B.run(dev,**kw))))
print(pd.DataFrame(rows).to_string())
