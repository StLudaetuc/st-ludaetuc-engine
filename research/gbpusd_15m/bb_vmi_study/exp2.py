import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys2 as B, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df); dev=d.iloc[:int(n*0.6)]
dup=(dev.c>dev.pdc).values; ddn=(dev.c<dev.pdc).values
CORE=((7*60,11*60),(12*60+30,16*60))
E={"B hold to session end":dict(exit_mode="session"),
   "B+F (core hours)":dict(exit_mode="session",windows=CORE),
   "B+H (daily trend)":dict(exit_mode="session",gate_long=dup,gate_short=ddn),
   "B+A (min stop 1.5ATR)":dict(exit_mode="session",min_stop_atr=1.5),
   "B+F+H [exploratory]":dict(exit_mode="session",windows=CORE,gate_long=dup,gate_short=ddn),
   "B+A+F+H [exploratory]":dict(exit_mode="session",windows=CORE,gate_long=dup,gate_short=ddn,min_stop_atr=1.5)}
rows=[]
for k,kw in E.items():
    t=B.run(dev,**kw); m=B.met(t)
    yr=t.groupby(t.t.dt.year).R.mean().round(3).to_dict()
    rows.append(dict(exp=k,**m,by_year=yr))
pd.set_option('display.width',250); print(pd.DataFrame(rows).to_string())
