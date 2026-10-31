import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys2 as B, pandas as pd
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df); val=d.iloc[int(n*0.6):int(n*0.8)]
CORE=((7*60,11*60),(12*60+30,16*60))
for mult in (1.0,1.5,2.0):
    t=B.run(val,exit_mode="session",windows=CORE,gate_long=(val.c>val.pdc).values,gate_short=(val.c<val.pdc).values,cost_mult=mult)
    print("VALIDATION B+F+H @",mult,"x", B.met(t))
