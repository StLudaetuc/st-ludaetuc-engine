import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, vmi as V, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=V.add_vmi(B.prep(df)); dev=d.iloc[:int(n*0.6)]
up=(dev.vmi>=1.0).values; dn=(dev.vmi<=-1.0).values
print("bars with VMI>=1:",up.mean().round(3)," <=-1:",dn.mean().round(3))
rows=[]
for rr in (1.5,2.0):
  for hi in (77,78,79):
    for lo in range(34,41):
        t=B.run(dev,lo,hi,"rev",rr=rr,gate=(up,dn),trade_from=8*60,trade_to=16*60)
        rows.append(dict(rr=rr,low=lo,high=hi,**B.met(t)))
print(pd.DataFrame(rows).to_string())
