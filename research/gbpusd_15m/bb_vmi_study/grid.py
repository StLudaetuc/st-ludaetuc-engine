import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, pandas as pd
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=B.prep(df); dev=d.iloc[:int(n*0.6)]
rows=[]
for mode in ("rev","brk"):
    for hi in (77,78,79):
        for lo in range(34,41):
            m=B.met(B.run(dev,lo,hi,mode)); rows.append(dict(mode=mode,low=lo,high=hi,**m))
r=pd.DataFrame(rows); print(r.to_string()); r.to_csv('grid_dev_baseline.csv',index=False)
