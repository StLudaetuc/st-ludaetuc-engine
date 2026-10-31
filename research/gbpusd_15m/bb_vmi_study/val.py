import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, vmi as V, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=V.add_vmi(B.prep(df)); val=d.iloc[int(n*0.6):int(n*0.8)]
up=(val.vmi>=1).values; dn=(val.vmi<=-1).values
rows=[]
for name,kw in (("rev",dict(mode="rev")),("brk",dict(mode="brk")),("vmi_fade",dict(mode="rev",gate=(dn,up),trade_from=480,trade_to=960))):
    for hi in (77,78,79):
        for lo in range(34,41):
            m=B.met(B.run(val,lo,hi,**kw)); rows.append(dict(sys=name,low=lo,high=hi,**m))
r=pd.DataFrame(rows); r.to_csv('grid_validation.csv',index=False)
print(r.groupby('sys').agg(n_min=('n','min'),n_max=('n','max'),pf_min=('pf','min'),pf_max=('pf','max'),avgR_min=('avgR','min'),avgR_max=('avgR','max')))
print("configs with PF>=1:", (r.pf>=1).sum(), "of", len(r))
