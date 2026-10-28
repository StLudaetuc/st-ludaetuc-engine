import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy as S, bbsys as B, vmi as V, pandas as pd, numpy as np
df=S.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'GBPUSD_M15_BA_OANDA_20211025_20261028.csv.gz'))
n=len(df); d=V.add_vmi(B.prep(df)); dev=d.iloc[:int(n*0.6)]
# gross (no spread, no slippage): mid fills
g=dev.copy()
for k in 'ohlc': g['bid_'+k]=g[k]; g['ask_'+k]=g[k]
up=(dev.vmi>=1.0).values; dn=(dev.vmi<=-1.0).values
for name,gate in (("with-flow pullback",(up,dn)),("fade migration",(dn,up))):
    for data,lab in ((dev,"net"),(g,"gross")):
        t=B.run(data,37,78,"rev",rr=1.5,gate=gate,trade_from=8*60,trade_to=16*60,slip=0 if lab=="gross" else 0.2e-4)
        print(f"{name:20s} {lab:5s}",B.met(t), "median stop pips", round(t.stop_pips.median(),1))
t=B.run(g,37,78,"rev",rr=1.5,trade_from=7*60,trade_to=17*60,slip=0); print("baseline rev gross",B.met(t))
