import sys,json; import os; H=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(H)); sys.path.insert(0,H)
import strategy as S, tf as T
m5=S.load(os.path.join(os.path.dirname(H),'data','GBPUSD_M5_BA_OANDA_20211025_20261028.csv.gz')); m1=S.load(os.path.join(os.path.dirname(H),'data','GBPUSD_M1_BA_OANDA_20241028_20261028.csv.gz'))
out=[]
def show(lab,x,period):
    res,sp,info=T.evaluate(x,lab)
    print(f"\n{lab} [{period}]: {info}  median spread/ATR={sp}")
    for (name,m),v in res.items():
        print(f"   {name:11s} {m}x  n={v['n']:4d} avgR={v['avgR']:+.3f} se={v['se']:.3f} PF={v['pf']}")
        out.append(dict(tf=lab,period=period,split=name,cost=m,n=v['n'],avgR=round(v['avgR'],4),se=round(v['se'],4),pf=v['pf'],spread_atr=sp,start=info[0],end=info[1],bars=info[2]))
for lab,rule in (("5m",None),("10m","10min"),("15m","15min"),("30m","30min"),("45m","45min"),("1h","60min")):
    show(lab, m5 if rule is None else T.resample(m5,rule), "5y")
for lab,rule in (("3m","3min"),("5m","5min"),("15m","15min")):
    show(lab, T.resample(m1,rule), "2y")
json.dump(out,open(os.path.join(H,'alltfs.json'),'w'),indent=1)
