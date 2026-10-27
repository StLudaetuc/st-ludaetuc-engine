import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
PIP={'EURUSD':1e-4,'AUDUSD':1e-4,'USDCAD':1e-4,'GBPUSD':1e-4,'USDJPY':1e-2}
def load(p):
    d=pd.read_pickle(f'data/{p}_5m.pkl'); d['c']=d.pc.shift(-1); d=d.dropna(); d=d[d.t.dt.dayofweek<5].reset_index(drop=True)
    d['ac']=d.c.diff().abs()/PIP[p]; return d
def label(d,p,stopm=3.0,rr=1.5,mx=48):
    c=d.c.values; sd=d.ac.mean()*stopm*PIP[p]; n=len(c); W={1:np.full(n,np.nan),-1:np.full(n,np.nan)}
    for dr in (1,-1):
        for i in range(n-mx):
            e=c[i]; st=e-dr*sd; tg=e+dr*sd*rr
            for j in range(i+1,i+mx+1):
                if (c[j]-st)*dr<=0: W[dr][i]=0;break
                if (c[j]-tg)*dr>=0: W[dr][i]=1;break
    return W
SPLIT=pd.Timestamp('2026-08-15',tz='UTC')
for p in sys.argv[1:]:
    d=load(p); W=label(d,p); c=d.c
    mid=(d.bbu+d.bbl)/2; half=(d.bbu-d.bbl)/2; z=(c-mid)/half
    slope=c-c.shift(24); ema=c.ewm(span=50).mean(); hi=c.rolling(12).max().shift(1); lo=c.rolling(12).min().shift(1)
    rows=[]
    for dr in (1,-1):
        g=pd.DataFrame({'t':d.t,'w':W[dr],'z':z*dr,'tr':(slope*dr>0)&((c-ema)*dr>0),'brk':(c-hi>0) if dr==1 else (lo-c>0),'rsi':d.rsi if dr==1 else 100-d.rsi,'h':d.t.dt.hour}); rows.append(g)
    D=pd.concat(rows).dropna(); D['oos']=D.t>=SPLIT
    def tab(m,name):
        g=D[m]; print(f'{p} {name:45s} n={len(g):6d} wr={g.w.mean():.3f} IS={g[~g.oos].w.mean():.3f} OOS={g[g.oos].w.mean():.3f}')
    tab(D.w.notna(),'all')
    tab(D.tr,'trend aligned'); tab(~D.tr,'not aligned')
    tab(D.tr&(D.z<-0.8),'trend + stretched pullback (z<-0.8)')
    tab(D.tr&(D.z>1.0),'trend + chasing (z>1)')
    tab(D.tr&D.brk,'trend + 12-bar breakout')
    tab(D.tr&D.brk&(D.rsi.between(55,70)),'trend + breakout + RSI 55-70')
    tab(~D.tr&(D.z<-1.0),'counter-trend fade z<-1')
    tab(D.tr&(D.z.between(-0.3,0.5)),'trend + mid band')
