exec(open('pairmodel.py').read().split("for p in sys.argv[1:]:")[0])
import pickle
res={}
for p in ['GBPUSD','EURUSD','AUDUSD','USDCAD','USDJPY']:
    d=load(p); c=d.c
    mid=(d.bbu+d.bbl)/2; half=(d.bbu-d.bbl)/2; z=(c-mid)/half
    slope=c-c.shift(24); ema=c.ewm(span=50).mean(); hi=c.rolling(12).max().shift(1); lo=c.rolling(12).min().shift(1)
    for stopm,rr in ((3.0,1.5),(4.5,1.5)):
        W=label(d,p,stopm,rr)
        rows=[]
        for dr in (1,-1):
            rows.append(pd.DataFrame({'t':d.t,'w':W[dr],'z':z*dr,'tr':(slope*dr>0)&((c-ema)*dr>0),'brk':(c-hi>0) if dr==1 else (lo-c>0),'rsi':d.rsi if dr==1 else 100-d.rsi,'h':d.t.dt.hour}))
        D=pd.concat(rows).dropna(); D['oos']=D.t>=SPLIT
        m = (D.tr&D.brk&D.rsi.between(55,70)) if p=='USDJPY' else ((~D.tr)&(D.z<-1.0))
        g=D[m]
        hb=g.groupby(pd.cut(g.h,[-1,5,11,16,20,23])).agg(n=('w','size'),wr=('w','mean'))
        print(p,'stop x',stopm, 'setup wr',round(g.w.mean(),3),'| by UTC hour block:',' '.join(f'{iv}:{r.n}/{r.wr:.3f}' for iv,r in hb.iterrows()))
