import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
PIP={'EURUSD':1e-4,'AUDUSD':1e-4,'USDCAD':1e-4,'GBPUSD':1e-4,'USDJPY':1e-2}
out={}
for p in PIP:
    d=pd.read_pickle(f'data/{p}_5m.pkl')
    d['c']=d.pc.shift(-1)              # close of this candle = next row's previous close
    d=d[d.t.dt.dayofweek<5].dropna()
    d['ac']=d.c.diff().abs()/PIP[p]
    d=d[(d.t.diff().dt.total_seconds()==300)]
    d['h']=d.t.dt.hour
    hv=d.groupby('h').ac.mean()
    pk=hv.max()
    core=[h for h in range(24) if hv[h]>=0.55*pk]
    # mean reversion test on closes: stretched (beyond band) vs chasing; label TP 1.5R/SL with stop = 3*mean|dc| pips, close path, 48 bars
    c=d.c.values; mid=(d.bbu+d.bbl).values/2; half=(d.bbu-d.bbl).values/2
    z=np.where(half>0,(c-mid)/half,np.nan)
    sd=d.ac.mean()*3*PIP[p]
    res=[]
    for i in range(0,len(c)-48,1):
        if np.isnan(z[i]): continue
        for dr in (1,-1):
            zz=z[i]*dr; e=c[i]; st=e-dr*sd; tg=e+dr*sd*1.5; w=np.nan
            for j in range(i+1,i+49):
                if (c[j]-st)*dr<=0: w=0;break
                if (c[j]-tg)*dr>=0: w=1;break
            res.append((zz,w))
    r=pd.DataFrame(res,columns=['z','w']).dropna()
    bins=pd.cut(r.z,[-9,-1,-0.3,0.3,1,9])
    wr=r.groupby(bins).w.mean()
    out[p]=dict(mean_abs_5m_pips=round(d.ac.mean(),2),core_hours=f'{min(core):02d}-{max(core)+1:02d}',core=core,
                stretched_wr=round(wr.iloc[0],3),chasing_wr=round(wr.iloc[-1],3),spread=round(wr.iloc[0]-wr.iloc[-1],3))
    print(p,{k:v for k,v in out[p].items() if k!='core'}, 'hourly:',' '.join(f'{h}:{hv[h]:.1f}' for h in range(24)))
pd.to_pickle(out,'data/pairprof.pkl')
