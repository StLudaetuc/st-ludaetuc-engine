import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import feats as F, numpy as np, pandas as pd
d=F.build(); n=len(d); dev=d.iloc[:int(n*.6)]
t=F.trades(d,dev)
print('dev trades',len(t),'stopped',round(t.stopped.mean(),3),'avgR',round(t.R.mean(),3),'| stopped avgR',round(t[t.stopped].R.mean(),2),'others avgR',round(t[~t.stopped].R.mean(),2))
FEATS=['stop_atr','pb_ext','bbw_pct','atr_pct','spread_atr','rng_atr','body_frac','vol_ratio','pdc_dist_atr','day_used','mom8_atr','lm','dow','trend_al']
rows=[]
for f in FEATS:
    x=t[[f,'R','stopped']].dropna()
    if x[f].nunique()<=7: g=x[f]
    else: g=pd.qcut(x[f],5,labels=False,duplicates='drop')
    s=x.groupby(g).agg(n=('R','size'),stop=('stopped','mean'),avgR=('R','mean'),sd=('R','std'))
    s['t']=s.avgR/(s.sd/np.sqrt(s.n))
    # worst bucket vs rest: t-stat of dropping it
    w=s.avgR.idxmin(); keep=x[g!=w].R; drop=x[g==w].R
    diff=keep.mean()-drop.mean(); se=np.sqrt(keep.var()/len(keep)+drop.var()/len(drop))
    pf=lambda r: r[r>0].sum()/-r[r<0].sum()
    rows.append(dict(feature=f,worst_bucket=w,drop_n=len(drop),stop_rate_drop=round(drop.lt(-0.95).mean(),3),diff_t=round(diff/se,2),pf_keep=round(pf(keep),3),pf_all=round(pf(x.R),3)))
    print(f'\n{f}\n',s.round(3).to_string())
print('\n',pd.DataFrame(rows).sort_values('diff_t',ascending=False).to_string(index=False))
t.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'dev_trades_features.csv'),index=False)
