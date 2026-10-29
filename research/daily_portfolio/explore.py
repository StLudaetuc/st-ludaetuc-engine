"""EXPLORATORY (not pre-registered): trend variants on higher-volatility groups, selected on DEVELOPMENT only, then validation; holdout reported but not used to select."""
import os,sys,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dp, numpy as np, pandas as pd
rates=dp.load_rates(); U=dp.universe()
kind={**{n:'FX' for n in U['FX']},**{n:'CFD' for g in ('METALS','INDICES','BONDS','COMMOD') for n in U[g]}}
def sig_lb(lbs):
    def f(C,idx): return (sum(np.sign(C/C.shift(L)-1) for L in lbs)/len(lbs)).where(C.shift(max(lbs)).notna())
    return f
def ewmac(C,idx):
    s=0
    for fa,sl in ((16,64),(32,128),(64,256)):
        r=C.pct_change(); v=r.ewm(span=36).std()*C
        s=s+((C.ewm(span=fa).mean()-C.ewm(span=sl).mean())/v).clip(-20,20)/10
    return (s/3).clip(-1,1).where(C.shift(256).notna())
VAR={'blend63-252':sig_lb((63,126,252)),'lb252':sig_lb((252,)),'lb126-252':sig_lb((126,252)),'ewmac':ewmac}
GR={'METALS':U['METALS'],'COMMOD':U['COMMOD'],'INDICES':U['INDICES'],'MET+COM':U['METALS']+U['COMMOD'],'MET+COM+IDX':U['METALS']+U['COMMOD']+U['INDICES']}
P=[('dev',None,'2018-12-31'),('val','2019-01-01','2022-12-31'),('hold','2023-01-01',None),('full',None,None)]
rows=[]
for g,names in GR.items():
    for v,fn in VAR.items():
        df,pos,ip=dp.run(names,kind,fn,cost_mult=1.0,rates=rates)
        m={p:dp.metrics(df,pos,ip,a,b) for p,a,b in P}
        rows.append(dict(group=g,variant=v,**{f'{p}_sh':m[p]['sharpe'] for p,_,_ in P},**{f'{p}_pfm':m[p]['pf_monthly'] for p,_,_ in P},**{f'{p}_pft':m[p]['pf_trades'] for p,_,_ in P}))
        print(rows[-1],flush=True)
r=pd.DataFrame(rows).sort_values('dev_sh',ascending=False); r.to_csv('results/explore_trend_variants.csv',index=False)
print(r[['group','variant','dev_sh','val_sh','hold_sh','full_sh','dev_pfm','val_pfm','hold_pfm']].to_string(index=False))
