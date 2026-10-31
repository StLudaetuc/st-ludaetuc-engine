import os,sys,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dp, pandas as pd, numpy as np
rates=dp.load_rates(); U=dp.universe()
kind={**{n:'FX' for n in U['FX']},**{n:'CFD' for g in ('METALS','INDICES','BONDS','COMMOD') for n in U[g]}}
P=[('development',None,'2018-12-31'),('validation','2019-01-01','2022-12-31'),('holdout','2023-01-01',None),('full',None,None)]
res={}
def go(label,names,fn,cm):
    df,pos,ip=dp.run(names,kind,fn,cost_mult=cm,rates=rates)
    res[(label,cm)]={p:dp.metrics(df,pos,ip,a,b) for p,a,b in P}
    for p,_,_ in P:
        m=res[(label,cm)][p]; print(f"{label:16s} {cm}x {p:11s} Sh={m['sharpe']} PFd={m['pf_daily']} PFm={m['pf_monthly']} PFt={m['pf_trades']} n={m['trades']} ret={m['ann_ret']}% vol={m['ann_vol']}% dd={m['maxdd_pct']}% cost={m['cost_pa']}% fin={m['fin_pa']}%")
    return df
out={}
for cm in (1.0,2.0):
    for g in ('FX','METALS','INDICES','BONDS','COMMOD'):
        out[('TREND_'+g,cm)]=go('TREND_'+g,U[g],dp.tsmom,cm)
    allx=U['FX']+U['METALS']+U['INDICES']+U['BONDS']; alln=allx+U['COMMOD']
    out[('TREND_ALLexCOM',cm)]=go('TREND_ALLexCOM',allx,dp.tsmom,cm)
    out[('TREND_ALL',cm)]=go('TREND_ALL',alln,dp.tsmom,cm)
    out[('CARRY_FX',cm)]=go('CARRY_FX',U['FX'],dp.make_carry(U['FX'],rates),cm)
    # combined: 50/50 of TREND_ALL and CARRY_FX daily pnl, each scaled to equal full-period vol
    a=out[('TREND_ALL',cm)].pnl; b=out[('CARRY_FX',cm)].pnl
    j=pd.concat([a,b],axis=1).fillna(0); j.columns=['t','c']
    comb=0.5*j.t/j.t.std()+0.5*j.c/j.c.std()
    cdf=pd.DataFrame(dict(pnl=comb*0.006,cost=0.0,fin=0.0))
    for p,s,e in P:
        r=cdf.pnl.loc[s:e]; m=r.resample('ME').sum(); ann=r.mean()*252; vol=r.std()*np.sqrt(252)
        pf=lambda x: float(x[x>0].sum()/-x[x<0].sum())
        print(f"COMBINED         {cm}x {p:11s} Sh={ann/vol:.2f} PFd={pf(r):.3f} PFm={pf(m):.3f}")
        res[('COMBINED',cm)]=res.get(('COMBINED',cm),{}); res[('COMBINED',cm)][p]=dict(sharpe=round(ann/vol,2),pf_daily=round(pf(r),3),pf_monthly=round(pf(m),3))
json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open('prereg_results.json','w'),indent=1)
for k,v in out.items(): v.to_csv(f"results_{k[0]}_{k[1]}x.csv.gz")
