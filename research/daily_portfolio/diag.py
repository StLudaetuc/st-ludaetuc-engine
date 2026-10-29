"""Diagnostic (not a pre-registered test): frictionless runs (no spread, no financing markup; pure rate differential kept) to separate signal from costs."""
import os,sys,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import dp, numpy as np, pandas as pd
os.environ.setdefault('DAILY_DIR','data/daily9')
rates=dp.load_rates(); U=dp.universe()
kind={**{n:'FX' for n in U['FX']},**{n:'CFD' for g in ('METALS','INDICES','BONDS','COMMOD') for n in U[g]}}
P=[('dev',None,'2018-12-31'),('val','2019-01-01','2022-12-31'),('hold','2023-01-01',None),('full',None,None)]
out={}
for lab,names,fn in (('TREND_FX',U['FX'],dp.tsmom),('TREND_METALS',U['METALS'],dp.tsmom),('TREND_INDICES',U['INDICES'],dp.tsmom),
                     ('TREND_BONDS',U['BONDS'],dp.tsmom),('TREND_COMMOD',U['COMMOD'],dp.tsmom),('CARRY_FX',U['FX'],dp.make_carry(U['FX'],rates))):
    df,pos,ip=dp.run(names,kind,fn,cost_mult=0.0,rates=rates)
    gross_notional=pos.abs().sum(axis=1).mean()
    row={p:dp.metrics(df,pos,ip,a,b)['sharpe'] for p,a,b in P}
    out[lab]=dict(frictionless_sharpe=row,avg_gross_leverage=round(float(gross_notional),2))
    print(lab,'frictionless Sharpe',row,'avg gross notional x equity',round(gross_notional,2))
json.dump(out,open('results/diag_frictionless.json','w'),indent=1)
