"""Run the frozen breakout B+F+H system on a given bid/ask OHLC frame (any timeframe)."""
import sys; sys.path.insert(0,'.')
import bbsys2 as B, pandas as pd, numpy as np
CORE=((7*60,11*60),(12*60+30,16*60))
def resample(df, rule):
    agg={}
    for side in ('bid','ask'):
        agg.update({f'{side}_o':'first',f'{side}_h':'max',f'{side}_l':'min',f'{side}_c':'last'})
    agg['volume']='sum'
    r=df[list(agg)].resample(rule, label='left', closed='left').agg(agg).dropna(subset=['bid_o'])
    for k in 'ohlc': r[k]=(r['bid_'+k]+r['ask_'+k])/2
    return r
def evaluate(df, label, holdout=False):
    d=B.prep(df); n=len(d)
    parts={'development':d.iloc[:int(n*0.6)],'validation':d.iloc[int(n*0.6):int(n*0.8)]}
    if holdout: parts['holdout']=d.iloc[int(n*0.8):]
    out={}
    for name,p in parts.items():
        for mult in (1.0,1.5,2.0):
            t=B.run(p,exit_mode="session",windows=CORE,gate_long=(p.c>p.pdc).values,gate_short=(p.c<p.pdc).values,cost_mult=mult)
            out[(name,mult)]=B.met(t)
    sp=((d.ask_c-d.bid_c)/d.atr).median()
    return out, round(float(sp),3), (str(d.index[0])[:10], str(d.index[-1])[:10], n)
