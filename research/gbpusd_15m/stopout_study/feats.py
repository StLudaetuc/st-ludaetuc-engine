"""Stop-out study for frozen 15m B+F+H. Features are measured on the SIGNAL bar (the bar before entry), so nothing is known only after entry."""
import os,sys,numpy as np,pandas as pd
H=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(H)
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'timeframe_study'))
import strategy as S, tf as T, bbsys2 as B
CORE=T.CORE
def build():
    m5=S.load(os.path.join(R,'data','GBPUSD_M5_BA_OANDA_20211025_20261028.csv.gz'))
    d=B.prep(T.resample(m5,'15min'))
    # extra signal-bar context (all use data up to and including the signal bar)
    d['spread_atr']=(d.ask_c-d.bid_c)/d.atr
    d['atr_pct']=d.atr.rolling(96*20).rank(pct=True)
    d['rng_atr']=(d.h-d.l)/d.atr
    d['body_frac']=(d.c-d.o).abs()/(d.h-d.l).replace(0,np.nan)
    d['vol_ratio']=d.volume/d.volume.rolling(96*5).median()
    d['pdc_dist_atr']=(d.c-d.pdc)/d.atr
    lon=d.index.tz_convert('Europe/London'); d['dow']=lon.dayofweek
    # day-so-far range vs 20-day avg daily range
    day=lon.date; dh=d.h.groupby(day).cummax(); dl=d.l.groupby(day).cummin()
    dr=(d.h.groupby(day).max()-d.l.groupby(day).min()); adr=dr.rolling(20).mean().shift(1)
    d['day_used']=((dh-dl).values/pd.Series(day).map(adr).values)
    d['mom8_atr']=(d.c-d.c.shift(8))/d.atr
    return d
def trades(d, part, cost_mult=1.0):
    t=B.run(part,exit_mode="session",windows=CORE,gate_long=(part.c>part.pdc).values,gate_short=(part.c<part.pdc).values,cost_mult=cost_mult)
    pos=d.index.get_indexer(t.t)-1                       # signal bar = bar before entry
    f=d.iloc[pos][['pb','bbw_pct','atr_pct','spread_atr','rng_atr','body_frac','vol_ratio','pdc_dist_atr','dow','day_used','mom8_atr','trend4h','atr']].reset_index(drop=True)
    t=pd.concat([t.reset_index(drop=True),f],axis=1)
    t['stop_atr']=t.stop_pips*B.PIP/t.atr
    t['pdc_dist_atr']=t.pdc_dist_atr*t.d                  # signed in trade direction
    t['mom8_atr']=t.mom8_atr*t.d
    t['trend_al']=(t.trend4h*t.d)
    t['pb_ext']=np.where(t.d==1,t.pb,100-t.pb)            # how extreme %B was, in trade direction
    t['stopped']=t.R<=-0.95
    return t
