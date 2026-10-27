from research import *
import warnings; warnings.filterwarnings('ignore')
p=lambda s: s.shift(1)
bbz_l=(f.c-f.bbb)/(f.bbu-f.bbb)            # +1 = at upper band
htfb=f.htf_c>f.htf_e; htfs=f.htf_c<f.htf_e

def gen(mode='pullback', stretch=1.0, rsi_x=35, lookback=6, trigger='reclaim', hours_block=(19,20,21,22,23),
        atr_min=1.8, atr_ratio_min=0.9, htf_req=True, cooldown=5, maxday=18):
    # stretch condition happened within last `lookback` bars
    stl=((bbz_l<=-stretch)|(f.rsi<=rsi_x)).astype(int).rolling(lookback).max()>0
    sts=((bbz_l>=stretch)|(f.rsi>=100-rsi_x)).astype(int).rolling(lookback).max()>0
    if trigger=='reclaim':   # bar closes above prior high & bullish body  (momentum turn confirmed)
        tl=(f.c>p(f.h))&(f.c>f.o); ts=(f.c<p(f.l))&(f.c<f.o)
    elif trigger=='rsi_cross':
        tl=(f.rsi>rsi_x)&(p(f.rsi)<=rsi_x); ts=(f.rsi<100-rsi_x)&(p(f.rsi)>=100-rsi_x)
    elif trigger=='wick':
        tl=(f.lw_r>=0.4)&(f.c>f.o); ts=(f.uw_r>=0.4)&(f.c<f.o)
    elif trigger=='none':
        tl=bbz_l<=-stretch; ts=bbz_l>=stretch
    common=(f.atr_pips>=atr_min)&(f.atr_ratio>=atr_ratio_min)&(~f.hour.isin(hours_block))&(f.dow<5)
    # don't buy if close already back above basis (we want price still "cheap")
    cheapL=bbz_l<0.3; cheapS=bbz_l>-0.3
    if mode=='pullback':
        Lc=stl&tl&common&cheapL&(htfb if htf_req else True)
        Sc=sts&ts&common&cheapS&(htfs if htf_req else True)
    elif mode=='fade':
        Lc=stl&tl&common&cheapL&htfs     # counter-HTF
        Sc=sts&ts&common&cheapS&htfb
    else:  # any direction
        Lc=stl&tl&common&cheapL; Sc=sts&ts&common&cheapS
    sig=[]; last=-10**9; day=None; nd=0
    Lv,Sv=Lc.values,Sc.values; td=f.t.dt.floor('D').values
    for i in range(250,N):
        if td[i]!=day: day=td[i]; nd=0
        if bool(Lv[i])==bool(Sv[i]): continue
        if i-last<cooldown or nd>=maxday: continue
        sig.append((i,1 if Lv[i] else -1,mode,'')); last=i; nd+=1
    return pd.DataFrame(sig,columns=['i','dir','model','regime'])

if __name__=='__main__':
    out=[]
    for mode in ('pullback','fade','any'):
      for trig in ('reclaim','rsi_cross','wick'):
        for stretch,rx in ((1.0,35),(1.2,30)):
          s=gen(mode,stretch,rx,trigger=trig)
          for sm,lb,pad,ms,rr in (('swing',6,0.3,8,1.5),('swing',6,0.3,10,1.2),('v52',12,0.0,8,1.5)):
            r=run(s,'stop' if trig!='none' else 'close',k=2,stop_mode=sm,swing_lb=lb,stop_atr_pad=pad,min_stop_pips=ms,rr=rr)
            out.append(split_stats(r,f'{mode}|{trig}|bb{stretch}/rsi{rx}|{sm}{lb}+{pad}|min{ms}|rr{rr}'))
    o=pd.DataFrame(out).sort_values('IS_R',ascending=False)
    pd.set_option('display.width',250)
    print(o[['label','n','win_rate','avg_R_gross','avg_R_net','profit_factor','IS_n','IS_wr','IS_R','OOS_n','OOS_wr','OOS_R','OOS_PF']].to_string())
