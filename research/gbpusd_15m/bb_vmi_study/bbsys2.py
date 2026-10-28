"""Breakout base (Bollinger %B zones reversed) with pre-declared extensions. Research only."""
import math, numpy as np, pandas as pd
PIP=1e-4
def prep(df, bb_len=20, bb_mult=2.0, atr_len=14):
    d=df.copy(); c=d.c
    m=c.rolling(bb_len).mean(); s=c.rolling(bb_len).std(ddof=0)
    d['pb']=(c-(m-bb_mult*s))/(2*bb_mult*s)*100
    d['bbw']=(4*bb_mult*s)/m                            # band width
    d['bbw_pct']=d.bbw.rolling(96*5).rank(pct=True)     # vs last 5 days
    tr=pd.concat([d.h-d.l,(d.h-c.shift()).abs(),(d.l-c.shift()).abs()],axis=1).max(axis=1)
    d['atr']=tr.ewm(alpha=1/atr_len,adjust=False).mean()
    lon=d.index.tz_convert('Europe/London'); d['lm']=lon.hour*60+lon.minute
    # 4H structure on completed UTC 4H buckets: +1 if last completed 4H close > close 6 completed 4H bars earlier (24h), -1 if lower
    key=d.index.floor('4h'); c4=d.c.groupby(key).last(); s4=np.sign(c4-c4.shift(6)).shift(1)   # shift(1): only completed buckets
    d['trend4h']=key.map(s4).values
    # daily: prior FX-day close (17:00 NY)
    ny=d.index.tz_convert('America/New_York'); dk=pd.Index((ny+pd.Timedelta(hours=7)).date)
    dc=d.c.groupby(dk).last(); prev=dc.shift(1)
    d['pdc']=dk.map(prev).values
    return d
def run(d, low=36, high=78, arm_bars=8, swing=2, rr=1.5, buf_atr=0.25, min_stop_atr=0.75,
        windows=((7*60,17*60),), flat=20*60+45, slip=0.2*PIP, exit_mode="fixed", trail_atr=2.0,
        gate_long=None, gate_short=None, cost_mult=1.0):
    o,h,l,c=(d[k].values for k in 'ohlc'); bo,bh,bl,ao,ah,al=(d[k].values for k in ('bid_o','bid_h','bid_l','ask_o','ask_h','ask_l'))
    bc,ac=d.bid_c.values,d.ask_c.values
    pb,atr,lm=d.pb.values,d.atr.values,d.lm.values; idx=d.index; n=len(d)
    hh=pd.Series(h).rolling(swing).max().shift(1).values; ll=pd.Series(l).rolling(swing).min().shift(1).values
    gL=np.ones(n,bool) if gate_long is None else gate_long; gS=np.ones(n,bool) if gate_short is None else gate_short
    lArm=sArm=None; lExt=sExt=math.nan; pos=None; pend=None; out=[]
    for i in range(1,n):
        if pend is not None:
            dd,stop,a=pend; half=(ao[i]-bo[i])/2*(cost_mult-1)
            e=(ao[i]+slip*cost_mult+half) if dd==1 else (bo[i]-slip*cost_mult-half)
            dist=(e-stop)*dd
            if dist>0: pos=dict(d=dd,e=e,sl=stop,tp=(e+dd*rr*dist) if exit_mode=="fixed" else math.nan,dist=dist,t=idx[i],lm=lm[i],atr=a,best=e)
            pend=None
        if pos is not None:
            dd=pos['d']; O,H,L=(bo[i],bh[i],bl[i]) if dd==1 else (ao[i],ah[i],al[i]); half=(ao[i]-bo[i])/2*(cost_mult-1)
            ex=None
            if (O<=pos['sl'] if dd==1 else O>=pos['sl']): ex=O-dd*(slip*cost_mult+half)
            elif exit_mode=="fixed" and (O>=pos['tp'] if dd==1 else O<=pos['tp']): ex=O
            elif (L<=pos['sl'] if dd==1 else H>=pos['sl']): ex=pos['sl']-dd*(slip*cost_mult+half)
            elif exit_mode=="fixed" and (H>=pos['tp'] if dd==1 else L<=pos['tp']): ex=pos['tp']
            elif flat<=lm[i]<22*60 and i+1<n: ex=(bo[i+1] if dd==1 else ao[i+1])-dd*(slip*cost_mult+half)
            if ex is not None:
                out.append((pos['t'],dd,(ex-pos['e'])*dd/pos['dist'],pos['dist']/PIP,pos['lm'])); pos=None
            elif exit_mode=="trail":
                # ratchet stop at bar close: trail_atr * entry ATR behind the best close (never loosens)
                cl=bc[i] if dd==1 else ac[i]
                pos['best']=max(pos['best'],cl) if dd==1 else min(pos['best'],cl)
                ns=pos['best']-dd*trail_atr*pos['atr']
                pos['sl']=max(pos['sl'],ns) if dd==1 else min(pos['sl'],ns)
        if math.isnan(pb[i]) or math.isnan(atr[i]): continue
        lz=pb[i]>=high; sz=pb[i]<=low          # breakout: strength arms longs, weakness arms shorts
        lw=lArm is not None and i-lArm<=arm_bars; sw=sArm is not None and i-sArm<=arm_bars
        if lz: lExt=min(lExt,l[i]) if lw else l[i]; lArm=i
        elif lw: lExt=min(lExt,l[i])
        if sz: sExt=max(sExt,h[i]) if sw else h[i]; sArm=i
        elif sw: sExt=max(sExt,h[i])
        lA=lArm is not None and i-lArm<=arm_bars; sA=sArm is not None and i-sArm<=arm_bars
        ok_time=any(a<=lm[i]<b for a,b in windows)
        if pos is None and pend is None and ok_time and i+1<n:
            if lA and c[i]>hh[i] and gL[i]:
                stop=min(lExt-buf_atr*atr[i], c[i]-min_stop_atr*atr[i]); pend=(1,stop,atr[i]); lArm=None
            elif sA and c[i]<ll[i] and gS[i]:
                stop=max(sExt+buf_atr*atr[i], c[i]+min_stop_atr*atr[i]); pend=(-1,stop,atr[i]); sArm=None
    return pd.DataFrame(out,columns=['t','d','R','stop_pips','lm'])
def met(t):
    if t.empty: return dict(n=0)
    r=t.R.values; gp,gl=r[r>0].sum(),-r[r<0].sum(); eq=np.cumsum(r)
    se=r.std()/np.sqrt(len(r))
    return dict(n=len(r),win=round(float((r>0).mean()),3),avgR=round(float(r.mean()),3),se=round(float(se),3),pf=round(float(gp/gl),3) if gl else None,
                ddR=round(float((np.maximum.accumulate(np.r_[0,eq])[1:]-eq).max()),1))
