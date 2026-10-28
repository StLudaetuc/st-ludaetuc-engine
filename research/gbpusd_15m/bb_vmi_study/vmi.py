import numpy as np, pandas as pd
def add_vmi(d, asia=(0,7*60), lon_start=7*60):
    """St Ludaetuc Value Migration Index. London-session tick-volume VWAP vs the Asian range,
    in units of the Asian half-range. Uses only bars up to and including the current bar."""
    lon=d.index.tz_convert('Europe/London'); lm=(lon.hour*60+lon.minute).values; date=lon.date
    tp=((d.h+d.l+d.c)/3).values; v=d.volume.values.astype(float)
    n=len(d); vmi=np.full(n,np.nan); cur=None; aH=aL=np.nan; pv=vv=0.0; nb=0
    for i in range(n):
        if date[i]!=cur:
            cur=date[i]; aH=aL=np.nan; pv=vv=0.0; nb=0
        if asia[0]<=lm[i]<asia[1]:
            aH=d.h.values[i] if np.isnan(aH) else max(aH,d.h.values[i]); aL=d.l.values[i] if np.isnan(aL) else min(aL,d.l.values[i])
        elif lm[i]>=lon_start and lm[i]<21*60 and not np.isnan(aH):
            pv+=tp[i]*v[i]; vv+=v[i]; nb+=1
            half=(aH-aL)/2
            if vv>0 and half>0 and nb>=4:
                vmi[i]=(pv/vv-(aH+aL)/2)/half
    d=d.copy(); d['vmi']=vmi; return d
