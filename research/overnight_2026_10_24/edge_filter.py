from v6 import *
import json
M=json.load(open('data/edge_model.json')); cf=M['coef']
pb=((f.c-f.bbl)/(f.bbu-f.bbl)).values
def edge_p(i,d):
    x=dict(htf=float((f.htf_c.values[i]>f.htf_e.values[i]) if d==1 else (f.htf_c.values[i]<f.htf_e.values[i])),
      trend=float(((f.ema20.values[i]>f.ema50.values[i]) and (C[i]>f.ema200.values[i])) if d==1 else ((f.ema20.values[i]<f.ema50.values[i]) and (C[i]<f.ema200.values[i]))),
      bbz=np.clip((pb[i]-0.5)*2*d,-2,2), rsi_c=np.clip(((f.rsi.values[i] if d==1 else 100-f.rsi.values[i])-50)/10,-3,3),
      vwap_d=np.clip(d*f.d_vwap_atr.values[i],-10,10)/5, macd_d=np.clip(d*f.macdh.values[i]/ATR[i],-1,1),
      atr_p=np.clip(f.atr_pips.values[i],0.5,6), atr_r=np.clip(f.atr_ratio.values[i],0.5,2), core=float(6<=f.hour.values[i]<=18), adx_s=np.clip(f.adx.values[i],5,60)/25)
    z=cf['intercept']+sum(cf[k]*x[k] for k in x); return 1/(1+np.exp(-z))
HB=(19,20,21,22,23,0,1,2,3,4,5)
base=bt.candidates_v52(f); best=gen('pullback',1.0,35,trigger='wick',hours_block=HB)
out=[]
for name,s,kw in (('v5.2 as-is',base,dict(entry_mode='close',stop_mode='v52',min_stop_pips=0,rr=1.6)),
                  ('v5.2 + min12 stop',base,dict(entry_mode='close',stop_mode='v52',min_stop_pips=12,rr=1.5)),
                  ('v6 best',best,dict(entry_mode='stop',k=2,stop_mode='swing',swing_lb=6,stop_atr_pad=0.3,min_stop_pips=12,rr=1.5))):
    for th in (0,0.38,0.40,0.42):
        r=run(s,filt=(lambda i,d,th=th: edge_p(i,d)>=th),cost=1.4,**kw)
        out.append(split_stats(r,f'{name} | edge_p>={th}'))
pd.set_option('display.width',250)
print(pd.DataFrame(out)[['label','n','win_rate','avg_R_gross','avg_R_net','profit_factor','early_stop_pct','IS_n','IS_R','OOS_n','OOS_R','OOS_PF']].to_string())
