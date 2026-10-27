import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings('ignore')
import sys; sys.path.insert(0,'.'); import bt
f=pd.read_pickle('data/feat.pkl'); N=len(f)
H,L,C,A=f.h.values,f.l.values,f.c.values,f.atr.values
RR=1.5; MINP=12.0
def label(d):
    out=np.full(N,np.nan)
    for i in range(250,N-2):
        a=A[i]
        if not a>0: continue
        e=C[i]; dist=max(a*2.0,MINP*bt.PIP); st=e-d*dist; tg=e+d*dist*RR
        for j in range(i+1,min(N,i+720)):
            if d==1:
                if L[j]<=st: out[i]=0; break
                if H[j]>=tg: out[i]=1; break
            else:
                if H[j]>=st: out[i]=0; break
                if L[j]<=tg: out[i]=1; break
    return out
SPLIT=pd.Timestamp('2026-08-01',tz='UTC')
pb=(f.c-f.bbl)/(f.bbu-f.bbl)
rows=[]
for d in (1,-1):
    g=pd.DataFrame({'t':f.t,'win':label(d)})
    g['htf']=((f.htf_c>f.htf_e) if d==1 else (f.htf_c<f.htf_e)).astype(float)
    g['trend']=(((f.ema20>f.ema50)&(f.c>f.ema200)) if d==1 else ((f.ema20<f.ema50)&(f.c<f.ema200))).astype(float)
    g['bbz']=np.clip(((pb-0.5)*2)*d,-2,2)
    rsi_d=f.rsi if d==1 else 100-f.rsi
    g['rsi_c']=np.clip((rsi_d-50)/10,-3,3)
    g['vwap_d']=np.clip(d*f.d_vwap_atr,-10,10)/5
    g['macd_d']=np.clip(d*f.macdh/f.atr,-1,1)
    g['atr_p']=np.clip(f.atr_pips,0.5,6)
    g['atr_r']=np.clip(f.atr_ratio,0.5,2)
    g['core']=f.hour.between(6,18).astype(float)
    g['adx_s']=np.clip(f.adx,5,60)/25
    g['dow']=f.dow
    rows.append(g)
D=pd.concat(rows); D=D[D.win.notna()&(D.dow<5)].dropna()
FE=['htf','trend','bbz','rsi_c','vwap_d','macd_d','atr_p','atr_r','core','adx_s']
def fit(X,y,l2=1.0,it=50):
    X=np.c_[np.ones(len(X)),X]; w=np.zeros(X.shape[1])
    for _ in range(it):
        p=1/(1+np.exp(-X@w)); W=p*(1-p)
        g=X.T@(p-y)+l2*np.r_[0,w[1:]]
        Hm=X.T@(X*W[:,None])+l2*np.diag(np.r_[0,np.ones(len(w)-1)])
        w-=np.linalg.solve(Hm,g)
    return w
def pred(w,X): return 1/(1+np.exp(-(np.c_[np.ones(len(X)),X]@w)))
def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
IS=D[D.t<SPLIT]; OS=D[D.t>=SPLIT]
w=fit(IS[FE].values,IS.win.values)
pi=pred(w,IS[FE].values); po=pred(w,OS[FE].values)
print('n IS',len(IS),'OOS',len(OS),'base',IS.win.mean().round(3),OS.win.mean().round(3))
print('AUC IS',round(auc(IS.win.values,pi),4),'OOS',round(auc(OS.win.values,po),4))
for n,c in zip(['intercept']+FE,w): print(f'{n:8s} {c:+.4f}')
OS=OS.assign(p=po)
OS['b']=pd.cut(OS.p,[0,.3,.34,.38,.42,.46,.5,1])
print(OS.groupby('b').agg(n=('win','size'),pred=('p','mean'),actual=('win','mean')).round(3).to_string())
be=1/(1+RR)
print('break-even (gross)',round(be,3))
# fit on all data for deployment
wa=fit(D[FE].values,D.win.values)
json.dump({'features':FE,'coef':dict(zip(['intercept']+FE,[round(x,4) for x in wa])),'rr':RR,'min_stop_pips':MINP,'auc_oos':round(auc(OS.win.values,po),4)},open('data/edge_model.json','w'),indent=1)
print(open('data/edge_model.json').read())
D.to_pickle('data/dirset12.pkl')
