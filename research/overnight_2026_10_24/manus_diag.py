import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
D=pd.read_pickle('data/dirset.pkl')
# replicate engine signal_score() trend terms on every bar/direction
ema_ok=np.where(D.d==1,D.ema20>=D.ema50,D.ema20<=D.ema50)
trend_b=np.where(D.d==1,(D.ema20>D.ema50)&(D.c>D.ema200),(D.ema20<D.ema50)&(D.c<D.ema200))
trend_against=np.where(D.d==1,(D.ema20<D.ema50)&(D.c<D.ema200),(D.ema20>D.ema50)&(D.c>D.ema200))
htf_ok=D.htf; htf_against=~D.htf
rsi_ok=np.where(D.d==1,D.rsi.between(35,62),D.rsi.between(38,65))
macd_ok=np.where(D.d==1,D.macdh>=0,D.macdh<=0)
score=45+np.where(ema_ok,8,-8)+np.where(trend_b,8,np.where(trend_against,-12,0))+np.where(htf_ok,10,-12)+np.where(rsi_ok,6,0)+np.where(macd_ok,4,0)
D['manus_sig']=score
q=pd.qcut(D.manus_sig,5,duplicates='drop')
print(D.groupby(q).apply(lambda g: pd.Series({'n':len(g),'wr':g.win.mean(),'IS':g[~g.oos].win.mean(),'OOS':g[g.oos].win.mean()})).round(3).to_string())
# real master-log check: valid trades scores
v=pd.read_pickle('data/valid_trades.pkl')
for c in ['manus_signal_quality_score','manus_context_score','Manus_Strategy_Fit_Score','manus_risk_score','manus_confidence_score','manus_predicted_win_probability','shadow_score_v0','manus_pre_gate_score']:
    x=pd.to_numeric(v[c],errors='coerce')
    if x.notna().sum()>10:
        print(c, 'n',x.notna().sum(),'corr w/ win',round(np.corrcoef(x[x.notna()],v.win[x.notna()])[0,1],3),'mean W',round(x[v.win==1].mean(),3),'mean L',round(x[v.win==0].mean(),3))
