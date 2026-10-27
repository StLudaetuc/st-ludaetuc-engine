import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
D=pd.read_pickle('data/dirset.pkl')
def tab(mask,name):
    g=D[mask]; print(f"{name:55s} n={len(g):6d} wr={g.win.mean():.3f} IS={g[~g.oos].win.mean():.3f} OOS={g[g.oos].win.mean():.3f}")
tab(D.win.notna(),'all')
tab(D.trend,'trend aligned'); tab(~D.trend,'not trend aligned')
tab(D.htf,'htf aligned'); tab(~D.htf,'htf against')
tab(D.trend&D.htf,'both aligned'); tab(~D.trend&~D.htf,'both against (counter-trend)')
D['bbz']=np.where(D.d==1,(D.c-D.bbb)/(D.bbu-D.bbb),(D.bbb-D.c)/(D.bbu-D.bbb))  # + = price on our side
for lo,hi in [(-9,-1.2),(-1.2,-0.8),(-0.8,-0.3),(-0.3,0.3),(0.3,0.8),(0.8,1.2),(1.2,9)]:
    tab((D.bbz>lo)&(D.bbz<=hi),f'bbz {lo}..{hi} (neg = buying below basis)')
for lo,hi in [(-99,-6),(-6,-3),(-3,-1),(-1,1),(1,3),(3,6),(6,99)]:
    tab((D.dist_vwap>lo)&(D.dist_vwap<=hi),f'dist_vwap {lo}..{hi}')
for lo,hi in [(0,25),(25,30),(30,35),(35,45),(45,55),(55,65),(65,100)]:
    tab((D.rsi_d>lo)&(D.rsi_d<=hi),f'rsi_d {lo}..{hi}')
# combos: trend-aligned but price pulled back below BB basis & close beyond vwap moderate
tab(D.htf&(D.bbz<-0.8),'htf aligned + deep pullback (bbz<-0.8)')
tab(D.htf&(D.bbz<-1.0)&(D.rsi_d<35),'htf aligned + BB lower + RSI<35')
tab(D.htf&D.trend&(D.bbz<-0.8),'trend+htf + bbz<-0.8')
tab(D.htf&D.trend&(D.bbz>0.8),'trend+htf + bbz>0.8 (chasing)')
tab(~D.htf&(D.bbz<-1.0)&(D.rsi_d<30),'counter-htf fade: BB outside + RSI<30')
tab((D.bbz<-1.0)&(D.rsi_d<30)&(D.atr_ratio<1.3),'fade BB+RSI<30, atr_ratio<1.3')
tab((D.bbz<-1.0)&(D.rsi_d<30)&(D.atr_ratio>=1.3),'fade BB+RSI<30, atr_ratio>=1.3')
