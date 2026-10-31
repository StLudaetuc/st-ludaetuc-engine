import json, pandas as pd, tvs
S='/tmp/claude-0/-home-user/46736265-72c4-5060-89a6-fa160e5a55e4/scratchpad'
M1=tvs.load(S+'/night/fine/M1_XAU_USD.csv.gz'); S5=tvs.load(S+'/night/fine/S5_XAU_USD.csv.gz'); M15=tvs.load(S+'/bb15/XAU_USD_M15_BA.csv.gz')
bars={'30s':tvs.resample(S5,'30s'),'45s':tvs.resample(S5,'45s'),'1m':M1,'5m':tvs.resample(M1,'5min'),'10m':tvs.resample(M1,'10min'),'15m_1y':tvs.resample(M1,'15min'),'15m_5y':M15}
TVWIN={('S1','10m'):'2026-04-05',('S2','30s'):'2026-10-19',('S2','45s'):'2026-10-13',('S2','5m'):'2026-07-19'}
TVSTAT={('S1','10m'):'TV: 34 tr, PF 2.201, win 76.5%, PnL +615',('S2','30s'):'TV: 44 tr, PF 3.662, win 77.3%, PnL +272',('S2','45s'):'TV: 35 tr, PF 2.075, win 68.6%, PnL +220',('S2','5m'):'TV: 45 tr, PF 2.175, win 64.4%, PnL +565'}
out={}
for strat,fn in (('S1',tvs.signals_s1),('S2',tvs.signals_s2)):
    for tf,b in bars.items():
        sig,a=fn(tvs.mid(b))
        for mode in ('tv','real'):
            tr=tvs.simulate(b,sig,a,mode=mode)
            key=f'{strat} {tf} {mode}'
            out[key]={'full':tvs.summary(tr),'span':f'{b.index[0].date()}..{b.index[-1].date()}'}
            if (strat,tf) in TVWIN:
                w=tr[tr.entry>=pd.Timestamp(TVWIN[(strat,tf)],tz='UTC')]
                out[key]['tv_window']=tvs.summary(w); out[key]['tv_reported']=TVSTAT[(strat,tf)]
                o=tr[tr.entry<pd.Timestamp(TVWIN[(strat,tf)],tz='UTC')]
                out[key]['before_window']=tvs.summary(o)
            if strat=='S2' and mode=='real':
                tr2=tvs.simulate(b,sig,a,mode='real',exits=True)
                out[key+' +SL/TP']={'full':tvs.summary(tr2)}
            print(key, out[key], flush=True)
json.dump(out,open('results.json','w'),indent=1,default=str)
