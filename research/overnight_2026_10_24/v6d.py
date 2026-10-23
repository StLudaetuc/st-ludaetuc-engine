from edge_filter import *
out=[]
for k in (1,2,3):
  for th in (0,0.35,0.36):
    r=run(best,'closebreak',k=k,stop_mode='swing',swing_lb=6,stop_atr_pad=0.3,min_stop_pips=12,rr=1.5,cost=1.4,filt=(lambda i,d,th=th: edge_p(i,d)>=th))
    out.append(split_stats(r,f'closebreak k={k} edge>={th}'))
    if k==2 and th==0.35:
        r['m']=pd.to_datetime(r.t,utc=True).dt.strftime('%Y-%m'); print(r.groupby('m').apply(lambda g: pd.Series({'n':len(g),'wr':(g.outcome=='win').mean(),'R':(g.r-g.cost_r).sum()})).round(2).to_string())
        r.to_pickle('data/v6_final_res.pkl')
for cost in (1.0,1.8):
    r=run(best,'closebreak',k=2,stop_mode='swing',swing_lb=6,stop_atr_pad=0.3,min_stop_pips=12,rr=1.5,cost=cost,filt=(lambda i,d: edge_p(i,d)>=0.35)); out.append(split_stats(r,f'closebreak k=2 edge>=0.35 cost{cost}'))
print(pd.DataFrame(out)[['label','n','win_rate','avg_R_gross','avg_R_net','total_R_net','profit_factor','early_stop_pct','IS_n','IS_R','OOS_n','OOS_wr','OOS_R','OOS_PF']].to_string())
