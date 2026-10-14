from collections import defaultdict
from datetime import timedelta
from .evidence import timestamp,instrument,number,canonical

TIMEFRAMES={'1m':60,'2m':120,'3m':180,'5m':300,'10m':600,'15m':900,'1h':3600}

def audit(rows,start,end,*,expected_slots=None,calendar_evidence=None):
    if not isinstance(rows,(list,tuple)) or len(rows)>100000:raise ValueError('bounded candle population')
    a,b=timestamp(start),timestamp(end)
    if b<=a:raise ValueError('valid half-open period')
    groups=defaultdict(list);errors=[]
    for i,row in enumerate(rows):
        try:
            symbol=instrument(row.get('instrument'));tf=row.get('timeframe')
            if tf not in TIMEFRAMES:raise ValueError('unknown timeframe')
            t=timestamp(row.get('event_time'))
            if not a<=t<b:raise ValueError('outside requested window')
            o,h,l,c=(number(row.get(k)) for k in ('open','high','low','close'))
            if min(o,h,l,c)<=0 or not l<=min(o,c)<=max(o,c)<=h:raise ValueError('OHLC inconsistent')
            v=number(row['volume']) if row.get('volume') is not None else None
            if v is not None and v<0:raise ValueError('negative volume')
            groups[(symbol,tf)].append((t,row,v))
        except (ValueError,TypeError,KeyError) as exc:errors.append({'row':i,'reason':str(exc)})
    report=[]
    for key,values in sorted(groups.items()):
        seen={};replays=conflicts=0
        for t,row,_ in values:
            digest=canonical({k:row.get(k) for k in ('open','high','low','close','volume')})
            if t in seen:
                if seen[t]==digest:replays+=1
                else:conflicts+=1
            else:seen[t]=digest
        times=sorted(seen);step=TIMEFRAMES[key[1]]
        offgrid=sum(t.timestamp()%step!=0 for t in times)
        gaps=[]
        for left,right in zip(times,times[1:]):
            missing=int((right-left).total_seconds()/step)-1
            if missing>0:gaps.append({'after':left.isoformat(),'before':right.isoformat(),'missing_grid_slots':missing})
        expected=(expected_slots or {}).get(key)
        missing=sorted(set(timestamp(t) for t in expected)-set(times)) if expected is not None else None
        extra=sorted(set(times)-set(timestamp(t) for t in expected)) if expected is not None else None
        verified=expected is not None and bool(calendar_evidence)
        report.append({'instrument':key[0],'timeframe':key[1],'rows':len(values),'distinct_times':len(times),'first':times[0].isoformat(),'last':times[-1].isoformat(),'duplicate_replays':replays,'conflicting_timestamps':conflicts,'out_of_order_arrivals':sum(values[i][0]<values[i-1][0] for i in range(1,len(values))),'off_grid':offgrid,'interior_gap_candidates':gaps,'missing_expected':len(missing) if missing is not None else None,'unexpected_slots':len(extra) if extra is not None else None,'calendar_verified':verified,'complete':verified and not missing and not extra and not errors and not conflicts and not offgrid,'volume_missing':sum(v is None for _,_,v in values),'received_at_latency':'UNAVAILABLE_UNLESS_SOURCE_EXPORT_INCLUDES_RECEIPT','indicators':'NOT_EXPOSED_BY_CURRENT_EXPORT'})
    return {'streams':report,'malformed':errors,'source_completeness':'UNPROVED' if not calendar_evidence else 'EXPLICIT_CALENDAR_ONLY'}
