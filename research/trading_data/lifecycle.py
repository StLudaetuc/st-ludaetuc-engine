from collections import defaultdict
from .evidence import timestamp,instrument,number,identity,canonical

def reconstruct(events,*,broker_manifest_complete=False):
    """Account/environment-scoped graph. No monetary accounting performed."""
    if type(broker_manifest_complete) is not bool:raise ValueError('explicit manifest verification boolean')
    if not isinstance(events,(list,tuple)) or len(events)>10000:raise ValueError('bounded events')
    requests={};predictions={};opens=defaultdict(list);fills=defaultdict(list);closes=defaultdict(list);exceptions=[];seen={};replays=0;attempts=[]
    for i,e in enumerate(events):
        try:
            scope=(identity(e.get('account_alias')),identity(e.get('broker_environment')))
            if scope[1] not in ('OANDA_PRACTICE','OANDA_LIVE'):raise ValueError('environment required')
            timestamp(e.get('event_time'));timestamp(e.get('received_at'));instrument(e.get('instrument'))
            event_id=identity(e.get('event_id'));key=scope+(event_id,)
            digest=canonical({k:v for k,v in e.items() if k!='received_at'})
            if key in seen:
                if seen[key]==digest:replays+=1;continue
                raise ValueError('CONFLICTING_EVENT')
            seen[key]=digest;p=e.get('payload');kind=e.get('kind')
            if not isinstance(p,dict):raise ValueError('payload missing')
            if kind=='prediction':
                signal=identity(p.get('signal_id'));pkey=scope+(signal,)
                if pkey in predictions:raise ValueError('CONFLICTING_SIGNAL')
                predictions[pkey]=e
            elif kind=='order_request':
                signal=identity(p.get('signal_id'));requests[scope+(identity(p.get('client_order_id')),)] = e
            elif kind=='trade_opened':
                signal=identity(p.get('signal_id'));status=p.get('fill_status')
                if status not in ('FILLED','CANCELLED','REJECTED','UNKNOWN'):raise ValueError('fill status required')
                attempts.append({'account_alias':scope[0],'broker_environment':scope[1],'signal_id':signal,'fill_status':status,'event_id':event_id})
                ids=p.get('broker_trade_ids',[])
                if not isinstance(ids,list) or len(ids)>100:raise ValueError('bounded broker trades')
                if status=='FILLED' and not ids:exceptions.append({'row':i,'reason':'FILLED_WITHOUT_BROKER_TRADE_ID'})
                if status!='FILLED' and ids:raise ValueError('NONFILLED_HAS_TRADE')
                for trade in ids:opens[scope+(identity(trade),)].append(e)
            elif kind in ('broker_fill','trade_outcome'):
                trade=identity(p.get('broker_trade_id'));identity(p.get('transaction_id'))
                if kind=='broker_fill':
                    identity(p.get('order_id'))
                    if p.get('action') not in ('OPEN','REDUCE','CLOSE'):raise ValueError('broker action required')
                    if abs(number(p.get('units')))<=0:raise ValueError('units required')
                    fills[scope+(trade,)].append(e)
                else:closes[scope+(trade,)].append(e)
            else:raise ValueError('unsupported lifecycle event')
        except (ValueError,TypeError,KeyError) as exc:exceptions.append({'row':i,'reason':str(exc)})
    results=[]
    population=set(opens)|set(fills)|set(closes)
    for key in sorted(population):
        oe=opens[key];be=sorted(fills[key],key=lambda e:(timestamp(e['event_time']),e['event_id']));ce=closes[key]
        signals={e['payload']['signal_id'] for e in oe}; reasons=[]
        if len(signals)>1:reasons.append('AMBIGUOUS_SIGNAL_LINK')
        signal=next(iter(signals)) if len(signals)==1 else None
        prediction=predictions.get(key[:2]+(signal,)) if signal else None
        if not oe:reasons.append('OPENING_MISSING')
        if prediction is None:reasons.append('PREDICTION_MISSING')
        if not be:reasons.append('BROKER_TRANSACTIONS_MISSING')
        actions=[e['payload']['action'] for e in be];full='CLOSE' in actions
        if ce and not full:reasons.append('INTERNAL_CLOSE_WITHOUT_BROKER_CLOSE')
        if full and not ce:reasons.append('INTERNAL_OUTCOME_MISSING_EOD_REVIEW_REQUIRED')
        if len(ce)>1:reasons.append('MULTIPLE_CLOSE_EVENTS_REVIEW_SLICES')
        opened=sum((abs(number(e['payload']['units'])) for e in be if e['payload']['action']=='OPEN'),number('0'))
        reduced=sum((abs(number(e['payload']['units'])) for e in be if e['payload']['action'] in ('REDUCE','CLOSE')),number('0'))
        if reduced>opened and opened:reasons.append('CLOSED_UNITS_EXCEED_OPENED')
        state='EXCEPTION' if reasons else 'CLOSED' if full else 'PARTIALLY_CLOSED' if 'REDUCE' in actions else 'OPEN'
        results.append({'account_alias':key[0],'broker_environment':key[1],'broker_trade_id':key[2],'signal_id':signal,'state':state,'reasons':reasons,'prediction_linked':prediction is not None,'broker_order_ids':sorted({e['payload']['order_id'] for e in be}),'broker_transaction_ids':[e['payload']['transaction_id'] for e in be],'event_ids':[e['event_id'] for e in oe+be+ce],'partial_close_slices':actions.count('REDUCE'),'remaining_units':str(opened-reduced) if be else None,'economic_result':'DEFER_TO_FINANCE_VERIFIED_CONTRACT'})
    linked=sum(not r['reasons'] for r in results);n=len(results)
    return {'attempts':attempts,'attempted_orders':len(attempts),'filled_orders':sum(r['fill_status']=='FILLED' for r in attempts),'cancelled_orders':sum(r['fill_status']=='CANCELLED' for r in attempts),'rejected_orders':sum(r['fill_status']=='REJECTED' for r in attempts),'trades':results,'malformed':exceptions,'replay_count':replays,'known_broker_trade_population':n,'fully_linked_population':linked,'linkage_percent':str(number(linked)*100/number(n)) if n else None,'broker_coverage_complete':broker_manifest_complete,'official_all_trades_reconciled':bool(broker_manifest_complete and n and linked==n and not exceptions),'unsubmitted_signals':[k[2] for k,e in predictions.items() if e['payload'].get('approval_status')=='approved' and not any((a['account_alias'],a['broker_environment'],a['signal_id'])==k for a in attempts)],'rejected_signals':[k[2] for k,e in predictions.items() if e['payload'].get('approval_status')=='rejected']}
