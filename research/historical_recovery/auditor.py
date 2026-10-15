"""Bounded offline population diagnostics and read-only dataset product."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
from .normalizer import iso_time, digest, normalize

MAX_RECORDS=50000

def audit(events,context,*,start,end,source_count=None,snapshot_complete=False):
    start,end=iso_time(start),iso_time(end)
    if not timedelta(0)<end-start<=timedelta(days=31):raise ValueError('INVALID_WINDOW')
    if len(events)>MAX_RECORDS:raise ValueError('BOUNDED_INPUT_REQUIRED')
    rows=[];exceptions=[]
    for index,event in enumerate(events):
        try:row=normalize(event,context)
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError):
            exceptions.append({'index':index,'code':'MALFORMED_RECORD_NOT_NORMALIZED'});continue
        rows.append(row)
    bybar=defaultdict(list);byevent=defaultdict(list);counts=Counter();features=Counter()
    previous_received=None;previous_open=None;late=out_of_order=0
    for row in rows:
        for flag in row['quality_flags']:counts[flag]+=1
        for group,has in row['features']['availability'].items():features[group]+=bool(has)
        k=(row['environment'],row['source_id'],row['instrument'],row['timeframe'],row['source_open_utc'])
        if row['source_open_utc']:
            t=iso_time(row['source_open_utc'])
            if not start<=t<end:
                counts['OUTSIDE_REQUESTED_SOURCE_WINDOW']+=1
                row['quality_flags']=sorted(set(row['quality_flags'])|{'OUTSIDE_REQUESTED_SOURCE_WINDOW'})
                row['research_eligible']=False
                row['eligibility']={k:'BLOCKED_OUTSIDE_WINDOW' for k in row['eligibility']}
                row['projection_digest']=digest({k:v for k,v in row.items() if k!='projection_digest'})
            else:bybar[k].append(row)
        byevent[row['event_id']].append(row)
        try:
            received=iso_time(row['received_at']);opened=iso_time(row['source_open_utc']) if row['source_open_utc'] else None
            # Sort by receipt before analysing source ordering below; input export
            # order is event_id order, not actual delivery order.
        except ValueError:received=opened=None
        if 'LATE_DELIVERY_FOR_DECISION_LINKAGE' in row['timing_flags']:late+=1
    receipts=[]
    for row in rows:
        try:receipts.append((iso_time(row['received_at']),str(row['event_id']),iso_time(row['source_open_utc'])))
        except ValueError:pass
    previous_by_stream={}
    receipt_rows=[]
    for row in rows:
        try:receipt_rows.append((iso_time(row['received_at']),str(row['event_id']),row['instrument'],row['timeframe'],iso_time(row['source_open_utc'])))
        except ValueError:pass
    for received,_,instrument,timeframe,opened in sorted(receipt_rows):
        key=(instrument,timeframe)
        if key in previous_by_stream and opened<previous_by_stream[key]:out_of_order+=1
        previous_by_stream[key]=opened
    duplicate_bar_groups=[];conflicts=[]
    for key,values in sorted(bybar.items(),key=lambda item:str(item[0])):
        if len(values)>1:
            group={'identity':list(key),'event_ids':[x['event_id'] for x in values],'record_count':len(values)}
            duplicate_bar_groups.append(group)
            # Full payload also compared: equal OHLCV with conflicting source
            # indicators/version/confirmation is not silently deduplicated.
            if len({x['payload_digest'] for x in values})>1:
                conflicts.append(group)
    event_conflicts=[k for k,v in byevent.items() if len({x['event_evidence_digest'] for x in v})>1]
    cal=context.calendar
    missing=None;unexpected=None;calendar_verified=bool(cal and cal.approved and cal.start<=start and end<=cal.end)
    if calendar_verified:
        expected={a.isoformat() for a,b in cal.slots if start<=a<end}
        actual={r['source_open_utc'] for r in rows if r['source_open_utc'] and start<=iso_time(r['source_open_utc'])<end and r['instrument']==cal.instrument and r['timeframe']==cal.timeframe}
        missing=sorted(expected-actual);unexpected=sorted(actual-expected)
    complete=bool(not any(len(v)>1 for v in byevent.values()) and snapshot_complete and source_count is not None and type(source_count) is int and source_count==len(events) and not exceptions and not event_conflicts)
    duplicate_ids={i for group in duplicate_bar_groups for i in group['event_ids']}
    conflict_ids={i for c in conflicts for i in c['event_ids']} | set(event_conflicts)
    for row in rows:
        if row['event_id'] in duplicate_ids and row['event_id'] not in conflict_ids:
            row['quality_flags']=sorted(set(row['quality_flags'])|{'DUPLICATE_BAR_IDENTITY'})
            row['validation_state']='UNPROVED';row['research_eligible']=False
            row['eligibility']={k:'BLOCKED_DUPLICATE_REQUIRES_REVIEW' for k in row['eligibility']}
            row['projection_digest']=digest({k:v for k,v in row.items() if k!='projection_digest'})
        if row['event_id'] in conflict_ids:
            row['quality_flags']=sorted(set(row['quality_flags'])|{'DATASET_DUPLICATE_CONFLICT'})
            row['validation_state']='INVALID';row['research_eligible']=False
            row['eligibility']={k:'BLOCKED_DUPLICATE_CONFLICT' for k in row['eligibility']}
            row['projection_digest']=digest({k:v for k,v in row.items() if k!='projection_digest'})
    result={'contract':'historical_quality_v1','window':{'start':start.isoformat(),'end':end.isoformat()},'scope':{'environment':context.environment,'source_id':context.source_id,'instrument':cal.instrument if cal else None,'timeframe':cal.timeframe if cal else None},'record_count':len(events),'normalized_count':len(rows),'unique_source_candles':len(bybar),'duplicate_bar_groups':duplicate_bar_groups,'conflicting_duplicates':conflicts,'repeated_event_records':sum(len(v)-1 for v in byevent.values()),'conflicting_event_ids':event_conflicts,'validation_flag_counts':dict(counts),'invalid_timestamp_counts':{k:v for k,v in counts.items() if any(t in k for t in ('TIME','RECEIPT','CALENDAR','CLOSE','SOURCE_OPEN'))},'invalid_ohlcv_count':counts.get('INVALID_OHLCV',0),'timeframe_inconsistency_count':counts.get('DECLARED_TIMEFRAME_MISMATCH',0),'session_calendar_exception_count':sum(r['nominal_grid_aligned'] is False and r['calendar_version'] is not None and r['research_eligible'] for r in rows),'late_delivery':late,'source_time_regressions_in_receipt_order':out_of_order,'receipt_order_limitation':'received_at can be replaced on retry; original delivery sequence unproved','nominal_off_grid_count':sum(r['nominal_grid_aligned'] is False for r in rows),'calendar_verified':calendar_verified,'missing_expected_intervals':missing,'unexpected_source_slots':unexpected,'feature_availability':dict(features),'recovery_counts':dict(Counter(r['recovery_state'] for r in rows)),'eligible_record_count':sum(r['research_eligible'] for r in rows),'source_population_status':'COMPLETE_CAPTURED_SNAPSHOT' if complete else 'INCOMPLETE_OR_UNPROVED','calendar_window_status':'COMPLETE_EXPECTED_SLOTS' if calendar_verified and not missing and not unexpected else 'INCOMPLETE_OR_UNPROVED','dataset_complete_for_research':bool(complete and calendar_verified and not missing and not unexpected and not duplicate_bar_groups and not conflicts and not event_conflicts and rows and all(r['research_eligible'] for r in rows)),'malformed_records':exceptions,'normalized_rows':rows}
    result['projection_population_hash']=digest([r['projection_digest'] for r in rows])
    result['manifest_hash']=digest({k:v for k,v in result.items() if k!='normalized_rows'})
    return result

def reader_page(report,*,manifest_hash,after_event_id=None,limit=100):
    """Offline trusted library, NOT a deployed HTTP/auth boundary.

    Caller must authenticate and authorise report scope before loading a private
    dataset. No SQL or secret fields. Cursor is bound to exact manifest and
    projection population; require a unique ordered event population.
    """
    if digest({k:v for k,v in report.items() if k not in ('normalized_rows','manifest_hash')})!=report['manifest_hash'] or digest([r['projection_digest'] for r in report['normalized_rows']])!=report['projection_population_hash']:raise ValueError('REPORT_INTEGRITY_MISMATCH')
    if any(r['projection_digest']!=digest({k:v for k,v in r.items() if k!='projection_digest'}) for r in report['normalized_rows']):raise ValueError('ROW_INTEGRITY_MISMATCH')
    if manifest_hash!=report['manifest_hash']:raise ValueError('MANIFEST_MISMATCH')
    if type(limit) is not int or not 1<=limit<=1000:raise ValueError('INVALID_LIMIT')
    rows=report['normalized_rows'];ids=[r['event_id'] for r in rows]
    if any(not isinstance(r.get('instrument'),str) or not isinstance(r.get('timeframe'),str) for r in rows):raise ValueError('INVALID_ROW_IDENTIFIERS')
    if any(not isinstance(i,str) for i in ids) or len(ids)!=len(set(ids)):raise ValueError('NONUNIQUE_EVENT_POPULATION')
    ordered=sorted(rows,key=lambda r:r['event_id'])
    if after_event_id is not None and after_event_id not in ids:raise ValueError('UNKNOWN_CURSOR')
    eligible=[r for r in ordered if after_event_id is None or r['event_id']>after_event_id]
    page=eligible[:limit]
    # Only safe, bounded product fields; never original payload or raw features.
    fields=('event_id','instrument','timeframe','source_id','environment','source_open_utc','source_close_utc','payload_digest','source_version','quality_flags','timing_flags','recovery_state','research_eligible','eligibility','calendar_version')
    return {'schema_version':'historical_research_reader_v1','manifest_hash':manifest_hash,'population_hash':report['projection_population_hash'],'window_status':report['source_population_status'],'research_window_complete':report['dataset_complete_for_research'],'rows':[{k:deepcopy(r[k]) for k in fields} | {'ohlcv':deepcopy(r['ohlcv']),'feature_availability':deepcopy(r['features']['availability'])} for r in page],'next_event_id':page[-1]['event_id'] if len(eligible)>limit else None}


def audit_streams(stream_exports,contexts,*,start,end,source_counts=None,snapshot_complete=False):
    """Explicit stream registry: absent streams remain unknown, never zero-complete.

    Process each bounded stream separately, with its own calendar/evidence
    context. Population counts are independently supplied, not inferred.
    """
    if len(contexts)>35 or not contexts:raise ValueError('EXPLICIT_STREAM_REGISTRY_REQUIRED')
    source_counts=source_counts or {}
    unexpected=set(stream_exports)-set(contexts)
    if unexpected:raise ValueError('UNREGISTERED_STREAM')
    results={}
    for key,context in sorted(contexts.items()):
        instrument,timeframe=key
        if key not in stream_exports:
            results[instrument+'/'+timeframe]={'status':'NOT_ACQUIRED','record_count':None,'dataset_complete_for_research':False};continue
        events=stream_exports[key]
        if any(not isinstance(e,dict) or e.get('instrument')!=instrument or e.get('timeframe')!=timeframe for e in events):raise ValueError('CROSS_STREAM_CONTAMINATION')
        results[instrument+'/'+timeframe]=audit(events,context,start=start,end=end,source_count=source_counts.get(key),snapshot_complete=snapshot_complete)
    return {'contract':'historical_multistream_v1','streams':results,'all_streams_complete':all(r.get('dataset_complete_for_research') is True for r in results.values())}
