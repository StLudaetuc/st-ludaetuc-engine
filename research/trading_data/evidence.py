import copy
import hashlib
import json
from threading import RLock
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation

INSTRUMENTS=frozenset(('GBPUSD','EURUSD','AUDUSD','USDCAD','USDJPY','XAUUSD','XAGUSD'))
KINDS=frozenset(('prediction','trade_opened','broker_fill','trade_outcome','market_data','order_request','correction'))

def timestamp(value):
    if not isinstance(value,str) or len(value)>64:raise ValueError('timestamp required')
    d=datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset() is None:raise ValueError('explicit source timezone required')
    return d.astimezone(timezone.utc)

def instrument(value):
    if not isinstance(value,str) or '{{' in value:raise ValueError('instrument missing/placeholder')
    key=value.replace('/','').replace('_','').upper()
    if key not in INSTRUMENTS:raise ValueError('unreviewed instrument')
    return key

def number(value):
    if isinstance(value,bool) or not isinstance(value,(str,int,Decimal)):raise ValueError('exact decimal required')
    try:d=Decimal(value)
    except InvalidOperation:raise ValueError('malformed decimal')
    if not d.is_finite() or abs(d)>=Decimal('1e18'):raise ValueError('invalid number')
    return d

def identity(value):
    if not isinstance(value,str) or not value or len(value)>160 or any(ord(c)<32 for c in value):raise ValueError('bounded source identity required')
    return value

def canonical(row):
    def decimal(v):
        if isinstance(v,Decimal):return str(v)
        raise TypeError('non-json source value')
    s=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False,default=decimal)
    if len(s.encode())>65536:raise ValueError('bounded evidence64KiB')
    return s

class Journal:
    """Reference append-only journal. Conflicts retained, never supersede silently.
    Receipt/source clocks do not determine canonical correction acceptance.
    A durable transactional writer is a separate reviewed integration step.
    """
    def __init__(self):self._facts=[];self._deliveries=[];self._heads={};self._lock=RLock()
    def append(self,row):
        with self._lock:return self._append(row)
    def _append(self,row):
        row=copy.deepcopy(row)
        for f in ('event_id','source_event_id','account_alias','broker_environment'):identity(row.get(f))
        if row['broker_environment'] not in ('OANDA_PRACTICE','OANDA_LIVE'):raise ValueError('explicit broker environment')
        if row.get('kind') not in KINDS or not isinstance(row.get('payload'),dict):raise ValueError('event schema')
        timestamp(row.get('event_time'));timestamp(row.get('received_at'));instrument(row.get('instrument'))
        body={k:v for k,v in row.items() if k not in ('received_at','delivery_id')}
        digest=hashlib.sha256(canonical(body).encode()).hexdigest()
        key=(row['account_alias'],row['broker_environment'],row['kind'],row['source_event_id'])
        versions=self._heads.setdefault(key,{})
        if digest in versions:state='EXACT_REPLAY';seq=versions[digest]
        else:
            state='CONFLICT' if versions else 'NEW_FACT'
            if row['kind']=='correction':
                previous=row['payload'].get('supersedes_event_id')
                matches=[r for r in self._facts if r['event']['event_id']==previous and r['event']['account_alias']==row['account_alias'] and r['event']['broker_environment']==row['broker_environment']]
                if len(matches)!=1 or not row['payload'].get('review_evidence'):raise ValueError('reviewed same-scope correction lineage required')
            if any(r['event']['event_id']==row['event_id'] and (r['event']['account_alias'],r['event']['broker_environment'])==(row['account_alias'],row['broker_environment']) for r in self._facts):state='CONFLICT'
            seq=len(self._facts)+1;versions[digest]=seq
            self._facts.append({'sequence':seq,'digest':digest,'classification':state,'event':row})
        self._deliveries.append({'sequence':len(self._deliveries)+1,'fact_sequence':seq,'received_at':row['received_at'],'state':state})
        return state
    @property
    def facts(self):return copy.deepcopy(tuple(self._facts))
    @property
    def deliveries(self):return copy.deepcopy(tuple(self._deliveries))
