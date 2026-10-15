"""Deterministic source preservation. No server-time fallback or indicator recomputation.

Policies/evidence are supplied by a trusted offline operator, not source payloads.
No approved real session calendar, source registry or evidence manifest is bundled.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re

UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
TIMEFRAMES = {'1m':60,'2m':120,'3m':180,'5m':300,'10m':600}
INSTRUMENTS = frozenset({'GBPUSD','EURUSD','AUDUSD','USDCAD','USDJPY','XAUUSD','XAGUSD'})
ALIAS_TF = {str(v//60):k for k,v in TIMEFRAMES.items()} | {k:k for k in TIMEFRAMES}
NAME = re.compile(r'^[A-Za-z0-9_.:/-]{1,160}$')
MAX_PAYLOAD_BYTES = 262144
FEATURE_GROUPS = {'EMA':('ema','ema_fast','ema_slow','ema_trend'), 'RSI':('rsi',), 'ATR':('atr','atr_percent_of_price'), 'ADX':('adx','di_plus','di_minus'), 'MACD':('macd','macd_line','macd_signal','macd_hist','macd_histogram'), 'BOLLINGER':('bb','bb_basis','bb_upper','bb_lower','bb_width','bollinger_bands')}

def canonical(value):
    """Versioned JSON-value digest encoding, not original transport-byte hashing.

    PostgreSQL JSONB does not retain original HTTP JSON bytes. Decimal values
    retain numeric lexical value through a tagged canonical representation.
    """
    def tree(x):
        if x is None:return ['null']
        if isinstance(x,bool):return ['boolean',x]
        if isinstance(x,Decimal):
            if not x.is_finite():raise ValueError('nonfinite decimal')
            return ['decimal',str(x)]
        if isinstance(x,datetime):return ['string',x.isoformat()]
        if isinstance(x,str):return ['string',x]
        if isinstance(x,int):return ['integer',str(x)]
        if isinstance(x,float):
            if not Decimal(str(x)).is_finite():raise ValueError('nonfinite float')
            return ['float',repr(x)]
        if isinstance(x,dict):
            if not all(isinstance(k,str) for k in x):raise ValueError('nonstring key')
            return ['object',[[k,tree(v)] for k,v in sorted(x.items())]]
        if isinstance(x,(list,tuple)):return ['array',[tree(v) for v in x]]
        raise ValueError('unsupported value')

    return json.dumps(tree(value),sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()

def digest(value):return sha256(canonical(value)).hexdigest()

def iso_time(value):
    if not isinstance(value,str) or len(value)>40:raise ValueError('ISO_TIME_REQUIRED')
    try:d=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError:raise ValueError('MALFORMED_ISO_TIME') from None
    if d.tzinfo is None:raise ValueError('NAIVE_SOURCE_TIME')
    d=d.astimezone(UTC)
    if not 2000<=d.year<=2100:raise ValueError('EXTREME_SOURCE_TIME')
    return d

def epoch_ms(value):
    if isinstance(value,bool) or not isinstance(value,(str,int,Decimal)):raise ValueError('INTEGER_MILLISECONDS_REQUIRED')
    if len(str(value))>20:raise ValueError('EXTREME_SOURCE_TIME')
    try:n=Decimal(str(value))
    except InvalidOperation:raise ValueError('MALFORMED_MILLISECONDS') from None
    if not n.is_finite() or n!=n.to_integral_value():raise ValueError('INTEGER_MILLISECONDS_REQUIRED')
    if Decimal('946684800')<=n<Decimal('4133980800'):raise ValueError('SECONDS_IN_MILLISECONDS_FIELD')
    try:d=EPOCH+timedelta(milliseconds=int(n))
    except OverflowError:raise ValueError('EXTREME_SOURCE_TIME') from None
    if not 2000<=d.year<=2100:raise ValueError('EXTREME_SOURCE_TIME')
    return d

def number(v):
    if isinstance(v,bool) or not isinstance(v,(str,int,float,Decimal)):raise ValueError('INVALID_NUMBER')
    if len(str(v))>80:raise ValueError('INVALID_NUMBER')
    try:d=Decimal(str(v))
    except InvalidOperation:raise ValueError('INVALID_NUMBER') from None
    if not d.is_finite():raise ValueError('INVALID_NUMBER')
    return d

@dataclass(frozen=True)
class Calendar:
    """Explicit UTC open-bar slots, bound to one stream and time window.

    Slots and close times must come from an approved, versioned session calendar.
    Empty expected slots need explicit reviewed_empty; not inferred from rows.
    """
    version: str
    instrument: str
    timeframe: str
    start: datetime
    end: datetime
    slots: tuple  # (aware UTC open, aware UTC close)
    approved: bool = False
    reviewed_empty: bool = False

    def __post_init__(self):
        if not NAME.fullmatch(self.version) or self.instrument not in INSTRUMENTS or self.timeframe not in TIMEFRAMES:raise ValueError('INVALID_CALENDAR')
        if self.start.tzinfo is None or self.end.tzinfo is None or not timedelta(0)<self.end-self.start<=timedelta(days=31):raise ValueError('INVALID_CALENDAR_WINDOW')
        if not self.slots and not self.reviewed_empty:raise ValueError('EMPTY_CALENDAR_UNREVIEWED')
        object.__setattr__(self,'start',self.start.astimezone(UTC))
        object.__setattr__(self,'end',self.end.astimezone(UTC))
        if type(self.approved) is not bool or type(self.reviewed_empty) is not bool:raise ValueError('CALENDAR_APPROVAL_MUST_BE_EXPLICIT')
        if len(self.slots)>50000:raise ValueError('CALENDAR_SLOT_BOUND')
        opens=[];previous_close=None
        for a,b in self.slots:
            if a.tzinfo is None or b.tzinfo is None or not self.start<=a<self.end or not a<b or b-a>timedelta(seconds=TIMEFRAMES[self.timeframe]):raise ValueError('INVALID_CALENDAR_SLOT')
            if previous_close is not None and a<previous_close:raise ValueError('OVERLAPPING_CALENDAR_SLOTS')
            previous_close=b
            opens.append(a)
        object.__setattr__(self,'slots',tuple((a.astimezone(UTC),b.astimezone(UTC)) for a,b in self.slots))
        if opens!=sorted(set(opens)):raise ValueError('DUPLICATE_OR_UNSORTED_CALENDAR')

@dataclass(frozen=True)
class Context:
    environment: str
    source_id: str
    provenance_reference: str
    # Exact event-evidence digests from independently checked acquisition,
    # never a boolean read from the payload itself.
    verified_event_digests: frozenset = field(default_factory=frozenset)
    known_versions: frozenset = field(default_factory=frozenset)
    calendar: Calendar | None = None
    max_receipt_lead_seconds: int = 5
    late_after_seconds: int = 300

    def __post_init__(self):
        if self.environment not in {'production','shadow','test','development'} or not NAME.fullmatch(self.source_id) or not NAME.fullmatch(self.provenance_reference):raise ValueError('INVALID_CONTEXT')
        if not 0<=self.max_receipt_lead_seconds<=60 or not 0<=self.late_after_seconds<=86400:raise ValueError('INVALID_CONTEXT_TIMING')

def features(payload, source_time, source_version, payload_digest):
    nested=payload.get('indicators')
    originals={'indicators':deepcopy(nested)} if 'indicators' in payload else {}
    for k,v in payload.items():
        if any(k.lower() in names or k.lower().startswith('ema_') or k.lower().startswith('bb_') for names in FEATURE_GROUPS.values()):originals[k]=deepcopy(v)
    leaves={k.lower():v for k,v in payload.items()}
    if isinstance(nested,dict):leaves.update({k.lower():v for k,v in nested.items()})
    availability={group:any(k in leaves and leaves[k] is not None for k in names) for group,names in FEATURE_GROUPS.items()}
    return {'original':originals,'availability':availability,'source_timestamp':source_time,'source_version':source_version,'payload_digest':payload_digest,'method':'PRESERVED_SOURCE_FIELDS_NO_RECOMPUTATION'}

def normalize(event,context):
    """Return a projection; leave the supplied event/payload unchanged.

    Malformed rows are returned as invalid projections, never repaired/rounded.
    Context integrity is validated separately; source self-declared verification
    flags do not influence trust.
    """
    flags=[];block=[];timing=[];opened=closed=stored=received=None;ohlcv={};raw={};event_hash=None;payload_hash=None
    out={'normalizer_version':'historical_normalizer_v1','digest_encoding':'canonical_json_value_v1','event_id':None,'instrument':None,'timeframe':None,'source_format':'UNKNOWN','source_version':None,'source_id':context.source_id,'environment':context.environment,'provenance_references':[context.provenance_reference],'quality_flags':flags,'timing_flags':timing,'eligibility_blockers':block}
    if not isinstance(event,dict):event={};flags.append('MALFORMED_EVENT')
    original=deepcopy(event)
    out['event_id']=event.get('event_id');out['instrument']=event.get('instrument');out['timeframe']=event.get('timeframe')
    for k in ('event_id','instrument','timeframe'):
        if not isinstance(event.get(k),str) or not NAME.fullmatch(event[k]):
            flags.append('INVALID_'+k.upper());out[k]=None
    if out.get('instrument') not in INSTRUMENTS:flags.append('UNSUPPORTED_INSTRUMENT')
    if out.get('timeframe') not in TIMEFRAMES:flags.append('UNSUPPORTED_TIMEFRAME')
    if event.get('event_type')!='market_data':flags.append('WRONG_EVENT_TYPE')
    if event.get('environment')!=context.environment:flags.append('ENVIRONMENT_MISMATCH')
    if event.get('origin')!=context.source_id:flags.append('SOURCE_IDENTITY_MISMATCH')
    try:
        event_hash=digest(event)
        raw=event.get('payload')
        if not isinstance(raw,dict):raise ValueError('MALFORMED_PAYLOAD')
        if len(canonical(raw))>MAX_PAYLOAD_BYTES:raise ValueError('PAYLOAD_TOO_LARGE')
        payload_hash=digest(raw)
    except (ValueError,TypeError,RecursionError):raw={};flags.append('PAYLOAD_NOT_CANONICAL_OR_OVERSIZED')
    version=raw.get('schema_version',raw.get('version',raw.get('script_version')))
    if version is None:version=event.get('schema_version')
    out['source_version']=version if isinstance(version,(str,int)) and not isinstance(version,bool) and (not isinstance(version,str) or NAME.fullmatch(version)) else None
    if version is not None and out['source_version'] is None:flags.append('MALFORMED_SOURCE_VERSION')
    out['event_schema_version']=deepcopy(event.get('schema_version'))
    out['declared_source_versions']={k:deepcopy(raw[k]) for k in ('schema_version','version','script_version') if k in raw}
    if len(out['declared_source_versions'])>1:block.append('MULTIPLE_VERSION_DECLARATIONS_REQUIRE_MAPPING')
    out['version_evidence_basis']='payload' if any(k in raw for k in ('schema_version','version','script_version')) else 'event_envelope_only'
    if out['version_evidence_basis']=='event_envelope_only':block.append('SOURCE_SCRIPT_VERSION_UNPROVED')
    if not isinstance(version,(str,int)) or isinstance(version,bool) or str(version) not in context.known_versions:block.append('SOURCE_VERSION_UNPROVED')
    # Exact aliases only; no silent removal of provider prefixes or underscores.
    symbol=raw.get('ticker',raw.get('symbol'))
    if symbol!=out['instrument']:flags.append('PAYLOAD_INSTRUMENT_MISMATCH')
    if 'ticker' in raw and 'symbol' in raw and raw['ticker']!=raw['symbol']:flags.append('CONFLICTING_INSTRUMENT_DECLARATIONS')
    declared=raw.get('interval',raw.get('timeframe'))
    if not isinstance(declared,str) or ALIAS_TF.get(declared)!=out['timeframe']:flags.append('DECLARED_TIMEFRAME_MISMATCH')
    if 'interval' in raw and 'timeframe' in raw and str(raw['interval']) in ALIAS_TF and str(raw['timeframe']) in ALIAS_TF and ALIAS_TF[str(raw['interval'])]!=ALIAS_TF[str(raw['timeframe'])]:flags.append('CONFLICTING_TIMEFRAME_DECLARATIONS')
    if raw.get('source') is not None and raw['source']!=context.source_id:flags.append('PAYLOAD_SOURCE_MISMATCH')
    has_iso='time' in raw;has_ms='time_open' in raw
    out['source_format']='DUAL_ISO_AND_EPOCH_MS' if has_iso and has_ms else 'LEGACY_ISO' if has_iso else 'V2_EPOCH_MS' if has_ms else 'UNKNOWN'
    if not has_iso and not has_ms:flags.append('MISSING_SOURCE_TIME')
    first=second=None
    if has_iso:
        try:first=iso_time(raw['time'])
        except ValueError as exc:flags.append(str(exc))
    if has_ms:
        try:second=epoch_ms(raw['time_open'])
        except ValueError as exc:flags.append(str(exc))
    if has_iso and has_ms and first!=second:flags.append('CONFLICTING_SOURCE_TIMES')
    opened=first if has_iso else second
    if 'time_close' in raw:
        try:closed=epoch_ms(raw['time_close']) if has_ms else iso_time(raw['time_close'])
        except ValueError as exc:flags.append('CLOSE_'+str(exc))
    if opened and closed and closed<=opened:flags.append('CLOSE_NOT_AFTER_OPEN')
    if opened and closed and out['timeframe'] in TIMEFRAMES and closed-opened>timedelta(seconds=TIMEFRAMES[out['timeframe']]):flags.append('CLOSE_DURATION_MISMATCH')
    if raw.get('confirmed') is False:block.append('UNCONFIRMED_CANDLE')
    if raw.get('confirmed') not in (True,False,None):flags.append('INVALID_CONFIRMATION_STATUS')
    if raw.get('confirmed') is not True:block.append('CONFIRMATION_UNPROVED')
    for name in ('event_time','received_at'):
        try:dt=iso_time(event.get(name))
        except ValueError:dt=None;flags.append('INVALID_'+name.upper())
        if name=='event_time':stored=dt
        else:received=dt
    if opened and received:
        if opened-received>timedelta(seconds=context.max_receipt_lead_seconds):flags.append('SOURCE_AFTER_RECEIPT')
        if closed and closed-received>timedelta(seconds=context.max_receipt_lead_seconds):flags.append('CLOSE_AFTER_RECEIPT')
        latency=received-(closed or opened)
        if latency>timedelta(seconds=context.late_after_seconds):timing.append('LATE_DELIVERY_FOR_DECISION_LINKAGE')
        out['receipt_latency_seconds']=str(Decimal(latency.days*86400+latency.seconds)+Decimal(latency.microseconds)/1000000)
    else:out['receipt_latency_seconds']=None
    if opened and stored and opened!=stored:out['stored_time_disagrees']=True
    else:out['stored_time_disagrees']=False
    cal=context.calendar;calendar_ok=False
    if cal is None or not cal.approved:block.append('CALENDAR_UNPROVED')
    elif cal.instrument!=out['instrument'] or cal.timeframe!=out['timeframe']:flags.append('CALENDAR_STREAM_MISMATCH')
    elif opened and not cal.start<=opened<cal.end:block.append('CALENDAR_WINDOW_UNCOVERED')
    elif opened:
        expected=dict(cal.slots)
        if opened not in expected:flags.append('SOURCE_OPEN_OUTSIDE_CALENDAR_SLOTS')
        elif closed and closed!=expected[opened]:flags.append('CLOSE_CALENDAR_MISMATCH')
        else:calendar_ok=True
    out['calendar_version']=cal.version if cal else None
    out['calendar_digest']=digest({'version':cal.version,'stream':[cal.instrument,cal.timeframe],'start':cal.start,'end':cal.end,'slots':cal.slots,'approved':cal.approved,'reviewed_empty':cal.reviewed_empty}) if cal else None
    # Diagnostic grid only; explicit calendar slots take precedence for legitimate
    # shortened/shifted session bars. Without a calendar the grid is unproved.
    if opened and out['timeframe'] in TIMEFRAMES:
        seconds=Decimal((opened-EPOCH).days*86400+(opened-EPOCH).seconds)+Decimal(opened.microsecond)/1000000
        out['nominal_grid_aligned']=seconds % TIMEFRAMES[out['timeframe']]==0
    else:out['nominal_grid_aligned']=None
    try:
        for k in ('open','high','low','close','volume'):ohlcv[k]=number(raw.get(k))
        if any(ohlcv[k]<=0 for k in ('open','high','low','close')) or ohlcv['volume']<0 or not ohlcv['low']<=min(ohlcv['open'],ohlcv['close'])<=max(ohlcv['open'],ohlcv['close'])<=ohlcv['high']:raise ValueError('INVALID_OHLCV')
    except ValueError:flags.append('INVALID_OHLCV');ohlcv={}
    provenance_ok=event_hash is not None and event_hash in context.verified_event_digests
    if not provenance_ok:block.append('PROVENANCE_UNPROVED')
    if out['source_format']=='DUAL_ISO_AND_EPOCH_MS':block.append('DUAL_SCHEMA_REVIEW_REQUIRED')
    stamp=opened.isoformat() if opened else None
    out.update({'source_open_utc':stamp,'source_close_utc':closed.isoformat() if closed else None,'stored_event_time':deepcopy(event.get('event_time')),'received_at':deepcopy(event.get('received_at')),'payload_digest':payload_hash,'event_evidence_digest':event_hash,'original_event':original,'ohlcv':{k:str(v) for k,v in ohlcv.items()},'features':features(raw,stamp,version,payload_hash)})
    flags[:]=sorted(set(flags));block[:]=sorted(set(block))
    if not closed:block.append('CLOSE_TIME_UNPROVED')
    out['validation_state']='INVALID' if flags else 'VALIDATED' if not block else 'UNPROVED'
    # Recovery classification concerns source-time trust, separately from OHLCV.
    time_invalid=any(f!='INVALID_OHLCV' for f in flags)
    out['recovery_state']='NOT_YET_ASSESSED' if not provenance_ok else 'UNRECOVERABLE' if opened is None or 'CONFLICTING_SOURCE_TIMES' in flags else 'RECOVERABLE_WITH_SOURCE_EVIDENCE' if time_invalid or block else 'ORIGINAL_TIMESTAMP_VERIFIED'
    # UNRECOVERABLE here refers only to this source record as supplied; additional
    # independent evidence can change a later projection, never the original.
    eligible=not flags and not block and calendar_ok and bool(closed)
    out['eligibility']={'descriptive_analysis':'ELIGIBLE' if eligible else 'BLOCKED','technical_indicator_research':'ELIGIBLE_SOURCE_FEATURES_ONLY' if eligible and any(out['features']['availability'].values()) else 'BLOCKED','signal_candle_linkage':'BLOCKED_LATE_DELIVERY_CAUSALITY' if timing else 'PENDING_SIGNAL_EVIDENCE','rejected_trade_replay':'PENDING_COMPLETE_WINDOW_AND_REPLAY_VALIDATION','backtesting':'PENDING_COMPLETE_WINDOW_AND_EXECUTION_MODEL','training_datasets':'PENDING_LABEL_LINEAGE_AND_SPLIT_VALIDATION'}
    out['research_eligible']=eligible
    try:out['projection_digest']=digest({k:v for k,v in out.items() if k!='projection_digest'})
    except (ValueError,TypeError,RecursionError):out['projection_digest']=None
    return out
