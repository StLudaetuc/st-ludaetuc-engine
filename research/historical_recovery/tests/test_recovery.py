import unittest,json,tempfile,io,os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from research.historical_recovery.normalizer import *
from research.historical_recovery.auditor import audit,reader_page
from research.historical_recovery.extract import acquire,write_export,validate_bounds,redact,json_bytes,main

START='2026-10-13T00:00:00+00:00';END='2026-10-13T00:03:00+00:00'

def fixture(kind='legacy',offset=0,event_id=None):
 t=iso_time(START)+timedelta(minutes=offset)
 payload={'ticker':'GBPUSD','interval':'1','schema_version':'legacy-logger-fixture-v1','source':'tradingview','confirmed':True,'open':'1.2','high':'1.3','low':'1.1','close':'1.25','volume':'100','indicators':{'ema_fast':'1.21','rsi':'55','atr':'0.01','adx':'20','macd_line':'0.002','bb_upper':'1.3','unmapped_source_field':None}}
 if kind=='legacy':payload['time']=t.isoformat();payload['time_close']=(t+timedelta(minutes=1)).isoformat()
 else:payload.update(schema_version='v2-logger-fixture-v2',time_open=int((t-EPOCH).total_seconds()*1000),time_close=int((t+timedelta(minutes=1)-EPOCH).total_seconds()*1000))
 return {'event_id':event_id or f'{offset+1:064x}','event_type':'market_data','origin':'tradingview','instrument':'GBPUSD','timeframe':'1m','event_time':(t+timedelta(seconds=3)).isoformat(),'received_at':(t+timedelta(minutes=1,seconds=2)).isoformat(),'schema_version':'engine-event-v1','environment':'test','retry_count':0,'payload':payload}

def calendar(approved=True):
 a=iso_time(START)
 return Calendar('SYNTHETIC_UTC_CALENDAR_v1','GBPUSD','1m',a,iso_time(END),tuple((a+timedelta(minutes=i),a+timedelta(minutes=i+1)) for i in range(3)),approved)

def context(*events,cal=True,provenance=True):
 return Context('test','tradingview','synthetic-fixture-manifest-v1',frozenset(digest(e) for e in events) if provenance else frozenset(),frozenset({'legacy-logger-fixture-v1','v2-logger-fixture-v2'}),calendar() if cal else None)

class NormalizerTests(unittest.TestCase):
 def norm(self,e):return normalize(e,context(e))
 def test_legacy_positive(self):
  e=fixture();r=self.norm(e);self.assertEqual(r['source_open_utc'],START);self.assertTrue(r['research_eligible']);self.assertEqual(r['recovery_state'],'ORIGINAL_TIMESTAMP_VERIFIED')
 def test_v2_positive(self):
  e=fixture('v2');r=self.norm(e);self.assertTrue(r['research_eligible']);self.assertEqual(r['source_open_utc'],START)
 def test_iso_timezone_conversion(self):self.assertEqual(iso_time('2026-10-13T01:00:00+01:00').isoformat(),START)
 def test_no_receipt_fallback(self):
  e=fixture();del e['payload']['time'];r=self.norm(e);self.assertIsNone(r['source_open_utc']);self.assertIn('MISSING_SOURCE_TIME',r['quality_flags']);self.assertFalse(r['research_eligible'])
 def test_malformed_source(self):
  for value in ['{{time}}','bad','',None,True,1791849600000]:
   e=fixture();e['payload']['time']=value;r=self.norm(e);self.assertIsNone(r['source_open_utc']);self.assertFalse(r['research_eligible'])
 def test_naive(self):
  e=fixture();e['payload']['time']='2026-10-13T00:00:00';self.assertIn('NAIVE_SOURCE_TIME',self.norm(e)['quality_flags'])
 def test_seconds_confusion(self):
  e=fixture('v2');e['payload']['time_open']//=1000;self.assertIn('SECONDS_IN_MILLISECONDS_FIELD',self.norm(e)['quality_flags'])
 def test_fractional_milliseconds(self):
  e=fixture('v2');e['payload']['time_open']=str(e['payload']['time_open'])+'.5';self.assertIn('INTEGER_MILLISECONDS_REQUIRED',self.norm(e)['quality_flags'])
 def test_float_ms_rejected(self):
  e=fixture('v2');e['payload']['time_open']=float(e['payload']['time_open']);self.assertFalse(self.norm(e)['research_eligible'])
 def test_extreme(self):
  for v in [-99999999999999999999,99999999999999999999,0,True,'NaN']:
   with self.assertRaises(ValueError):epoch_ms(v)
 def test_extreme_iso(self):
  with self.assertRaises(ValueError):iso_time('1900-01-01T00:00:00Z')
 def test_dual_disagreement(self):
  e=fixture();e['payload']['time_open']=int((iso_time(START)-EPOCH).total_seconds()*1000)+60000;r=self.norm(e);self.assertIn('CONFLICTING_SOURCE_TIMES',r['quality_flags'])
 def test_dual_agreement_held(self):
  e=fixture('v2');e['payload']['time']=START;r=self.norm(e);self.assertIn('DUAL_SCHEMA_REVIEW_REQUIRED',r['eligibility_blockers']);self.assertFalse(r['research_eligible'])
 def test_close_before_open(self):
  e=fixture();e['payload']['time_close']=START;self.assertIn('CLOSE_NOT_AFTER_OPEN',self.norm(e)['quality_flags'])
 def test_missing_close(self):
  e=fixture();del e['payload']['time_close'];r=self.norm(e);self.assertIn('CLOSE_TIME_UNPROVED',r['eligibility_blockers']);self.assertFalse(r['research_eligible'])
 def test_long_close(self):
  e=fixture();e['payload']['time_close']=END;self.assertIn('CLOSE_DURATION_MISMATCH',self.norm(e)['quality_flags'])
 def test_off_grid_not_rounded(self):
  e=fixture();e['payload']['time']='2026-10-13T00:00:01+00:00';r=self.norm(e);self.assertEqual(r['source_open_utc'],'2026-10-13T00:00:01+00:00');self.assertIn('SOURCE_OPEN_OUTSIDE_CALENDAR_SLOTS',r['quality_flags'])
 def test_no_calendar_not_false_invalid(self):
  e=fixture();r=normalize(e,context(e,cal=False));self.assertEqual(r['validation_state'],'UNPROVED');self.assertIn('CALENDAR_UNPROVED',r['eligibility_blockers']);self.assertFalse(r['research_eligible'])
 def test_unapproved_calendar(self):
  e=fixture();r=normalize(e,replace(context(e),calendar=calendar(False)));self.assertFalse(r['research_eligible'])
 def test_dst_fall(self):
  z=ZoneInfo('Europe/London');a=datetime(2026,10,25,1,30,tzinfo=z,fold=0);b=datetime(2026,10,25,1,30,tzinfo=z,fold=1);self.assertNotEqual(iso_time(a.isoformat()),iso_time(b.isoformat()))
 def test_dst_spring_utc(self):self.assertEqual((iso_time('2026-03-29T02:00:00+01:00')-iso_time('2026-03-29T00:59:00Z')).total_seconds(),60)
 def test_session_shift_allowed_explicit(self):
  e=fixture();a=iso_time(START)+timedelta(seconds=15);b=a+timedelta(seconds=30);e['payload'].update(time=a.isoformat(),time_close=b.isoformat());c=Calendar('SYNTHETIC_SHORT_BAR_v1','GBPUSD','1m',iso_time(START),iso_time(END),((a,b),),True);r=normalize(e,replace(context(e),calendar=c));self.assertTrue(r['research_eligible']);self.assertFalse(r['nominal_grid_aligned'])
 def test_declared_timeframe(self):
  e=fixture();e['payload']['interval']='5';self.assertIn('DECLARED_TIMEFRAME_MISMATCH',self.norm(e)['quality_flags'])
 def test_cross_symbol(self):
  e=fixture();e['payload']['ticker']='EURUSD';self.assertIn('PAYLOAD_INSTRUMENT_MISMATCH',self.norm(e)['quality_flags'])
 def test_source_identity(self):
  e=fixture();e['origin']='unknown';self.assertIn('SOURCE_IDENTITY_MISMATCH',self.norm(e)['quality_flags'])
 def test_environment(self):
  e=fixture();e['environment']='production';self.assertIn('ENVIRONMENT_MISMATCH',self.norm(e)['quality_flags'])
 def test_wrong_event_type(self):
  e=fixture();e['event_type']='prediction';self.assertIn('WRONG_EVENT_TYPE',self.norm(e)['quality_flags'])
 def test_invalid_ohlcv(self):
  for key,val in [('low','2'),('high','0'),('volume','-1'),('open','NaN'),('close',True),('volume',None)]:
   e=fixture();e['payload'][key]=val;self.assertIn('INVALID_OHLCV',self.norm(e)['quality_flags'])
 def test_missing_indicators_not_fabricated(self):
  e=fixture('v2');del e['payload']['indicators'];r=self.norm(e);self.assertTrue(r['research_eligible']);self.assertFalse(any(r['features']['availability'].values()));self.assertEqual(r['features']['original'],{})
 def test_original_features_preserved(self):
  e=fixture();r=self.norm(e);self.assertEqual(r['features']['original']['indicators'],e['payload']['indicators']);self.assertTrue(all(r['features']['availability'].values()))
 def test_no_input_mutation(self):
  e=fixture();before=deepcopy(e);r=self.norm(e);r['original_event']['payload']['indicators']['rsi']='99';self.assertEqual(e,before)
 def test_receipt_before_open(self):
  e=fixture();e['received_at']='2026-10-12T23:00:00Z';self.assertIn('SOURCE_AFTER_RECEIPT',self.norm(e)['quality_flags'])
 def test_delayed_arrival(self):
  e=fixture();e['received_at']='2026-10-13T02:00:00Z';r=self.norm(e);self.assertNotIn('SOURCE_AFTER_RECEIPT',r['quality_flags']);self.assertIn('LATE_DELIVERY_FOR_DECISION_LINKAGE',r['timing_flags']);self.assertTrue(r['research_eligible']);self.assertEqual(r['eligibility']['signal_candle_linkage'],'BLOCKED_LATE_DELIVERY_CAUSALITY')
 def test_stored_time_disagreement_preserved(self):
  e=fixture();r=self.norm(e);self.assertTrue(r['stored_time_disagrees']);self.assertEqual(r['stored_event_time'],e['event_time'])
 def test_provenance_missing(self):
  e=fixture();r=normalize(e,context(e,provenance=False));self.assertEqual(r['recovery_state'],'NOT_YET_ASSESSED');self.assertFalse(r['research_eligible'])
 def test_payload_self_verification_ignored(self):
  e=fixture();e['payload']['verified']=True;r=normalize(e,context(e,provenance=False));self.assertFalse(r['research_eligible'])
 def test_unknown_version(self):
  e=fixture();e['payload']['schema_version']='uninspected';self.assertIn('SOURCE_VERSION_UNPROVED',self.norm(e)['eligibility_blockers'])
 def test_envelope_not_script_version(self):
  e=fixture();del e['payload']['schema_version'];r=self.norm(e);self.assertIn('SOURCE_SCRIPT_VERSION_UNPROVED',r['eligibility_blockers'])
 def test_confirmation_unknown(self):
  e=fixture();del e['payload']['confirmed'];self.assertFalse(self.norm(e)['research_eligible'])
 def test_json_digest_type_collision(self):self.assertNotEqual(digest(Decimal('1')),digest({'__decimal__':'1'}));self.assertNotEqual(digest(1),digest('1'))
 def test_deterministic(self):e=fixture();self.assertEqual(self.norm(e),self.norm(e))

class AuditTests(unittest.TestCase):
 def report(self,rows,complete=True,count=None):return audit(rows,context(*rows),start=START,end=END,source_count=len(rows) if count is None else count,snapshot_complete=complete)
 def test_complete_positive(self):
  r=self.report([fixture(offset=i) for i in range(3)]);self.assertTrue(r['dataset_complete_for_research']);self.assertEqual(r['unique_source_candles'],3)
 def test_partial_missing_intervals(self):
  r=self.report([fixture()]);self.assertEqual(len(r['missing_expected_intervals']),2);self.assertFalse(r['dataset_complete_for_research'])
 def test_snapshot_unproved(self):self.assertFalse(self.report([fixture(offset=i) for i in range(3)],False)['dataset_complete_for_research'])
 def test_count_mismatch(self):self.assertEqual(self.report([fixture()],count=99)['source_population_status'],'INCOMPLETE_OR_UNPROVED')
 def test_duplicate_event(self):
  e=fixture();r=self.report([e,e]);self.assertEqual(r['repeated_event_records'],1);self.assertFalse(r['dataset_complete_for_research'])
 def test_duplicate_bar_new_event_id(self):
  e=fixture();r=self.report([e,dict(e,event_id='a'*64)]);self.assertEqual(len(r['duplicate_bar_groups']),1);self.assertFalse(r['conflicting_duplicates'])
 def test_conflicting_ohlcv(self):
  e=fixture();b=deepcopy(e);b['event_id']='b'*64;b['payload']['close']='1.26';r=self.report([e,b]);self.assertEqual(len(r['conflicting_duplicates']),1);self.assertFalse(any(x['research_eligible'] for x in r['normalized_rows']))
 def test_conflicting_indicator(self):
  e=fixture();b=deepcopy(e);b['event_id']='b'*64;b['payload']['indicators']['rsi']='66';self.assertTrue(self.report([e,b])['conflicting_duplicates'])
 def test_conflicting_same_id(self):
  e=fixture();b=deepcopy(e);b['received_at']='2026-10-13T00:02:00Z';self.assertTrue(self.report([e,b])['conflicting_event_ids'])
 def test_receipt_order(self):
  a,b=fixture(),fixture(offset=1);a['received_at']='2026-10-13T00:03:30Z';r=self.report([a,b]);self.assertEqual(r['source_time_regressions_in_receipt_order'],1)
 def test_export_order_not_receipt_order(self):
  a,b=fixture(),fixture(offset=1);self.assertEqual(self.report([b,a])['source_time_regressions_in_receipt_order'],0)
 def test_no_calendar_missing_unproved(self):
  e=fixture();r=audit([e],context(e,cal=False),start=START,end=END);self.assertIsNone(r['missing_expected_intervals'])
 def test_empty_not_complete(self):self.assertFalse(self.report([])['dataset_complete_for_research'])
 def test_bounded_window(self):
  with self.assertRaises(ValueError):audit([],context(),start=START,end='2027-01-01T00:00:00Z')
 def test_reader_pages(self):
  r=self.report([fixture(offset=i) for i in range(3)]);a=reader_page(r,manifest_hash=r['manifest_hash'],limit=2);b=reader_page(r,manifest_hash=r['manifest_hash'],after_event_id=a['next_event_id'],limit=2);self.assertEqual(len(a['rows'])+len(b['rows']),3);self.assertIsNone(b['next_event_id']);self.assertNotIn('original_event',a['rows'][0])
 def test_reader_bad_manifest(self):
  r=self.report([fixture()]);
  with self.assertRaises(ValueError):reader_page(r,manifest_hash='bad')
 def test_reader_unknown_cursor(self):
  r=self.report([fixture()]);
  with self.assertRaises(ValueError):reader_page(r,manifest_hash=r['manifest_hash'],after_event_id='bad')
 def test_reader_tampering(self):
  r=self.report([fixture()]);r['normalized_rows'][0]['ohlcv']['close']='999'
  with self.assertRaises(ValueError):reader_page(r,manifest_hash=r['manifest_hash'])
 def test_reader_duplicate_ids(self):
  e=fixture();r=self.report([e,e]);
  with self.assertRaises(ValueError):reader_page(r,manifest_hash=r['manifest_hash'])

class FakeCursor:
 def __init__(self,conn):self.c=conn;self.description=None;self.result=[]
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def execute(self,sql,params=()):
  self.c.queries.append((sql,params))
  if 'current_setting' in sql:self.result=[('on' if self.c.readonly else 'off','repeatable read','synthetic-snapshot','synthetic-reader')]
  elif 'COUNT(*)' in sql:self.result=[(len(self.c.rows),)]
  elif sql.startswith('SELECT event_id'):
   after=params[-2] if 'event_id COLLATE "C" >' in sql else None;size=params[-1];rows=[x for x in self.c.rows if after is None or x['event_id']>after][:size]
   if self.c.corrupt_page and len(self.c.queries)>4 and rows:rows=[self.c.rows[0]]
   self.description=[(k,) for k in self.c.rows[0]] if self.c.rows else [];self.result=[tuple(x.values()) for x in rows]
  else:self.result=[]
 def fetchone(self):return self.result[0]
 def fetchall(self):return self.result
class FakeConnection:
 def __init__(self,rows,readonly=True,corrupt_page=False):self.rows=rows;self.queries=[];self.rollback_count=0;self.readonly=readonly;self.corrupt_page=corrupt_page;self.closed=False
 def set_session(self,**kw):self.options=kw
 def cursor(self):return FakeCursor(self)
 def rollback(self):self.rollback_count+=1
 def close(self):self.closed=True

class AcquisitionTests(unittest.TestCase):
 def acquire(self,conn,**kw):return acquire(conn,instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END,**kw)
 def test_positive_pagination(self):
  c=FakeConnection([fixture(offset=i) for i in range(3)]);rows,m=self.acquire(c,batch_size=1);self.assertEqual(len(rows),3);self.assertTrue(m['complete_captured_snapshot']);self.assertEqual(len(m['pages']),3);self.assertEqual(c.rollback_count,1);self.assertTrue(c.options['readonly'])
 def test_cap_partial(self):
  c=FakeConnection([fixture(offset=i) for i in range(3)]);rows,m=self.acquire(c,batch_size=1,max_rows=2);self.assertFalse(m['complete_captured_snapshot']);self.assertEqual(m['exported_count'],2)
 def test_resume_not_full_snapshot(self):
  c=FakeConnection([fixture(offset=i) for i in range(3)]);rows,m=self.acquire(c,after_event_id=fixture()['event_id']);self.assertFalse(m['complete_captured_snapshot']);self.assertEqual(len(rows),2)
 def test_nonmonotonic_page(self):
  c=FakeConnection([fixture(offset=i) for i in range(3)],corrupt_page=True)
  with self.assertRaises(ValueError):self.acquire(c,batch_size=1)
  self.assertEqual(c.rollback_count,1)
 def test_readwrite_denied(self):
  c=FakeConnection([fixture()],readonly=False)
  with self.assertRaises(ValueError):self.acquire(c)
  self.assertEqual(c.rollback_count,1)
 def test_cross_instrument_contamination(self):
  e=fixture();e['instrument']='EURUSD';c=FakeConnection([e])
  with self.assertRaises(ValueError):self.acquire(c)
 def test_bounds(self):
  for size in [0,-1,1001]:
   with self.assertRaises(ValueError):validate_bounds('GBPUSD','1m','test',START,END,size,1000)
 def test_redaction(self):
  r,paths=redact({'token':'secret','nested':{'account_id':'account','text':'postgres://user:password@host'},'rsi':'55'});self.assertNotIn('password',json.dumps(r));self.assertNotIn('secret',json.dumps(r));self.assertEqual(r['rsi'],'55');self.assertEqual(len(paths),3)
 def test_write_private_and_no_overwrite(self):
  c=FakeConnection([fixture()]);rows,m=self.acquire(c)
  with tempfile.TemporaryDirectory() as tmp:
   target=Path(tmp)/'export';write_export(target,rows,m);self.assertEqual((target/'events.jsonl').stat().st_mode & 0o777,0o600);self.assertEqual(target.stat().st_mode & 0o777,0o700)
   with self.assertRaises(FileExistsError):write_export(target,rows,m)
 def test_no_credentials_error_output(self):
  stream=io.StringIO()
  with patch.dict(os.environ,{},clear=True),patch('sys.stdout',stream):code=main(['--instrument','GBPUSD','--timeframe','1m','--environment','test','--start',START,'--end',END,'--output','unused'])
  self.assertEqual(code,1);self.assertNotIn('password',stream.getvalue());self.assertFalse(Path('unused').exists())
 def test_sql_no_writes(self):
  c=FakeConnection([fixture()]);self.acquire(c)
  self.assertTrue(all(sql.startswith(('SELECT ','SET LOCAL ')) for sql,params in c.queries));self.assertFalse(any('DATABASE_URL' in sql for sql,params in c.queries))
 def test_decimal_json_roundtrip(self):self.assertEqual(json.loads(json_bytes({'price':Decimal('1.234567890123456789')}),parse_float=Decimal)['price'],Decimal('1.234567890123456789'))


class ProjectionTests(unittest.TestCase):
 def test_export_project_chain(self):
  from research.historical_recovery.project import project_export
  c=FakeConnection([fixture(offset=i) for i in range(3)])
  rows,m=acquire(c,instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'export';write_export(p,rows,m);r=project_export(p,start=START,end=END)
   self.assertEqual(r['record_count'],3);self.assertFalse(r['dataset_complete_for_research']);self.assertTrue(r['receipt_snapshot_complete']);self.assertEqual(r['recovery_counts'],{'NOT_YET_ASSESSED':3})
 def test_export_tamper_denied(self):
  from research.historical_recovery.project import project_export
  rows,m=acquire(FakeConnection([fixture()]),instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'export';write_export(p,rows,m);(p/'events.jsonl').write_text((p/'events.jsonl').read_text().replace('1.25','1.29'))
   with self.assertRaises(ValueError):project_export(p,start=START,end=END)
 def test_decimal_and_datetime_digest_export(self):
  from research.historical_recovery.project import project_export
  e=fixture();e['event_time']=iso_time(e['event_time']);e['received_at']=iso_time(e['received_at']);e['payload']['close']=Decimal('1.250001')
  rows,m=acquire(FakeConnection([e]),instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'export';write_export(p,rows,m);self.assertEqual(project_export(p,start=START,end=END)['record_count'],1)
 def test_strict_duplicate_json(self):
  from research.historical_recovery.project import strict_json
  with self.assertRaises(ValueError):strict_json('{"x":1,"x":2}')
 def test_strict_nan_json(self):
  from research.historical_recovery.project import strict_json
  with self.assertRaises(ValueError):strict_json('{"x":NaN}')
 def test_reader_result_contract(self):
  r=audit([fixture()],context(fixture()),start=START,end=END)
  p=reader_page(r,manifest_hash=r['manifest_hash']);self.assertEqual(p['schema_version'],'historical_research_reader_v1');self.assertFalse(p['research_window_complete'])
 def test_invalid_row_does_not_abort_population(self):
  e=fixture();r=audit([e,{'event_type':'prediction'}],context(e),start=START,end=END);self.assertEqual(r['record_count'],2);self.assertFalse(r['dataset_complete_for_research'])
 def test_future_source_close_rejected(self):
  e=fixture();e['received_at']=START;r=normalize(e,context(e));self.assertIn('CLOSE_AFTER_RECEIPT',r['quality_flags'])
 def test_negative_volume(self):
  e=fixture();e['payload']['volume']=-1;self.assertFalse(normalize(e,context(e))['research_eligible'])
 def test_event_digest_provenance_stale(self):
  e=fixture();c=context(e);e['payload']['indicators']['rsi']='99';self.assertIn('PROVENANCE_UNPROVED',normalize(e,c)['eligibility_blockers'])

class MultiStreamTests(unittest.TestCase):
 def test_missing_stream_not_zero(self):
  from research.historical_recovery.auditor import audit_streams
  r=audit_streams({}, {('GBPUSD','1m'):context()},start=START,end=END);self.assertIsNone(r['streams']['GBPUSD/1m']['record_count']);self.assertFalse(r['all_streams_complete'])
 def test_cross_stream_rejected(self):
  from research.historical_recovery.auditor import audit_streams
  with self.assertRaises(ValueError):audit_streams({('EURUSD','1m'):[fixture()]},{('EURUSD','1m'):context()},start=START,end=END)
 def test_one_stream(self):
  from research.historical_recovery.auditor import audit_streams
  rows=[fixture(offset=i) for i in range(3)];key=('GBPUSD','1m');r=audit_streams({key:rows},{key:context(*rows)},start=START,end=END,source_counts={key:3},snapshot_complete=True);self.assertTrue(r['all_streams_complete'])

class HardeningTests(unittest.TestCase):
 def test_non_string_identifiers_isolate(self):
  e=fixture();e['event_id']={'secret':'do-not-expose'};e['instrument']={};r=audit([e,fixture(offset=1)],context(fixture(offset=1)),start=START,end=END);self.assertEqual(r['record_count'],2);self.assertFalse(r['dataset_complete_for_research'])
 def test_version_object_not_exposed(self):
  e=fixture();e['payload']['schema_version']={'secret':'hidden'};r=normalize(e,context(e));self.assertIsNone(r['source_version']);self.assertIn('MALFORMED_SOURCE_VERSION',r['quality_flags'])
 def test_reader_row_bound(self):
  r=audit([fixture()],context(fixture()),start=START,end=END)
  for v in [0,1001,True]:
   with self.assertRaises(ValueError):reader_page(r,manifest_hash=r['manifest_hash'],limit=v)
 def test_manifest_summary_tamper(self):
  r=audit([fixture()],context(fixture()),start=START,end=END);r['dataset_complete_for_research']=True
  with self.assertRaises(ValueError):reader_page(r,manifest_hash=r['manifest_hash'])
 def test_unreviewed_empty_calendar(self):
  with self.assertRaises(ValueError):Calendar('SYNTHETIC_EMPTY_v1','GBPUSD','1m',iso_time(START),iso_time(END),(),True)
 def test_same_clock_determinism_not_walltime(self):
  e=fixture();a=normalize(e,context(e));b=normalize(e,context(e));self.assertEqual(a['projection_digest'],b['projection_digest'])

class AmbiguityTests(unittest.TestCase):
 def test_multiple_versions_held(self):
  e=fixture();e['payload']['script_version']='different-source-script-v2';self.assertIn('MULTIPLE_VERSION_DECLARATIONS_REQUIRE_MAPPING',normalize(e,context(e))['eligibility_blockers'])
 def test_instrument_conflict(self):
  e=fixture();e['payload']['symbol']='EURUSD';self.assertIn('CONFLICTING_INSTRUMENT_DECLARATIONS',normalize(e,context(e))['quality_flags'])
 def test_timeframe_conflict(self):
  e=fixture();e['payload']['timeframe']='5m';self.assertIn('CONFLICTING_TIMEFRAME_DECLARATIONS',normalize(e,context(e))['quality_flags'])

class DuplicateWindowTest(unittest.TestCase):
 def test_complete_slots_plus_replay_not_research_complete(self):
  rows=[fixture(offset=i) for i in range(3)];rows.append(dict(rows[0],event_id='b'*64))
  r=audit(rows,context(*rows),start=START,end=END,source_count=4,snapshot_complete=True)
  self.assertFalse(r['dataset_complete_for_research']);self.assertEqual(r['eligible_record_count'],2)


class RawDigestLineageTest(unittest.TestCase):
 def test_redacted_payload_retains_original_digest(self):
  from research.historical_recovery.project import project_export
  e=fixture();e['payload']['token']='SYNTHETIC_SECRET';original=digest(e['payload'])
  rows,m=acquire(FakeConnection([e]),instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'export';write_export(p,rows,m);r=project_export(p,start=START,end=END);record=r['normalized_rows'][0]
   self.assertEqual(record['source_original_payload_digest'],original);self.assertNotEqual(record['payload_digest'],original);self.assertTrue(record['redaction_applied']);self.assertNotIn('SYNTHETIC_SECRET',json.dumps(r))

if __name__=='__main__':unittest.main(verbosity=2)
