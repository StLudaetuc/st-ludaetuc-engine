import json,unittest,tempfile,hashlib
from dataclasses import replace
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
from research.historical_recovery.tests.test_recovery import fixture,context,calendar,FakeConnection,START,END
from research.historical_recovery.normalizer import *
from research.historical_recovery.auditor import audit
from research.historical_recovery.extract import acquire,redact,write_export,json_bytes,verify_reader_identity
from research.historical_recovery.project import project_export,strict_json
from research.historical_recovery.anchors import TrustedAnchor

# Test-only external expectation stand-in. Real anchors require independent custody.
def make_anchor(rows,m,folder):
 return TrustedAnchor(m['manifest_hash'],hashlib.sha256((folder/'export_audit.json').read_bytes()).hexdigest(),m['source_count'],len(rows),m['scope'],m['window'],m['acquired_at'],m['operator_identity_reference'],m['contract'],m['redaction_policy'],tuple((r['event_id'],r['acquisition_source_event_digest'],r['acquisition_redacted_event_digest'],r['acquisition_source_payload_digest']) for r in rows),'SYNTHETIC_INDEPENDENT_CUSTODY_FIXTURE')
def export(e,p):
 rows,m=acquire(FakeConnection(e),instrument='GBPUSD',timeframe='1m',environment='test',start=START,end=END);write_export(p,rows,m);return rows,m

class TimeV11(unittest.TestCase):
 def test_timezone_hashes_601_three_zones(self):
  for i in range(601):
   e=fixture(offset=i%3);t=iso_time(e['event_time']);receipt=iso_time(e['received_at']);h=[]
   for z in ('UTC','Europe/London','America/New_York'):
    e['event_time']=t.astimezone(ZoneInfo(z));e['received_at']=receipt.astimezone(ZoneInfo(z));h.append(digest(e))
   self.assertEqual(len(set(h)),1)
 def test_fixed_precision(self):self.assertEqual(canonical_timestamp(datetime(2026,10,13,tzinfo=timezone.utc)),'2026-10-13T00:00:00.000000Z')
 def test_roundtrip_timezone_fidelity(self):
  x=datetime(2026,10,13,tzinfo=ZoneInfo('Asia/Tokyo'));self.assertEqual(digest(x),digest(strict_json(json_bytes(x))))
 def test_naive_canonical_rejected(self):
  with self.assertRaises(ValueError):digest(datetime(2026,10,13))
 def test_submicrosecond_iso_rejected(self):
  for x in ('2026-10-13T00:00:00.1234567Z','2026-10-13T00:00:00,1234567+00:00'):
   with self.assertRaises(ValueError):iso_time(x)
 def test_microsecond_iso_preserved(self):self.assertEqual(iso_time('2026-10-13T00:00:00.123456Z').microsecond,123456)
 def test_exponent_ms_rejected(self):
  for x in ('1.7918496e12',Decimal('1.7918496E+12'),1791849600000.0):
   with self.assertRaises(ValueError):epoch_ms(x)
 def test_integer_ms_valid(self):self.assertEqual(epoch_ms('1791849600000'),epoch_ms(1791849600000))
 def test_original_time_preserved(self):
  e=fixture();e['payload']['time']='2026-10-13T01:00:00+01:00';r=normalize(e,context(e));self.assertEqual(r['original_event']['payload']['time'],e['payload']['time'])

class AnchorV11(unittest.TestCase):
 def test_unanchored_consistency_not_authenticity(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p);r=project_export(p,start=START,end=END,context=context(fixture()));self.assertEqual(r['evidence_authenticity']['state'],'UNPROVED');self.assertFalse(r['normalized_rows'][0]['research_eligible'])
 def test_required_anchor_missing(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p)
   with self.assertRaisesRegex(ValueError,'INDEPENDENT_ANCHOR_REQUIRED'):project_export(p,start=START,end=END,require_anchor=True)
 def test_valid_expected_anchor(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';rows,m=export([fixture()],p);a=make_anchor(rows,m,p);r=project_export(p,start=START,end=END,trusted_anchor=a,require_anchor=True);self.assertEqual(r['evidence_authenticity']['state'],'MATCHED_INDEPENDENT_EXPECTATION')
 def test_recomputed_tampering_wrong_external_anchor(self):
  with tempfile.TemporaryDirectory() as t:
   p,q=Path(t)/'one',Path(t)/'two';rows,m=export([fixture()],p);a=make_anchor(rows,m,p);e=fixture();e['payload']['close']='1.26';export([e],q)
   # Entire forged export is internally self-consistent; pinned expectation rejects it.
   self.assertEqual(project_export(q,start=START,end=END)['evidence_authenticity']['state'],'UNPROVED')
   with self.assertRaisesRegex(ValueError,'EXPECTED_MANIFEST_MISMATCH'):project_export(q,start=START,end=END,trusted_anchor=a)
 def test_wrong_expected_audit(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';rows,m=export([fixture()],p);a=replace(make_anchor(rows,m,p),expected_audit_sha256='a'*64)
   with self.assertRaisesRegex(ValueError,'EXPECTED_AUDIT_MISMATCH'):project_export(p,start=START,end=END,trusted_anchor=a)
 def test_wrong_expected_records(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';rows,m=export([fixture()],p);a=make_anchor(rows,m,p);a=replace(a,records=((rows[0]['event_id'],'a'*64,rows[0]['acquisition_redacted_event_digest'],rows[0]['acquisition_source_payload_digest']),))
   with self.assertRaisesRegex(ValueError,'EXPECTED_RECORD_POPULATION_MISMATCH'):project_export(p,start=START,end=END,trusted_anchor=a)
 def test_missing_independent_audit(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p);(p/'export_audit.json').unlink()
   with self.assertRaises(OSError):project_export(p,start=START,end=END)
 def test_modified_audit_evidence(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p);a=json.loads((p/'export_audit.json').read_text());a['events_file_sha256']='b'*64;(p/'export_audit.json').write_text(json.dumps(a))
   with self.assertRaisesRegex(ValueError,'EXPORT_AUDIT_INTEGRITY'):project_export(p,start=START,end=END)

class CoverageV11(unittest.TestCase):
 def report(self,rows,ctx=None):return audit(rows,ctx or context(*rows),start=START,end=END,source_count=len(rows),snapshot_complete=True)
 def test_invalid_slot_not_complete(self):
  rows=[fixture(offset=i) for i in range(3)];rows[0]['payload']['close']='999';r=self.report(rows);self.assertNotEqual(r['calendar_window_status'],'COMPLETE_EXPECTED_SLOTS');self.assertEqual(len(r['invalid_expected_slots']),1);self.assertEqual(r['missing_expected_intervals'],[]);self.assertEqual(len(r['valid_expected_slots']),2)
 def test_complete_positive(self):self.assertTrue(self.report([fixture(offset=i) for i in range(3)])['dataset_complete_for_research'])
 def test_conflict_demotes_aggregate(self):
  e,b=fixture(),fixture(event_id='b'*64);b['payload']['close']='1.26';r=self.report([e,b]);self.assertNotIn('ORIGINAL_TIMESTAMP_VERIFIED',r['recovery_counts']);self.assertEqual(len(r['conflicting_expected_slots']),1);self.assertEqual(r['eligible_record_count'],0);self.assertEqual(len(r['normalized_rows']),2)
 def test_indicator_conflicts_preserved(self):
  e,b=fixture(),fixture(event_id='b'*64);b['payload']['indicators']['rsi']='65';r=self.report([e,b]);self.assertEqual(r['normalized_rows'][1]['features']['original']['indicators']['rsi'],'65');self.assertEqual(r['eligible_record_count'],0)
 def test_unproved_slot_separate(self):
  rows=[fixture(offset=i) for i in range(3)];r=self.report(rows,context(*rows,provenance=False));self.assertEqual(len(r['unproved_expected_slots']),3);self.assertEqual(r['invalid_expected_slots'],[])
 def test_unknown_calendar_stays_unknown(self):
  r=self.report([fixture()],context(fixture(),cal=False));self.assertIsNone(r['valid_expected_slots']);self.assertIsNone(r['research_missing_or_ineligible_slots'])
 def test_instrument_bind(self):
  e=fixture();c=replace(context(e),expected_instrument='EURUSD');self.assertFalse(normalize(e,c)['research_eligible'])
 def test_timeframe_bind(self):
  e=fixture();c=replace(context(e),expected_timeframe='5m');self.assertFalse(normalize(e,c)['research_eligible'])
 def test_integer_version_not_granted(self):
  e=fixture();e['payload']['schema_version']=1;c=replace(context(e),known_versions=frozenset({'1'}));self.assertFalse(normalize(e,c)['research_eligible'])
 def test_alias_features_preserved(self):
  e=fixture();e['payload']['EMA20']='1.234';e['payload']['indicators']['RSI_14']='53';r=normalize(e,context(e));self.assertEqual(r['features']['original']['EMA20'],'1.234');self.assertEqual(r['features']['field_provenance']['rsi_14']['group'],'RSI')
 def test_version_registry_invalid(self):
  with self.assertRaises(ValueError):replace(context(),known_versions=frozenset({1}))

class RedactionV11(unittest.TestCase):
 def test_nested_mixed_case_keys(self):
  for key in ('passphrase','api_key','key','auth','API-Key','PaSsWoRd','access_token','client_secret','connectionString'):
   data={'nested':[{'safe_price':'1.234',key:'SYNTHETIC_SECRET'}]};clean,paths=redact(data);self.assertNotIn('SYNTHETIC_SECRET',json.dumps(clean));self.assertEqual(clean['nested'][0]['safe_price'],'1.234');self.assertTrue(paths)
 def test_password_dsn(self):
  for value in ('host=localhost user=reader password=SYNTHETIC','postgresql://reader:SYNTHETIC@localhost/db','host=x passphrase="SYNTHETIC"'):
   self.assertEqual(redact(value)[0],'[REDACTED]')
 def test_jwt(self):self.assertEqual(redact('eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJTWU5USEVUSUMifQ.c3ludGhldGlj')[0],'[REDACTED]')
 def test_bearer(self):self.assertEqual(redact('Bearer SYNTHETIC_TOKEN')[0],'[REDACTED]')
 def test_positive_market_values(self):
  d={'open':'1.2','volume':Decimal('123.456'),'indicators':{'RSI14':53,'MACD':{'line':'0.003'}},'ticker':'GBPUSD'};self.assertEqual(redact(d)[0],d)
 def test_nonsecret_monkey_key(self):self.assertEqual(redact({'monkey':3,'authenticity_state':'UNPROVED'})[0],{'monkey':3,'authenticity_state':'UNPROVED'})
 def test_paths_do_not_echo_secret_key(self):
  _,paths=redact({'passwordSYNTHETIC':1});self.assertNotIn('SYNTHETIC',str(paths))

class PrivilegeV11(unittest.TestCase):
 def conn(self,row):
  class Cursor:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def execute(self,sql):self.sql=sql
   def fetchone(self):return row
  class Connection:
   rolled=0
   def set_session(self,**kw):self.options=kw
   def cursor(self):return Cursor()
   def rollback(self):self.rolled+=1
  return Connection()
 def test_select_only_positive_mock_not_pg(self):
  c=self.conn(('dedicated_reader',False,True,False,False,False,False));self.assertEqual(verify_reader_identity(c,'dedicated_reader')['identity'],'dedicated_reader');self.assertEqual(c.rolled,1)
 def test_writer_role_rejected(self):
  for index in (1,3,4,5,6):
   row=['dedicated_reader',False,True,False,False,False,False];row[index]=True;c=self.conn(tuple(row))
   with self.assertRaises(ValueError):verify_reader_identity(c,'dedicated_reader')
   self.assertEqual(c.rolled,1)
 def test_wrong_login_rejected(self):
  with self.assertRaises(ValueError):verify_reader_identity(self.conn(('writer',False,True,False,False,False,False)),'dedicated_reader')
 def test_no_select_rejected(self):
  with self.assertRaises(ValueError):verify_reader_identity(self.conn(('dedicated_reader',False,False,False,False,False,False)),'dedicated_reader')
 def test_unsafe_destination(self):
  with self.assertRaises(ValueError):write_export('relative',[],{})
 def test_symlink_ancestor(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'real').mkdir();(root/'link').symlink_to(root/'real',target_is_directory=True)
   with self.assertRaises(ValueError):write_export(root/'link'/'output',[],{})

class ProjectionScopeV11(unittest.TestCase):
 def test_explicit_context_instrument_not_overridden(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p)
   with self.assertRaisesRegex(ValueError,'CONTEXT_STREAM_MISMATCH'):project_export(p,start=START,end=END,context=replace(context(fixture(),cal=False),expected_instrument='EURUSD'))
 def test_explicit_context_timeframe_not_overridden(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'export';export([fixture()],p)
   with self.assertRaisesRegex(ValueError,'CONTEXT_STREAM_MISMATCH'):project_export(p,start=START,end=END,context=replace(context(fixture(),cal=False),expected_timeframe='5m'))
