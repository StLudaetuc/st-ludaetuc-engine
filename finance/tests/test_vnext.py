import unittest
import copy
import json
from pathlib import Path
from datetime import timedelta
from dataclasses import replace
from decimal import Decimal
from test_core import START,END,coverage,fixture_scope,scoped,expense
from test_corrections import bridged,csv_text
import test_corrections as tc
from finance.policy import AuthorityJournal,instant,prorate,elapsed_seconds
from finance.schedule import Schedule,SyncRun,VERSION,next_sync_due,sync_status
from finance.reconciliation import reconcile_run,reconcile_scheduled
from finance.serializer import METRIC_REGISTRY,serialize
from finance.core import import_expenses,ExpenseLedger
from finance.oanda import NON_CASH_TYPES,normalise

LIVE=fixture_scope('live')
def calendar(verified=True,grace=60):
    return Schedule(VERSION,'Europe/London',(1,2,3,4,5),0,30,grace,'evidence:test-config' if verified else None,verified)

class VNextTests(unittest.TestCase):
    def test_N7_DST_end(self):
        cases={'2026-10-23T22:00:00Z':'2026-10-23T23:30:00Z','2026-10-24T00:00:00Z':'2026-10-27T00:30:00Z','2026-10-26T23:50:00Z':'2026-10-27T00:30:00Z'}
        for event,due in cases.items():self.assertEqual(next_sync_due(event,calendar()),instant(due))
    def test_N7_DST_start(self):
        self.assertEqual(next_sync_due('2027-03-27T01:00:00Z',calendar()),instant('2027-03-29T23:30:00Z'))
    def test_N7_exact_run_boundary_next_batch(self):
        self.assertEqual(next_sync_due('2026-10-13T23:30:00Z',calendar()),instant('2026-10-14T23:30:00Z'))
    def test_N7_unverified_configuration_not_assumed(self):
        self.assertFalse(sync_status(START,END,calendar(False),[])['verified'])
    def test_N7_missing_run_after_grace(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z','2026-10-13T23:31:00Z',calendar(),[])['reason'],'missed_or_incomplete_sync_run')
    def test_N7_before_grace_pending(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z','2026-10-13T23:30:59Z',calendar(),[])['state'],'PENDING_EOD_SYNC')
    def sync(self,**kw):
        d=dict(run_id='r',scheduled_at='2026-10-13T23:30:00Z',started_at='2026-10-13T23:30:01Z',completed_at='2026-10-13T23:35:00Z',state='COMPLETED',watermark_start=START,watermark_end='2026-10-13T23:30:00Z',complete=True,evidence_count=1,reviewed_empty=False,manifest_ref='manifest:test');d.update(kw);return SyncRun(**d)
    def test_N7_late_catchup_resolves(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z','2026-10-13T23:36:00Z',calendar(),[self.sync()])['state'],'MATCHED')
    def test_N7_duplicate_runs_idempotent(self):
        r=self.sync();self.assertEqual(sync_status('2026-10-13T20:00:00Z',END,calendar(),[r,r])['state'],'MATCHED')
    def test_N7_conflicting_run_exception(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z',END,calendar(),[self.sync(),self.sync(evidence_count=2)])['reason'],'conflicting_sync_run')
    def test_N7_partial_results_not_complete(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z',END,calendar(),[self.sync(complete=False,state='PARTIAL')])['state'],'INVESTIGATION_REQUIRED')
    def test_N7_future_completion_not_evidence(self):
        self.assertEqual(sync_status('2026-10-13T20:00:00Z','2026-10-13T23:31:00Z',calendar(),[self.sync()])['state'],'INVESTIGATION_REQUIRED')
    def test_N7_zero_requires_reviewed_empty(self):
        with self.assertRaises(ValueError):self.sync(evidence_count=0)
    def test_N7_grace_explicit(self):
        with self.assertRaises(ValueError):calendar(grace=-1)
    def test_M2_opened_no_broker_predeadline_pending(self):
        b,o,t,p=bridged();r=reconcile_run([],o,[],p,as_of=START,outcome_due_at=END,account='practice1',scope=fixture_scope('practice'))
        self.assertEqual(r.results[0].state,'PENDING_EOD_SYNC')
    def test_M2_outcome_no_closure_predeadline_pending(self):
        b,o,t,p=bridged();b[0]['action']='OPEN';r=reconcile_run(b,o,t,p,as_of=START,outcome_due_at=END,account='practice1',scope=fixture_scope('practice'))
        self.assertEqual(r.results[0].state,'PENDING_EOD_SYNC')
    def test_M2_after_deadline_strict(self):
        b,o,t,p=bridged();r=reconcile_run([],o,[],p,as_of=END,outcome_due_at=END,account='practice1',scope=fixture_scope('practice'))
        self.assertEqual(r.results[0].state,'INTERNAL_ONLY')
    def test_N3_coverage_periods_coexist(self):
        j=AuthorityJournal();october=coverage(1);nov=replace(october,period_start=instant(END),period_end=instant(END)+timedelta(days=30))
        j.append('coverage','a',october,START);j.append('coverage','a',nov,END)
        self.assertEqual(j.coverage_for('a',START,END)[3],october)
    def test_N3_withdrawal_only_own_period(self):
        j=AuthorityJournal();c=coverage(1);j.append('coverage','a',c,START);j.append('coverage','a',replace(c,state='WITHDRAWN'),END)
        self.assertEqual(j.coverage_for('a',START,END)[3].state,'WITHDRAWN')
    def test_H5_registry_covers_all_schema_metrics(self):
        def walk(x):
            if isinstance(x,dict):
                for key,v in x.get('properties',{}).items():
                    if key in ('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs'):
                        self.assertLessEqual(set(v['properties']),set(METRIC_REGISTRY))
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        for f in ('reader_products','weekly_report'):walk(json.loads(Path('finance/contracts/'+f+'.schema.json').read_text()))
    def test_H5_verified_null_rejected(self):
        helper=tc.SerializerTests();p,j,e=helper.factory();p['metrics']['net_trading']['value']=None;e[('metrics','net_trading')]=copy.deepcopy(p['metrics']['net_trading'])
        with self.assertRaises(ValueError):helper.validate(p,j,e)
    def test_H5_weekly_without_metric_windows_rejected(self):
        p,j,e=tc.SerializerTests().factory('weekly')
        with self.assertRaises(ValueError):serialize(p,'weekly',journal=j,coverage_key='a',scope=LIVE,observed_count=1,expected_metrics=e,reconciliation_key='a')
    def test_H5_unratified_money_unavailable(self):
        p,j,e=tc.SerializerTests().factory()
        with self.assertRaises(ValueError):serialize(p,'reader',journal=j,coverage_key='a',scope=LIVE,observed_count=1,expected_metrics=e,reconciliation_key='a')
    def test_L7_csv_formula_identity_rejected_atomic(self):
        l=ExpenseLedger()
        with self.assertRaises(ValueError):import_expenses(csv_text([expense(expense_id='-A1',verification='unverified',verified_by='')]),l)
        self.assertEqual(len(l.history),0)
    def test_N6_all_reviewed_non_cash_types(self):
        for kind in NON_CASH_TYPES:
            with self.subTest(kind=kind):
                rows,meta=normalise({'id':'1','time':START,'type':kind,'accountID':'x'},'a','x',LIVE,'GBP','1');self.assertEqual(rows,());self.assertTrue(meta['non_cash'])
    def test_N6_unknown_type_fails_closed(self):
        with self.assertRaises(ValueError):normalise({'id':'1','time':START,'type':'ALIEN','accountID':'x','accountBalance':'1'},'a','x',LIVE,'GBP','1')
    def test_N6_non_cash_with_cash_rejected(self):
        with self.assertRaises(ValueError):normalise({'id':'1','time':START,'type':'MARKET_ORDER','accountID':'x','pl':'3'},'a','x',LIVE,'GBP','1')
    def test_L1_constructed_recognition_vector(self):
        amount=Decimal('1234.559980108419');full=Decimal(365*86400+1);cut=Decimal(31*86400)
        self.assertEqual(prorate(amount,cut,full),Decimal('104.853036081600'))
