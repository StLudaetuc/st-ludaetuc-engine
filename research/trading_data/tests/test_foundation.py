import copy
import unittest
from datetime import datetime,timedelta,timezone
from decimal import Decimal
from research.trading_data.evidence import Journal,instrument,timestamp,number
from research.trading_data.candles import audit
from research.trading_data.lifecycle import reconstruct
from research.trading_data.replay import replay,feature_at
from research.trading_data.explorer import query,Principal

START='2026-10-13T00:00:00Z';END='2026-10-14T00:00:00Z'
def event(kind='prediction',eid='p',**payload):
    if kind=='broker_fill':payload.setdefault('order_id','order-1')
    return dict(event_id=eid,source_event_id=eid,account_alias='practice-a',broker_environment='OANDA_PRACTICE',instrument='GBPUSD',event_time=START,received_at=START,kind=kind,payload=payload or {'signal_id':'s','approval_status':'approved'})
def chain():
    return [event(),event('trade_opened','o',signal_id='s',fill_status='FILLED',broker_trade_ids=['t']),event('broker_fill','b',broker_trade_id='t',transaction_id='1',action='OPEN',units='10'),event('broker_fill','c',broker_trade_id='t',transaction_id='2',action='CLOSE',units='10'),event('trade_outcome','x',broker_trade_id='t',transaction_id='2')]
def candle(**kw):
    r=dict(instrument='GBPUSD',timeframe='1m',event_time=START,open='100',high='102',low='99',close='101',volume='1');r.update(kw);return r
def plan(**kw):
    p=dict(direction='long',entry='100',stop_loss='99',take_profit='102',order_type='MARKET_NEXT_BAR',plan_available_at=START);p.update(kw);return p
def bar(**kw):
    c=dict(bar_open='2026-10-13T00:01:00Z',bar_close='2026-10-13T00:02:00Z',open='100',high='103',low='99.5',close='102');c.update(kw);return c
def replay_test(p=None,bars=None,**kw):
    options=dict(decision_at=START,horizon_end=END,spread='0',slippage='0',coverage_verified=True,prerequisites_verified=True);options.update(kw);return replay(p or plan(),bars or [bar()],**options)

class EvidenceTests(unittest.TestCase):
    def test_exact_replay_preserves_fact_receipts(self):
        j=Journal();e=event();self.assertEqual(j.append(e),'NEW_FACT');e['received_at']=END;self.assertEqual(j.append(e),'EXACT_REPLAY');self.assertEqual(len(j.facts),1);self.assertEqual(len(j.deliveries),2)
    def test_conflict_retained_without_overwrite(self):
        j=Journal();e=event();j.append(e);e['payload']['signal_id']='changed';self.assertEqual(j.append(e),'CONFLICT');self.assertEqual(j.facts[0]['event']['payload']['signal_id'],'s')
    def test_external_mutation_no_history_change(self):
        j=Journal();j.append(event());f=j.facts;f[0]['event']['payload']['signal_id']='bad';self.assertEqual(j.facts[0]['event']['payload']['signal_id'],'s')
    def test_account_scope_collision_is_separate(self):
        j=Journal();e=event();j.append(e);e['account_alias']='other';self.assertEqual(j.append(e),'NEW_FACT')
    def test_correction_requires_existing_scoped_lineage(self):
        j=Journal()
        with self.assertRaises(ValueError):j.append(event('correction','r',supersedes_event_id='missing',review_evidence='evidence:test'))
    def test_valid_correction_retains_prior(self):
        j=Journal();j.append(event());j.append(event('correction','r',supersedes_event_id='p',review_evidence='evidence:test'));self.assertEqual(len(j.facts),2)
    def test_bad_source_timestamp_not_receipt_fallback(self):
        for value in ('','{{time}}','2026-10-13','garbage',None):
            with self.subTest(value=value),self.assertRaises((ValueError,TypeError)):timestamp(value)
    def test_instrument_aliases_and_placeholder(self):
        for s in ('GBP/USD','GBP_USD','GBPUSD'):self.assertEqual(instrument(s),'GBPUSD')
        with self.assertRaises(ValueError):instrument('{{ticker}}')
    def test_nonfinite_number_rejected(self):
        for x in ('NaN','Infinity','-Infinity',True,1.1):
            with self.subTest(x=x),self.assertRaises(ValueError):number(x)

class CandleTests(unittest.TestCase):
    def test_no_calendar_no_completeness(self):self.assertFalse(audit([candle()],START,END)['streams'][0]['complete'])
    def test_explicit_slots_complete(self):self.assertTrue(audit([candle()],START,END,expected_slots={('GBPUSD','1m'):[START]},calendar_evidence='evidence:calendar')['streams'][0]['complete'])
    def test_missing_expected_bar(self):self.assertEqual(audit([candle()],START,END,expected_slots={('GBPUSD','1m'):[START,'2026-10-13T00:01:00Z']},calendar_evidence='evidence:calendar')['streams'][0]['missing_expected'],1)
    def test_duplicate_exact_vs_conflict(self):
        self.assertEqual(audit([candle(),candle(),candle(close='100')],START,END)['streams'][0]['conflicting_timestamps'],1)
    def test_ohlc_invalid_quarantined(self):self.assertEqual(len(audit([candle(low='103')],START,END)['malformed']),1)
    def test_offgrid_receipt_timestamp_flagged(self):self.assertEqual(audit([candle(event_time='2026-10-13T00:00:03.100Z')],START,END)['streams'][0]['off_grid'],1)
    def test_out_of_order_detected(self):self.assertEqual(audit([candle(event_time='2026-10-13T00:01:00Z'),candle()],START,END)['streams'][0]['out_of_order_arrivals'],1)
    def test_volume_absent_is_not_zero(self):self.assertEqual(audit([candle(volume=None)],START,END)['streams'][0]['volume_missing'],1)

class LifecycleTests(unittest.TestCase):
    def test_full_chain_linked_but_manifest_unknown(self):
        r=reconstruct(chain());self.assertEqual(r['fully_linked_population'],1);self.assertFalse(r['official_all_trades_reconciled'])
    def test_out_of_order_events_deterministic(self):self.assertEqual(reconstruct(chain())['trades'],reconstruct(list(reversed(chain())))['trades'])
    def test_partial_and_multiple_fills(self):
        e=chain()[:3];e+=[event('broker_fill','b2',broker_trade_id='t',transaction_id='3',action='OPEN',units='5'),event('broker_fill','red',broker_trade_id='t',transaction_id='4',action='REDUCE',units='2')]
        r=reconstruct(e)['trades'][0];self.assertEqual(r['state'],'PARTIALLY_CLOSED');self.assertEqual(r['remaining_units'],'13')
    def test_cancelled_not_executed(self):
        e=[event(),event('trade_opened','o',signal_id='s',fill_status='CANCELLED',broker_trade_ids=[])];r=reconstruct(e);self.assertEqual(r['filled_orders'],0);self.assertEqual(r['cancelled_orders'],1);self.assertEqual(r['trades'],[]);self.assertEqual(r['unsubmitted_signals'],[])
    def test_missing_broker_id_explicit(self):self.assertEqual(reconstruct([event('trade_opened','o',signal_id='s',fill_status='FILLED',broker_trade_ids=[])])['malformed'][0]['reason'],'FILLED_WITHOUT_BROKER_TRADE_ID')
    def test_orphan_outcome_retained(self):self.assertIn('OPENING_MISSING',reconstruct(chain()[-1:])['trades'][0]['reasons'])
    def test_missing_prediction(self):self.assertIn('PREDICTION_MISSING',reconstruct(chain()[1:])['trades'][0]['reasons'])
    def test_exact_duplicate_closure(self):self.assertEqual(reconstruct(chain()+[chain()[-1]])['replay_count'],1)
    def test_conflicting_outcome_prevents_clean(self):
        e=chain();bad=copy.deepcopy(e[-1]);bad['payload']['reason']='conflicting';r=reconstruct(e+[bad],broker_manifest_complete=True);self.assertFalse(r['official_all_trades_reconciled']);self.assertEqual(r['malformed'][0]['reason'],'CONFLICTING_EVENT')
    def test_cross_account_ids_do_not_join(self):
        e=chain();e[-1]['account_alias']='other';self.assertEqual(len(reconstruct(e)['trades']),2)
    def test_ambiguous_signals(self):
        e=chain()+[event('trade_opened','o2',signal_id='s2',fill_status='FILLED',broker_trade_ids=['t'])];self.assertIn('AMBIGUOUS_SIGNAL_LINK',reconstruct(e)['trades'][0]['reasons'])
    def test_malformed_does_not_abort(self):
        e=chain()+[event('broker_fill','bad',broker_trade_id='t',transaction_id='3',action='CLOSE',units='NaN')];self.assertEqual(len(reconstruct(e)['malformed']),1)

class SchemaTests(unittest.TestCase):
    def test_daily_schema_and_example(self):
        import json
        from pathlib import Path
        from jsonschema import Draft202012Validator,FormatChecker
        root=Path(__file__).resolve().parents[1]
        schema=json.loads((root/'daily_report.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema,format_checker=FormatChecker()).validate(json.loads((root/'evidence/daily_report_20261013.json').read_text()))
    def test_sql_proposal_grammar_not_runtime(self):
        from pathlib import Path
        from pglast import parse_sql,parse_plpgsql
        text=(Path(__file__).resolve().parents[1]/'retention_candidate.sql').read_text()
        self.assertTrue(parse_sql(text));self.assertTrue(parse_plpgsql(text))

class ReplayTests(unittest.TestCase):
    def test_same_candle_ambiguous_not_real_loss(self):self.assertEqual(replay_test(bars=[bar(low='98')])['state'],'AMBIGUOUS_SAME_CANDLE')
    def test_unknown_costs_no_result(self):self.assertEqual(replay_test(spread=None)['state'],'COST_ASSUMPTIONS_MISSING')
    def test_missing_data_no_result(self):self.assertEqual(replay_test(coverage_verified=False)['state'],'INSUFFICIENT_DATA')
    def test_rejection_does_not_equal_mistake(self):self.assertFalse(replay_test()['is_rejection_error'])
    def test_hypothetical_never_realised(self):self.assertEqual(replay_test()['financial_reality'],'SIMULATED')
    def test_limit_entry_exit_order_ambiguous(self):self.assertEqual(replay_test(p=plan(order_type='LIMIT'))['state'],'AMBIGUOUS_ENTRY_EXIT_SEQUENCE')
    def test_decision_bar_not_usable(self):self.assertEqual(replay_test(bars=[bar(bar_open=START)])['state'],'INSUFFICIENT_DATA')
    def test_lookahead_feature_rejected(self):
        with self.assertRaises(ValueError):feature_at('50',END,START)
    def test_plan_lookahead_rejected(self):self.assertEqual(replay_test(p=plan(plan_available_at=END))['state'],'INVALID_OR_UNRECONSTRUCTABLE_PLAN')
    def test_abnormal_spread(self):self.assertEqual(replay_test(spread='2')['state'],'ABNORMAL_COST_ASSUMPTIONS')
    def test_wrong_plan_geometry(self):self.assertEqual(replay_test(p=plan(stop_loss='103'))['state'],'INVALID_OR_UNRECONSTRUCTABLE_PLAN')
    def test_prerequisites_missing(self):self.assertEqual(replay_test(prerequisites_verified=False)['state'],'EXECUTION_PREREQUISITES_UNPROVED')

class ExplorerTests(unittest.TestCase):
    def principal(self):return Principal('synthetic-ceo',True,frozenset({'practice-a'}),frozenset({'research:read'}))
    def read(self,p=None,**kw):return query(chain(),p or self.principal(),account_alias='practice-a',start=START,end=END,audit_log=[],**kw)
    def test_unauthenticated_denied(self):
        with self.assertRaises(PermissionError):self.read(Principal('x',False,frozenset({'practice-a'}),frozenset({'research:read'})))
    def test_cross_account_denied(self):
        with self.assertRaises(PermissionError):self.read(Principal('x',True,frozenset({'other'}),frozenset({'research:read'})))
    def test_unbounded_export_denied(self):
        for limit in (0,501,True,-1):
            with self.subTest(limit=limit),self.assertRaises(ValueError):self.read(limit=limit)
    def test_sensitive_raw_excluded(self):
        e=event();e['payload'].update(token='secret',account_snapshot={'balance':'1'},reasons=[{'password':'secret'}]);out=query([e],self.principal(),account_alias='practice-a',start=START,end=END,audit_log=[]);self.assertNotIn('secret',str(out))
    def test_audit_and_truncation(self):
        logs=[];out=query(chain(),self.principal(),account_alias='practice-a',start=START,end=END,limit=1,audit_log=logs);self.assertTrue(out['truncated']);self.assertEqual(len(logs),1)

if __name__=='__main__':unittest.main()
