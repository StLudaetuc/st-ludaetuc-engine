import copy
import csv
import io
import json
import unittest
from datetime import timedelta
from decimal import Decimal, getcontext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from test_core import START, END, expense, fixture_scope, coverage, scoped, component
import finance.core as core
from finance.policy import *
from finance.reconciliation import *
from finance.oanda import normalise_batch
from finance.serializer import serialize,TRADING_KEYS,ECONOMIC_KEYS,CASH_ECONOMIC_KEYS,FORECAST_KEYS

LIVE=fixture_scope('live'); PRACTICE=fixture_scope('practice')

def account(alias, value, scope=LIVE, complete=True):
    return {'account':alias,'scope':scope,'net_trading':money(value) if value is not None else None,'coverage_complete':complete}

def expenses(value='12', complete=True):
    return {'cash_spend':money(value) if value is not None else None,'economic_expense':money(value) if value is not None else None,'coverage_complete':complete}

def project(rows, spend='12', complete=True, count=1, basis='closed_trades'):
    return core.project_economics(rows,expenses(spend,complete),coverage(len(rows),complete),START,END,trade_count=count,denominator_basis=basis)

def reconciliation(b=None,o=None,t=None,p=None,now=END):
    return reconcile_run(b or [],o or [],t or [],p or [],as_of=now,outcome_due_at=END,account='practice1',scope=PRACTICE)

def bridged(value='4'):
    return ([scoped(trade_id='1',action='CLOSE',realised_pnl=value)],
            [scoped(trade_id='1',signal_id='s')],
            [scoped(trade_id='1',realised_pnl=value)], [scoped(signal_id='s')])

def csv_text(rows):
    from dataclasses import fields
    names=[f.name for f in fields(core.Expense)]
    stream=io.StringIO();writer=csv.DictWriter(stream,names);writer.writeheader()
    for row in rows:
        d={k:('' if v is None else str(v)) for k,v in row.__dict__.items()};d['recurring']='true' if row.recurring else 'false';writer.writerow(d)
    stream.seek(0);return stream

class CorrectionTests(unittest.TestCase):
    def test_H1_future_source_cannot_beat_withdrawal(self):
        j=AuthorityJournal();j.append('coverage','a',coverage(1),'2100-01-01T00:00:00Z')
        j.append('coverage','a',replace(coverage(1),state='WITHDRAWN'),START)
        self.assertEqual(j.latest('coverage','a')[3].state,'WITHDRAWN')
        self.assertEqual([r[0] for r in j.history],[1,2])
    def test_H1_verification_withdrawal_and_revision(self):
        l=core.ExpenseLedger();l.append(expense(verification='unverified',verified_by=''))
        l.verify('e1',1,'invoice_verified','CEO','evidence:invoice','2100-01-01T00:00:00Z')
        self.assertEqual(l.spend(START,END),10)
        l.verify('e1',1,'withdrawn','CEO','evidence:withdraw',START)
        self.assertEqual(l.spend(START,END),0)
        self.assertEqual(len(l.verifications.history),2)
    def test_H1_verification_not_carried_to_new_revision(self):
        l=core.ExpenseLedger();l.append(expense(verification='unverified',verified_by=''))
        l.verify('e1',1,'invoice_verified','CEO','evidence:invoice',START)
        l.append(expense(revision=2,correction_reason='changed',verification='unverified',verified_by=''))
        self.assertEqual(l.effective[0].verification,'unverified')
    def test_H1_concurrent_server_orders(self):
        j=AuthorityJournal()
        with ThreadPoolExecutor(8) as pool:list(pool.map(lambda i:j.append('coverage','a',i,START),range(100)))
        self.assertEqual([r[0] for r in j.history],list(range(1,101)))
    def test_H2_missing_broker_evidence_is_unavailable(self):
        r=core.broker_totals([],'a',LIVE,START,END,coverage(0,False))
        self.assertIsNone(r['net_trading']);self.assertFalse(r['coverage_complete'])
    def test_legacy_pnl_without_currency_cost_completeness_unknown(self):
        c=replace(coverage(1),currency_confirmed=False,costs_complete=False)
        self.assertIsNone(core.broker_totals([component()],'practice1',PRACTICE,START,END,c)['net_trading'])
    def test_H2_reviewed_empty_is_real_zero(self):
        r=core.broker_totals([],'a',LIVE,START,END,coverage(0))
        self.assertEqual(r['net_trading'],0)
    def test_H2_verified_cannot_be_empty(self):
        with self.assertRaises(ValueError):replace(coverage(0),state='VERIFIED')
    def test_H2_count_mismatch_is_unknown(self):
        self.assertIsNone(core.broker_totals([],'a',LIVE,START,END,coverage(1))['net_trading'])
    def test_H2_missing_expenses_cannot_break_even(self):
        r=project([account('a','100')],None,False)
        self.assertIsNone(r['remaining_cash_break_even_gbp']);self.assertIsNone(r['operationally_profitable'])
    def test_H2_explicit_empty_project_can_zero(self):
        r=core.project_economics([],expenses('0'),coverage(0),START,END,trade_count=0,denominator_basis='closed_trades')
        self.assertEqual(r['cash_net_economic_gbp'],0);self.assertFalse(r['operationally_profitable'])
    def test_H3_two_accounts_subtract_once(self):
        r=project([account('a','95'),account('b','45')])
        self.assertEqual(r['net_economic_gbp'],128);self.assertEqual(r['cash_spend_gbp'],12)
    def test_H3_duplicate_account_rejected(self):
        with self.assertRaises(ValueError):project([account('a','95'),account('a','95')])
    def test_H3_incomplete_one_account_unavailable(self):
        self.assertIsNone(project([account('a','95'),account('b',None,complete=False)])['net_economic_gbp'])
    def test_H3_practice_excluded_not_summed(self):
        self.assertEqual(project([account('a','5'),account('practice','9999',PRACTICE)])['net_economic_gbp'],-7)
    def test_H4_no_dimension_defaults(self):
        with self.assertRaises(TypeError):Scope('OANDA_LIVE')
        with self.assertRaises(TypeError):core.BrokerComponent(account='a',transaction_id='1',component_key='pl',at=START,kind='realised_pnl',amount='1',currency='GBP',fx_to_gbp='1')
    def test_H4_legacy_production_is_not_live(self):
        with self.assertRaises(ValueError):Scope('production','PRODUCTION','ACTUAL','REALISED',None)
    def test_H4_simulated_cannot_be_realised(self):
        with self.assertRaises(ValueError):Scope('OANDA_LIVE','PRODUCTION','SIMULATED','REALISED','COUNTERFACTUAL')
    def test_M1_negative_tie_away_zero(self):
        self.assertEqual(quantise(Decimal('-0.0000000000005')),Decimal('-0.000000000001'))
    def test_M1_positive_tie_away_zero(self):
        self.assertEqual(quantise(Decimal('0.0000000000005')),Decimal('0.000000000001'))
    def test_M1_conversion_round_per_cash_component(self):
        self.assertEqual(convert('0.000000000001','0.5'),Decimal('0.000000000001'))
    def test_M1_context_independence(self):
        old=getcontext().prec
        try:
            getcontext().prec=4
            self.assertEqual(convert('123456789.123456789012','0.765432109876'),Decimal('94497790.577283938656'))
        finally:getcontext().prec=old
    def test_M2_partial_slices_match_lifetime_outcome(self):
        b,o,t,p=bridged('10');b=[scoped(trade_id='1',action='REDUCE',transaction_id='reduce',realised_pnl='4'),scoped(trade_id='1',action='CLOSE',transaction_id='close',realised_pnl='6')]
        r=reconciliation(b,o,t,p);self.assertTrue(r.clean);self.assertEqual(r.closed_trades,1);self.assertEqual(r.partial_close_slices,1);self.assertEqual(r.transaction_count,2)
    def test_M2_eod_boundary(self):
        b,o,t,p=bridged()
        before=reconciliation(b,o,[],p,START);after=reconciliation(b,o,[],p,END)
        self.assertEqual(before.results[0].state,'PENDING_EOD_SYNC');self.assertEqual(after.results[0].state,'BROKER_ONLY')
    def test_M2_late_match_not_false_failure(self):
        self.assertEqual(reconciliation(*bridged()).results[0].state,'MATCHED')
    def test_M2_missing_broker_is_internal_only(self):
        b,o,t,p=bridged();self.assertEqual(reconciliation([],o,t,p).results[0].state,'INTERNAL_ONLY')
    def test_M2_value_mismatch(self):
        b,o,t,p=bridged();t[0]['realised_pnl']='5';self.assertEqual(reconciliation(b,o,t,p).results[0].state,'VALUE_MISMATCH')
    def test_M2_malformed_does_not_abort_good_trade(self):
        b,o,t,p=bridged();r=reconciliation(b+[{'trade_id':'bad'},None],o,t,p)
        self.assertEqual(next(x.state for x in r.results if x.trade_id=='1'),'MATCHED');self.assertFalse(r.clean);self.assertEqual(len(r.malformed),2)
    def test_M2_unhashable_identity_does_not_abort(self):
        b,o,t,p=bridged();r=reconciliation(b+[{'trade_id':{'malformed':'id'}}],o,t,p)
        self.assertEqual(r.results[0].state,'MATCHED');self.assertEqual(len(r.malformed),1)
    def test_M2_non_decimal_does_not_abort(self):
        b,o,t,p=bridged();b[0]['realised_pnl']='not-money';r=reconciliation(b,o,t,p);self.assertEqual(r.results[0].state,'INVESTIGATION_REQUIRED')
    def test_M2_exact_replay_ignored(self):
        b,o,t,p=bridged();r=reconciliation(b+b,o,t,p);self.assertTrue(r.clean);self.assertEqual(r.duplicate_replays,1)
    def test_M2_receipt_time_change_is_replay(self):
        b,o,t,p=bridged();late=dict(b[0],received_at=END);r=reconciliation(b+[late],o,t,p);self.assertTrue(r.clean)
    def test_M2_conflicting_event_not_last_write_wins(self):
        b,o,t,p=bridged();r=reconciliation(b+[dict(b[0],realised_pnl='99')],o,t,p);self.assertEqual(r.results[0].state,'INVESTIGATION_REQUIRED')
    def test_M3_nested_pnl_mismatch_quarantines(self):
        tx=dict(accountID='raw',id='1',time=START,type='ORDER_FILL',accountBalance='100',pl='10',financing='0',commission='0',guaranteedExecutionFee='0',tradesClosed=[{'realizedPL':'9'}])
        good=dict(tx,id='2',tradesClosed=[{'realizedPL':'10'}])
        accepted,errors=normalise_batch([tx,good],'practice1','raw',PRACTICE,'GBP','1')
        self.assertEqual(len(accepted),1);self.assertEqual(errors[0]['reason'],'nested_attribution_mismatch')
    def test_M3_nested_missing_not_zero(self):
        tx=dict(accountID='raw',id='1',time=START,type='DAILY_FINANCING',accountBalance='100',financing='-2',positionFinancings=[{}])
        accepted,errors=normalise_batch([tx],'practice1','raw',PRACTICE,'GBP','1');self.assertFalse(accepted);self.assertEqual(errors[0]['reason'],'nested_attribution_missing')
    def test_M4_self_verification_rejected(self):
        with self.assertRaises(ValueError):core.import_expenses(csv_text([expense()]),core.ExpenseLedger())
    def test_M4_formula_neutralised(self):
        l=core.ExpenseLedger();row=expense(verification='unverified',verified_by='',vendor='=HYPERLINK("x")',description='@SUM(1)')
        core.import_expenses(csv_text([row]),l);self.assertTrue(l.effective[0].vendor.startswith("'="));self.assertTrue(l.effective[0].description.startswith("'@"))
    def test_M4_retry_idempotent_with_formula(self):
        l=core.ExpenseLedger();row=expense(verification='unverified',verified_by='',vendor='=x')
        self.assertEqual(core.import_expenses(csv_text([row]),l),1);self.assertEqual(core.import_expenses(csv_text([row]),l),0)
    def test_M4_unsafe_reference_rejected(self):
        for ref in ('https://x.example/?token=secret','file:/etc/passwd','../secret','=HYPERLINK("x")'):
            with self.subTest(ref=ref),self.assertRaises(ValueError):core.import_expenses(csv_text([expense(verification='unverified',verified_by='',evidence_ref=ref)]),core.ExpenseLedger())
    def test_M4_registry(self):
        with self.assertRaises(ValueError):expense(currency='ZZZ')
    def test_M4_bounded_size(self):
        with self.assertRaises(ValueError):core.import_expenses(io.StringIO('x'*1048577),core.ExpenseLedger())
    def test_M4_late_invalid_row_atomic(self):
        l=core.ExpenseLedger();a=expense(verification='unverified',verified_by='');b=expense(expense_id='e2',source_key='source:2',verification='unverified',verified_by='',evidence_ref='https://bad')
        with self.assertRaises(ValueError):core.import_expenses(csv_text([a,b]),l)
        self.assertEqual(l.history,())
    def test_M4_concurrent_duplicate_import(self):
        l=core.ExpenseLedger();row=expense(verification='unverified',verified_by='')
        with ThreadPoolExecutor(2) as pool:r=list(pool.map(lambda _:core.import_expenses(csv_text([row]),l),range(2)))
        self.assertEqual(sorted(r),[0,1]);self.assertEqual(len(l.history),1)
    def test_M5_source_time_correction_not_order(self):
        l=core.ExpenseLedger();l.append(expense(incurred_at='2099-01-01T00:00:00Z'))
        l.append(expense(revision=2,incurred_at=START,correction_reason='actual bill date'))
        self.assertEqual(l.effective[0].revision,2);self.assertEqual(l.spend(START,END),10)
    def test_M6_practice_non_cash(self):
        total=core.broker_totals([component()],'practice1',PRACTICE,START,END,coverage(1))
        r=core.practice_unit_economics(total,trade_count=1,denominator_basis='closed_trades')
        self.assertEqual(r['basis'],'HYPOTHETICAL_NON_CASH');self.assertIsNone(r['actual_net_profit'])
    def test_M6_attempted_orders_not_profit_denominator(self):
        r=project([account('a','100')],basis='attempted_orders',count=155)
        self.assertIsNone(r['net_profit_per_trade_gbp'])
    def test_operational_112_filled_43_cancelled(self):
        o=[scoped(trade_id=str(i),event_id=str(i),fill_status='FILLED' if i<112 else 'CANCELLED') for i in range(155)]
        r=reconciliation([],o,[],[]);self.assertEqual((r.attempted_orders,r.filled_orders,r.cancelled_orders,r.closed_trades),(155,112,43,0))
    def test_M7_missing_run_rate(self):
        r=core.run_rate([],START,coverage(0,False));self.assertIsNone(r['month']);self.assertEqual(r['state'],'UNKNOWN')
    def test_M7_reviewed_zero(self):
        self.assertEqual(core.run_rate([],START,coverage(0))['month'],0)
    def test_M7_estimates_and_unknown_differ(self):
        rows=[core.Subscription('e','5',START,END,'estimate'),core.Subscription('u','7',START,END,'unverified')]
        r=core.run_rate(rows,START,coverage(2));self.assertIsNone(r['month']);self.assertEqual((r['excluded_estimates'],r['unknown_records']),(1,1))
    def test_M8_prepaid_service_recognition(self):
        l=core.ExpenseLedger();l.append(expense(original_amount='365',paid_at=START,billing_start=START,billing_end='2027-10-01T00:00:00Z',recognition_policy='straight_line_seconds_v1'))
        r=core.expense_totals(l,START,END,coverage(1));self.assertEqual(r['cash_spend'],365);self.assertEqual(r['economic_expense'],31)
    def test_M8_policy_missing_unavailable(self):
        l=core.ExpenseLedger();l.append(expense(paid_at=START))
        r=core.expense_totals(l,START,END,coverage(1));self.assertEqual(r['cash_spend'],10);self.assertIsNone(r['economic_expense']);self.assertEqual(r['policy_missing'],('e1',))
    def test_M8_cash_date_not_incurred_date(self):
        l=core.ExpenseLedger();l.append(expense(paid_at=END,recognition_policy='point_in_time_v1'))
        r=core.expense_totals(l,START,END,coverage(1));self.assertEqual((r['cash_spend'],r['economic_expense']),(0,10))
    def test_M8_refund_service_recognition(self):
        l=core.ExpenseLedger();l.append(expense(original_amount='-31',paid_at=START,recognition_policy='straight_line_seconds_v1'))
        self.assertEqual(core.expense_totals(l,START,END,coverage(1))['economic_expense'],-31)
    def test_M8_cumulative_recognition_is_additive(self):
        l=core.ExpenseLedger();l.append(expense(original_amount='1',paid_at=START,billing_start=START,billing_end='2026-10-04T00:00:00Z',recognition_policy='straight_line_seconds_v1'))
        values=[]
        for i in range(3):
            left=instant(START)+timedelta(days=i);right=left+timedelta(days=1)
            values.append(core.expense_totals(l,left,right,coverage(1))['economic_expense'])
        self.assertEqual(add(values),1);self.assertEqual(values[-1],Decimal('0.333333333333'))
    def test_M9_latest_run_uses_server_identity(self):
        j=ReconciliationJournal('practice1',PRACTICE)
        j.record('future',reconciliation(*bridged()),'2100-01-01T00:00:00Z')
        b,o,t,p=bridged();j.record('later',reconciliation(b,o,[],p),START)
        self.assertEqual(j.latest_run()[3][0],'later')
    def test_M9_cross_account_resolution_rejected(self):
        j=ReconciliationJournal('other',LIVE)
        with self.assertRaises(ValueError):j.record('r',reconciliation(*bridged()),START)
    def test_M9_resolution_and_withdrawal(self):
        b,o,t,p=bridged();old=reconciliation(b,o,[],p);new=reconciliation(b,o,t,p)
        j=ReconciliationJournal('practice1',PRACTICE);j.record('old',old,END);j.record('new',new,END)
        j.resolve('old','1','new','CEO','evidence:match','2100-01-01T00:00:00Z')
        j.withdraw('old','1','CEO','evidence:withdraw',START)
        self.assertEqual(j.latest_resolution('old','1')[3][0],'WITHDRAWN');self.assertEqual(old.results[0].state,'BROKER_ONLY')
    def test_M9_stale_matched_evidence_cannot_resolve(self):
        b,o,t,p=bridged();j=ReconciliationJournal('practice1',PRACTICE)
        j.record('stale',reconciliation(b,o,t,p),START);j.record('new-failure',reconciliation(b,o,[],p),END)
        with self.assertRaises(ValueError):j.resolve('new-failure','1','stale','CEO','evidence:stale',END)
    def test_cancelled_conflicting_trade_never_denominator(self):
        b,o,t,p=bridged();o[0]['fill_status']='CANCELLED';r=reconciliation(b,o,t,p)
        self.assertEqual(r.closed_trades,0);self.assertEqual(r.results[0].state,'INVESTIGATION_REQUIRED')
    def test_M9_no_resolution_without_matched_evidence(self):
        b,o,t,p=bridged();j=ReconciliationJournal('practice1',PRACTICE);j.record('old',reconciliation(b,o,[],p),END);j.record('bad',reconciliation(b,o,[],p),END)
        with self.assertRaises(ValueError):j.resolve('old','1','bad','CEO','evidence:bad',END)

class SerializerTests(unittest.TestCase):
    def factory(self,kind='reader',product='finance-summary'):
        d=json.loads(Path('finance/contracts/'+('reader_products' if kind=='reader' else 'weekly_report')+'.schema.json').read_text())
        schema=next(v for v in d['oneOf'] if v['properties']['product']['const']==product) if kind=='reader' else d
        p={}
        for key,rule in schema['properties'].items():
            if 'const' in rule:p[key]=rule['const']
            elif key in ('as_of','period_start','period_end'):p[key]=END if key!='period_start' else START
            elif key in ('trade_count','evidence_count'):p[key]=1
            elif key in ('exception_count','pending_eod_count','unresolved_exception_count','recognition_policy_missing_count'):p[key]=0
            elif key in ('coverage_complete','currency_confirmed','costs_complete'):p[key]=True
            elif key=='first_evidence_id':p[key]='first'
            elif key=='last_evidence_id':p[key]='last'
            elif key=='simulation_type':p[key]=None
            elif key in ('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs'):
                p[key]={}
                for name,mrule in rule['properties'].items():
                    if mrule['$ref'].endswith('/drawdown'):
                        p[key][name]={'value':None,'unit':'fraction_of_capital','state':'unavailable','classification':'UNKNOWN','reason':'no approved drawdown input','evidence_id':None}
                    else:p[key][name]={'value':'1.000000000000','currency':'GBP','state':'verified','classification':'FACT','reason':None,'evidence_id':'evidence:fixture'}
            else:p[key]='fixture'
        p.update(broker_environment='OANDA_LIVE',system_environment='PRODUCTION',financial_reality='ACTUAL',financial_state='REALISED',scope_level='PROJECT',classification='MIXED' if kind=='weekly' else 'FACT',basis='ACTUAL_CASH',coverage_state='VERIFIED',reconciliation_state='MATCHED',denominator_basis='closed_trades')
        for group in ('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs'):
            if 'estimated_spend' in p.get(group,{}):
                p[group]['estimated_spend'].update(state='estimate',classification='ESTIMATED')
                p['classification']='MIXED'
        p['source_manifest_id']='manifest:fixture'
        j=AuthorityJournal();j.append('coverage','a',coverage(1),START);j.append('project_inception','a',instant(START),START)
        run=Run((Result('t','MATCHED',(),1),),(),0,1,1,0,1,0,1,'fixture',LIVE,True)
        j.append('reconciliation','a',run,START)
        expected={(g,k):copy.deepcopy(v) for g in ('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs') for k,v in p.get(g,{}).items()}
        return p,j,expected
    def validate(self,p,j,e,kind='reader',scope=LIVE):
        contexts={(g,k):dict(start=START,end=END,coverage_key='a',observed_count=1,reconciliation_key='a',window_kind='lifetime' if g=='lifetime' or k in {'lifetime_spend','remaining_lifetime_break_even'} else 'recurring' if g=='run_rate' else 'period',inception_key='a',evidence_domain='BROKER' if k in TRADING_KEYS else 'PROJECT' if k in ECONOMIC_KEYS|CASH_ECONOMIC_KEYS else 'RECURRING' if g=='run_rate' else 'SNAPSHOT_OR_ASSUMPTION' if k in FORECAST_KEYS else 'EXPENSES') for g,k in e}
        return serialize(p,kind,journal=j,coverage_key='a',scope=scope,observed_count=1,expected_metrics=e,reconciliation_key='a',metric_contexts=contexts,policy_approvals={k:'attestation:test-only' for k in ('money_policy','reporting_currency','fx_source','recognition')})
    def test_H5_reader_good_and_repeatable(self):
        p,j,e=self.factory();self.assertEqual(self.validate(p,j,e),self.validate(p,j,e))
    def test_H5_positive_practice_is_hypothetical_only(self):
        p,j,e=self.factory();p.update(broker_environment='OANDA_PRACTICE',basis='HYPOTHETICAL_NON_CASH',classification='MIXED')
        for key in ('net_economic','cash_net_economic','remaining_cash_break_even'):
            p['metrics'][key].update(value=None,state='unavailable',classification='UNKNOWN',reason='Practice is non-cash')
        for key in ('gross_pnl','financing','other_broker_costs','net_trading'):
            p['metrics'][key].update(state='hypothetical',classification='HYPOTHETICAL')
        expected={(g,k):copy.deepcopy(v) for g,k in e for v in [p[g][k]]}
        j.append('reconciliation','a',Run((Result('t','MATCHED',(),1),),(),0,1,1,0,1,0,1,'fixture',PRACTICE,True),END)
        self.validate(p,j,expected,scope=PRACTICE)
    def test_H5_weekly_good(self):
        p,j,e=self.factory('weekly');self.validate(p,j,e,'weekly')
    def test_H5_all_reader_products(self):
        schema=json.loads(Path('finance/contracts/reader_products.schema.json').read_text())
        for v in schema['oneOf']:
            product=v['properties']['product']['const'];p,j,e=self.factory(product=product)
            if product=='finance-forecast-inputs':p['classification']='MIXED'
            with self.subTest(product=product):self.validate(p,j,e)
    def test_H5_practice_as_live_profit_reader(self):
        p,j,e=self.factory();p['broker_environment']='OANDA_PRACTICE'
        with self.assertRaises(ValueError):self.validate(p,j,e,scope=PRACTICE)
    def test_H5_practice_as_live_profit_weekly(self):
        p,j,e=self.factory('weekly');p['broker_environment']='OANDA_PRACTICE'
        with self.assertRaises(ValueError):self.validate(p,j,e,'weekly',PRACTICE)
    def test_H5_simulated_realised_reader_and_weekly(self):
        for kind in ('reader','weekly'):
            p,j,e=self.factory(kind);p['financial_reality']='SIMULATED';p['simulation_type']='COUNTERFACTUAL'
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate(p,j,e,kind)
    def test_H5_future_assertion_withdrawal_blocks_old_fact(self):
        p,j,e=self.factory();j.append('coverage','a',replace(coverage(1),state='WITHDRAWN'),'2100-01-01T00:00:00Z')
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_incomplete_as_verified_reader_and_weekly(self):
        for kind in ('reader','weekly'):
            p,j,e=self.factory(kind);j.append('coverage','a',coverage(1,False),END)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate(p,j,e,kind)
    def test_H5_unresolved_cannot_clean_break_even(self):
        p,j,e=self.factory(product='finance-break-even');b,o,t,pr=bridged();j.append('reconciliation','a',reconciliation(b,o,[],pr),END)
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_estimate_cannot_fact(self):
        p,j,e=self.factory();p['metrics']['spent']['state']='estimate'
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_cannot_change_producer_number(self):
        p,j,e=self.factory();p['metrics']['spent']['value']='1000'
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_unknown_cannot_fact(self):
        p,j,e=self.factory();p['metrics']['spent'].update(state='unavailable',value=None,classification='FACT')
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_no_arbitrary_extra_field(self):
        p,j,e=self.factory();p['secret']='raw-token'
        with self.assertRaises(Exception):self.validate(p,j,e)
    def test_H5_wrong_denominator(self):
        p,j,e=self.factory('weekly');p['denominator_basis']='attempted_orders'
        with self.assertRaises(ValueError):self.validate(p,j,e,'weekly')
    def test_H5_reconciliation_state_cannot_be_relabelled(self):
        p,j,e=self.factory();p['reconciliation_state']='PENDING_EOD_SYNC'
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_fabricated_trade_count(self):
        p,j,e=self.factory('weekly');p['trade_count']=155
        with self.assertRaises(ValueError):self.validate(p,j,e,'weekly')
    def test_H5_missing_recognition_policy_not_economic_fact(self):
        p,j,e=self.factory();p['recognition_policy_missing_count']=1
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_false_source_manifest_rejected(self):
        p,j,e=self.factory();p['source_manifest_id']='manifest:other'
        with self.assertRaises(ValueError):self.validate(p,j,e)
    def test_H5_precision_overflow(self):
        p,j,e=self.factory();p['metrics']['spent']['value']='0.1234567890123'
        with self.assertRaises(Exception):self.validate(p,j,e)

if __name__=='__main__':unittest.main()
