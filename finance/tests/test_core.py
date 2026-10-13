import io
import unittest
from dataclasses import replace, FrozenInstanceError
from decimal import Decimal, getcontext
from finance.core import *
from finance.oanda import normalise

START='2026-10-01T00:00:00Z'
END='2026-11-01T00:00:00Z'
def expense(**kw):
    args=dict(expense_id='e1', revision=1, source_key='invoice:render:123:line1',incurred_at=START,
        vendor='Render',description='Monthly service',category='infrastructure',original_amount='10',
        currency='GBP',fx_to_gbp='1',fx_source='GBP identity',fx_observed_at=START,evidence_ref='invoice:123',
        verification='invoice_verified',verified_by='CEO',service='engine',environment='practice',
        billing_start=START,billing_end=END)
    args.update(kw)
    return Expense(**args)
def component(**kw):
    args=dict(account='practice1',environment='practice',transaction_id='1',component_key='pl',at=START,
              kind='realised_pnl',amount='100',currency='GBP',fx_to_gbp='1')
    args.update(kw)
    return BrokerComponent(**args)
def scoped(**kw):
    return dict(account='practice1',environment='practice',currency='GBP',**kw)

class FinanceTests(unittest.TestCase):
    def test_decimal_arithmetic(self):
        l=ExpenseLedger();l.append(expense(original_amount='0.1'));l.append(expense(expense_id='e2',source_key='2',original_amount='0.2'))
        self.assertEqual(l.spend(START,END),Decimal('0.3'))
    def test_fx(self):
        self.assertEqual(expense(currency='USD',original_amount='20',fx_to_gbp='0.75').gbp,15)
    def test_bad_money(self):
        for v in (1.1, 'NaN','Infinity','-Infinity','1e19','0.0000000000001'):
            with self.subTest(v=v),self.assertRaises((ValueError,ArithmeticError)):money(v)
    def test_bad_currency(self):
        with self.assertRaises(ValueError):expense(currency='usd')
    def test_bad_fx(self):
        for r in ('0','-1','2'):
            with self.assertRaises(ValueError):expense(fx_to_gbp=r)
    def test_duplicate_retry(self):
        l=ExpenseLedger();self.assertTrue(l.append(expense()));self.assertFalse(l.append(expense()));self.assertEqual(l.spend(START,END),10)
    def test_duplicate_source(self):
        l=ExpenseLedger();l.append(expense())
        with self.assertRaises(ValueError):l.append(expense(expense_id='e2'))
    def test_conflicting_duplicate(self):
        l=ExpenseLedger();l.append(expense())
        with self.assertRaises(ValueError):l.append(expense(original_amount='12'))
    def test_correction(self):
        l=ExpenseLedger();l.append(expense());l.append(expense(revision=2,original_amount='12',correction_reason='Invoice correction'))
        self.assertEqual(l.spend(START,END),12);self.assertEqual(l.history[0].gbp,10);self.assertEqual(len(l.history),2)
    def test_frozen(self):
        with self.assertRaises(FrozenInstanceError):expense().vendor='X'
    def test_revision_gap(self):
        with self.assertRaises(ValueError):ExpenseLedger().append(expense(revision=2,correction_reason='gap'))
    def test_source_immutable(self):
        l=ExpenseLedger();l.append(expense())
        with self.assertRaises(ValueError):l.append(expense(revision=2,source_key='changed',correction_reason='change'))
    def test_estimates(self):
        l=ExpenseLedger();l.append(expense(verification='estimate'));self.assertEqual(l.spend(START,END),0);self.assertEqual(l.spend(START,END,False),10)
    def test_manual_verification(self):
        l=ExpenseLedger();l.append(expense(verification='manual_verified'));self.assertEqual(l.spend(START,END),10)
    def test_verifier_required(self):
        with self.assertRaises(ValueError):expense(verified_by='')
    def test_estimate_promotion(self):
        l=ExpenseLedger();l.append(expense(verification='estimate'));l.append(expense(revision=2,correction_reason='CEO verified invoice'))
        self.assertEqual(l.spend(START,END),10);self.assertEqual(l.history[0].verification,'estimate')
    def test_refund(self):
        l=ExpenseLedger();l.append(expense(original_amount='-3'));self.assertEqual(l.spend(START,END),-3)
    def test_half_open_period(self):
        l=ExpenseLedger();l.append(expense(incurred_at=END));self.assertEqual(l.spend(START,END),0)
    def test_timezone_required(self):
        with self.assertRaises(ValueError):expense(incurred_at='2026-10-01')
    def test_bad_billing_period(self):
        with self.assertRaises(ValueError):expense(billing_end=START)
    def test_atomic_csv(self):
        l=ExpenseLedger();e=expense();keys=list(e.__dict__);s=io.StringIO();w=csv.DictWriter(s,keys);w.writeheader()
        r={k:str(v) for k,v in e.__dict__.items()};r['recurring']='false';w.writerow(r);r['expense_id']='e2';w.writerow(r);s.seek(0)
        with self.assertRaises(ValueError):import_expenses(s,l)
        self.assertEqual(l.history,())
    def test_run_rate(self):
        r=run_rate([Subscription('render','30',START,END,'invoice_verified')],START)
        self.assertEqual(r['month'],30);self.assertEqual(r['year'],360)
        self.assertAlmostEqual(float(r['day']),360/365.25)
    def test_run_rate_excludes_estimates(self):
        self.assertEqual(run_rate([Subscription('x','30',START,END,'estimate')],START)['year'],0)
    def test_run_rate_expired(self):
        self.assertEqual(run_rate([Subscription('x','30',START,END,'invoice_verified')],END)['year'],0)
    def test_run_rate_overlap(self):
        r=Subscription('x','30',START,END,'invoice_verified')
        with self.assertRaises(ValueError):run_rate([r,r],START)
    def test_financing_and_fees(self):
        r=[component(),component(component_key='f',kind='financing',amount='-3'),component(component_key='c',kind='commission',amount='-2')]
        t=broker_totals(r,'practice1','practice',START,END);self.assertEqual(t['net_trading'],95);self.assertEqual(t['financing'],-3)
    def test_broker_dedup(self):
        r=component();self.assertEqual(broker_totals([r,r],'practice1','practice',START,END)['net_trading'],100)
    def test_broker_conflicting_duplicate(self):
        with self.assertRaises(ValueError):broker_totals([component(),component(amount='2')],'practice1','practice',START,END)
    def test_broker_scope(self):
        self.assertEqual(broker_totals([component(account='other')],'practice1','practice',START,END)['net_trading'],0)
    def test_deposits_excluded(self):
        self.assertEqual(broker_totals([component(kind='capital_transfer')],'practice1','practice',START,END)['net_trading'],0)
    def test_broker_null(self):
        with self.assertRaises(ValueError):component(amount=None)
    def test_break_even(self):
        t={'net_trading':Decimal('95')};e=economics('120',t,5,'live',True)
        self.assertEqual(e['net_economic_gbp'],-25);self.assertEqual(e['remaining_break_even_gbp'],25);self.assertEqual(e['allocated_cost_per_trade_gbp'],24);self.assertEqual(e['net_profit_per_trade_gbp'],-5)
    def test_profitable(self):
        e=economics('20',{'net_trading':Decimal('95')},5,'live',True)
        self.assertTrue(e['operationally_profitable']);self.assertEqual(e['remaining_break_even_gbp'],0)
    def test_practice_not_real_profit(self):
        e=economics('20',{'net_trading':Decimal('95')},5,'practice',True)
        self.assertIsNone(e['net_economic_gbp']);self.assertEqual(e['unit_economics_basis'],'practice_broker_actual')
    def test_shadow_exclusion(self):
        self.assertIsNone(economics('20',{'net_trading':Decimal('95')},5,'live',True,system_environment='SHADOW')['net_economic_gbp'])
    def test_counterfactual_exclusion(self):
        self.assertIsNone(economics('20',{'net_trading':Decimal('95')},5,'live',True,financial_reality='SIMULATED',simulation_type='COUNTERFACTUAL')['net_economic_gbp'])
    def test_incomplete(self):
        self.assertIsNone(economics('20',{'net_trading':Decimal('95')},5,'live',False)['remaining_break_even_gbp'])
    def test_no_trades(self):
        self.assertIsNone(economics('20',{'net_trading':Decimal('0')},0,'live',True)['allocated_cost_per_trade_gbp'])
    def reconciliation(self,b,o,t,p,now=END,due=END):
        return reconcile(b,o,t,p,now,due,'practice1','practice')
    def test_reconciliation_complete(self):
        self.assertEqual(self.reconciliation([scoped(trade_id='1',closed=True,realised_pnl='4')],[scoped(trade_id='1',signal_id='s')],[scoped(trade_id='1',realised_pnl='4')],[scoped(signal_id='s')]),())
    def test_missing_outcome(self):
        self.assertIn(('trade','1','outcome_overdue'),self.reconciliation([scoped(trade_id='1',closed=True)],[],[],[]))
    def test_pending_eod(self):
        self.assertIn(('trade','1','pending_eod'),self.reconciliation([scoped(trade_id='1',closed=True)],[],[],[],START,END))
    def test_pnl_mismatch(self):
        self.assertIn(('trade','1','pnl_mismatch'),self.reconciliation([scoped(trade_id='1',closed=True,realised_pnl='3')],[],[scoped(trade_id='1',realised_pnl='4')],[]))
    def test_missing_pnl(self):
        self.assertIn(('trade','1','pnl_missing'),self.reconciliation([scoped(trade_id='1',closed=True)],[],[scoped(trade_id='1',realised_pnl=None)],[]))
    def test_missing_prediction(self):
        self.assertIn(('trade','1','prediction_missing'),self.reconciliation([scoped(trade_id='1',closed=False)],[scoped(trade_id='1',signal_id='s')],[],[]))
    def test_reconciliation_duplicate(self):
        r=scoped(trade_id='1',closed=False);self.assertIn(('broker','1','duplicate_identity'),self.reconciliation([r,r],[],[],[]))
    def test_reconciliation_scope(self):
        self.assertIn(('broker','','scope_missing'),self.reconciliation([dict(trade_id='1')],[],[],[]))
    def test_currency_missing(self):
        r=scoped(trade_id='1',closed=True,realised_pnl='3');del r['currency']
        self.assertIn(('broker','1','currency_missing'),self.reconciliation([r],[],[scoped(trade_id='1',realised_pnl='3')],[]))
    def test_currency_mismatch(self):
        r=scoped(trade_id='1',closed=True,realised_pnl='3');r['currency']='USD'
        self.assertIn(('trade','1','currency_mismatch'),self.reconciliation([r],[],[scoped(trade_id='1',realised_pnl='3')],[]))
    def test_unknown_broker_state(self):
        self.assertIn(('trade','1','broker_state_missing'),self.reconciliation([scoped(trade_id='1',closed=None)],[],[],[]))
    def test_repeatability(self):
        r=[component(),component(component_key='f',kind='financing',amount='-3')]
        self.assertEqual(broker_totals(r,'practice1','practice',START,END),broker_totals(list(reversed(r)),'practice1','practice',START,END))
    def tx(self,**kw):
        r=dict(accountID='private-export-id',id='1',time=START,type='ORDER_FILL',accountBalance='1000',pl='100',financing='-3',commission='2',guaranteedExecutionFee='1',tradesClosed=[{'realizedPL':'100'}],halfSpreadCost='5');r.update(kw);return r
    def adapt(self,tx):
        return normalise(tx,'practice1','private-export-id','practice','GBP','1')
    def test_adapter_no_nested_double_count(self):
        rows,d=self.adapt(self.tx());self.assertEqual(broker_totals(rows,'practice1','practice',START,END)['net_trading'],94);self.assertEqual(d['halfSpreadCost'],'5')
    def test_adapter_missing(self):
        tx=self.tx();del tx['financing']
        with self.assertRaises(ValueError):self.adapt(tx)
    def test_adapter_unknown(self):
        with self.assertRaises(ValueError):self.adapt(self.tx(type='NEW_TYPE'))
    def test_adapter_scope(self):
        with self.assertRaises(ValueError):self.adapt(self.tx(accountID='wrong'))
    def test_daily_financing(self):
        rows,_=self.adapt(self.tx(type='DAILY_FINANCING',positionFinancings=[{'financing':'-3'}]));self.assertEqual(broker_totals(rows,'practice1','practice',START,END)['financing'],-3)
    def test_negative_fee_rejected(self):
        with self.assertRaises(ValueError):self.adapt(self.tx(commission='-2'))

if __name__=='__main__':unittest.main()
