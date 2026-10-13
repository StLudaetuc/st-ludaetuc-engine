"""Independent adversarial tests (cross-provider review) for finance @ 189587ce.

Stdlib only. Tests named test_finding_* assert a property the review requires
and are EXPECTED TO FAIL at the reviewed SHA; every other test asserts a
property that holds and must keep holding.

    python -m unittest discover -s finance/tests_independent -t .
"""
import ast
import io
import unittest
from decimal import Decimal
from pathlib import Path

from finance import core, oanda
from finance.core import BrokerComponent, Expense, ExpenseLedger, Subscription

ROOT = Path(__file__).resolve().parent.parent
T0 = '2026-10-01T00:00:00Z'


def expense(**over):
    base = dict(expense_id='e1', revision=1, source_key='inv-1:line-1', incurred_at=T0, vendor='Render',
                description='hosting', category='infrastructure', original_amount='10.00', currency='GBP',
                fx_to_gbp='1', fx_source='identity', fx_observed_at=T0, evidence_ref='invoice:render:2026-10',
                verification='invoice_verified', service='engine', environment='PRODUCTION',
                billing_start=T0, billing_end='2026-11-01T00:00:00Z', recurring=True, verified_by='ceo')
    base.update(over)
    return Expense(**base)


CSV_HEADER = ('expense_id,revision,source_key,incurred_at,vendor,description,category,original_amount,currency,'
              'fx_to_gbp,fx_source,fx_observed_at,evidence_ref,verification,service,environment,billing_start,'
              'billing_end,recurring,correction_reason,verified_by\n')


def csv_row(**over):
    row = dict(expense_id='e1', revision='1', source_key='k1', incurred_at=T0, vendor='Render', description='hosting',
               category='infrastructure', original_amount='10.00', currency='GBP', fx_to_gbp='1', fx_source='identity',
               fx_observed_at=T0, evidence_ref='invoice:render:1', verification='invoice_verified', service='engine',
               environment='PRODUCTION', billing_start=T0, billing_end='2026-11-01T00:00:00Z', recurring='true',
               correction_reason='', verified_by='ceo')
    row.update(over)
    import csv as _csv
    out = io.StringIO()
    _csv.writer(out, lineterminator='\n').writerow([row[k] for k in CSV_HEADER.strip().split(',')])
    return out.getvalue()


def comp(tx, key, kind, amount, env='live', at=T0, account='a1', fx='1', ccy='GBP'):
    return BrokerComponent(account, env, tx, key, at, kind, amount, ccy, fx)


ZERO = {k: Decimal(0) for k in ('realised_pnl', 'financing', 'commission', 'conversion', 'other_adjustment',
                                'capital_transfer', 'net_trading')}


# ----------------------------------------------------------------------------------------------- 3 decimal
class DecimalIntegrity(unittest.TestCase):
    def test_no_binary_float_in_authoritative_finance_code(self):
        for name in ('core.py', 'oanda.py'):
            tree = ast.parse((ROOT / name).read_text())
            floats = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
            calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'float']
            self.assertEqual((floats, calls), ([], []), name)

    def test_money_rejects_float_nan_inf_and_excess_precision(self):
        for bad in (0.1, 1, True, 'NaN', 'sNaN', 'Infinity', '-inf', '1e18', '0.0000000000001', '', 'abc'):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, ArithmeticError, TypeError)):
                    core.money(bad)
        self.assertEqual(core.money('0.000000000001'), Decimal('1E-12'))
        self.assertEqual(core.money('-12345.000000000001'), Decimal('-12345.000000000001'))

    def test_finding_money_accepts_the_full_sql_money_domain(self):
        # SQL finance.money accepts abs<1e18 with scale<=12. core.money() computes abs() under the default
        # 28-digit context; values within ~1e-10 of the bound round up to 1E+18 and are rejected (LOW, boundary).
        for v in ('999999999999999999.999999999999', '-999999999999999999.9999999999'):
            with self.subTest(v=v):
                try:
                    self.assertEqual(core.money(v), Decimal(v))
                except ValueError:
                    self.fail(f'{v} is valid finance.money in SQL but rejected by core.money()')

    def test_fx_conversion_exact_no_rounding_drift(self):
        e = expense(original_amount='0.000000000001', currency='USD', fx_to_gbp='0.999999999999')
        self.assertEqual(e.gbp, Decimal('0.000000000000999999999999'))
        e = expense(original_amount='12345678901234.123456789012', currency='USD', fx_to_gbp='0.123456789012')
        self.assertEqual(e.gbp, Decimal('1524157875319.561166019561153483936144'))

    def test_refund_nets_against_spend(self):
        led = ExpenseLedger()
        led.append(expense())
        led.append(expense(expense_id='e2', source_key='refund-1', original_amount='-4.00', description='refund'))
        self.assertEqual(led.spend(T0, '2026-11-01T00:00:00Z'), Decimal('6.00'))


# ----------------------------------------------------------------------------------------------- 4 ledger
class ExpenseLedgerAdversarial(unittest.TestCase):
    def test_duplicate_source_line_under_new_expense_id_rejected(self):
        led = ExpenseLedger()
        led.append(expense())
        with self.assertRaises(ValueError):
            led.append(expense(expense_id='e2'))

    def test_idempotent_and_conflicting_retry(self):
        led = ExpenseLedger()
        self.assertTrue(led.append(expense()))
        self.assertFalse(led.append(expense()))
        with self.assertRaises(ValueError):
            led.append(expense(original_amount='11.00'))
        self.assertEqual(len(led.history), 1)

    def test_revision_gap_and_source_change_rejected(self):
        led = ExpenseLedger()
        led.append(expense())
        with self.assertRaises(ValueError):
            led.append(expense(revision=3, correction_reason='x'))
        with self.assertRaises(ValueError):
            led.append(expense(revision=2, correction_reason='x', source_key='other'))

    def test_correction_preserves_original_evidence(self):
        led = ExpenseLedger()
        led.append(expense())
        led.append(expense(revision=2, original_amount='12.00', correction_reason='invoice re-issued'))
        self.assertEqual([r.original_amount for r in led.history], [Decimal('10.00'), Decimal('12.00')])
        self.assertEqual(led.spend(T0, '2026-11-01T00:00:00Z'), Decimal('12.00'))

    def test_estimate_and_unverified_excluded_from_verified_spend(self):
        led = ExpenseLedger()
        led.append(expense(verification='estimate', verified_by=''))
        led.append(expense(expense_id='e2', source_key='k2', verification='unverified', verified_by=''))
        self.assertEqual(led.spend(T0, '2026-11-01T00:00:00Z'), Decimal(0))

    def test_missing_date_currency_or_fx_rejected(self):
        for over in (dict(incurred_at=''), dict(incurred_at='2026-10-01T00:00:00'), dict(currency=''),
                     dict(currency='gbp'), dict(fx_to_gbp=''), dict(fx_to_gbp='0'), dict(currency='USD', fx_to_gbp='-1'),
                     dict(currency='GBP', fx_to_gbp='1.1'), dict(evidence_ref=''), dict(verified_by='')):
            with self.subTest(over=over):
                with self.assertRaises((ValueError, ArithmeticError, TypeError, AttributeError)):
                    expense(**over)

    def test_batch_import_is_atomic(self):
        led = ExpenseLedger()
        led.append(expense(source_key='k0', expense_id='e0'))
        stream = io.StringIO(CSV_HEADER + csv_row(expense_id='e1', source_key='k1')
                             + csv_row(expense_id='e2', source_key='k0'))  # second row collides
        with self.assertRaises(ValueError):
            core.import_expenses(stream, led)
        self.assertEqual([r.expense_id for r in led.history], ['e0'])

    def test_overlapping_subscription_schedules_rejected(self):
        subs = [Subscription('render', '30', T0, '2027-01-01T00:00:00Z', 'invoice_verified'),
                Subscription('render', '40', '2026-12-01T00:00:00Z', '2027-02-01T00:00:00Z', 'invoice_verified')]
        with self.assertRaises(ValueError):
            core.run_rate(subs, '2026-12-15T00:00:00Z')

    def test_finding_run_rate_reports_excluded_uncertain_schedules(self):
        subs = [Subscription('api', '100', T0, '2027-01-01T00:00:00Z', 'estimate')]
        result = core.run_rate(subs, '2026-10-15T00:00:00Z')
        self.assertTrue(any('uncertain' in k or 'excluded' in k for k in result),
                        'run rate silently drops estimate/unverified schedules: 0 looks like "no subscriptions"')


# ----------------------------------------------------------------------------------------------- 10 import
class HistoricalImportSecurity(unittest.TestCase):
    def _import(self, **over):
        return core.import_expenses(io.StringIO(CSV_HEADER + csv_row(**over)), ExpenseLedger())

    def test_malformed_headers_and_bom_fail_closed(self):
        for text in ('﻿' + CSV_HEADER + csv_row(), CSV_HEADER.replace('vendor', 'vendr') + csv_row(),
                     CSV_HEADER + csv_row().rstrip('\n') + ',extra\n'):
            with self.subTest(text=text[:20]):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    core.import_expenses(io.StringIO(text), ExpenseLedger())

    def test_finding_formula_injection_rejected(self):
        for payload in ('=HYPERLINK("http://x","y")', '+cmd|"/c calc"!A1', '-2+3', '@SUM(1,1)'):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError, msg='spreadsheet formula accepted into evidence text'):
                    self._import(vendor=payload)

    def test_finding_evidence_ref_scheme_and_path_restricted(self):
        for ref in ('javascript:alert(1)', 'file:///etc/passwd', '../../../etc/passwd', 'http://attacker.example/x'):
            with self.subTest(ref=ref):
                with self.assertRaises(ValueError, msg='unsafe evidence reference accepted'):
                    self._import(evidence_ref=ref)

    def test_finding_unsupported_currency_rejected(self):
        for ccy in ('XXX', 'ABC', 'ZZZ'):
            with self.subTest(ccy=ccy):
                with self.assertRaises(ValueError):
                    self._import(currency=ccy, fx_to_gbp='1.5')

    def test_finding_bulk_import_cannot_self_assert_verification(self):
        with self.assertRaises(ValueError, msg='CSV row self-declares invoice_verified; verification spoofable'):
            self._import(verification='invoice_verified', verified_by='anyone')

    def test_finding_import_bounds_input_size(self):
        self.assertIn('max_rows', core.import_expenses.__code__.co_varnames,
                      'no row/size bound on historical import (memory exhaustion)')


# ----------------------------------------------------------------------------------------------- 5 broker
def fill(**over):
    tx = {'id': '1001', 'accountID': '001-004-1-001', 'time': '2026-10-01T12:00:00.000000000Z', 'type': 'ORDER_FILL',
          'accountBalance': '10050.0000', 'pl': '60.0000', 'financing': '-1.2500', 'commission': '2.0000',
          'guaranteedExecutionFee': '0.0000',
          'tradesClosed': [{'tradeID': '7', 'units': '-500', 'realizedPL': '25.0000'},
                           {'tradeID': '8', 'units': '-500', 'realizedPL': '35.0000'}]}
    tx.update(over)
    return tx


def norm(tx, env='live'):
    return oanda.normalise(tx, 'acct-alias', '001-004-1-001', env, 'GBP', '1')


class BrokerEconomics(unittest.TestCase):
    def test_signs_and_no_nested_double_count(self):
        comps, diag = norm(fill())
        totals = core.broker_totals(comps, 'acct-alias', 'live', T0, '2026-11-01T00:00:00Z')
        self.assertEqual(totals['realised_pnl'], Decimal('60'))      # top-level only; nested 25+35 not added
        self.assertEqual(totals['financing'], Decimal('-1.25'))
        self.assertEqual(totals['commission'], Decimal('-2'))
        self.assertEqual(totals['net_trading'], Decimal('56.75'))
        self.assertIn('tradesClosed', diag)

    def test_transfers_excluded_from_trading(self):
        comps, _ = norm({'id': '9', 'accountID': '001-004-1-001', 'time': T0, 'type': 'TRANSFER_FUNDS',
                         'accountBalance': '1', 'amount': '5000'})
        totals = core.broker_totals(comps, 'acct-alias', 'live', T0, '2026-11-01T00:00:00Z')
        self.assertEqual((totals['capital_transfer'], totals['net_trading']), (Decimal(5000), Decimal(0)))

    def test_unknown_missing_float_and_negative_fee_quarantined(self):
        for tx in (fill(type='DIVIDEND_ADJUSTMENT'), fill(type='MARGIN_CALL_ENTER'), {k: v for k, v in fill().items() if k != 'pl'},
                   fill(pl=60.0), fill(commission='-2.0'), fill(accountID='101-004-1-001'), fill(time='2026-10-01T12:00:00')):
            with self.subTest(tx=tx.get('type')):
                with self.assertRaises((ValueError, TypeError)):
                    norm(tx)

    def test_duplicate_ingestion_idempotent_conflict_rejected(self):
        comps, _ = norm(fill())
        totals = core.broker_totals(comps + comps, 'acct-alias', 'live', T0, '2026-11-01T00:00:00Z')
        self.assertEqual(totals['realised_pnl'], Decimal('60'))
        altered, _ = norm(fill(pl='61.0000'))
        with self.assertRaises(ValueError):
            core.broker_totals(comps + altered, 'acct-alias', 'live', T0, '2026-11-01T00:00:00Z')

    def test_finding_nested_trade_attribution_must_reconcile_to_top_level_pl(self):
        with self.assertRaises(ValueError, msg='nested realizedPL (10) disagrees with top-level pl (60) undetected'):
            norm(fill(tradesClosed=[{'tradeID': '7', 'units': '-1', 'realizedPL': '10.0000'}]))


# ----------------------------------------------------------------------------------------------- 6/9 practice + break-even
class PracticeVersusActual(unittest.TestCase):
    def test_practice_never_produces_economic_conclusion(self):
        totals = dict(ZERO, net_trading=Decimal('1000000'))
        r = core.economics('10', totals, 5, 'practice', True)
        self.assertIsNone(r['net_economic_gbp'])
        self.assertIsNone(r['remaining_break_even_gbp'])
        self.assertIsNone(r['operationally_profitable'])

    def test_estimates_and_simulation_suppress_conclusion(self):
        totals = dict(ZERO, net_trading=Decimal('100'))
        for kw in (dict(simulation_type='SHADOW'), dict(financial_state='UNREALISED'), dict(system_environment='DEMO'),
                   dict(financial_reality='SIMULATED')):
            with self.subTest(kw=kw):
                self.assertIsNone(core.economics('10', totals, 1, 'live', True, **kw)['net_economic_gbp'])
        self.assertIsNone(core.economics('10', totals, 1, 'live', False)['net_economic_gbp'])

    def test_break_even_boundaries(self):
        totals = dict(ZERO, net_trading=Decimal('100'))
        exact = core.economics('100', totals, 4, 'live', True)
        self.assertEqual((exact['net_economic_gbp'], exact['remaining_break_even_gbp'], exact['operationally_profitable']),
                         (Decimal(0), Decimal(0), False))
        short = core.economics('100.01', totals, 4, 'live', True)
        self.assertEqual(short['remaining_break_even_gbp'], Decimal('0.01'))

    def test_finding_live_scope_must_be_explicit_not_defaulted(self):
        totals = dict(ZERO, net_trading=Decimal('100'))
        with self.assertRaises(TypeError, msg='system_environment/financial_reality default to PRODUCTION/ACTUAL'):
            core.economics('10', totals, 1, 'live', True)

    def test_finding_practice_unit_economics_not_mixed_with_actual_spend(self):
        totals = dict(ZERO, net_trading=Decimal('1000'))
        r = core.economics('10', totals, 5, 'practice', True)
        self.assertIsNone(r['net_profit_per_trade_gbp'],
                          'Practice P&L minus actual project spend reported as net profit per trade')

    def test_finding_empty_evidence_cannot_conclude_break_even(self):
        r = core.economics('0', dict(ZERO), 0, 'live', True)
        self.assertIsNone(r['remaining_break_even_gbp'],
                          'no expenses, no trades, coverage asserted => "remaining break-even 0"')


# ----------------------------------------------------------------------------------------------- 7 reconciliation
def rec(broker=(), opened=(), outcomes=(), predictions=(), as_of='2026-10-01T20:00:00Z',
        due='2026-10-01T23:00:00Z'):
    return core.reconcile(list(broker), list(opened), list(outcomes), list(predictions), as_of, due, 'a1', 'live')


S = dict(account='a1', environment='live')


class Reconciliation(unittest.TestCase):
    chain = dict(predictions=[dict(S, signal_id='sig1')], opened=[dict(S, trade_id='t1', signal_id='sig1')])

    def test_matched_chain_has_no_exceptions(self):
        r = rec(broker=[dict(S, trade_id='t1', closed=True, realised_pnl='5.00', currency='GBP')],
                outcomes=[dict(S, trade_id='t1', realised_pnl='5.0', currency='GBP')], **self.chain)
        self.assertEqual(r, ())

    def test_pending_eod_then_overdue_exactly_at_deadline(self):
        broker = [dict(S, trade_id='t1', closed=True, realised_pnl='5', currency='GBP')]
        self.assertIn(('trade', 't1', 'pending_eod'), rec(broker=broker, **self.chain))
        self.assertIn(('trade', 't1', 'outcome_overdue'), rec(broker=broker, as_of='2026-10-01T23:00:00Z', **self.chain))

    def test_value_currency_duplicate_and_false_closure(self):
        b = [dict(S, trade_id='t1', closed=True, realised_pnl='5', currency='GBP')]
        self.assertIn(('trade', 't1', 'pnl_mismatch'),
                      rec(broker=b, outcomes=[dict(S, trade_id='t1', realised_pnl='5.01', currency='GBP')], **self.chain))
        self.assertIn(('trade', 't1', 'currency_mismatch'),
                      rec(broker=b, outcomes=[dict(S, trade_id='t1', realised_pnl='5', currency='USD')], **self.chain))
        dup = rec(broker=b, outcomes=[dict(S, trade_id='t1', realised_pnl='5', currency='GBP')] * 2, **self.chain)
        self.assertIn(('outcome', 't1', 'duplicate_identity'), dup)
        open_b = [dict(S, trade_id='t1', closed=False, realised_pnl=None, currency='GBP')]
        self.assertIn(('trade', 't1', 'outcome_without_broker_closure'),
                      rec(broker=open_b, outcomes=[dict(S, trade_id='t1', realised_pnl='5', currency='GBP')], **self.chain))

    def test_wrong_scope_rows_flagged(self):
        self.assertIn(('broker', '', 'scope_missing'),
                      rec(broker=[dict(trade_id='t1', account='a2', environment='live', closed=True)], **self.chain))

    def test_finding_partial_close_is_not_a_duplicate_failure(self):
        partials = [dict(S, trade_id='t1', transaction_id='100', closed=False, realised_pnl='2', currency='GBP'),
                    dict(S, trade_id='t1', transaction_id='101', closed=True, realised_pnl='3', currency='GBP')]
        r = rec(broker=partials, outcomes=[dict(S, trade_id='t1', realised_pnl='5', currency='GBP')], **self.chain)
        self.assertEqual(r, (), f'partial close (2+3=5) reported as {r}')

    def test_finding_broker_sync_lag_before_eod_is_pending_not_missing(self):
        r = rec(**self.chain)  # opened, broker EOD export not yet in, before deadline
        self.assertNotIn(('trade', 't1', 'broker_record_missing'), r,
                         'temporary broker-export lag before EOD reported as missing broker record')

    def test_finding_malformed_pnl_is_an_exception_not_a_crash(self):
        b = [dict(S, trade_id='t1', closed=True, realised_pnl='5', currency='GBP')]
        try:
            r = rec(broker=b, outcomes=[dict(S, trade_id='t1', realised_pnl=5.0, currency='GBP')], **self.chain)
        except ValueError:
            self.fail('one float realised_pnl aborts the whole reconciliation run')
        self.assertTrue(r)


# ----------------------------------------------------------------------------------------------- 11/12 contracts
class ContractSemantics(unittest.TestCase):
    def _schema(self, name):
        import json
        return json.loads((ROOT / 'contracts' / name).read_text())

    def test_reader_products_bounded_and_no_private_identifiers(self):
        import json
        text = json.dumps(self._schema('reader_products.schema.json'))
        for forbidden in ('accountID', 'account_id', 'invoice_number', 'iban', 'card', '"sql"', '"filter"', '"query"'):
            self.assertNotIn(forbidden, text)
        self.assertNotIn('"type": "array"', text)

    def test_finding_weekly_schema_forbids_verified_actual_from_practice_or_incomplete(self):
        text = (ROOT / 'contracts' / 'weekly_report.schema.json').read_text()
        self.assertRegex(text, r'"if"\s*:', 'no conditional rule: Practice/SIMULATED/incomplete report may label '
                                            'net_economic_pnl "verified"')

    def test_market_data_v2_rejects_placeholders_and_has_no_strategy_fields(self):
        import json
        from finance import market_data
        example = json.loads((ROOT / 'contracts' / 'market_data_v2.example.json').read_text())
        market_data.validate(dict(example))
        for key, val in (('ticker', '{{ticker}}'), ('close', '{{close}}'), ('timeframe', '{{interval}}'),
                         ('time', '{{time}}'), ('is_confirmed', 'true')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    market_data.validate(dict(example, **{key: val}))
        for feature in ('signal', 'probability', 'rsi', 'ema', 'atr', 'score', 'prediction', 'strategy'):
            self.assertNotIn(feature, market_data.FIELDS)


if __name__ == '__main__':
    unittest.main()
