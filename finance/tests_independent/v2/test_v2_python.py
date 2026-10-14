"""Independent adversarial tests for Finance candidate 4da760ba (Python, no database).

Written by the independent reviewer, not the implementation provider. Tests named
``test_finding_*`` encode a property the reviewer requires; they FAIL BY DESIGN while
the finding is open. Every other test is a regression or a confirmed property.

Run from the repository root of a checkout of the reviewed SHA:
    python -m unittest discover -s finance/tests_independent/v2 -t .
"""
import copy
import io
import json
import sys
import threading
import unittest
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import finance.core as core
from finance.policy import (Scope, Coverage, AuthorityJournal, money, quantise, convert, divide,
                            instant, POLICY_VERSION)
from finance.reconciliation import reconcile_run, ReconciliationJournal, Run, Result
from finance.oanda import normalise, normalise_batch

S, E = '2026-10-01T00:00:00Z', '2026-11-01T00:00:00Z'
LIVE = Scope('OANDA_LIVE', 'PRODUCTION', 'ACTUAL', 'REALISED', None)
PRACTICE = Scope('OANDA_PRACTICE', 'PRODUCTION', 'ACTUAL', 'REALISED', None)
ROOT = Path(core.__file__).resolve().parent


def cov(count, state=None, start=S, end=E, unresolved=0):
    state = state or ('VERIFIED' if count else 'REVIEWED_EMPTY')
    has = state == 'VERIFIED' or (state not in ('REVIEWED_EMPTY',) and count)
    return Coverage(state, count, start, end, 'first' if has else None, 'last' if has else None,
                    'manifest:review', 'independent-reviewer', unresolved, True, True)


def mk_expense(**kw):
    args = dict(expense_id='e1', revision=1, source_key='invoice:x:1', incurred_at=S, vendor='Render',
                description='svc', category='infrastructure', original_amount='10', currency='GBP',
                fx_to_gbp='1', fx_source='identity', fx_observed_at=S, evidence_ref='invoice:x1',
                verification='unverified', service='engine', environment='PRODUCTION',
                billing_start=S, billing_end=E, paid_at=S, recognition_policy='point_in_time_v1')
    args.update(kw)
    return core.Expense(**args)


def verified_ledger(*rows):
    ledger = core.ExpenseLedger()
    for row in rows:
        ledger.append(row)
        ledger.verify(row.expense_id, row.revision, 'invoice_verified', 'reviewer', 'invoice:ok', S)
    return ledger


def acct(alias, value, scope=LIVE, complete=True):
    return {'account': alias, 'scope': scope, 'coverage_complete': complete,
            'net_trading': money(value) if value is not None else None}


def row(**kw):
    base = dict(account='a1', scope=LIVE, currency='GBP', event_time=S, received_at=S)
    base.update(kw)
    return base


# --------------------------------------------------------------------------- H1
class H1Ordering(unittest.TestCase):
    def test_future_dated_verification_cannot_beat_later_withdrawal(self):
        ledger = core.ExpenseLedger()
        ledger.append(mk_expense())
        ledger.verify('e1', 1, 'invoice_verified', 'r', 'invoice:a', '2199-01-01T00:00:00Z')
        ledger.verify('e1', 1, 'withdrawn', 'r', 'evidence:w', '2000-01-01T00:00:00Z')
        self.assertEqual(ledger.spend(S, E), 0)

    def test_verification_never_carries_to_new_revision(self):
        ledger = verified_ledger(mk_expense())
        ledger.append(mk_expense(revision=2, original_amount='99', correction_reason='fix'))
        self.assertEqual(ledger.spend(S, E), 0)
        with self.assertRaises(ValueError):  # stale revision cannot be verified
            ledger.verify('e1', 1, 'invoice_verified', 'r', 'invoice:a', S)

    def test_journal_orders_are_unique_and_monotonic_under_threads(self):
        journal, out = AuthorityJournal(), []
        def work(i):
            out.append(journal.append('coverage', 'k', i, S))
        threads = [threading.Thread(target=work, args=(i,)) for i in range(64)]
        [t.start() for t in threads]; [t.join() for t in threads]
        self.assertEqual(sorted(out), list(range(1, 65)))
        self.assertEqual([r[0] for r in journal.history], list(range(1, 65)))


# --------------------------------------------------------------------------- H2
class H2EmptyEvidence(unittest.TestCase):
    def test_unknown_coverage_yields_null_not_zero(self):
        ledger = core.ExpenseLedger()
        for state in ('UNKNOWN', 'INCOMPLETE', 'WITHDRAWN'):
            with self.subTest(state=state):
                e = core.expense_totals(ledger, S, E, cov(0, state))
                self.assertIsNone(e['cash_spend']); self.assertIsNone(e['economic_expense'])
                p = core.project_economics([], e, cov(0, state), S, E, trade_count=0, denominator_basis='closed_trades')
                self.assertIsNone(p['cash_net_economic_gbp']); self.assertIsNone(p['remaining_cash_break_even_gbp'])

    def test_count_mismatch_is_unavailable(self):
        ledger = verified_ledger(mk_expense(), mk_expense(expense_id='e2', source_key='invoice:x:2'))
        self.assertIsNone(core.expense_totals(ledger, S, E, cov(1))['cash_spend'])
        self.assertEqual(core.expense_totals(ledger, S, E, cov(2))['cash_spend'], 20)

    def test_broker_without_matching_coverage_is_unavailable(self):
        rows = [core.BrokerComponent('a1', LIVE, 't1', 'pl', S, 'realised_pnl', '5', 'GBP', '1')]
        self.assertIsNone(core.broker_totals(rows, 'a1', LIVE, S, E, None)['net_trading'])
        self.assertIsNone(core.broker_totals(rows, 'a1', LIVE, S, E, cov(2))['net_trading'])
        self.assertEqual(core.broker_totals(rows, 'a1', LIVE, S, E, cov(1))['net_trading'], 5)

    def test_verified_coverage_cannot_be_empty(self):
        with self.assertRaises(ValueError):
            Coverage('VERIFIED', 0, S, E, None, None, 'manifest:x', 'r', 0, True, True)

    def test_coverage_for_october_does_not_support_november(self):
        self.assertFalse(cov(1).supports('2026-11-01T00:00:00Z', '2026-12-01T00:00:00Z', 1))


# --------------------------------------------------------------------------- H3
class H3ProjectLevel(unittest.TestCase):
    def setUp(self):
        self.exp = {'cash_spend': money('12'), 'economic_expense': money('12'), 'coverage_complete': True}

    def test_spend_deducted_once_across_live_accounts_practice_excluded(self):
        accounts = [acct('a', '95'), acct('b', '45'), acct('p', '9999', PRACTICE)]
        r = core.project_economics(accounts, self.exp, cov(3), S, E, trade_count=2, denominator_basis='closed_trades')
        self.assertEqual(r['actual_net_trading_gbp'], 140)
        self.assertEqual(r['cash_net_economic_gbp'], 128)
        self.assertEqual(r['eligible_accounts'], 2)
        self.assertFalse(r['practice_profit_applied_to_break_even'])

    def test_incomplete_eligible_account_blocks_project(self):
        r = core.project_economics([acct('a', '95'), acct('b', None, complete=False)], self.exp, cov(2), S, E,
                                   trade_count=1, denominator_basis='closed_trades')
        self.assertIsNone(r['cash_net_economic_gbp'])

    def test_truthy_non_bool_completeness_rejected(self):
        a = acct('a', '95'); a['coverage_complete'] = 'yes'
        r = core.project_economics([a], self.exp, cov(1), S, E, trade_count=1, denominator_basis='closed_trades')
        self.assertIsNone(r['cash_net_economic_gbp'])

    def test_duplicate_account_rejected(self):
        with self.assertRaises(ValueError):
            core.project_economics([acct('a', '1'), acct('a', '1')], self.exp, cov(2), S, E, trade_count=0,
                                   denominator_basis='closed_trades')


# --------------------------------------------------------------------------- H4
class H4ExplicitScope(unittest.TestCase):
    def test_scope_has_no_defaults(self):
        with self.assertRaises(TypeError):
            Scope('OANDA_LIVE')

    def test_legacy_and_lowercase_labels_rejected(self):
        for args in [('production', 'PRODUCTION', 'ACTUAL', 'REALISED', None),
                     ('live', 'PRODUCTION', 'ACTUAL', 'REALISED', None),
                     ('OANDA_LIVE', 'production', 'ACTUAL', 'REALISED', None),
                     ('OANDA_LIVE', 'PRODUCTION', 'SIMULATED', 'REALISED', 'SHADOW'),
                     ('OANDA_LIVE', 'PRODUCTION', 'ACTUAL', 'REALISED', 'SHADOW'),
                     ('OANDA_LIVE', 'PRODUCTION', 'SIMULATED', 'UNREALISED', None)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Scope(*args)

    def test_only_live_production_actual_realised_is_cash(self):
        self.assertTrue(LIVE.live_cash)
        for s in (PRACTICE, Scope('OANDA_LIVE', 'SHADOW', 'ACTUAL', 'REALISED', None),
                  Scope('OANDA_LIVE', 'PRODUCTION', 'ACTUAL', 'UNREALISED', None),
                  Scope('OANDA_LIVE', 'PRODUCTION', 'SIMULATED', 'UNREALISED', 'COUNTERFACTUAL')):
            with self.subTest(scope=s):
                self.assertFalse(s.live_cash)

    def test_adapter_requires_explicit_scope_object(self):
        tx = dict(accountID='x', id='1', time=S, type='DAILY_FINANCING', accountBalance='1', financing='-1')
        with self.assertRaises(ValueError):
            normalise(tx, 'a1', 'x', 'live', 'GBP', '1')


# --------------------------------------------------------------------------- H5 serializer
GROUPS = ('metrics', 'lifetime', 'this_week', 'run_rate', 'unit_economics', 'forecast_inputs')


def schema_for(kind, product='finance-summary'):
    d = json.loads((ROOT / 'contracts' / ('reader_products.schema.json' if kind == 'reader'
                                          else 'weekly_report.schema.json')).read_text())
    return next(v for v in d['oneOf'] if v['properties']['product']['const'] == product) if kind == 'reader' else d


def build(kind='reader', product='finance-summary', scope=LIVE, start=S, end=E):
    """Independent valid-payload builder driven only by the published schema."""
    sch, p = schema_for(kind, product), {}
    for key, rule in sch['properties'].items():
        if 'const' in rule: p[key] = rule['const']
        elif key == 'period_start': p[key] = start
        elif key in ('as_of', 'period_end'): p[key] = end
        elif key in ('trade_count', 'evidence_count'): p[key] = 1
        elif key.endswith('_count'): p[key] = 0
        elif key in ('coverage_complete', 'currency_confirmed', 'costs_complete'): p[key] = True
        elif key == 'first_evidence_id': p[key] = 'first'
        elif key == 'last_evidence_id': p[key] = 'last'
        elif key == 'simulation_type': p[key] = None
        elif key in GROUPS:
            p[key] = {}
            for name, m in rule['properties'].items():
                if m['$ref'].endswith('/drawdown'):
                    p[key][name] = {'value': None, 'unit': 'fraction_of_capital', 'state': 'unavailable',
                                    'classification': 'UNKNOWN', 'reason': 'none', 'evidence_id': None}
                else:
                    p[key][name] = {'value': '1.000000000000', 'currency': 'GBP', 'state': 'verified',
                                    'classification': 'FACT', 'reason': None, 'evidence_id': 'evidence:m'}
        else: p[key] = 'fixture'
    p.update(broker_environment=scope.broker_environment, system_environment=scope.system_environment,
             financial_reality=scope.financial_reality, financial_state=scope.financial_state,
             simulation_type=scope.simulation_type, scope_level='PROJECT', basis='ACTUAL_CASH',
             classification='MIXED' if kind == 'weekly' else 'FACT', coverage_state='VERIFIED',
             reconciliation_state='MATCHED', denominator_basis='closed_trades', source_manifest_id='manifest:review',
             account_alias='a1')
    j = AuthorityJournal()
    j.append('coverage', 'k', cov(1, start=start, end=end), S)
    j.append('reconciliation', 'k', Run((Result('t', 'MATCHED', (), 1),), (), 0, 1, 1, 0, 1, 0, 1, 'a1', scope), S)
    return p, j


def expected(p):
    return {(g, k): copy.deepcopy(v) for g in GROUPS for k, v in p.get(g, {}).items()}


def ser(p, j, kind='reader', scope=LIVE, exp=None):
    from finance.serializer import serialize
    return serialize(p, kind, journal=j, coverage_key='k', scope=scope, observed_count=1,
                     expected_metrics=exp if exp is not None else expected(p), reconciliation_key='k')


def practiceify(p):
    from finance.serializer import ECONOMIC_KEYS, CASH_ECONOMIC_KEYS, TRADING_KEYS
    p.update(basis='HYPOTHETICAL_NON_CASH', classification='MIXED')
    for g in GROUPS:
        for k, m in p.get(g, {}).items():
            if k in ECONOMIC_KEYS | CASH_ECONOMIC_KEYS:
                m.update(value=None, state='unavailable', classification='UNKNOWN', reason='Practice is non-cash')
            elif k in TRADING_KEYS:
                m.update(state='hypothetical', classification='HYPOTHETICAL')
    return p


class H5Serializer(unittest.TestCase):
    def test_baselines_serialize(self):
        p, j = build(); ser(p, j)
        p, j = build('weekly'); ser(p, j, 'weekly')

    def test_practice_cannot_carry_economic_conclusion(self):
        p, j = build(scope=PRACTICE); p['basis'] = 'HYPOTHETICAL_NON_CASH'; p['classification'] = 'MIXED'
        with self.assertRaises(ValueError):
            ser(p, j, scope=PRACTICE)

    def test_finding_practice_realised_pnl_cannot_be_labelled_fact(self):
        """Weekly keys realised_pnl / broker_costs / other_trading_costs are outside TRADING_KEYS,
        so a Practice report can publish them as verified FACT."""
        p, j = build('weekly', scope=PRACTICE); practiceify(p)
        self.assertEqual(p['this_week']['realised_pnl']['classification'], 'FACT')
        with self.assertRaises(ValueError, msg='Practice realised_pnl serialized as verified FACT'):
            ser(p, j, 'weekly', PRACTICE)

    def test_finding_verified_fact_requires_a_value(self):
        p, j = build(); p['metrics']['net_trading'].update(value=None)
        with self.assertRaises(ValueError, msg='state=verified/FACT with value=null accepted'):
            ser(p, j)

    def test_unavailable_must_be_null_with_reason(self):
        p, j = build(); p['metrics']['net_trading'].update(state='unavailable', classification='UNKNOWN', reason=None)
        with self.assertRaises(Exception):
            ser(p, j)

    def test_withdrawn_coverage_blocks_fact_and_self_declared_completeness(self):
        p, j = build(); j.append('coverage', 'k', cov(1, 'WITHDRAWN'), '1999-01-01T00:00:00Z')
        with self.assertRaises(ValueError):
            ser(p, j)

    def test_forged_value_rejected(self):
        p, j = build(); exp = expected(p); p['metrics']['net_trading']['value'] = '2.000000000000'
        with self.assertRaises(ValueError):
            ser(p, j, exp=exp)

    def test_rewritten_reconciliation_state_rejected(self):
        p, j = build(); j.append('reconciliation', 'k', Run((Result('t', 'VALUE_MISMATCH', ('x',), 1),), (), 0, 1, 1, 0, 0, 0, 1, 'a1', LIVE), S)
        with self.assertRaises(ValueError):
            ser(p, j)

    def test_precision_beyond_policy_rejected_by_schema(self):
        p, j = build(); p['metrics']['net_trading']['value'] = '1.0000000000001'
        with self.assertRaises(Exception):
            ser(p, j, exp=expected(p))

    def test_missing_jsonschema_fails_closed(self):
        saved = sys.modules.get('jsonschema'); sys.modules['jsonschema'] = None
        try:
            p, j = build()
            with self.assertRaises(RuntimeError):
                ser(p, j)
        finally:
            if saved is None: del sys.modules['jsonschema']
            else: sys.modules['jsonschema'] = saved

    def test_finding_lifetime_metrics_bound_to_lifetime_coverage(self):
        """Weekly lifetime break-even is accepted as FACT while the only coverage checked is the week."""
        week_start = '2026-10-26T00:00:00Z'
        p, j = build('weekly', start=week_start, end=E)
        self.assertEqual(p['lifetime']['remaining_break_even']['classification'], 'FACT')
        with self.assertRaises(ValueError, msg='lifetime FACT accepted on one-week coverage only'):
            ser(p, j, 'weekly')

    def test_residual_provenance_is_caller_trusted(self):
        """Documents trust boundary R-WRITER-02: any in-process caller can build the journal."""
        p, j = build()
        forged = AuthorityJournal(); forged.append('coverage', 'k', cov(1), S)
        forged.append('reconciliation', 'k', j.latest('reconciliation', 'k')[3], S)
        self.assertTrue(ser(p, forged))


# --------------------------------------------------------------------------- M1
class M1MoneyPolicy(unittest.TestCase):
    def test_declared_policy(self):
        self.assertEqual(POLICY_VERSION, 'money_scale12_half_away_v1')

    def test_ties_away_from_zero_and_negative_zero(self):
        self.assertEqual(quantise(Decimal('0.0000000000005')), Decimal('0.000000000001'))
        self.assertEqual(quantise(Decimal('-0.0000000000005')), Decimal('-0.000000000001'))
        self.assertEqual(str(quantise(Decimal('-0.0000000000004'))), '0E-12')

    def test_differs_from_half_even_only_on_exact_ties(self):
        from decimal import ROUND_HALF_EVEN
        q = Decimal('0.000000000001')
        self.assertNotEqual(quantise(Decimal('0.0000000000025')), Decimal('0.0000000000025').quantize(q, ROUND_HALF_EVEN))
        self.assertEqual(quantise(Decimal('0.00000000000251')), Decimal('0.00000000000251').quantize(q, ROUND_HALF_EVEN))

    def test_money_boundary_rejections(self):
        for bad in (1.5, 'NaN', 'Infinity', '1e18', '0.0000000000001', 'x' * 65):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                money(bad)

    def test_fx_single_rounding(self):
        self.assertEqual(convert('123456789.123456789012', '0.765432109876'), Decimal('94497790.577283938656'))


# --------------------------------------------------------------------------- M2 / operational
class M2Reconciliation(unittest.TestCase):
    def run_(self, b, o, t, p, now=E, due=E):
        return reconcile_run(b, o, t, p, as_of=now, outcome_due_at=due, account='a1', scope=LIVE)

    def chain(self, fill='FILLED'):
        return ([row(trade_id='T', transaction_id='x1', action='OPEN'),
                 row(trade_id='T', transaction_id='x2', action='REDUCE', realised_pnl='1.5'),
                 row(trade_id='T', transaction_id='x3', action='CLOSE', realised_pnl='2.5')],
                [row(trade_id='T', event_id='o1', signal_id='s', fill_status=fill)],
                [row(trade_id='T', event_id='c1', realised_pnl='4.0')],
                [row(signal_id='s', event_id='p1')])

    def test_partial_close_rolls_up(self):
        r = self.run_(*self.chain())
        self.assertEqual([x.state for x in r.results], ['MATCHED'])
        self.assertEqual((r.closed_trades, r.partial_close_slices), (1, 1))

    def test_malformed_row_is_quarantined_not_fatal(self):
        b, o, t, p = self.chain()
        b.append(row(trade_id='U', transaction_id='y1', action='CLOSE', realised_pnl=1.25))
        r = self.run_(b, o, t, p)
        self.assertEqual({x.trade_id: x.state for x in r.results}, {'T': 'MATCHED', 'U': 'INVESTIGATION_REQUIRED'})

    def test_pending_then_overdue_exactly_at_deadline(self):
        b, o, _, p = self.chain()
        due = '2026-10-20T23:30:00Z'
        before = self.run_(b, o, [], p, now='2026-10-20T23:29:59.999999Z', due=due)
        at = self.run_(b, o, [], p, now=due, due=due)
        self.assertEqual(before.results[0].state, 'PENDING_EOD_SYNC')
        self.assertEqual(at.results[0].state, 'BROKER_ONLY')

    def test_finding_broker_export_lag_before_eod_is_pending(self):
        """Engine trade_opened has arrived; broker export not yet synced; deadline not reached."""
        _, o, _, p = self.chain()
        r = self.run_([], o, [], p, now=S, due=E)
        self.assertEqual(r.results[0].state, 'PENDING_EOD_SYNC',
                         'pre-EOD broker lag reported as %s %s' % (r.results[0].state, r.results[0].reasons))

    def test_finding_broker_close_lag_before_eod_is_pending(self):
        """Outcome posted by the EOD bridge before the broker CLOSE slice is exported."""
        b, o, t, p = self.chain()
        r = self.run_(b[:1], o, t, p, now=S, due=E)
        self.assertEqual(r.results[0].state, 'PENDING_EOD_SYNC',
                         'pre-EOD close lag reported as %s %s' % (r.results[0].state, r.results[0].reasons))

    def test_cancelled_never_counts_as_closed_trade(self):
        _, o, _, p = self.chain(fill='CANCELLED')
        r = self.run_([], o, [], p)
        self.assertEqual((r.attempted_orders, r.filled_orders, r.cancelled_orders, r.closed_trades), (1, 0, 1, 0))
        b, o, t, p = self.chain(fill='CANCELLED')
        self.assertEqual(self.run_(b, o, t, p).results[0].state, 'INVESTIGATION_REQUIRED')

    def test_http_200_without_fill_evidence_is_not_execution(self):
        b, o, t, p = self.chain(); del o[0]['fill_status']
        r = self.run_(b, o, t, p)
        self.assertEqual(r.results[0].state, 'INVESTIGATION_REQUIRED'); self.assertEqual(r.closed_trades, 0)

    def test_value_mismatch_and_conflicting_replay(self):
        b, o, t, p = self.chain(); t[0]['realised_pnl'] = '4.01'
        self.assertEqual(self.run_(b, o, t, p).results[0].state, 'VALUE_MISMATCH')
        b, o, t, p = self.chain(); b.append(dict(b[1], realised_pnl='9'))
        self.assertEqual(self.run_(b, o, t, p).results[0].state, 'INVESTIGATION_REQUIRED')
        b, o, t, p = self.chain(); b.append(dict(b[1], received_at='2026-10-02T00:00:00Z'))
        r = self.run_(b, o, t, p)
        self.assertEqual((r.results[0].state, r.duplicate_replays), ('MATCHED', 1))


# --------------------------------------------------------------------------- M3
class M3Attribution(unittest.TestCase):
    def tx(self, **kw):
        t = dict(accountID='X', id='1', time=S, type='ORDER_FILL', accountBalance='1000', pl='10',
                 financing='0', commission='0', guaranteedExecutionFee='0', tradesClosed=[{'realizedPL': '10'}])
        t.update(kw); return t

    def test_nested_mismatch_quarantined_adjacent_accepted(self):
        ok, bad = self.tx(), self.tx(id='2', tradesClosed=[{'realizedPL': '9'}])
        accepted, exceptions = normalise_batch([bad, ok], 'a1', 'X', LIVE, 'GBP', '1')
        self.assertEqual(len(accepted), 1)
        self.assertEqual(exceptions[0]['reason'], 'nested_attribution_mismatch')

    def test_finding_routine_non_cash_oanda_types_are_not_investigations(self):
        """A real OANDA stream contains MARKET_ORDER/STOP_LOSS_ORDER/ORDER_CANCEL etc. for every trade."""
        stream = [dict(accountID='X', id='1', time=S, type='MARKET_ORDER', accountBalance='1000'), self.tx(id='2')]
        _, exceptions = normalise_batch(stream, 'a1', 'X', LIVE, 'GBP', '1')
        self.assertEqual(exceptions, (), 'routine non-cash transaction raised %s' % (exceptions,))


# --------------------------------------------------------------------------- M4
class M4Import(unittest.TestCase):
    def csv(self, rows, extra=None):
        names = [f.name for f in fields(core.Expense)]
        out = io.StringIO(); out.write(','.join(names) + '\n')
        for r in rows:
            d = {f: getattr(r, f) for f in names}
            d.update(extra or {})
            vals = []
            for f in names:
                v = d[f]
                v = '' if v is None else ('true' if v is True else 'false' if v is False else
                                          v.isoformat() if isinstance(v, datetime) else str(v))
                vals.append('"%s"' % v.replace('"', '""'))
            out.write(','.join(vals) + '\n')
        out.seek(0); return out

    def test_formula_text_neutralised(self):
        ledger = core.ExpenseLedger()
        core.import_expenses(self.csv([mk_expense()], {'description': '=HYPERLINK("http://x")', 'vendor': ' @SUM(1)'}), ledger)
        r = ledger.history[0]
        self.assertTrue(r.description.startswith("'=")); self.assertTrue(r.vendor.startswith("' @"))

    def test_rejections(self):
        for extra in ({'verification': 'invoice_verified'}, {'verified_by': 'me'}, {'evidence_ref': 'javascript:alert'},
                      {'evidence_ref': 'file:../../etc'}, {'currency': 'XXX', 'fx_to_gbp': '1.1'},
                      {'expense_id': '=cmd|x'}, {'description': 'a\tb'}):
            with self.subTest(extra=extra), self.assertRaises((ValueError, TypeError)):
                core.import_expenses(self.csv([mk_expense()], extra), core.ExpenseLedger())

    def test_size_bounds(self):
        rows = [mk_expense(expense_id='e%d' % i, source_key='invoice:x:%d' % i) for i in range(1001)]
        with self.assertRaises(ValueError):
            core.import_expenses(self.csv(rows), core.ExpenseLedger())
        with self.assertRaises(ValueError):
            core.import_expenses(io.StringIO('a' * (1024 * 1024 + 10)), core.ExpenseLedger())

    def test_atomic_on_late_failure(self):
        ledger = core.ExpenseLedger()
        rows = [mk_expense(), mk_expense(expense_id='e2', source_key='invoice:x:2')]
        text = self.csv(rows).getvalue()
        head, sep, tail = text.rpartition('"GBP"')
        with self.assertRaises(ValueError):  # second (last) row invalid after a valid first row
            core.import_expenses(io.StringIO(head + '"ZZZ"' + tail), ledger)
        self.assertEqual(ledger.history, ())


# --------------------------------------------------------------------------- M6/M7/M8/M9
class MediumOthers(unittest.TestCase):
    def test_M6_practice_unit_economics_hypothetical(self):
        rows = [core.BrokerComponent('p', PRACTICE, 't', 'pl', S, 'realised_pnl', '50', 'GBP', '1')]
        totals = core.broker_totals(rows, 'p', PRACTICE, S, E, cov(1))
        r = core.practice_unit_economics(totals, trade_count=5, denominator_basis='closed_trades')
        self.assertEqual((r['basis'], r['actual_net_profit'], r['hypothetical_net_per_trade']), ('HYPOTHETICAL_NON_CASH', None, 10))
        with self.assertRaises(ValueError):
            core.practice_unit_economics(totals, trade_count=5, denominator_basis='attempted_orders')

    def test_M7_run_rate_unknown_is_null_and_counts_exposed(self):
        subs = [core.Subscription('a', '30', S, E, 'invoice_verified'), core.Subscription('b', '5', S, E, 'estimate'),
                core.Subscription('c', '7', S, E, 'unverified')]
        r = core.run_rate(subs, S, cov(3))
        self.assertIsNone(r['year']); self.assertEqual((r['excluded_estimates'], r['unknown_records']), (1, 1))
        r = core.run_rate([], S, cov(0))
        self.assertEqual((r['state'], r['year']), ('VERIFIED', 0))

    def test_M8_straight_line_is_additive_and_exact(self):
        e = mk_expense(original_amount='1000.000000000001', recognition_policy='straight_line_seconds_v1',
                       billing_start='2026-10-01T00:00:00Z', billing_end='2027-10-01T00:00:00Z')
        ledger = verified_ledger(e)
        cuts = ['2026-09-01T00:00:00Z', '2026-10-25T00:59:59Z', '2026-10-25T01:00:01Z', '2027-02-14T13:13:13Z', '2027-12-01T00:00:00Z']
        parts = [core.expense_totals(ledger, a, b, cov(1, start=a, end=b))['economic_expense'] for a, b in zip(cuts, cuts[1:])]
        self.assertEqual(sum(parts), e.gbp)

    def test_M8_missing_policy_or_payment_is_unavailable(self):
        ledger = verified_ledger(mk_expense(recognition_policy=None, paid_at=None))
        r = core.expense_totals(ledger, S, E, cov(1))
        self.assertIsNone(r['economic_expense']); self.assertIsNone(r['cash_spend'])

    def test_M9_resolution_lifecycle(self):
        j = ReconciliationJournal('a1', LIVE)
        bad = reconcile_run([], [row(trade_id='T', event_id='o', signal_id='s', fill_status='FILLED')], [], [row(signal_id='s', event_id='p')],
                            as_of=E, outcome_due_at=E, account='a1', scope=LIVE)
        good = reconcile_run(*M2Reconciliation().chain(), as_of=E, outcome_due_at=E, account='a1', scope=LIVE)
        j.record('r2', good, S)
        j.record('r1', bad, S)
        with self.assertRaises(ValueError):  # matched evidence accepted BEFORE the exception
            j.resolve('r1', 'T', 'r2', 'rev', 'evidence:x', S)
        j.record('r3', good, S)
        j.resolve('r1', 'T', 'r3', 'rev', 'evidence:x', S)
        self.assertEqual(j.latest_resolution('r1', 'T')[3][0], 'RESOLVED')
        j.withdraw('r1', 'T', 'rev', 'evidence:y', S)
        self.assertEqual(j.latest_resolution('r1', 'T')[3][0], 'WITHDRAWN')


# --------------------------------------------------------------------------- D: operational timing
LONDON = ZoneInfo('Europe/London')


def reference_next_sync(closed_at):
    """Reviewer reference: Closed Trade Data Sync runs 00:30 Europe/London, Tuesday-Saturday."""
    t = instant(closed_at).astimezone(LONDON)
    day = t.date()
    for i in range(0, 9):
        d = day + timedelta(days=i)
        run = datetime(d.year, d.month, d.day, 0, 30, tzinfo=LONDON)
        if run > t and run.isoweekday() in (2, 3, 4, 5, 6):
            return run.astimezone(timezone.utc)
    raise AssertionError('unreachable')


class D_OperationalTiming(unittest.TestCase):
    def test_reference_schedule_across_dst_end_2026(self):
        cases = {  # closure (UTC) -> next sync (UTC); clocks go back 2026-10-25 01:00 UTC
            '2026-10-23T22:00:00Z': '2026-10-23T23:30:00+00:00',   # Fri BST -> Sat 00:30 BST
            '2026-10-23T23:40:00Z': '2026-10-27T00:30:00+00:00',   # missed Sat run -> Tue (GMT)
            '2026-10-26T23:45:00Z': '2026-10-27T00:30:00+00:00',   # Mon GMT -> Tue 00:30 GMT
            '2026-10-19T23:31:00Z': '2026-10-20T23:30:00+00:00',   # Tue 00:31 BST, just missed -> Wed 00:30 BST
            '2026-10-25T21:00:00Z': '2026-10-27T00:30:00+00:00',   # Sun open -> no Monday run -> Tue
        }
        for closed, due in cases.items():
            with self.subTest(closed=closed):
                self.assertEqual(reference_next_sync(closed).isoformat(), due)

    def test_candidate_state_machine_is_correct_when_given_true_due(self):
        b, o, _, p = M2Reconciliation().chain()
        due = reference_next_sync('2026-10-23T23:40:00Z').isoformat()
        r = reconcile_run(b, o, [], p, as_of='2026-10-26T12:00:00Z', outcome_due_at=due, account='a1', scope=LIVE)
        self.assertEqual(r.results[0].state, 'PENDING_EOD_SYNC')

    def test_finding_due_time_is_derived_not_caller_asserted(self):
        """No schedule/calendar model exists: outcome_due_at is free caller input, so a naive
        'next day 00:30 UTC' assumption silently misclassifies BST closures."""
        api = {n for m in (core,) for n in dir(m)} | set(dir(sys.modules['finance.reconciliation']))
        self.assertTrue(any('sync' in n.lower() and 'due' in n.lower() or 'schedule' in n.lower() for n in api),
                        'no EOD schedule/due-time derivation; Europe/London 00:30 Tue-Sat and DST are unmodelled')

    def test_practice_profit_never_reduces_break_even(self):
        exp = {'cash_spend': money('100'), 'economic_expense': money('100'), 'coverage_complete': True}
        r = core.project_economics([acct('p', '100000', PRACTICE)], exp, cov(1), S, E, trade_count=0, denominator_basis='closed_trades')
        self.assertEqual(r['remaining_cash_break_even_gbp'], 100)
        self.assertEqual(r['eligible_accounts'], 0)


if __name__ == '__main__':
    unittest.main()
