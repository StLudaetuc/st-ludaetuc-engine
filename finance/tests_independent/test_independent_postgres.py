"""Independent PostgreSQL 18 tests (cross-provider review) for finance @ 189587ce.

Requires psycopg 3 and FIN_REVIEW_PG_ADMIN_DSN pointing at a DISPOSABLE cluster
(superuser on a throwaway instance; every test creates and drops its own
database). Never point this at any St Ludaetuc database. Skipped otherwise.
test_finding_* tests assert required properties and fail at the reviewed SHA.
"""
import importlib.util
import os
import random
import threading
import unittest
from decimal import Decimal
from pathlib import Path

from finance import core

ROOT = Path(__file__).resolve().parent.parent
ADMIN = os.environ.get('FIN_REVIEW_PG_ADMIN_DSN', '')
ENABLED = bool(ADMIN) and importlib.util.find_spec('psycopg') is not None
MIGRATION = (ROOT / 'migrations' / '001_foundation.sql').read_text()


@unittest.skipUnless(ENABLED, 'FIN_REVIEW_PG_ADMIN_DSN not set: PostgreSQL evidence UNPROVED')
class FinancePostgres(unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
        self.psycopg = psycopg
        self.db = f"fin_review_{random.randrange(10**9)}"
        with psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute(f'CREATE DATABASE {self.db}')
        info = conninfo_to_dict(ADMIN)
        info['dbname'] = self.db
        self.dsn = make_conninfo(**info)
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.conn.execute(MIGRATION)

    def tearDown(self):
        self.conn.close()
        with self.psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS {self.db} WITH (FORCE)')

    def q(self, sql, params=None):
        cur = self.conn.execute(sql, params)
        return cur.fetchall() if cur.description else None

    # ----------------------------------------------------------------------- fixtures
    def expense(self, eid, rev=1, at='2026-10-05T10:00:00Z', amount='10', ccy='GBP', fx='1',
                verification='invoice_verified', recorded_at=None, reason=None):
        if rev == 1:
            self.q('INSERT INTO finance.expense_identity(expense_id, source_key) VALUES (%s, %s)', (eid, 'src:' + eid))
        cols = ('expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,'
                'fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification,verified_by,'
                'correction_reason')
        vals = [eid, rev, at, 'v', 'd', 'infrastructure', amount, ccy, fx, 'fx', at, '2026-01-01Z', '2027-01-01Z', False,
                's', 'PRODUCTION', 'ev', verification,
                'reviewer' if verification in ('invoice_verified', 'manual_verified') else None, reason]
        if recorded_at:
            cols += ',recorded_at'
            vals.append(recorded_at)
        self.q(f'INSERT INTO finance.expense_revision({cols}) VALUES ({",".join(["%s"] * len(vals))})', vals)

    def account(self, key, broker='OANDA_LIVE', system='PRODUCTION', ccy='GBP'):
        self.q('INSERT INTO finance.broker_account VALUES (%s,%s,%s,%s,%s)', (key, broker, system, 'ACTUAL', ccy))

    def covered(self, key, start, end, run='r', assertion='c', recorded_at=None, complete=True, first='1', last='9'):
        self.q('INSERT INTO finance.reconciliation_run VALUES (%s,%s,%s,%s,%s,%s,true,true,%s,%s,%s)',
               (run + key, key, start, end, end, end, first, last, 'ev'))
        cols, vals = ('assertion_id,account_key,period_start,period_end,expenses_complete,broker_complete,'
                      'reconciliation_run_id,reviewed_by,evidence_ref'), [assertion + key, key, start, end, complete, complete,
                                                                          run + key, 'reviewer', 'ev']
        if recorded_at:
            cols += ',recorded_at'
            vals.append(recorded_at)
        self.q(f'INSERT INTO finance.coverage_assertion({cols}) VALUES ({",".join(["%s"] * len(vals))})', vals)

    def summary(self, key, start, end):
        cur = self.conn.execute('SELECT * FROM finance.period_summary(%s,%s,%s)', (key, start, end))
        return dict(zip([d.name for d in cur.description], cur.fetchone()))

    # ----------------------------------------------------------------------- 1 migration
    def test_migration_refuses_rerun_and_grants_nothing_to_public(self):
        with self.assertRaises(self.psycopg.Error) as ctx:
            self.conn.execute(MIGRATION)
        self.assertEqual(ctx.exception.sqlstate, '42P06')  # duplicate_schema: re-application refused
        self.conn.execute('ROLLBACK')  # the script's own BEGIN left the session in an aborted transaction
        self.assertEqual(self.q("SELECT has_schema_privilege('public', 'finance', 'USAGE')")[0][0], False)

    def test_append_only_rejects_update_delete_truncate(self):
        self.expense('e1')
        for stmt in ('UPDATE finance.expense_revision SET original_amount = 1', 'DELETE FROM finance.expense_revision',
                     'TRUNCATE finance.expense_revision', 'UPDATE finance.expense_identity SET source_key = %s'):
            with self.subTest(stmt):
                with self.assertRaises(self.psycopg.errors.RaiseException):
                    self.conn.execute(stmt, ('x',) if '%s' in stmt else None)

    def test_revision_gap_duplicate_source_and_concurrent_corrections(self):
        self.expense('e1')
        with self.assertRaises(self.psycopg.errors.RaiseException):
            self.expense('e1', rev=3, reason='gap')
        with self.assertRaises(self.psycopg.errors.UniqueViolation):
            self.q("INSERT INTO finance.expense_identity VALUES ('e2', 'src:e1', now())")
        barrier, results = threading.Barrier(2), []

        def correct(amount):
            import psycopg
            with psycopg.connect(self.dsn) as c:
                barrier.wait()
                try:
                    with c.transaction():
                        c.execute('INSERT INTO finance.expense_revision(expense_id, revision, incurred_at, vendor, description,'
                                  ' category, original_amount, currency, fx_to_gbp, fx_source, fx_observed_at, billing_start,'
                                  ' billing_end, recurring, service, environment, evidence_ref, verification, verified_by,'
                                  ' correction_reason) SELECT expense_id, 2, incurred_at, vendor, description,'
                                  ' category, %s, currency, fx_to_gbp, fx_source, fx_observed_at, billing_start, billing_end,'
                                  ' recurring, service, environment, evidence_ref, verification, verified_by, %s'
                                  ' FROM finance.expense_revision WHERE expense_id = %s AND revision = 1',
                                  (amount, 'race', 'e1'))
                        c.execute('SELECT pg_sleep(0.2)')
                    results.append('ok')
                except psycopg.Error as exc:
                    results.append(type(exc).__name__)
        threads = [threading.Thread(target=correct, args=(a,)) for a in ('11', '12')]
        [t.start() for t in threads]
        [t.join(20) for t in threads]
        self.assertEqual(sorted(results).count('ok'), 1, results)
        self.assertEqual(self.q("SELECT count(*) FROM finance.expense_revision WHERE expense_id = 'e1'")[0][0], 2)

    # ----------------------------------------------------------------------- 2 parity
    def test_parity_spend_randomised(self):
        rng = random.Random(20261013)
        led = core.ExpenseLedger()
        for i in range(300):
            amount = str(Decimal(rng.randint(-50000, 500000)) / Decimal(10 ** rng.randint(0, 12)))
            ccy, fx = rng.choice([('GBP', '1'), ('USD', '0.7612345678'), ('EUR', '0.857142857142')])
            ver = rng.choice(['invoice_verified', 'manual_verified', 'estimate', 'unverified'])
            day = rng.randint(1, 360)
            at = f'2026-01-01T00:00:00Z'
            import datetime as dt
            at = (dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=day, seconds=rng.randint(0, 86399))).isoformat()
            self.expense(f'e{i}', at=at, amount=amount, ccy=ccy, fx=fx, verification=ver)
            led.append(core.Expense(f'e{i}', 1, 'src:e' + str(i), at, 'v', 'd', 'infrastructure', amount, ccy, fx, 'fx', at,
                                    'ev', ver, 's', 'PRODUCTION', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z',
                                    verified_by='reviewer' if ver.endswith('verified') else ''))
        self.account('live')
        for _ in range(25):
            a, b = sorted(rng.sample(range(0, 370), 2))
            import datetime as dt
            start = (dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=a)).isoformat()
            end = (dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=b)).isoformat()
            with self.subTest(start=start, end=end):
                self.assertEqual(self.summary('live', start, end)['observed_spend_gbp'], led.spend(start, end))
        self.assertEqual(self.q('SELECT observed_verified_gbp FROM finance.lifetime_spend')[0][0],
                         led.spend('2000-01-01T00:00:00Z', '2100-01-01T00:00:00Z'))

    def test_parity_broker_components_randomised(self):
        rng = random.Random(7)
        self.account('live', ccy='USD')
        comps = []
        for i in range(200):
            at = f'2026-10-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00Z'
            self.q("INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,transaction_type,"
                   "financial_state,raw_evidence_ref,raw_sha256,parser_version) VALUES ('live',%s,%s,'X','REALISED','ev',%s,'p')",
                   (str(i), at, 'a' * 64))
            for key, kind in (('pl', 'realised_pnl'), ('fin', 'financing'), ('com', 'commission'), ('cv', 'conversion'),
                              ('adj', 'other_adjustment'), ('tr', 'capital_transfer')):
                if rng.random() < 0.6:
                    amount = str(Decimal(rng.randint(-10 ** 7, 10 ** 7)) / Decimal(10 ** rng.randint(0, 8)))
                    fx = rng.choice(['0.7612345678', '0.75', '1.000000000001'])
                    self.q("INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,"
                           "fx_to_gbp,fx_source,fx_observed_at) VALUES ('live',%s,%s,%s,%s,%s,'fx',%s)",
                           (str(i), key, kind, amount, fx, at))
                    comps.append(core.BrokerComponent('live', 'live', str(i), key, at, kind, amount, 'USD', fx))
        py = core.broker_totals(comps, 'live', 'live', '2026-10-01T00:00:00Z', '2026-11-01T00:00:00Z')
        sql = self.summary('live', '2026-10-01T00:00:00Z', '2026-11-01T00:00:00Z')
        self.assertEqual(sql['gross_realised_gbp'], py['realised_pnl'])
        self.assertEqual(sql['financing_gbp'], py['financing'])
        self.assertEqual(sql['other_costs_gbp'], py['commission'] + py['conversion'] + py['other_adjustment'])
        self.assertEqual(sql['net_trading_gbp'], py['net_trading'])

    def test_parity_run_rate_month_and_year(self):
        self.q("INSERT INTO finance.subscription_schedule VALUES ('s1','sub',NULL,'2026-01-01Z','2027-01-01Z',"
               "'10.123456789012','invoice_verified','ev','reviewer')")
        sql = self.q("SELECT month_gbp, year_gbp FROM finance.run_rate('2026-06-01Z')")[0]
        py = core.run_rate([core.Subscription('sub', '10.123456789012', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z',
                                              'invoice_verified')], '2026-06-01T00:00:00Z')
        self.assertEqual(sql, (py['month'], py['year']))

    def test_finding_parity_run_rate_day(self):
        self.q("INSERT INTO finance.subscription_schedule VALUES ('s1','sub',NULL,'2026-01-01Z','2027-01-01Z',10,"
               "'invoice_verified','ev','reviewer')")
        sql = self.q("SELECT day_gbp FROM finance.run_rate('2026-06-01Z')")[0][0]
        py = core.run_rate([core.Subscription('sub', '10', '2026-01-01T00:00:00Z', '2027-01-01T00:00:00Z',
                                              'invoice_verified')], '2026-06-01T00:00:00Z')['day']
        self.assertEqual(sql, py, 'SQL numeric division (20 dp) and Python Decimal (80 digits) disagree; no rounding policy')

    # ----------------------------------------------------------------------- 6/8/9 coverage, practice, break-even
    def test_practice_excluded_even_when_labelled_production(self):
        self.account('practice', broker='OANDA_PRACTICE', system='PRODUCTION')
        self.covered('practice', '2026-10-01Z', '2026-11-01Z')
        s = self.summary('practice', '2026-10-01Z', '2026-11-01Z')
        self.assertTrue(s['coverage_complete'])
        self.assertIsNone(s['net_economic_gbp'])
        self.assertIsNone(s['remaining_break_even_gbp'])

    def test_estimate_unverified_and_open_exception_block_conclusion(self):
        self.account('live')
        self.covered('live', '2026-10-01Z', '2026-11-01Z')
        self.assertTrue(self.summary('live', '2026-10-01Z', '2026-11-01Z')['coverage_complete'])
        self.expense('est', verification='estimate')
        s = self.summary('live', '2026-10-01Z', '2026-11-01Z')
        self.assertEqual((s['coverage_complete'], s['net_economic_gbp']), (False, None))

    def test_unresolved_reconciliation_exception_blocks_conclusion(self):
        self.account('live')
        self.covered('live', '2026-10-01Z', '2026-11-01Z')
        self.q("INSERT INTO finance.reconciliation_exception VALUES ('rlive','x1','t1','pnl_mismatch','ev')")
        self.assertIsNone(self.summary('live', '2026-10-01Z', '2026-11-01Z')['net_economic_gbp'])

    def test_capital_transfers_never_profit(self):
        self.account('live')
        self.q("INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,transaction_type,financial_state,"
               "raw_evidence_ref,raw_sha256,parser_version) VALUES ('live','1','2026-10-02Z','TRANSFER_FUNDS','REALISED','ev',%s,'p')",
               ('b' * 64,))
        self.q("INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,"
               "fx_observed_at) VALUES ('live','1','amount','capital_transfer',5000,1,'id','2026-10-02Z')")
        self.covered('live', '2026-10-01Z', '2026-11-01Z')
        s = self.summary('live', '2026-10-01Z', '2026-11-01Z')
        self.assertEqual((s['net_trading_gbp'], s['net_economic_gbp']), (Decimal(0), Decimal(0)))

    def test_finding_empty_evidence_cannot_show_break_even(self):
        self.account('live')
        self.covered('live', '2026-10-01Z', '2026-11-01Z', first=None, last=None)
        s = self.summary('live', '2026-10-01Z', '2026-11-01Z')
        self.assertIsNone(s['remaining_break_even_gbp'],
                          f'no expenses, no broker rows, no transaction range => conclusion {s}')

    def test_finding_future_dated_assertion_cannot_pin_latest(self):
        self.account('live')
        self.covered('live', '2026-10-01Z', '2026-11-01Z', assertion='old', recorded_at='2099-01-01T00:00:00Z')
        self.q("INSERT INTO finance.coverage_assertion(assertion_id,account_key,period_start,period_end,expenses_complete,"
               "broker_complete,reconciliation_run_id,reviewed_by,evidence_ref) VALUES ('withdrawn','live','2026-10-01Z',"
               "'2026-11-01Z',false,false,'rlive','reviewer','ev')")
        self.assertFalse(self.summary('live', '2026-10-01Z', '2026-11-01Z')['coverage_complete'],
                         'a later withdrawal is ignored because a caller-supplied recorded_at of 2099 stays "latest"')

    def test_finding_recorded_at_is_server_assigned(self):
        self.expense('e1', recorded_at='2001-01-01T00:00:00Z')
        self.assertGreater(self.q("SELECT recorded_at FROM finance.expense_revision")[0][0].year, 2025,
                           'evidence recording time is caller-controlled (backdating)')

    def test_finding_project_spend_subtracted_once_across_accounts(self):
        self.expense('e1', amount='100')
        for key in ('live-a', 'live-b'):
            self.account(key)
            self.covered(key, '2026-10-01Z', '2026-11-01Z')
        per_account = [self.summary(k, '2026-10-01Z', '2026-11-01Z')['net_economic_gbp'] for k in ('live-a', 'live-b')]
        has_project_fn = self.q("SELECT count(*) FROM pg_proc WHERE pronamespace = 'finance'::regnamespace "
                                "AND proname LIKE 'project%%'")[0][0]
        self.assertTrue(has_project_fn and sum(per_account) == Decimal(-100),
                        f'each account subtracts the full project spend {per_account}; no project-level function')

    def test_finding_run_rate_exposes_uncertain_schedules(self):
        self.q("INSERT INTO finance.subscription_schedule VALUES ('s1','api',NULL,'2026-01-01Z','2027-01-01Z',100,"
               "'estimate','ev',NULL)")
        cols = [d.name for d in self.conn.execute("SELECT * FROM finance.run_rate('2026-06-01Z')").description]
        self.assertTrue(any('uncertain' in c for c in cols), f'run_rate {cols} reports 0 with an estimate excluded')

    def test_finding_views_are_security_invoker(self):
        rows = self.q("SELECT c.relname FROM pg_class c WHERE c.relnamespace = 'finance'::regnamespace AND c.relkind = 'v' "
                      "AND NOT coalesce('security_invoker=true' = ANY(c.reloptions), false)")
        self.assertEqual(rows, [], 'views execute with owner rights; a future Reader grant on a view bypasses table ACLs')


if __name__ == '__main__':
    unittest.main()
