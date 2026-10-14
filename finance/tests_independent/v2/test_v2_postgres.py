"""Independent PostgreSQL 18 runtime tests for Finance candidate 4da760ba.

Requires psycopg 3 and FIN_REVIEW_PG_ADMIN_DSN pointing at a DISPOSABLE, LOCAL cluster
(socket-only recommended). Every test creates and drops its own database named
finance_rev_<random>_isolated. Never point this at a shared or Render-hosted database.

``test_finding_*`` tests FAIL BY DESIGN while the finding is open.
"""
import os
import random
import re
import subprocess
import sys
import threading
import unittest
import uuid
from decimal import Decimal, getcontext
from pathlib import Path

ADMIN = os.environ.get('FIN_REVIEW_PG_ADMIN_DSN', '')
ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (ROOT / 'migrations' / '001_foundation.sql').read_text()
ACCEPTANCE = (ROOT / 'tests' / 'postgres_acceptance.sql').read_text()
S, E = '2026-10-01T00:00:00Z', '2026-11-01T00:00:00Z'
NOV_E = '2026-12-01T00:00:00Z'
SHA = 'a' * 64


@unittest.skipUnless(ADMIN, 'FIN_REVIEW_PG_ADMIN_DSN not set: PostgreSQL runtime evidence UNPROVED')
class PG(unittest.TestCase):
    migrate = True

    @classmethod
    def setUpClass(cls):
        import psycopg
        cls.psycopg = psycopg
        with psycopg.connect(ADMIN, autocommit=True) as c:
            v = c.execute('SHOW server_version_num').fetchone()[0]
            assert 180000 <= int(v) < 190000, 'PostgreSQL 18 required, got %s' % v
            assert c.execute("SELECT current_setting('listen_addresses')").fetchone()[0] in ('', 'localhost', '127.0.0.1'), \
                'refusing: cluster listens on a network address'

    def setUp(self):
        self.db = 'finance_rev_%s_isolated' % uuid.uuid4().hex[:10]
        with self.psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute('CREATE DATABASE %s' % self.db)
        self.dsn = re.sub(r'dbname=\S+', 'dbname=' + self.db, ADMIN) if 'dbname=' in ADMIN else ADMIN + ' dbname=' + self.db
        self.c = self.psycopg.connect(self.dsn, autocommit=True)
        if self.migrate:
            self.c.execute(MIGRATION)

    def tearDown(self):
        self.c.close()
        with self.psycopg.connect(ADMIN, autocommit=True) as c:
            c.execute('DROP DATABASE %s WITH (FORCE)' % self.db)

    # ---------------------------------------------------------------- helpers
    def q(self, sql, *args):
        cur = self.c.execute(sql, args or None)
        return cur.fetchall() if cur.description else None

    def one(self, sql, *args):
        return self.q(sql, *args)[0][0]

    def account(self, key, env='OANDA_LIVE', sysenv='PRODUCTION'):
        self.q("INSERT INTO finance.broker_account(account_key,broker_environment,system_environment,financial_reality,"
               "financial_state,simulation_type,currency) VALUES(%s,%s,%s,'ACTUAL','REALISED',NULL,'GBP')", key, env, sysenv)

    def fill(self, acct, tx, at, pl, financing='0', commission='0', fee='0', ttype='ORDER_FILL'):
        self.q("INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,source_received_at,transaction_type,"
               "raw_evidence_ref,raw_sha256,parser_version) VALUES(%s,%s,%s,%s,%s,'evidence:tx',%s,'review')", acct, tx, at, at, ttype, SHA)
        if ttype != 'ORDER_FILL':
            return
        for key, kind, amount in (('pl', 'realised_pnl', pl), ('financing', 'financing', financing),
                                  ('commission', 'commission', commission), ('guaranteedExecutionFee', 'other_adjustment', fee)):
            self.q("INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,"
                   "fx_observed_at) VALUES(%s,%s,%s,%s,%s,1,'identity',%s)", acct, tx, key, kind, Decimal(amount), at)

    def event(self, acct, ident, at, etype='trade_opened', fill='FILLED', sha=SHA):
        self.q("INSERT INTO finance.engine_event_evidence(account_key,source_event_id,source_event_type,event_time,source_received_at,"
               "fill_status,source_evidence_ref,source_sha256) VALUES(%s,%s,%s,%s,%s,%s,'evidence:ev',%s)", acct, ident, etype, at, at, fill, sha)

    def recrun(self, run_id, acct, n_b, n_i, start=S, end=E, results=(('r1', 't1', 'MATCHED'),)):
        self.q("INSERT INTO finance.reconciliation_run(run_id,account_key,period_start,period_end,source_as_of,outcome_due_at,"
               "broker_evidence_count,internal_evidence_count,broker_complete,internal_complete,broker_reviewed_empty,"
               "internal_reviewed_empty,evidence_ref) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,true,true,%s,%s,'evidence:run')",
               run_id, acct, start, end, end, end, n_b, n_i, n_b == 0, n_i == 0)
        for rid, trade, state in results:
            self.q("INSERT INTO finance.reconciliation_result(run_id,result_id,trade_id,state,reason,evidence_ref) "
                   "VALUES(%s,%s,%s,%s,'review','evidence:res')", run_id, rid, trade, state)

    def coverage(self, aid, domain, scope, count, start=S, end=E, state=None, run_id=None, conn=None):
        state = state or ('VERIFIED' if count else 'REVIEWED_EMPTY')
        first = 'f' if state == 'VERIFIED' else None
        (conn or self.c).execute(
            "INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,"
            "first_evidence_id,last_evidence_id,reconciliation_run_id,currency_confirmed,costs_complete,reviewed_by,evidence_ref) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,true,true,'reviewer','manifest:cov') RETURNING server_order",
            (aid, domain, scope, start, end, state, count, first, first, run_id)).fetchone()

    def expense(self, eid, amount='10', paid=S, policy='point_in_time_v1', bs=S, be=E, incurred=S, fx='1', ccy='GBP'):
        self.q("INSERT INTO finance.expense_identity(expense_id,source_key) VALUES(%s,%s)", eid, 'invoice:' + eid)
        self.q("INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,paid_at,vendor,description,category,"
               "original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,recognition_policy,"
               "service,environment,evidence_ref,verification) VALUES(%s,1,%s,%s,'v','d','infrastructure',%s,%s,%s,'fx',%s,%s,%s,"
               "false,%s,'svc','PRODUCTION','invoice:x','unverified')", eid, incurred, paid, Decimal(amount), ccy, Decimal(fx), S, bs, be, policy)

    def verify(self, decision, eid, state='invoice_verified', conn=None):
        return (conn or self.c).execute(
            "INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref) "
            "VALUES(%s,%s,1,%s,'reviewer','evidence:v') RETURNING server_order", (decision, eid, state)).fetchone()[0]

    def live_month(self, acct='a', start=S, end=E, tx='t1', pl='100', at=None, run_id=None):
        at = at or start
        self.fill(acct, tx, at, pl)
        self.event(acct, 'ev-' + tx, at)
        rid = run_id or 'run-' + acct + '-' + tx
        self.recrun(rid, acct, 1, 1, start, end, (('m-' + tx, tx, 'MATCHED'),))
        return rid


# ======================================================================== migration
class Migration(PG):
    migrate = False

    def test_clean_apply_then_reapply_refused_without_change(self):
        self.c.execute(MIGRATION)
        before = self.one("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE nspname='finance'")
        with self.assertRaises(self.psycopg.errors.DuplicateSchema):
            self.c.execute(MIGRATION)
        self.c.execute('ROLLBACK') if self.c.info.transaction_status else None
        self.assertEqual(before, self.one("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE nspname='finance'"))

    def test_failed_migration_leaves_no_partial_schema(self):
        broken = MIGRATION.replace('REVOKE ALL ON ALL TABLES', 'SELECT 1/0;\nREVOKE ALL ON ALL TABLES')
        with self.assertRaises(self.psycopg.errors.DivisionByZero):
            self.c.execute(broken)
        self.c.execute('ROLLBACK')
        self.assertIsNone(self.one("SELECT to_regnamespace('finance')"))

    def test_finding_author_acceptance_script_passes_on_pg18(self):
        self.c.execute(MIGRATION)
        try:
            self.c.execute(ACCEPTANCE)
        except self.psycopg.Error as exc:
            self.fail('candidate postgres_acceptance.sql fails on PostgreSQL 18: %s' % str(exc).splitlines()[0])

    def test_acceptance_remainder_passes_with_reviewer_truncate_fix(self):
        """Only change: the TRUNCATE probe also accepts the FK-check error PG raises before the trigger."""
        fixed = ACCEPTANCE.replace("BEGIN TRUNCATE finance.expense_revision; RAISE EXCEPTION 'truncate allowed'; EXCEPTION WHEN raise_exception THEN",
                                   "BEGIN TRUNCATE finance.expense_revision; RAISE EXCEPTION 'truncate allowed'; EXCEPTION WHEN raise_exception OR feature_not_supported THEN")
        self.assertNotEqual(fixed, ACCEPTANCE)
        self.c.execute(MIGRATION)
        self.c.execute(fixed)


# ======================================================================== immutability / triggers
class Immutability(PG):
    TABLES = ['expense_identity', 'expense_revision', 'expense_verification', 'subscription_schedule', 'broker_account',
              'broker_transaction', 'broker_component', 'broker_trade_action', 'broker_snapshot', 'engine_event_evidence',
              'reconciliation_run', 'reconciliation_result', 'exception_resolution', 'coverage_assertion', 'report_snapshot']

    def test_update_delete_truncate_and_cascade_blocked(self):
        self.account('a'); self.live_month(); self.expense('e1'); self.verify('v1', 'e1')
        self.coverage('c1', 'EXPENSES', 'PROJECT', 1)
        self.q("INSERT INTO finance.subscription_schedule(schedule_id,subscription_id,effective_from,effective_to,monthly_gbp,verification,evidence_ref) VALUES('s','x',%s,%s,1,'unverified','evidence:s')", S, E)
        self.q("INSERT INTO finance.broker_trade_action(account_key,transaction_id,trade_id,action,units,price,reason,evidence_ref) VALUES('a','t1','T','OPEN',1,1,'r','evidence:a')")
        self.q("INSERT INTO finance.broker_snapshot(snapshot_id,account_key,as_of,balance,nav,evidence_ref,completeness) VALUES('n','a',%s,1,1,'evidence:n','complete')", S)
        self.q("INSERT INTO finance.report_snapshot(report_id,formula_version,source_manifest_id,bounded_output) VALUES('r','f','m','{}')")
        # exception_resolution cannot hold any row on this SHA (see test_finding_sql_exception_resolution_insertable);
        # its row-level guards are therefore checked by TRUNCATE only.
        populated = [t for t in self.TABLES if self.one('SELECT count(*) FROM finance.%s' % t)]
        self.assertEqual(sorted(set(self.TABLES) - set(populated)), ['exception_resolution'])
        for t in self.TABLES:
            stmts = ['TRUNCATE finance.%s CASCADE' % t]
            if t in populated:
                stmts += ['UPDATE finance.%s SET server_order=server_order' % t, 'DELETE FROM finance.%s' % t]
            for stmt in stmts:
                with self.subTest(stmt=stmt), self.assertRaises(self.psycopg.Error):
                    self.c.execute(stmt)
        self.assertEqual(self.one('SELECT count(*) FROM finance.expense_revision'), 1)

    def test_server_order_and_clock_forced(self):
        self.account('a')
        row = self.q("INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,"
                     "currency_confirmed,costs_complete,reviewed_by,evidence_ref,server_order,server_received_at) VALUES('x','EXPENSES',"
                     "'PROJECT',%s,%s,'UNKNOWN',0,false,false,'r','manifest:x',999999999,'2199-01-01Z') RETURNING server_order,"
                     "server_received_at<now()+interval '1 minute'", S, E)[0]
        self.assertNotEqual(row[0], 999999999); self.assertTrue(row[1])

    def test_constraint_and_profile_rejections(self):
        self.account('a'); self.fill('a', 't', S, '1')
        bad = [
            ("INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,source_received_at,transaction_type,raw_evidence_ref,raw_sha256,parser_version) VALUES('a','t2',%s,%s,'ORDER_FILL','e','zz','p')", (S, S)),
            ("INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,currency_confirmed,costs_complete,reviewed_by,evidence_ref) VALUES('v0','EXPENSES','PROJECT',%s,%s,'VERIFIED',0,true,true,'r','manifest:x')", (S, E)),
            ("INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,currency_confirmed,costs_complete,reviewed_by,evidence_ref) VALUES('v1','EXPENSES','PROJECT',%s,%s,'UNKNOWN',0,true,true,'r','javascript:alert')", (S, E)),
        ]
        for sql, args in bad:
            with self.subTest(sql=sql[:60]), self.assertRaises(self.psycopg.Error):
                self.c.execute(sql, args)
        self.q("INSERT INTO finance.broker_transaction(account_key,transaction_id,event_time,source_received_at,transaction_type,raw_evidence_ref,raw_sha256,parser_version) VALUES('a','t3',%s,%s,'ORDER_FILL','evidence:x',%s,'p')", S, S, SHA)
        for key, kind, amount in (('commission', 'commission', 2), ('pl', 'commission', -1), ('halfSpreadCost', 'other_adjustment', -1)):
            with self.subTest(component=key), self.assertRaises(self.psycopg.Error):
                self.c.execute("INSERT INTO finance.broker_component(account_key,transaction_id,component_key,kind,amount,fx_to_gbp,fx_source,fx_observed_at) VALUES('a','t3',%s,%s,%s,1,'id',%s)", (key, kind, amount, S))
        with self.assertRaises(self.psycopg.Error):  # importer cannot self-verify
            self.c.execute("INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('z','invoice:z')")
            self.c.execute("INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification) VALUES('z',1,%s,'v','d','other',1,'GBP',1,'i',%s,%s,%s,false,'s','e','invoice:z','invoice_verified')", (S, S, S, E))

    def test_unclassified_transaction_blocks_account_completeness(self):
        self.account('a'); rid = self.live_month()
        self.fill('a', 'mo', S, '0', ttype='MARKET_ORDER')
        self.coverage('cb', 'BROKER', 'a', 2, run_id=rid)
        self.assertFalse(self.q('SELECT coverage_complete FROM finance.account_totals(%s,%s,%s)', 'a', S, E)[0][0])

    def test_finding_conflicting_engine_event_duplicates_rejected(self):
        self.account('a'); self.event('a', 'ev1', S)
        with self.assertRaises(self.psycopg.Error, msg='same source_event_id with different content accepted as a second row'):
            self.event('a', 'ev1', S, fill='CANCELLED', sha='b' * 64)


# ======================================================================== H1: authority ordering under concurrency
class H1Concurrency(PG):
    def test_serial_future_source_cannot_beat_withdrawal(self):
        self.expense('e1')
        self.c.execute("INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref,source_decided_at) VALUES('v1','e1',1,'invoice_verified','r','evidence:v','2199-01-01Z')")
        self.verify('w1', 'e1', 'withdrawn')
        self.assertEqual(self.one('SELECT count(*) FROM finance.verified_expenses'), 0)

    def _race(self, first_sql, first_args, second_sql, second_args):
        """T1 obtains its server_order first but commits LAST; T2 commits in between."""
        t1 = self.psycopg.connect(self.dsn); t2 = self.psycopg.connect(self.dsn, autocommit=True)
        try:
            o1 = t1.execute(first_sql, first_args).fetchone()[0]       # T1 open, order N
            o2 = t2.execute(second_sql, second_args).fetchone()[0]     # T2 order N+1, committed
            t1.commit()                                                # T1 commits last
            return o1, o2
        finally:
            t1.close(); t2.close()

    def test_finding_withdrawal_committed_last_is_authoritative_verification(self):
        self.expense('e1')
        o_w, o_v = self._race(
            "INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref) VALUES('w','e1',1,'withdrawn','r','evidence:w') RETURNING server_order", None,
            "INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref) VALUES('v','e1',1,'invoice_verified','r','evidence:v') RETURNING server_order", None)
        state = self.q("SELECT decision FROM finance.latest_verifications WHERE expense_id='e1'")[0][0]
        self.assertLess(o_w, o_v)
        self.assertEqual(state, 'withdrawn', 'withdrawal (order %s) committed AFTER verification (order %s) but '
                         'latest_verifications returns %r; verified_expenses=%s' % (o_w, o_v, state,
                         self.one('SELECT count(*) FROM finance.verified_expenses')))

    def test_finding_withdrawal_committed_last_is_authoritative_coverage(self):
        self.expense('e1'); self.verify('v', 'e1')
        sql = ("INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,"
               "first_evidence_id,last_evidence_id,currency_confirmed,costs_complete,reviewed_by,evidence_ref) VALUES(%s,'EXPENSES','PROJECT',"
               "%s,%s,%s,1,%s,%s,true,true,'r','manifest:c') RETURNING server_order")
        self._race(sql, ('withdraw', S, E, 'WITHDRAWN', None, None), sql, ('verify', S, E, 'VERIFIED', 'e1', 'e1'))
        cash = self.one('SELECT cash_spend_gbp FROM finance.expense_totals(%s,%s)', S, E)
        self.assertIsNone(cash, 'expense coverage withdrawn by the last-committed decision, yet cash_spend=%s' % cash)

    def test_order_is_assigned_before_revision_lock(self):
        """Documents: a_server_order fires before expense_sequence's row lock (trigger name order)."""
        names = [r[0] for r in self.q("SELECT tgname FROM pg_trigger WHERE tgrelid='finance.expense_revision'::regclass AND tgname IN ('a_server_order','expense_sequence') ORDER BY tgname")]
        self.assertEqual(names, ['a_server_order', 'expense_sequence'])


# ======================================================================== H2/H3/period/coverage semantics
class Economics(PG):
    def _project_fixture(self):
        self.account('a'); self.account('b'); self.account('p', env='OANDA_PRACTICE')
        ra = self.live_month('a', tx='ta', pl='95'); rb = self.live_month('b', tx='tb', pl='45'); rp = self.live_month('p', tx='tp', pl='9999')
        for acct, rid in (('a', ra), ('b', rb), ('p', rp)):
            self.coverage('cb-' + acct, 'BROKER', acct, 1, run_id=rid)
        self.expense('e1', '12'); self.verify('v1', 'e1')
        self.coverage('ce', 'EXPENSES', 'PROJECT', 1)
        self.coverage('ci', 'PROJECT_INVENTORY', 'PROJECT', 3)

    def test_H3_spend_once_practice_excluded_and_python_parity(self):
        self._project_fixture()
        sql = self.q('SELECT actual_net_trading_gbp,cash_net_economic_gbp,remaining_cash_break_even_gbp FROM finance.project_totals(%s,%s)', S, E)[0]
        self.assertEqual(sql, (Decimal('140'), Decimal('128'), Decimal('0')))
        import finance.core as core
        from finance.policy import Scope, Coverage
        live = Scope('OANDA_LIVE', 'PRODUCTION', 'ACTUAL', 'REALISED', None)
        practice = Scope('OANDA_PRACTICE', 'PRODUCTION', 'ACTUAL', 'REALISED', None)
        cov = lambda n: Coverage('VERIFIED', n, S, E, 'f', 'f', 'manifest:c', 'r', 0, True, True)
        accounts = [{'account': k, 'scope': s, 'coverage_complete': True, 'net_trading': Decimal(v)} for k, s, v in
                    (('a', live, '95'), ('b', live, '45'), ('p', practice, '9999'))]
        py = core.project_economics(accounts, {'cash_spend': Decimal('12'), 'economic_expense': Decimal('12'), 'coverage_complete': True},
                                    cov(3), S, E, trade_count=2, denominator_basis='closed_trades')
        self.assertEqual((py['actual_net_trading_gbp'], py['cash_net_economic_gbp'], py['remaining_cash_break_even_gbp']), sql)

    def test_H2_no_or_unknown_coverage_is_null_reviewed_empty_is_zero(self):
        self.assertIsNone(self.one('SELECT net_economic_gbp FROM finance.project_totals(%s,%s)', S, E))
        self.coverage('u', 'EXPENSES', 'PROJECT', 0, state='UNKNOWN'); self.coverage('i', 'PROJECT_INVENTORY', 'PROJECT', 0, state='UNKNOWN')
        self.assertIsNone(self.one('SELECT cash_net_economic_gbp FROM finance.project_totals(%s,%s)', S, E))
        self.coverage('u2', 'EXPENSES', 'PROJECT', 0); self.coverage('i2', 'PROJECT_INVENTORY', 'PROJECT', 0)
        self.assertEqual(self.one('SELECT cash_net_economic_gbp FROM finance.project_totals(%s,%s)', S, E), 0)

    def test_practice_labelled_production_never_reaches_break_even(self):
        self.account('p', env='OANDA_PRACTICE'); rid = self.live_month('p', pl='1000000')
        self.coverage('cb', 'BROKER', 'p', 1, run_id=rid); self.expense('e1', '10'); self.verify('v', 'e1')
        self.coverage('ce', 'EXPENSES', 'PROJECT', 1); self.coverage('ci', 'PROJECT_INVENTORY', 'PROJECT', 1)
        row = self.q('SELECT actual_net_trading_gbp,remaining_cash_break_even_gbp FROM finance.project_totals(%s,%s)', S, E)[0]
        self.assertEqual(row, (Decimal('0'), Decimal('10')))

    def test_unresolved_exception_blocks_and_estimate_blocks(self):
        self._project_fixture()
        self.recrun('run-a2', 'a', 1, 1, results=(('m', 'ta', 'VALUE_MISMATCH'),))
        self.coverage('cb-a2', 'BROKER', 'a', 1, run_id='run-a2')
        self.assertIsNone(self.one('SELECT net_economic_gbp FROM finance.project_totals(%s,%s)', S, E))

    def test_finding_coverage_of_one_period_survives_assertion_of_the_next(self):
        self.account('a'); r1 = self.live_month('a', tx='oct', at='2026-10-05T00:00:00Z')
        self.coverage('c-oct', 'BROKER', 'a', 1, run_id=r1)
        self.assertTrue(self.one('SELECT coverage_complete FROM finance.account_totals(%s,%s,%s)', 'a', S, E))
        r2 = self.live_month('a', tx='nov', at='2026-11-05T00:00:00Z', start=E, end=NOV_E)
        self.coverage('c-nov', 'BROKER', 'a', 1, start=E, end=NOV_E, run_id=r2)
        self.assertTrue(self.one('SELECT coverage_complete FROM finance.account_totals(%s,%s,%s)', 'a', E, NOV_E))
        self.assertTrue(self.one('SELECT coverage_complete FROM finance.account_totals(%s,%s,%s)', 'a', S, E),
                        'October lost completeness when November was asserted (latest_coverage/latest_runs key ignores period)')

    def test_finding_reconciliation_results_must_cover_population(self):
        self.account('a')
        for i in range(3):
            self.fill('a', 't%d' % i, S, '10'); self.event('a', 'ev%d' % i, S)
        self.recrun('r', 'a', 3, 3, results=(('only', 't0', 'MATCHED'),))
        self.coverage('cb', 'BROKER', 'a', 3, run_id='r')
        self.assertFalse(self.one('SELECT coverage_complete FROM finance.account_totals(%s,%s,%s)', 'a', S, E),
                         '3 broker transactions, 1 MATCHED result: reconciliation accepted as complete')

    def test_finding_sql_exception_resolution_insertable(self):
        """resolution_evidence() aliases a table as `old`, which collides with PL/pgSQL OLD: every insert fails."""
        self.account('a')
        self.recrun('old', 'a', 1, 1, results=(('x', 'T', 'VALUE_MISMATCH'),))
        self.recrun('new', 'a', 1, 1, results=(('m', 'T', 'MATCHED'),))
        for sql in ("INSERT INTO finance.exception_resolution(decision_id,run_id,result_id,state,matched_run_id,matched_result_id,reviewed_by,evidence_ref) VALUES('d1','old','x','RESOLVED','new','m','r','evidence:r')",
                    "INSERT INTO finance.exception_resolution(decision_id,run_id,result_id,state,reviewed_by,evidence_ref) VALUES('d2','old','x','WITHDRAWN','r','evidence:w')"):
            with self.subTest(sql=sql[60:120]):
                try:
                    self.c.execute(sql)
                except self.psycopg.Error as exc:
                    self.fail('valid resolution decision rejected: %s' % str(exc).splitlines()[0])

    def test_cancelled_orders_are_not_filled_denominators(self):
        self.account('a'); self.event('a', 'o1', S, fill='FILLED'); self.event('a', 'o2', S, fill='CANCELLED')
        self.assertEqual(self.q('SELECT attempted_orders,filled_orders,cancelled_orders FROM finance.order_denominators')[0], (2, 1, 1))

    def test_weekly_views_use_utc_weeks(self):
        """Policy observation: a Monday 00:15 BST payment is attributed to the previous UTC week."""
        self.expense('e1', paid='2026-10-18T23:15:00Z'); self.verify('v', 'e1')
        self.assertEqual(str(self.one('SELECT period FROM finance.weekly_cash_spend')), '2026-10-12 00:00:00')


# ======================================================================== privileges
class Privileges(PG):
    def setUp(self):
        self.roles = []
        super().setUp()

    def tearDown(self):
        self.c.execute('RESET ROLE')
        for r in self.roles:
            self.c.execute('DROP OWNED BY %s' % r); self.c.execute('DROP ROLE %s' % r)
        super().tearDown()

    def test_public_has_no_schema_table_or_function_access(self):
        self.assertFalse(self.one("SELECT has_schema_privilege('public','finance','USAGE')"))
        self.assertFalse(self.one("SELECT has_function_privilege('public','finance.project_totals(timestamptz,timestamptz)','EXECUTE')"))
        role = 'rev_nobody_' + uuid.uuid4().hex[:8]; self.roles.append(role)
        self.c.execute('CREATE ROLE %s NOLOGIN' % role)
        self.c.execute('SET ROLE %s' % role)
        with self.assertRaises(self.psycopg.errors.InsufficientPrivilege):
            self.c.execute('SELECT * FROM finance.coverage_assertion')
        self.c.execute('RESET ROLE')

    def test_narrow_writer_cannot_verify_and_needs_undocumented_grants(self):
        role = 'rev_writer_' + uuid.uuid4().hex[:8]; self.roles.append(role)
        self.c.execute('CREATE ROLE %s NOLOGIN' % role)
        self.c.execute('GRANT USAGE ON SCHEMA finance TO %s; GRANT INSERT ON finance.expense_identity, finance.expense_revision TO %s' % (role, role))
        self.c.execute('SET ROLE %s' % role)
        with self.assertRaises(self.psycopg.errors.InsufficientPrivilege):   # sequence/identity lock grants not designed
            self.c.execute("INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('w','invoice:w')")
        with self.assertRaises(self.psycopg.errors.InsufficientPrivilege):
            self.c.execute("INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref) VALUES('x','w',1,'invoice_verified','r','evidence:x')")
        self.c.execute('RESET ROLE')

    def test_finding_views_are_security_invoker(self):
        bad = [r[0] for r in self.q("SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE nspname='finance' "
                                    "AND relkind='v' AND NOT coalesce('security_invoker=true'=ANY(reloptions),false)")]
        self.assertEqual(bad, [], 'views without security_invoker: %s' % bad)

    def test_owner_residuals(self):
        """Documents owner power (L3): trigger disable and sequence rewind (duplicate server_order)."""
        self.c.execute("SELECT setval('finance.authority_order', 1)")
        self.coverage('x1', 'EXPENSES', 'PROJECT', 0, state='UNKNOWN')
        self.c.execute("SELECT setval('finance.authority_order', 1)")
        self.coverage('x2', 'EXPENSES', 'PROJECT', 0, state='UNKNOWN')
        self.assertEqual(self.one('SELECT count(DISTINCT server_order) FROM finance.coverage_assertion'), 1)


# ======================================================================== M1/M8 runtime SQL-Python parity
class Parity(PG):
    def test_q_matches_quantise_including_ties(self):
        from finance.policy import quantise
        rnd = random.Random(20261014); vals = []
        for _ in range(3000):
            digits = ''.join(rnd.choice('0123456789') for _ in range(rnd.randint(1, 30)))
            scale = rnd.randint(0, 24)
            v = Decimal(digits).scaleb(-scale) * rnd.choice((1, -1))
            vals.append(v)
        vals += [Decimal(s) for s in ('0.0000000000005', '-0.0000000000005', '0.0000000000015', '-0.0000000000025', '999999999999999999.9999999999995')]
        got = self.q('SELECT finance.q(x) FROM unnest(%s::numeric[]) WITH ORDINALITY AS u(x,i) ORDER BY i', vals)
        bad = [(v, g[0], quantise(v)) for v, g in zip(vals, got) if g[0] != quantise(v)]
        self.assertEqual(bad, [])

    def test_run_rate_day_parity_random(self):
        """SQL computes finance.q(a/b): numeric division already rounds at a dynamic scale (>=16 significant digits),
        then q() rounds again. Python divides at 80 digits and rounds once."""
        from finance.policy import divide, multiply
        rnd = random.Random(7); mism = []
        monthly = [Decimal(rnd.randint(1, 10**7)).scaleb(-rnd.randint(0, 12)) for _ in range(20000)]
        got = self.q("SELECT finance.q(m*12/365.25) FROM unnest(%s::numeric[]) WITH ORDINALITY AS u(m,i) ORDER BY i", monthly)
        for m, g in zip(monthly, got):
            py = divide(multiply(m, Decimal(12)), Decimal('365.25'))
            if g[0] != py:
                mism.append((m, g[0], py))
        if mism:
            m = mism[0][0]
            self.coverage('rc', 'RECURRING', 'PROJECT', 1)
            self.q("INSERT INTO finance.subscription_schedule(schedule_id,subscription_id,effective_from,effective_to,monthly_gbp,verification,evidence_ref) VALUES('s','x',%s,%s,%s,'invoice_verified','evidence:s')", S, E, m)
            day_sql = self.one('SELECT day_gbp FROM finance.run_rate(%s,%s,%s)', S, S, E)
            import finance.core as core
            from finance.policy import Coverage
            day_py = core.run_rate([core.Subscription('x', str(m), S, E, 'invoice_verified')], S,
                                   Coverage('VERIFIED', 1, S, E, 'f', 'f', 'manifest:c', 'r', 0, True, True))['day']
            self.fail('%d/20000 run-rate day mismatches; e.g. monthly=%s finance.run_rate day=%s Python run_rate day=%s'
                      % (len(mism), m, day_sql, day_py))

    def test_recognition_parity_random(self):
        import finance.core as core
        from finance.policy import Coverage
        rnd = random.Random(99); ledger = core.ExpenseLedger(); n = 60
        base = 1790812800  # 2026-10-01T00:00:00Z
        from datetime import datetime, timezone
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
        for i in range(n):
            amount = Decimal(rnd.randint(1, 10**9)).scaleb(-rnd.randint(0, 6))
            fx = Decimal(rnd.randint(5 * 10**11, 2 * 10**12)).scaleb(-12)
            ccy = rnd.choice(('USD', 'EUR'))
            bs = base + rnd.randint(-40, 40) * 86400 + rnd.randint(0, 86399)
            be = bs + rnd.randint(1, 400) * 86400 + rnd.randint(0, 86399)
            policy = rnd.choice(('straight_line_seconds_v1', 'point_in_time_v1'))
            self.expense('x%d' % i, str(amount), paid=iso(bs), policy=policy, bs=iso(bs), be=iso(be), incurred=iso(bs), fx=str(fx), ccy=ccy)
            self.verify('v%d' % i, 'x%d' % i)
            row = core.Expense('x%d' % i, 1, 'invoice:x%d' % i, iso(bs), 'v', 'd', 'infrastructure', str(amount), ccy, str(fx), 'fx', S,
                               'invoice:x', 'unverified', 'svc', 'PRODUCTION', iso(bs), iso(be), paid_at=iso(bs), recognition_policy=policy)
            ledger.append(row); ledger.verify('x%d' % i, 1, 'invoice_verified', 'r', 'evidence:v', S)
        wide_s, wide_e = '2020-01-01T00:00:00Z', '2030-01-01T00:00:00Z'
        self.coverage('ce', 'EXPENSES', 'PROJECT', n, start=wide_s, end=wide_e)
        cov = Coverage('VERIFIED', n, wide_s, wide_e, 'f', 'f', 'manifest:c', 'r', 0, True, True)
        mism = []
        for _ in range(150):
            a = base + rnd.randint(-60, 60) * 86400 + rnd.randint(0, 86399); b = a + rnd.randint(1, 120) * 86400 + rnd.randint(0, 86399)
            sql = self.q('SELECT cash_spend_gbp,economic_expense_gbp FROM finance.expense_totals(%s,%s)', iso(a), iso(b))[0]
            py = core.expense_totals(ledger, iso(a), iso(b), cov)
            if sql != (py['cash_spend'], py['economic_expense']):
                mism.append((iso(a), iso(b), sql, (py['cash_spend'], py['economic_expense'])))
        self.assertEqual(mism, [], '%d/150 expense_totals windows differ, first: %s' % (len(mism), mism[:1]))

    def test_finding_recognition_single_rounding_parity_vector(self):
        """Whole-second annual service period (+1 s) cut at a month boundary. SQL numeric division rounds at its
        dynamic result scale, then finance.q() rounds again (double rounding); Python rounds once at 80 digits."""
        import finance.core as core
        from finance.policy import Coverage
        g, bs, be = '1234.559980108419', '2026-10-01T00:00:00Z', '2027-10-01T00:00:01Z'
        self.expense('y', g, paid=bs, policy='straight_line_seconds_v1', bs=bs, be=be, incurred=bs); self.verify('vy', 'y')
        self.coverage('cy', 'EXPENSES', 'PROJECT', 1, start='2020-01-01T00:00:00Z', end='2030-01-01T00:00:00Z')
        sql = self.one('SELECT economic_expense_gbp FROM finance.expense_totals(%s,%s)', S, E)
        ledger = core.ExpenseLedger()
        ledger.append(core.Expense('y', 1, 'invoice:y', bs, 'v', 'd', 'infrastructure', g, 'GBP', '1', 'fx', S, 'invoice:x',
                                   'unverified', 'svc', 'PRODUCTION', bs, be, paid_at=bs, recognition_policy='straight_line_seconds_v1'))
        ledger.verify('y', 1, 'invoice_verified', 'r', 'evidence:v', S)
        py = core.expense_totals(ledger, S, E, Coverage('VERIFIED', 1, '2020-01-01T00:00:00Z', '2030-01-01T00:00:00Z', 'f', 'f',
                                                       'manifest:c', 'r', 0, True, True))['economic_expense']
        self.assertEqual(sql, py, 'October recognition: SQL %s vs Python %s' % (sql, py))

    def test_broker_totals_parity(self):
        import finance.core as core
        from finance.policy import Scope
        live = Scope('OANDA_LIVE', 'PRODUCTION', 'ACTUAL', 'REALISED', None)
        rnd = random.Random(5); self.account('a'); comps = []
        for i in range(120):
            vals = [Decimal(rnd.randint(-10**8, 10**8)).scaleb(-rnd.randint(0, 12)) for _ in range(2)]
            com = Decimal(rnd.randint(0, 10**6)).scaleb(-rnd.randint(0, 12)); fee = Decimal(rnd.randint(0, 10**4)).scaleb(-4)
            self.fill('a', 't%d' % i, S, str(vals[0]), str(vals[1]), str(-com), str(-fee))
            for key, kind, v in (('pl', 'realised_pnl', vals[0]), ('financing', 'financing', vals[1]),
                                 ('commission', 'commission', -com), ('guaranteedExecutionFee', 'other_adjustment', -fee)):
                comps.append(core.BrokerComponent('a', live, 't%d' % i, key, S, kind, str(v), 'GBP', '1'))
        sql = self.one('SELECT observed_net_gbp FROM finance.account_totals(%s,%s,%s)', 'a', S, E)
        self.assertEqual(sql, core.broker_totals(comps, 'a', live, S, E, None)['observed_net_trading'])


# ======================================================================== author runner isolation
class RunnerIsolation(unittest.TestCase):
    def test_runner_refuses_non_local_or_external_socket(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT.parent))
        for sock in ('/tmp', 'localhost', str(ROOT.parent / 'no_such_dir')):
            r = subprocess.run([sys.executable, '-m', 'finance.tests.postgres_local_validation', '--socket', sock, '--user', 'x'],
                               cwd=ROOT.parent, env=env, capture_output=True, text=True, timeout=30)
            with self.subTest(sock=sock):
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('local scratch Unix socket required', r.stderr + r.stdout)


if __name__ == '__main__':
    unittest.main()
