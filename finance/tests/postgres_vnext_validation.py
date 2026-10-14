"""Opt-in runtime tests; NEVER a shared/remote target.

Creates/drops fresh local finance_vnext_<random>_isolated databases. Requires a
repository-local Unix socket, socket-only PostgreSQL18, and psycopg3. Test-only
synthetic fixtures. This is supplied but must not be described as executed until
its actual runtime output is recorded.
"""
import argparse
import os
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from decimal import Decimal
from finance.policy import prorate

ROOT=Path(__file__).resolve().parents[1]

def main():
    args=argparse.ArgumentParser();args.add_argument('--socket',required=True);args.add_argument('--user',required=True);a=args.parse_args()
    socket=Path(a.socket).resolve()
    if not socket.is_relative_to(ROOT.parent) or not (socket/'.s.PGSQL.5432').exists(): raise SystemExit('repository-local Unix socket required')
    for name in tuple(os.environ):
        if name.startswith('PG'):os.environ.pop(name)
    import psycopg
    from psycopg import sql
    params=dict(host=str(socket),user=a.user,dbname='postgres',connect_timeout=5)
    with psycopg.connect(**params,autocommit=True) as admin:
        assert 180000<=int(admin.execute('SHOW server_version_num').fetchone()[0])<190000
        assert admin.execute('SHOW listen_addresses').fetchone()[0]=='', 'socket-only cluster required'
        name='finance_vnext_'+uuid.uuid4().hex[:12]+'_isolated'
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    params['dbname']=name
    try:
        with psycopg.connect(**params,autocommit=True) as c:
            def execute(q): return c.execute(q)
            migration=(ROOT/'migrations/001_foundation.sql').read_text()
            # A failed fresh migration must leave no schema behind.
            try:execute(migration.replace('COMMIT;','SELECT 1/0; COMMIT;'))
            except psycopg.Error:execute('ROLLBACK')
            assert execute("SELECT to_regnamespace('finance')").fetchone()[0] is None
            execute(migration)
            try:execute(migration);raise AssertionError('migration reapplied')
            except psycopg.errors.DuplicateSchema:execute('ROLLBACK')
            execute((ROOT/'tests/postgres_acceptance.sql').read_text())
            assert execute('SELECT finance.q_ratio(1234.559980108419*2678400,31536001)').fetchone()[0]==prorate(Decimal('1234.559980108419'),Decimal(2678400),Decimal(31536001))
            execute("INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('e','invoice:test')")
            execute("""INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification)
             VALUES('e',1,'2026-10-01Z','test','test','other',10,'GBP',1,'identity','2026-10-01Z','2026-10-01Z','2026-11-01Z',false,'test','VALIDATION','evidence:test','unverified')""".replace('2026-10-01Z','2026-10-01T00:00:00Z').replace('2026-11-01Z','2026-11-01T00:00:00Z'))
            execute("INSERT INTO finance.broker_account(account_key,broker_environment,system_environment,financial_reality,financial_state,currency) VALUES('a','OANDA_PRACTICE','VALIDATION','ACTUAL','REALISED','GBP')")
            def verify(ident,state):return f"INSERT INTO finance.expense_verification(decision_id,expense_id,revision,decision,reviewed_by,evidence_ref) VALUES('{ident}','e',1,'{state}','test','evidence:test')"
            def coverage(ident,state):return f"INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,currency_confirmed,costs_complete,reviewed_by,evidence_ref) VALUES('{ident}','EXPENSES','PROJECT','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','{state}',0,true,true,'test','evidence:test')"
            def run(ident):return f"INSERT INTO finance.reconciliation_run(run_id,account_key,period_start,period_end,source_as_of,outcome_due_at,broker_evidence_count,internal_evidence_count,broker_complete,internal_complete,broker_reviewed_empty,internal_reviewed_empty,evidence_ref) VALUES('{ident}','a','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','2026-11-02T00:00:00Z','2026-11-02T00:00:00Z',0,0,true,true,true,true,'evidence:test')"
            def race(first,second,latest_query,expected):
                # T2 must wait for T1's commit; it cannot leapfrog the same key.
                started=Event()
                with psycopg.connect(**params) as t1:
                    t1.execute(first)
                    def insert_second():
                        with psycopg.connect(**params) as t2:
                            t2.execute("SET LOCAL lock_timeout='5s'");started.set();t2.execute(second);t2.commit()
                    with ThreadPoolExecutor(1) as pool:
                        future=pool.submit(insert_second);assert started.wait(2)
                        # Independent connection verifies an actual lock waiter.
                        for _ in range(200):
                            if c.execute("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock')").fetchone()[0]:break
                        else:raise AssertionError('same-key transaction did not block')
                        assert not future.done();t1.commit();future.result(timeout=7)
                assert c.execute(latest_query).fetchone()[0]==expected
            race(verify('v1','invoice_verified'),verify('v2','withdrawn'),"SELECT decision FROM finance.latest_verifications WHERE expense_id='e'",'withdrawn')
            race(verify('v3','withdrawn'),verify('v4','invoice_verified'),"SELECT decision FROM finance.latest_verifications WHERE expense_id='e'",'invoice_verified')
            race(coverage('c1','UNKNOWN'),coverage('c2','WITHDRAWN'),"SELECT state FROM finance.latest_coverage WHERE domain='EXPENSES'",'WITHDRAWN')
            race(run('r1'),run('r2'),"SELECT run_id FROM finance.latest_runs WHERE account_key='a'",'r2')
            execute("INSERT INTO finance.reconciliation_result(run_id,result_id,trade_id,state,reason,evidence_ref) VALUES('r1','bad','trade','VALUE_MISMATCH','test','evidence:test'),('r2','good','trade','MATCHED','test','evidence:test')")
            def resolution(ident,state):return f"INSERT INTO finance.exception_resolution(decision_id,run_id,result_id,state,matched_run_id,matched_result_id,reviewed_by,evidence_ref) VALUES('{ident}','r1','bad','{state}',"+("'r2','good'" if state=='RESOLVED' else 'NULL,NULL')+",'test','evidence:test')"
            race(resolution('d1','RESOLVED'),resolution('d2','WITHDRAWN'),"SELECT state FROM finance.latest_resolutions WHERE run_id='r1'",'WITHDRAWN')
            execute(run('r3'))
            try:execute(resolution('stale','RESOLVED'));raise AssertionError('stale resolution accepted')
            except psycopg.errors.RaiseException:pass
            # Explicit compare-and-set cannot ignore a later accepted decision.
            try:execute(verify('cas','withdrawn').replace('evidence_ref)', 'evidence_ref,supersedes_server_order)').replace("'evidence:test')","'evidence:test',0)"));raise AssertionError('stale CAS accepted')
            except psycopg.errors.RaiseException:pass
            execute('BEGIN');execute(coverage('rollback','UNKNOWN'));execute('ROLLBACK')
            assert execute("SELECT count(*) FROM finance.coverage_assertion WHERE assertion_id='rollback'").fetchone()[0]==0
            role='finance_test_'+uuid.uuid4().hex[:8]
            execute(sql.SQL('CREATE ROLE {}').format(sql.Identifier(role)))
            try:
                execute(sql.SQL('SET ROLE {}').format(sql.Identifier(role)))
                try:execute('SELECT * FROM finance.latest_coverage');raise AssertionError('PUBLIC read allowed')
                except psycopg.errors.InsufficientPrivilege:pass
            finally:execute('RESET ROLE');execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))
            print('PASS PG18 clean migration/rollback/reapplication, acceptance, exact recognition, 4 authority races, resolution/withdrawal/stale/CAS, rollback and privilege denial')
    finally:
        params['dbname']='postgres'
        with psycopg.connect(**params,autocommit=True) as admin:admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))

if __name__=='__main__':main()
