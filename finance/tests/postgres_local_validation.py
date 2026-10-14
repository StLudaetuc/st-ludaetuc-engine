"""Explicit opt-in local-only PG18 acceptance, parity and concurrent INSERT checks.

Requires existing empty database named finance_validation_isolated and psql on
PATH. Refuses remote hosts, URIs, inherited libpq service configuration and shared
DB names. No cloud/shared connection attempt when the local runtime is absent.
"""
import argparse
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from finance.policy import quantise, convert, divide

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--user',required=True)
    args=p.parse_args();socket=Path(args.socket).resolve()
    if not socket.is_relative_to(ROOT.parent) or not socket.is_dir() or not (socket/'.s.PGSQL.5432').exists():
        raise SystemExit('local scratch Unix socket required; no TCP/URI/remote fallback')
    env={k:v for k,v in os.environ.items() if not k.startswith('PG')}
    command=['psql','-X','-v','ON_ERROR_STOP=1','-h',str(socket),'-U',args.user,'-d','finance_validation_isolated','-A','-t']
    def sql(text):return subprocess.run(command+['-c',text],env=env,text=True,capture_output=True,check=True,timeout=30).stdout.strip()
    version=int(sql("SHOW server_version_num"));assert 180000<=version<190000
    assert sql("SELECT count(*) FROM pg_namespace WHERE nspname NOT IN ('pg_catalog','information_schema','public') AND nspname NOT LIKE 'pg_toast%' AND nspname NOT LIKE 'pg_temp%'")=='0', 'fresh database required'
    assert sql("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'")=='0', 'public objects require fresh database'
    sql((ROOT/'migrations/001_foundation.sql').read_text());sql((ROOT/'tests/postgres_acceptance.sql').read_text())
    vectors=[(quantise(Decimal('0.0000000000005')),'finance.q(0.0000000000005)'),
             (quantise(Decimal('-0.0000000000005')),'finance.q(-0.0000000000005)'),
             (convert('123456789.123456789012','0.765432109876'),'finance.q(123456789.123456789012*0.765432109876)'),
             (divide(Decimal(360),Decimal('365.25')),'finance.q(360/365.25)')]
    for expected,expression in vectors:assert Decimal(sql('SELECT '+expression))==expected
    sql("INSERT INTO finance.expense_identity(expense_id,source_key) VALUES('concurrent','source:concurrent')")
    insert="""INSERT INTO finance.expense_revision(expense_id,revision,incurred_at,vendor,description,category,original_amount,currency,fx_to_gbp,fx_source,fx_observed_at,billing_start,billing_end,recurring,service,environment,evidence_ref,verification,correction_reason) VALUES('concurrent',%s,'2026-10-01Z','fixture','fixture','other',10,'GBP',1,'identity','2026-10-01Z','2026-10-01Z','2026-11-01Z',false,'fixture','VALIDATION','evidence:fixture','unverified','fixture')"""
    insert=insert.replace('2026-10-01Z','2026-10-01T00:00:00Z').replace('2026-11-01Z','2026-11-01T00:00:00Z')
    sql(insert%1)
    def revision(_):
        try:sql(insert%2);return 'accepted'
        except subprocess.CalledProcessError as exc:
            if 'revision gap or duplicate' not in exc.stderr:raise
            return 'rejected'
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(revision,range(2)))
    assert sorted(results)==['accepted','rejected'];assert sql("SELECT count(*) FROM finance.expense_revision WHERE expense_id='concurrent'")=='2'
    def ordering(i):
        return sql("INSERT INTO finance.coverage_assertion(assertion_id,domain,scope_key,period_start,period_end,state,evidence_count,currency_confirmed,costs_complete,reviewed_by,evidence_ref,server_order) VALUES('order-%s','EXPENSES','PROJECT','2026-10-01T00:00:00Z','2026-11-01T00:00:00Z','UNKNOWN',0,false,false,'fixture','evidence:fixture',999999999) RETURNING server_order"%i)
    with ThreadPoolExecutor(2) as pool:orders=list(pool.map(ordering,range(2)))
    assert len(set(orders))==2 and '999999999' not in orders
    print('PostgreSQL 18 migration, acceptance, 4 parity vectors, revision race and forced-order checks PASS')

if __name__=='__main__':main()
