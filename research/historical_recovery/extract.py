"""Operator-only bounded read-only acquisition. No public endpoint fallback.

Credentials are read by libpq from an operator-owned PGSERVICE/pgpass setup.
No DSN command argument, connection string, account data or exception text is
printed. The operator must inspect actual least-privilege service identity.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from .normalizer import iso_time, INSTRUMENTS, TIMEFRAMES, digest, canonical_timestamp

MAX_ROWS=50000
MAX_EXPORT_BYTES=64*1024*1024
SELECT_FIELDS='event_id,event_type,origin,instrument,timeframe,event_time,received_at,schema_version,environment,retry_count,payload'
WHERE="event_type = 'market_data' AND instrument = %s AND timeframe = %s AND environment = %s AND received_at >= %s AND received_at < %s"
SENSITIVE=re.compile(r'(token|secret|password|passphrase|authorization|cookie|credential|account|connection|dsn)|^(key|auth|apikey|privatekey|accesskey|clientsecret)$',re.I)
SECRET_VALUE=re.compile(r"(?i)(?:bearer\s|postgres(?:ql)?://|(?:^|\s)(?:password|passphrase|sslpassword)\s*=|(?:api[_-]?key|auth|token)\s*[=:]|(?:^|[^A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:$|[^A-Za-z0-9_-]))")

# Bounded JSON encoding retains Decimal source numeric values as JSON numbers,
# not strings. It cannot recover pre-JSONB original HTTP lexical bytes.
def json_bytes(value):
    from decimal import Decimal
    def encode(v):
        if isinstance(v,Decimal):
            if not v.is_finite():raise ValueError('NONFINITE_JSON')
            return str(v)
        if isinstance(v,datetime):return json.dumps(canonical_timestamp(v))
        if isinstance(v,dict):
            if not all(isinstance(k,str) for k in v):raise ValueError('NONSTRING_KEY')
            return '{'+','.join(json.dumps(k)+':'+encode(x) for k,x in sorted(v.items()))+'}'
        if isinstance(v,list):return '['+','.join(encode(x) for x in v)+']'
        return json.dumps(v,allow_nan=False,ensure_ascii=True)
    return encode(value).encode()

def redact(value,path='$',removed=None):
    removed=[] if removed is None else removed
    if isinstance(value,dict):
        clean={}
        for k,v in value.items():
            if SENSITIVE.search(re.sub(r'[^a-z0-9]','',k.lower())):removed.append(path+'.[sensitive-key]');clean[k]='[REDACTED]'
            else:clean[k]=redact(v,path+'.'+k,removed)[0]
        return clean,removed
    if isinstance(value,list):return [redact(v,path+'[]',removed)[0] for v in value],removed
    if isinstance(value,str) and (SECRET_VALUE.search(value)):
        removed.append(path);return '[REDACTED]',removed
    return deepcopy(value),removed

def validate_bounds(instrument,timeframe,environment,start,end,batch_size,max_rows):
    a,b=iso_time(start),iso_time(end)
    if instrument not in INSTRUMENTS or timeframe not in TIMEFRAMES or environment not in {'production','shadow','test','development'}:raise ValueError('INVALID_SCOPE')
    if not timedelta(0)<b-a<=timedelta(days=7):raise ValueError('INVALID_RECEIPT_WINDOW')
    if type(batch_size) is not int or not 1<=batch_size<=1000 or type(max_rows) is not int or not 1<=max_rows<=MAX_ROWS:raise ValueError('INVALID_BATCH_BOUNDS')
    return a,b

def acquire(conn,*,instrument,timeframe,environment,start,end,batch_size=250,max_rows=5000,after_event_id=None):
    """One connection/transaction, one fixed snapshot for count and all pages.

    Do not run separate invocations as one claimed snapshot. Resume requires a
    new snapshot and explicit comparison; there is no cross-session fake cursor.
    A row cap creates an explicitly INCOMPLETE snapshot export.
    """
    a,b=validate_bounds(instrument,timeframe,environment,start,end,batch_size,max_rows)
    if after_event_id is not None and not re.fullmatch(r'[0-9a-f]{64}',after_event_id):raise ValueError('INVALID_CURSOR')
    params=(instrument,timeframe,environment,a,b)
    export_bytes=0
    original_count=0;rows=[];pages=[];snapshot=None;identity=None;last=after_event_id
    # set_session only sets this connection's upcoming transaction modes.
    # No schema, row, network or server configuration mutation.
    conn.set_session(readonly=True,isolation_level='REPEATABLE READ',autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '10s'")
            cur.execute("SET LOCAL lock_timeout = '1s'")
            cur.execute("SELECT current_setting('transaction_read_only'), current_setting('transaction_isolation'), pg_current_snapshot()::text, current_user")
            ro,level,snapshot,identity=cur.fetchone()
            if ro!='on' or level!='repeatable read':raise ValueError('READ_ONLY_SNAPSHOT_REQUIRED')
            cur.execute('SELECT COUNT(*) FROM public.events WHERE '+WHERE,params)
            original_count=cur.fetchone()[0]
            while len(rows)<max_rows:
                take=min(batch_size,max_rows-len(rows));cursor_clause=' AND event_id COLLATE "C" > %s' if last is not None else ''
                sql='SELECT '+SELECT_FIELDS+' FROM public.events WHERE '+WHERE+cursor_clause+' ORDER BY event_id COLLATE "C" LIMIT %s'
                cur.execute(sql,params+((last,) if last is not None else ())+ (take,))
                names=[getattr(c,'name',None) or c[0] for c in cur.description]
                values=[dict(zip(names,row)) for row in cur.fetchall()]
                if not values:break
                for value in values:
                    encoded=json_bytes(value)
                    export_bytes+=len(encoded)
                    if len(encoded)>524288 or export_bytes>MAX_EXPORT_BYTES:raise ValueError('EXPORT_BYTE_BOUND_EXCEEDED')
                ids=[r['event_id'] for r in values]
                if ids!=sorted(set(ids)) or (last is not None and ids[0]<=last):raise ValueError('PAGINATION_NONMONOTONIC')
                if any(r['instrument']!=instrument or r['timeframe']!=timeframe or r['environment']!=environment or r['event_type']!='market_data' for r in values):raise ValueError('CROSS_SCOPE_RESPONSE')
                pages.append({'page':len(pages)+1,'after':last,'last':ids[-1],'rows':len(values),'source_page_digest':digest(values),'snapshot':snapshot})
                rows.extend(values);last=ids[-1]
                if len(values)<take:break
    finally:
        conn.rollback()  # Always close the snapshot without committing anything.
    captured=len(rows)==original_count and after_event_id is None
    raw_digests=[digest(row) for row in rows]
    redacted=[];paths=[]
    for row in rows:
        clean,removed=redact(row);paths.extend(removed)
        clean['acquisition_source_event_digest']=digest(row)
        clean['acquisition_source_payload_digest']=digest(row['payload'])
        clean['acquisition_redacted_event_digest']=digest({k:v for k,v in clean.items() if not k.startswith('acquisition_')})
        redacted.append(clean)
    manifest={'contract':'historical_acquisition_v1.1','digest_encoding':'canonical_json_value_v2_utc6','timestamp_encoding':'utc6_z_v1','operator_identity_reference':identity,'acquired_at':datetime.now(timezone.utc).isoformat(),'database_identity':identity,'scope':{'instrument':instrument,'timeframe':timeframe,'environment':environment},'window_basis':'received_at_UTC_not_source_bar_time','window':{'start':a.isoformat(),'end':b.isoformat()},'source_count':original_count,'exported_count':len(rows),'complete_captured_snapshot':captured,'historical_source_completeness':'UNPROVED','snapshot':snapshot,'isolation':'repeatable read / read only','pages':pages,'resume_after_event_id':last if not captured else None,'prior_cursor':after_event_id,'redacted_field_paths':sorted(set(paths)),'redaction_policy':'structured_secret_redaction_v2','unredacted_population_digest':digest(raw_digests),'redacted_population_digest':digest(redacted),'raw_transport_bytes_retained':False,'redaction_notice':'Private-only export. Key/pattern filtering is defense in depth, not a guarantee that arbitrary source text contains no secrets. Operator must review before sharing.'}
    manifest['manifest_hash']=digest(manifest)
    return redacted,manifest

def write_export(directory,rows,manifest):
    """New private directory only; never overwrite existing evidence."""
    directory=Path(directory)
    if not directory.is_absolute() or '..' in directory.parts:raise ValueError('ABSOLUTE_PRIVATE_DESTINATION_REQUIRED')
    for parent in directory.parents:
        if parent.is_symlink():raise ValueError('SYMLINK_DESTINATION_REJECTED')
    directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    try:
        for filename,data in [('events.jsonl',b''.join(json_bytes(r)+b'\n' for r in rows)),('manifest.json',json_bytes(manifest)+b'\n')]:
            fd=os.open(directory/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f:f.write(data)
        audit={'action':'READ_ONLY_MARKET_DATA_EXPORT','manifest_hash':manifest['manifest_hash'],'events_file_sha256':sha256((directory/'events.jsonl').read_bytes()).hexdigest(),'manifest_file_sha256':sha256((directory/'manifest.json').read_bytes()).hexdigest(),'source_count':manifest['source_count'],'exported_count':len(rows),'status':'COMPLETE_CAPTURED_SNAPSHOT' if manifest['complete_captured_snapshot'] else 'PARTIAL','not_authoritative_research':True}
        fd=os.open(directory/'export_audit.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(json_bytes(audit)+b'\n')
    except Exception:
        # Preserve partial files for investigation; never silently claim complete.
        raise ValueError('EXPORT_WRITE_FAILED_PARTIAL_FILES_MAY_EXIST') from None

def verify_reader_identity(conn,expected_login):
    """Future CLI preflight; no role/grant changes. Not a full privilege audit."""
    if not isinstance(expected_login,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',expected_login):raise ValueError('APPROVED_READER_LOGIN_REQUIRED')
    conn.set_session(readonly=True,isolation_level='REPEATABLE READ',autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute("""SELECT current_user, r.rolsuper OR r.rolcreaterole OR r.rolcreatedb OR r.rolreplication OR r.rolbypassrls,
              has_table_privilege(current_user,'public.events','SELECT'),
              has_table_privilege(current_user,'public.events','INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER,REFERENCES'),
              has_database_privilege(current_user,current_database(),'CREATE'),
              has_schema_privilege(current_user,'public','CREATE'),
              EXISTS (SELECT 1 FROM pg_roles inherited WHERE pg_has_role(current_user,inherited.oid,'MEMBER')
                AND (inherited.rolsuper OR inherited.rolcreaterole OR inherited.rolcreatedb OR inherited.rolreplication OR inherited.rolbypassrls
                  OR has_table_privilege(inherited.rolname,'public.events','INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER,REFERENCES')))
              FROM pg_roles r WHERE r.rolname=current_user""")
            identity,elevated,can_select,can_write,can_create_db,can_create_schema,member_write=cur.fetchone()
            if identity!=expected_login or elevated or not can_select or can_write or can_create_db or can_create_schema or member_write:raise ValueError('DEDICATED_SELECT_ONLY_IDENTITY_REQUIRED')
    finally:conn.rollback()
    return {'identity':identity,'check':'ROLE_FLAGS_EVENTS_TABLE_DDL_MEMBERSHIP','residual':'Operator must separately review function execution, other objects and network/custody policy.'}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('instrument','timeframe','environment','start','end','output'):parser.add_argument('--'+k,required=True)
    parser.add_argument('--batch-size',type=int,default=250);parser.add_argument('--max-rows',type=int,default=5000);parser.add_argument('--after-event-id');parser.add_argument('--approved-reader-login');parser.add_argument('--approved-execution-context')
    args=vars(parser.parse_args(argv));output=args.pop('output');reader=args.pop('approved_reader_login');execution_context=args.pop('approved_execution_context')
    conn=None
    try:
        validate_bounds(**{k:v for k,v in args.items() if k!='after_event_id'})
        if not reader or not execution_context or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}',execution_context):raise ValueError('APPROVED_ISOLATED_EXECUTION_CONTEXT_REQUIRED')
        if not os.environ.get('PGSERVICE'):raise ValueError('PGSERVICE_REQUIRED')
        import psycopg2
        conn=psycopg2.connect(connect_timeout=5)
        from psycopg2.extras import register_default_jsonb
        from decimal import Decimal
        register_default_jsonb(conn,loads=lambda text: json.loads(text,parse_float=Decimal,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('NONFINITE_SOURCE_JSON'))))
        identity_check=verify_reader_identity(conn,reader)
        rows,manifest=acquire(conn,**args)
        manifest['reader_identity_check']=identity_check;manifest['approved_execution_context']=execution_context
        manifest['manifest_hash']=digest({k:v for k,v in manifest.items() if k!='manifest_hash'})
        write_export(output,rows,manifest)
        print(json.dumps({'status':'COMPLETE_CAPTURED_SNAPSHOT' if manifest['complete_captured_snapshot'] else 'PARTIAL','exported_count':len(rows),'manifest_hash':manifest['manifest_hash']}))
        return 0
    except Exception:
        # No exception repr, DSN, password, SQL row or payload can reach stdout.
        print('{"status":"FAILED","error":"ACQUISITION_FAILED_SEE_OPERATOR_RUNBOOK"}')
        return 1
    finally:
        if conn is not None:conn.close()

if __name__=='__main__':raise SystemExit(main())
