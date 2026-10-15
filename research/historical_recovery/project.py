"""Offline projections from private operator exports. Defaults to unproved trust.

Do not serve this CLI as an unauthenticated web product. Context is independent
operator-reviewed evidence, not copied source assertions or governance approval.
"""
import argparse,json
from decimal import Decimal
from pathlib import Path
from .normalizer import Calendar,Context,digest,iso_time,MAX_PAYLOAD_BYTES
from .extract import json_bytes
from .auditor import audit

MAX_FILE_BYTES=64*1024*1024

def strict_json(text):
    def pairs(items):
        r={}
        for k,v in items:
            if k in r:raise ValueError('DUPLICATE_JSON_KEY')
            r[k]=v
        return r
    def nonfinite(s):raise ValueError('NONFINITE_JSON')
    return json.loads(text,parse_float=Decimal,parse_constant=nonfinite,object_pairs_hook=pairs)

def load_json(path,max_bytes=MAX_FILE_BYTES):
    path=Path(path)
    if path.stat().st_size>max_bytes:raise ValueError('FILE_TOO_LARGE')
    return strict_json(path.read_text())

def project_export(directory,*,start,end,context=None):
    directory=Path(directory);manifest=load_json(directory/'manifest.json');events_path=directory/'events.jsonl'
    if events_path.stat().st_size>MAX_FILE_BYTES:raise ValueError('FILE_TOO_LARGE')
    if manifest.get('contract')!='historical_acquisition_v1':raise ValueError('UNKNOWN_MANIFEST')
    if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest.get('manifest_hash'):raise ValueError('MANIFEST_INTEGRITY')
    rows=[]
    with events_path.open() as f:
        for line in f:
            if len(line)>MAX_PAYLOAD_BYTES*2 or len(rows)>=50000:raise ValueError('INPUT_BOUNDS')
            row=strict_json(line)
            rows.append(row)
    if digest(rows)!=manifest['redacted_population_digest']:raise ValueError('POPULATION_INTEGRITY')
    if len(rows)!=manifest['exported_count']:raise ValueError('POPULATION_COUNT')
    bare=[]
    for row in rows:
        source={k:v for k,v in row.items() if not k.startswith('acquisition_')}
        if digest(source)!=row.get('acquisition_redacted_event_digest'):raise ValueError('REDACTED_ROW_INTEGRITY')
        bare.append(source)
    if context is None:
        context=Context(manifest['scope']['environment'],'tradingview','unreviewed-acquisition-'+manifest['manifest_hash'][:16])
    if context.environment!=manifest['scope']['environment']:raise ValueError('CONTEXT_SCOPE_MISMATCH')
    # Evidence window is receipt-based: source-time coverage cannot be promoted
    # merely because a received-at snapshot was complete.
    result=audit(bare,context,start=start,end=end,source_count=manifest['source_count'],snapshot_complete=manifest['complete_captured_snapshot'])
    lineage={r['event_id']:r for r in rows}
    for projected in result['normalized_rows']:
        original=lineage[projected['event_id']]
        projected['source_original_event_digest']=original['acquisition_source_event_digest']
        projected['source_original_payload_digest']=original['acquisition_source_payload_digest']
        projected['redaction_applied']=bool(manifest['redacted_field_paths'])
        projected['projection_digest']=digest({k:v for k,v in projected.items() if k!='projection_digest'})
    result['projection_population_hash']=digest([r['projection_digest'] for r in result['normalized_rows']])
    result['acquisition_manifest_hash']=manifest['manifest_hash']
    result['receipt_snapshot_complete']=manifest['complete_captured_snapshot']
    result['source_window_population_complete']=False
    result['dataset_complete_for_research']=False
    result['source_population_status']='RECEIPT_SNAPSHOT_ONLY_SOURCE_WINDOW_UNPROVED'
    result['manifest_hash']=digest({k:v for k,v in result.items() if k not in ('normalized_rows','manifest_hash')})
    return result

def context_from_file(path):
    cfg=load_json(path,1024*1024)
    if cfg.get('contract')!='historical_context_v1':raise ValueError('INVALID_CONTEXT_CONTRACT')
    cal=cfg.get('calendar')
    if cal:
        cal=Calendar(cal['version'],cal['instrument'],cal['timeframe'],iso_time(cal['start']),iso_time(cal['end']),tuple((iso_time(a),iso_time(b)) for a,b in cal['slots']),cal.get('approved') is True,cal.get('reviewed_empty') is True)
    return Context(cfg['environment'],cfg['source_id'],cfg['provenance_reference'],frozenset(cfg.get('verified_event_digests',[])),frozenset(cfg.get('known_versions',[])),cal)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--output',required=True);p.add_argument('--reviewed-context')
    a=p.parse_args(argv)
    try:
        report=project_export(a.export,start=a.start,end=a.end,context=context_from_file(a.reviewed_context) if a.reviewed_context else None)
        # Exclusive output, private permissions; no overwrite or production write.
        import os
        fd=os.open(a.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(json_bytes(report)+b'\n')
        print(json.dumps({'status':'OFFLINE_UNAPPROVED_PROJECTION','manifest_hash':report['manifest_hash'],'records':report['record_count']}));return 0
    except Exception:
        print('{"status":"FAILED","error":"PROJECTION_FAILED_NO_TRUTH_PROMOTED"}');return 1

if __name__=='__main__':raise SystemExit(main())
