"""Offline evidence hygiene; never grants release authority or accesses a service."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import re
from pathlib import Path
import subprocess

DATA=frozenset(('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy'))
TRADING=frozenset(('/market-data','/evaluate','/trade-opened','/trade-outcome'))
FORBIDDEN=frozenset(('2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c','c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44'))

def publication_errors(m):
    errors=[]
    if m.get('status')=='VERIFIED_PUBLISHED':
        for k in ('destination_url','original_archive_sha256','source_sha','source_tree'):
            if not m.get(k):errors.append('missing '+k)
        for k in ('destination_hash_verified','access_verified'):
            if m.get(k) is not True:errors.append('unverified '+k)
    if m.get('independent_assurance') is True and m.get('author_test_evidence') is True:errors.append('author tests are not independent assurance')
    return errors

def artifact_errors(m):
    errors=[]
    if m.get('release_sha') in FORBIDDEN or m.get('recovery_sha') in FORBIDDEN:errors.append('prohibited bare recovery/release target')
    for field in ('merge_result_sha','approved_result_sha','merge_result_tree','approved_result_tree'):
        if not re.fullmatch('[0-9a-f]{40}',m.get(field,'') or ''):errors.append('invalid '+field)
    if m.get('fallback_activation') and not m.get('fallback_files_present'):errors.append('fallback files absent')
    if m.get('merge_result_sha')!=m.get('approved_result_sha'):errors.append('result SHA not approved before ref change')
    if m.get('merge_result_tree')!=m.get('approved_result_tree'):errors.append('result tree mismatch')
    if not m.get('merge_result_sha') or not m.get('approved_result_tree'):errors.append('result identity unavailable')
    if m.get('dependencies_reproducible') is not True:errors.append('build not reproducible')
    return errors

def _utc(v):
    t=datetime.fromisoformat(v.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('timezone required')
    return t.astimezone(timezone.utc)

def summarise_access(rows,start,end,complete=False):
    lo,hi=_utc(start),_utc(end)
    if hi<=lo:raise ValueError('invalid window')
    seen={};counts=Counter();ignored=duplicates=invalid=0
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('record_id'),str) or not row['record_id']:
            invalid+=1;continue
        identity=row['record_id']
        if identity in seen:
            duplicates+=1
            if row!=seen[identity]:invalid+=1
            continue
        seen[identity]=row
        try:at=_utc(row['at'])
        except (KeyError,ValueError,TypeError,AttributeError):invalid+=1;continue
        route=row.get('route');status=row.get('status')
        if type(status) is not int or not 100<=status<=599 or route not in DATA|TRADING|{'/analytics/access-status'}:
            invalid+=1;continue
        if not lo<=at<hi:ignored+=1;continue
        counts[(route,status)]+=1
    disclosure=sum(n for (r,s),n in counts.items() if r in DATA and 200<=s<300)
    return {'coverage':'SUPPLIED_COMPLETE_WINDOW' if complete and not invalid else 'UNKNOWN_OR_PARTIAL','observed_responses':sum(counts.values()),'duplicates':duplicates,'invalid':invalid,'outside_window':ignored,'unexpected_data_2xx':disclosure,'rows':[{'route':r,'status':s,'count':n} for (r,s),n in sorted(counts.items())],'limits':'transport only; no identity, persistence, fill or dataset completeness claim'}

def source_inventory(repo,sha):
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    paths=set(git('ls-tree','-r','--name-only',sha).splitlines())
    required={'security/analytics_release/containment_fallback.py','security/analytics_release/fallback_app.py'}
    return {'sha':git('rev-parse',sha+'^{commit}'),'tree':git('rev-parse',sha+'^{tree}'),'fallback_files_present':required<=paths,'runtime_blobs':{p:git('rev-parse',sha+':'+p) for p in ('engine.py','analytics_access_control.py','requirements.txt')}}

def file_manifest(folder):
    root=Path(folder)
    return [{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()]
