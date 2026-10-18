"""Offline, read-only gate. PASS means mechanically consistent, never release authority."""
import argparse,json,re,subprocess,sys
from pathlib import Path
BASE='2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c'
SECURITY='3467133b2208a09e1289651b5be0b45342695d52'
PREP='5f0d07565629fba30d0c2221229fd2483472e701'
FORBIDDEN={BASE,'c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44'}
GATES=('independent_security_review','independent_fallback_review','actual_runtime_match','consumer_acknowledgements','monitoring_access','private_audit_preservation','fallback_operator_drill','ceo_exact_sha_approval','operator_identity','deployment_window','stop_thresholds')
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def check(repo,record):
    failures=[];sha=record.get('release_sha','');fallback=record.get('fallback_sha','')
    for k,v in [('release_sha',sha),('fallback_sha',fallback)]:
        if not re.fullmatch('[0-9a-f]{40}',v or '') or v in FORBIDDEN:failures.append(k+': unsafe or missing exact SHA')
    if failures:return failures
    # This package covers only immutable reviewed candidates; any successor needs fresh review.
    if sha not in (SECURITY,PREP):failures.append('release artifact outside this rehearsal')
    if fallback!=PREP:failures.append('fallback artifact outside this rehearsal')
    try:
        if git(repo,'status','--porcelain'):failures.append('working tree not clean')
        for f in ('engine.py','analytics_access_control.py','requirements.txt'):
            if git(repo,'rev-parse',sha+':'+f)!=git(repo,'rev-parse',SECURITY+':'+f):failures.append('runtime drift: '+f)
        for ref in (sha,fallback):
            if subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',BASE,ref],capture_output=True).returncode:failures.append('ancestry invalid')
        if record.get('expected_main_sha')!=BASE:failures.append('main drift requires new delta review')
        if record.get('auto_deploy')!='yes-on-main-commit':failures.append('auto-deploy unconfirmed')
        for gate in GATES:
            # References are operator attestations, not cryptographic proof of contents.
            v=record.get(gate)
            if not isinstance(v,dict) or v.get('status')!='VERIFIED' or not v.get('evidence_ref') or not v.get('verified_by') or not v.get('verified_at'):failures.append(gate+': evidence missing')
    except subprocess.CalledProcessError:failures.append('Git object unavailable')
    return failures
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    failures=check(a.repo,json.loads(a.record.read_text()))
    print(json.dumps({'decision':'HOLD' if failures else 'MECHANICALLY_CONSISTENT_NOT_AUTHORITY','failures':failures},indent=2));sys.exit(bool(failures))
