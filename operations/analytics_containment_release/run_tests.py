"""Clean child environments, unedited logs and explicit exit metadata."""
import argparse,datetime,hashlib,importlib.metadata,json,os,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--label',required=True);a=p.parse_args()
root=Path(__file__).absolute().parents[2];out=root/'operations/analytics_containment_release/evidence';out.mkdir(exist_ok=True)
if not a.label.replace('_','').isalnum():raise SystemExit('unsafe label')
exe=str(Path(a.python).absolute());env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
commands=[('tools',[exe,'-m','unittest','operations.analytics_containment_release.test_tools','-v']),('security',[exe,'-W','default','-m','unittest','discover','-s','security/analytics_access_control','-p','test_*.py','-v']),('fallback',[exe,'-m','unittest','security.analytics_release.test_containment_fallback','-v']),('operations',[exe,'operations/analytics_containment_release/rehearsal.py'])]
meta={'schema':'ops_test_execution_v1','platform':platform.platform(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':{x:hashlib.sha256((root/x).read_bytes()).hexdigest() for x in ['engine.py','analytics_access_control.py','security/analytics_release/containment_fallback.py','security/analytics_release/fallback_app.py','operations/analytics_containment_release/rehearsal.py','operations/analytics_containment_release/run_tests.py','operations/analytics_containment_release/local_entry.py','operations/analytics_containment_release/release_gate.py','operations/analytics_containment_release/audit_summary.py','operations/analytics_containment_release/test_tools.py','operations/analytics_containment_release/verify_http.py']},'runs':[]}
inventory=subprocess.run([exe,'-c','import sys,json,importlib.metadata; print(json.dumps({"python":sys.version,"packages":sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())},indent=2))'],env=env,capture_output=True,text=True,check=True)
(out/(a.label+'_environment.json')).write_text(inventory.stdout)
for name,cmd in commands:
    t=datetime.datetime.now(datetime.timezone.utc).isoformat()
    r=subprocess.run(cmd,cwd=root,env=env,capture_output=True,timeout=180)
    (out/(a.label+'_'+name+'.log')).write_bytes(r.stdout+r.stderr)
    meta['runs'].append(dict(name=name,command=cmd,started_at=t,exit_code=r.returncode,log=a.label+'_'+name+'.log',log_sha256=hashlib.sha256(r.stdout+r.stderr).hexdigest()))
    print(name,r.returncode,flush=True)
(out/(a.label+'_execution.json')).write_text(json.dumps(meta,indent=2)+'\n')
sys.exit(any(x['exit_code'] for x in meta['runs']))
