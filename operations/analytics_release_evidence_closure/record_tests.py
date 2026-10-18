"""Author recorder; writes exclusively in closure namespace. No service calls."""
import datetime,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).absolute().parents[2];OUT=ROOT/'operations/analytics_release_evidence_closure/evidence'
exe=str(Path(sys.argv[1]).absolute());label=sys.argv[2]
if not label.isalnum():raise SystemExit('unsafe label')
env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
commands=[('closure',[exe,'-m','unittest','operations.analytics_release_evidence_closure.test_evidence_checks','-v']),('security',[exe,'-W','default','-m','unittest','discover','-s','security/analytics_access_control','-p','test_*.py','-v']),('fallback',[exe,'-m','unittest','security.analytics_release.test_containment_fallback','-v']),('priorTools',[exe,'-m','unittest','operations.analytics_containment_release.test_tools','-v']),('priorRehearsal',[exe,'operations/analytics_containment_release/rehearsal.py'])]
meta={'author':'GPTWork, not independent assurance','label':label,'platform':platform.platform(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':None,'runs':[],'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'operations/analytics_release_evidence_closure').glob('*.py'))}}
r=subprocess.run([exe,'-c','import sys,sysconfig,platform,importlib.metadata,json; print(json.dumps(dict(python=sys.version,abi=sysconfig.get_config_var("SOABI"),platform=platform.platform(),packages=sorted((d.metadata["Name"],d.version) for d in importlib.metadata.distributions())),indent=2))'],env=env,capture_output=True,text=True,check=True);meta['environment']=json.loads(r.stdout)
for name,cmd in commands:
 r=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,timeout=180)
 b=r.stdout+r.stderr;log=label+'_'+name+'.log';(OUT/log).write_bytes(b)
 meta['runs'].append(dict(name=name,command=cmd,exit_code=r.returncode,log=log,log_sha256=hashlib.sha256(b).hexdigest()));print(name,r.returncode,flush=True)
(OUT/(label+'.json')).write_text(json.dumps(meta,indent=2)+'\n');sys.exit(any(r['exit_code'] for r in meta['runs']))
