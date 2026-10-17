"""Run the candidate in a clean child process; no production variables inherited."""
import argparse,subprocess,os,json,hashlib,platform
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--label',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[4];out=root/'research/historical_recovery/evidence/v11'
command=[str(Path(a.python).absolute()),'-W','default','-m','unittest','discover','-s','research/historical_recovery/tests','-v']
env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'}
source={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'research/historical_recovery').rglob('*.py') if '__pycache__' not in f.parts}
started=datetime.now(timezone.utc).isoformat()
with (out/(a.label+'.log')).open('wb') as log:r=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
metadata={'command':command,'exit_code':r.returncode,'started_at':started,'ended_at':datetime.now(timezone.utc).isoformat(),'os':platform.platform(),'environment_keys':list(env),'production_environment_inherited':False,'source_sha256':source}
(out/(a.label+'.metadata.json')).write_text(json.dumps(metadata,indent=2));print(json.dumps({'label':a.label,'exit_code':r.returncode}));raise SystemExit(r.returncode)
