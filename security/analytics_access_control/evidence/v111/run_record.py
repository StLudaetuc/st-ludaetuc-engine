import json,os,platform,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
REPO=BASE/'repo'; E=REPO/'security/analytics_access_control/evidence/v111'

def run(python,label,args):
 env={'PATH':'/usr/bin:/bin','HOME':str(BASE/'runner-home'),'PYTHONDONTWRITEBYTECODE':'1'}
 started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 command=[str(Path(python).absolute()),'-W','default']+args
 with (E/(label+'.log')).open('wb') as output:
  r=subprocess.run(command,cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=120)
 metadata={'label':label,'started_at':started,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'command':command,'cwd':'repository_root','exit_code':r.returncode,'environment_keys':sorted(env),'production_environment_inherited':False,'os':platform.platform()}
 (E/(label+'.metadata.json')).write_text(json.dumps(metadata,indent=2))
 print(json.dumps(metadata))
 return r.returncode
if __name__=='__main__':run(sys.argv[1],sys.argv[2],sys.argv[3:])
