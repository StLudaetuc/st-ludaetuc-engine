"""Isolated real supported-launcher checks. No DB or external service access."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import urllib.error

p=argparse.ArgumentParser();p.add_argument('runtime',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];runtime=a.runtime.resolve();env={'PATH':str(runtime/'bin')+':/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'};results=[]
for label,port,args in [('missing',None,[]),('zero','0',[]),('too-high','65536',[]),('negative','-1',[]),('text','10000;bad',[]),('argument','10000',['extra'])]:
 e=dict(env)
 if port is not None:e['PORT']=port
 r=subprocess.run(['sh','release/launch.sh']+args,cwd=root,env=e,capture_output=True,text=True,timeout=5);results.append({'case':label,'exit':r.returncode,'stderr':r.stderr,'pass':r.returncode!=0})
s=socket.socket();s.bind(('0.0.0.0',0));port=s.getsockname()[1];s.close();env['PORT']=str(port)
log=a.out.with_suffix('.server.log');f=log.open('w');proc=subprocess.Popen(['sh','release/launch.sh'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def get(url):
 try:
  with client.open(url,timeout=2) as r:return r.status,r.read().decode()
 except urllib.error.HTTPError as e:return e.code,e.read().decode()
 except urllib.error.URLError as e:return None,str(e.reason)
try:
 for _ in range(100):
  if proc.poll() is not None:raise RuntimeError('launcher exited; see '+str(log))
  try:
   if get(f'http://127.0.0.1:{port}/health')[0]==200:break
  except OSError:pass
  time.sleep(.1)
 else:raise RuntimeError('startup timeout')
 ip=next(x for x in socket.gethostbyname_ex(socket.gethostname())[2] if not x.startswith('127.') and ':' not in x)
 for path in ['/health','/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/export%0A']:
  local_status,_=get(f'http://127.0.0.1:{port}'+path)
  results.append({'case':'loopback'+path,'status':local_status,'pass':local_status==200 if path=='/health' else local_status in [401,503]})
  status,body=get(f'http://{ip}:{port}'+path);ok=status==200 if path=='/health' else status in [401,503]
  results.append({'case':'nonloopback'+path,'ip':ip,'status':status,'pass':ok,'body':body if path=='/health' or status is None else 'denied-body-not-retained'})
finally:
 proc.terminate()
 try:proc.wait(5)
 except subprocess.TimeoutExpired:proc.kill();proc.wait()
 f.close()
a.out.write_text(json.dumps(results,indent=2)+'\n');print(json.dumps({'checks':len(results),'failed':sum(not x['pass'] for x in results)}));raise SystemExit(any(not x['pass'] for x in results))
