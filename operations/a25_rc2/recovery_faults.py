"""AUTHOR deployment-state fault rehearsal; clean tree/runtime, not OCI/PG proof."""
import argparse,json,os,pathlib,shutil,socket,subprocess,tempfile,time,http.client
p=argparse.ArgumentParser();p.add_argument('rb',type=pathlib.Path);p.add_argument('runtime',type=pathlib.Path);p.add_argument('out',type=pathlib.Path);a=p.parse_args();rb=a.rb.resolve();runtime=a.runtime.resolve();rows=[]
EVAL={'payload':{'meta':{'signal_id':'SYNTH-FI-1'},'system':{'strategy_id':'STL_SYNTH'},'instrument':{'symbol':'EURUSD'},'market':{'timeframe':'2','bar_time_utc':'2026-10-18T05:00:00Z'},'signal':{'direction':'long','entry_model':'trend_pullback'}},'account':{'balance':10000,'currency':'USD','nav':10000}}
def check(tree,venv,label):
 s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();f=a.out.with_name(label+'.log').open('w');proc=subprocess.Popen(['sh','release/launch.sh'],cwd=tree,env={'PATH':str(venv/'bin')+':/usr/bin:/bin','PORT':str(port),'PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT);result={'case':label,'ready':False,'persistence':'NOT_TESTED_NO_DB'}
 def req(method,path,body=None):
  c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request(method,path,body=json.dumps(body) if body else None,headers={'Content-Type':'application/json'});r=c.getresponse();r.read();c.close();return r.status
 try:
  for _ in range(60):
   if proc.poll() is not None:break
   try:
    if req('GET','/health')==200:result['ready']=True;break
   except OSError:pass
   time.sleep(.05)
  if result['ready']:result.update(health=req('GET','/health'),evaluate=req('POST','/evaluate',EVAL),analytics=req('GET','/analytics/export'))
  result['exit_before_stop']=proc.poll()
 finally:
  proc.terminate()
  try:proc.wait(3)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  f.close()
 return result
with tempfile.TemporaryDirectory(prefix='rc2-recovery-fault-') as td:
 d=pathlib.Path(td)
 for name in ['S1-engine-syntax','S2-deployed-dependency-drift','S5-post-build-trading-corruption']:
  tree=d/name;shutil.copytree(rb,tree,ignore=shutil.ignore_patterns('.git','__pycache__'));venv=runtime
  if name.startswith('S1'):
   with (tree/'engine.py').open('a') as f:f.write('\nthis is ! syntax\n')
  elif name.startswith('S2'):
   venv=d/'broken-runtime';shutil.copytree(runtime,venv,symlinks=True);file=next(venv.glob('lib/python*/site-packages/starlette/__init__.py'));file.write_text(file.read_text()+'\n# SYNTHETIC_DEPLOYMENT_DRIFT\n')
  else:
   with (tree/'engine.py').open('a') as f:f.write("\nengine.evaluate = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('SYNTHETIC_TRADING_FAULT'))\n")
  broken=check(tree,venv,name);clean=check(rb,runtime,name+'-clean-retained-state');rows.append({'fault':name,'broken':broken,'clean':clean,'deployment_response_restored':clean.get('evaluate')==200 and clean.get('analytics')==503 and (not broken['ready'] or broken.get('evaluate')!=200)})
rows.append({'fault':'S3/S4/shared-source-dependency','result':'NOT_RECOVERABLE_BY_CLAIM; actual PG/shared-source matrix not executed here'})
a.out.write_text(json.dumps({'label':'AUTHOR clean directory/runtime deployment-domain rehearsal; NO image/PG/Production claim','results':rows},indent=2)+'\n');print(json.dumps({'deployment_cases':3,'response_restored':sum(x.get('deployment_response_restored',False) for x in rows),'persistent_trading_continuity':'NOT_DEMONSTRATED'}));raise SystemExit(not all(x.get('deployment_response_restored',True) for x in rows))
