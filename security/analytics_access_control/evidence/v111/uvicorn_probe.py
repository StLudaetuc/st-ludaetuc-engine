"""Loopback real-Uvicorn author regression; never contacts production or a DB."""
import sys,os,json,hashlib,threading,time,socket,http.client,subprocess,types
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import urlencode
sys.path.insert(0,str(Path.cwd()))
if os.environ.get('DATABASE_URL'):raise RuntimeError('DATABASE_URL must be absent')
import uvicorn,engine
from analytics_access_control import AnalyticsAccessControl
from fastapi.routing import APIRoute
SC={'account_alias':'practice-A','environment':'production','broker_environment':'practice'}
TOKEN='x'*48 # SYNTHETIC ONLY
CFG={'version':1,'identities':[{'key_id':'reader-1','service_id':'synthetic-uvicorn-probe','token_sha256':hashlib.sha256(TOKEN.encode()).hexdigest(),'permissions':['analytics:status','analytics:export','analytics:strategy-report','analytics:prediction-accuracy'],'scopes':[SC],'expires_at':'2026-11-01T00:00:00Z'}]}
DATA=('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy')
DB=[];HANDLERS=[];EFFECTS=[];AUDIT=[]
engine._get_pool=lambda: (DB.append('pool_tripwire') or None)
for r in list(engine.app.routes):
 if r.path in DATA:
  fn=r.dependant.call
  def guarded(*a,_fn=fn,_path=r.path,**kw):HANDLERS.append(_path);return _fn(*a,**kw)
  r.dependant.call=guarded
 elif r.path in ('/evaluate','/market-data','/trade-opened','/trade-outcome'):
  def standin(_path=r.path):EFFECTS.append(_path);return {'synthetic_passthrough':_path}
  engine.app.routes[engine.app.routes.index(r)]=APIRoute(r.path,standin,methods=['POST'])

def query(p):return dict(SC,since_hours='720',lookahead_minutes='240',max_predictions='100') if p.endswith('prediction-accuracy') else dict(SC,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1000')

def probe(root,control=False):
 cls=AnalyticsAccessControl
 if control:
  # Old immutable implementation, loaded only in this local negative control.
  source=subprocess.check_output(['git','show','c035f9d2d74c3eb0d7f2dc1086f2e1095b3bea44:analytics_access_control.py'],text=True)
  m=types.ModuleType('old_negative_control');exec(compile(source,'old_candidate_control','exec'),m.__dict__);cls=m.AnalyticsAccessControl
 for mw in engine.app.user_middleware:
  if mw.cls.__name__=='AnalyticsAccessControl':mw.cls=cls;mw.kwargs.update(config_provider=lambda:json.dumps(CFG),clock=lambda:datetime(2026,10,17,tzinfo=timezone.utc),audit_sink=AUDIT.append)
 engine.app.middleware_stack=None
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 config=uvicorn.Config(engine.app,host='127.0.0.1',port=port,root_path=root,log_level='error',access_log=False,lifespan='on')
 server=uvicorn.Server(config);thread=threading.Thread(target=server.run,kwargs={'sockets':[sock]},daemon=True);thread.start()
 deadline=time.monotonic()+12
 while not server.started and time.monotonic()<deadline:time.sleep(.05)
 if not server.started:raise RuntimeError('LOOPBACK_SERVER_NOT_STARTED')
 results=[]
 def request(path,auth=False,method='GET'):
  before=(len(HANDLERS),len(DB),len(EFFECTS))
  conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
  headers={'Authorization':'Bearer reader-1.'+TOKEN} if auth else {}
  target=path+('?'+urlencode(query(path)) if path in DATA and auth else '')
  conn.request(method,target,headers=headers);response=conn.getresponse();body=response.read();conn.close()
  payload=json.loads(body)
  row={'root_path':root,'wire_path':path,'authenticated':auth,'method':method,'status':response.status,'error':payload.get('error'),'legacy_handler_calls':len(HANDLERS)-before[0],'pool_calls':len(DB)-before[1],'synthetic_trading_effects':len(EFFECTS)-before[2]}
  results.append(row)
  if path in DATA and not control:
   assert response.status==(503 if auth else 401),row
   assert row['error']==('analytics_scope_unavailable' if auth else 'unauthorized'),row
   assert row['legacy_handler_calls']==row['pool_calls']==0,row
  elif control:
   assert row['legacy_handler_calls']==row['pool_calls']==1,row
  else:
   assert response.status==200,row
   assert row['legacy_handler_calls']==row['pool_calls']==0,row
   if method=='POST':assert payload=={'synthetic_passthrough':path} and row['synthetic_trading_effects']==1,row
   else:assert payload['status']=='ok',row
 try:
  for p in DATA:
   request(p)
   if not control:request(p,True)
  if not control:
   request('/health')
   for p in ('/evaluate','/market-data','/trade-opened','/trade-outcome'):request(p,method='POST')
 finally:
  server.should_exit=True;thread.join(timeout=10);sock.close()
  if thread.is_alive():raise RuntimeError('LOOPBACK_SERVER_SHUTDOWN_FAILED')
 return results

if __name__=='__main__':
 records=[]
 for root in ('','/api','/x'):records.extend(probe(root))
 controls=probe('/api',True)
 print(json.dumps({'schema':'author_uvicorn_regression_v111','uvicorn':uvicorn.__version__,'candidate_probe_count':len(records),'old_candidate_negative_controls':len(controls),'candidate_cases':records,'negative_control_cases':controls,'candidate_legacy_handler_calls':sum(x['legacy_handler_calls'] for x in records),'candidate_pool_calls':sum(x['pool_calls'] for x in records),'claim':'loopback real ASGI/HTTP, synthetic trading standins, no real DB or production dependency verification'},indent=2))
