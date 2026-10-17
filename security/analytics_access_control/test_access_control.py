import asyncio,ast,hashlib,json,subprocess,unittest
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
from analytics_access_control import AnalyticsAccessControl
NOW=datetime(2026,10,15,tzinfo=timezone.utc)
SECRET='x'*48 # SYNTHETIC, NOT A CREDENTIAL
SCOPE={'account_alias':'practice-A','environment':'production','broker_environment':'practice'}
CONFIG={'version':1,'identities':[{'key_id':'reader-1','service_id':'research-reader','token_sha256':hashlib.sha256(SECRET.encode()).hexdigest(),'permissions':['analytics:status','analytics:export','analytics:strategy-report','analytics:prediction-accuracy'],'scopes':[SCOPE],'expires_at':'2026-11-01T00:00:00Z'}]}
class Tests(unittest.TestCase):
 def setUp(self):self.events=[];self.calls=[]
 def request(self,path='/analytics/export',auth=True,query=None,method='GET',headers=None,sink=None,config=None,root_path=''):
  sent=[]
  async def downstream(scope,receive,send):
   self.calls.append(scope['path']);await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':b'untouched'})
  app=AnalyticsAccessControl(downstream,lambda:json.dumps(CONFIG) if config is None else config,sink or self.events.append,lambda:NOW)
  async def receive():return {'type':'http.request','body':b'','more_body':False}
  async def send(m):sent.append(m)
  h=headers if headers is not None else [(b'authorization',('Bearer reader-1.'+SECRET).encode())] if auth else []
  q=query if query is not None else dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1000')
  asyncio.run(app({'type':'http','path':path,'root_path':root_path,'method':method,'headers':h,'query_string':urlencode(q,doseq=True).encode()},receive,send))
  return sent[0]['status'],sent[-1].get('body',b''),dict(sent[0]['headers'])
 def test_status_positive(self):
  s,b,h=self.request('/analytics/access-status',query=SCOPE);self.assertEqual(s,200);self.assertEqual(json.loads(b)['scope'],SCOPE);self.assertEqual(h[b'cache-control'],b'no-store');self.assertFalse(self.calls)
 def test_containment(self):
  for p in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy']:
   self.assertEqual(self.request(p,query=SCOPE if p.endswith('accuracy') else None)[0],503)
  self.assertFalse(self.calls)
 def test_all_unauthenticated(self):
  for p in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/access-status','/analytics/new-route','/analytics']:
   self.assertEqual(self.request(p,auth=False)[0],401)
  self.assertFalse(self.calls)
 def test_auth_adversarial(self):
  for h in [[(b'authorization',b'Basic foo')],[(b'authorization',b'Bearer reader-1.short')],[(b'authorization',b'\xff')],[(b'authorization',b'Bearer reader-1.'+b'y'*48)],[(b'authorization',b'Bearer unknown.'+SECRET.encode())],[(b'authorization',b'bad'),(b'Authorization',b'bad')]]:
   self.assertEqual(self.request(headers=h)[0],401)
 def test_query_token_not_auth(self):self.assertEqual(self.request(auth=False,query={'token':SECRET})[0],401)
 def test_invalid_configs(self):
  for c in ['', '{',' '*70000,'{"version":1,"version":1,"identities":[]}']:
   self.assertEqual(self.request(config=c)[0],503)
 def test_unknown_permission_config(self):
  c=json.loads(json.dumps(CONFIG));c['identities'][0]['permissions']=['*'];self.assertEqual(self.request(config=json.dumps(c))[0],503)
 def test_duplicate_identity(self):
  c=json.loads(json.dumps(CONFIG));c['identities']*=2;self.assertEqual(self.request(config=json.dumps(c))[0],503)
 def test_expired(self):
  c=json.loads(json.dumps(CONFIG));c['identities'][0]['expires_at']=NOW.isoformat();self.assertEqual(self.request(config=json.dumps(c))[0],401)
 def test_permission_denied(self):
  c=json.loads(json.dumps(CONFIG));c['identities'][0]['permissions']=['analytics:status'];self.assertEqual(self.request(config=json.dumps(c))[0],403)
 def test_scope_account(self):self.assertEqual(self.request(query=dict(SCOPE,account_alias='practice-B'))[0],403)
 def test_scope_system_environment(self):self.assertEqual(self.request(query=dict(SCOPE,environment='shadow'))[0],403)
 def test_scope_broker_environment(self):self.assertEqual(self.request(query=dict(SCOPE,broker_environment='live'))[0],403)
 def test_missing_scope(self):self.assertEqual(self.request(query={})[0],400)
 def test_missing_window(self):self.assertEqual(self.request(query=SCOPE)[0],400)
 def test_windows(self):
  for a,b in [('2026-01-01T00:00:00Z','2026-10-01T00:00:00Z'),('2026-10-14T00:00:00Z','2026-10-13T00:00:00Z'),('2026-10-15T00:00:00Z','2026-10-16T00:00:00Z'),('2026-10-13','2026-10-14'),('{{time}}','2026-10-14T00:00:00Z')]:
   self.assertEqual(self.request(query=dict(SCOPE,start=a,end=b))[0],400)
 def test_limit_bounds(self):
  for v in ['0','-1','1001','1000000000','nan','1.5']:
   self.assertEqual(self.request(query=dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit=v))[0],400)
 def test_accuracy_bounds(self):
  for k,v in [('since_hours','745'),('lookahead_minutes','1441'),('max_predictions','101'),('max_predictions','0')]:self.assertEqual(self.request('/analytics/prediction-accuracy',query=dict(SCOPE,**{k:v}))[0],400)
 def test_duplicate_query(self):self.assertEqual(self.request(query=list(SCOPE.items())+[('account_alias','practice-B')])[0],400)
 def test_unknown_query(self):self.assertEqual(self.request(query=dict(SCOPE,token=SECRET))[0],400)
 def test_large_query(self):self.assertEqual(self.request(query=dict(SCOPE,instrument='x'*3000))[0],400)
 def test_injection_filter(self):self.assertEqual(self.request(query=dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',instrument="x';SELECT"))[0],400)
 def test_variants(self):
  for p in ['/analytics/export/','//analytics/export','/ANALYTICS/export']:
   self.assertEqual(self.request(p,auth=False)[0],401);self.assertEqual(self.request(p)[0],404)
 def test_methods(self):
  for m in ['POST','PUT','DELETE','OPTIONS']:self.assertEqual(self.request(method=m)[0],405)
  self.assertEqual(self.request(method='HEAD')[1],b'')
 def test_root_path_unauthenticated_containment(self):
  # Uvicorn's root_path may be prepended to path while Starlette strips it.
  for root in ('/api','/x'):
   for p in ('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy'):
    self.assertEqual(self.request(path=root+p,root_path=root,auth=False)[0],401)
  self.assertFalse(self.calls)
 def test_root_path_authenticated_containment(self):
  # Accuracy accepts its own bounded lookahead query, not export dates/limit.
  queries={
   '/analytics/export':dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1000'),
   '/analytics/strategy-report':dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1000'),
   '/analytics/prediction-accuracy':dict(SCOPE,since_hours='720',lookahead_minutes='240',max_predictions='100'),
  }
  for root in ('','/api','/x'):
   for path,query in queries.items():
    with self.subTest(root_path=root,route=path):
     status,body,_=self.request(path=root+path,root_path=root,query=query)
     self.assertEqual(status,503)
     self.assertEqual(json.loads(body)['error'],'analytics_scope_unavailable')
     # Reaching downstream would execute a legacy handler; none may run.
     self.assertFalse(self.calls)
  self.assertFalse(self.calls)
 def test_root_path_unrelated_routes_unchanged(self):
  self.assertEqual(self.request(path='/api/health',root_path='/api',auth=False)[1],b'untouched')
  self.assertEqual(self.calls,['/api/health'])
 def test_root_path_requires_prefix_boundary(self):
  # A root_path which only shares a partial textual prefix must not be stripped.
  self.assertEqual(self.request(path='/apix/health',root_path='/api',auth=False)[1],b'untouched')
 def test_trading_passthrough(self):
  for p in ['/evaluate','/market-data','/trade-opened','/trade-outcome','/health']:
   self.assertEqual(self.request(p,auth=False,method='POST')[1],b'untouched')
  self.assertEqual(len(self.calls),5);self.assertFalse(self.events)
 def test_audit(self):
  self.request(auth=False);self.request();self.request('/analytics/access-status',query=SCOPE);self.assertEqual(len(self.events),3);self.assertEqual(len({x['request_id'] for x in self.events}),3)
 def test_audit_failure(self):
  def fail(e):raise RuntimeError('postgres://password@host')
  s,b,_=self.request('/analytics/access-status',query=SCOPE,sink=fail);self.assertEqual(s,503);self.assertNotIn(b'password',b)
 def test_no_secrets(self):
  _,b,_=self.request();self.assertNotIn(SECRET,b.decode()+json.dumps(self.events));self.assertNotIn('token_sha256',json.dumps(self.events))
 def test_engine_ast_unchanged(self):
  b=ast.parse(Path('engine.py').read_text())
  b.body=[n for n in b.body if not (isinstance(n,ast.ImportFrom) and n.module=='analytics_access_control') and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='add_middleware')]
  # Compare within the executing interpreter: ast.dump varies by Python version.
  baseline = subprocess.run(
   ['git','show','2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c:engine.py'],
   check=True,capture_output=True,text=True,
  ).stdout
  self.assertEqual(ast.dump(b),ast.dump(ast.parse(baseline)))
if __name__=='__main__':unittest.main(verbosity=2)
