"""Author-run equivalent probes; NOT Claude's unavailable 32-test source.

Imports real engine.app with DATABASE_URL absent. Analytics pool access trips.
Trading endpoints are synthetic in-memory stand-ins registered on the real router;
this demonstrates middleware passthrough, not a repeated trading/DB differential.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import os,json,hashlib,unittest
from unittest.mock import patch
from datetime import datetime,timezone
from fastapi.testclient import TestClient
if os.environ.get('DATABASE_URL'):raise RuntimeError('DATABASE_URL must be absent')
import engine
TOKEN='x'*48 # SYNTHETIC TEST ONLY
SC={'account_alias':'practice-A','environment':'production','broker_environment':'practice'}
NOW=datetime(2026,10,17,tzinfo=timezone.utc)
CFG={'version':1,'identities':[{'key_id':'reader-1','service_id':'synthetic-assurance','token_sha256':hashlib.sha256(TOKEN.encode()).hexdigest(),'permissions':['analytics:status','analytics:export','analytics:strategy-report','analytics:prediction-accuracy'],'scopes':[SC],'expires_at':'2026-11-01T00:00:00Z'}]}
DB=[];HANDLERS=[];EFFECTS=[]
engine._get_pool=lambda: (DB.append('tripwire') or None)
original_routes={r.path:r.endpoint for r in engine.app.routes if hasattr(r,'endpoint')}
for route in engine.app.routes:
 if route.path in ('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy'):
  fn=route.dependant.call
  def guarded(*args,_fn=fn,_path=route.path,**kwargs):
   HANDLERS.append(_path)
   return _fn(*args,**kwargs)
  route.dependant.call=guarded
 elif route.path in ('/evaluate','/market-data','/trade-opened','/trade-outcome'):
  def standin(_route=route.path):EFFECTS.append(_route);return {'synthetic_passthrough':_route}
  # Replace route locally, no source modification or DB effects.
  from fastapi.routing import APIRoute
  replacement=APIRoute(route.path,standin,methods=['POST'])
  engine.app.routes[engine.app.routes.index(route)]=replacement
for middleware in engine.app.user_middleware:
 if middleware.cls.__name__=='AnalyticsAccessControl':middleware.kwargs['clock']=lambda:NOW
AUTH={'Authorization':'Bearer reader-1.'+TOKEN}
DATA=('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy')

def valid_query(path):
 return dict(SC,since_hours='720',lookahead_minutes='240',max_predictions='100') if path.endswith('prediction-accuracy') else dict(SC,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1000')

class EquivalentAssurance(unittest.TestCase):
 def setUp(self):DB.clear();HANDLERS.clear();EFFECTS.clear()
 def request(self,root,path,auth=False,method='GET',query=None):
  with patch.dict(os.environ,{'ANALYTICS_SERVICE_IDENTITIES_JSON':json.dumps(CFG)}):
   with TestClient(engine.app,root_path=root) as c:
    DB.clear() # Startup's schema check returns without connecting; test reads after startup.
    return c.request(method,root+path,params=query,headers=AUTH if auth else {},follow_redirects=False)
 def test_30_no_registered_analytics_handler_access(self):
  for root in ('','/api','/x'):
   for path in DATA:
    self.request(root,path);self.request(root,path,True,query=valid_query(path))
  self.assertEqual(DB,[]);self.assertEqual(HANDLERS,[])
 def test_31_prefix_boundary_unrelated(self):
  from analytics_access_control import AnalyticsAccessControl
  seen=[]
  async def next_app(scope,receive,send):seen.append(scope['path'])
  import asyncio
  asyncio.run(AnalyticsAccessControl(next_app)({'type':'http','path':'/apix/health','root_path':'/api'},None,None));self.assertEqual(seen,['/apix/health'])
 def test_32_status_all_roots(self):
  for root in ('','/api','/x'):
   r=self.request(root,'/analytics/access-status',True,query=SC);self.assertEqual(r.status_code,200);self.assertEqual(r.json()['scope'],SC)

# 18 auth/root/route probes + 3 real health probes + 8 trading passthrough probes.
for i,root in enumerate(('','/api','/x')):
 for j,path in enumerate(DATA):
  for auth in (False,True):
   def case(self,root=root,path=path,auth=auth):
    r=self.request(root,path,auth,query=valid_query(path) if auth else None)
    self.assertEqual(r.status_code,503 if auth else 401)
    self.assertEqual(r.json()['error'],'analytics_scope_unavailable' if auth else 'unauthorized')
    self.assertEqual(HANDLERS,[]);self.assertEqual(DB,[])
   setattr(EquivalentAssurance,f'test_route_{i}_{j}_{int(auth)}',case)
 def health(self,root=root):
  r=self.request(root,'/health');self.assertEqual(r.status_code,200);self.assertEqual(r.json()['status'],'ok');self.assertEqual(DB,[])
 setattr(EquivalentAssurance,f'test_health_{i}',health)
for i,root in enumerate(('/api','/x')):
 for j,path in enumerate(('/evaluate','/market-data','/trade-opened','/trade-outcome')):
  def trading(self,root=root,path=path):
   r=self.request(root,path,method='POST');self.assertEqual(r.status_code,200);self.assertEqual(r.json(),{'synthetic_passthrough':path});self.assertEqual(EFFECTS,[path]);self.assertEqual(DB,[])
  setattr(EquivalentAssurance,f'test_trading_{i}_{j}',trading)

if __name__=='__main__':unittest.main(verbosity=2)
