"""AUTHOR supplemental 236-call differential; log_event stub, NOT PostgreSQL.
Reconstructed author fixture, not the unavailable original reviewer fixture.
"""
import ast,copy,hashlib,json,subprocess,sys,types
from pathlib import Path
from fastapi.testclient import TestClient
ROOT=Path(__file__).absolute().parents[1];sys.path.insert(0,str(ROOT))
import engine
from security.analytics_release.containment_fallback import ContainmentOnly
BASE='2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c'
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),default=lambda x:x.isoformat() if hasattr(x,'isoformat') else str(x))
fixtures=[]
for node in ast.parse((ROOT/'operations/analytics_containment_release/rehearsal.py').read_text()).body:
 if isinstance(node,ast.Assign):
  try:
   val=ast.literal_eval(node.value)
   if isinstance(val,dict) and '/evaluate' in val:payloads=val
  except (ValueError,TypeError):pass
fixtures=[('GET','/health',None)]*20
for i in range(54):
 for path,payload in payloads.items():
  data=copy.deepcopy(payload);data['synthetic_sequence']=i
  fixtures.append(('POST',path,data))
assert len(fixtures)==236
src=subprocess.check_output(['git','-C',str(ROOT),'show',BASE+':engine.py']).decode();base=types.ModuleType('author_baseline');sys.modules[base.__name__]=base;exec(compile(src,'immutable_baseline_engine','exec'),base.__dict__)
def run(mod,app):
 events=[]
 def log(**kw):events.append(copy.deepcopy(kw));return kw.get('event_id') or f'synthetic-event-{len(events)}'
 mod.log_event=log
 def no_db():raise AssertionError('NO_DATABASE_CONNECTION_ALLOWED')
 mod._get_pool=no_db;responses=[]
 with TestClient(app) as c:
  for method,path,body in fixtures:
   r=c.request(method,path,json=body) if body is not None else c.request(method,path)
   responses.append({'status':r.status_code,'body':r.json()})
 return dict(responses=responses,event_records=json.loads(canon(events)),events_hash=hashlib.sha256(canon(events).encode()).hexdigest(),schema_hash=hashlib.sha256((mod._EVENTS_SCHEMA_SQL+mod._REFERENCE_SCHEMA_SQL).encode()).hexdigest())
if __name__=='__main__':
 result={'baseline':run(base,base.app),'primary':run(engine,engine.app),'fallback':run(engine,ContainmentOnly(engine.app))}
 diffs={k:sum(a!=b for a,b in zip(v['responses'],result['baseline']['responses'])) for k,v in result.items()}
 print(json.dumps(dict(claim='AUTHOR EVIDENCE; mock persistence, not original reviewer fixture',fixture_sha256=hashlib.sha256(canon(fixtures).encode()).hexdigest(),request_count=len(fixtures),ordering='20 health then 54 rounds of 4 trading routes',normalized_volatile_fields=[],excluded_routes=['/analytics/*','/docs','/openapi.json'],response_differences=diffs,modes=result,A13='INCOMPLETE: real PG and original fixture unavailable'),indent=2))
