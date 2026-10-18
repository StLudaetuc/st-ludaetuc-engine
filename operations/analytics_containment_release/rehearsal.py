"""Local-only supporting engineering rehearsal. No Render acceptance or assurance."""
import asyncio
import contextlib
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import tempfile
import unittest
from urllib.parse import urlencode

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT))
if any(os.environ.get(k) for k in ('DATABASE_URL','PGSERVICE','PGHOST','PGPASSWORD','ANALYTICS_SERVICE_IDENTITIES_JSON')):
    raise RuntimeError('Use documented clean environment; inherited connection/identity settings prohibited')
import psycopg2
import psycopg2.pool

def blocked(*a, **kw):
    raise AssertionError('DATABASE_CONNECTION_PROHIBITED')
psycopg2.connect = blocked
psycopg2.pool.ThreadedConnectionPool = blocked
import engine
from analytics_access_control import AnalyticsAccessControl
from security.analytics_release.containment_fallback import ContainmentOnly
from fastapi.testclient import TestClient
import uvicorn

BASE = '2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c'
DATA = ('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy')
TRADING = ('/evaluate','/market-data','/trade-opened','/trade-outcome')
SC = dict(account_alias='synthetic-A',environment='production',broker_environment='practice')
TOKEN = 'S'*48 # public synthetic test fixture, never a production identity
CFG = {'version':1,'identities':[dict(key_id='synthetic',service_id='local-rehearsal',token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),permissions=['analytics:export','analytics:strategy-report','analytics:prediction-accuracy','analytics:status'],scopes=[SC],expires_at='2099-01-01T00:00:00Z')]}
RECORDS = []
AUDIT = []
HANDLERS = []
engine._get_pool = blocked
for r in engine.app.routes:
    if r.path in DATA:
        def forbidden(*a, _path=r.path, **kw):
            HANDLERS.append(_path)
            raise AssertionError('LEGACY_HANDLER_PROHIBITED')
        r.dependant.call = forbidden

def configure(mode='primary', audit_fail=False):
    engine.app.middleware_stack = None
    for m in engine.app.user_middleware:
        if m.cls.__name__ == 'AnalyticsAccessControl':
            m.kwargs['config_provider'] = lambda: json.dumps(CFG)
            def sink(e):
                if audit_fail: raise OSError('synthetic audit failure')
                AUDIT.append(e)
            m.kwargs['audit_sink'] = sink
    return ContainmentOnly(engine.app) if mode == 'fallback' else engine.app

def query(p):
    if p.endswith('prediction-accuracy'):
        return dict(SC,since_hours='24',lookahead_minutes='60',max_predictions='10')
    return dict(SC,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z',limit='1')

def request(port,path,auth=False,method='GET',payload=None):
    target=path+('?' + urlencode(query(path)) if auth else '')
    headers={'Authorization':'Bearer synthetic.'+TOKEN} if auth else {}
    body=None
    if payload is not None: body=json.dumps(payload);headers['Content-Type']='application/json'
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    c.request(method,target,body,headers);r=c.getresponse();data=r.read();c.close()
    return r.status, json.loads(data) if data else None

@contextlib.contextmanager
def server(app,root=''):
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    config=uvicorn.Config(app,host='127.0.0.1',port=port,root_path=root,lifespan='on',access_log=False,log_level='error')
    srv=uvicorn.Server(config);t=threading.Thread(target=srv.run,kwargs={'sockets':[sock]},daemon=True);t.start()
    deadline=time.monotonic()+8
    while not srv.started and t.is_alive() and time.monotonic()<deadline:time.sleep(.02)
    try:
        if not srv.started:raise RuntimeError('LOCAL_STARTUP_FAILED')
        yield port
    finally:
        srv.should_exit=True;t.join(8);sock.close()
        if t.is_alive():raise RuntimeError('LOCAL_SHUTDOWN_FAILED')

PAYLOADS={
 '/evaluate':{'payload':{'meta':{'signal_id':'synthetic-signal'},'system':{'strategy_id':'synthetic','strategy_version':'1'},'instrument':{'symbol':'GBPUSD'},'market':{'timeframe':'2m','bar_time_utc':'2026-10-13T12:00:00Z'},'signal':{'direction':'long','entry_model':'trend_pullback'},'price':{'close':1.3},'indicators':{'atr':.001}},'account':{'balance':10000,'equity':10000}},
 '/market-data':{'ticker':'OANDA:GBPUSD','interval':'2','time':'2026-10-13T12:00:00Z','open':1.3,'close':1.3},
 '/trade-opened':{'trade_id':'synthetic-signal','instrument':'GBPUSD','opened_at':'2026-10-13T12:00:00Z','oanda_order_response':{'orderFillTransaction':{'id':'synthetic-fill','tradeOpened':{'tradeID':'synthetic-trade'},'time':'2026-10-13T12:00:00Z'}}},
 '/trade-outcome':{'trade_id':'synthetic-signal','instrument':'GBPUSD','closed_at':'2026-10-13T12:10:00Z','outcome':'win','realized_pnl':1.25},
}

def fake_logger(calls):
    def log(**kw):calls.append(kw);return 'synthetic-event'
    return log

class OperationsRehearsal(unittest.TestCase):
    def test_01_real_http_primary_and_fallback_roots(self):
        for mode in ('primary','fallback'):
            for root in ('','/api','/x'):
                with server(configure(mode),root) as port:
                    for p in DATA:
                        for auth in (False,True):
                            status,body=request(port,p,auth)
                            self.assertEqual(status,503 if mode=='fallback' or auth else 401)
                            self.assertEqual(body['error'],'analytics_containment_active' if mode=='fallback' else 'analytics_scope_unavailable' if auth else 'unauthorized')
                            RECORDS.append(dict(mode=mode,root=root,path=p,auth=auth,status=status,error=body['error']))
                    for p in ('//analytics//export','/ANALYTICS/export',root+'/analytics/export' if root else '/analytics/export/'):
                        status,body=request(port,p)
                        # Some doubled/root-prefixed wire variants do not resolve a route.
                        self.assertIn(status,(401,503,404))
                        self.assertEqual(HANDLERS,[])
                        RECORDS.append(dict(mode=mode,root=root,path=p,status=status,error=body.get('error') if isinstance(body,dict) else None,variant='noncanonical; 404 is routing rejection, never dataset'))
                    self.assertEqual(request(port,'/health')[0],200)
        self.assertEqual(HANDLERS,[])
        self.assertTrue(AUDIT)
        self.assertTrue(all(e['event']=='analytics_access' for e in AUDIT))

    def test_02_actual_trading_handlers_match_baseline(self):
        # Baseline engine loaded only in memory, never started as a public service.
        src=subprocess.check_output(['git','show',BASE+':engine.py'],cwd=ROOT,text=True)
        import types
        baseline=types.ModuleType('synthetic_baseline');sys.modules[baseline.__name__]=baseline
        exec(compile(src,'immutable_baseline_engine','exec'),baseline.__dict__)
        baseline._get_pool=blocked
        expected=[]
        for app,module,label in ((baseline.app,baseline,'baseline'),(configure(),engine,'primary'),(configure('fallback'),engine,'fallback')):
            calls=[];module.log_event=fake_logger(calls)
            with TestClient(app) as c:
                outputs=[]
                for p in TRADING:
                    r=c.post(p,json=PAYLOADS[p]);self.assertEqual(r.status_code,200,(p,r.text));outputs.append(r.json())
                # Invalid request handling also remains equivalent.
                outputs.append(c.post('/market-data',content='[').status_code)
                outputs.append(c.post('/trade-outcome',json={}).status_code)
            if label=='baseline':expected=[outputs,calls]
            else:self.assertEqual([outputs,calls],expected)
        RECORDS.append(dict(test='actual_trading_baseline_differential',routes=4,variants=3,db='in-memory log_event stub; no SQL effects tested'))

    def test_03_real_http_trading_handlers(self):
        engine.log_event=fake_logger([])
        for mode in ('primary','fallback'):
            with server(configure(mode)) as port:
                from operations.analytics_containment_release.verify_http import verify
                self.assertTrue(verify('http://127.0.0.1:'+str(port),mode)['passed'])
                for p in TRADING:self.assertEqual(request(port,p,method='POST',payload=PAYLOADS[p])[0],200)

    def test_04_audit_failure_fails_closed(self):
        with TestClient(configure(audit_fail=True)) as c:
            r=c.get(DATA[0]);self.assertEqual(r.status_code,503);self.assertEqual(r.json()['error'],'audit_unavailable')
        self.assertEqual(HANDLERS,[])

    def test_05_fallback_covers_primary_bypass(self):
        async def bypass(scope,receive,send):
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'synthetic legacy marker'})
        c=TestClient(ContainmentOnly(bypass)) # no lifespan support in this synthetic inner app
        self.assertEqual(c.get(DATA[0]).status_code,503)
        self.assertEqual(c.get('/health').status_code,200)

    def test_06_fallback_covers_inner_middleware_failure(self):
        async def broken(scope,receive,send):raise RuntimeError('synthetic inner middleware failure')
        # Avoid lifespan, deliberately exercise request failure only.
        c=TestClient(ContainmentOnly(broken),raise_server_exceptions=False)
        self.assertEqual(c.get(DATA[0]).status_code,503)
        self.assertEqual(c.get('/health').status_code,500)

    def test_07_fallback_cannot_fix_startup_failure(self):
        async def startup_broken(scope,receive,send):
            if scope['type']=='lifespan':
                await receive();await send({'type':'lifespan.startup.failed','message':'synthetic startup failure'});return
            raise RuntimeError('not started')
        for app in (startup_broken,ContainmentOnly(startup_broken)):
            with self.assertRaisesRegex(RuntimeError,'LOCAL_STARTUP_FAILED'):
                with server(app):self.fail('must not become ready')

    def test_08_fallback_websocket_and_head(self):
        with TestClient(configure('fallback')) as c:
            self.assertEqual(c.head(DATA[0]).content,b'')
            from starlette.websockets import WebSocketDisconnect
            with self.assertRaises(WebSocketDisconnect) as result:
                with c.websocket_connect(DATA[0]):pass
            self.assertEqual(result.exception.code,1008)

    def test_09_default_audit_stdout(self):
        from analytics_access_control import emit_audit
        output=io.StringIO()
        e=dict(event='analytics_access',request_id='synthetic',http_status=401,result='unauthorized')
        with contextlib.redirect_stdout(output):emit_audit(e)
        self.assertEqual(json.loads(output.getvalue()),e)
        self.assertNotIn(TOKEN,output.getvalue())

    def test_10_env_root_path_and_cli_override(self):
        for mode in ('primary','fallback'):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','UVICORN_ROOT_PATH':'/api','OPS_REHEARSAL_MODE':mode}
            cmd=[sys.executable,'-m','uvicorn','operations.analytics_containment_release.local_entry:create_app','--factory','--host','127.0.0.1','--port',str(port),'--log-level','info']
            with tempfile.TemporaryFile() as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                ready=False
                try:
                    deadline=time.monotonic()+15
                    while time.monotonic()<deadline:
                        if proc.poll() is not None:break
                        try:
                            if request(port,'/health')[0]==200:ready=True;break
                        except OSError:pass
                        time.sleep(.05)
                    if ready:
                        self.assertEqual(request(port,DATA[0])[0],503 if mode=='fallback' else 401)
                finally:
                    if proc.poll() is None:proc.terminate()
                    proc.wait(8)
                    log.seek(0);content=log.read().decode()
                    RECORDS.append(dict(test='real_uvicorn_cli',mode=mode,root_path_env='/api',exit_code=proc.returncode,startup=ready,log=content))
                self.assertTrue(ready,content)
                self.assertIn('/api/analytics/export',content)
                self.assertIn('Application startup complete',content)
                if mode=='primary':self.assertIn('analytics_access',content)

    def test_11_health_is_not_database_readiness(self):
        with TestClient(configure()) as c:
            body=c.get('/health').json()
            self.assertEqual(body['status'],'ok');self.assertFalse(body['events_db_configured'])
        RECORDS.append(dict(test='health_limit',finding='200 health with no configured DB; not DB readiness'))

    def test_12_outer_failure_is_not_a_safe_fallback(self):
        async def outer_broken(scope,receive,send):raise RuntimeError('synthetic outer boundary failure')
        c=TestClient(outer_broken,raise_server_exceptions=False)
        self.assertEqual(c.get(DATA[0]).status_code,500)
        RECORDS.append(dict(test='failure_limit',finding='fallback failure or engine startup failure requires separately approved incident response; no safe old deployment'))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OperationsRehearsal)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'schema':'operations_rehearsal_v1','cases':RECORDS,'audit_events':len(AUDIT),'legacy_handler_calls':len(HANDLERS),'claim':'author supporting local synthetic evidence; not independent fallback approval or Render deployment acceptance'},indent=2))
    sys.exit(not result.wasSuccessful())
