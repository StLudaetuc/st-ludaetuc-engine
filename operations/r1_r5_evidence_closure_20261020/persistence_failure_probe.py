"""SUPPLEMENTAL AUTHOR real connection-refusal probe; no PostgreSQL server."""
import socket,subprocess,http.client,json,time,os
from pathlib import Path
root=Path(__file__).absolute().parents[1];ws=root.parent;out=[]
# Bind but do not listen: reserve a loopback-only port that deterministically refuses TCP.
refused=socket.socket();refused.bind(('127.0.0.1',0));badport=refused.getsockname()[1]
for mode,tree,entry in [('candidate',ws/'candidate-checkout','app:app'),('fallback',ws/'candidate-checkout','security.analytics_release.fallback_app:app'),('RB',ws/'recovery-checkout','release.recovery_entry:app')]:
 s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','DATABASE_URL':f'postgresql://synthetic@127.0.0.1:{badport}/synthetic?connect_timeout=1'}
 with (root/f'logs/{mode}-actual-connect-refused.log').open('wb') as f:
  p=subprocess.Popen([str(ws/'envG/bin/python'),'-m','uvicorn',entry,'--host','127.0.0.1','--port',str(port)],cwd=tree,env=env,stdout=f,stderr=subprocess.STDOUT)
  try:
   for _ in range(100):
    try:
     c=http.client.HTTPConnection('127.0.0.1',port,timeout=2);c.request('GET','/health');r=c.getresponse();health=dict(status=r.status,body=json.loads(r.read()));c.close();break
    except OSError:time.sleep(.05)
   else:raise RuntimeError('startup failed')
   c=http.client.HTTPConnection('127.0.0.1',port,timeout=3);c.request('POST','/market-data',body=json.dumps({'ticker':'OANDA:GBPUSD','interval':'1','time':'2026-10-20T00:00:00Z','close':1.3}),headers={'Content-Type':'application/json'});r=c.getresponse();market=dict(status=r.status,body=json.loads(r.read()));c.close()
   c=http.client.HTTPConnection('127.0.0.1',port,timeout=2);c.request('GET','/analytics/export');r=c.getresponse();analytics=r.status;r.read();c.close()
   assert health['status']==200 and market['status']==500
   out.append(dict(mode=mode,health=health,market_data=market,analytics_status=analytics,persistence='FAILED: real loopback TCP connection refused; no rows inserted; no PostgreSQL server connected'))
  finally:p.terminate();p.wait(timeout=10)
refused.close();(root/'actual-connect-refusal-results.json').write_text(json.dumps({'claim':'SUPPLEMENTAL AUTHOR EVIDENCE; true TCP refusal, not a PG integration test','results':out},indent=2));print([(x['mode'],x['health']['status'],x['market_data']['status']) for x in out])
