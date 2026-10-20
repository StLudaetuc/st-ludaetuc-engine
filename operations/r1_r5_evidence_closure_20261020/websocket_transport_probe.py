"""SUPPLEMENTAL AUTHOR raw WebSocket handshake, no PG/database claims."""
import subprocess,socket,time,http.client,json
from pathlib import Path
r=Path(__file__).absolute().parents[1];ws=r.parent;results=[]
for label in ['G','R']:
 for mode,tree,entry in [('primary',ws/'candidate-checkout','app:app'),('fallback',ws/'candidate-checkout','security.analytics_release.fallback_app:app'),('RB',ws/'recovery-checkout','release.recovery_entry:app')]:
  s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
  with (r/f'logs/{label}-{mode}-ws.log').open('wb') as f:
   proc=subprocess.Popen([str(r/f'env-{label}/bin/python'),'-m','uvicorn',entry,'--ws','websockets','--host','127.0.0.1','--port',str(port)],cwd=tree,env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
   try:
    for _ in range(100):
     try:
      c=http.client.HTTPConnection('127.0.0.1',port,timeout=1);c.request('GET','/health');z=c.getresponse();z.read();c.close();break
     except OSError:time.sleep(.05)
    else:raise RuntimeError('startup failure')
    probes=[]
    for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/access-status']:
     s=socket.create_connection(('127.0.0.1',port),timeout=2);s.sendall((f'GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n').encode());b=s.recv(65536);s.close();line=b.split(b'\r\n')[0].decode();assert ' 403 ' in line and ' 101 ' not in line;probes.append(dict(path=path,status_line=line,accepted=False))
   finally:proc.terminate();proc.wait(timeout=10)
  results.append(dict(set=label,mode=mode,probes=probes))
(r/'websocket-transport-results.json').write_text(json.dumps(dict(claim='SUPPLEMENTAL AUTHOR EVIDENCE; 17.2 test-only websockets wheel; no PG, not original full matrix',results=results),indent=2));print('handshakes denied before 101',sum(len(x['probes']) for x in results))
