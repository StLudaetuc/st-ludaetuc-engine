"""AUTHOR EVIDENCE: real entrypoint semantics and exact fallback audit counts."""
import contextlib,io,json,unittest
from fastapi.testclient import TestClient
from security.analytics_release.fallback_app import app
class LocalProbe(unittest.TestCase):
 def test_literal_fallback_probes(self):
  stream=io.StringIO()
  with contextlib.redirect_stderr(stream),TestClient(app) as c:
   self.assertEqual(c.get('/health').status_code,200)
   for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/access-status']:
    r=c.get(path);self.assertEqual(r.status_code,503);self.assertEqual(r.json()['error'],'analytics_containment_active');self.assertEqual(r.headers['x-request-id'],r.json()['request_id'])
   self.assertEqual(c.head('/analytics/export').content,b'');self.assertEqual(c.options('/analytics/export').status_code,503)
   from starlette.websockets import WebSocketDisconnect
   with self.assertRaises(WebSocketDisconnect):
    with c.websocket_connect('/analytics/export'):pass
  lines=[json.loads(x) for x in stream.getvalue().splitlines() if x.startswith('{')]
  self.assertEqual(len(lines),7);self.assertTrue(all(e['event']=='analytics_containment' for e in lines));print(json.dumps({'claim':'AUTHOR EVIDENCE','fallback_denials':7,'audit_lines':lines}))
if __name__=='__main__':unittest.main(verbosity=2)
