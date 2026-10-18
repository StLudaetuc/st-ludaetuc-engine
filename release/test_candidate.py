"""AUTHOR EVIDENCE: fallback C2/C3 controls with the actual entrypoint."""
import asyncio, contextlib, importlib, io, json, unittest
from unittest.mock import patch
from security.analytics_release.containment_fallback import ContainmentOnly
class CandidateTests(unittest.TestCase):
 def invoke(self,path,method='GET',kind='http',sink=None):
  entry=importlib.import_module('security.analytics_release.fallback_app')
  out=[]; hit=[]
  async def downstream(s,r,w):hit.append(s)
  async def receive():return {'type':'http.request','body':b''}
  async def send(x):out.append(x)
  stream=sink or io.StringIO()
  with patch.object(entry.app,'app',downstream),contextlib.redirect_stderr(stream):
   asyncio.run(entry.app({'type':kind,'path':path,'root_path':'','method':method,'headers':[(b'authorization',b'SYNTHETIC_PRIVATE')],'query_string':b'secret=SYNTHETIC_PRIVATE'},receive,send))
  return out,hit,stream
 def test_real_entrypoint_suffixes_and_non_string_paths(self):
  for path in ['/analytics/export\n','/analytics/export%0A','/analytics/access-status',None,1,b'/analytics/export',{},[]]:
   with self.subTest(path=path):
    out,hit,stream=self.invoke(path);self.assertFalse(hit);self.assertEqual(out[0]['status'],503);self.assertEqual(len(stream.getvalue().splitlines()),1)
 def test_audit_request_id_redaction_utc(self):
  for method in ['GET','HEAD','OPTIONS']:
   out,_,stream=self.invoke('/analytics/export',method);e=json.loads(stream.getvalue());self.assertEqual(e['event'],'analytics_containment');self.assertTrue(e['at'].endswith('+00:00'));self.assertNotIn('SYNTHETIC_PRIVATE',stream.getvalue());self.assertEqual(e['request_id'],dict(out[0]['headers'])[b'x-request-id'].decode());self.assertEqual(e['status'],503)
 def test_ws_audit_before_accept(self):
  out,hit,stream=self.invoke('/analytics/export',kind='websocket');self.assertEqual(out,[{'type':'websocket.close','code':1008}]);self.assertFalse(hit);self.assertEqual(json.loads(stream.getvalue())['status'],1008)
 def test_broken_log_transport_denies(self):
  class Broken:
   def write(self,*a):raise OSError('SYNTHETIC_SINK_FAILURE')
   def flush(self):raise OSError('SYNTHETIC_SINK_FAILURE')
  out,hit,_=self.invoke('/analytics/export',sink=Broken());self.assertFalse(hit);self.assertEqual(out[0]['status'],503)
 def test_denial_burst_exactly_once(self):
  for _ in range(200):
   _,hit,stream=self.invoke('/analytics/export');self.assertFalse(hit);self.assertEqual(len(stream.getvalue().splitlines()),1)
if __name__=='__main__':unittest.main(verbosity=2)
