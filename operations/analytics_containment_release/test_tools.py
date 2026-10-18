import json,unittest
from unittest.mock import patch
from operations.analytics_containment_release import release_gate as gate
from operations.analytics_containment_release.audit_summary import summarise
class ToolTests(unittest.TestCase):
 def test_missing_approval_holds(self):self.assertTrue(gate.check('.',{}))
 def test_forbidden_rollback(self):
  for s in gate.FORBIDDEN:self.assertTrue(gate.check('.',{'release_sha':s,'fallback_sha':s}))
 def test_empty_logs_unknown(self):self.assertIn('not zero traffic',summarise([])['coverage'])
 def test_data_success_detected(self):self.assertEqual(summarise([json.dumps(dict(event='analytics_access',http_status=200,route='/analytics/export',result='access_status'))])['unexpected_data_2xx'],1)
 def test_status_success_not_data_leak(self):self.assertEqual(summarise([json.dumps(dict(event='analytics_access',http_status=200,route='/analytics/access-status',result='access_status'))])['unexpected_data_2xx'],0)
 def test_denials_counted(self):self.assertEqual(summarise([json.dumps(dict(event='analytics_access',http_status=401,route='/analytics/export',result='unauthorized'))])['valid_analytics_events'],1)
 def test_malformed_not_silent(self):self.assertEqual(summarise(['bad'])['unclassified_lines'],1)
 def test_unknown_result_unclassified(self):self.assertEqual(summarise([json.dumps(dict(event='analytics_access',http_status=401,route='/analytics/export',result='NEW'))])['unclassified_lines'],1)
class ProbeSafetyTests(unittest.TestCase):
 def test_remote_default_denied(self):
  from operations.analytics_containment_release.verify_http import target
  with self.assertRaises(ValueError):target('https://st-ludaetuc-engine.onrender.com')
 def test_unrelated_remote_always_denied(self):
  from operations.analytics_containment_release.verify_http import target
  with self.assertRaises(ValueError):target('https://example.com',True)
 def test_url_secrets_rejected(self):
  from operations.analytics_containment_release.verify_http import target
  for url in ('http://user:pass@localhost:123','http://localhost:123/?token=fake'):
   with self.assertRaises(ValueError):target(url)
 def test_success_body_never_read(self):
  from operations.analytics_containment_release.verify_http import verify
  from unittest.mock import MagicMock,patch
  c=MagicMock();c.getresponse.return_value.status=200
  with patch('http.client.HTTPConnection',return_value=c):r=verify('http://127.0.0.1:1234')
  self.assertFalse(r['passed']);self.assertEqual(len(r['requests']),3);c.getresponse.return_value.read.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
