"""Real FastAPI integration without importing engine or running startup DDL."""
import unittest,json,hashlib
from datetime import datetime,timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from analytics_access_control import AnalyticsAccessControl
from test_access_control import CONFIG,SCOPE,SECRET,NOW
class FastAPITests(unittest.TestCase):
 def setUp(self):
  self.events=[];self.hit=[];app=FastAPI()
  app.add_middleware(AnalyticsAccessControl,config_provider=lambda:json.dumps(CONFIG),audit_sink=self.events.append,clock=lambda:NOW)
  for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy']:
   def handler():self.hit.append('SENSITIVE_HANDLER');return {'password':'must_never_be_visible'}
   app.add_api_route(path,handler,methods=['GET'])
  @app.post('/evaluate')
  def evaluate():self.hit.append('evaluate');return {'unchanged':True}
  self.client=TestClient(app);self.auth={'Authorization':'Bearer reader-1.'+SECRET}
 def test_unauthenticated_all_routes(self):
  for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy']:
   self.assertEqual(self.client.get(path).status_code,401)
  self.assertFalse(self.hit)
 def test_authorized_status(self):
  r=self.client.get('/analytics/access-status',params=SCOPE,headers=self.auth);self.assertEqual(r.status_code,200);self.assertEqual(r.json()['scope'],SCOPE)
 def test_no_handler_on_authorized_export(self):
  r=self.client.get('/analytics/export',params=dict(SCOPE,start='2026-10-13T00:00:00Z',end='2026-10-14T00:00:00Z'),headers=self.auth);self.assertEqual(r.status_code,503);self.assertNotIn('password',r.text);self.assertFalse(self.hit)
 def test_trading_request(self):self.assertEqual(self.client.post('/evaluate',json={'a':1}).json(),{'unchanged':True});self.assertEqual(self.hit,['evaluate']);self.assertFalse(self.events)
 def test_encoded_path(self):self.assertEqual(self.client.get('/%61nalytics/export').status_code,401);self.assertFalse(self.hit)
 def test_trailing_slash_no_redirect(self):self.assertEqual(self.client.get('/analytics/export/',follow_redirects=False).status_code,401)
 def test_openapi_does_not_execute_data(self):self.assertEqual(self.client.get('/openapi.json').status_code,200);self.assertFalse(self.hit)
 def test_host_and_forwarded_headers_not_identity(self):self.assertEqual(self.client.get('/analytics/export',headers={'X-Forwarded-User':'admin','X-API-Key':SECRET}).status_code,401)
 def test_websocket_denied(self):
  from starlette.websockets import WebSocketDisconnect
  with self.assertRaises(WebSocketDisconnect):
   with self.client.websocket_connect('/analytics/export'):pass
  self.assertFalse(self.hit)
