import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from operations.a25_rc2 import drill
from release import runtime_preflight as pre

class GuardTests(unittest.TestCase):
    def inventory(self):
        return {'environment':'staging','production':False,'old_service_id':'old','new_service_id':'new','complete_path_inventory':True,'private_peer_inventory':True,'network_mode':'synthetic-loopback','approval_kind':'CEO_STAGING_ONLY','approval_reference':'SYNTHETIC-UNIT-TEST','approval_expires_epoch':time.time()+30,'targets':[{'id':'local','url':'http://127.0.0.1:12345','service_id':'old','environment':'staging','vantage':'private-peer'}]}
    def test_dry_run_needs_no_network(self):
        with patch('socket.create_connection',side_effect=AssertionError('unexpected network')):
            self.assertEqual(drill.validate_inventory(self.inventory())['environment'],'staging')
    def test_execution_requires_explicit_staging_approval(self):
        for key,value in [('environment','production'),('production',True),('approval_kind','NOT_APPROVED'),('approval_reference',None),('approval_expires_epoch',0),('network_mode','real-platform'),('complete_path_inventory',False),('private_peer_inventory',False),('new_service_id','old')]:
            m=self.inventory();m[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):drill.validate_inventory(m,True)
    def test_nonloopback_and_credential_urls_rejected(self):
        for url in ['http://st-ludaetuc-engine.onrender.com','http://localhost:80','http://192.0.2.1','http://user:secret@127.0.0.1','http://127.0.0.1/path','http://127.0.0.1?x=y','https://example.invalid']:
            m=self.inventory();m['targets'][0]['url']=url
            with self.subTest(url=url),self.assertRaises(ValueError):drill.validate_inventory(m,True)
    def test_unknown_service_and_vantage_rejected(self):
        for key,value in [('service_id','unlisted'),('vantage','operator-allowlisted'),('environment','production')]:
            m=self.inventory();m['targets'][0][key]=value
            with self.assertRaises(ValueError):drill.validate_inventory(m,True)
    def test_local_execution_inventory_accepted(self):
        drill.validate_inventory(self.inventory(),True)
    def test_runtime_wrong_python_and_platform_rejected(self):
        with patch.object(pre.platform,'python_version',return_value='3.13.0'),patch.object(pre.platform,'machine',return_value='aarch64'):
            errors=pre.verify(phase='build')['errors'];self.assertTrue(any('3.14.3' in e for e in errors));self.assertIn('requires Linux x86_64',errors)
    def test_override_rejected(self):
        for value in ['','3.14.6','3.13']:
            with patch.dict(os.environ,{'PYTHON_VERSION':value}):self.assertTrue(any('PYTHON_VERSION' in e for e in pre.verify(phase='build')['errors']))
    def test_lock_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'release').mkdir()
            for f in ['requirements.txt','release/G.lock','release/runtime_contract.json']:(r/f).write_bytes((pre.ROOT/f).read_bytes())
            (r/'requirements.txt').write_bytes((r/'requirements.txt').read_bytes()+b'# drift\n')
            e=pre.verify(r,'build')['errors'];self.assertIn('requirements lock digest mismatch',e)
    def test_missing_wrong_duplicate_and_extra_distributions(self):
        with patch.object(pre.md,'distributions',return_value=[]):self.assertTrue(any('missing distribution' in e for e in pre.verify()['errors']))
        class D:
            metadata={'Name':'unapproved-package'};version='1';files=[]
        with patch.object(pre.md,'distributions',return_value=[D(),D()]):
            e=pre.verify()['errors'];self.assertTrue(any('unexpected runtime' in s for s in e));self.assertTrue(any('duplicate installed' in s for s in e))

if __name__=='__main__':unittest.main(verbosity=2)
