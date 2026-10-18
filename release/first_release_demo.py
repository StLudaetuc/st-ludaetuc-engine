"""AUTHOR EVIDENCE: request-path model, not actual Render enforcement."""
import asyncio,contextlib,io,json
from security.analytics_release.containment_fallback import ContainmentOnly
async def old_instance(scope,receive,send):
 body=b'SYNTHETIC_BASELINE_ANALYTICS' if '/analytics/' in scope['path'] else b'SYNTHETIC_TRADING_OK'
 await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':body})
async def invoke(app,path):
 out=[]
 async def receive():return {'type':'http.request','body':b''}
 async def send(msg):out.append(msg)
 await app({'type':'http','path':path,'root_path':'','method':'GET'},receive,send)
 return {'status':out[0]['status'],'leaked':b'SYNTHETIC_BASELINE_ANALYTICS' in out[-1].get('body',b'')}
async def main():
 ingress=ContainmentOnly(old_instance);rows=[]
 with contextlib.redirect_stderr(io.StringIO()):
  for phase in ['before','candidate_build_failed_old_still_serving','candidate_start_failed_old_still_serving','after']:
   for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/access-status','//analytics//export','/ANALYTICS/export','/analytics/export\n','/analytics/export%0A']:
    rows.append(dict(phase=phase,path=path,through_independent_boundary=await invoke(ingress,path),direct_origin=await invoke(old_instance,path)))
 assert all(not r['through_independent_boundary']['leaked'] and r['through_independent_boundary']['status']==503 for r in rows)
 assert any(r['direct_origin']['leaked'] for r in rows)
 return dict(claim='AUTHOR EVIDENCE; synthetic ASGI model only',old_instance_kept_serving=True,independent_ingress_protected_in_model=True,direct_origin_bypass_control_fired=True,A25='NOT MET: no demonstrated all-path independent boundary for the existing public origin',rows=rows)
if __name__=='__main__':print(json.dumps(asyncio.run(main()),indent=2))
