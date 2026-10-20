"""AUTHOR real-engine re-test of original Claude edge model. NOT platform/PG proof.
usage: python model_challenge.py REVIEW_EDGE_MODEL BASELINE PRIMARY RB RUNTIME OUT
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import httpx

EDGE,BASE,PRIMARY,RB,RUNTIME,OUT=sys.argv[1:7]
OUT=Path(OUT).resolve();OUT.parent.mkdir(exist_ok=True,parents=True)
HERE=Path(__file__).resolve().parent
PY=str(Path(RUNTIME).resolve()/'bin/python')
def port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p
def start(tree,app,name):
    p=port();events=OUT.parent/(name+'.events.jsonl');events.write_text('');f=(OUT.parent/(name+'.server.log')).open('w')
    proc=subprocess.Popen([PY,str(HERE/'synthetic_engine.py'),str(Path(tree).resolve()),app,str(p),str(events)],env={'PATH':str(Path(RUNTIME).resolve()/'bin')+':/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'},stdout=f,stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            if httpx.get(f'http://127.0.0.1:{p}/health',trust_env=False).status_code==200:return {'proc':proc,'port':p,'events':events,'log':f}
        except httpx.HTTPError:pass
        if proc.poll() is not None:raise RuntimeError('engine start failed '+name)
        time.sleep(.05)
    raise RuntimeError('timeout '+name)
def stop(x):
    x['proc'].terminate()
    try:x['proc'].wait(3)
    except subprocess.TimeoutExpired:x['proc'].kill();x['proc'].wait()
    x['log'].close()
def counts(x):
    a=[json.loads(l) for l in x['events'].read_text().splitlines()];return {'handlers':sum(y['kind']=='ANALYTICS_HANDLER' for y in a),'synthetic_select_calls':sum(y['kind']=='SYNTHETIC_SQL_CALL' and y['verb']=='SELECT' for y in a)}
async def run():
    statefile=OUT.parent/'edge-state.json';os.environ['EDGE_STATE']=str(statefile)
    spec=importlib.util.spec_from_file_location('claude_original_edge',EDGE);edge=importlib.util.module_from_spec(spec);spec.loader.exec_module(edge)
    protected=start(PRIMARY,'app:app','primary');recovery=start(RB,'release.recovery_entry:app','rb');results=[]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=edge.app),base_url='http://edge.synthetic',trust_env=False) as client:
      async def probes(state,old,label):
        statefile.write_text(json.dumps(state));rows=[]
        for host in state['hosts']:
          for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/export%0A','//analytics//export','/%61nalytics/export']:
            r=await client.get(path,headers={'Host':host,'X-Forwarded-Host':'origin.synthetic'});rows.append({'phase':label,'host':host,'path':path,'vantage':'model-public','status':r.status_code})
        if old['proc'].poll() is None:
          for path in ['/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy']:
            r=await edge._client.get(f"http://127.0.0.1:{old['port']}"+path);rows.append({'phase':label,'vantage':'loopback-stand-in-for-private-peer','path':path,'status':r.status_code})
        else:rows.append({'phase':label,'vantage':'loopback-stand-in-for-private-peer','status':None,'reason':'old process stopped'})
        return rows
      cases=[]
      for option in ['O0','A','A2','B','D1','D2','E','C','F','FS']:
        for failure in ['success','build_fail','start_fail']:
          cases.append((option,failure))
      cases += [('FS','health_fail'),('FS','resume_old_challenge'),('FS','suspend_edge_only_challenge'),('FS','new_service_old_source_challenge'),('FS','unknown_preview_challenge')]
      try:
       for i,(option,failure) in enumerate(cases):
        old=start(BASE,'engine:app',f'old-{i}');before=counts(old)
        state={'hosts':{'default.synthetic':{'service':'old'},'custom.synthetic':{'service':'old'},'gateway.synthetic':{'service':'old','gateway_deny_analytics':option in ['A','A2']}},'services':{'old':{'active':f"127.0.0.1:{old['port']}"}}}
        control=await probes(state,old,'positive-control');control_ok=any(r['vantage']=='model-public' and 200<=r['status']<300 for r in control) and counts(old)['handlers']>before['handlers'] and counts(old)['synthetic_select_calls']>before['synthetic_select_calls']
        if option=='A2':state['hosts']['default.synthetic']['disabled']=True
        if option in ['D1','E','F']:state['services']['old']['maintenance']=True
        if option=='D2':state['services']['old']['allowlist']=['operator.synthetic']
        if option=='FS':
          state['services']['old']['suspended']=True
          if failure!='suspend_edge_only_challenge':stop(old)
        if option in ['C','F','FS']:
          state['hosts']['new.synthetic']={'service':'new'};state['services']['new']={'active':None}
        initial=counts(old);obs=await probes(state,old,'after-lock-before-deploy')
        # Real disposable start fault; build failure is deterministic preflight in
        # a separate test. Here only retained-instance routing is modelled.
        target='new' if option in ['C','F','FS'] else 'old'
        if failure not in ['build_fail','start_fail','health_fail']:
          state['services'][target]['active']=f"127.0.0.1:{protected['port']}"
          if option in ['D1','D2','E']:state['services']['old'].pop('maintenance',None);state['services']['old'].pop('allowlist',None)
          if option in ['C','F','FS']:state['hosts']['custom.synthetic']['service']='new'
        if option=='E':
          # Model isolated private peers by excluding that vantage only. Do not
          # report unprobed private traffic as an observed denial.
          obs=[r for r in obs if r['vantage']=='model-public']
          if failure=='success':state['services']['old']['active']=f"127.0.0.1:{old['port']}"
        if failure=='resume_old_challenge':
          old=start(BASE,'engine:app',f'resumed-{i}');state['services']['old']={'active':f"127.0.0.1:{old['port']}"};initial=counts(old)
        if failure=='new_service_old_source_challenge':
          old=start(BASE,'engine:app',f'rollback-{i}');state['services']['new']['active']=f"127.0.0.1:{old['port']}";initial=counts(old)
        if failure=='unknown_preview_challenge':
          old=start(BASE,'engine:app',f'preview-{i}');state['hosts']['preview.synthetic']={'service':'preview'};state['services']['preview']={'active':f"127.0.0.1:{old['port']}"};initial=counts(old)
        obs+=await probes(state,old,'after-deploy-result')
        if option=='E':obs=[r for r in obs if r['vantage']=='model-public']
        after=counts(old);exposed=any(r['status'] is not None and 200<=r['status']<300 for r in obs)
        expected=not(option=='FS' and failure in ['success','build_fail','start_fail','health_fail']) and not(option=='E' and failure!='success')
        results.append({'case':i,'option':option,'failure':failure,'control_valid':control_ok,'exposed':exposed,'expected_exposed':expected,'observation_matches':exposed==expected,'handler_delta':after['handlers']-initial['handlers'],'synthetic_select_delta':after['synthetic_select_calls']-initial['synthetic_select_calls'],'observations':obs,'private_isolation': 'ASSUMED_NOT_PROBED' if option=='E' else 'loopback process only'})
        if old['proc'].poll() is None:stop(old)
       # R-B real protected process, same synthetic DB adapter, not image proof.
       r=await edge._client.get(f"http://127.0.0.1:{recovery['port']}/analytics/export");results.append({'rb_analytics_status':r.status_code,'rb_handler_counts':counts(recovery),'pass':r.status_code==503 and counts(recovery)['handlers']==0})
      finally:
       stop(protected);stop(recovery)
    await edge._client.aclose()
    output={'label':'AUTHOR REAL ENGINE + ORIGINAL SYNTHETIC EDGE MODEL; NO POSTGRESQL OR RENDER PROOF','cases':results,'limitation':'private peers represented by local direct process access; SQL calls are a synthetic adapter, no persistence/SQL semantics verified; failure routing is modelled','result':'MATCHED_MODEL_EXPECTATIONS' if all(x.get('observation_matches',x.get('pass',False)) and x.get('control_valid',True) for x in results) else 'FAIL'}
    OUT.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'scenarios':len(results)-1,'result':output['result']}));return output['result']!='MATCHED_MODEL_EXPECTATIONS'
raise SystemExit(asyncio.run(run()))
