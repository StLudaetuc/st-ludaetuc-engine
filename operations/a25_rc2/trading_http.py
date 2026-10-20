"""Original corpus over real local HTTP. No DB configured: no persistence claim.
Original nontrading /openapi and /nonexistent probes reported separately.
"""
import base64,hashlib,http.client,json,os,pathlib,socket,subprocess,sys,time
CORPUS,BASE,PRIMARY,RB,RUNTIME,OUT=sys.argv[1:];items=json.loads(pathlib.Path(CORPUS).read_text());results={}
for name,tree,app in [('baseline',BASE,'engine:app'),('primary',PRIMARY,'app:app'),('rb',RB,'release.recovery_entry:app')]:
 s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();log=pathlib.Path(OUT).with_name(name+'-trading.log').open('w')
 proc=subprocess.Popen([str(pathlib.Path(RUNTIME).resolve()/'bin/python'),'-m','uvicorn',app,'--host','127.0.0.1','--port',str(port)],cwd=tree,env={'PATH':str(pathlib.Path(RUNTIME).resolve()/'bin')+':/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
 try:
  for _ in range(100):
   try:
    c=http.client.HTTPConnection('127.0.0.1',port,timeout=2);c.request('GET','/health');r=c.getresponse();r.read();c.close()
    if r.status==200:break
   except OSError:pass
   time.sleep(.05)
  rows=[]
  for x in items:
   c=http.client.HTTPConnection('127.0.0.1',port,timeout=3);headers={'Host':'fixture.synthetic'}
   if x['content_type']:headers['Content-Type']=x['content_type']
   c.request(x['method'],x['path'],body=base64.b64decode(x['body_b64']) if x['body_b64'] else None,headers=headers);r=c.getresponse();b=r.read();h=[(k.lower(),v) for k,v in r.getheaders() if k.lower()!='date'];rows.append({'status':r.status,'headers':h,'body':b.decode(errors='replace')});c.close()
  results[name]=rows
 finally:
  proc.terminate();proc.wait(5);log.close()
diffs=[]
for i,x in enumerate(items):
 for n in ['primary','rb']:
  if results[n][i]!=results['baseline'][i]:diffs.append({'index':i,'path':x['path'],'target':n,'trading':x['path'] not in ['/openapi.json','/nonexistent'],'baseline':results['baseline'][i],'actual':results[n][i]})
out={'label':'AUTHOR original236 corpus; real local HTTP; NO DATABASE configured; not PG/schema/view/row acceptance','corpus_sha256':hashlib.sha256(pathlib.Path(CORPUS).read_bytes()).hexdigest(),'requests':len(items),'trading_requests':sum(x['path'] not in ['/openapi.json','/nonexistent'] for x in items),'trading_differences':sum(x['trading'] for x in diffs),'nontrading_differences':sum(not x['trading'] for x in diffs),'differences':diffs,'raw':results,'persistence':'NOT_TESTED'};pathlib.Path(OUT).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['raw','differences']}));raise SystemExit(out['trading_differences']!=0)
