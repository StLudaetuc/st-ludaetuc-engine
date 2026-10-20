"""SUPPLEMENTAL AUTHOR adapter: original corpus, no PG, no runtime source edits."""
import base64,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1];WS=ROOT.parent;H=ROOT/'original-fallback/analytics_fallback_independent_review_20261018/harness'
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 sys.path.insert(0,str(H));sys.argv=[str(H/'trading_diff.py'),str(WS/'envG/bin/python'),str(ROOT/'unused-output.json')]
 import real_uvicorn_matrix as matrix
 matrix.TREES['fallback']=str(WS/'candidate-checkout')
 import trading_diff as original
 corpus=original.corpus();encoded=[dict(method=m,path=p,body_b64=None if b is None else base64.b64encode(b).decode(),content_type=c) for m,p,b,c in corpus]
 (ROOT/'original-corpus-236.json').write_text(json.dumps(encoded,indent=2))
 allowed={'/health','/evaluate','/market-data','/trade-outcome','/trade-opened'}
 keep=[(i,item) for i,item in enumerate(corpus) if item[1].rstrip('/') in allowed]
 omitted=[{'index':i,'method':x[0],'path':x[1]} for i,x in enumerate(corpus) if x[1].rstrip('/') not in allowed]
 # Preserve the original 234-call subset and disclose the two author-added repeats.
 items=[x for _,x in keep]+[corpus[0],corpus[1]]
 assert len(corpus)==236 and len(keep)==234 and len(items)==236
 (ROOT/'trading-only-corpus-236.json').write_text(json.dumps({'claim':'234 original requests plus two disclosed author-added repeats; NOT original 236 fixture','original_indices':[i for i,_ in keep],'omitted':omitted,'added_original_indices':[0,1],'requests':[dict(method=m,path=p,body_b64=None if b is None else base64.b64encode(b).decode(),content_type=c) for m,p,b,c in items]},indent=2))
 results={};runtimes={}
 for label in ['G','R']:
  exe=WS/f'env{label}/bin/python';inv=subprocess.run([str(exe),'-m','pip','inspect','--local'],capture_output=True,check=True);(ROOT/f'logs/{label}-pip-inspect.json').write_bytes(inv.stdout)
  for mode,tree,entry in [('baseline',WS/'baseline-supplement','engine:app'),('primary',WS/'candidate-checkout','engine:app'),('fallback',WS/'candidate-checkout','security.analytics_release.fallback_app:app'),('RB',WS/'recovery-checkout','release.recovery_entry:app')]:
   s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();log=ROOT/f'logs/{label}-{mode}-original-http.log';env={'PATH':'/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','HITS_FILE':str(ROOT/f'logs/{label}-{mode}.hits')}
   with log.open('wb') as f:
    proc=subprocess.Popen([str(exe),str(H/'launcher.py'),str(tree),entry,'--host','127.0.0.1','--port',str(port)],env=env,stdout=f,stderr=subprocess.STDOUT)
    try:
     for _ in range(100):
      try:
       if matrix.status_of(matrix.raw(port,matrix.req('GET','/health'),1))==200:break
      except OSError:time.sleep(.05)
     else:raise RuntimeError('server startup failed')
     out=[]
     for m,p,b,c in items:
      response=matrix.raw(port,matrix.req(m,p,[('Content-Type',c)] if c else [],b or b''));headers,_,body=response.partition(b'\r\n\r\n');head=[line for line in headers.decode('latin1').split('\r\n')[1:] if not line.lower().startswith('date:')];out.append({'method':m,'path':p,'status':matrix.status_of(response),'headers':head,'body':body.decode('utf-8','replace')})
    finally:proc.terminate();proc.wait(timeout=10)
   results[label+'-'+mode]=out
 baseline=results['G-baseline'];comparison={k:sum(a!=b for a,b in zip(v,baseline)) for k,v in results.items()}
 result={'claim':'SUPPLEMENTAL AUTHOR EVIDENCE; original fixture adapter, real HTTP with no PG configured, no stubs','original_corpus_count':len(corpus),'original_corpus_hash':digest(encoded),'original_trading_subset_count':len(keep),'nontrading_originals_excluded':omitted,'author_extension_original_indices':[0,1],'trading_request_count':236,'date_header_normalized':True,'fixture_sha256':hashlib.sha256((ROOT/'trading-only-corpus-236.json').read_bytes()).hexdigest(),'response_differences_vs_G_baseline':comparison,'results':results,'database':'NOT CONFIGURED; event persistence/schema/row hash unknown; A13 INCOMPLETE'}
 (ROOT/'original-fixture-http-results.json').write_text(json.dumps(result,indent=2));print(comparison)
if __name__=='__main__':main()
