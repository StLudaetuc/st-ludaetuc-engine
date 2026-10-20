"""SUPPLEMENTAL AUTHOR RUN: original reviewer source, database portion unavailable."""
import subprocess,json,hashlib,sys,time
from pathlib import Path
root=Path(__file__).absolute().parents[1];work=Path.cwd();h=root/'original-fallback/analytics_fallback_independent_review_20261018/harness/test_independent_fallback.py';results=[]
for label in ['G','R']:
 for repeat in [1,2]:
  exe=work/f'env{label}/bin/python';log=root/f'logs/{label}-original-unit-{repeat}.log';start=time.monotonic()
  env={'PATH':'/usr/bin:/bin','TREE':str(work/'candidate-checkout'),'PYTHONDONTWRITEBYTECODE':'1','HITS_FILE':str(root/f'logs/{label}-{repeat}.hits')}
  with log.open('wb') as f:r=subprocess.run([str(exe),str(h)],cwd=work/'candidate-checkout',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
  results.append({'set':label,'repeat':repeat,'exit':r.returncode,'seconds':time.monotonic()-start,'log':str(log.relative_to(root)),'source_sha256':hashlib.sha256(h.read_bytes()).hexdigest(),'database':'NOT CONFIGURED; handler/oracle/lifespan assertions only, not PG assurance','command':[str(exe),str(h)]});print(results[-1],flush=True)
(root/'original-unit-results.json').write_text(json.dumps({'claim':'SUPPLEMENTAL AUTHOR EVIDENCE; not independent findings; original script unchanged','runs':results},indent=2))
