"""SUPPLEMENTAL AUTHOR host rebuild of frozen artifacts; not container evidence."""
import subprocess,json,hashlib
from pathlib import Path
r=Path(__file__).absolute().parents[1];ws=r.parent;records=[]
for label,sha,expected in [('candidate','d7d9b9c9d55027806d9f5f6cc8d20ee8a456ba67','0801321dd167106f0511306a127f1efe9a2a194fe02aeff222e417853f695cad'),('RB','8c3a7ca2a5f767ef49cbd8cd1cd5023a97de9e9c','0b3df73c3eac2a30fa1e459839aa01962f8ee48312186bccb52270a158aef209')]:
 epoch=subprocess.check_output(['git','--git-dir='+str(r/'history.git'),'show','-s','--format=%ct',sha]).decode().strip()
 digests=[]
 for n,exe in [(1,r/'env-G/bin/python'),(2,r/'env-R/bin/python')]:
  folder=r/f'host-builds/{label}-{n}';folder.mkdir(parents=True,exist_ok=True);source=folder/'source.tar';source.write_bytes(subprocess.check_output(['git','--git-dir='+str(r/'history.git'),'archive','--format=tar',sha]));recipe=folder/'build_payload.py';recipe.write_bytes(subprocess.check_output(['git','--git-dir='+str(r/'history.git'),'show',sha+':release/build_payload.py']));artifact=folder/'application.tar.gz';cmd=[str(exe),str(recipe),'--source',str(source),'--wheels',str(ws/'private/operations/wheelhouse'),'--out',str(artifact),'--epoch',epoch];p=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin','SOURCE_DATE_EPOCH':epoch},capture_output=True,text=True,check=True);(r/f'logs/{label}-host-build-{n}.log').write_text(p.stdout+p.stderr);digests.append(hashlib.sha256(artifact.read_bytes()).hexdigest())
 records.append({'label':label,'frozen_commit':sha,'source_date_epoch':epoch,'digests':digests,'match_each_other':len(set(digests))==1,'match_original':all(d==expected for d in digests),'interpretation':'packaging reads and validates frozen G wheel inventory even when executed from separate G/R tooling envs; not an R application artifact or pinned image build'})
(r/'host-rebuild-results.json').write_text(json.dumps({'claim':'SUPPLEMENTAL AUTHOR repeatability only','records':records,'A12':'INCOMPLETE; no container image executed'},indent=2));print(records)
