"""Read-only release verification. Full mode requires a post-freeze attestation."""
import argparse,hashlib,json,re,subprocess,sys,tarfile,io
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
REQUIRED={'engine.py':'5fbf9ae9eeeba63fa8d2e2be5d61e611a058033e','analytics_access_control.py':'9f1f84985d92f2cb16df55351f3c2e0bd9dc877a','app.py':'417a5cc99d4852b424fb2e093afbad9cdddd5d57','security/analytics_access_control':'5533972ec65e2b6a55e55fc941946105d8fffcd8','security/analytics_release/DEPLOYMENT_RUNBOOK.md':'a9c0d28c07c57bc4549fef50018fae5f4e5bc02f'}
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],stderr=subprocess.PIPE).decode().strip()
def digest(b):return hashlib.sha256(b).hexdigest()
def verify(commit,manifest,attestation=None,artifact=None,source_only=False):
 errors=[];checks=[]
 def check(name,ok):checks.append({'check':name,'ok':bool(ok)});errors.extend([] if ok else [name])
 sha=git('rev-parse',commit+'^{commit}');tree=git('rev-parse',sha+'^{tree}');m=json.loads(manifest.read_text())
 check('manifest bytes equal tracked manifest',manifest.read_bytes()==subprocess.check_output(['git','-C',str(ROOT),'show',sha+':release/RELEASE_MANIFEST.json']))
 check('full commit SHA input',bool(re.fullmatch('[0-9a-f]{40}',commit)) and sha==commit)
 check('mandatory protected manifest entries',all(m['protected'].get(p)==v for p,v in REQUIRED.items()))
 for p,want in m['protected'].items():check('protected '+p,git('rev-parse',sha+':'+p)==want)
 for p,want in m['file_sha256'].items():check('sha256 '+p,digest(subprocess.check_output(['git','-C',str(ROOT),'show',sha+':'+p]))==want)
 check('mandatory baseline identity',m['baseline']=='2ca3fa49f1b6e44be55c5ccdc1dbddc44d9f5b3c')
 check('baseline ancestry',subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',m['baseline'],sha],capture_output=True).returncode==0)
 if not source_only:
  check('external attestation supplied',attestation is not None)
  check('artifact supplied',artifact is not None)
  if attestation:
   a=json.loads(attestation.read_text());check('attested commit',a['commit']==sha);check('attested tree',a['tree']==tree);check('manifest digest',a['manifest_sha256']==digest(manifest.read_bytes()))
   if artifact:
    check('artifact digest',a['artifact_sha256']==digest(artifact.read_bytes()))
    with tarfile.open(artifact) as bundle:
     check('packaged frozen source',bundle.extractfile('source.tar').read()==subprocess.check_output(['git','-C',str(ROOT),'archive','--format=tar',sha]))
     for wheel in m['wheels']:
      check('wheel '+wheel['wheel'],digest(bundle.extractfile('wheelhouse/'+wheel['wheel']).read())==wheel['sha256'])
 return {'claim':'AUTHOR EVIDENCE','mode':'SOURCE_ONLY' if source_only else 'FULL','commit':sha,'tree':tree,'checks':checks,'errors':errors}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('commit');p.add_argument('manifest',type=Path);p.add_argument('--attestation',type=Path);p.add_argument('--artifact',type=Path);p.add_argument('--source-only',action='store_true');a=p.parse_args()
 try:r=verify(a.commit,a.manifest,a.attestation,a.artifact,a.source_only)
 except Exception as e:r={'errors':[str(e)],'claim':'AUTHOR EVIDENCE'}
 print(json.dumps(r,indent=2));sys.exit(bool(r['errors']))
