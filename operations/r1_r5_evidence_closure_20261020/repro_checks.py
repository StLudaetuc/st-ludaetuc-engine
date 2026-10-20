"""SUPPLEMENTAL AUTHOR offline/hash/SBOM checks; no container claim."""
import hashlib,json,subprocess,tempfile,shutil,zipfile,email,sys
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1];WS=ROOT.parent;out=[]
for label,wheels,lock in [('G',WS/'private/operations/wheelhouse',WS/'candidate-checkout/requirements.txt'),('R',WS/'wheelR',WS/'candidate-checkout/release/R.lock')]:
 exe=ROOT/f'env-{label}/bin/python';inv=subprocess.check_output([str(exe),'-m','pip','inspect','--local']);(ROOT/f'logs/{label}-fresh-inventory.json').write_bytes(inv)
 packages={x['metadata']['name'].lower().replace('_','-'):x['metadata']['version'] for x in json.loads(inv)['installed']};inventory=[]
 for w in sorted(wheels.glob('*.whl')):
  with zipfile.ZipFile(w) as z:
   m=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))));wheel=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
  name=m['Name'].lower().replace('_','-');assert packages[name]==m['Version'];inventory.append(dict(name=name,version=m['Version'],wheel=w.name,sha256=hashlib.sha256(w.read_bytes()).hexdigest(),requires_python=m['Requires-Python'],requires_dist=m.get_all('Requires-Dist',[]),tags=[s[5:] for s in wheel.splitlines() if s.startswith('Tag: ')],source_built=False))
 for mode in ['clean','tamper-wheel','tamper-lock']:
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);wh=p/'wheels';shutil.copytree(wheels,wh);lk=p/'requirements.txt';lk.write_bytes(lock.read_bytes())
   if mode=='tamper-wheel':
    f=sorted(wh.glob('*.whl'))[0];f.write_bytes(f.read_bytes()+b'SYNTHETIC_TAMPER')
   if mode=='tamper-lock':
    import re
    lk.write_text(re.sub(r'(?<=sha256:)[0-9a-f]{64}','0'*64,lk.read_text(),count=1))
   cmd=[str(exe),'-m','pip','--isolated','install','--dry-run','--ignore-installed','--no-index','--find-links',str(wh),'-r',str(lk)];r=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin'},capture_output=True,text=True,timeout=45);(ROOT/f'logs/{label}-{mode}.log').write_text(r.stdout+r.stderr);assert (r.returncode==0) if mode=='clean' else (r.returncode!=0 and 'DO NOT MATCH THE HASHES' in r.stdout+r.stderr);out.append(dict(set=label,control=mode,exit=r.returncode,expected_result=True))
 (ROOT/f'{label}-wheel-inventory.json').write_text(json.dumps(inventory,indent=2))
# Verify selected G SBOM components against exact verified wheel inventory.
sbom=json.loads((WS/'candidate-checkout/release/sbom.cdx.json').read_text());g=json.loads((ROOT/'G-wheel-inventory.json').read_text());expected={(r['name'],r['version'],r['sha256']) for r in g};actual={(r['name'].lower().replace('_','-'),r['version'],r['hashes'][0]['content']) for r in sbom['components']};assert actual==expected
(ROOT/'repro-check-results.json').write_text(json.dumps(dict(claim='SUPPLEMENTAL AUTHOR EVIDENCE; exact Python3.14.3; hash-enforced offline wheels; not container reproducibility',tests=out,sbom_wheel_match=True,container_build='INCOMPLETE: docker/podman absent; no verified image was executed'),indent=2));print('offline controls',len(out),'SBOM matches G wheels')
