"""Deterministic stdlib packaging; no network. AUTHOR EVIDENCE."""
import argparse,gzip,hashlib,io,json,tarfile,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--wheels',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epoch',type=int,required=True);a=p.parse_args()
if sys.version_info[:3]!=(3,14,3):raise SystemExit('requires exact Python 3.14.3')
files={'source.tar':a.source.read_bytes()}
with tarfile.open(a.source) as t:
 m=json.load(t.extractfile('release/RELEASE_MANIFEST.json'))
 for w in m['wheels']:
  b=(a.wheels/w['wheel']).read_bytes()
  if hashlib.sha256(b).hexdigest()!=w['sha256']:raise SystemExit('wheel digest mismatch: '+w['wheel'])
  files['wheelhouse/'+w['wheel']]=b
files['INSTALL.txt']=b'Extract source.tar into app/. With Python 3.14.3 create a venv, then pip install --no-index --find-links ../wheelhouse -r requirements.txt. Start only using release/launch.sh.\n'
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('wb') as f:
 with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=9) as gz:
  with tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as t:
   for name,data in sorted(files.items()):
    n=tarfile.TarInfo(name);n.size=len(data);n.mtime=a.epoch;n.uid=n.gid=0;n.uname=n.gname='';n.mode=0o644;t.addfile(n,io.BytesIO(data))
print(json.dumps({'artifact':str(a.out),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'bytes':a.out.stat().st_size,'claim':'AUTHOR EVIDENCE'}))
