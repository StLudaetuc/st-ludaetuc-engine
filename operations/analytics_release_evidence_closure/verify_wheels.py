"""Local wheel integrity negative control; no package installation or service calls."""
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path

def verify(folder,lock):
    wheels=sorted(folder.glob('*.whl'))
    if len(wheels)!=15:raise ValueError('expected complete15-wheel observed runtime selection')
    manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in wheels]
    with tempfile.TemporaryDirectory() as d:
        dest=Path(d)
        for p in wheels:os.link(p,dest/p.name)
        selected=dest/wheels[0].name
        data=selected.read_bytes();selected.unlink();selected.write_bytes(data+b'\nSYNTHETIC_TAMPER_CONTROL\n')
        cmd=[sys.executable,'-m','pip','--isolated','install','--dry-run','--ignore-installed','--no-index','--find-links',str(dest),'--require-hashes','-r',str(lock.absolute())]
        r=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin'},capture_output=True,text=True,timeout=60)
        if r.returncode==0 or 'DO NOT MATCH THE HASHES' not in r.stdout+r.stderr:raise RuntimeError('hash tampering detector not proven')
        return {'schema':'wheel_integrity_v1','selected_wheels':manifest,'tamper_negative_control':{'changed_wheel':selected.name,'command':cmd,'exit_code':r.returncode,'expected_hash_rejection':True,'log':r.stdout+r.stderr},'claim':'local archive integrity only; not production runtime provenance'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wheelhouse',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(verify(a.wheelhouse,a.lock),indent=2)+'\n')
