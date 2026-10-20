"""Compare two OCI exports, verify descriptor graph and attest image identity."""
import argparse,hashlib,json,pathlib,tarfile
p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('out');a=p.parse_args()
def inspect(file):
 with tarfile.open(file) as t:
  members={m.name:m for m in t.getmembers() if m.isfile()}
  def read(name):return t.extractfile(members[name]).read()
  index=json.loads(read('index.json'));seen={}
  def visit(d):
   algo,digest=d['digest'].split(':');assert algo=='sha256';b=read('blobs/sha256/'+digest);assert hashlib.sha256(b).hexdigest()==digest and len(b)==d['size'];seen[d['digest']]=len(b)
   if d['mediaType'].endswith('manifest.v1+json'):
    m=json.loads(b);visit(m['config'])
    for x in m['layers']:visit(x)
  for d in index['manifests']:visit(d)
  return {'oci_index':index,'blobs':seen,'archive_sha256':hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()}
x,y=inspect(a.first),inspect(a.second);same=x['oci_index']==y['oci_index'] and x['blobs']==y['blobs'];result={'label':'AUTHOR clean no-cache OCI build comparison; not platform acceptance','first':x,'second':y,'identical_image_descriptors_and_blobs':same};pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(not same)
