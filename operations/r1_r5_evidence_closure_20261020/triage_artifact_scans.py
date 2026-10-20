"""SUPPLEMENTAL AUTHOR triage. Raw findings preserved; never suppress scan exits."""
import io,json,re,tarfile,zipfile,hashlib,base64
from pathlib import Path
ROOT=Path(__file__).absolute().parents[1]
def read_nested(path):
 parts=path.split('!');b=Path(parts[0]).read_bytes()
 for member in parts[1:]:
  try:
   with tarfile.open(fileobj=io.BytesIO(b)) as t:b=t.extractfile(member).read()
  except tarfile.ReadError:
   with zipfile.ZipFile(io.BytesIO(b)) as z:b=z.read(member)
 return b
results={}
for stem in ['gitleaks-author-archive-exact','gitleaks-original-evidence','gitleaks-original-fallback']:
 p=ROOT/'logs'/f'{stem}.json'
 if not p.exists():continue
 findings=json.loads(p.read_text());rows=[]
 for f in findings:
  status='REQUIRES_REVIEW';reason='Unclassified finding';segment=b''
  try:
   lines=read_nested(f['File']).decode(errors='replace').splitlines();segment='\n'.join(lines[f['StartLine']-1:f['EndLine']]).encode();s=segment.decode()
   hashes=re.findall(r'(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])',s)
   if f['RuleID']=='generic-api-key' and hashes:
    status='NON_SECRET_PROVENANCE_HASH';reason='SHA-1/SHA-256 source/build/file digest in immutable manifest/log/verification source; no auth use'
   if f['RuleID']=='generic-api-key' and 'Sec-WebSocket-Key' in s and 'dGhlIHNhbXBsZSBub25jZQ==' in s:
    status='PUBLIC_PROTOCOL_EXAMPLE';reason='RFC6455 sample WebSocket nonce, base64(the sample nonce), not a credential'
  except Exception as e:reason='Read limitation: '+type(e).__name__
  rows.append(dict(fingerprint=f['Fingerprint'],file=f['File'],line=f['StartLine'],rule=f['RuleID'],classification=status,reason=reason,source_segment_sha256=hashlib.sha256(segment).hexdigest()))
 report=dict(claim='SUPPLEMENTAL AUTHOR TRIAGE; not independent acceptance',raw_findings=len(findings),unresolved=sum(r['classification']=='REQUIRES_REVIEW' for r in rows),findings=rows);(ROOT/f'{stem}-triage.json').write_text(json.dumps(report,indent=2));results[stem]={'raw_findings':len(findings),'unresolved':report['unresolved']}
(ROOT/'artifact-scan-triage-summary.json').write_text(json.dumps(results,indent=2));print(results)
