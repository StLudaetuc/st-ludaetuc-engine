"""Classify source hashes and explicit fixtures without printing potential secrets."""
import json,subprocess,re,base64,hashlib
from pathlib import Path
root=Path(__file__).absolute().parents[1];history=root/'history.git';report=json.loads((root/'logs/gitleaks-full-history-final.json').read_text());rows=[]
for f in report:
 b=subprocess.check_output(['git','--git-dir='+str(history),'show',f['Commit']+':'+f['File']]);lines=b.decode().splitlines();segment='\n'.join(lines[f['StartLine']-1:f['EndLine']]);classification='REQUIRES_REVIEW';reason='unclassified; never assume synthetic'
 if f['RuleID']=='generic-api-key':
  literals=re.findall(r'(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])',segment)
  if literals and (any(w in segment.lower() for w in ['sha','hash','digest','app.py','engine.py','manifest']) or f['File'].endswith('execution.json')):
   classification='NON_SECRET_PROVENANCE_HASH';reason='Hex Git object / SHA-256 source or execution digest; public immutable source metadata, not an authentication value'
 if classification=='REQUIRES_REVIEW':
  for value in re.findall(r'(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])',segment):
   t=subprocess.run(['git','--git-dir='+str(history),'cat-file','-t',value],capture_output=True,text=True)
   if t.returncode==0 and t.stdout.strip() in ['blob','tree','commit']:
    classification='NON_SECRET_VERIFIED_GIT_OBJECT';reason='40-hex value resolves to an actual Git object in the complete source mirror'
 if f['RuleID']=='jwt' and f['File'].endswith('tests/test_v11.py'):
  tokens=re.findall(r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',segment)
  if tokens and ('payload' in segment.lower() or 'eyJ' in segment):
   classification='SYNTHETIC_TEST_FIXTURE';reason='Literal JWT-like input inside redaction test; no issuer/account credential authority; verify test provenance separately'
 rows.append({'fingerprint':f['Fingerprint'],'rule':f['RuleID'],'commit':f['Commit'],'file':f['File'],'start_line':f['StartLine'],'end_line':f['EndLine'],'classification':classification,'reason':reason,'source_segment_sha256':hashlib.sha256(segment.encode()).hexdigest()})
(root/'gitleaks-history-triage.json').write_text(json.dumps({'claim':'SUPPLEMENTAL AUTHOR TRIAGE; not independent acceptance','raw_findings':len(report),'findings':rows,'unresolved':sum(x['classification']=='REQUIRES_REVIEW' for x in rows)},indent=2));print('history findings',len(rows),'unresolved',sum(x['classification']=='REQUIRES_REVIEW' for x in rows));print([(r['file'],r['start_line']) for r in rows if r['classification']=='REQUIRES_REVIEW'])
