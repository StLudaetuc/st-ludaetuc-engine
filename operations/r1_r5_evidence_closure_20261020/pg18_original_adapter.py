"""SUPPLEMENTAL AUTHOR infrastructure adapter, NOT EXECUTED against PG here.
Runs recovered original matrix on an explicitly disposable loopback PG18 only.
No implementation edits. No passwords. No remote hosts or production data-dir.
"""
import argparse,importlib,json,os,re,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--python',type=Path,required=True);p.add_argument('--psql',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--pg-log',type=Path,required=True);p.add_argument('--harness',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if os.getuid()==0:raise SystemExit('BLOCKED: PostgreSQL test infrastructure requires a non-root execution identity')
 data=a.data_dir.resolve()
 if not (str(data).startswith('/workspace/scratch/') or str(data).startswith('/tmp/st-ludaetuc-pg18-')):raise SystemExit('BLOCKED: unapproved data-directory prefix')
 marker=data/'ST_LUDAETUC_DISPOSABLE_ONLY'
 if not marker.is_file() or marker.read_text().strip()!='SYNTHETIC_ONLY_NO_PRODUCTION':raise SystemExit('BLOCKED: missing explicit disposable-only marker')
 pid=data/'postmaster.pid'
 if not pid.is_file() or str(data) not in pid.read_text():raise SystemExit('BLOCKED: local test postmaster provenance absent')
 clean={'PATH':'/usr/bin:/bin','PGCONNECT_TIMEOUT':'3','PGPASSFILE':'/dev/null','PGSERVICEFILE':'/dev/null'}
 def sql(q):return subprocess.check_output([str(a.psql),'-h','127.0.0.1','-p','55432','-U','synth','-d','postgres','-Atc',q],env=clean,text=True).strip()
 if not sql('SHOW server_version').startswith('18.'):raise SystemExit('requires actual PostgreSQL 18 server')
 if Path(sql('SHOW data_directory')).resolve()!=a.data_dir.resolve():raise SystemExit('data directory mismatch; will not run')
 if sql('SHOW log_statement')!='all' or '[db=%d]' not in sql('SHOW log_line_prefix'):raise SystemExit('requires statement log and db-qualified log prefix')
 # No arbitrary inherited DB, credential or identity environment enters server tests.
 os.environ.clear();os.environ.update(clean)
 out=a.output_dir.absolute();out.mkdir(parents=True,exist_ok=True);(out/'ev/uvicorn_logs').mkdir(parents=True,exist_ok=True);(out/'ev/hits').mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(a.harness.absolute()));sys.argv=[str(a.harness/'real_uvicorn_matrix.py'),str(a.python.absolute()),str(out/'matrix.json')];m=importlib.import_module('real_uvicorn_matrix')
 m.PSQL=str(a.psql.absolute());m.PGLOG=str(a.pg_log.absolute());m.S=str(out);m.HARNESS=str(a.harness.absolute());m.TREES={'baseline':str(a.baseline.absolute()),'candidate':str(a.candidate.absolute()),'fallback':str(a.candidate.absolute()),'recovery':str(a.recovery.absolute())};m.FALLBACK_FILE=str(a.candidate.absolute()/'security/analytics_release/containment_fallback.py');m.TAG='supplemental';m.OUT=str(out/'matrix.json');m.PY=str(a.python.absolute())
 # Original 13 configs remain unchanged; R-B is a disclosed extra config.
 m.CONFIGS=list(m.CONFIGS)+[('SUPPLEMENTAL R-B mandatory entrypoint','recovery','release.recovery_entry:app',[],{},True)]
 m.main()
 report={'claim':'SUPPLEMENTAL AUTHOR rerun/adaptation; not independent acceptance','original_configs':13,'supplemental_RB_configs':1,'server_version':sql('SHOW server_version'),'status':'RAW_RESULTS_REQUIRE_REVIEW','original_script_sha256':__import__('hashlib').sha256((a.harness/'real_uvicorn_matrix.py').read_bytes()).hexdigest()}
 (out/'adapter-provenance.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
