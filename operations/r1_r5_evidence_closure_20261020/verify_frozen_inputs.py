"""Read-only guard for supplemental tests; never changes any source ref."""
import hashlib,json,subprocess,sys
from pathlib import Path
EXPECTED={'candidate':('d7d9b9c9d55027806d9f5f6cc8d20ee8a456ba67','89c51f474652c5eed22f4608c39d5f1a308276d8'),'recovery':('8c3a7ca2a5f767ef49cbd8cd1cd5023a97de9e9c','312f3e2b4e12d02f7ae9f4d61a25fefdd1a1b356')}
def check(repo):
 rows=[]
 for name,(sha,tree) in EXPECTED.items():
  got=subprocess.check_output(['git','-C',str(repo),'rev-parse',sha+'^{tree}']).decode().strip();rows.append(dict(name=name,commit=sha,expected_tree=tree,actual_tree=got,ok=got==tree))
 return rows
if __name__=='__main__':
 rows=check(Path(sys.argv[1]));print(json.dumps({'claim':'SUPPLEMENTAL AUTHOR evidence','checks':rows},indent=2));sys.exit(not all(r['ok'] for r in rows))
