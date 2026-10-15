"""Synthetic offline demonstration; not a production source or approved calendar."""
import json
from .tests.test_recovery import fixture,context,START,END
from .auditor import audit,reader_page

def demo():
    rows=[fixture('legacy'),fixture('v2',offset=1),fixture('v2',offset=2)]
    result=audit(rows,context(*rows),start=START,end=END,source_count=3,snapshot_complete=True)
    return {'classification':'SYNTHETIC_ONLY_NOT_PRODUCTION_EVIDENCE','input_events':rows,'quality_summary':{k:v for k,v in result.items() if k!='normalized_rows'},'reader_page':reader_page(result,manifest_hash=result['manifest_hash']),'normalized_examples':result['normalized_rows'][:2]}

if __name__=='__main__':print(json.dumps(demo(),indent=2))
