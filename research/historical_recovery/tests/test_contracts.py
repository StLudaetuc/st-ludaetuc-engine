import json,unittest
from pathlib import Path
from jsonschema import Draft202012Validator
from research.historical_recovery.demo import demo
ROOT=Path(__file__).resolve().parents[1]
class Contracts(unittest.TestCase):
 def test_population_schema(self):
  schema=json.loads((ROOT/'quality_report.schema.json').read_text());Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(json.loads((ROOT/'evidence/population_summary.json').read_text()))
 def test_reader_schema(self):
  schema=json.loads((ROOT/'reader_contract.schema.json').read_text());Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(demo()['reader_page'])
 def test_unknown_not_zero(self):
  s=json.loads((ROOT/'evidence/population_summary.json').read_text());self.assertEqual(len(s['streams']),35);self.assertTrue(all(x['invalid_timestamp_count'] is None for x in s['streams']));self.assertEqual(s['raw_records_validated'],0)
