"""PG18 grammar checks are syntax-only, never claimed as database execution."""
import json
import unittest
from pathlib import Path
from pglast import parse_sql
from pglast.parser import parse_plpgsql_json, get_postgresql_version

class SQLCandidateSyntaxTests(unittest.TestCase):
    def test_pg18_parser(self):
        self.assertEqual(get_postgresql_version()[0],18)
    def test_migration_sql_grammar(self):
        self.assertGreater(len(parse_sql(Path('finance/migrations/001_foundation.sql').read_text())),0)
    def test_migration_procedural_grammar(self):
        self.assertTrue(json.loads(parse_plpgsql_json(Path('finance/migrations/001_foundation.sql').read_text())))
    def test_acceptance_sql_and_procedural_grammar(self):
        text=Path('finance/tests/postgres_acceptance.sql').read_text()
        self.assertTrue(parse_sql(text));self.assertTrue(json.loads(parse_plpgsql_json(text)))
