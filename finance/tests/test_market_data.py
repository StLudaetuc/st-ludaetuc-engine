import json
import unittest
from pathlib import Path
from finance.market_data import validate

class MarketDataTests(unittest.TestCase):
    def fixture(self):return json.loads(Path('finance/contracts/market_data_v2.example.json').read_text())
    def test_good(self):self.assertEqual(validate(self.fixture())['ticker'],'GBPUSD')
    def test_bad_placeholder_json(self):
        with self.assertRaises(json.JSONDecodeError):json.loads('{"open":{{open}}}')
    def test_quoted_placeholder_rejected(self):
        p=self.fixture();p['ticker']='{{ticker}}'
        with self.assertRaises(ValueError):validate(p)
    def test_numeric_placeholder_rejected(self):
        p=self.fixture();p['open']='{{open}}'
        with self.assertRaises(ValueError):validate(p)
    def test_nan_rejected(self):
        p=self.fixture();p['open']=float('nan')
        with self.assertRaises(ValueError):validate(p)
    def test_aliases(self):
        p=self.fixture();p['interval']='5'
        with self.assertRaises(ValueError):validate(p)
    def test_bad_ohlc(self):
        p=self.fixture();p['high']=1
        with self.assertRaises(ValueError):validate(p)
    def test_unknown_field(self):
        p=self.fixture();p['rsi']=50
        with self.assertRaises(ValueError):validate(p)
    def test_null_volume_and_time_close(self):
        p=self.fixture();p['volume']=None;p['bar_close_time']=None;validate(p)
    def test_bool_not_number(self):
        p=self.fixture();p['open']=True
        with self.assertRaises(ValueError):validate(p)
    def test_contracts_json(self):
        for p in Path('finance/contracts').glob('*.json'):
            with self.subTest(p=p):json.loads(p.read_text())
