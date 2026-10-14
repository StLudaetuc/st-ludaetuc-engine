"""Offline raw-evidence validation candidate, not connected to engine routes."""
import math

FIELDS={'schema_version','source','ticker','ticker_id','provider','asset_type','base_currency','quote_currency',
        'timeframe','bar_open_time','bar_close_time','open','high','low','close','volume','minimum_tick',
        'is_confirmed','interval','time'}

def validate(payload):
    if not isinstance(payload,dict) or set(payload)!=FIELDS:
        raise ValueError('bounded market-data-v2 object required')
    if payload['schema_version']!='market_data_v2' or payload['source']!='tradingview':
        raise ValueError('unsupported schema/source')
    for key,limit in [('ticker',64),('ticker_id',128),('provider',64),('asset_type',32),('timeframe',16),('interval',16)]:
        value=payload[key]
        if not isinstance(value,str) or not 0<len(value)<=limit or '{' in value or '}' in value:
            raise ValueError('invalid raw identity: '+key)
    for key in ('base_currency','quote_currency'):
        value=payload[key]
        if value is not None and (not isinstance(value,str) or not 1<=len(value)<=12 or '{' in value or '}' in value):
            raise ValueError('invalid raw currency')
    for key in ('bar_open_time','bar_close_time','time'):
        value=payload[key]
        if key=='bar_close_time' and value is None:continue
        if type(value) is not int or not 0<=value<=9007199254740991:
            raise ValueError('UTC epoch milliseconds required')
    if payload['bar_close_time'] is not None and payload['bar_close_time']<=payload['bar_open_time']:
        raise ValueError('invalid bar times')
    for key in ('open','high','low','close','volume','minimum_tick'):
        value=payload[key]
        if key=='volume' and value is None:continue
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('finite raw number required: '+key)
    if payload['minimum_tick']<=0 or (payload['volume'] is not None and payload['volume']<0):
        raise ValueError('invalid tick/volume')
    if not payload['low']<=min(payload['open'],payload['close'])<=max(payload['open'],payload['close'])<=payload['high']:
        raise ValueError('invalid OHLC envelope')
    if type(payload['is_confirmed']) is not bool:
        raise ValueError('explicit confirmation required')
    if payload['interval']!=payload['timeframe'] or payload['time']!=payload['bar_open_time']:
        raise ValueError('legacy aliases must preserve lineage')
    return payload
