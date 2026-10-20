"""A25-RC2 guarded staging collection kit. Dry-run default. No Render mutator.

Platform step output is an operator worksheet. A future authorised operator must
validate API semantics and execute it outside this sealed author kit.
"""
import argparse
import hashlib
import http.client
import ipaddress
import json
import pathlib
import socket
import time
from urllib.parse import urlsplit

PATHS = ['/analytics/export', '/analytics/strategy-report', '/analytics/prediction-accuracy', '/analytics/export%0A', '//analytics//export', '/%61nalytics/export', '/ANALYTICS/export', '/analytics%2Fexport', '/analytics/export?probe=synthetic']
STEPS = ['inventory', 'positive-control', 'suspend-old', 'build-failure', 'start-failure', 'health-failure', 'protected-deploy', 'verify-new', 'transfer-domain', 'rollback-new-staging-only', 'resume-old-negative-control-staging-only', 're-suspend-old', 'verify-cutover']

def validate_inventory(m, execute=False):
    if m.get('environment') != 'staging' or m.get('production') is not False:
        raise ValueError('requires explicit isolated staging inventory')
    if not m.get('targets') or not m.get('old_service_id') or not m.get('new_service_id') or m['old_service_id'] == m['new_service_id']:
        raise ValueError('requires distinct staging service identities and targets')
    if not m.get('complete_path_inventory') or not m.get('private_peer_inventory'):
        raise ValueError('P1–P13 inventory and private peers required')
    for t in m['targets']:
        u = urlsplit(t['url'])
        if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment or u.path not in ('', '/'):
            raise ValueError('invalid origin URL')
        if t.get('service_id') not in (m['old_service_id'], m['new_service_id']) or t.get('environment') != 'staging':
            raise ValueError('target not a staging service in inventory')
        if t.get('vantage') not in ('external-non-allowlisted', 'private-peer'):
            raise ValueError('explicit vantage required')
    if execute:
        if m.get('approval_kind') != 'CEO_STAGING_ONLY' or not m.get('approval_reference') or m.get('approval_expires_epoch', 0) < time.time():
            raise ValueError('unexpired separate CEO staging approval required')
        if m.get('network_mode') != 'synthetic-loopback':
            # Author kit intentionally cannot send to real staging/Production. Later
            # independent scope must validate a platform adapter and network policy.
            raise ValueError('real platform execution sealed pending separate staging authority and adapter review')
        for t in m['targets']:
            u = urlsplit(t['url'])
            try:
                if not ipaddress.ip_address(u.hostname).is_loopback:
                    raise ValueError('synthetic execution requires literal loopback IP')
            except ValueError:
                raise ValueError('synthetic execution requires literal loopback IP')
    return m

def probe(target, path, method='GET', websocket=False):
    u = urlsplit(target['url'])
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=2)
    headers = {'Host': target.get('host', u.hostname), 'X-Forwarded-Host': 'synthetic.invalid', 'X-Forwarded-Prefix': '/analytics', 'X-Forwarded-For': '192.0.2.1'}
    if websocket:
        headers.update({'Connection': 'Upgrade', 'Upgrade': 'websocket', 'Sec-WebSocket-Version': '13', 'Sec-WebSocket-Key': 'c3ludGhldGljLXRlc3QxMg=='})
    start = time.time()
    try:
        conn.request(method, path, headers=headers)
        r = conn.getresponse()
        # Never store analytics bodies; stop collection immediately on exposure.
        status = r.status
        r.close()
        return {'at': start, 'target_id': target['id'], 'vantage': target['vantage'], 'path': path, 'method': method, 'websocket': websocket, 'status': status, 'exposure': 200 <= status < 300 or status == 101}
    except (OSError, http.client.HTTPException) as e:
        return {'at': start, 'target_id': target['id'], 'vantage': target['vantage'], 'path': path, 'method': method, 'websocket': websocket, 'status': None, 'error': type(e).__name__, 'exposure': False}
    finally:
        conn.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['steps', 'probes', 'trading', 'collect-sql'])
    p.add_argument('inventory', type=pathlib.Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--interval', type=float, default=5)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--source-log', type=pathlib.Path)
    a = p.parse_args()
    m = validate_inventory(json.loads(a.inventory.read_text()), a.execute)
    if not 0 < a.interval <= 10 or not 1 <= a.rounds <= 360:
        p.error('bounded rounds and interval <=10 seconds required')
    if a.mode == 'steps':
        print(json.dumps({'mode': 'OPERATOR_WORKSHEET_ONLY', 'steps': STEPS, 'platform_mutations': False, 'requires': 'separate CEO staging authority, verified Render semantics, isolated resources and operator execution'}, indent=2))
        return
    if not a.execute:
        print(json.dumps({'mode': 'DRY_RUN', 'operation': a.mode, 'target_ids': [t['id'] for t in m['targets']], 'rounds': a.rounds, 'interval': a.interval, 'network_calls': 0, 'sql_source': 'operator-exported staging statement log only'}))
        return
    if a.mode == 'collect-sql':
        if not a.source_log:
            p.error('--source-log required; database connectivity intentionally absent')
        b = a.source_log.read_bytes()
        print(json.dumps({'label': 'operator-exported synthetic SQL log', 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(), 'select_lines': sum('select' in l.lower() for l in b.decode(errors='replace').splitlines()), 'source': str(a.source_log)}))
        return
    for _ in range(a.rounds):
        tick = time.monotonic()
        for t in m['targets']:
            if a.mode == 'trading':
                u = urlsplit(t['url'])
                c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=2)
                body = json.dumps({'event_type': 'market_data', 'origin': 'a25-rc2-synthetic', 'instrument': 'GBPUSD', 'timeframe': '1m', 'time': '2026-01-01T00:00:00Z', 'open': 1.2, 'high': 1.3, 'low': 1.1, 'close': 1.2})
                try:
                    c.request('POST', '/market-data', body=body, headers={'Content-Type': 'application/json', 'Host': t.get('host', u.hostname)})
                    r = c.getresponse(); print(json.dumps({'at': time.time(), 'target': t['id'], 'sent': 1, 'status': r.status, 'acked': 200 <= r.status < 300, 'persisted': 'MUST_VERIFY_FROM_FRESH_DB_CONNECTION'}), flush=True);r.close()
                except OSError as e:
                    print(json.dumps({'at': time.time(), 'target': t['id'], 'sent': 1, 'acked': False, 'error': type(e).__name__}), flush=True)
                finally:
                    c.close()
            else:
                for path in PATHS:
                    for method in ['GET', 'HEAD', 'POST', 'OPTIONS']:
                        row = probe(t, path, method)
                        print(json.dumps(row), flush=True)
                        if row['exposure']:
                            raise SystemExit('STOP: analytics exposure; body not retained')
                row = probe(t, '/analytics/export', websocket=True)
                print(json.dumps(row), flush=True)
                if row['exposure']:
                    raise SystemExit('STOP: websocket exposure')
        elapsed = time.monotonic() - tick
        if elapsed > 10:
            raise SystemExit('STOP: probe coverage exceeded10s; add parallel authorised vantage workers')
        if a.rounds > 1:
            time.sleep(max(0, a.interval - elapsed))

if __name__ == '__main__':
    main()
