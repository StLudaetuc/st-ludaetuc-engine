"""Fail-closed analytics containment. No credentials, SQL or trading changes.

Legacy analytics have no proven account/environment ownership. They cannot be
re-enabled by a permissive configuration flag. Authenticated callers can inspect
access status; data routes return 503 pending an independently reviewed scoped
reader. This intentional availability trade-off is documented in the handover.
"""
import hashlib
import hmac
import json
import sys
import threading
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl

MAX_CONFIG_BYTES = 65536
MAX_QUERY_BYTES = 2048
MAX_WINDOW_DAYS = 31
MAX_ROWS = 1000
NAME = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
TOKEN = re.compile(r'^Bearer ([A-Za-z0-9_-]{1,64})\.([A-Za-z0-9_-]{43,128})$')
PERMISSIONS = frozenset({'analytics:status', 'analytics:export', 'analytics:strategy-report', 'analytics:prediction-accuracy'})
ROUTES = {'/analytics/access-status': 'analytics:status', '/analytics/export': 'analytics:export', '/analytics/strategy-report': 'analytics:strategy-report', '/analytics/prediction-accuracy': 'analytics:prediction-accuracy'}
ENVIRONMENTS = frozenset({'production', 'shadow', 'development', 'test'})
BROKER_ENVIRONMENTS = frozenset({'practice', 'live', 'none'})
AUDIT_LOCK = threading.Lock()

def emit_audit(event):
    # Direct structured stdout avoids silent logger-level filtering. Render log
    # retention/export must be verified by the operator before production use.
    line = json.dumps(event,sort_keys=True,allow_nan=False) + '\n'
    with AUDIT_LOCK:
        if sys.stdout.write(line) != len(line):
            raise OSError('incomplete audit write')
        sys.stdout.flush()


def utc_time(value):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError('invalid time')
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('timezone required')
    return d.astimezone(timezone.utc)


def load_identities(raw, now):
    if not raw or len(raw.encode()) > MAX_CONFIG_BYTES:
        raise ValueError('configuration unavailable')
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    data = json.loads(raw, object_pairs_hook=unique_pairs)
    if not isinstance(data, dict) or set(data) != {'version', 'identities'} or type(data['version']) is not int or data['version'] != 1:
        raise ValueError('invalid configuration')
    ids = data['identities']
    if not isinstance(ids, list) or not 1 <= len(ids) <= 32:
        raise ValueError('invalid identities')
    seen = set()
    digests = set()
    for row in ids:
        if not isinstance(row, dict) or set(row) != {'key_id','service_id','token_sha256','permissions','scopes','expires_at'}:
            raise ValueError('invalid identity')
        if not all(isinstance(row[k], str) and NAME.fullmatch(row[k]) for k in ('key_id','service_id')):
            raise ValueError('invalid identity names')
        digest = row['token_sha256']
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or digest in digests or row['key_id'] in seen:
            raise ValueError('invalid identity digest')
        seen.add(row['key_id']); digests.add(digest)
        if not isinstance(row['permissions'],list) or not row['permissions'] or not all(isinstance(x,str) and x in PERMISSIONS for x in row['permissions']):
            raise ValueError('invalid permissions')
        utc_time(row['expires_at'])
        scopes = row['scopes']
        if not isinstance(scopes,list) or not 1 <= len(scopes) <= 16:
            raise ValueError('invalid scopes')
        for s in scopes:
            if not isinstance(s,dict) or set(s) != {'account_alias','environment','broker_environment'}:
                raise ValueError('invalid scope')
            if not isinstance(s['account_alias'],str) or not NAME.fullmatch(s['account_alias']) or s['environment'] not in ENVIRONMENTS or s['broker_environment'] not in BROKER_ENVIRONMENTS:
                raise ValueError('invalid scope values')
    return ids


class AnalyticsAccessControl:
    """ASGI middleware: no legacy data handler can run through this boundary."""
    def __init__(self, app, config_provider=None, audit_sink=None, clock=None):
        self.app = app
        self.config_provider = config_provider or (lambda: os.environ.get('ANALYTICS_SERVICE_IDENTITIES_JSON',''))
        self.audit_sink = audit_sink or emit_audit
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def __call__(self, scope, receive, send):
        path = scope.get('path','')
        # Protect both the incoming ASGI path and the route path after an ASGI
        # root_path prefix. Uvicorn can prepend root_path to scope['path'], while
        # Starlette removes that prefix before matching application routes.
        # Matching only scope['path'] can silently bypass containment.
        root_path = scope.get('root_path') or ''
        route_path = path
        if isinstance(root_path, str) and root_path and path.startswith(root_path):
            remainder = path[len(root_path):]
            if not remainder or remainder.startswith('/'):
                route_path = remainder or '/'
        def is_analytics(candidate):
            if not isinstance(candidate, str):
                return False
            normalized = '/' + '/'.join(part for part in candidate.split('/') if part)
            folded = normalized.casefold()
            return folded == '/analytics' or folded.startswith('/analytics/')
        protected = is_analytics(path) or is_analytics(route_path)
        if scope['type'] not in ('http','websocket') or not protected:
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        subject = None
        selected_scope = None
        async def finish(status, code, payload=None):
            nonlocal subject
            event = {'event':'analytics_access','request_id':request_id,'at':self.clock().isoformat(),'service_id':subject,'route':path if path in ROUTES else 'unknown_analytics','result':code,'http_status':status,'scope':selected_scope}
            try:
                self.audit_sink(event)
            except Exception:
                status, code, payload = 503, 'audit_unavailable', None
            if scope['type'] == 'websocket':
                await send({'type':'websocket.close','code':1008})
                return
            headers = [(b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'pragma',b'no-cache'),(b'vary',b'Authorization'),(b'x-content-type-options',b'nosniff'),(b'x-request-id',request_id.encode())]
            if status == 401: headers.append((b'www-authenticate',b'Bearer'))
            body = json.dumps(payload if payload is not None else {'error':code,'request_id':request_id},allow_nan=False).encode()
            await send({'type':'http.response.start','status':status,'headers':headers})
            await send({'type':'http.response.body','body':b'' if scope.get('method') == 'HEAD' else body})
        auth = [v for k,v in scope.get('headers',[]) if k.lower() == b'authorization']
        if len(auth) != 1 or len(auth[0]) > 256:
            await finish(401,'unauthorized'); return
        try:
            match = TOKEN.fullmatch(auth[0].decode('ascii'))
        except UnicodeError:
            match = None
        if not match:
            await finish(401,'unauthorized'); return
        try:
            identities = load_identities(self.config_provider(),self.clock())
        except Exception:
            await finish(503,'access_configuration_unavailable'); return
        key_id, token = match.groups()
        digest = hashlib.sha256(token.encode()).hexdigest()
        principal = next((r for r in identities if r['key_id'] == key_id),None)
        if principal is None or not hmac.compare_digest(principal['token_sha256'],digest) or utc_time(principal['expires_at']) <= self.clock():
            await finish(401,'unauthorized'); return
        subject = principal['service_id']
        if scope['type'] == 'websocket' or scope.get('method') != 'GET':
            await finish(405,'method_not_allowed'); return
        if path not in ROUTES:
            await finish(404,'analytics_route_unavailable'); return
        if ROUTES[path] not in principal['permissions']:
            await finish(403,'forbidden'); return
        try:
            raw = scope.get('query_string',b'')
            if len(raw) > MAX_QUERY_BYTES:
                raise ValueError('query too large')
            pairs = parse_qsl(raw.decode('ascii'),keep_blank_values=True,max_num_fields=20,errors='strict')
            query = dict(pairs)
            if len(query) != len(pairs): raise ValueError('duplicate parameter')
            base = {'account_alias','environment','broker_environment'}
            allowed = base | ({'start','end','limit','instrument','timeframe'} if path == '/analytics/export' else {'start','end','limit'} if path == '/analytics/strategy-report' else {'since_hours','lookahead_minutes','max_predictions'} if path == '/analytics/prediction-accuracy' else set())
            if not base <= query.keys() or not query.keys() <= allowed: raise ValueError('query fields')
            wanted = {k:query[k] for k in base}
            if wanted not in principal['scopes']:
                await finish(403,'forbidden_scope'); return
            selected_scope = wanted
            if path in ('/analytics/export','/analytics/strategy-report'):
                if not {'start','end'} <= query.keys():raise ValueError('explicit window required')
                start,end = utc_time(query['start']),utc_time(query['end'])
                if not timedelta(0) < end-start <= timedelta(days=MAX_WINDOW_DAYS) or end > self.clock()+timedelta(minutes=5):raise ValueError('window bounds')
                limit = int(query.get('limit','1000'))
                if not 1 <= limit <= MAX_ROWS:raise ValueError('row bounds')
                for field in ('instrument','timeframe'):
                    if field in query and not re.fullmatch('[A-Za-z0-9_]{1,20}',query[field]):raise ValueError('invalid filter')
            elif path == '/analytics/prediction-accuracy':
                for field, default, maximum in [('since_hours',720,744),('lookahead_minutes',240,1440),('max_predictions',100,100)]:
                    if not 1 <= int(query.get(field,str(default))) <= maximum:raise ValueError('query bounds')
        except Exception:
            await finish(400,'invalid_analytics_query'); return
        if path == '/analytics/access-status':
            await finish(200,'access_status',{'service_id':subject,'scope':selected_scope,'legacy_data_access':'unavailable_pending_verified_scope','request_id':request_id}); return
        # Fail closed even for a valid principal. Legacy views/payloads cannot
        # establish account and environment isolation. No DB query is issued.
        await finish(503,'analytics_scope_unavailable')
