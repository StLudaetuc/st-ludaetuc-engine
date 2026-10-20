"""AUTHOR ONLY synthetic database adapter for real engine HTTP/model challenges.
No PostgreSQL claim. Engine source remains unmodified. No external connections.
"""
import importlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT, APP, PORT, LOG = sys.argv[1:5]
sys.path.insert(0, str(Path(ROOT).resolve()))
import engine
LOCK=threading.Lock()
def event(kind, **fields):
    with LOCK:
        with open(LOG,'a') as f:f.write(json.dumps({'kind':kind,**fields})+'\n')
class Cursor:
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def execute(self,sql,params=()):event('SYNTHETIC_SQL_CALL',verb=sql.strip().split()[0].upper())
    def fetchall(self):return []
    def fetchone(self):return None
class Connection:
    def cursor(self,*a,**kw):return Cursor()
    def commit(self):event('SYNTHETIC_COMMIT_CALL_NOT_DURABILITY')
    def rollback(self):pass
class Pool:
    def getconn(self):return Connection()
    def putconn(self,*a):pass
engine._get_pool=lambda:Pool()
codes={getattr(engine,n).__code__:n for n in ('strategy_report','analytics_export','prediction_accuracy','_analytics_query')}
sys.monitoring.use_tool_id(4,'rc2-author-hitmon')
def on_start(code,offset):
    if code in codes:event('ANALYTICS_HANDLER',name=codes[code])
sys.monitoring.register_callback(4,sys.monitoring.events.PY_START,on_start)
for c in codes:sys.monitoring.set_local_events(4,c,sys.monitoring.events.PY_START)
import uvicorn
uvicorn.run(APP,host='127.0.0.1',port=int(PORT),log_level='error')
