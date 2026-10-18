"""Bounded denial checks. Production requires explicit operator approval flag.
No authentication, query params, response bodies, redirects or trading requests.
"""
import argparse,http.client,json,sys
from urllib.parse import urlsplit
ROUTES=('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy')
def target(url,approved=False):
 u=urlsplit(url)
 if u.username or u.password or u.query or u.fragment or u.path not in ('','/') or not u.hostname:raise ValueError('invalid origin')
 local=u.hostname in ('127.0.0.1','localhost','::1')
 if local:
  if u.scheme!='http':raise ValueError('local HTTP origin required')
 elif not (approved and u.scheme=='https' and u.hostname=='st-ludaetuc-engine.onrender.com' and u.port in (None,443)):
  raise ValueError('remote check requires exact approved production origin and explicit operator approval')
 return u

def verify(url,mode='primary',approved=False):
 u=target(url,approved);rows=[]
 expected=401 if mode=='primary' else 503
 for route in ROUTES:
  cls=http.client.HTTPSConnection if u.scheme=='https' else http.client.HTTPConnection
  c=cls(u.hostname,u.port,timeout=5)
  try:
   c.request('GET',route,headers={'Cache-Control':'no-cache'})
   r=c.getresponse()
   # Never read or save bodies, including an unexpected data-bearing success.
   rows.append({'route':route,'status':r.status,'expected':expected,'passed':r.status==expected})
  except (OSError,http.client.HTTPException) as exc:rows.append({'route':route,'passed':False,'error_class':type(exc).__name__})
  finally:c.close()
 return {'mode':mode,'requests':rows,'passed':all(x['passed'] for x in rows),'limits':'status-only; does not establish body semantics, cache/proxy path coverage, current SHA or trading/database health'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--origin',required=True);p.add_argument('--mode',choices=['primary','fallback'],default='primary');p.add_argument('--operator-approved-production-check',action='store_true');a=p.parse_args()
 r=verify(a.origin,a.mode,a.operator_approved_production_check);print(json.dumps(r,indent=2));sys.exit(not r['passed'])
