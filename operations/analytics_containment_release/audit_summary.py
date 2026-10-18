"""Summarise supplied private structured logs; never fetches, prints raw lines or tokens."""
import argparse,collections,json,sys

def summarise(lines):
    counts=collections.Counter();malformed=0;seen=0
    for line in lines:
        seen+=1
        if len(line)>65536:malformed+=1;continue
        try:e=json.loads(line)
        except (ValueError,TypeError):malformed+=1;continue
        if not isinstance(e,dict) or e.get('event')!='analytics_access':continue
        status=e.get('http_status');route=e.get('route');result=e.get('result')
        if type(status) is not int or route not in ('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy','/analytics/access-status','unknown_analytics') or result not in ('unauthorized','analytics_scope_unavailable','audit_unavailable','access_configuration_unavailable','analytics_route_unavailable','method_not_allowed','forbidden','forbidden_scope','invalid_analytics_query','access_status'):
            malformed+=1;continue
        counts[(route,status)]+=1
    unexpected=sum(n for (route,status),n in counts.items() if route in ('/analytics/export','/analytics/strategy-report','/analytics/prediction-accuracy') and 200<=status<300)
    return {'input_lines':seen,'valid_analytics_events':sum(counts.values()),'unclassified_lines':malformed,'counts':[{'route':r,'http_status':s,'count':n} for (r,s),n in sorted(counts.items())],'unexpected_data_2xx':unexpected,'coverage':'SUPPLIED_LOGS_ONLY; not zero traffic if empty; fallback emits no analytics_access events'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('file',type=argparse.FileType('r'));a=p.parse_args();r=summarise(a.file);print(json.dumps(r,indent=2));sys.exit(bool(r['unexpected_data_2xx'] or r['unclassified_lines']))
