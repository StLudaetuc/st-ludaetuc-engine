"""Bounded offline reader reference. Identity verification is an external gate.
No network server, public endpoint, live SQL or arbitrary query execution here.
"""
import copy
from dataclasses import dataclass
from datetime import timedelta
from .evidence import timestamp

@dataclass(frozen=True)
class Principal:
    subject: str
    authenticated: bool
    account_aliases: frozenset
    permissions: frozenset

FIELDS=frozenset(('event_id','kind','event_time','received_at','instrument','timeframe','account_alias','broker_environment'))
PAYLOAD_FIELDS=frozenset(('signal_id','broker_trade_id','broker_trade_ids','transaction_id','action','units','direction','fill_status','approval_status','rejection_reason_code','strategy_version','entry_model','setup_class','session_name','open','high','low','close','volume','atr','rsi','stop_loss','take_profit','exit_reason','prediction_linked','state','reasons'))

def query(rows,principal,*,account_alias,start,end,limit=100,instrument=None,search_id=None,audit_log):
    if not isinstance(principal,Principal) or principal.authenticated is not True or not principal.subject or 'research:read' not in principal.permissions or account_alias not in principal.account_aliases:
        audit_log.append({'operation':'research:read','decision':'DENIED','subject':principal.subject[:160] if isinstance(principal,Principal) and isinstance(principal.subject,str) else None})
        raise PermissionError('deny by default')
    a,b=timestamp(start),timestamp(end)
    if b<=a or b-a>timedelta(days=31) or type(limit) is not int or not 1<=limit<=500:raise ValueError('bounded read window/limit')
    if not isinstance(rows,(list,tuple)) or len(rows)>10000:raise ValueError('bounded source snapshot')
    if search_id is not None and (not isinstance(search_id,str) or len(search_id)>160):raise ValueError('bounded identity lookup')
    selected=[]
    for r in sorted(rows,key=lambda r:(r.get('event_time',''),r.get('event_id',''))):
        if r.get('account_alias')!=account_alias or not a<=timestamp(r['event_time'])<b:continue
        if instrument is not None and r.get('instrument')!=instrument:continue
        p=r.get('payload',{})
        if search_id is not None and search_id not in (p.get('signal_id'),p.get('broker_trade_id')):continue
        safe={k:copy.deepcopy(v) for k,v in r.items() if k in FIELDS}
        safe['payload']={k:copy.deepcopy(v) for k,v in p.items() if k in PAYLOAD_FIELDS and (v is None or type(v) in (bool,int,float) or isinstance(v,str) and len(v)<=256 or isinstance(v,list) and len(v)<=100 and all(isinstance(x,str) and len(x)<=160 for x in v))}
        # Never expose raw requests/responses/account snapshots or financial records.
        selected.append(safe)
    audit_log.append({'subject':principal.subject,'operation':'research:read','account_alias':account_alias,'start':start,'end':end,'limit':limit,'returned':min(limit,len(selected))})
    return {'rows':selected[:limit],'truncated':len(selected)>limit,'snapshot_count':len(selected),'source_completeness':'UNPROVED','financial_result':'DEFER_TO_FINANCE'}
