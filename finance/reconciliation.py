"""Bounded, deterministic reconciliation; exceptions/resolutions are append-only.

Broker slices use transaction_id and OPEN/REDUCE/CLOSE (legacy closed bool can
be accepted only inside explicit scope). Outcome is a lifetime trade rollup.
Pre-EOD missing closure outcomes are pending; late matching arrivals resolve via
new runs. Event time, receipt time and server decision order are distinct.
"""
from collections import defaultdict
from threading import RLock
from functools import wraps
from dataclasses import dataclass
from .policy import Scope, instant, currency, money, add, AuthorityJournal, evidence_ref

STATES = {'MATCHED','BROKER_ONLY','INTERNAL_ONLY','VALUE_MISMATCH','PENDING_EOD_SYNC','INVESTIGATION_REQUIRED'}

@dataclass(frozen=True)
class Result:
    trade_id: str
    state: str
    reasons: tuple
    broker_slice_count: int
    event_time: object = None
    received_at: object = None

@dataclass(frozen=True)
class Run:
    results: tuple
    malformed: tuple
    duplicate_replays: int
    attempted_orders: int
    filled_orders: int
    cancelled_orders: int
    closed_trades: int
    partial_close_slices: int
    transaction_count: int
    account: str
    scope: Scope
    schedule_verified: bool = False
    @property
    def clean(self):
        return not self.malformed and all(r.state == 'MATCHED' for r in self.results)


def reconcile_run(broker, opened, outcomes, predictions, *, as_of, outcome_due_at, account, scope):
    as_of, due = instant(as_of), instant(outcome_due_at)
    if not isinstance(scope,Scope) or not account:
        raise ValueError('explicit reconciliation scope required')
    if any(not isinstance(rows,(list,tuple)) or len(rows)>10000 for rows in (broker,opened,outcomes,predictions)):
        raise ValueError('bounded input collections required')
    errors, conflicts, replay = [], set(), 0
    def parse(rows, kind):
        nonlocal replay
        output, seen = [], {}
        for row in rows:
            try:
                if not isinstance(row,dict): raise ValueError('malformed_record')
                ident = row.get('signal_id') if kind == 'prediction' else row.get('trade_id')
                if not isinstance(ident,str) or not ident or len(ident)>128: raise ValueError('missing_identity')
                if row.get('account') != account or row.get('scope') != scope: raise ValueError('scope_missing')
                time, receipt = instant(row.get('event_time')), instant(row.get('received_at'))
                key = row.get('transaction_id') if kind == 'broker' else row.get('event_id')
                if not isinstance(key,str) or not key or len(key)>128: raise ValueError('missing_event_identity')
                if kind in {'broker','outcome'}:
                    if not row.get('currency'): raise ValueError('currency_missing')
                    currency(row['currency'])
                normal = dict(row, event_time=time, received_at=receipt)
                if kind == 'broker':
                    action = row.get('action')
                    if action is None:
                        if type(row.get('closed')) is not bool: raise ValueError('broker_state_missing')
                        action = 'CLOSE' if row['closed'] else 'OPEN'
                    if action not in {'OPEN','REDUCE','CLOSE'}: raise ValueError('broker_state_missing')
                    normal['action'] = action
                    if action != 'OPEN':
                        if row.get('realised_pnl') is None: raise ValueError('pnl_missing')
                        normal['realised_pnl'] = money(row['realised_pnl'])
                if kind == 'outcome':
                    if row.get('realised_pnl') is None: raise ValueError('pnl_missing')
                    normal['realised_pnl'] = money(row['realised_pnl'])
                if key in seen:
                    if {k:v for k,v in seen[key].items() if k!='received_at'} == {k:v for k,v in normal.items() if k!='received_at'}:
                        replay += 1; continue
                    conflicts.add(ident); raise ValueError('conflicting_event')
                seen[key] = normal; output.append(normal)
            except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
                ident = row.get('trade_id') or row.get('signal_id') or '' if isinstance(row,dict) else ''
                if not isinstance(ident,str) or len(ident)>128:ident=''
                errors.append((kind, ident, str(exc)))
                if ident: conflicts.add(ident)
        return output
    b,o,t,p = parse(broker,'broker'),parse(opened,'opened'),parse(outcomes,'outcome'),parse(predictions,'prediction')
    grouped = defaultdict(list)
    for row in b: grouped[row['trade_id']].append(row)
    opened_map, outcome_map = {}, {}
    for rows,target,kind in ((o,opened_map,'opened'),(t,outcome_map,'outcome')):
        for row in rows:
            key=row['trade_id']
            if key in target:
                conflicts.add(key);errors.append((kind,key,'duplicate_identity'))
            else: target[key]=row
    signals = {row['signal_id'] for row in p}
    results = []
    for ident in sorted(set(grouped)|set(opened_map)|set(outcome_map)|conflicts):
        slices = grouped.get(ident,[]); opened_row=opened_map.get(ident); outcome=outcome_map.get(ident)
        reasons=[]; state='MATCHED'
        cancelled = opened_row and opened_row.get('fill_status') == 'CANCELLED'
        if ident in conflicts:
            state='INVESTIGATION_REQUIRED';reasons.append('malformed_or_conflicting_evidence')
        elif cancelled:
            if slices or outcome:
                state='INVESTIGATION_REQUIRED';reasons.append('cancelled_order_has_trade')
        elif not slices:
            state='PENDING_EOD_SYNC' if as_of < due else 'INTERNAL_ONLY';reasons.append('pending_broker_sync' if as_of < due else 'broker_record_missing')
        elif opened_row is None:
            state='BROKER_ONLY';reasons.append('opened_missing')
        elif opened_row.get('fill_status') != 'FILLED':
            state='INVESTIGATION_REQUIRED';reasons.append('fill_status_missing')
        elif opened_row.get('signal_id') not in signals:
            state='INVESTIGATION_REQUIRED';reasons.append('prediction_missing')
        else:
            cash_slices = [r for r in slices if r['action'] in {'REDUCE','CLOSE'}]
            closed = any(r['action']=='CLOSE' for r in slices)
            if outcome is None and closed:
                state='PENDING_EOD_SYNC' if as_of < due else 'BROKER_ONLY'
                reasons.append('pending_eod' if as_of < due else 'outcome_overdue')
            elif outcome is not None:
                if not closed:
                    state='PENDING_EOD_SYNC' if as_of < due else 'INVESTIGATION_REQUIRED';reasons.append('pending_broker_closure' if as_of < due else 'outcome_without_broker_closure')
                elif len({r['currency'] for r in cash_slices}|{outcome['currency']}) != 1:
                    state='VALUE_MISMATCH';reasons.append('currency_mismatch')
                elif add(r['realised_pnl'] for r in cash_slices) != outcome['realised_pnl']:
                    state='VALUE_MISMATCH';reasons.append('pnl_mismatch')
        last=max(slices,key=lambda r:(r['event_time'],r['transaction_id'])) if slices else outcome or opened_row
        results.append(Result(ident,state,tuple(reasons),len(slices),last.get('event_time') if last else None,last.get('received_at') if last else None))
    return Run(tuple(results),tuple(sorted(errors)),replay,len(o),sum(r.get('fill_status')=='FILLED' for r in o),
               sum(r.get('fill_status')=='CANCELLED' for r in o),sum(r.state=='MATCHED' and opened_map.get(r.trade_id,{}).get('fill_status')=='FILLED' and any(s['action']=='CLOSE' for s in grouped.get(r.trade_id,[])) for r in results),
               sum(r['action']=='REDUCE' for r in b),len(b),account,scope)


def reconcile(broker,opened,outcomes,predictions,as_of,outcome_due_at,account,scope):
    """Diagnostic adapter for old tuple consumers; governed output is reconcile_run."""
    run=reconcile_run(broker,opened,outcomes,predictions,as_of=as_of,outcome_due_at=outcome_due_at,account=account,scope=scope)
    issues=list(run.malformed)
    for r in run.results:
        issues.extend(('trade',r.trade_id,reason) for reason in r.reasons)
    return tuple(sorted(issues))

def locked(method):
    @wraps(method)
    def wrapper(self,*args,**kwargs):
        with self._lock:return method(self,*args,**kwargs)
    return wrapper

class ReconciliationJournal:
    def __init__(self,account,scope):
        if not account or not isinstance(scope,Scope):raise ValueError('explicit journal scope required')
        self.account,self.scope=account,scope
        self._lock=RLock()
        self._journal=AuthorityJournal();self._issues={};self._orders={}
    @locked
    def record(self, run_id, run, source_at):
        if run_id in self._issues: raise ValueError('immutable run identity')
        if not isinstance(run,Run) or run.account!=self.account or run.scope!=self.scope: raise ValueError('scoped Run required')
        self._issues[run_id]=run
        order=self._journal.append('run','CURRENT',(run_id,run),source_at)
        self._orders[run_id]=order
        return order
    @locked
    def resolve(self, run_id, trade_id, new_run_id, reviewer, evidence, source_at):
        evidence_ref(evidence)
        if not reviewer or run_id not in self._issues or new_run_id not in self._issues:
            raise ValueError('reviewed resolution and new evidence run required')
        old=self._issues[run_id];new=self._issues[new_run_id]
        if self._orders[new_run_id]<=self._orders[run_id]:raise ValueError('resolution requires later server-accepted evidence')
        if not any(r.trade_id==trade_id and r.state!='MATCHED' for r in old.results):
            raise ValueError('unresolved source exception required')
        if new.malformed or not any(r.trade_id==trade_id and r.state=='MATCHED' for r in new.results):
            raise ValueError('resolution must be supported by matched new evidence')
        return self._journal.append('resolution',(run_id,trade_id),('RESOLVED',new_run_id,reviewer,evidence),source_at)
    @locked
    def withdraw(self,run_id,trade_id,reviewer,evidence,source_at):
        evidence_ref(evidence)
        if not reviewer or run_id not in self._issues: raise ValueError('reviewed withdrawal required')
        return self._journal.append('resolution',(run_id,trade_id),('WITHDRAWN',None,reviewer,evidence),source_at)
    @locked
    def latest_resolution(self,run_id,trade_id):
        return self._journal.latest('resolution',(run_id,trade_id))

    @locked
    def latest_run(self):
        return self._journal.latest('run','CURRENT')


def reconcile_scheduled(broker, opened, outcomes, predictions, *, event_at, as_of, account, scope, schedule, sync_runs):
    from .schedule import sync_status
    status=sync_status(event_at,as_of,schedule,sync_runs)
    run=reconcile_run(broker,opened,outcomes,predictions,as_of=as_of,outcome_due_at=status['due_at'],account=account,scope=scope)
    from dataclasses import replace
    run=replace(run,schedule_verified=status['verified'])
    if status['state']=='INVESTIGATION_REQUIRED':
        from dataclasses import replace
        results=tuple(replace(r,state='INVESTIGATION_REQUIRED',reasons=r.reasons+(status['reason'],)) if status['state']=='INVESTIGATION_REQUIRED' else r for r in run.results)
        return replace(run,results=results)
    return run
