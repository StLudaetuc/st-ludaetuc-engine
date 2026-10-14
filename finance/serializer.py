"""Single deterministic Reader/weekly egress boundary; no LLM trusted metadata.

The trusted producer supplies current journal identity, scope, observed counts and
expected values. Consumer prose/JSON cannot self-upgrade certainty or arithmetic.
Schema validation precedes semantic checking; canonical output is sorted/bounded.
"""
import json
from pathlib import Path
from .policy import Scope, Coverage, AuthorityJournal, money, instant, quantise, POLICY_VERSION
from .reconciliation import Run

# Exhaustive metric taxonomy. Unknown schema additions must fail closed.
CASH_ECONOMIC_KEYS={'cash_net_economic','remaining_cash_break_even'}
ECONOMIC_KEYS={'net_economic','net_economic_pnl','remaining_lifetime_break_even','remaining_break_even',
               'period_operating_gap','net_result','net_profit_per_trade'}
TRADING_KEYS={'gross_pnl','gross_trading_pnl','financing','other_broker_costs','net_trading','commission',
              'conversion_costs','other_adjustments','gross_profit_per_trade','financing_per_trade',
              'realised_expectancy','observed_expectancy','realised_pnl','broker_costs','other_trading_costs'}
COST_KEYS={'spent','cash_spend','economic_expense','infrastructure','api_ai','subscriptions','rd_testing',
           'domains','data','tooling','other','total_spend','lifetime_spend','total_spent','other_operating_costs',
           'monthly_operating_cost','operating_costs','cost_per_day','cost_per_month','cost_per_year','cost_per_trade'}
ESTIMATE_KEYS={'estimated_spend'}
FORECAST_KEYS={'capital','financing_assumption','drawdown_assumption'}
METRIC_REGISTRY={**{k:'CASH_ECONOMIC' for k in CASH_ECONOMIC_KEYS},**{k:'ECONOMIC' for k in ECONOMIC_KEYS},
 **{k:'TRADING' for k in TRADING_KEYS},**{k:'COST' for k in COST_KEYS},**{k:'ESTIMATE' for k in ESTIMATE_KEYS},
 **{k:'FORECAST' for k in FORECAST_KEYS}}
GROUPS=('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs')

def serialize(payload, kind, *, journal, coverage_key, scope, observed_count, expected_metrics, reconciliation_key, metric_contexts=None, policy_approvals=None):
    if kind not in {'reader','weekly'} or not isinstance(journal,AuthorityJournal) or not isinstance(scope,Scope):
        raise ValueError('explicit trusted producer context required')
    encoded=json.dumps(payload,allow_nan=False,ensure_ascii=False)
    if len(encoded.encode())>65536:
        raise ValueError('bounded 64 KiB output required')
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError('Finance serializer dependency unavailable; refuse output') from exc
    name='reader_products' if kind=='reader' else 'weekly_report'
    schema=json.loads((Path(__file__).parent/'contracts'/(name+'.schema.json')).read_text())
    Draft202012Validator(schema,format_checker=FormatChecker()).validate(payload)
    wire_scope=Scope(payload['broker_environment'],payload['system_environment'],payload['financial_reality'],payload['financial_state'],payload['simulation_type'])
    if wire_scope!=scope or payload['money_policy']!=POLICY_VERSION:
        raise ValueError('producer scope/precision mismatch')
    start,end=instant(payload['period_start']),instant(payload['period_end'])
    if end<=start: raise ValueError('invalid report period')
    decision=journal.coverage_for(coverage_key,start,end)
    coverage=decision[3] if decision else None
    reconcile=journal.latest('reconciliation',reconciliation_key) if reconciliation_key is not None else None
    run=reconcile[3] if reconcile else None
    clean=isinstance(run,Run) and run.clean and run.schedule_verified and run.transaction_count == observed_count and run.scope == scope and run.account == payload['account_alias']
    needs_reconciliation=kind=='weekly' or payload.get('product')!='finance-costs'
    complete=isinstance(coverage,Coverage) and coverage.supports(start,end,observed_count) and (clean or not needs_reconciliation)
    if payload['classification']=='FACT' and not complete:
        raise ValueError('incomplete report cannot be FACT')
    if payload['coverage_complete'] is not complete:
        raise ValueError('coverage cannot be self-declared')
    if not isinstance(coverage,Coverage) or (payload['coverage_state'],payload['evidence_count'],payload['first_evidence_id'],payload['last_evidence_id'],payload['unresolved_exception_count']) != (coverage.state,coverage.evidence_count,coverage.first_evidence_id,coverage.last_evidence_id,coverage.unresolved_exceptions):
        raise ValueError('coverage metadata must match current server-ordered evidence')
    if payload['source_manifest_id']!=coverage.manifest_ref:
        raise ValueError('source manifest must match current trusted evidence')
    if payload['currency_confirmed'] is not coverage.currency_confirmed or payload['costs_complete'] is not coverage.costs_complete:
        raise ValueError('currency/cost completeness cannot be self-declared')
    if isinstance(run,Run):
        priority=('INVESTIGATION_REQUIRED','VALUE_MISMATCH','BROKER_ONLY','INTERNAL_ONLY','PENDING_EOD_SYNC','MATCHED')
        states={r.state for r in run.results}
        canonical='INVESTIGATION_REQUIRED' if run.malformed else next((s for s in priority if s in states),'MATCHED')
        if payload['reconciliation_state']!=canonical:
            raise ValueError('reconciliation state cannot be rewritten')
        counts={'attempted_orders':run.attempted_orders,'filled_orders':run.filled_orders,'closed_trades':run.closed_trades,
                'partial_close_slices':run.partial_close_slices,'transactions':run.transaction_count}
        if payload['trade_count'] is not None and payload['trade_count']!=counts[payload['denominator_basis']]:
            raise ValueError('denominator must match deterministic reconciliation population')
        exceptions=len(run.malformed)+sum(r.state!='MATCHED' for r in run.results)
        pending=sum(r.state=='PENDING_EOD_SYNC' for r in run.results)
        if payload['exception_count']!=exceptions or payload['pending_eod_count']!=pending:
            raise ValueError('exception/pending counts cannot be rewritten')
    if payload['reconciliation_state']=='MATCHED' and not clean:
        raise ValueError('no verified clean reconciliation')
    if scope.broker_environment=='OANDA_PRACTICE' and payload['basis']!='HYPOTHETICAL_NON_CASH':
        raise ValueError('Practice cannot masquerade as actual cash economics')
    metrics=[]
    for group in GROUPS:
        for key,value in payload.get(group,{}).items():
            metrics.append((group,key,value))
    if set(expected_metrics) != {(g,k) for g,k,_ in metrics}:
        raise ValueError('trusted expected-value manifest must cover every metric')
    classifications=[]
    for group,key,metric in metrics:
        state,classification=metric['state'],metric['classification'];value=metric['value']
        classifications.append(classification)
        if key not in METRIC_REGISTRY: raise ValueError('unregistered financial metric')
        if state=='verified' and (value is None or classification!='FACT'): raise ValueError('verified metric requires non-null FACT')
        if key in ESTIMATE_KEYS and value is not None and (state!='estimate' or classification!='ESTIMATED'):
            raise ValueError('estimated spend cannot become FACT')
        # Every asserted metric needs its own trusted window context. Weekly
        # lifetime and recurring evidence must never borrow final-week coverage.
        metric_complete=complete
        if value is not None:
            context=(metric_contexts or {}).get((group,key))
            if not isinstance(context,dict): raise ValueError('metric reporting-window context required')
            ms,me=instant(context['start']),instant(context['end'])
            domain='BROKER' if key in TRADING_KEYS else 'PROJECT' if key in ECONOMIC_KEYS|CASH_ECONOMIC_KEYS else 'RECURRING' if group=='run_rate' else 'SNAPSHOT_OR_ASSUMPTION' if key in FORECAST_KEYS else 'EXPENSES'
            if context.get('evidence_domain')!=domain: raise ValueError('metric evidence domain mismatch')
            window_kind='lifetime' if group=='lifetime' or key in {'lifetime_spend','remaining_lifetime_break_even'} else ('recurring' if group=='run_rate' else 'period')
            if context.get('window_kind')!=window_kind or me<=ms: raise ValueError('metric window kind mismatch')
            if window_kind=='period' and (ms,me)!=(start,end): raise ValueError('period metric window mismatch')
            if window_kind=='lifetime':
                inception=journal.latest('project_inception',context.get('inception_key'))
                if not inception or ms!=instant(inception[3]) or ms>start or me!=end: raise ValueError('trusted lifetime inception window required')
            md=journal.coverage_for(context['coverage_key'],ms,me);mc=md[3] if md else None
            metric_complete=isinstance(mc,Coverage) and mc.supports(ms,me,context['observed_count'])
            if key in TRADING_KEYS|ECONOMIC_KEYS|CASH_ECONOMIC_KEYS:
                rd=journal.latest('reconciliation',context['reconciliation_key']);rr=rd[3] if rd else None
                metric_complete=metric_complete and isinstance(rr,Run) and rr.clean and rr.schedule_verified and rr.account==payload['account_alias'] and rr.scope==scope and rr.transaction_count==context['observed_count']
        if classification=='FACT' and not metric_complete: raise ValueError('metric window evidence incomplete')
        if value is not None and classification=='FACT' and metric.get('currency')=='GBP':
            required={'money_policy','reporting_currency','fx_source'}
            if key in ECONOMIC_KEYS|{'economic_expense'}: required.add('recognition')
            if not required<=set(policy_approvals or {}): raise ValueError('unratified policy: metric unavailable or conditional only')
            if any(not isinstance(policy_approvals[k],str) or not policy_approvals[k].startswith('attestation:') for k in required): raise ValueError('policy approval evidence required')
        if metric.get('evidence_id') is not None:
            from .policy import evidence_ref
            evidence_ref(metric['evidence_id'])
        if metric != expected_metrics[(group,key)]:
            raise ValueError('metric differs from deterministic producer')
        if value is not None:
            money(value)
            if metric.get('currency')=='GBP' and quantise(money(value)) != money(value):
                raise ValueError('monetary policy mismatch')
        if classification=='FACT' and (state!='verified' or not metric_complete):
            raise ValueError('ESTIMATED/UNKNOWN/incomplete cannot be FACT')
        if state in {'estimate','approved_assumption'} and classification!='ESTIMATED':
            raise ValueError('assumptions/estimates must remain estimates')
        if state=='unavailable' and (value is not None or classification!='UNKNOWN' or not metric.get('reason')):
            raise ValueError('unavailable requires null and explicit reason')
        if state=='observed_incomplete' and classification!='UNKNOWN':
            raise ValueError('incomplete evidence is UNKNOWN')
        if state=='hypothetical' and classification!='HYPOTHETICAL':
            raise ValueError('hypothetical must be labelled non-cash')
        if key in ECONOMIC_KEYS|CASH_ECONOMIC_KEYS and value is not None and (not scope.live_cash or not complete or not clean):
            raise ValueError('economic profit/break-even requires actual live complete clean evidence')
        if key in ECONOMIC_KEYS|{'economic_expense'} and value is not None and payload['recognition_policy_missing_count']:
            raise ValueError('unapproved/missing recognition policy cannot become economic fact')
        if key in TRADING_KEYS and scope.broker_environment=='OANDA_PRACTICE' and value is not None and classification!='HYPOTHETICAL':
            raise ValueError('Practice trading metrics are hypothetical non-cash')
        if group=='unit_economics' and scope.broker_environment=='OANDA_PRACTICE' and value is not None and classification!='HYPOTHETICAL': raise ValueError('Practice unit economics must be hypothetical')
        if (group=='unit_economics' or key.endswith('_per_trade')) and value is not None and (payload['denominator_basis']!='closed_trades' or not payload['trade_count']):
            raise ValueError('executed closed-trade denominator required')
        if scope.financial_reality=='SIMULATED' and key in TRADING_KEYS and value is not None and classification!='HYPOTHETICAL':
            raise ValueError('simulated cannot become realised fact')
    if payload['classification']=='FACT' and any(c!='FACT' for c in classifications):
        raise ValueError('mixed estimates/unknown cannot label the report FACT')
    if payload['classification']=='ESTIMATED' and any(c!='ESTIMATED' for c in classifications):
        raise ValueError('classification mismatch')
    return json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False)
