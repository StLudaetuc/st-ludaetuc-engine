"""Single deterministic Reader/weekly egress boundary; no LLM trusted metadata.

The trusted producer supplies current journal identity, scope, observed counts and
expected values. Consumer prose/JSON cannot self-upgrade certainty or arithmetic.
Schema validation precedes semantic checking; canonical output is sorted/bounded.
"""
import json
from pathlib import Path
from .policy import Scope, Coverage, AuthorityJournal, money, instant, quantise, POLICY_VERSION
from .reconciliation import Run

CASH_ECONOMIC_KEYS={'cash_net_economic','remaining_cash_break_even'}
ECONOMIC_KEYS={'net_economic','net_economic_pnl','remaining_lifetime_break_even','remaining_break_even',
               'period_operating_gap','net_result','net_profit_per_trade'}
TRADING_KEYS={'gross_pnl','gross_trading_pnl','financing','other_broker_costs','net_trading','commission',
              'conversion_costs','other_adjustments','gross_profit_per_trade','financing_per_trade','realised_expectancy'}
GROUPS=('metrics','lifetime','this_week','run_rate','unit_economics','forecast_inputs')

def serialize(payload, kind, *, journal, coverage_key, scope, observed_count, expected_metrics, reconciliation_key):
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
    decision=journal.latest('coverage',coverage_key)
    coverage=decision[3] if decision else None
    reconcile=journal.latest('reconciliation',reconciliation_key) if reconciliation_key is not None else None
    run=reconcile[3] if reconcile else None
    clean=isinstance(run,Run) and run.clean and run.transaction_count == observed_count and run.scope == scope and run.account == payload['account_alias']
    needs_reconciliation=kind=='weekly' or payload.get('product')!='finance-costs'
    complete=isinstance(coverage,Coverage) and coverage.supports(start,end,observed_count) and (clean or not needs_reconciliation)
    if payload['classification']=='FACT' and not complete:
        raise ValueError('incomplete report cannot be FACT')
    if payload['coverage_complete'] is not complete:
        raise ValueError('coverage cannot be self-declared')
    if not isinstance(coverage,Coverage) or (payload['coverage_state'],payload['evidence_count'],payload['first_evidence_id'],payload['last_evidence_id'],payload['unresolved_exception_count']) != (coverage.state,coverage.evidence_count,coverage.first_evidence_id,coverage.last_evidence_id,coverage.unresolved_exceptions):
        raise ValueError('coverage metadata must match current server-ordered evidence')
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
        if metric.get('evidence_id') is not None:
            from .policy import evidence_ref
            evidence_ref(metric['evidence_id'])
        if metric != expected_metrics[(group,key)]:
            raise ValueError('metric differs from deterministic producer')
        if value is not None:
            money(value)
            if metric.get('currency')=='GBP' and quantise(money(value)) != money(value):
                raise ValueError('monetary policy mismatch')
        if classification=='FACT' and (state!='verified' or not complete):
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
        if (group=='unit_economics' or key.endswith('_per_trade')) and value is not None and (payload['denominator_basis']!='closed_trades' or not payload['trade_count']):
            raise ValueError('executed closed-trade denominator required')
        if scope.financial_reality=='SIMULATED' and key in TRADING_KEYS and value is not None and classification!='HYPOTHETICAL':
            raise ValueError('simulated cannot become realised fact')
    if payload['classification']=='FACT' and any(c!='FACT' for c in classifications):
        raise ValueError('mixed estimates/unknown cannot label the report FACT')
    if payload['classification']=='ESTIMATED' and any(c!='ESTIMATED' for c in classifications):
        raise ValueError('classification mismatch')
    return json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False)
