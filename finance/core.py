"""Deterministic accounting primitives. No network, credentials or LLM arithmetic.

All money enters as decimal strings. Estimates and Practice results never become
actual project economics. Immutable revisions replace the effective value, not
the original evidence. SQL is the persistence contract; these pure functions are
the executable reporting and import-validation reference.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import csv
import re

STATES = {'invoice_verified', 'manual_verified', 'estimate', 'unverified'}
CATEGORIES = {'infrastructure', 'api_ai', 'subscriptions', 'rd_testing', 'domains', 'data', 'tooling', 'other'}

def money(value):
    if not isinstance(value, (str, Decimal)):
        raise ValueError('money must be a decimal string, never float')
    result = Decimal(value)
    if not result.is_finite() or abs(result) >= Decimal('1e18') or result.as_tuple().exponent < -12:
        raise ValueError('non-finite, excessive magnitude or precision')
    return result

def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timezone required')
    return result.astimezone(timezone.utc)

def currency(value):
    if not re.fullmatch('[A-Z]{3}', value):
        raise ValueError('currency must be uppercase three-letter code')
    return value

@dataclass(frozen=True)
class Expense:
    expense_id: str
    revision: int
    source_key: str
    incurred_at: datetime
    vendor: str
    description: str
    category: str
    original_amount: Decimal
    currency: str
    fx_to_gbp: Decimal
    fx_source: str
    fx_observed_at: datetime
    evidence_ref: str
    verification: str
    service: str
    environment: str
    billing_start: datetime
    billing_end: datetime
    recurring: bool = False
    correction_reason: str = ''
    verified_by: str = ''

    def __post_init__(self):
        for key in ('original_amount', 'fx_to_gbp'):
            object.__setattr__(self, key, money(getattr(self, key)))
        for key in ('incurred_at', 'billing_start', 'billing_end', 'fx_observed_at'):
            object.__setattr__(self, key, instant(getattr(self, key)))
        currency(self.currency)
        if self.revision < 1 or not all((self.expense_id, self.source_key, self.vendor, self.description,
                                       self.fx_source, self.evidence_ref, self.service, self.environment)):
            raise ValueError('missing identity or evidence')
        if self.fx_to_gbp <= 0 or (self.currency == 'GBP' and self.fx_to_gbp != 1):
            raise ValueError('invalid GBP conversion')
        if self.billing_end <= self.billing_start or self.category not in CATEGORIES or self.verification not in STATES:
            raise ValueError('invalid classification or billing period')
        if self.verification in {'invoice_verified', 'manual_verified'} and not self.verified_by:
            raise ValueError('verified expenditure needs verifier')
        if self.revision > 1 and not self.correction_reason:
            raise ValueError('correction reason required')

    @property
    def gbp(self):
        with localcontext() as ctx:
            ctx.prec = 80
            return self.original_amount * self.fx_to_gbp

class ExpenseLedger:
    def __init__(self):
        self._history = {}
        self._sources = {}

    def append(self, row):
        history = self._history.get(row.expense_id, ())
        if row.revision <= len(history):
            if history[row.revision - 1] == row:
                return False
            raise ValueError('conflicting duplicate revision')
        if row.revision != len(history) + 1:
            raise ValueError('revision gap')
        owner = self._sources.get(row.source_key)
        if owner is not None and owner != row.expense_id:
            raise ValueError('duplicate invoice/source line')
        if history and row.source_key != history[0].source_key:
            raise ValueError('source identity cannot change')
        self._sources[row.source_key] = row.expense_id
        self._history[row.expense_id] = history + (row,)
        return True

    @property
    def history(self):
        return tuple(row for key in sorted(self._history) for row in self._history[key])

    @property
    def effective(self):
        return tuple(self._history[key][-1] for key in sorted(self._history))

    def spend(self, start, end, verified=True):
        start, end = instant(start), instant(end)
        if end <= start:
            raise ValueError('empty period')
        with localcontext() as ctx:
            ctx.prec = 80
            return sum((r.gbp for r in self.effective if start <= r.incurred_at < end and
                        (not verified or r.verification in {'invoice_verified', 'manual_verified'})), Decimal(0))

def import_expenses(stream, ledger):
    """Atomic validation: conflicting rows reject the whole batch. CSV money stays text."""
    candidate = ExpenseLedger()
    for row in ledger.history:
        candidate.append(row)
    count = 0
    for raw in csv.DictReader(stream):
        raw['revision'] = int(raw['revision'])
        if raw['recurring'] not in ('true', 'false'):
            raise ValueError('recurring must be true/false')
        raw['recurring'] = raw['recurring'] == 'true'
        count += candidate.append(Expense(**raw))
    ledger._history, ledger._sources = candidate._history, candidate._sources
    return count

@dataclass(frozen=True)
class Subscription:
    subscription_id: str
    monthly_gbp: Decimal
    effective_from: datetime
    effective_to: datetime
    verification: str
    def __post_init__(self):
        object.__setattr__(self, 'monthly_gbp', money(self.monthly_gbp))
        object.__setattr__(self, 'effective_from', instant(self.effective_from))
        object.__setattr__(self, 'effective_to', instant(self.effective_to))
        if self.monthly_gbp < 0 or self.effective_to <= self.effective_from or self.verification not in STATES:
            raise ValueError('invalid subscription')

def run_rate(subscriptions, as_of):
    """Annualised convention: month * 12, year / 365.25; never actual spend."""
    as_of = instant(as_of)
    ids = set()
    with localcontext() as ctx:
        ctx.prec = 80
        monthly = Decimal(0)
        for r in subscriptions:
            if r.effective_from <= as_of < r.effective_to:
                if r.subscription_id in ids:
                    raise ValueError('overlapping subscription schedules')
                ids.add(r.subscription_id)
                if r.verification in {'invoice_verified', 'manual_verified'}:
                    monthly += r.monthly_gbp
        return {'month': monthly, 'year': monthly * 12, 'day': monthly * 12 / Decimal('365.25')}

@dataclass(frozen=True)
class BrokerComponent:
    account: str
    environment: str
    transaction_id: str
    component_key: str
    at: datetime
    kind: str
    amount: Decimal
    currency: str
    fx_to_gbp: Decimal
    trade_id: str = ''
    def __post_init__(self):
        if self.environment not in {'practice', 'live'} or self.kind not in {'realised_pnl', 'financing', 'commission', 'conversion', 'other_adjustment', 'capital_transfer'}:
            raise ValueError('invalid broker classification')
        if not all((self.account, self.transaction_id, self.component_key)):
            raise ValueError('broker identity required')
        object.__setattr__(self, 'at', instant(self.at))
        currency(self.currency)
        object.__setattr__(self, 'amount', money(self.amount))
        object.__setattr__(self, 'fx_to_gbp', money(self.fx_to_gbp))
        if self.fx_to_gbp <= 0 or (self.currency == 'GBP' and self.fx_to_gbp != 1):
            raise ValueError('invalid broker FX')

def broker_totals(components, account, environment, start, end):
    start, end = instant(start), instant(end)
    if end <= start or environment not in {'practice', 'live'} or not account:
        raise ValueError('invalid broker scope/period')
    seen, totals = {}, {k: Decimal(0) for k in ('realised_pnl', 'financing', 'commission', 'conversion', 'other_adjustment', 'capital_transfer')}
    with localcontext() as ctx:
        ctx.prec = 80
        for row in components:
            if row.account != account or row.environment != environment or not start <= row.at < end:
                continue
            key = (row.transaction_id, row.component_key)
            if key in seen:
                if seen[key] != row:
                    raise ValueError('conflicting broker duplicate')
                continue
            seen[key] = row
            totals[row.kind] += row.amount * row.fx_to_gbp
        totals['net_trading'] = sum((totals[k] for k in ('realised_pnl', 'financing', 'commission', 'conversion', 'other_adjustment')), Decimal(0))
    return totals

def economics(spend, totals, trades, environment, complete, *, system_environment='PRODUCTION',
              financial_reality='ACTUAL', financial_state='REALISED', simulation_type=None):
    """Observed subtotals remain labelled. Incomplete truth suppresses conclusions."""
    spend = money(spend)
    if type(complete) is not bool or isinstance(trades, bool) or not isinstance(trades, int) or trades < 0 or environment not in {'practice', 'live'}:
        raise ValueError('invalid scope or denominator')
    net = totals['net_trading']
    with localcontext() as ctx:
        ctx.prec = 80
        actual_live = (environment == 'live' and system_environment == 'PRODUCTION' and
                       financial_reality == 'ACTUAL' and financial_state == 'REALISED' and simulation_type is None)
        economic = net - spend if complete and actual_live else None
        return {'observed_spend_gbp': spend, 'observed_net_trading_gbp': net,
                'complete': complete, 'environment': environment,
                'net_economic_gbp': economic,
                'remaining_break_even_gbp': max(-economic, Decimal(0)) if economic is not None else None,
                'operationally_profitable': economic > 0 if economic is not None else None,
                'allocated_cost_per_trade_gbp': spend / trades if complete and trades else None,
                'net_profit_per_trade_gbp': (net - spend) / trades if complete and trades else None,
                'unit_economics_basis': 'practice_broker_actual' if environment == 'practice' else 'scoped_live'}

def reconcile(broker, opened, outcomes, predictions, as_of, outcome_due_at, account, environment):
    """Scoped input lists, one account/environment, explicit EOD deadline supplied.

    Broker closure records contain trade_id, realised_pnl (account currency).
    Opened records use trade_id=broker id and signal_id=Pine id. Outbound adapters
    must retain account/environment scope, partial-close transactions and evidence.
    """
    as_of, outcome_due_at = instant(as_of), instant(outcome_due_at)
    if not account or environment not in {'practice', 'live'}:
        raise ValueError('scope required')
    result = []
    def index(rows, kind, key='trade_id'):
        data = {}
        for row in rows:
            if row.get('account') != account or row.get('environment') != environment:
                result.append((kind, '', 'scope_missing'))
                continue
            if kind in {'broker','outcome'} and not row.get('currency'):
                result.append((kind, row.get(key) or '', 'currency_missing'))
            ident = row.get(key)
            if not ident:
                result.append((kind, '', 'missing_identity'))
                continue
            if ident in data:
                result.append((kind, ident, 'duplicate_identity'))
            data[ident] = row
        return data
    b, o, t, p = index(broker, 'broker'), index(opened, 'opened'), index(outcomes, 'outcome'), index(predictions, 'prediction', 'signal_id')
    for ident in sorted(set(b) | set(o) | set(t)):
        if ident not in b:
            result.append(('trade', ident, 'broker_record_missing'))
        if ident not in o:
            result.append(('trade', ident, 'opened_missing'))
        elif not o[ident].get('signal_id') or o[ident]['signal_id'] not in p:
            result.append(('trade', ident, 'prediction_missing'))
        if ident in b and type(b[ident].get('closed')) is not bool:
            result.append(('trade', ident, 'broker_state_missing'))
        if ident in b and b[ident].get('closed') is True:
            if ident not in t:
                result.append(('trade', ident, 'outcome_overdue' if as_of >= outcome_due_at else 'pending_eod'))
            else:
                left, right = b[ident].get('realised_pnl'), t[ident].get('realised_pnl')
                if left is None or right is None:
                    result.append(('trade', ident, 'pnl_missing'))
                elif not b[ident].get('currency') or not t[ident].get('currency'):
                    pass # explicit currency_missing already emitted; do not compare unknown units
                elif b[ident]['currency'] != t[ident]['currency']:
                    result.append(('trade',ident,'currency_mismatch'))
                elif money(left) != money(right):
                    result.append(('trade', ident, 'pnl_mismatch'))
        elif ident in t:
            result.append(('trade', ident, 'outcome_without_broker_closure'))
    return tuple(sorted(result))
