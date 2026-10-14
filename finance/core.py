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

from .policy import (money, instant, currency, quantise, convert, divide, add,
                     Scope, Coverage, AuthorityJournal, evidence_ref, sheet_text, elapsed_seconds, multiply, prorate)
from threading import RLock

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
    paid_at: datetime | None = None
    recognition_policy: str | None = None

    def __post_init__(self):
        for key in ('original_amount', 'fx_to_gbp'):
            object.__setattr__(self, key, money(getattr(self, key)))
        for key in ('incurred_at', 'billing_start', 'billing_end', 'fx_observed_at'):
            object.__setattr__(self, key, instant(getattr(self, key)))
        currency(self.currency)
        if self.paid_at is not None:
            object.__setattr__(self, 'paid_at', instant(self.paid_at))
        if self.recognition_policy not in {None, 'straight_line_seconds_v1', 'point_in_time_v1'}:
            raise ValueError('unapproved recognition policy')
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
        return convert(self.original_amount, self.fx_to_gbp)

class ExpenseLedger:
    def __init__(self):
        self._lock = RLock()
        self.verifications = AuthorityJournal()
        self._history = {}
        self._sources = {}

    def append(self, row):
        with self._lock:
            return self._append(row)

    def _append(self, row):
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

    def verify(self, expense_id, revision, state, reviewer, evidence, source_at):
        evidence_ref(evidence)
        if state not in STATES | {'withdrawn'} or not reviewer:
            raise ValueError('reviewed verification decision required')
        with self._lock:
            if expense_id not in self._history or self._history[expense_id][-1].revision != revision:
                raise ValueError('verification must reference current revision')
            value = ('unverified' if state=='withdrawn' else state, reviewer, revision, evidence)
            return self.verifications.append('verification', expense_id, value, source_at)

    @property
    def history(self):
        with self._lock:
            return tuple(row for key in sorted(self._history) for row in self._history[key])

    @property
    def effective(self):
        with self._lock:
            return self._effective()

    def _effective(self):
        rows = []
        for key in sorted(self._history):
            row = self._history[key][-1]
            decision = self.verifications.latest('verification', key)
            if decision is not None:
                state, reviewer, revision, evidence = decision[3]
                row = replace(row, verification=state if row.revision==revision else 'unverified', verified_by=reviewer if row.revision==revision else '')
            rows.append(row)
        return tuple(rows)

    def spend(self, start, end, verified=True):
        start, end = instant(start), instant(end)
        if end <= start:
            raise ValueError('empty period')
        with localcontext() as ctx:
            ctx.prec = 80
            return sum((r.gbp for r in self.effective if start <= r.incurred_at < end and
                        (not verified or r.verification in {'invoice_verified', 'manual_verified'})), Decimal(0))

def import_expenses(stream, ledger):
    """Bounded untrusted staging. Verification is a separate trusted review action."""
    from dataclasses import fields
    text = stream.read(1048577)
    if len(text.encode('utf-8')) > 1048576:
        raise ValueError('CSV exceeds 1 MiB')
    import io
    with ledger._lock:
        candidate = ExpenseLedger()
        for row in ledger.history:
            candidate.append(row)
        reader = csv.DictReader(io.StringIO(text))
        allowed = {f.name for f in fields(Expense)}
        required = {f.name for f in fields(Expense) if f.default is __import__('dataclasses').MISSING}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) - allowed or not required <= set(reader.fieldnames):
            raise ValueError('invalid CSV header')
        count = 0
        for index, raw in enumerate(reader):
            if index >= 1000 or None in raw or any(v is None or len(v) > 512 for v in raw.values()):
                raise ValueError('bounded CSV rows/fields required')
            if raw['verification'] not in {'estimate', 'unverified'} or raw.get('verified_by'):
                raise ValueError('import cannot self-verify')
            for key in ('expense_id','source_key'):
                if not re.fullmatch(r'[A-Za-z0-9_:.\-]{1,200}', raw[key]):
                    raise ValueError('unsafe identity')
            evidence_ref(raw['evidence_ref'])
            for key in ('vendor','description','fx_source','service','environment','correction_reason'):
                if key in raw:
                    raw[key] = sheet_text(raw[key])
            raw['revision'] = int(raw['revision'])
            if raw['recurring'] not in ('true', 'false'):
                raise ValueError('recurring must be true/false')
            raw['recurring'] = raw['recurring'] == 'true'
            for key in ('paid_at','recognition_policy'):
                if key in raw and raw[key] in ('', 'None'):
                    raw[key] = None
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

def run_rate(subscriptions, as_of, coverage):
    """Known zero, excluded estimates, and UNKNOWN are different states."""
    as_of = instant(as_of)
    ids, active, verified = set(), [], []
    for row in subscriptions:
        if row.effective_from <= as_of < row.effective_to:
            if row.subscription_id in ids:
                raise ValueError('overlapping subscription schedules')
            ids.add(row.subscription_id); active.append(row)
            if row.verification in {'invoice_verified','manual_verified'}:
                verified.append(row)
    complete = (isinstance(coverage, Coverage) and coverage.supports(coverage.period_start, coverage.period_end, len(active))
                and len(active) == len(verified) and coverage.period_start <= as_of < coverage.period_end)
    month = add(r.monthly_gbp for r in verified) if complete else None
    return {'state': 'VERIFIED' if complete else 'UNKNOWN', 'month': month,
            'year': multiply(month,Decimal(12)) if month is not None else None,
            'day': divide(multiply(month,Decimal(12)), Decimal('365.25')) if month is not None else None,
            'observed_month': add(r.monthly_gbp for r in verified),
            'excluded_estimates': sum(r.verification=='estimate' for r in active),
            'unknown_records':sum(r.verification=='unverified' for r in active), 'evidence_count': len(active)}

@dataclass(frozen=True)
class BrokerComponent:
    account: str
    scope: Scope
    transaction_id: str
    component_key: str
    at: datetime
    kind: str
    amount: Decimal
    currency: str
    fx_to_gbp: Decimal
    trade_id: str = ''
    def __post_init__(self):
        if not isinstance(self.scope, Scope) or self.kind not in {'realised_pnl', 'financing', 'commission', 'conversion', 'other_adjustment', 'capital_transfer'}:
            raise ValueError('invalid broker classification')
        if not all((self.account, self.transaction_id, self.component_key)):
            raise ValueError('broker identity required')
        object.__setattr__(self, 'at', instant(self.at))
        currency(self.currency)
        object.__setattr__(self, 'amount', money(self.amount))
        object.__setattr__(self, 'fx_to_gbp', money(self.fx_to_gbp))
        if self.fx_to_gbp <= 0 or (self.currency == 'GBP' and self.fx_to_gbp != 1):
            raise ValueError('invalid broker FX')

def broker_totals(components, account, scope, start, end, coverage):
    start, end = instant(start), instant(end)
    if end <= start or not isinstance(scope, Scope) or not account:
        raise ValueError('explicit broker scope/period required')
    seen, totals = {}, {k: Decimal(0) for k in ('realised_pnl','financing','commission','conversion','other_adjustment','capital_transfer')}
    for row in components:
        if row.account != account or row.scope != scope or not start <= row.at < end:
            continue
        key = (row.transaction_id, row.component_key)
        if key in seen:
            if seen[key] != row:
                raise ValueError('conflicting broker duplicate')
            continue
        seen[key] = row
        totals[row.kind] = add([totals[row.kind], convert(row.amount, row.fx_to_gbp)])
    count = len({key[0] for key in seen})
    complete = isinstance(coverage, Coverage) and coverage.supports(start, end, count)
    observed = add(totals[k] for k in ('realised_pnl','financing','commission','conversion','other_adjustment'))
    return {**totals, 'observed_net_trading': observed, 'net_trading': observed if complete else None,
            'coverage_complete': complete, 'evidence_count': count, 'scope': scope, 'account': account}


def expense_totals(ledger, start, end, coverage):
    """Cash and economic recognition are distinct; policy absence is unavailable."""
    start, end = instant(start), instant(end)
    if end <= start:
        raise ValueError('valid period required')
    # Coverage count is ALL effective expenditure evidence, not just the paid cohort.
    rows = list(ledger.effective)
    verified = [r for r in rows if r.verification in {'invoice_verified','manual_verified'}]
    complete = isinstance(coverage, Coverage) and coverage.supports(start, end, len(rows)) and len(rows) == len(verified)
    cash_known = all(r.paid_at is not None for r in verified)
    cash = add(r.gbp for r in verified if r.paid_at is not None and start <= r.paid_at < end)
    recognised, unknown = [], []
    for row in verified:
        if row.recognition_policy is None:
            unknown.append(row.expense_id)
        elif row.recognition_policy == 'point_in_time_v1':
            if start <= row.incurred_at < end:
                recognised.append(row.gbp)
        else:
            full = elapsed_seconds(row.billing_end-row.billing_start)
            left = min(full,max(Decimal(0),elapsed_seconds(start-row.billing_start)))
            right = min(full,max(Decimal(0),elapsed_seconds(end-row.billing_start)))
            # Difference of cumulative rounded allocations: adjacent periods sum
            # exactly to the full converted amount, including the final residue.
            recognised.append(add([prorate(row.gbp,right,full),prorate(row.gbp,left,full).copy_negate()]))
    return {'cash_spend': cash if complete and cash_known else None,
            'economic_expense': add(recognised) if complete and not unknown else None,
            'coverage_complete': complete, 'evidence_count': len(rows), 'policy_missing': tuple(unknown),
            'observed_cash_spend': cash, 'recognition_policy': 'explicit_per_expense'}


def project_economics(accounts, expenses, account_inventory, start, end, *, trade_count, denominator_basis):
    """Sum eligible broker cash FIRST, subtract project spend ONCE. No defaults."""
    if type(trade_count) is not int or trade_count < 0 or denominator_basis not in {'attempted_orders','filled_orders','closed_trades','partial_close_slices','transactions'}:
        raise ValueError('explicit denominator required')
    aliases = [r['account'] for r in accounts]
    if len(set(aliases)) != len(aliases):
        raise ValueError('duplicate account')
    if not all(isinstance(r.get('scope'), Scope) for r in accounts):
        raise ValueError('explicit account dimensions required')
    eligible = [r for r in accounts if r['scope'].live_cash]
    inventory_complete = (isinstance(account_inventory,Coverage) and account_inventory.supports(start,end,len(accounts)))
    broker_complete = inventory_complete and all(r.get('coverage_complete') is True and r.get('net_trading') is not None for r in eligible)
    net = add(r['net_trading'] for r in eligible) if broker_complete else None
    cash = expenses.get('cash_spend')
    economic = expenses.get('economic_expense')
    complete = broker_complete and expenses.get('coverage_complete') is True and cash is not None
    cash_result = add([net,cash.copy_negate()]) if complete else None
    economic_result = add([net,economic.copy_negate()]) if broker_complete and expenses.get('coverage_complete') is True and economic is not None else None
    valid_denominator = denominator_basis == 'closed_trades' and trade_count > 0 and bool(eligible)
    return {'scope_level':'PROJECT', 'eligible_accounts':len(eligible), 'coverage_complete':complete,
            'actual_net_trading_gbp':net, 'cash_spend_gbp':cash, 'economic_expense_gbp':economic,
            'cash_net_economic_gbp':cash_result, 'net_economic_gbp':economic_result,
            'remaining_cash_break_even_gbp':quantise(max(cash_result.copy_negate(),Decimal(0))) if cash_result is not None else None,
            'operationally_profitable':economic_result > 0 if economic_result is not None else None,
            'allocated_cost_per_trade_gbp':divide(economic,Decimal(trade_count)) if valid_denominator and economic_result is not None else None,
            'net_profit_per_trade_gbp':divide(economic_result,Decimal(trade_count)) if valid_denominator and economic_result is not None else None,
            'denominator_basis':denominator_basis, 'trade_count':trade_count,
            'practice_profit_applied_to_break_even':False}


def practice_unit_economics(totals, *, trade_count, denominator_basis):
    if not isinstance(totals.get('scope'),Scope) or totals['scope'].broker_environment != 'OANDA_PRACTICE':
        raise ValueError('Practice scope required')
    if type(trade_count) is not int or trade_count < 0 or denominator_basis != 'closed_trades':
        raise ValueError('closed-trade denominator required; cancelled attempts excluded')
    value = totals.get('net_trading')
    return {'basis':'HYPOTHETICAL_NON_CASH','actual_net_profit':None,
            'hypothetical_net_per_trade':divide(value,Decimal(trade_count)) if value is not None and trade_count else None,
            'trade_count':trade_count,'denominator_basis':denominator_basis}

# Re-export governed lifecycle. No row-level parse failure may abort the run.
from .reconciliation import reconcile, reconcile_run, ReconciliationJournal
