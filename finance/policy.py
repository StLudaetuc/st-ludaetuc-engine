"""Pure validation boundary. No default scope, implicit completeness or clock authority."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from datetime import datetime, timezone
from threading import RLock
import re

POLICY_VERSION = 'money_scale12_half_away_v1'
QUANTUM = Decimal('0.000000000001')
CURRENCIES = frozenset({'GBP', 'USD', 'EUR', 'AED', 'CAD', 'JPY', 'AUD', 'CHF'})
# Candidate registry, not a declaration of CEO policy approval. Other currencies fail closed.

def money(value):
    if not isinstance(value, (str, Decimal)) or isinstance(value, str) and len(value) > 64:
        raise ValueError('bounded decimal string required; never float')
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('invalid decimal') from exc
    if not result.is_finite() or result.copy_abs() >= Decimal('1e18') or result.as_tuple().exponent < -12:
        raise ValueError('invalid monetary precision/magnitude')
    return result

def quantise(value):
    with localcontext() as ctx:
        ctx.prec = 80
        result = value.quantize(QUANTUM, rounding=ROUND_HALF_UP)
        return result.copy_abs() if result == 0 else result

def add(values):
    with localcontext() as ctx:
        ctx.prec = 80
        return quantise(sum(values, Decimal(0)))

def convert(amount, rate):
    with localcontext() as ctx:
        ctx.prec = 80
        return quantise(money(amount) * money(rate))

def divide(numerator, denominator):
    with localcontext() as ctx:
        ctx.prec = 80
        if denominator == 0:
            raise ValueError('zero denominator')
        return quantise(numerator / denominator)

def instant(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid timestamp') from exc
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('explicit timezone required')
    return result.astimezone(timezone.utc)

def currency(value):
    if value not in CURRENCIES:
        raise ValueError('currency outside candidate governed registry')
    return value

def evidence_ref(value):
    """Evidence IDs only. No dereferenceable URLs, paths, credentials or spreadsheets."""
    if not isinstance(value, str) or not re.fullmatch(r'(invoice|receipt|evidence|manifest|attestation|statement|source|billing|fx|assumption):[A-Za-z0-9_.-]{1,160}', value):
        raise ValueError('unsafe evidence reference; use an opaque evidence ID')
    return value

def sheet_text(value):
    if not isinstance(value, str) or len(value) > 512 or any(ord(c) < 32 for c in value):
        raise ValueError('invalid/bounded spreadsheet text')
    # Preserve idempotence; apostrophe is a display escape, never executable formula.
    return "'" + value if value.lstrip().startswith(('=', '+', '-', '@')) else value

@dataclass(frozen=True)
class Scope:
    broker_environment: str
    system_environment: str
    financial_reality: str
    financial_state: str
    simulation_type: str | None
    def __post_init__(self):
        if self.broker_environment not in {'OANDA_PRACTICE', 'OANDA_LIVE'}:
            raise ValueError('explicit broker environment required')
        if self.system_environment not in {'DEVELOPMENT','VALIDATION','DEMO','SHADOW','REPLAY','PRODUCTION'}:
            raise ValueError('explicit system environment required')
        if self.financial_reality not in {'ACTUAL', 'SIMULATED'} or self.financial_state not in {'REALISED', 'UNREALISED'}:
            raise ValueError('explicit financial dimensions required')
        if self.financial_reality == 'SIMULATED':
            if self.simulation_type not in {'COUNTERFACTUAL','SHADOW','REPLAY','VALIDATION'} or self.financial_state == 'REALISED':
                raise ValueError('simulated cannot be realised broker cash')
        elif self.simulation_type is not None:
            raise ValueError('ACTUAL cannot have simulation context')
    @property
    def live_cash(self):
        return (self.broker_environment == 'OANDA_LIVE' and self.system_environment == 'PRODUCTION'
                and self.financial_reality == 'ACTUAL' and self.financial_state == 'REALISED'
                and self.simulation_type is None)

@dataclass(frozen=True)
class Coverage:
    state: str
    evidence_count: int
    period_start: datetime
    period_end: datetime
    first_evidence_id: str | None
    last_evidence_id: str | None
    manifest_ref: str
    reviewed_by: str
    unresolved_exceptions: int
    currency_confirmed: bool
    costs_complete: bool
    def __post_init__(self):
        object.__setattr__(self, 'period_start', instant(self.period_start))
        object.__setattr__(self, 'period_end', instant(self.period_end))
        evidence_ref(self.manifest_ref)
        if type(self.currency_confirmed) is not bool or type(self.costs_complete) is not bool:
            raise ValueError('currency and cost-completeness attestations required')
        if self.state not in {'VERIFIED','REVIEWED_EMPTY','UNKNOWN','INCOMPLETE','WITHDRAWN'}:
            raise ValueError('invalid coverage state')
        if type(self.evidence_count) is not int or self.evidence_count < 0 or type(self.unresolved_exceptions) is not int or self.unresolved_exceptions < 0:
            raise ValueError('evidence counts required')
        if self.period_end <= self.period_start:
            raise ValueError('invalid coverage period')
        if self.state == 'VERIFIED' and (self.evidence_count == 0 or not self.first_evidence_id or not self.last_evidence_id):
            raise ValueError('VERIFIED requires evidence count and range')
        if self.state == 'REVIEWED_EMPTY' and (self.evidence_count != 0 or self.first_evidence_id is not None or self.last_evidence_id is not None):
            raise ValueError('reviewed empty must be explicitly empty')
        if self.state in {'VERIFIED','REVIEWED_EMPTY'} and not self.reviewed_by:
            raise ValueError('reviewed evidence required')
    def supports(self, start, end, observed_count):
        return (type(observed_count) is int and self.state in {'VERIFIED','REVIEWED_EMPTY'} and not self.unresolved_exceptions and self.currency_confirmed and self.costs_complete
                and self.period_start <= instant(start) < instant(end) <= self.period_end
                and observed_count == self.evidence_count)

class AuthorityJournal:
    """Trusted in-process reference; durable authority uses INSERT-only SQL functions.

    Acceptance order is assigned under server lock; callers cannot supply identity
    or receive a mutable journal list. Source timestamps never sort authority.
    """
    def __init__(self):
        self._lock, self._rows = RLock(), []
    def append(self, domain, key, value, source_at):
        with self._lock:
            sequence = len(self._rows) + 1
            self._rows.append((sequence, domain, key, value, instant(source_at)))
            return sequence
    def latest(self, domain, key):
        with self._lock:
            return next((r for r in reversed(self._rows) if r[1:3] == (domain, key)), None)
    def coverage_for(self, key, start, end):
        start,end=instant(start),instant(end)
        with self._lock:
            return next((r for r in reversed(self._rows) if r[1:3]==('coverage',key) and isinstance(r[3],Coverage) and (r[3].period_start,r[3].period_end)==(start,end)),None)
    @property
    def history(self):
        with self._lock:
            return tuple(self._rows)


def elapsed_seconds(delta):
    with localcontext() as ctx:
        ctx.prec = 80
        return Decimal(delta.days)*86400 + Decimal(delta.seconds) + Decimal(delta.microseconds)/1000000


def multiply(left, right):
    with localcontext() as ctx:
        ctx.prec = 80
        return quantise(left * right)


def prorate(amount, overlap, full):
    with localcontext() as ctx:
        ctx.prec = 80
        return quantise(amount * overlap / full)
