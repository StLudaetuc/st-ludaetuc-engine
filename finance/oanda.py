"""Conservative offline OANDA export adapter; NO broker calls.

Only reviewed ORDER_FILL / DAILY_FINANCING / TRANSFER_FUNDS profiles accepted.
Top-level account cash amounts are authoritative; nested trade/position amounts
are attribution evidence, never additional ledger cash. Unknown/missing fields
quarantine the whole transaction, not zero-fill it.
"""
from .core import BrokerComponent
from .policy import money, instant, Scope, add

PROFILE = 'oanda_v20_account_cash_v1'

def normalise(transaction, account_alias, expected_account_id, scope, account_currency, fx_to_gbp):
    if not isinstance(scope, Scope) or scope.financial_reality != 'ACTUAL' or scope.financial_state != 'REALISED':
        raise ValueError('explicit actual broker scope required')
    if transaction.get('accountID') != expected_account_id or not expected_account_id:
        raise ValueError('account mismatch')
    for key in ('id', 'time', 'type', 'accountBalance'):
        if transaction.get(key) is None:
            raise ValueError('missing broker field: ' + key)
    money(transaction['accountBalance'])
    instant(transaction['time'])
    kind = transaction['type']
    if kind == 'ORDER_FILL':
        fields = [('pl', 'realised_pnl', 1), ('financing', 'financing', 1),
                  ('commission', 'commission', -1), ('guaranteedExecutionFee', 'other_adjustment', -1)]
    elif kind == 'DAILY_FINANCING':
        fields = [('financing', 'financing', 1)]
    elif kind == 'TRANSFER_FUNDS':
        fields = [('amount', 'capital_transfer', 1)]
    else:
        raise ValueError('unclassified_transaction: ' + kind)
    components = []
    for field, component_kind, sign in fields:
        if transaction.get(field) is None:
            raise ValueError('missing broker field: ' + field)
        value = money(transaction[field])
        if sign < 0 and value < 0:
            raise ValueError('broker fee magnitude must be nonnegative')
        components.append(BrokerComponent(account_alias, scope, transaction['id'], field,
                                          transaction['time'], component_kind, value.copy_negate() if sign < 0 else value,
                                          account_currency, fx_to_gbp))
    # Conversion/spread metrics may already be embedded in home-currency pl.
    # No additive conversion cost without a division-specific reviewed profile.
    diagnostics = {k: transaction[k] for k in ('halfSpreadCost', 'homeConversionFactors', 'tradesClosed',
                    'tradeReduced', 'tradeOpened', 'positionFinancings', 'reason', 'orderID') if k in transaction}
    if kind == 'ORDER_FILL':
        nested = list(transaction.get('tradesClosed') or [])
        if transaction.get('tradeReduced'):
            nested.append(transaction['tradeReduced'])
        if nested:
            if any(not isinstance(r,dict) or r.get('realizedPL') is None for r in nested):
                raise ValueError('nested_attribution_missing')
            if add(money(r['realizedPL']) for r in nested) != money(transaction['pl']):
                raise ValueError('nested_attribution_mismatch')
    if kind == 'DAILY_FINANCING' and transaction.get('positionFinancings'):
        rows = transaction['positionFinancings']
        if any(not isinstance(r,dict) or r.get('financing') is None for r in rows):
            raise ValueError('nested_attribution_missing')
        if add(money(r['financing']) for r in rows) != money(transaction['financing']):
            raise ValueError('nested_attribution_mismatch')
    return tuple(components), diagnostics


def normalise_batch(transactions, account_alias, expected_account_id, scope, account_currency, fx_to_gbp):
    if not isinstance(transactions,(list,tuple)) or len(transactions)>10000:
        raise ValueError('bounded broker export required')
    accepted, exceptions, seen = [], [], {}
    for index, tx in enumerate(transactions):
        try:
            if not isinstance(tx,dict): raise ValueError('malformed_transaction')
            ident = tx.get('id')
            if ident in seen:
                if seen[ident] == tx: continue
                raise ValueError('conflicting_event')
            rows, attribution = normalise(tx, account_alias, expected_account_id, scope, account_currency, fx_to_gbp)
            seen[ident] = dict(tx)
            accepted.append((rows,attribution))
        except (ValueError,TypeError,KeyError,ArithmeticError) as exc:
            exceptions.append({'row_index':index,'state':'INVESTIGATION_REQUIRED','reason':str(exc)})
    return tuple(accepted), tuple(exceptions)
