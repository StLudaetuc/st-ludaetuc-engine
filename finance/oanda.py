"""Conservative offline OANDA export adapter; NO broker calls.

Only reviewed ORDER_FILL / DAILY_FINANCING / TRANSFER_FUNDS profiles accepted.
Top-level account cash amounts are authoritative; nested trade/position amounts
are attribution evidence, never additional ledger cash. Unknown/missing fields
quarantine the whole transaction, not zero-fill it.
"""
from .core import BrokerComponent, money, instant

PROFILE = 'oanda_v20_account_cash_v1'

def normalise(transaction, account_alias, expected_account_id, environment, account_currency, fx_to_gbp):
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
        components.append(BrokerComponent(account_alias, environment, transaction['id'], field,
                                          transaction['time'], component_kind, value.copy_negate() if sign < 0 else value,
                                          account_currency, fx_to_gbp))
    # Conversion/spread metrics may already be embedded in home-currency pl.
    # No additive conversion cost without a division-specific reviewed profile.
    diagnostics = {k: transaction[k] for k in ('halfSpreadCost', 'homeConversionFactors', 'tradesClosed',
                    'tradeReduced', 'tradeOpened', 'positionFinancings', 'reason', 'orderID') if k in transaction}
    return tuple(components), diagnostics
