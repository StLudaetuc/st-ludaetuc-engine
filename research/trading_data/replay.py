"""Hypothetical R-only research replay; never broker/accounting truth."""
from .evidence import timestamp,number

def feature_at(value,available_at,decision_at):
    if timestamp(available_at)>timestamp(decision_at):raise ValueError('LOOKAHEAD_FEATURE')
    return value

def replay(plan,candles,*,decision_at,horizon_end,spread,slippage,coverage_verified,prerequisites_verified):
    if not isinstance(candles,(list,tuple)) or len(candles)>10000:raise ValueError('bounded replay')
    base={'classification':'HYPOTHETICAL','financial_reality':'SIMULATED','net_r':None,'is_rejection_error':False}
    def result(state,**kw):return dict(base,state=state,**kw)
    try:
        start,end=timestamp(decision_at),timestamp(horizon_end)
        if end<=start:raise ValueError('invalid horizon')
        direction=plan['direction'];entry,sl,tp=(number(plan[k]) for k in ('entry','stop_loss','take_profit'))
        if direction not in ('long','short') or (direction=='long' and not 0<sl<entry<tp) or (direction=='short' and not 0<tp<entry<sl):raise ValueError('invalid plan geometry')
        if plan.get('order_type') not in ('MARKET_NEXT_BAR','LIMIT'):raise ValueError('explicit execution semantics')
        if plan.get('plan_available_at') is None or timestamp(plan['plan_available_at'])>start:raise ValueError('plan unavailable at decision')
    except (ValueError,KeyError,TypeError):return result('INVALID_OR_UNRECONSTRUCTABLE_PLAN')
    if coverage_verified is not True:return result('INSUFFICIENT_DATA')
    if prerequisites_verified is not True:return result('EXECUTION_PREREQUISITES_UNPROVED')
    if spread is None or slippage is None:return result('COST_ASSUMPTIONS_MISSING')
    spread,slippage=number(spread),number(slippage)
    if spread<0 or slippage<0 or spread>=abs(entry-sl):return result('ABNORMAL_COST_ASSUMPTIONS')
    rows=[]
    for c in candles:
        try:
            opened,closed=timestamp(c['bar_open']),timestamp(c['bar_close'])
            if closed<=opened:raise ValueError('bar interval')
            if opened<=start or closed>end:continue # never consume decision-bar future range
            o,h,l,cl=(number(c[k]) for k in ('open','high','low','close'))
            if not 0<l<=min(o,cl)<=max(o,cl)<=h:raise ValueError('OHLC')
            rows.append((opened,c,o,h,l))
        except (ValueError,TypeError,KeyError):return result('INSUFFICIENT_DATA')
    if not rows:return result('INSUFFICIENT_DATA')
    rows.sort(key=lambda r:r[0]);filled=False;fill_price=None
    for opened,c,o,h,l in rows:
        entering=not filled
        if entering:
            if plan['order_type']=='LIMIT' and not l<=entry<=h:continue
            fill_price=(o if plan['order_type']=='MARKET_NEXT_BAR' else entry)+(spread/2+slippage)*(1 if direction=='long' else -1)
            if (direction=='long' and not sl<fill_price<tp) or (direction=='short' and not tp<fill_price<sl):return result('GAP_INVALIDATES_PLAN')
            filled=True
        # Candles are midpoint proxies; exits use adverse side and slippage.
        stop_hit=l-spread/2<=sl if direction=='long' else h+spread/2>=sl
        target_hit=h-spread/2>=tp if direction=='long' else l+spread/2<=tp
        if stop_hit and target_hit:return result('AMBIGUOUS_SAME_CANDLE',filled=True)
        if entering and plan['order_type']=='LIMIT' and (stop_hit or target_hit):return result('AMBIGUOUS_ENTRY_EXIT_SEQUENCE',filled=True)
        if stop_hit or target_hit:
            exit_price=(sl if stop_hit else tp)-slippage*(1 if direction=='long' else -1)
            risk=abs(fill_price-sl)
            r=(exit_price-fill_price)*(1 if direction=='long' else -1)/risk
            return result('HYPOTHETICAL_LOSS' if r<0 else 'HYPOTHETICAL_PROFIT' if r>0 else 'HYPOTHETICAL_BREAKEVEN',net_r=str(r),filled=True,assumptions='EXPLICIT_MIDPOINT_SPREAD_SLIPPAGE_PROXY')
    return result('HYPOTHETICAL_NOT_FILLED' if not filled else 'HORIZON_OPEN',filled=filled)
