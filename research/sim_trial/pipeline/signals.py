"""Trend targets for the simulation trial, computed exactly as dp.run() in research/daily_portfolio does.

target weight (fraction of equity, signed) = signal * VOL_TARGET / annual_vol / sqrt(N)
signal = mean(sign(close / close.shift(L) - 1) for L in LOOKBACKS), only once 252 days of history exist."""
import numpy as np
import pandas as pd
import requests

from config import LOOKBACKS, VOL_SPAN, VOL_MIN_PERIODS, VOL_TARGET, OANDA_PRACTICE_BASE


def tsmom(closes: pd.DataFrame) -> pd.DataFrame:
    s = sum(np.sign(closes / closes.shift(L) - 1) for L in LOOKBACKS) / float(len(LOOKBACKS))
    return s.where(closes.shift(max(LOOKBACKS)).notna())


def target_weights(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily target weights known at each close (same formula as dp.run's `tgt`)."""
    vol = closes.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_MIN_PERIODS).std() * np.sqrt(252)
    sig = tsmom(closes)
    navail = sig.notna().sum(axis=1).clip(lower=1)
    tgt = (sig * VOL_TARGET / vol).div(np.sqrt(navail), axis=0)
    return tgt.replace([np.inf, -np.inf], np.nan)


def latest_targets(closes: pd.DataFrame):
    """Targets from the last complete close, plus diagnostics for the run log."""
    tgt = target_weights(closes)
    last = closes.index[-1]
    vol = closes.pct_change().ewm(span=VOL_SPAN, min_periods=VOL_MIN_PERIODS).std() * np.sqrt(252)
    return {
        "as_of_close": str(last.date()),
        "weights": {k: (None if pd.isna(v) else float(v)) for k, v in tgt.iloc[-1].items()},
        "signal": {k: (None if pd.isna(v) else float(v)) for k, v in tsmom(closes).iloc[-1].items()},
        "annual_vol": {k: (None if pd.isna(v) else float(v)) for k, v in vol.iloc[-1].items()},
        "last_close": {k: float(v) for k, v in closes.iloc[-1].items()},
    }


def fetch_oanda_daily(instruments, token, count=400, session=None):
    """Daily mid closes from the OANDA practice API, 09:00 New York alignment (as in the backtest).
    Only complete candles are used. Index = New York date of the candle open."""
    s = session or requests.Session()
    out = {}
    for inst in instruments:
        r = s.get(f"{OANDA_PRACTICE_BASE}/instruments/{inst}/candles",
                  params={"granularity": "D", "price": "M", "count": count,
                          "dailyAlignment": 9, "alignmentTimezone": "America/New_York"},
                  headers={"Authorization": f"Bearer {token}"}, timeout=30)
        r.raise_for_status()
        rows = [(c["time"], float(c["mid"]["c"])) for c in r.json()["candles"] if c.get("complete")]
        idx = pd.to_datetime([t for t, _ in rows], utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
        ser = pd.Series([v for _, v in rows], index=idx)
        out[inst] = ser[~ser.index.duplicated(keep="last")]
    return pd.DataFrame(out).sort_index()
