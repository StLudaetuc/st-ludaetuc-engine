"""Python replica of pine/research/STL_GBPUSD_15m_LiquiditySweep_RESEARCH_v1.pine.

RESEARCH ONLY - not imported by engine.py, not used for live trading.

Signals are computed on mid prices ((bid+ask)/2) of confirmed 15-minute bars.
Fills use the stored bid/ask: longs buy at the ask and sell at the bid, shorts
the reverse, plus a fixed slippage per fill. Every rule below has a one-line
market reason; defaults are round and were fixed before any result was seen.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, asdict
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
PIP = 0.0001


@dataclass(frozen=True)
class Params:
    # Kill zones in session-timezone minutes since midnight (start inclusive, end exclusive).
    london_kz: tuple = (7 * 60, 11 * 60)        # London open: first liquidity of the day is taken here
    ny_kz: tuple = (12 * 60 + 30, 16 * 60)     # New York open overlap: second liquidity window
    asia: tuple = (0, 7 * 60)                  # Asian range: the overnight pool London hunts
    displacement_bars: int = 4                 # a real reversal shows intent within an hour
    displacement_atr: float = 1.0              # body at least one average bar: not noise
    atr_len: int = 14                          # standard volatility yardstick
    setup_expiry_bars: int = 8                 # 2 hours: a retrace that takes longer is a new market
    stop_buffer_pips: float = 1.5              # ~90th-percentile London/NY spread in this data
    min_stop_pips: float = 5.0                 # below this, ~1.5 pips of costs exceed 30% of R
    rr: float = 2.0                            # minimum reward:risk (fixed target)
    flat_minute: int = 20 * 60 + 45            # be flat before the 21:00-22:00 UTC rollover spread
    skip_nfp: bool = True                      # first-Friday 12:30-14:30 London (approximate NFP calendar)
    use_htf_bias: bool = True                  # do not fade 1H and 4H swing structure
    slippage_pips: float = 0.2                 # per market fill
    cost_mult: float = 1.0                     # stress multiplier on spread + slippage
    entry_mode: str = "next_open"              # "next_open" (brief) | "limit_mid" (sensitivity only)


# ---------------------------------------------------------------- data ----
def load(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, parse_dates=["time_utc"])
    df = df.set_index("time_utc").sort_index()
    for k in "ohlc":
        df[k] = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2.0
    return df


def _atr(h, l, c, n):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    out = np.full(len(tr), np.nan)
    a = None
    for i, x in enumerate(tr):           # Wilder RMA, same as ta.atr
        if i < n - 1:
            continue
        a = np.mean(tr[:n]) if a is None else (a * (n - 1) + x) / n
        out[i] = a
    return out


class HTFStructure:
    """Higher-timeframe swing structure built from completed bars only.
    Bias +1 when the last two confirmed swing highs AND lows are rising, -1 when
    both are falling, else 0. Swings are 2-left/2-right fractals, so a swing is
    only known two HTF bars after it forms (no lookahead)."""

    def __init__(self):
        self.key = None
        self.cur = None            # [high, low]
        self.hs, self.ls = [], []  # completed bar highs/lows (rolling)
        self.sh, self.sl = [], []  # confirmed swing highs / lows
        self.bias = 0

    def update(self, key, high, low):
        if self.key is None:
            self.key, self.cur = key, [high, low]
            return
        if key != self.key:
            self.hs.append(self.cur[0]); self.ls.append(self.cur[1])
            self.hs, self.ls = self.hs[-5:], self.ls[-5:]
            if len(self.hs) == 5:
                if self.hs[2] > max(self.hs[0], self.hs[1], self.hs[3], self.hs[4]):
                    self.sh = (self.sh + [self.hs[2]])[-2:]
                if self.ls[2] < min(self.ls[0], self.ls[1], self.ls[3], self.ls[4]):
                    self.sl = (self.sl + [self.ls[2]])[-2:]
                if len(self.sh) == 2 and len(self.sl) == 2:
                    up = self.sh[1] > self.sh[0] and self.sl[1] > self.sl[0]
                    dn = self.sh[1] < self.sh[0] and self.sl[1] < self.sl[0]
                    self.bias = 1 if up else (-1 if dn else 0)
            self.key, self.cur = key, [high, low]
        else:
            self.cur[0] = max(self.cur[0], high); self.cur[1] = min(self.cur[1], low)


@dataclass
class Trade:
    direction: int
    level: str
    session: str
    sweep_time: str
    signal_time: str
    entry_time: str
    entry: float
    stop: float
    target: float
    stop_pips: float
    exit_time: str = ""
    exit: float = math.nan
    exit_reason: str = ""
    r: float = math.nan


def run(df: pd.DataFrame, p: Params = Params()) -> list[Trade]:
    o, h, l, c = (df[k].to_numpy() for k in "ohlc")
    bo, bh, bl, ao, ah, al = (df[k].to_numpy() for k in ("bid_o", "bid_h", "bid_l", "ask_o", "ask_h", "ask_l"))
    atr = _atr(h, l, c, p.atr_len)
    idx = df.index
    lon = idx.tz_convert(LONDON)
    ny = idx.tz_convert(NEW_YORK)
    lmin = (lon.hour * 60 + lon.minute).to_numpy()
    # FX trading day: 17:00 New York to 17:00 New York (OANDA daily candle).
    tday = (ny + timedelta(hours=7)).date
    utc_h = idx.hour.to_numpy()
    utc_d = idx.normalize()
    slip = p.slippage_pips * PIP * p.cost_mult
    buf = p.stop_buffer_pips * PIP
    n = len(df)

    h1, h4 = HTFStructure(), HTFStructure()
    trades: list[Trade] = []
    day = None
    dH = dL = pdH = pdL = asH = asL = math.nan
    consumed = {}
    setup = None      # dict while hunting displacement / retrace
    pos = None        # open Trade
    pend = None       # entry to fill at next bar open

    def in_kz(m):
        if p.london_kz[0] <= m < p.london_kz[1]:
            return "london"
        if p.ny_kz[0] <= m < p.ny_kz[1]:
            return "new_york"
        return None

    def cost_adj(price, side_is_buy, spread_half):
        # stress: widen half-spread and slippage by cost_mult around mid
        extra = (p.cost_mult - 1.0) * spread_half
        return price + (slip + extra if side_is_buy else -(slip + extra))

    for i in range(n):
        # ---------------- fills at this bar's open (orders from bar i-1) ----
        if pend is not None:
            d = pend["d"]
            half = (ao[i] - bo[i]) / 2.0
            fill = cost_adj(ao[i] if d == 1 else bo[i], d == 1, half)
            dist = (fill - pend["stop"]) * d
            if dist >= p.min_stop_pips * PIP:
                pos = Trade(d, pend["level"], pend["session"], pend["sweep_time"], pend["signal_time"],
                            str(idx[i]), fill, pend["stop"], fill + d * p.rr * dist, dist / PIP)
            pend = None
        if pos is not None:
            d = pos.direction
            # exits on the side you close at: longs sell at bid, shorts buy at ask
            O, H, L = (bo[i], bh[i], bl[i]) if d == 1 else (ao[i], ah[i], al[i])
            half = (ao[i] - bo[i]) / 2.0
            ex = None
            hit_sl = (L <= pos.stop) if d == 1 else (H >= pos.stop)
            hit_tp = (H >= pos.target) if d == 1 else (L <= pos.target)
            gap_sl = (O <= pos.stop) if d == 1 else (O >= pos.stop)
            gap_tp = (O >= pos.target) if d == 1 else (O <= pos.target)
            if gap_sl:
                ex, why = cost_adj(O, d == -1, half), "stop_gap"
            elif gap_tp:
                ex, why = O, "target_gap"
            elif hit_sl:                      # stop first if both hit in one bar: conservative
                ex, why = cost_adj(pos.stop, d == -1, half), "stop"
            elif hit_tp:
                ex, why = pos.target, "target"
            if ex is not None:
                pos.exit, pos.exit_time, pos.exit_reason = ex, str(idx[i]), why
                pos.r = (ex - pos.entry) * d / (abs(pos.entry - pos.stop))
                trades.append(pos); pos = None

        # ---------------- bar-close logic on bar i --------------------------
        if tday[i] != day:
            if day is not None and not math.isnan(dH):
                pdH, pdL = dH, dL
            day, dH, dL, asH, asL = tday[i], h[i], l[i], math.nan, math.nan
            consumed = {"asia_high": False, "asia_low": False, "pdh": False, "pdl": False}
            setup = None
        else:
            dH, dL = max(dH, h[i]), min(dL, l[i])
        m = lmin[i]
        asia_done = m >= p.asia[1] and m < 17 * 60
        if p.asia[0] <= m < p.asia[1]:
            asH = h[i] if math.isnan(asH) else max(asH, h[i])
            asL = l[i] if math.isnan(asL) else min(asL, l[i])

        h1.update(utc_d[i] + pd.Timedelta(hours=int(utc_h[i])), h[i], l[i])
        h4.update(utc_d[i] + pd.Timedelta(hours=int(utc_h[i]) // 4 * 4), h[i], l[i])

        # time exit: flat before rollover (fills at next open)
        if pos is not None and m >= p.flat_minute and m < 22 * 60:
            nxt = i + 1
            if nxt < n:
                d = pos.direction
                half = (ao[nxt] - bo[nxt]) / 2.0
                px = cost_adj(bo[nxt] if d == 1 else ao[nxt], d == -1, half)
                pos.exit, pos.exit_time, pos.exit_reason = px, str(idx[nxt]), "time"
                pos.r = (px - pos.entry) * d / abs(pos.entry - pos.stop)
                trades.append(pos); pos = None
            continue

        if math.isnan(atr[i]):
            continue
        sess = in_kz(m)
        lon_i = lon[i]
        nfp = p.skip_nfp and lon_i.weekday() == 4 and lon_i.day <= 7 and 12 * 60 + 30 <= m < 14 * 60 + 30

        # levels available for sweeping
        levels_hi = [("pdh", pdH)] + ([("asia_high", asH)] if asia_done else [])
        levels_lo = [("pdl", pdL)] + ([("asia_low", asL)] if asia_done else [])
        swept_hi = [name for name, v in levels_hi if not math.isnan(v) and not consumed[name] and h[i] > v]
        swept_lo = [name for name, v in levels_lo if not math.isnan(v) and not consumed[name] and l[i] < v]
        closed_back_hi = [name for name in swept_hi if c[i] < dict(levels_hi)[name]]
        closed_back_lo = [name for name in swept_lo if c[i] > dict(levels_lo)[name]]
        for name in swept_hi + swept_lo:
            consumed[name] = True       # liquidity taken: the level is spent either way

        # ---- manage an existing setup ----
        if setup is not None:
            d = setup["d"]
            beyond = (d == -1 and h[i] > setup["ext"]) or (d == 1 and l[i] < setup["ext"])
            accepted = (d == -1 and c[i] > setup["lvl"]) or (d == 1 and c[i] < setup["lvl"])
            if setup["state"] == "displace" and accepted:
                setup = None            # closed beyond the swept level: a breakout, not a sweep
            elif setup["state"] == "displace" and beyond:
                setup["ext"] = h[i] if d == -1 else l[i]   # the stop run is still going:
                setup["bar"] = i                           # extend it, restart the window
            elif setup["state"] == "retrace" and beyond:
                setup = None            # extreme broken after displacement: the stop would be hit
            elif setup["state"] == "displace":
                if i - setup["bar"] > p.displacement_bars:
                    setup = None
                elif i - 2 >= setup["bar"]:
                    body = abs(c[i - 1] - o[i - 1])
                    disp_ok = body >= p.displacement_atr * atr[i - 1] and (c[i - 1] - o[i - 1]) * d > 0
                    fvg = (l[i - 2] > h[i]) if d == -1 else (h[i - 2] < l[i])
                    if disp_ok and fvg:
                        lo_, hi_ = (h[i], l[i - 2]) if d == -1 else (h[i - 2], l[i])
                        setup.update(state="retrace", fvg_lo=lo_, fvg_hi=hi_, fvg_bar=i,
                                     disp_time=str(idx[i - 1]))
            elif setup["state"] == "retrace":
                mid = (setup["fvg_lo"] + setup["fvg_hi"]) / 2.0
                if p.entry_mode == "limit_mid" and i > setup["fvg_bar"] and sess and not nfp and pos is None and pend is None:
                    # resting limit at the gap midpoint, placed at the close of the FVG bar
                    touched = (al[i] <= mid) if d == 1 else (bh[i] >= mid)
                    if touched:
                        stop = setup["ext"] - buf if d == 1 else setup["ext"] + buf
                        dist = (mid - stop) * d
                        lvl_name, sweep_t = setup["level"], setup["time"]
                        setup = None
                        if dist >= p.min_stop_pips * PIP:
                            t = Trade(d, lvl_name, sess, sweep_t, str(idx[i]), str(idx[i]), mid, stop, mid + d * p.rr * dist, dist / PIP)
                            # same bar: assume the worst - if the stop is also inside this bar, it is hit
                            if (d == 1 and bl[i] <= stop) or (d == -1 and ah[i] >= stop):
                                half = (ao[i] - bo[i]) / 2.0
                                t.exit, t.exit_time, t.exit_reason = cost_adj(stop, d == -1, half), str(idx[i]), "stop_same_bar"
                                t.r = (t.exit - t.entry) * d / dist
                                trades.append(t)
                            else:
                                pos = t
                        continue
                if i - setup["fvg_bar"] > p.setup_expiry_bars:
                    setup = None
                elif (d == -1 and c[i] > setup["fvg_hi"]) or (d == 1 and c[i] < setup["fvg_lo"]):
                    setup = None        # gap fully filled and closed through: imbalance gone
                elif p.entry_mode == "next_open" and sess and not nfp and pos is None and pend is None and \
                        ((d == -1 and h[i] >= mid) or (d == 1 and l[i] <= mid)):
                    stop = setup["ext"] + buf if d == -1 else setup["ext"] - buf
                    pend = dict(d=d, stop=stop, level=setup["level"], session=sess,
                                sweep_time=setup["time"], signal_time=str(idx[i]))
                    setup = None

        # ---- new sweep -> new setup (only in a kill zone, flat, no setup) ----
        if setup is None and pos is None and pend is None and sess and not nfp:
            short_ok = bool(closed_back_hi) and not swept_lo
            long_ok = bool(closed_back_lo) and not swept_hi
            if p.use_htf_bias:
                short_ok = short_ok and h1.bias != 1 and h4.bias != 1
                long_ok = long_ok and h1.bias != -1 and h4.bias != -1
            if short_ok:
                lvl = max(dict(levels_hi)[k] for k in closed_back_hi)
                setup = dict(d=-1, ext=h[i], lvl=lvl, bar=i, state="displace", level="+".join(closed_back_hi), time=str(idx[i]))
            elif long_ok:
                lvl = min(dict(levels_lo)[k] for k in closed_back_lo)
                setup = dict(d=1, ext=l[i], lvl=lvl, bar=i, state="displace", level="+".join(closed_back_lo), time=str(idx[i]))
    return trades


def to_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])
