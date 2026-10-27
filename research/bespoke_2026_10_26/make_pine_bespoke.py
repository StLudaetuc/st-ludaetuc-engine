"""Builds the bespoke per-instrument candidate generators from each instrument's
own current script (pair constants, session timezone, calibration notes kept).

Usage: python make_pine_bespoke.py <pine_dir>
"""
import sys, pathlib

PINE = pathlib.Path(sys.argv[1])

# Per-instrument settings. Evidence for each is written into the script's
# own change log (see CHANGELOG below).
CFG = {
    "EURUSD": dict(template="fx", version="6.0.0", short="STLEUCG60", prefix="STL-EURUSD-2M-V6", slug="stl_eurusd_2m_candidate_generator_v6",
                   min_stop=9.5, min_atr=1.35, win_a=(6, 20), win_b=(0, 0), enable_d=True, enable_e=False, rr=1.5),
    "AUDUSD": dict(template="fx", version="6.0.0", short="STLAUCG60", prefix="STL-AUDUSD-2M-V6", slug="stl_audusd_2m_candidate_generator_v6",
                   min_stop=11.0, min_atr=1.2, win_a=(0, 6), win_b=(12, 20), enable_d=True, enable_e=False, rr=1.5),
    "USDCAD": dict(template="fx", version="6.0.0", short="STLUCCG60", prefix="STL-USDCAD-2M-V6", slug="stl_usdcad_2m_candidate_generator_v6",
                   min_stop=15.0, min_atr=1.45, win_a=(6, 20), win_b=(0, 0), enable_d=True, enable_e=False, rr=1.5),
    "USDJPY": dict(template="fx", version="6.0.0", short="STLUJCG60", prefix="STL-USDJPY-2M-V6", slug="stl_usdjpy_2m_candidate_generator_v6",
                   min_stop=15.0, min_atr=2.3, win_a=(0, 6), win_b=(12, 16), enable_d=False, enable_e=True, rr=1.5),
    "XAUUSD": dict(template="metal", version="2.0.0", short=None, prefix="STL-XAUUSD-2M-V2", slug="stl_xauusd_2m_candidate_generator_v2",
                   min_stop=40.0, min_atr=3.0, win_a=(7, 20), win_b=(0, 0), enable_d=True, enable_e=False, rr=1.5),
    "XAGUSD": dict(template="metal", version="2.0.0", short=None, prefix="STL-XAGUSD-2M-V2", slug="stl_xagusd_2m_candidate_generator_v2",
                   min_stop=30.0, min_atr=2.5, win_a=(7, 20), win_b=(0, 0), enable_d=True, enable_e=False, rr=1.5),
}

COMMON_EVIDENCE = """// Shared research basis (2026-10-24 overnight study, GBP/USD 2m, 87,216
// bars): v5.2's trigger had no directional edge (-0.05R gross, -0.27R net
// per trade); entries that chase short-term extension lose; stretched
// pullbacks confirmed by a reclaim close win; stops under ~8x round-trip
// cost bleed. GBP/USD v6.0.0 (pullback reclaim, 12-pip floor, 1.5R) was
// +0.07R to +0.10R net per trade, positive in-sample and out-of-sample.
//
// Bespoke calibration for THIS instrument (2026-10-26) used its own
// "<PAIR> Market Data" 5m sheet, Jun-Sep 2026 (~25,000 bars), closes
// only, compared side by side with GBP/USD from the same 5m source so
// every number below is relative to the pair that was fully backtested.
// The 5m sheets have no high/low columns, so these are close-path
// estimates (no spread cost), NOT a full 2m backtest like GBP/USD's.
"""

CHANGELOG = {
    "EURUSD": """// EUR/USD findings:
//   - Mean-reverting, but less than GBP/USD: stretched-vs-chasing win-rate
//     gap 5.8 points (GBP/USD 9.9). Fading a >1-sd stretch against the
//     2-hour trend: 44.7% at 1.5R (44.4% Jun-mid Aug / 45.3% after),
//     GBP/USD 46.5%. -> same Model D (pullback reclaim), unchanged rules.
//   - Volatility 0.75x GBP/USD (mean 5m close-to-close 1.38 vs 1.83 pips).
//     -> minimum ATR 1.35 pips (GBP 1.8 x 0.75).
//   - Stop floor 9.5 pips: GBP's 12 x 0.75 = 9 pips, raised to ~8.5x the
//     ~1.1-pip EUR/USD round-trip cost so costs stay near 0.12R.
//   - Hours: Asia (00-06 UTC) was the weakest block for EUR/USD fades
//     (36-41%); 06-20 UTC kept. 20-24 UTC excluded for rollover spreads.
""",
    "AUDUSD": """// AUD/USD findings:
//   - Mean-reverting: stretched-vs-chasing gap 8.1 points (GBP/USD 9.9).
//     Fade of a >1-sd stretch: 44.9% at 1.5R (45.5% / 43.9%). -> Model D.
//   - Volatility 0.67x GBP/USD (1.23 vs 1.83 pips) -> minimum ATR 1.2 pips.
//   - Stop floor 11 pips: volatility alone says 8, but AUD/USD's ~1.3-pip
//     round-trip cost needs ~11 to keep costs near 0.12R.
//   - Hours: AUD/USD is the one pair whose Asian session works (00-06 UTC
//     44-45%), and its London morning is the weak block (06-12 UTC 38-42%).
//     -> two windows: 00:00-05:59 and 12:00-19:59 UTC.
""",
    "USDCAD": """// USD/CAD findings:
//   - Mean-reverting, the weakest of the four (gap 5.1 points), fade of a
//     >1-sd stretch 45.0% at 1.5R (44.9% / 45.1%). -> Model D.
//   - Volatility 0.80x GBP/USD (1.46 vs 1.83 pips) -> minimum ATR 1.45.
//   - Stop floor 15 pips: USD/CAD has the widest spread of the majors
//     (~1.8 pips round trip), so the floor is cost-driven, not volatility-
//     driven (volatility alone says 9.6).
//   - Hours: 16-20 UTC was USD/CAD's strongest block (49-52%), Asia weak
//     (40-44%). -> 06:00-19:59 UTC.
""",
    "USDJPY": """// USD/JPY findings - this pair is DIFFERENT:
//   - NOT mean-reverting at short horizons: stretched entries won LESS than
//     chasing ones (40.0% vs 42.1%), the only pair where that held. A
//     trend-aligned stretched pullback (Model D's premise) won 27.7%.
//     -> Model D is OFF by default for USD/JPY.
//   - What worked: trend-aligned BREAKOUT continuation - close above the
//     prior 1-hour closing high, with the 2-hour slope and the ~4-hour EMA
//     agreeing and RSI 55-70: 42.2% at 1.5R overall (41.8% / 42.8%), and
//     by session 46-47% in Tokyo (00-06 UTC) and 44-45% around the New York
//     open (12-16 UTC), but 37-40% in the London morning and 33-36% in the
//     New York afternoon. -> new Model E (trend_breakout_continuation), on
//     by default, armed only 00:00-05:59 and 12:00-15:59 UTC.
//   - The edge is thin: ~45% at 1.5R against a ~44% break-even once a
//     ~1.4-pip cost is charged on a 15-pip stop. Treat USD/JPY as the most
//     speculative of the six until its live trades say otherwise.
//   - Volatility 1.28x GBP/USD (2.34 vs 1.83 pips) -> minimum ATR 2.3 pips,
//     stop floor 15 pips (12 x 1.28).
""",
    "XAUUSD": """// XAU/USD (Gold) - NO PRICE DATA was available for this instrument (the
// 1m/2m/3m/5m/10m XAU/USD sheets are empty, the master log has no gold
// rows, and no outside data source is reachable), so nothing here is
// backtested. What is bespoke:
//   - Built on gold's own v1.2.0 script: its New York session windows,
//     pip size (0.1), and the existing bb_rsi_armed_reversal model are
//     kept. That model stays available but is OFF by default - it has no
//     live record and engine v4 scores it as research-only.
//   - Model D (pullback reclaim) is the default, because it is the only
//     setup with a positive backtest on any instrument; on 4 of 5 FX pairs
//     the stretched-fade effect it relies on was present. Model E
//     (breakout continuation) is included OFF, to switch on if gold's live
//     data shows it trends like USD/JPY rather than reverting.
//   - Stop floor 40 pips = $4.00: ~8.5x a ~$0.45 round trip ($0.30-0.40
//     spread + slippage), the same cost ratio GBP/USD's 12 pips gives.
//   - Arms 07:00-19:59 UTC (London open to NY afternoon), outside the
//     thin Asian hours and the 21:00 UTC rollover.
// Run it in the Strategy Tester before enabling alerts.
""",
    "XAGUSD": """// XAG/USD (Silver) - NO PRICE DATA was available for this instrument (no
// silver market-data sheet, no silver rows in the master log, no outside
// data source reachable), so nothing here is backtested. What is bespoke:
//   - Built on silver's own v1.2.0 script: New York session windows, pip
//     size (0.01), and the existing bb_rsi_armed_reversal model (kept,
//     OFF by default - no live record, research-only in engine v4).
//   - Model D (pullback reclaim) default, Model E available OFF, for the
//     same reasons as gold.
//   - Stop floor 30 pips = $0.30: ~8.5x a ~$0.035 round trip. Silver's
//     spread is large relative to its moves, so this floor is wider in
//     ATR terms than gold's.
//   - Arms 07:00-19:59 UTC.
// Run it in the Strategy Tester before enabling alerts.
""",
}


def build(sym):
    c = CFG[sym]
    src = (PINE / f"STL_{sym}_2m_CandidateGenerator_v1.pine").read_text()

    def rep(old, new, count=1):
        nonlocal src
        n = src.count(old)
        assert n == count, (sym, n, old[:90])
        src = src.replace(old, new)

    metal = c["template"] == "metal"
    name = {"XAUUSD": "XAU/USD (Gold)", "XAGUSD": "XAG/USD (Silver)"}.get(sym, f"{sym[:3]}/{sym[3:]}")

    # --- header / identity ---------------------------------------------------
    if metal:
        old_title = src.split('title                   = "', 1)[1].split('"', 1)[0]
        rep(f'title                   = "{old_title}"', f'title                   = "{old_title.rsplit(" v", 1)[0]} v2.0"')
        old_short = src.split('shorttitle              = "', 1)[1].split('"', 1)[0]
        rep(f'shorttitle              = "{old_short}"', f'shorttitle              = "{old_short[:-2]}20"')
        head_marker = f"// ST LUDAETUC {name} 2-MINUTE CANDIDATE GENERATOR v1.0.0\n"
    else:
        rep(f'title                   = "St Ludaetuc {sym} 2m Candidate Generator v5.0"', f'title                   = "St Ludaetuc {sym} 2m Candidate Generator v6.0"')
        old_short = src.split('shorttitle              = "', 1)[1].split('"', 1)[0]
        rep(f'shorttitle              = "{old_short}"', f'shorttitle              = "{c["short"]}"')
        head_marker = f"// ST LUDAETUC {name} 2-MINUTE CANDIDATE GENERATOR v5.2.0\n"

    hours = f'{c["win_a"][0]:02d}:00-{c["win_a"][1]-1:02d}:59 UTC' + (f' and {c["win_b"][0]:02d}:00-{c["win_b"][1]-1:02d}:59 UTC' if c["win_b"][1] > c["win_b"][0] else "")
    log = f"""// ST LUDAETUC {name} 2-MINUTE CANDIDATE GENERATOR v{c['version']}
//
// ----------------------------------------------------------------------------
// CHANGE LOG v{c['version']} (bespoke {name} build, 2026-10-26)
// ----------------------------------------------------------------------------
{COMMON_EVIDENCE.replace('<PAIR>', name.split(' ')[0])}//
{CHANGELOG[sym]}//
// Settings this build ships with (all are inputs, group "Model D/E"):
//   stop floor {c['min_stop']} pips, target {c['rr']}R, minimum ATR {c['min_atr']} pips,
//   ATR/50-bar-average >= 0.90, arms {hours},
//   Model D {'ON' if c['enable_d'] else 'OFF'}, Model E {'ON' if c['enable_e'] else 'OFF'}.
// The alert payload schema is unchanged; entry_model arrives as
// "trend_pullback_reclaim" (Model D) or "trend_breakout_continuation"
// (Model E). Needs Manus engine v4.1.0+, which knows both models and this
// instrument's matching stop floor.
//
// Earlier notes retained below for history.
//
// ----------------------------------------------------------------------------
"""
    rep(head_marker, log + "// (previous header)\n" + head_marker)

    ver_old = "1.2.0" if metal else "5.2.0"
    rep(f'string strategyVersionString = "{ver_old}"', f'string strategyVersionString = "{c["version"]}"')
    old_slug = src.split('string strategyIdSlug        = "', 1)[1].split('"', 1)[0]
    rep(f'string strategyIdSlug        = "{old_slug}"', f'string strategyIdSlug        = "{c["slug"]}"')
    old_prefix = src.split('string signalIdPrefix        = "', 1)[1].split('"', 1)[0]
    rep(f'string signalIdPrefix        = "{old_prefix}"', f'string signalIdPrefix        = "{c["prefix"]}"')
    rep('string engineExpectedVersion = "3.11.0"', 'string engineExpectedVersion = "4.1.0"')
    old_rev = src.split('string strategyRevisionDate  = "', 1)[1].split('"', 1)[0]
    rep(f'string strategyRevisionDate  = "{old_rev}"', 'string strategyRevisionDate  = "2026-10-26"')

    # --- risk inputs ---------------------------------------------------------
    old_rr = src.split("float targetRiskMultiplier = input.float(\n     ", 1)[1].split(",", 1)[0]
    rep(f"float targetRiskMultiplier = input.float(\n     {old_rr},", f"float targetRiskMultiplier = input.float(\n     {c['rr']:.2f},")
    old_hold = src.split("int simulatedMaximumHoldingBars = input.int(\n     ", 1)[1].split(",", 1)[0]
    rep(f"int simulatedMaximumHoldingBars = input.int(\n     {old_hold},", "int simulatedMaximumHoldingBars = input.int(\n     720,")

    # --- legacy models default off -------------------------------------------
    legacy = ["enableBbRsiArmedReversal"] if metal else ["enableTrendPullback", "enableMomentumContinuation", "enableRangeReversal"]
    for v in legacy:
        if f"bool {v} = input.bool(\n     false," in src:
            continue  # already off in this instrument's script
        rep(f"bool {v} = input.bool(\n     true,", f"bool {v} = input.bool(\n     false,")

    # --- new inputs ------------------------------------------------------------
    inputs = f'''

// ----------------------------------------------------------------------------
// v{c['version']} - Model D (trend pullback reclaim) and Model E (trend breakout
// continuation), bespoke {name} defaults - see change log
// ----------------------------------------------------------------------------

string groupV6 = "Model D/E - shared filters and risk"

float v6MinStopPips = input.float(
     {c['min_stop']},
     "Minimum stop distance, pips",
     minval = 0.0,
     step = 0.5,
     tooltip = "Sized to ~8.5x this instrument's round-trip cost (spread + slippage). Keep in step with Manus engine INSTRUMENTS min_stop_pips.",
     group = groupV6
)

int v6SwingLookback = input.int(
     6,
     "Swing lookback for stop placement, bars",
     minval = 2,
     maxval = 50,
     group = groupV6
)

float v6StopAtrPad = input.float(
     0.30,
     "Stop padding beyond swing, ATR",
     minval = 0.0,
     step = 0.05,
     group = groupV6
)

float v6MinAtrPips = input.float(
     {c['min_atr']},
     "Minimum ATR, pips",
     minval = 0.0,
     step = 0.05,
     group = groupV6
)

float v6MinAtrRatio = input.float(
     0.90,
     "Minimum ATR / 50-bar ATR average",
     minval = 0.0,
     step = 0.05,
     tooltip = "Skips dead, contracting markets.",
     group = groupV6
)

int v6WindowAStartUtc = input.int(
     {c['win_a'][0]},
     "Trading window A: from hour (UTC, inclusive)",
     minval = 0,
     maxval = 23,
     group = groupV6
)

int v6WindowAEndUtc = input.int(
     {c['win_a'][1]},
     "Trading window A: until hour (UTC, exclusive)",
     minval = 1,
     maxval = 24,
     group = groupV6
)

int v6WindowBStartUtc = input.int(
     {c['win_b'][0]},
     "Trading window B: from hour (UTC, inclusive)",
     minval = 0,
     maxval = 23,
     tooltip = "Set start = end to disable window B.",
     group = groupV6
)

int v6WindowBEndUtc = input.int(
     {c['win_b'][1]},
     "Trading window B: until hour (UTC, exclusive)",
     minval = 0,
     maxval = 24,
     group = groupV6
)

string groupV6D = "Model D - trend pullback reclaim"

bool enablePullbackReclaim = input.bool(
     {'true' if c['enable_d'] else 'false'},
     "Enable trend-pullback-reclaim candidates",
     group = groupV6D
)

float v6StretchZ = input.float(
     1.0,
     "Stretch: Bollinger z beyond",
     minval = 0.2,
     step = 0.1,
     tooltip = "Price counts as stretched against the trend once it closes this many Bollinger half-widths beyond the basis (1.0 = on or outside the band).",
     group = groupV6D
)

float v6RsiStretch = input.float(
     35.0,
     "Stretch: RSI at or below (longs) / at or above 100 minus this (shorts)",
     minval = 5.0,
     maxval = 50.0,
     step = 1.0,
     group = groupV6D
)

int v6StretchLookback = input.int(
     6,
     "Stretch must have happened within N bars",
     minval = 1,
     maxval = 30,
     group = groupV6D
)

float v6WickRatio = input.float(
     0.40,
     "Rejection wick, minimum fraction of bar range",
     minval = 0.0,
     maxval = 1.0,
     step = 0.05,
     group = groupV6D
)

float v6CheapZ = input.float(
     0.30,
     "Arm only while Bollinger z is still below (longs) / above minus (shorts)",
     step = 0.05,
     group = groupV6D
)

int v6ConfirmBars = input.int(
     2,
     "Confirmation window, bars",
     minval = 1,
     maxval = 10,
     tooltip = "A later bar must CLOSE beyond the armed bar's extreme within this many bars, or the setup is cancelled.",
     group = groupV6D
)

string groupV6E = "Model E - trend breakout continuation"

bool enableBreakoutContinuation = input.bool(
     {'true' if c['enable_e'] else 'false'},
     "Enable trend-breakout-continuation candidates",
     group = groupV6E
)

int eBreakoutLookback = input.int(
     30,
     "Breakout: close above the highest close of the prior N bars",
     minval = 2,
     maxval = 200,
     tooltip = "30 two-minute bars = 1 hour.",
     group = groupV6E
)

int eSlopeBars = input.int(
     60,
     "Trend: close above (below) the close N bars ago",
     minval = 2,
     maxval = 500,
     tooltip = "60 two-minute bars = 2 hours.",
     group = groupV6E
)

int eTrendEmaLength = input.int(
     125,
     "Trend: close above (below) this EMA",
     minval = 5,
     maxval = 1000,
     tooltip = "125 two-minute bars ~ a 50-period EMA on the 5-minute chart.",
     group = groupV6E
)

float eRsiLow = input.float(
     55.0,
     "RSI band, lower (longs; shorts use 100 minus the upper)",
     minval = 0.0,
     maxval = 100.0,
     step = 1.0,
     group = groupV6E
)

float eRsiHigh = input.float(
     70.0,
     "RSI band, upper (longs; shorts use 100 minus the lower)",
     minval = 0.0,
     maxval = 100.0,
     step = 1.0,
     group = groupV6E
)
'''
    if metal:
        anchor = "string groupArmedReversal = \"BB/RSI armed exhaustion reversal model\""
        rep(anchor, inputs.lstrip("\n") + "\n\n// ----------------------------------------------------------------------------\n\n" + anchor)
    else:
        anchor = '''     tooltip = "Research-only candidate type. Manus should initially score it conservatively.",
     group = groupModels
)
'''
        rep(anchor, anchor + inputs)

    # --- model D + E logic ----------------------------------------------------
    logic = '''// ============================================================================
// 9b. MODEL D (TREND PULLBACK RECLAIM) AND MODEL E (TREND BREAKOUT
//     CONTINUATION) - bespoke v6 family
// ============================================================================

float bbHalfWidth =
     bbUpper - bbBasis

float bbZ = na
if bbHalfWidth > 0.0
    bbZ :=
         (close - bbBasis) / bbHalfWidth

int barHourUtc =
     hour(time, "UTC")

inUtcWindow(int startHour, int endHour) =>
    bool result = false
    if endHour > startHour
        result := barHourUtc >= startHour and barHourUtc < endHour
    else if endHour < startHour
        result := barHourUtc >= startHour or barHourUtc < endHour
    result

bool v6CoreHours =
     inUtcWindow(v6WindowAStartUtc, v6WindowAEndUtc) or
     inUtcWindow(v6WindowBStartUtc, v6WindowBEndUtc)

bool v6VolatilityOk =
     atrPips >= v6MinAtrPips and
     not na(atrRatio) and
     atrRatio >= v6MinAtrRatio

// --- Model D: arm on a stretched, rejected pullback against the HTF trend,
// confirm when a later bar closes beyond the armed bar's extreme.

bool longStretchBar =
     (not na(bbZ) and bbZ <= -v6StretchZ) or
     rsi <= v6RsiStretch

bool shortStretchBar =
     (not na(bbZ) and bbZ >= v6StretchZ) or
     rsi >= 100.0 - v6RsiStretch

bool longStretchRecent =
     ta.highest(longStretchBar ? 1 : 0, v6StretchLookback) > 0

bool shortStretchRecent =
     ta.highest(shortStretchBar ? 1 : 0, v6StretchLookback) > 0

bool v6ArmCommon =
     enablePullbackReclaim and
     v6CoreHours and
     v6VolatilityOk and
     not na(bbZ)

bool longArmTrigger =
     v6ArmCommon and
     htfBullish and
     longStretchRecent and
     lowerWickRatio >= v6WickRatio and
     close > open and
     bbZ < v6CheapZ

bool shortArmTrigger =
     v6ArmCommon and
     htfBearish and
     shortStretchRecent and
     upperWickRatio >= v6WickRatio and
     close < open and
     bbZ > -v6CheapZ

var float longArmHigh = na
var float longArmLow = na
var int longArmBar = na
var float longArmBbZ = na
var float longArmWick = na

var float shortArmHigh = na
var float shortArmLow = na
var int shortArmBar = na
var float shortArmBbZ = na
var float shortArmWick = na

bool longArmLive =
     not na(longArmBar) and
     bar_index > longArmBar and
     bar_index - longArmBar <= v6ConfirmBars

bool shortArmLive =
     not na(shortArmBar) and
     bar_index > shortArmBar and
     bar_index - shortArmBar <= v6ConfirmBars

bool longArmInvalidated =
     longArmLive and
     low < longArmLow

bool shortArmInvalidated =
     shortArmLive and
     high > shortArmHigh

bool longPullbackReclaim =
     longArmLive and
     not longArmInvalidated and
     close > longArmHigh

bool shortPullbackReclaim =
     shortArmLive and
     not shortArmInvalidated and
     close < shortArmLow

int v6BarsSinceArm = na
float v6ArmBbZ = na
float v6ArmWick = na
if longPullbackReclaim
    v6BarsSinceArm := bar_index - longArmBar
    v6ArmBbZ := longArmBbZ
    v6ArmWick := longArmWick
else if shortPullbackReclaim
    v6BarsSinceArm := bar_index - shortArmBar
    v6ArmBbZ := shortArmBbZ
    v6ArmWick := shortArmWick

if longArmInvalidated or longPullbackReclaim or (not na(longArmBar) and bar_index - longArmBar > v6ConfirmBars)
    longArmBar := na
    longArmHigh := na
    longArmLow := na

if shortArmInvalidated or shortPullbackReclaim or (not na(shortArmBar) and bar_index - shortArmBar > v6ConfirmBars)
    shortArmBar := na
    shortArmHigh := na
    shortArmLow := na

if barstate.isconfirmed and longArmTrigger and not shortArmTrigger
    longArmBar := bar_index
    longArmHigh := high
    longArmLow := low
    longArmBbZ := bbZ
    longArmWick := lowerWickRatio

if barstate.isconfirmed and shortArmTrigger and not longArmTrigger
    shortArmBar := bar_index
    shortArmHigh := high
    shortArmLow := low
    shortArmBbZ := bbZ
    shortArmWick := upperWickRatio

// --- Model E: trend-aligned breakout continuation, entered on the breakout
// bar's close. Trend = HTF agreement + 2-hour slope + ~4-hour EMA.

float eTrendEma =
     ta.ema(close, eTrendEmaLength)

float ePriorHighClose =
     ta.highest(close, eBreakoutLookback)[1]

float ePriorLowClose =
     ta.lowest(close, eBreakoutLookback)[1]

bool eCommon =
     enableBreakoutContinuation and
     v6CoreHours and
     v6VolatilityOk

bool longBreakoutContinuation =
     eCommon and
     htfBullish and
     close > close[eSlopeBars] and
     close > eTrendEma and
     close > ePriorHighClose and
     rsi >= eRsiLow and
     rsi <= eRsiHigh

bool shortBreakoutContinuation =
     eCommon and
     htfBearish and
     close < close[eSlopeBars] and
     close < eTrendEma and
     close < ePriorLowClose and
     rsi >= 100.0 - eRsiHigh and
     rsi <= 100.0 - eRsiLow

float v6SwingLow =
     ta.lowest(low, v6SwingLookback)

float v6SwingHigh =
     ta.highest(high, v6SwingLookback)


'''
    if metal:
        rep("bool longBbRsiArmedReversal =\n", logic + "bool longBbRsiArmedReversal =\n")
        rep("bool rawLongCandidate =\n     longBbRsiArmedReversal\n",
            "bool rawLongCandidate =\n     longPullbackReclaim or\n     longBreakoutContinuation or\n     longBbRsiArmedReversal\n")
        rep("bool rawShortCandidate =\n     shortBbRsiArmedReversal\n",
            "bool rawShortCandidate =\n     shortPullbackReclaim or\n     shortBreakoutContinuation or\n     shortBbRsiArmedReversal\n")
        for side in ("long", "short"):
            S = side.capitalize()
            rep(f'''get{S}Model() =>
    string result = "none"

    if {side}BbRsiArmedReversal''', f'''get{S}Model() =>
    string result = "none"

    if {side}PullbackReclaim
        result := "trend_pullback_reclaim"
    else if {side}BreakoutContinuation
        result := "trend_breakout_continuation"
    else if {side}BbRsiArmedReversal''')
        rep('''    if model == "bb_rsi_armed_reversal"
        result := "reversal"''', '''    if model == "bb_rsi_armed_reversal"
        result := "reversal"
    else if model == "trend_pullback_reclaim"
        result := "trend"
    else if model == "trend_breakout_continuation"
        result := "momentum"''')
        old_policy = src.split('"\\"candidate_policy\\":\\"', 1)[1].split('\\"', 1)[0]
    else:
        rep('''// ============================================================================
// 10. MODEL SELECTION
// ============================================================================
''', logic + '''// ============================================================================
// 10. MODEL SELECTION
// ============================================================================
''')
        rep("bool rawLongCandidate =\n     longTrendPullback or", "bool rawLongCandidate =\n     longPullbackReclaim or\n     longBreakoutContinuation or\n     longTrendPullback or")
        rep("bool rawShortCandidate =\n     shortTrendPullback or", "bool rawShortCandidate =\n     shortPullbackReclaim or\n     shortBreakoutContinuation or\n     shortTrendPullback or")
        for side in ("long", "short"):
            rep(f'''    if {side}TrendPullback
        result := "trend_pullback"''', f'''    if {side}PullbackReclaim
        result := "trend_pullback_reclaim"
    else if {side}BreakoutContinuation
        result := "trend_breakout_continuation"
    else if {side}TrendPullback
        result := "trend_pullback"''')
        rep('''    if model == "trend_pullback"
        result := "trend"
    else if model == "momentum_continuation"''', '''    if model == "trend_pullback"
        result := "trend"
    else if model == "trend_pullback_reclaim"
        result := "trend"
    else if model == "trend_breakout_continuation"
        result := "momentum"
    else if model == "momentum_continuation"''')
        old_policy = "standard_candidate_v5"

    # --- governance: v6 models apply their own volatility filter ----------------
    rep('''bool commonCandidatePass =
     barstate.isconfirmed and
     validChart and
     validSession and
     volatilityPass and''', '''// Models D and E apply their own ATR / hour filters, so the legacy
// volatility band only gates the legacy model(s).
bool v6Candidate =
     longPullbackReclaim or
     shortPullbackReclaim or
     longBreakoutContinuation or
     shortBreakoutContinuation

bool commonCandidatePass =
     barstate.isconfirmed and
     validChart and
     validSession and
     (volatilityPass or v6Candidate) and''')

    # --- stops ---------------------------------------------------------------------
    rep('''float proposedLongStop =
     math.min(
         longAtrStop,
         longStructureStop
     )''', '''float proposedLongStop =
     math.min(
         longAtrStop,
         longStructureStop
     )

if longModel == "trend_pullback_reclaim" or longModel == "trend_breakout_continuation"
    proposedLongStop :=
         math.min(
             v6SwingLow - atr * v6StopAtrPad,
             close - v6MinStopPips * pipSize
         )''')
    rep('''float proposedShortStop =
     math.max(
         shortAtrStop,
         shortStructureStop
     )''', '''float proposedShortStop =
     math.max(
         shortAtrStop,
         shortStructureStop
     )

if shortModel == "trend_pullback_reclaim" or shortModel == "trend_breakout_continuation"
    proposedShortStop :=
         math.max(
             v6SwingHigh + atr * v6StopAtrPad,
             close + v6MinStopPips * pipSize
         )''')

    # --- strength ----------------------------------------------------------------
    for side in ("Long", "Short"):
        rep(f'''calculate{side}Strength(string model) =>
    float score = 50.0
''', f'''calculate{side}Strength(string model) =>
    float score = 50.0

    if model == "trend_pullback_reclaim"
        // Depth of the stretch and quality of the rejection.
        if not na(v6ArmBbZ)
            score += math.min(math.abs(v6ArmBbZ), 2.0) * 6.0
        if not na(v6ArmWick)
            score += math.min(v6ArmWick, 0.8) * 15.0
''')

    # --- payload extensions --------------------------------------------------------
    rep(f'''         "\\"candidate_policy\\":\\"{old_policy}\\","''', f'''         "\\"candidate_policy\\":\\"bespoke_{sym.lower()}_v{c['version'].split('.')[0]}\\","

    payload := payload +
         "\\"v6_bars_since_arm\\":" +
         (na(v6BarsSinceArm) ? "null" : str.tostring(v6BarsSinceArm)) +
         ","

    payload := payload +
         "\\"v6_arm_bb_z\\":" +
         nullableFloat(v6ArmBbZ, "#.####") +
         ","

    payload := payload +
         "\\"v6_arm_wick_ratio\\":" +
         nullableFloat(v6ArmWick, "#.####") +
         ","

    payload := payload +
         "\\"v6_bb_z\\":" +
         nullableFloat(bbZ, "#.####") +
         ","

    payload := payload +
         "\\"v6_min_stop_pips\\":" +
         str.tostring(v6MinStopPips, "#.##") +
         ","''')

    major = c["version"].split(".")[0]
    out = PINE / f"STL_{sym}_2m_CandidateGenerator_v{major}.pine"
    out.write_text(src)
    return out


for s in CFG:
    print(build(s))
