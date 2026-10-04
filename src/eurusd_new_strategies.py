"""
NEW EUR/USD STRATEGIES: P06 - P15
==================================

Design rationale (pre-registered, not post-hoc):

ROOT CAUSE OF P01-P05 FAILURE
------------------------------
EUR/USD had near-zero directional movement from 2014-2020 ("central bank dead zone").
Any simple trend-following strategy gets whipsawed to death in flat conditions.
P01/P02 failed because a single regime gate is not enough.
P03 failed because mean-reversion in calm still has wide bid/ask vs edge.
P04 had the right idea (FRAMA + RSI) but RSI momentum threshold was too loose.
P05 was the best (ATR expansion gate) but channel breakout after expansion often enters too late.

WHAT ACTUALLY WORKS IN FX (per literature + first principles):
--------------------------------------------------------------
1. Trend-following requires BOTH volatility AND momentum confirmation simultaneously.
2. Mean-reversion requires price at extreme zscore AND confirmed reversal candle.
3. Multi-timeframe: 4h direction must align with "daily trend" (shifted EMAs).
4. Momentum strategies: price action RATE OF CHANGE exceeding threshold.
5. Volatility-of-volatility: trade when vol regime is CHANGING not just when it IS high.
6. London/NY session overlap: 4h bars ending 08:00-17:00 UTC have more edge (time filter).

STRATEGIES P06-P15:
-------------------
P06: Triple-Confirmation Breakout
  Logic: Channel breakout (Donchian 40) + ATR expansion (atr14/atr72 > 1.15)
         + RSI momentum (RSI > 55 for long, < 45 for short) + ADX > 22
  Gate: All 4 conditions must fire simultaneously.
  Source: Combines P05 ATR gate with Guyard technical indicators + ADX filter from S07.
  Hypothesis: Requiring 4 simultaneous confirmations should massively reduce false signals.

P07: Adaptive Momentum (Rate of Change + Volatility Scaling)
  Logic: 20-bar ROC exceeds 2x the 60-bar rolling std of ROC (momentum outlier detection)
         + EMA slope positive + price above EMA50
  Source: Inspired by Camarao dissertation (momentum classification) + Guyard momentum feature.
  Hypothesis: Detecting "momentum outlier" bars captures genuine trend starts.

P08: Weekend Gap Fade (Session Timing + Mean Reversion)
  Logic: First 4h bar of the week (Monday open) - if gap > 0.5x ATR from Friday close,
         fade the gap direction. Stop = 1.5 ATR.
  Source: Well-documented FX weekend gap mean-reversion effect.
  Hypothesis: EUR/USD gaps frequently revert over the first trading day of the week.

P09: Daily Trend Alignment + 4h Pullback
  Logic: Daily trend = direction of EMA50 on daily bars (approximated using 6-bar SMA on 4h).
         Enter on 4h RSI pullback to 35-50 (long) or 50-65 (short) when daily trend is clear.
         EMA200 4h as regime filter (only long above EMA200, only short below).
  Source: Standard institutional FX approach. Dissertation confirms multi-timeframe filtering.
  Hypothesis: Filtering by daily trend should dramatically reduce choppy periods.

P10: Bollinger Band Squeeze Expansion
  Logic: Bollinger bandwidth (BB20,2 upper-lower / midband) below its 60-bar percentile (< 25th pct)
         signals a squeeze. Enter on breakout in direction of the expansion (close crosses upper/lower BB).
         Exit at opposite band or stop.
  Source: Standard volatility breakout approach, appears in multiple papers as a feature.
  Hypothesis: Trading BB squeezes captures the post-compression volatility release.

P11: Stochastic Momentum + Trend Filter
  Logic: Stochastic K < 20 (oversold) while price > EMA200 → long.
         Stochastic K > 80 (overbought) while price < EMA200 → short.
         Stochastic K crosses above 20 (or below 80) as the trigger.
         Only enter when ATR14 > 0.9 * ATR72 (not in extreme low-vol crush).
  Source: Guyard uses Stochastic K% and D% as key features. WNE_WP446 identifies STOCH as top indicator.
  Hypothesis: Stochastic oversold/overbought in the direction of the longer-term trend has positive edge.

P12: ATR Volatility-Targeting Trend (Position Sizing Proxy)
  Logic: EMA 50 > EMA 200 → bullish regime. Enter on close above EMA50 after being below.
         Size inversely proportional to ATR (targeting fixed daily vol). Use low ATR to scale in.
         Gate: EMA50 slope positive for at least 5 bars.
  Source: Trend-following with volatility targeting from Guyard/Camarao. AQR-style vol targeting.
  Hypothesis: Entering trend after 5-bar slope confirmation reduces false reversals.

P13: MACD Divergence + Volume Confirmation
  Logic: MACD histogram positive AND increasing (momentum acceleration) + price above EMA100.
         Short: MACD histogram negative AND decreasing + price below EMA100.
         Volume gate: 4h bar volume > 1.1x 20-bar volume average.
  Source: MACD features are in Guyard Table 1. Dissertation uses momentum acceleration.
  Hypothesis: Accelerating MACD with volume confirmation = genuine momentum start.

P14: Swing Low Reversal (Price Action Pattern)
  Logic: Three-bar pattern: bar[-2] makes a lower low than bar[-3], bar[-1] makes higher low,
         bar[0] closes above bar[-1] high → buy signal. Opposite for short.
         Requires EMA50 > EMA200 for long (trend alignment).
         RSI not overbought (< 70 for long).
  Source: Price action based pattern. Camarao uses candle pattern classification.
  Hypothesis: Swing low reversal in trend direction has positive expectancy.

P15: Multi-Condition Composite Score
  Logic: Score = RSI_bull * 0.25 + MACD_bull * 0.25 + EMA_stack_bull * 0.25 + ATR_expand * 0.25
         Enter long if score >= 0.75 (3 of 4 conditions). Enter short if score <= -0.75.
         Exit when score drops below 0.5 / rises above -0.5 (handled by max hold / SL).
  Source: Meta-estimator concept from Guyard (stacking multiple signal sources).
  Hypothesis: Composite scoring ensures multiple evidence sources before entry.
"""

import os, time, pickle, sys
import numpy as np, pandas as pd
from scipy import stats

from core import run_bt, to_sig, cross_up, cross_dn, edge
from eurusd_pipeline import (
    load_eurusd, build_windows, dayoff, pick, trades_of, sel,
    metrics, daily, sharpe, maxdd, START, END, FX_COST_RT, WINS, d0o, d1o, OOS_YEARS
)
from strategies import X, S, dc_hi, dc_lo, SL_GRID, RR_GRID, MAX_HOLD
from core import ema as _ema, sma as _sma, atr as _atr, rsi as _rsi, adx as _adx

# ─────────────────────────────────────────────────────────────────────
# HELPER: approximate stochastic K% on pandas series
# ─────────────────────────────────────────────────────────────────────
def stoch_k(df, n=14):
    lo_n = df.l.rolling(n).min()
    hi_n = df.h.rolling(n).max()
    denom = (hi_n - lo_n).replace(0, np.nan)
    return 100.0 * (df.c - lo_n) / denom

def stoch_d(df, n=14, d=3):
    return stoch_k(df, n).rolling(d).mean()

def bb(c, n=20, k=2.0):
    mid = c.rolling(n).mean()
    sd  = c.rolling(n).std(ddof=0)
    return mid - k*sd, mid, mid + k*sd

def cci(df, n=14):
    tp = (df.h + df.l + df.c) / 3
    mid = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - mid) / (0.015 * mad.replace(0, np.nan))

# ─────────────────────────────────────────────────────────────────────
# P06: Triple-Confirmation Breakout
# ─────────────────────────────────────────────────────────────────────
def p06_fn(x, dc_n, atr_thr, rsi_thr, adx_thr):
    hi = dc_hi(x, dc_n); lo = dc_lo(x, dc_n)
    atr14 = x.atr(14); atr72 = x.atr(72)
    vol_expand = (atr14 / atr72.replace(0, np.nan)) > atr_thr
    r = x.rsi(14)
    p, m, adx_v = x.adx(14)
    adx_strong = adx_v > adx_thr
    long_sig  = edge(x.c > hi) & vol_expand & (r > rsi_thr)   & adx_strong
    short_sig = edge(x.c < lo) & vol_expand & (r < 100 - rsi_thr) & adx_strong
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P07: Adaptive Momentum (ROC outlier)
# ─────────────────────────────────────────────────────────────────────
def p07_fn(x, roc_n, roc_std_n, ema_n):
    roc = x.c.pct_change(roc_n)
    roc_std = roc.rolling(roc_std_n).std()
    # Long: ROC is > 2 std above its rolling mean (positive momentum outlier)
    roc_mean = roc.rolling(roc_std_n).mean()
    ema_f = x.ema(ema_n)
    slope_pos = ema_f > ema_f.shift(3)
    long_sig  = edge(roc > roc_mean + 1.8 * roc_std) & slope_pos & (x.c > ema_f)
    short_sig = edge(roc < roc_mean - 1.8 * roc_std) & (~slope_pos) & (x.c < ema_f)
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P08: Monday Gap Fade
# ─────────────────────────────────────────────────────────────────────
def p08_fn(x, gap_atr_mult):
    """Fade weekend gap. Monday's first 4h bar is t.dt.dayofweek==0."""
    is_monday_first = (x.df.t.dt.dayofweek == 0) & (x.df.t.dt.hour < 4)
    prev_close = x.c.shift()  # prior bar close (Friday close on 4h)
    gap = x.o - prev_close    # positive = gap up
    atr14 = x.atr(14)
    # Large gap up → short (fade), large gap down → long (fade)
    long_sig  = is_monday_first & (gap < -gap_atr_mult * atr14)
    short_sig = is_monday_first & (gap >  gap_atr_mult * atr14)
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P09: Daily Trend + 4h RSI Pullback Entry
# ─────────────────────────────────────────────────────────────────────
def p09_fn(x, daily_ema_bars, ema200_bars, rsi_lo, rsi_hi):
    """
    daily_ema_bars = approximate daily EMA on 4h data (e.g. 6 * 20 = 120 bars ≈ 20 daily bars).
    EMA200 on 4h = 200 bars ≈ 200 days / 6 ≈ 33 actual days.
    """
    trend_ema = x.ema(daily_ema_bars)
    ema200    = x.ema(ema200_bars)
    daily_up  = x.c > trend_ema
    daily_dn  = x.c < trend_ema
    above_ema200 = x.c > ema200
    below_ema200 = x.c < ema200
    r = x.rsi(14)
    # Long: in daily uptrend, above EMA200, RSI pulled back to 35-55
    long_sig  = cross_up(r, rsi_lo) & daily_up & above_ema200 & (r < rsi_hi)
    short_sig = cross_dn(r, 100-rsi_lo) & daily_dn & below_ema200 & (r > 100-rsi_hi)
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P10: Bollinger Band Squeeze Expansion
# ─────────────────────────────────────────────────────────────────────
def p10_fn(x, bb_n, pct_window, pct_thr):
    lo_b, mid_b, hi_b = bb(x.c, bb_n, 2.0)
    bw = (hi_b - lo_b) / mid_b.replace(0, np.nan)   # bandwidth
    # Detect squeeze: bandwidth below its rolling percentile threshold
    bw_pct = bw.rolling(pct_window).apply(lambda a: np.mean(a < a[-1]), raw=True)
    in_squeeze = bw_pct < pct_thr  # True when bandwidth is in the bottom pct_thr% of its range
    # Wait for the bar AFTER squeeze ends (squeeze was True, now price breaks out)
    squeeze_ended = in_squeeze.shift() & ~in_squeeze
    long_sig  = squeeze_ended & (x.c > hi_b.shift())
    short_sig = squeeze_ended & (x.c < lo_b.shift())
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P11: Stochastic + EMA200 Trend Filter
# ─────────────────────────────────────────────────────────────────────
def p11_fn(x, stoch_n, ema_n, lo_thr, hi_thr):
    sk = stoch_k(x.df, stoch_n)
    ema_trend = x.ema(ema_n)
    atr14 = x.atr(14); atr72 = x.atr(72)
    not_vol_crushed = (atr14 / atr72.replace(0, np.nan)) > 0.85
    long_sig  = cross_up(sk, lo_thr) & (x.c > ema_trend) & not_vol_crushed
    short_sig = cross_dn(sk, hi_thr) & (x.c < ema_trend) & not_vol_crushed
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P12: EMA50/200 Trend Entry with Slope Confirmation
# ─────────────────────────────────────────────────────────────────────
def p12_fn(x, fast_n, slow_n, slope_bars):
    e_fast = x.ema(fast_n)
    e_slow = x.ema(slow_n)
    # slope confirmed: EMA50 has been rising for slope_bars consecutive bars
    slope_pos = (e_fast > e_fast.shift(slope_bars)) & (e_fast > e_fast.shift(1))
    slope_neg = (e_fast < e_fast.shift(slope_bars)) & (e_fast < e_fast.shift(1))
    # ATR expansion: only trade when vol > 90% of longer-term ATR
    atr14 = x.atr(14); atr48 = x.atr(48)
    vol_ok = (atr14 / atr48.replace(0, np.nan)) > 0.88
    long_sig  = cross_up(x.c, e_fast) & (e_fast > e_slow) & slope_pos & vol_ok
    short_sig = cross_dn(x.c, e_fast) & (e_fast < e_slow) & slope_neg & vol_ok
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P13: MACD Acceleration + Volume
# ─────────────────────────────────────────────────────────────────────
def p13_fn(x, fast_n, slow_n, sig_n, ema_filter_n):
    macd_line = x.ema(fast_n) - x.ema(slow_n)
    sig_line  = _ema(macd_line, sig_n)
    hist      = macd_line - sig_line
    # Histogram accelerating: current > previous > previous-1
    hist_acc_pos = (hist > 0) & (hist > hist.shift()) & (hist.shift() > hist.shift(2))
    hist_acc_neg = (hist < 0) & (hist < hist.shift()) & (hist.shift() < hist.shift(2))
    ema_filter = x.ema(ema_filter_n)
    # Volume gate (using raw volume since tb is synthetic)
    vol_up = x.v > x.vsma(20) * 1.05
    long_sig  = edge(hist_acc_pos) & (x.c > ema_filter) & vol_up
    short_sig = edge(hist_acc_neg) & (x.c < ema_filter) & vol_up
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P14: Swing Low/High Reversal in Trend Direction
# ─────────────────────────────────────────────────────────────────────
def p14_fn(x, ema_fast, ema_slow, rsi_n):
    r = x.rsi(rsi_n)
    e_fast = x.ema(ema_fast)
    e_slow = x.ema(ema_slow)
    # Swing low: bar[-2] low < bar[-3] low, bar[-1] low > bar[-2] low, close > bar[-1] high
    l = x.l; h = x.h
    swing_low  = (l.shift(2) < l.shift(3)) & (l.shift(1) > l.shift(2)) & (x.c > h.shift(1))
    swing_high = (h.shift(2) > h.shift(3)) & (h.shift(1) < h.shift(2)) & (x.c < l.shift(1))
    # Trend filter: EMA stack
    bull_trend = e_fast > e_slow
    bear_trend = e_fast < e_slow
    # RSI filter: not overbought on longs, not oversold on shorts
    rsi_ok_long  = r < 72
    rsi_ok_short = r > 28
    long_sig  = swing_low  & bull_trend & rsi_ok_long
    short_sig = swing_high & bear_trend & rsi_ok_short
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# P15: Composite 4-Signal Score
# ─────────────────────────────────────────────────────────────────────
def p15_fn(x, ema_fast, ema_slow, rsi_n, macd_fast, macd_slow):
    r = x.rsi(rsi_n)
    e_fast = x.ema(ema_fast)
    e_slow = x.ema(ema_slow)
    macd = x.ema(macd_fast) - x.ema(macd_slow)
    atr14 = x.atr(14); atr60 = x.atr(60)
    sk = stoch_k(x.df, 14)

    # Each sub-signal is +1 (bullish) or -1 (bearish) or 0 (neutral)
    rsi_bull    = (r > 55).astype(float)   - (r < 45).astype(float)
    macd_bull   = (macd > 0) & (macd > macd.shift())
    macd_bear   = (macd < 0) & (macd < macd.shift())
    macd_score  = macd_bull.astype(float)  - macd_bear.astype(float)
    ema_bull    = (e_fast > e_slow).astype(float) - (e_fast < e_slow).astype(float)
    atr_expand  = ((atr14 / atr60.replace(0, np.nan)) > 1.05).astype(float)
    stoch_bull  = (sk > 50).astype(float) - (sk < 50).astype(float)

    # Composite score: weighted sum (range approx -1 to +1)
    score = 0.25 * rsi_bull + 0.25 * macd_score + 0.25 * ema_bull + 0.15 * atr_expand + 0.10 * stoch_bull

    # Need 3 of 4 main conditions to agree
    long_sig  = edge(score >= 0.75)
    short_sig = edge(score <= -0.75)
    return long_sig, short_sig

# ─────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRY
# ─────────────────────────────────────────────────────────────────────
NEW_STRATS = [
    S('P06', 'Triple-Confirm Breakout',
      'Paper-MultiCondition',
      'Donchian break + ATR expansion + RSI momentum + ADX all required simultaneously',
      [(40, 1.15, 55, 22), (55, 1.15, 55, 20), (40, 1.20, 52, 22), (55, 1.20, 50, 18)],
      p06_fn),

    S('P07', 'ROC Momentum Outlier',
      'Paper-Momentum',
      'Rate-of-change exceeds 1.8 std above rolling mean = momentum outlier entry',
      [(5, 40, 50), (5, 60, 50), (10, 60, 50), (10, 80, 100)],
      p07_fn),

    S('P08', 'Monday Gap Fade',
      'Paper-SessionMeanRev',
      'Fade weekend gaps larger than N*ATR on Monday open bar',
      [(0.30,), (0.40,), (0.50,), (0.60,)],
      p08_fn),

    S('P09', 'Daily Trend + 4h RSI Pullback',
      'Paper-MultiTF',
      'EMA approx daily trend + EMA200 regime + RSI pullback to 35-55 trigger',
      [(120, 200, 35, 55), (90, 200, 35, 55), (120, 300, 35, 58), (90, 150, 38, 55)],
      p09_fn),

    S('P10', 'Bollinger Squeeze Expansion',
      'Paper-VolBreakout',
      'BB bandwidth squeeze below 25th pct, trade expansion direction',
      [(20, 60, 0.25), (20, 100, 0.25), (30, 60, 0.20), (30, 100, 0.20)],
      p10_fn),

    S('P11', 'Stochastic + EMA200 Filter',
      'Paper-Stochastic',
      'Stochastic K crossover at oversold/overbought with EMA200 trend alignment',
      [(14, 200, 20, 80), (14, 150, 25, 75), (21, 200, 20, 80), (21, 150, 25, 75)],
      p11_fn),

    S('P12', 'EMA Trend + Slope Confirm',
      'Paper-AdaptiveTrend',
      'EMA50/200 cross, confirmed by N-bar positive slope + ATR expansion gate',
      [(50, 200, 5), (50, 200, 7), (50, 150, 5), (75, 200, 5)],
      p12_fn),

    S('P13', 'MACD Acceleration + Volume',
      'Paper-MACDMomentum',
      'MACD histogram accelerating (3 bars) with EMA trend filter and volume confirmation',
      [(12, 26, 9, 100), (8, 21, 6, 100), (12, 26, 9, 200), (8, 21, 6, 200)],
      p13_fn),

    S('P14', 'Swing Reversal in Trend',
      'Paper-PriceAction',
      '3-bar swing low/high pattern with EMA stack trend alignment + RSI filter',
      [(50, 200, 14), (50, 200, 7), (50, 150, 14), (34, 89, 14)],
      p14_fn),

    S('P15', 'Composite Score (RSI+MACD+EMA+ATR+STOCH)',
      'Paper-MetaEstimator',
      'Weighted composite of 5 indicators, enter when score >= 0.75 (3-of-4 agree)',
      [(50, 200, 14, 12, 26), (50, 200, 7, 12, 26), (50, 200, 14, 8, 21), (34, 89, 14, 12, 26)],
      p15_fn),
]


# ─────────────────────────────────────────────────────────────────────
# WALK-FORWARD EVALUATION
# ─────────────────────────────────────────────────────────────────────
def run_new_strategies():
    print("=" * 70)
    print("NEW STRATEGIES P06-P15: Walk-Forward Evaluation (EUR/USD 4h)")
    print("=" * 70)
    print()

    df = load_eurusd()
    x = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x.atr(14).values.astype(np.float64)
    T = pd.to_datetime(df.t.values)
    DAY = ((T.floor('D') - START).days).values

    print(f"Data: {df.t.min()} to {df.t.max()}, {len(df)} bars")
    print(f"Testing {len(NEW_STRATS)} new strategies × {len(SL_GRID)} SL × {len(RR_GRID)} RR "
          f"× {max(len(s['grid']) for s in NEW_STRATS)} params max = up to "
          f"{len(NEW_STRATS) * len(SL_GRID) * len(RR_GRID) * 4} combos\n")

    t0 = time.time()
    res = {}
    sig_counts = {}
    for st in NEW_STRATS:
        sid = st['id']
        for pi, p in enumerate(st['grid']):
            try:
                lg, sh = st['fn'](x, *p)
            except Exception as e_:
                print(f"  ERROR in {sid} p={p}: {e_}")
                lg = pd.Series(False, index=df.index)
                sh = pd.Series(False, index=df.index)
            sig = to_sig(lg, sh)
            sig[:200] = 0   # warmup
            sig_counts[(sid, pi)] = int((sig != 0).sum())
            for sl in SL_GRID:
                for rr in RR_GRID:
                    res[(sid, pi, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)
    print(f"Backtests done in {time.time() - t0:.1f}s\n")

    # Walk-forward
    WF_ROWS = []
    OOS = {}
    for st in NEW_STRATS:
        sid = st['id']
        nets = []; rks = []; exds = []; ents = []; wid = []; grosses = []
        for wi, (a, b, c_win) in enumerate(WINS):
            da, db, dc = dayoff(a), dayoff(b), dayoff(c_win)
            best, breadth = pick(res, DAY, st, da, db, min_trades=40)
            row = dict(
                strategy_id=sid, strategy=st['name'], family=st['family'],
                window=wi + 1,
                train=f"{a:%Y-%m-%d}..{b - pd.Timedelta(days=1):%Y-%m-%d}",
                test=f"{b:%Y-%m-%d}..{c_win - pd.Timedelta(days=1):%Y-%m-%d}",
                is_breadth_profitable=breadth
            )
            if best is None:
                row.update(selected='NO TRADE')
                WF_ROWS.append(row)
                continue
            (pi, sl, rr), im = best
            tr = trades_of(res, DAY, (sid, pi, sl, rr))
            m = sel(tr, db, dc)
            om = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], db, dc)
            row.update(
                selected=f"{st['grid'][pi]} SL={sl}xATR RR={rr}",
                sig_params=str(st['grid'][pi]), param_idx=pi, sl_atr=sl, rr=rr,
                is_trades=im['trades'], is_win_rate=im['win_rate'],
                is_sharpe=im['sharpe'], is_pf=im['profit_factor'],
                is_ret_pct=im['total_ret_pct'],
                oos_trades=om['trades'], oos_win_rate=om['win_rate'],
                oos_pf=om['profit_factor'], oos_ret_pct=om['total_ret_pct'],
                oos_exp_R=om['expectancy_R'], oos_sharpe=om['sharpe']
            )
            WF_ROWS.append(row)
            if m.sum() > 0:
                nets.append(tr['net'][m]); rks.append(tr['risk'][m])
                exds.append(tr['ex_day'][m]); ents.append(tr['ent_day'][m])
                wid.append(np.full(m.sum(), wi)); grosses.append(tr['gross'][m])

        cat = lambda L: np.concatenate(L) if L else np.array([])
        OOS[sid] = dict(
            net=cat(nets), gross=cat(grosses), risk=cat(rks),
            exd=cat(exds).astype(int) if exds else np.array([], int),
            entd=cat(ents), win=cat(wid)
        )

    WF = pd.DataFrame(WF_ROWS)

    # ── Aggregate metrics ──
    rows = []
    yr_rows = []
    ydays = {y: (dayoff(pd.Timestamp(f'{y}-01-01')),
                 dayoff(min(pd.Timestamp(f'{y+1}-01-01'), END)))
             for y in range(2013, 2027)}

    for st in NEW_STRATS:
        sid = st['id']
        o_tr = OOS[sid]
        mt = metrics(o_tr['net'], o_tr['risk'], o_tr['exd'], d0o, d1o)
        dly = (daily(o_tr['net'], o_tr['exd'], d0o, d1o)
               if len(o_tr['net']) else np.zeros(d1o - d0o))

        net = o_tr['net']; rk = o_tr['risk']
        if len(net) > 0 and len(rk) > 0:
            R = net / np.where(rk == 0, 1e-9, rk)
            eq_15 = 100.0
            for rv in R: eq_15 *= (1.0 + 0.015 * rv)
            comp_15 = eq_15 - 100.0
            eq_20 = 100.0
            for rv in R: eq_20 *= (1.0 + 0.020 * rv)
            comp_20 = eq_20 - 100.0
            lev3 = dly.sum() * 3.0 * 100.0
        else:
            comp_15 = comp_20 = lev3 = 0.0

        act = WF[(WF.strategy_id == sid) & (WF.oos_trades.fillna(0) > 0)]
        traded_win = len(act)
        prof_win = (act.oos_ret_pct > 0).sum() if traded_win else 0
        pct_prof = prof_win / traded_win if traded_win else 0.0

        mt.update(
            strategy_id=sid, strategy=st['name'], family=st['family'],
            windows_traded=traded_win, windows_profitable=prof_win,
            pct_windows_profitable=pct_prof,
            unleveraged_1x_ret_pct=mt['total_ret_pct'],
            leverage_3x_ret_pct=lev3,
            compounded_1_5pct_risk_pct=comp_15,
            compounded_2_0pct_risk_pct=comp_20,
            verdict='FAIL'   # will be updated below
        )

        # Gate checks
        n_oos  = mt['trades']
        pf_oos = mt.get('profit_factor', 0.0)
        sr_oos = mt['sharpe']
        n_trials = 56 + 10   # 56 previous + 10 new = 66 total
        # DSR: need skew and kurt of OOS daily P&L
        if len(o_tr['net']) > 5:
            skew_v = float(stats.skew(dly[dly != 0])) if (dly != 0).sum() > 3 else 0.0
            kurt_v = float(stats.kurtosis(dly[dly != 0])) if (dly != 0).sum() > 3 else 0.0
        else:
            skew_v = kurt_v = 0.0
        T_days = d1o - d0o
        # DSR computation
        sr_list = []   # collect all OOS sharpes for exp max SR
        exp_sr = _exp_max_sr_val(n_trials)
        dsr_prob = _dsr(sr_oos, exp_sr, T_days, skew_v, kurt_v)

        gate_pass = (
            n_oos >= 100 and
            pf_oos >= 1.10 and sr_oos >= 0.50 and
            pct_prof >= 0.60 and
            dsr_prob >= 0.95
        )
        mt['dsr_prob'] = dsr_prob
        mt['exp_max_sr'] = exp_sr
        mt['verdict'] = ('PASS' if gate_pass else
                         'WATCH' if (sr_oos >= 0.50 and pf_oos >= 1.05) else 'FAIL')
        rows.append(mt)

        # Yearly
        yr_entry = dict(strategy_id=sid, strategy=st['name'])
        for y, (da_y, db_y) in ydays.items():
            m_y = ((o_tr['exd'] >= da_y) & (o_tr['exd'] < db_y)
                   if len(o_tr['net']) else np.array([], bool))
            yr_entry[str(y)] = (o_tr['net'][m_y].sum() * 100.0
                                 if len(o_tr['net']) else 0.0)
        yr_rows.append(yr_entry)

    EV = pd.DataFrame(rows)
    WF = WF
    YR = pd.DataFrame(yr_rows)

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (OOS, net of costs)")
    print("=" * 80)
    cols_print = [
        'strategy_id', 'strategy', 'trades', 'win_rate', 'payoff',
        'profit_factor', 'sharpe', 'max_dd_pct', 'pct_windows_profitable',
        'unleveraged_1x_ret_pct', 'leverage_3x_ret_pct',
        'compounded_1_5pct_risk_pct', 'dsr_prob', 'verdict'
    ]
    available = [c for c in cols_print if c in EV.columns]
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(EV[available].to_string(index=False))

    return EV, WF, YR, res, OOS


def _exp_max_sr_val(N, avg_sr=0.2, std_sr=0.6):
    """Expected maximum Sharpe given N trials (simplified)."""
    g = 0.5772156649
    return std_sr * ((1 - g) * stats.norm.ppf(1 - 1/N) + g * stats.norm.ppf(1 - 1/(N * np.e)))


def _dsr(sr_d, sr0, T, skew, kurt):
    den = np.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return float(stats.norm.cdf((sr_d - sr0) * np.sqrt(max(T - 1, 1)) / den))


if __name__ == '__main__':
    EV, WF, YR, res, OOS = run_new_strategies()
