import numpy as np, pandas as pd
from core import *
from strategies import X, S, dc_hi, dc_lo

def er(c, n):
    change = (c - c.shift(n)).abs()
    volatility = c.diff().abs().rolling(n).sum()
    return change / volatility.replace(0, np.nan)

# H01_FX: Regime-Gated Mean Reversion (Bollinger Fade in Low ADX)
def h01_fx(x, n, k, adx_thr):
    _, _, a = x.adx(14)
    mu = x.sma(n); sd = x.c.rolling(n).std()
    low_adx = a < adx_thr
    return cross_up(x.c, mu - k * sd) & low_adx, cross_dn(x.c, mu + k * sd) & low_adx

# H02_FX: Multi-Timeframe Macro Trend Following (4h Cross + Macro EMA)
def h02_fx(x, f, s, macro_n):
    ef = x.ema(f); es = x.ema(s); em = x.ema(macro_n)
    return cross_up(ef, es) & (x.c > em), cross_dn(ef, es) & (x.c < em)

# H03_FX: London Session Expansion Breakout (08:00 UTC Asian Range Break)
def h03_fx(x, b):
    # Asian range: 00:00 to 08:00 UTC (bars at 00:00 and 04:00)
    in_asian = x.hour < 8
    hi = x.h.where(in_asian).groupby(x.day).transform('max')
    lo = x.l.where(in_asian).groupby(x.day).transform('min')
    in_london = (x.hour >= 8) & (x.hour < 16)
    a = x.atr(14)
    return edge((x.c > hi + b * a) & in_london), edge((x.c < lo - b * a) & in_london)

# H04_FX: Kaufman Efficiency Ratio Breakout
def h04_fx(x, n, thr):
    eff = er(x.c, n)
    hi = dc_hi(x, n); lo = dc_lo(x, n)
    return edge(x.c > hi) & (eff > thr), edge(x.c < lo) & (eff > thr)

# H05_FX: Volatility Compression Squeeze (ATR Ratio)
def h05_fx(x, n_slow, comp_ratio, k):
    a14 = x.atr(14); aslow = x.atr(n_slow)
    sq = (a14 / aslow.replace(0, np.nan)) < comp_ratio
    hi = dc_hi(x, k); lo = dc_lo(x, k)
    return edge(x.c > hi) & sq.shift(), edge(x.c < lo) & sq.shift()

# H06_FX: RSI Trend Pullback
def h06_fx(x, n, low_thr, trend_n):
    r = x.rsi(n); em = x.ema(trend_n)
    bull = x.c > em; bear = x.c < em
    return cross_up(r, low_thr) & bull, cross_dn(r, 100 - low_thr) & bear

EURUSD_HYPO_STRATS = [
    S('H01_FX', 'Regime Mean-Reversion', 'FX-Hypothesis', 'Bollinger reversion gated by ADX low-trend regime',
      [(20, 2.0, 20), (20, 2.5, 20), (20, 2.0, 15), (30, 2.0, 20)], h01_fx),
    S('H02_FX', 'Multi-TF Macro Trend', 'FX-Hypothesis', '4h EMA cross aligned with macro Daily EMA trend',
      [(12, 26, 600), (20, 50, 600), (9, 21, 300), (12, 26, 300)], h02_fx),
    S('H03_FX', 'London Session Breakout', 'FX-Hypothesis', 'Breakout of Asian range during London/NY overlap',
      [(0.0,), (0.2,), (0.5,), (1.0,)], h03_fx),
    S('H04_FX', 'Efficiency Ratio Breakout', 'FX-Hypothesis', 'Donchian breakout only in high Kaufman efficiency regime',
      [(20, 0.35), (20, 0.45), (40, 0.30), (40, 0.40)], h04_fx),
    S('H05_FX', 'Vol Compression Breakout', 'FX-Hypothesis', 'Donchian breakout out of multi-week ATR compression',
      [(72, 0.70, 20), (72, 0.75, 20), (120, 0.65, 30), (120, 0.70, 30)], h05_fx),
    S('H06_FX', 'RSI Trend Pullback', 'FX-Hypothesis', 'RSI oversold/overbought pullback in direction of macro EMA',
      [(7, 30, 200), (7, 25, 200), (14, 35, 200), (14, 30, 100)], h06_fx),
]
