import numpy as np, pandas as pd
from core import *
from strategies import X, S, dc_hi, dc_lo

def er(c, n):
    """Kaufman Efficiency Ratio: |C_t - C_{t-n}| / Sum(|C_i - C_{i-1}|)"""
    change = (c - c.shift(n)).abs()
    volatility = c.diff().abs().rolling(n).sum()
    return change / volatility.replace(0, np.nan)

# H01: Multi-Timeframe Trend Following (1h Signal + 4h Trend Filter)
def h01(x, f, s, s4, m4):
    ef = x.ema(f); es = x.ema(s)
    e_s4 = x.ema(s4); e_m4 = x.ema(m4)
    macro_bull = e_s4 > e_m4; macro_bear = e_s4 < e_m4
    return cross_up(ef, es) & macro_bull, cross_dn(ef, es) & macro_bear

# H02: Volatility-Regime Trend Following (Kaufman Efficiency Ratio Filter)
def h02(x, n, thr):
    eff = er(x.c, n)
    hi = dc_hi(x, n); lo = dc_lo(x, n)
    regime = eff > thr
    return edge(x.c > hi) & regime, edge(x.c < lo) & regime

# H03: Cross-Timeframe Momentum Pullback (Macro Trend + RSI Pullback)
def h03(x, n, low_thr):
    r = x.rsi(n)
    macro_ema = x.ema(800)  # ~200 bars on 4h
    bull = x.c > macro_ema; bear = x.c < macro_ema
    return cross_up(r, low_thr) & bull, cross_dn(r, 100 - low_thr) & bear

# H04: Regime-Gated Mean Reversion (Low Volatility / Ranging Market Filter)
def h04(x, n, k, adx_thr):
    _, _, adx_val = x.adx(14)
    mu = x.sma(n); sd = x.c.rolling(n).std()
    ranging = adx_val < adx_thr
    return cross_up(x.c, mu - k * sd) & ranging, cross_dn(x.c, mu + k * sd) & ranging

# H05: Taker-Buy Volume Flow Imbalance with Trend Filter
def h05(x, n, z_thr, m):
    z = zscore(x.tbr(), n)
    e = x.ema(m)
    return cross_up(z, z_thr) & (x.c > e), cross_dn(z, -z_thr) & (x.c < e)

# H06: Volatility Compression Squeeze & Expansion (ATR Ratio Breakout)
def h06(x, n_slow, comp_ratio, k):
    a14 = x.atr(14); a_slow = x.atr(n_slow)
    compressed = (a14 / a_slow.replace(0, np.nan)) < comp_ratio
    hi = dc_hi(x, k); lo = dc_lo(x, k)
    return edge(x.c > hi) & compressed.shift(), edge(x.c < lo) & compressed.shift()

# H07: Volume-Weighted Momentum Breakout (VWAP Deviation + Volume Surge)
def h07(x, thr, mult):
    dev = (x.c - x.vwap()) / x.atr(14).replace(0, np.nan)
    vs = x.v > mult * x.vsma(20)
    e = x.ema(50)
    return cross_up(dev, thr) & vs & (x.c > e), cross_dn(dev, -thr) & vs & (x.c < e)

# H08: Dual-Thrust Range Breakout with Volatility Filter
def h08(x, k1, k2):
    dh = x.h.groupby(x.day).max(); dl = x.l.groupby(x.day).min(); dc = x.c.groupby(x.day).last()
    ph = x.day.map(dh.shift()); pl = x.day.map(dl.shift()); pc = x.day.map(dc.shift())
    rng1 = ph - pc; rng2 = pc - pl
    rg = pd.concat([rng1, rng2], axis=1).max(axis=1)
    d_open = x.day.map(x.o.groupby(x.day).first())
    buy_trig = d_open + k1 * rg; sell_trig = d_open - k2 * rg
    a14 = x.atr(14); a_filt = a14 > sma(a14, 50)
    return edge(x.c > buy_trig) & a_filt, edge(x.c < sell_trig) & a_filt

HYPO_STRATS = [
    S('H01', 'Multi-TF Trend Following', 'New-Hypothesis', '1h EMA cross filtered by 4h macro trend',
      [(12, 26, 200, 800), (20, 50, 200, 800), (9, 21, 80, 400), (12, 26, 80, 400)], h01),
    S('H02', 'Volatility-Regime Trend', 'New-Hypothesis', 'Donchian breakout only in high Kaufman Efficiency regime',
      [(20, 0.40), (20, 0.50), (40, 0.35), (40, 0.45)], h02),
    S('H03', 'Cross-TF Momentum Pullback', 'New-Hypothesis', '4h macro trend + 1h RSI oversold/overbought pullback',
      [(7, 30), (7, 25), (14, 35), (14, 30)], h03),
    S('H04', 'Regime-Gated Mean Reversion', 'New-Hypothesis', 'Bollinger reversion gated by ADX low-trend regime',
      [(20, 2.0, 20), (20, 2.5, 20), (20, 2.0, 15), (30, 2.0, 20)], h04),
    S('H05', 'Taker-Flow Imbalance Trend', 'New-Hypothesis', 'Taker-buy volume Z-score extreme aligned with EMA trend',
      [(24, 1.5, 50), (24, 2.0, 50), (48, 1.5, 100), (48, 2.0, 100)], h05),
    S('H06', 'Vol Compression Breakout', 'New-Hypothesis', 'Donchian breakout following ATR compression',
      [(72, 0.70, 20), (72, 0.75, 20), (120, 0.65, 30), (120, 0.70, 30)], h06),
    S('H07', 'VWAP Dev + Volume Surge', 'New-Hypothesis', 'VWAP ATR deviation with volume surge and trend agreement',
      [(1.0, 1.5), (1.5, 1.5), (1.0, 2.0), (1.5, 2.0)], h07),
    S('H08', 'Dual-Thrust Vol Breakout', 'New-Hypothesis', 'Dual-thrust range breakout with ATR expansion filter',
      [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7), (0.5, 0.7)], h08),
]
