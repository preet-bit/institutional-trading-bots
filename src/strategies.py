import numpy as np, pandas as pd
from core import *

class X:
    """Indicator cache bound to one dataframe."""
    def __init__(s, df):
        s.df = df; s.c_ = {}
        s.o, s.h, s.l, s.c, s.v = df.o, df.h, df.l, df.c, df.v
        s.hour = df.t.dt.hour; s.minute = df.t.dt.minute; s.day = df.t.dt.floor('D')
    def g(s, key, fn):
        if key not in s.c_: s.c_[key] = fn()
        return s.c_[key]
    def ema(s, n): return s.g(('ema', n), lambda: ema(s.c, n))
    def sma(s, n): return s.g(('sma', n), lambda: sma(s.c, n))
    def atr(s, n=14): return s.g(('atr', n), lambda: atr(s.df, n))
    def rsi(s, n=14): return s.g(('rsi', n), lambda: rsi(s.c, n))
    def vsma(s, n): return s.g(('vsma', n), lambda: sma(s.v, n))
    def vwap(s): return s.g('vwap', lambda: daily_vwap(s.df))
    def adx(s, n): return s.g(('adx', n), lambda: adx(s.df, n))
    def tbr(s): return s.g('tbr', lambda: s.df.tb / s.v.replace(0, np.nan))

def S(id_, name, fam, desc, grid, fn): return dict(id=id_, name=name, family=fam, desc=desc, grid=grid, fn=fn)

def dc_hi(x, n): return x.h.rolling(n).max().shift()
def dc_lo(x, n): return x.l.rolling(n).min().shift()

STRATS = []
A = STRATS.append

# ---------------- TREND ----------------
A(S('S01', 'EMA Cross', 'Trend', 'Fast EMA crosses slow EMA', [(9, 21), (12, 26), (20, 50), (50, 200)],
    lambda x, f, s: (cross_up(x.ema(f), x.ema(s)), cross_dn(x.ema(f), x.ema(s)))))
A(S('S02', 'SMA Cross', 'Trend', 'Fast SMA crosses slow SMA', [(10, 30), (20, 50), (50, 100), (50, 200)],
    lambda x, f, s: (cross_up(x.sma(f), x.sma(s)), cross_dn(x.sma(f), x.sma(s)))))
def s03(x, f, m, s):
    up = (x.ema(f) > x.ema(m)) & (x.ema(m) > x.ema(s)); dn = (x.ema(f) < x.ema(m)) & (x.ema(m) < x.ema(s))
    return up & cross_up(x.c, x.ema(f)), dn & cross_dn(x.c, x.ema(f))
A(S('S03', 'EMA Stack Pullback', 'Trend', 'EMA f>m>s stack, price reclaims fast EMA', [(8, 21, 55), (13, 34, 89), (21, 55, 144)], s03))
def s04(x, f, s, g):
    m = x.ema(f) - x.ema(s); sg = ema(m, g)
    return cross_up(m, sg), cross_dn(m, sg)
A(S('S04', 'MACD Signal Cross', 'Trend', 'MACD line crosses signal line', [(12, 26, 9), (8, 17, 9), (24, 52, 18), (6, 13, 5)], s04))
def s05(x, f, s):
    m = x.ema(f) - x.ema(s)
    return cross_up(m, 0.0), cross_dn(m, 0.0)
A(S('S05', 'MACD Zero Cross', 'Trend', 'MACD line crosses zero', [(12, 26), (8, 21), (24, 52), (5, 35)], s05))
def s06(x, n, m):
    d = supertrend(x.df, n, m)
    return (d == 1) & (d.shift() == -1), (d == -1) & (d.shift() == 1)
A(S('S06', 'Supertrend Flip', 'Trend', 'Supertrend direction flip', [(10, 2.0), (10, 3.0), (14, 3.0), (20, 4.0)], s06))
def s07(x, n, thr):
    p, m, a = x.adx(n)
    return cross_up(p, m) & (a > thr), cross_dn(p, m) & (a > thr)
A(S('S07', 'ADX / DI Cross', 'Trend', '+DI/-DI cross while ADX above threshold', [(14, 20), (14, 25), (20, 20), (20, 25)], s07))
def s08(x, n):
    e = x.ema(n); sl = e - e.shift(5)
    return cross_up(x.c, e) & (sl > 0), cross_dn(x.c, e) & (sl < 0)
A(S('S08', 'Price/EMA Cross + Slope', 'Trend', 'Close crosses EMA with EMA slope agreeing', [(50,), (100,), (200,), (400,)], s08))
def s09(x, a, b, cc):
    t = (x.h.rolling(a).max() + x.l.rolling(a).min()) / 2; k = (x.h.rolling(b).max() + x.l.rolling(b).min()) / 2
    sa = ((t + k) / 2).shift(b); sb = ((x.h.rolling(cc).max() + x.l.rolling(cc).min()) / 2).shift(b)
    top = pd.concat([sa, sb], axis=1).max(axis=1); bot = pd.concat([sa, sb], axis=1).min(axis=1)
    return cross_up(t, k) & (x.c > top), cross_dn(t, k) & (x.c < bot)
A(S('S09', 'Ichimoku TK Cross + Cloud', 'Trend', 'Tenkan/Kijun cross, price outside the cloud', [(9, 26, 52), (6, 13, 26), (12, 36, 72), (20, 60, 120)], s09))
def _ha(df):
    o, h, l, c = [df[k].values for k in ('o', 'h', 'l', 'c')]
    hc = (o + h + l + c) / 4; ho = np.empty(len(c)); ho[0] = (o[0] + c[0]) / 2
    for i in range(1, len(c)): ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    return hc, ho
def s10(x, n, k):
    hc, ho = x.g('ha', lambda: _ha(x.df)); bull = pd.Series(hc > ho, index=x.df.index)
    e = x.ema(n)
    up = bull & ~bull.shift().fillna(False)
    dn = ~bull & bull.shift().fillna(True)
    if k == 2:
        up = up.shift().fillna(False) & bull; dn = dn.shift().fillna(False) & ~bull
    return up & (x.c > e), dn & (x.c < e)
A(S('S10', 'Heikin-Ashi Flip + EMA', 'Trend', 'Heikin-Ashi colour flip (k bars confirm) with EMA filter', [(50, 1), (100, 1), (100, 2), (200, 2)], s10))
def s11(x, n):
    ua = x.h.rolling(n + 1).apply(lambda a: n - (n - np.argmax(a)), raw=True) * 100 / n
    da = x.l.rolling(n + 1).apply(lambda a: n - (n - np.argmin(a)), raw=True) * 100 / n
    return cross_up(ua, da), cross_dn(ua, da)
A(S('S11', 'Aroon Cross', 'Trend', 'Aroon-up crosses Aroon-down', [(14,), (25,), (50,), (100,)], s11))
def s12(x, n):
    sl = x.g(('lrs', n), lambda: linreg_slope(x.c, n))
    return cross_up(sl, 0.0), cross_dn(sl, 0.0)
A(S('S12', 'Linear-Regression Slope Turn', 'Trend', 'Rolling regression slope changes sign', [(20,), (50,), (100,), (200,)], s12))
def s13(x, n):
    h = x.g(('hma', n), lambda: hma(x.c, n))
    up = (h > h.shift()) & (h.shift() <= h.shift(2)); dn = (h < h.shift()) & (h.shift() >= h.shift(2))
    return up, dn
A(S('S13', 'Hull MA Turn', 'Trend', 'Hull MA changes direction', [(21,), (55,), (89,), (144,)], s13))

# ---------------- BREAKOUT ----------------
A(S('S14', 'Donchian Breakout', 'Breakout', 'Close breaks prior N-bar high/low', [(20,), (55,), (96,), (192,)],
    lambda x, n: (edge(x.c > dc_hi(x, n)), edge(x.c < dc_lo(x, n)))))
A(S('S15', 'Donchian + Volume', 'Breakout', 'Donchian break confirmed by volume spike', [(20, 1.5), (55, 1.5), (96, 1.2), (96, 2.0)],
    lambda x, n, m: (edge(x.c > dc_hi(x, n)) & (x.v > m * x.vsma(20)), edge(x.c < dc_lo(x, n)) & (x.v > m * x.vsma(20)))))
def s16(x, n, m):
    e = x.ema(n); a = x.atr(14)
    return edge(x.c > e + m * a), edge(x.c < e - m * a)
A(S('S16', 'Keltner Breakout', 'Breakout', 'Close outside EMA +/- m*ATR channel', [(20, 2.0), (20, 3.0), (50, 2.0), (50, 3.0)], s16))
def s17(x, n, k):
    mu = x.sma(n); sd = x.c.rolling(n).std()
    return edge(x.c > mu + k * sd), edge(x.c < mu - k * sd)
A(S('S17', 'Bollinger Breakout', 'Breakout', 'Close outside Bollinger band', [(20, 2.0), (20, 2.5), (50, 2.0), (50, 3.0)], s17))
def s18(x, k, n):
    mu = x.sma(20); sd = x.c.rolling(20).std(); kc = x.ema(20); a = x.atr(14)
    sq = ((mu + 2 * sd) < (kc + 1.5 * a)) & ((mu - 2 * sd) > (kc - 1.5 * a))
    held = sq.rolling(k).sum().shift() == k
    rel = held & ~sq
    m = x.c - x.sma(n)
    return rel & (m > 0), rel & (m < 0)
A(S('S18', 'Squeeze Breakout', 'Breakout', 'Bollinger-inside-Keltner squeeze releases; trade momentum side', [(6, 20), (12, 20), (6, 50), (12, 50)], s18))
def s19(x, n):
    pos = x.df.groupby(x.day).cumcount()
    hi = x.h.where(pos < n).groupby(x.day).cummax().groupby(x.day).transform('max')
    lo = x.l.where(pos < n).groupby(x.day).cummin().groupby(x.day).transform('min')
    ok = pos >= n
    return edge((x.c > hi) & ok), edge((x.c < lo) & ok)
A(S('S19', 'Opening-Range Breakout (UTC day)', 'Breakout', 'Break of first-N-bar range of the UTC day', [(4,), (8,), (16,), (24,)], s19))
def s20(x, b):
    dh = x.h.groupby(x.day).max(); dl = x.l.groupby(x.day).min()
    ph = x.day.map(dh.shift()); pl = x.day.map(dl.shift()); a = x.atr(14)
    return edge(x.c > ph + b * a), edge(x.c < pl - b * a)
A(S('S20', 'Prior-Day High/Low Break', 'Breakout', 'Close beyond previous UTC day extreme + buffer*ATR', [(0.0,), (0.25,), (0.5,), (1.0,)], s20))
def s21(x, e, u):
    inr = x.hour < e
    hi = x.h.where(inr).groupby(x.day).transform('max'); lo = x.l.where(inr).groupby(x.day).transform('min')
    win = (x.hour >= e) & (x.hour < u)
    return edge((x.c > hi) & win), edge((x.c < lo) & win)
A(S('S21', 'Asian-Range Breakout', 'Breakout', 'Break of 00:00-E:00 UTC range during E:00-U:00', [(8, 16), (8, 20), (6, 14), (12, 20)], s21))
def s22(x, n):
    ins = (x.h.shift() < x.h.shift(2)) & (x.l.shift() > x.l.shift(2)); e = x.ema(n)
    return ins & (x.c > x.h.shift(2)) & (x.c > e), ins & (x.c < x.l.shift(2)) & (x.c < e)
A(S('S22', 'Inside-Bar Breakout + Trend', 'Breakout', 'Break of mother bar after inside bar, with EMA trend', [(20,), (50,), (100,), (200,)], s22))
def s23(x, n):
    rg = x.h - x.l; nr = rg.shift() == rg.rolling(n).min().shift()
    return nr & (x.c > x.h.shift()), nr & (x.c < x.l.shift())
A(S('S23', 'NR-N Breakout', 'Breakout', 'Break of narrowest-range-of-N bar', [(4,), (7,), (10,), (20,)], s23))
def s24(x, m, q):
    rg = x.h - x.l; a = x.atr(14).shift(); pos = (x.c - x.l) / rg.replace(0, np.nan)
    big = rg > m * a
    return big & (pos >= q), big & (pos <= 1 - q)
A(S('S24', 'ATR Thrust Continuation', 'Breakout', 'Wide-range bar closing near its extreme: continue', [(1.5, 0.8), (2.0, 0.8), (2.0, 0.9), (3.0, 0.8)], s24))
def s25(x, n, thr):
    a = x.adx(14)[2]
    return edge(x.c > dc_hi(x, n)) & (a > thr), edge(x.c < dc_lo(x, n)) & (a > thr)
A(S('S25', 'Donchian + ADX Filter', 'Breakout', 'Donchian break only when ADX above threshold', [(48, 20), (48, 25), (96, 20), (96, 25)], s25))

# ---------------- MEAN REVERSION ----------------
def s26(x, n, lo):
    r = x.rsi(n); return cross_up(r, lo), cross_dn(r, 100 - lo)
A(S('S26', 'RSI Extreme Reversal', 'MeanRev', 'RSI crosses back out of oversold/overbought', [(14, 30), (14, 25), (7, 20), (7, 25)], s26))
def s27(x, n, k):
    mu = x.sma(n); sd = x.c.rolling(n).std()
    return cross_up(x.c, mu - k * sd), cross_dn(x.c, mu + k * sd)
A(S('S27', 'Bollinger Reversion', 'MeanRev', 'Close crosses back inside Bollinger band', [(20, 2.0), (20, 2.5), (20, 3.0), (50, 2.5)], s27))
def s28(x, n, z):
    zz = x.g(('z', n), lambda: zscore(x.c, n)); return cross_up(zz, -z), cross_dn(zz, z)
A(S('S28', 'Z-Score Reversion', 'MeanRev', 'Price z-score crosses back from extreme', [(48, 2.0), (96, 2.0), (96, 2.5), (192, 2.5)], s28))
def s29(x, n, thr):
    k = stoch(x.df, n, 3); d = sma(k, 3)
    return cross_up(k, d) & (k < thr), cross_dn(k, d) & (k > 100 - thr)
A(S('S29', 'Stochastic Extreme Cross', 'MeanRev', '%K crosses %D inside extreme zone', [(14, 20), (14, 10), (21, 20), (9, 15)], s29))
def s30(x, n, lv):
    tp = (x.h + x.l + x.c) / 3; mad = tp.rolling(n).apply(lambda a: np.abs(a - a.mean()).mean(), raw=True) if False else (tp - sma(tp, n)).abs().rolling(n).mean()
    cci = (tp - sma(tp, n)) / (0.015 * mad)
    return cross_up(cci, -lv), cross_dn(cci, lv)
A(S('S30', 'CCI Extreme Reversal', 'MeanRev', 'CCI crosses back from +/-level', [(20, 100), (20, 150), (40, 150), (14, 200)], s30))
def s31(x, t, n):
    r = rsi(x.c, 2); e = x.ema(n)
    return edge((r < t) & (x.c > e)), edge((r > 100 - t) & (x.c < e))
A(S('S31', 'RSI(2) Pullback in Trend', 'MeanRev', 'RSI(2) extreme in direction of EMA trend', [(5, 200), (10, 200), (5, 400), (10, 100)], s31))
def s32(x, k):
    dev = (x.c - x.vwap()) / x.atr(14); return cross_up(dev, -k), cross_dn(dev, k)
A(S('S32', 'Daily-VWAP Deviation Reversion', 'MeanRev', 'Price stretched k ATR from daily VWAP reverts', [(1.5,), (2.0,), (2.5,), (3.0,)], s32))
def s33(x, n, m):
    e = x.ema(n); a = x.atr(14); return cross_up(x.c, e - m * a), cross_dn(x.c, e + m * a)
A(S('S33', 'Keltner Fade', 'MeanRev', 'Close back inside Keltner channel', [(20, 2.0), (20, 2.5), (40, 2.5), (40, 3.0)], s33))
def s34(x, n, lo):
    tp = (x.h + x.l + x.c) / 3; mf = tp * x.v; d = tp.diff()
    pos = mf.where(d > 0, 0.0).rolling(n).sum(); neg = mf.where(d < 0, 0.0).rolling(n).sum()
    m = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return cross_up(m, lo), cross_dn(m, 100 - lo)
A(S('S34', 'MFI Extreme Reversal', 'MeanRev', 'Money-flow index crosses back from extreme', [(14, 20), (14, 10), (7, 15), (21, 25)], s34))
def s35(x, k, f):
    dn = (x.c < x.c.shift()).rolling(k).sum() == k; up = (x.c > x.c.shift()).rolling(k).sum() == k
    if f: e = x.ema(f); return edge(dn & (x.c > e)), edge(up & (x.c < e))
    return edge(dn), edge(up)
A(S('S35', 'Consecutive-Bar Exhaustion', 'MeanRev', 'k consecutive closes one way: fade (optionally with trend)', [(3, 0), (5, 0), (4, 200), (6, 200)], s35))
def s36(x, n, k):
    d = (x.c - x.ema(n)) / x.atr(14); return cross_up(d, -k), cross_dn(d, k)
A(S('S36', 'EMA-Distance Reversion', 'MeanRev', 'Price k ATR from EMA reverts', [(20, 3.0), (50, 3.0), (50, 4.0), (100, 5.0)], s36))
def s37(x, n, w):
    body = (x.c - x.o).abs().replace(0, 1e-9); lw = pd.concat([x.o, x.c], axis=1).min(axis=1) - x.l
    uw = x.h - pd.concat([x.o, x.c], axis=1).max(axis=1); rg = (x.h - x.l).replace(0, np.nan)
    lo = x.l <= x.l.rolling(n).min().shift(); hi = x.h >= x.h.rolling(n).max().shift()
    return (lw > w * body) & (lw / rg > 0.6) & lo, (uw > w * body) & (uw / rg > 0.6) & hi
A(S('S37', 'Pin-Bar Rejection at Extreme', 'MeanRev', 'Long-wick rejection candle at N-bar low/high', [(20, 2.0), (48, 2.0), (48, 3.0), (96, 3.0)], s37))

# ---------------- MOMENTUM ----------------
def s38(x, n, k):
    m = (x.c / x.c.shift(n) - 1) / ((x.atr(14) / x.c) * np.sqrt(n)); return cross_up(m, k), cross_dn(m, -k)
A(S('S38', 'Volatility-Scaled Momentum', 'Momentum', 'N-bar return in ATR-sigma units crosses threshold', [(16, 1.5), (32, 1.5), (32, 2.0), (96, 2.0)], s38))
def s39(x, n, m):
    r = x.rsi(n); e = x.ema(m); return cross_up(r, 50) & (x.c > e), cross_dn(r, 50) & (x.c < e)
A(S('S39', 'RSI-50 Cross + Trend', 'Momentum', 'RSI crosses 50 with EMA trend filter', [(14, 100), (14, 200), (21, 200), (7, 50)], s39))
def s40(x, n, lo):
    r = x.rsi(n); mn = r.rolling(n).min(); mx = r.rolling(n).max(); sr = (r - mn) / (mx - mn).replace(0, np.nan)
    k = sma(sr, 3); d = sma(k, 3)
    return cross_up(k, d) & (k < lo), cross_dn(k, d) & (k > 1 - lo)
A(S('S40', 'Stochastic-RSI Cross', 'Momentum', 'StochRSI K/D cross in low/high zone', [(14, 0.2), (14, 0.3), (21, 0.2), (7, 0.2)], s40))
def s41(x, n, k):
    r = x.c / x.c.shift(n) - 1; return cross_up(r, ema(r, k)) & (r > 0), cross_dn(r, ema(r, k)) & (r < 0)
A(S('S41', 'Momentum Acceleration', 'Momentum', 'ROC crosses its own EMA, same sign', [(8, 4), (16, 8), (32, 16), (64, 32)], s41))
def s42(x, n, k):
    m = (x.c - x.c.shift(n)) / x.atr(14); return edge(m > k), edge(m < -k)
A(S('S42', 'Impulse Continuation', 'Momentum', 'N-bar move larger than k ATR: continue', [(4, 2.0), (4, 3.0), (8, 3.0), (16, 4.0)], s42))
def s43(x, k, f):
    hh = ((x.h > x.h.shift()) & (x.c > x.c.shift())).rolling(k).sum() == k
    ll = ((x.l < x.l.shift()) & (x.c < x.c.shift())).rolling(k).sum() == k
    e = x.ema(f); return edge(hh & (x.c > e)), edge(ll & (x.c < e))
A(S('S43', 'Higher-High Streak', 'Momentum', 'k consecutive higher-high/higher-close bars with trend filter', [(3, 50), (4, 100), (5, 100), (3, 200)], s43))

# ---------------- VOLUME / ORDER FLOW ----------------
def s44(x, m, n):
    sp = x.v > m * x.vsma(n); a = x.atr(14)
    return sp & (x.c > x.o) & ((x.c - x.o) > 0.5 * a), sp & (x.c < x.o) & ((x.o - x.c) > 0.5 * a)
A(S('S44', 'Volume-Spike Continuation', 'Volume', 'Volume spike with a decisive candle: continue', [(2.0, 50), (3.0, 50), (3.0, 200), (4.0, 200)], s44))
def s45(x, n, k):
    z = x.g(('tz', n), lambda: zscore(x.tbr(), n)); return cross_up(z, k), cross_dn(z, -k)
A(S('S45', 'Taker-Buy Imbalance Continuation', 'Volume', 'Aggressive buy share z-score extreme: follow', [(96, 1.5), (96, 2.0), (288, 2.0), (288, 2.5)], s45))
def s46(x, n, k):
    z = x.g(('tz', n), lambda: zscore(x.tbr(), n)); return cross_up(z, -k), cross_dn(z, k)
A(S('S46', 'Taker-Buy Imbalance Exhaustion', 'Volume', 'Aggressive flow z-score extreme reverts: fade', [(96, 2.0), (96, 2.5), (288, 2.0), (288, 3.0)], s46))
def s47(x, n):
    ob = (np.sign(x.c.diff()).fillna(0) * x.v).cumsum(); e = ema(ob, n); t = x.ema(200)
    return cross_up(ob, e) & (x.c > t), cross_dn(ob, e) & (x.c < t)
A(S('S47', 'OBV-EMA Cross + Trend', 'Volume', 'On-balance volume crosses its EMA with EMA200 trend', [(20,), (50,), (100,), (200,)], s47))
def s48(x, n, m):
    e = x.ema(n); vv = x.v > m * x.vsma(20)
    return cross_up(x.c, x.vwap()) & vv & (x.c > e), cross_dn(x.c, x.vwap()) & vv & (x.c < e)
A(S('S48', 'Daily-VWAP Cross + Volume', 'Volume', 'Price crosses daily VWAP on volume, with EMA trend', [(50, 1.0), (50, 1.5), (200, 1.0), (200, 1.5)], s48))

# ---------------- VOLATILITY / TIME ----------------
def s49(x, r, n):
    ex = (x.atr(14) / x.atr(96)) > r; e = x.ema(n)
    return edge(ex & (x.c > e)), edge(ex & (x.c < e))
A(S('S49', 'Volatility-Expansion Trend', 'Vol/Time', 'ATR(14)/ATR(96) expansion, trade side of EMA', [(1.2, 50), (1.5, 50), (1.2, 200), (1.5, 200)], s49))
def s50(x, H, k):
    at = (x.hour == (H - 1) % 24) & (x.minute == 45); r = x.c / x.c.shift(k) - 1
    return at & (r > 0), at & (r < 0)
A(S('S50', 'Session-Open Momentum', 'Vol/Time', 'At fixed UTC hour trade direction of preceding k bars', [(8, 32), (13, 16), (13, 32), (0, 32)], s50))

assert len(STRATS) == 50
SL_GRID = [1.0, 2.0, 3.0]
RR_GRID = [1.0, 2.0, 3.0]
MAX_HOLD = 96
