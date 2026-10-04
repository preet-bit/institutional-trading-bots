import numpy as np, pandas as pd
from numba import njit

import os
CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc_15m_data_2018_to_2025.csv'))
FEE = 0.0004      # taker fee per side
SLIP = 0.0002     # slippage per side
COST_RT = 2 * (FEE + SLIP)   # 0.12% round trip, deducted from every trade


def load():
    df = pd.read_csv(CSV).dropna(subset=['Open time'])
    df['t'] = pd.to_datetime(df['Open time'])
    # two corrupt rows (price ~108k) on 2018-07-07 -> drop
    bad = df.index[df.Close > 50000][df.index[df.Close > 50000] < 20000]
    df = df.drop(bad)
    df = df.drop_duplicates('t').sort_values('t').reset_index(drop=True)
    df = df.rename(columns={'Open': 'o', 'High': 'h', 'Low': 'l', 'Close': 'c', 'Volume': 'v',
                            'Number of trades': 'nt', 'Taker buy base asset volume': 'tb'})
    return df[['t', 'o', 'h', 'l', 'c', 'v', 'nt', 'tb']].reset_index(drop=True)


@njit(cache=True)
def run_bt(o, h, l, c, sig, atr, sl_m, rr, max_hold):
    """sig[i] in {+1,-1,0} known at close of bar i -> enter at open of i+1.
    SL = sl_m*ATR, TP = rr*SL. If SL and TP both touched in one bar -> SL (conservative).
    Gap through stop/target at open fills at the open. Timeout exits at close."""
    n = len(c)
    cap = 200000
    ei = np.empty(cap, np.int64); xi = np.empty(cap, np.int64)
    dr = np.empty(cap, np.int8); gr = np.empty(cap); rk = np.empty(cap)
    k = 0
    i = 0
    while i < n - 2:
        d = sig[i]
        if d != 0 and atr[i] > 0 and atr[i] == atr[i]:
            e = o[i + 1]
            risk = sl_m * atr[i]
            sl = e - d * risk
            tp = e + d * risk * rr
            xp = c[min(n - 1, i + max_hold)]
            xj = min(n - 1, i + max_hold)
            for j in range(i + 1, min(n, i + 1 + max_hold)):
                if j > i + 1:
                    if d == 1:
                        if o[j] <= sl: xp = o[j]; xj = j; break
                        if o[j] >= tp: xp = o[j]; xj = j; break
                    else:
                        if o[j] >= sl: xp = o[j]; xj = j; break
                        if o[j] <= tp: xp = o[j]; xj = j; break
                if d == 1:
                    hs = l[j] <= sl; ht = h[j] >= tp
                else:
                    hs = h[j] >= sl; ht = l[j] <= tp
                if hs: xp = sl; xj = j; break
                if ht: xp = tp; xj = j; break
            if k < cap:
                ei[k] = i + 1; xi[k] = xj; dr[k] = d
                gr[k] = d * (xp / e - 1.0); rk[k] = risk / e
                k += 1
            i = max(xj, i + 1)
        else:
            i += 1
    return ei[:k], xi[:k], dr[:k], gr[:k], rk[:k]


# ---------------- indicators (all causal) ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rma(s, n): return s.ewm(alpha=1.0 / n, adjust=False).mean()

def tr(df):
    pc = df.c.shift()
    return pd.concat([df.h - df.l, (df.h - pc).abs(), (df.l - pc).abs()], axis=1).max(axis=1)

def atr(df, n=14): return rma(tr(df), n)

def rsi(s, n=14):
    d = s.diff(); up = rma(d.clip(lower=0), n); dn = rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def stoch(df, n=14, k=3):
    ll = df.l.rolling(n).min(); hh = df.h.rolling(n).max()
    kk = 100 * (df.c - ll) / (hh - ll).replace(0, np.nan)
    return sma(kk, k)

def adx(df, n=14):
    up = df.h.diff(); dn = -df.l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * rma(pd.Series(pdm, index=df.index), n) / a
    mdi = 100 * rma(pd.Series(mdm, index=df.index), n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pdi, mdi, rma(dx, n)

def cross_up(a, b):
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift() <= b.shift())

def cross_dn(a, b):
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift() >= b.shift())

def edge(cond):
    cond = cond.fillna(False)
    return cond & ~cond.shift().fillna(False)

@njit(cache=True)
def _supertrend(h, l, c, a, mult):
    n = len(c); dirn = np.zeros(n, np.int8)
    ub = np.full(n, np.nan); lb = np.full(n, np.nan)
    d = 1
    for i in range(1, n):
        if a[i] != a[i]: continue
        hl2 = (h[i] + l[i]) / 2
        u = hl2 + mult * a[i]; lo = hl2 - mult * a[i]
        pu = ub[i - 1]; pl = lb[i - 1]
        if pu == pu:
            if not (u < pu or c[i - 1] > pu): u = pu
            if not (lo > pl or c[i - 1] < pl): lo = pl
        ub[i] = u; lb[i] = lo
        if d == 1 and c[i] < lo: d = -1
        elif d == -1 and c[i] > u: d = 1
        dirn[i] = d
    return dirn

def supertrend(df, n, mult):
    return pd.Series(_supertrend(df.h.values, df.l.values, df.c.values, atr(df, n).values, mult), index=df.index)

def wma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    v = np.convolve(s.values, w[::-1], 'valid') / w.sum()
    return pd.Series(np.concatenate([np.full(n - 1, np.nan), v]), index=s.index)

def hma(s, n):
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))

def daily_vwap(df):
    day = df.t.dt.floor('D'); tp = (df.h + df.l + df.c) / 3
    pv = (tp * df.v).groupby(day).cumsum(); vv = df.v.groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)

def zscore(s, n): return (s - sma(s, n)) / s.rolling(n).std()

def linreg_slope(s, n):
    idx = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    return idx.rolling(n).cov(s) / idx.rolling(n).var()

def to_sig(long, short):
    long = long.fillna(False).values; short = short.fillna(False).values
    s = np.zeros(len(long), np.int8)
    s[long & ~short] = 1; s[short & ~long] = -1
    return s
