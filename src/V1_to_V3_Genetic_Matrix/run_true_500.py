import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, time, sys, json, random
import numpy as np, pandas as pd
from scipy import stats

from core import run_bt, to_sig, edge
from eurusd_pipeline import dayoff, pick, trades_of, sel, metrics, daily
from strategies import X, SL_GRID, RR_GRID

# -------------------------------------------------------------------------
# 1. GENERATE 500 STRUCTURALLY UNIQUE STRATEGIES (GENETIC/COMBINATORIAL)
# -------------------------------------------------------------------------
# We will define a set of indicator functions. 
# A strategy will be a random combination of 2 to 4 conditions.

random.seed(42) # For reproducibility

class Condition:
    def __init__(self, name, long_fn, short_fn):
        self.name = name
        self.long_fn = long_fn
        self.short_fn = short_fn

def stoch_k(df, n=14):
    lo_n = df.l.rolling(n).min()
    hi_n = df.h.rolling(n).max()
    denom = (hi_n - lo_n).replace(0, np.nan)
    return 100.0 * (df.c - lo_n) / denom

CONDITIONS = [
    # Trend
    Condition("Price > SMA200", lambda x: x.c > x.sma(200), lambda x: x.c < x.sma(200)),
    Condition("Price > SMA50", lambda x: x.c > x.sma(50), lambda x: x.c < x.sma(50)),
    Condition("EMA20 > EMA50", lambda x: x.ema(20) > x.ema(50), lambda x: x.ema(20) < x.ema(50)),
    Condition("MACD > 0", lambda x: (x.ema(12) - x.ema(26)) > 0, lambda x: (x.ema(12) - x.ema(26)) < 0),
    Condition("MACD Hist Up", 
              lambda x: ((x.ema(12)-x.ema(26)) - x.ema(12).ewm(span=9).mean()) > 0,
              lambda x: ((x.ema(12)-x.ema(26)) - x.ema(12).ewm(span=9).mean()) < 0),
    
    # Momentum
    Condition("RSI > 50", lambda x: x.rsi(14) > 50, lambda x: x.rsi(14) < 50),
    Condition("RSI < 40 (Pullback)", lambda x: x.rsi(14) < 40, lambda x: x.rsi(14) > 60),
    Condition("Stoch K < 30 (Oversold)", lambda x: stoch_k(x.df, 14) < 30, lambda x: stoch_k(x.df, 14) > 70),
    Condition("Stoch K > D", lambda x: stoch_k(x.df, 14) > stoch_k(x.df, 14).rolling(3).mean(), 
                             lambda x: stoch_k(x.df, 14) < stoch_k(x.df, 14).rolling(3).mean()),
    Condition("ROC10 > 0", lambda x: x.c.pct_change(10) > 0, lambda x: x.c.pct_change(10) < 0),
    
    # Volatility & Breakout
    Condition("ATR Expand", lambda x: (x.atr(14) / x.atr(72).replace(0, np.nan)) > 1.05, 
                            lambda x: (x.atr(14) / x.atr(72).replace(0, np.nan)) > 1.05),
    Condition("Close > BB Upper", 
              lambda x: x.c > (x.sma(20) + 2*x.c.rolling(20).std()), 
              lambda x: x.c < (x.sma(20) - 2*x.c.rolling(20).std())),
    Condition("Close > Donchian20", 
              lambda x: x.c > x.h.rolling(20).max().shift(), 
              lambda x: x.c < x.l.rolling(20).min().shift()),
              
    # Volume
    Condition("Volume Spike", lambda x: x.v > (x.vsma(20) * 1.3), lambda x: x.v > (x.vsma(20) * 1.3)),
    
    # Price Action
    Condition("Green Candle", lambda x: x.c > x.o, lambda x: x.c < x.o),
    Condition("Higher High", lambda x: x.h > x.h.shift(1), lambda x: x.l < x.l.shift(1)),
    Condition("Swing Low", 
              lambda x: (x.l.shift(1) < x.l.shift(2)) & (x.c > x.h.shift(1)),
              lambda x: (x.h.shift(1) > x.h.shift(2)) & (x.c < x.l.shift(1)))
]

def generate_500_strategies():
    strats = []
    seen_names = set()
    
    while len(strats) < 500:
        # Pick 2 to 4 random conditions
        num_conds = random.choice([2, 3, 3, 4])
        conds = random.sample(CONDITIONS, num_conds)
        
        name = " AND ".join([c.name for c in conds])
        if name in seen_names:
            continue
            
        seen_names.add(name)
        
        # Capture the conditions in a closure
        def make_fn(cond_list):
            def fn(x, dummy):
                long_sig = pd.Series(True, index=x.df.index)
                short_sig = pd.Series(True, index=x.df.index)
                for c in cond_list:
                    long_sig = long_sig & c.long_fn(x)
                    short_sig = short_sig & c.short_fn(x)
                # Ensure we only trigger on the edge (when it becomes true)
                return edge(long_sig), edge(short_sig)
            return fn
            
        sid = f"S{len(strats)+1:03d}"
        strats.append({
            'id': sid,
            'name': name,
            'family': 'Combinatorial',
            'grid': [(0,)], # dummy param
            'fn': make_fn(conds)
        })
        
    return strats


# -------------------------------------------------------------------------
# 2. RUN BACKTEST
# -------------------------------------------------------------------------
def load_futures(name='ES_daily'):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
    df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    df['nt'] = df['v']
    df['tb'] = df['v'] / 2.0
    return df

def main():
    print("Generating 500 structurally unique strategies...")
    STRATS = generate_500_strategies()
    
    print("Loading ES Futures Daily data...")
    df = load_futures('ES_daily')
    x = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x.atr(14).values.astype(np.float64)
    
    START = df.t.min(); END = df.t.max()
    T = pd.to_datetime(df.t.values)
    DAY = ((T.floor('D') - START).days).values

    WINS = []
    ts = pd.Timestamp('2003-01-01')
    while ts < END:
        te = min(ts + pd.DateOffset(months=12), END)
        WINS.append((ts - pd.DateOffset(months=24), ts, te))
        ts = ts + pd.DateOffset(months=12)

    d0o = (WINS[0][1] - START).days
    d1o = (END - START).days
    FUTURES_COST_RT = 0.00015
    MAX_HOLD_DAILY = 40

    t0 = time.time()
    res = {}
    for st in STRATS:
        sid = st['id']
        try:
            lg, sh = st['fn'](x, 0)
        except Exception:
            lg = pd.Series(False, index=df.index); sh = pd.Series(False, index=df.index)
        sig = to_sig(lg, sh); sig[:200] = 0
        
        # Reduced param grid for speed (only testing 3 standard SL/RR combos instead of 9)
        for sl, rr in [(1.0, 2.0), (2.0, 1.0), (2.0, 2.0)]:
            res[(sid, 0, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD_DAILY)
    
    print(f"Backtests done in {time.time() - t0:.1f}s")

    WF_ROWS = []
    OOS = {}
    for st in STRATS:
        sid = st['id']
        nets = []; rks = []; exds = []; grosses = []
        for wi, (a, b, c_win) in enumerate(WINS):
            da, db, dc = (a - START).days, (b - START).days, (c_win - START).days
            
            # Simple pick: best IS sharpe
            best = None; best_sr = -999
            for sl, rr in [(1.0, 2.0), (2.0, 1.0), (2.0, 2.0)]:
                tr = trades_of(res, DAY, (sid, 0, sl, rr), cost=FUTURES_COST_RT)
                m = sel(tr, da, db, purge=True)
                if m.sum() < 5: continue
                mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], da, db)
                if mt['total_ret_pct'] > 0 and mt['sharpe'] > best_sr:
                    best_sr = mt['sharpe']; best = (sl, rr)
            
            if best is None: continue
            
            sl, rr = best
            tr = trades_of(res, DAY, (sid, 0, sl, rr), cost=FUTURES_COST_RT)
            m = sel(tr, db, dc)
            
            if m.sum() > 0:
                nets.append(tr['net'][m]); rks.append(tr['risk'][m])
                exds.append(tr['ex_day'][m]); grosses.append(tr['gross'][m])

        cat = lambda L: np.concatenate(L) if L else np.array([])
        OOS[sid] = dict(net=cat(nets), risk=cat(rks), exd=cat(exds).astype(int) if exds else np.array([], int))

    # Compile Final JSON for Dashboard
    dashboard_data = []
    for st in STRATS:
        sid = st['id']
        o_tr = OOS[sid]
        mt = metrics(o_tr['net'], o_tr['risk'], o_tr['exd'], d0o, d1o)
        
        # Compounding sim (1.5% risk)
        net = o_tr['net']; rk = o_tr['risk']
        comp_15 = 0.0
        if len(net) > 0 and len(rk) > 0:
            R = net / np.where(rk == 0, 1e-9, rk)
            eq_15 = 100.0
            for rv in R: eq_15 *= (1.0 + 0.015 * rv)
            comp_15 = eq_15 - 100.0
            
        dashboard_data.append({
            'id': sid,
            'name': st['name'],
            'trades': mt['trades'],
            'win_rate': round(mt['win_rate']*100, 1) if not np.isnan(mt['win_rate']) else 0,
            'pf': round(mt.get('profit_factor', 0), 2) if mt.get('profit_factor', 0) != np.inf else 0,
            'sharpe': round(mt['sharpe'], 2),
            'compounded': round(comp_15, 1)
        })
        
    dashboard_data.sort(key=lambda x: x['sharpe'], reverse=True)
    
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f)
        
    print("Saved dashboard_data.json")

if __name__ == '__main__':
    main()
