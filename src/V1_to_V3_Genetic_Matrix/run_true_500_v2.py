import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, time, sys, json, random
import numpy as np, pandas as pd
from scipy import stats

from core import run_bt, to_sig, edge
from eurusd_pipeline import dayoff, pick, trades_of, sel, metrics, daily
from strategies import X, SL_GRID, RR_GRID

random.seed(99) # New seed for new strategies

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

# V2 Conditions: Focused on high-expectancy institutional concepts
CONDITIONS = [
    # Macro Trend Filters (Strict)
    Condition("MacroBull (EMA50 > EMA200)", lambda x: x.ema(50) > x.ema(200), lambda x: x.ema(50) < x.ema(200)),
    Condition("Price > SMA100", lambda x: x.c > x.sma(100), lambda x: x.c < x.sma(100)),
    Condition("Trend Accelerating (MACD Hist > 0)", 
              lambda x: ((x.ema(12)-x.ema(26)) - x.ema(12).ewm(span=9).mean()) > 0,
              lambda x: ((x.ema(12)-x.ema(26)) - x.ema(12).ewm(span=9).mean()) < 0),
    Condition("ADX > 20 (Trending Environment)", lambda x: x.adx(14)[2] > 20, lambda x: x.adx(14)[2] > 20),
    
    # Mean Reversion Triggers (Deep pullbacks)
    Condition("Deep Pullback (RSI < 35)", lambda x: x.rsi(14) < 35, lambda x: x.rsi(14) > 65),
    Condition("Stoch K < 20 (Extreme Oversold)", lambda x: stoch_k(x.df, 14) < 20, lambda x: stoch_k(x.df, 14) > 80),
    Condition("Price < Lower Keltner", 
              lambda x: x.c < (x.sma(20) - 1.5*x.atr(20)), 
              lambda x: x.c > (x.sma(20) + 1.5*x.atr(20))),
    Condition("3 Down Days in a Row", 
              lambda x: (x.c < x.o) & (x.c.shift(1) < x.o.shift(1)) & (x.c.shift(2) < x.o.shift(2)),
              lambda x: (x.c > x.o) & (x.c.shift(1) > x.o.shift(1)) & (x.c.shift(2) > x.o.shift(2))),
    
    # Volatility / Volume Confirmation
    Condition("Volume > SMA20", lambda x: x.v > x.vsma(20), lambda x: x.v > x.vsma(20)),
    Condition("Volatility Crush (ATR14 < ATR72)", 
              lambda x: (x.atr(14) / x.atr(72).replace(0, np.nan)) < 0.95, 
              lambda x: (x.atr(14) / x.atr(72).replace(0, np.nan)) < 0.95),
              
    # Breakout / Structural
    Condition("Inside Bar Breakout", 
              lambda x: (x.h.shift(1) < x.h.shift(2)) & (x.l.shift(1) > x.l.shift(2)) & (x.c > x.h.shift(1)),
              lambda x: (x.h.shift(1) < x.h.shift(2)) & (x.l.shift(1) > x.l.shift(2)) & (x.c < x.l.shift(1))),
    Condition("Close > Highest(5)", 
              lambda x: x.c > x.h.rolling(5).max().shift(), 
              lambda x: x.c < x.l.rolling(5).min().shift())
]

def generate_v2_strategies():
    strats = []; seen = set()
    while len(strats) < 500:
        # Enforce smarter combinations: 1 Trend + 1 Trigger + 1 Confirm
        trend = random.choice(CONDITIONS[0:4])
        trigger = random.choice(CONDITIONS[4:8])
        confirm = random.choice(CONDITIONS[8:])
        
        conds = [trend, trigger, confirm]
        
        # 30% chance to add a 4th random condition
        if random.random() < 0.3:
            extra = random.choice([c for c in CONDITIONS if c not in conds])
            conds.append(extra)
            
        name = " AND ".join([c.name for c in conds])
        if name in seen: continue
        seen.add(name)
        
        def make_fn(cond_list):
            def fn(x, dummy):
                long_sig = pd.Series(True, index=x.df.index)
                short_sig = pd.Series(True, index=x.df.index)
                for c in cond_list:
                    long_sig = long_sig & c.long_fn(x)
                    short_sig = short_sig & c.short_fn(x)
                return edge(long_sig), edge(short_sig)
            return fn
            
        strats.append({
            'id': f"V{len(strats)+1:03d}",
            'name': name,
            'family': 'V2_SmartCombo',
            'grid': [(0,)],
            'fn': make_fn(conds)
        })
    return strats

def load_futures(name='ES_daily'):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
    df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    df['nt'] = df['v']; df['tb'] = df['v'] / 2.0
    return df

def run_asset(df, STRATS, COST_RT, MAX_HOLD_DAILY):
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

    res = {}
    for st in STRATS:
        sid = st['id']
        try:
            lg, sh = st['fn'](x, 0)
        except Exception:
            lg = pd.Series(False, index=df.index); sh = pd.Series(False, index=df.index)
        sig = to_sig(lg, sh); sig[:200] = 0
        
        for sl, rr in [(1.0, 2.0), (2.0, 1.0), (2.0, 2.0)]:
            res[(sid, 0, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD_DAILY)

    OOS = {}
    for st in STRATS:
        sid = st['id']
        nets = []; rks = []; exds = []
        for wi, (a, b, c_win) in enumerate(WINS):
            da, db, dc = (a - START).days, (b - START).days, (c_win - START).days
            best = None; best_sr = -999
            for sl, rr in [(1.0, 2.0), (2.0, 1.0), (2.0, 2.0)]:
                tr = trades_of(res, DAY, (sid, 0, sl, rr), cost=COST_RT)
                m = sel(tr, da, db, purge=True)
                if m.sum() < 5: continue
                mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], da, db)
                if mt['total_ret_pct'] > 0 and mt['sharpe'] > best_sr:
                    best_sr = mt['sharpe']; best = (sl, rr)
            if best is None: continue
            sl, rr = best
            tr = trades_of(res, DAY, (sid, 0, sl, rr), cost=COST_RT)
            m = sel(tr, db, dc)
            if m.sum() > 0:
                nets.append(tr['net'][m]); rks.append(tr['risk'][m])
                exds.append(tr['ex_day'][m])

        cat = lambda L: np.concatenate(L) if L else np.array([])
        OOS[sid] = dict(net=cat(nets), risk=cat(rks), exd=cat(exds))
        
    return OOS

def main():
    print("Generating 500 V2 Smart-Combo strategies...")
    STRATS = generate_v2_strategies()
    
    print("Running Backtests on ES (S&P 500) and NQ (Nasdaq 100) simultaneously...")
    df_es = load_futures('ES_daily')
    df_nq = load_futures('NQ_daily')
    
    # Assuming similar slippage/comm profile as % of value
    t0 = time.time()
    OOS_ES = run_asset(df_es, STRATS, COST_RT=0.00015, MAX_HOLD_DAILY=40)
    print(f"ES done in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    OOS_NQ = run_asset(df_nq, STRATS, COST_RT=0.00015, MAX_HOLD_DAILY=40)
    print(f"NQ done in {time.time()-t0:.1f}s")

    # 3. Combine into Portfolio
    dashboard_data = []
    
    # Common date range for daily aggregation
    START = min(df_es.t.min(), df_nq.t.min())
    END = max(df_es.t.max(), df_nq.t.max())
    d0o = (pd.Timestamp('2003-01-01') - START).days
    d1o = (END - START).days
    
    for st in STRATS:
        sid = st['id']
        
        # Combine trades
        es_net = OOS_ES[sid]['net']; es_rk = OOS_ES[sid]['risk']; es_exd = OOS_ES[sid]['exd']
        nq_net = OOS_NQ[sid]['net']; nq_rk = OOS_NQ[sid]['risk']; nq_exd = OOS_NQ[sid]['exd']
        
        all_net = np.concatenate([es_net, nq_net]) if len(es_net) or len(nq_net) else np.array([])
        all_rk = np.concatenate([es_rk, nq_rk]) if len(es_rk) or len(nq_rk) else np.array([])
        all_exd = np.concatenate([es_exd, nq_exd]) if len(es_exd) or len(nq_exd) else np.array([])
        
        if len(all_net) > 0:
            # Sort chronologically by exit date
            idx = np.argsort(all_exd)
            all_net = all_net[idx]; all_rk = all_rk[idx]; all_exd = all_exd[idx]
            
            mt = metrics(all_net, all_rk, all_exd.astype(int), d0o, d1o)
            
            # Compounding (1.5% fixed risk per trade across the combined portfolio sequence)
            R = all_net / np.where(all_rk == 0, 1e-9, all_rk)
            eq_15 = 100.0
            for rv in R: eq_15 *= (1.0 + 0.015 * rv)
            comp_15 = eq_15 - 100.0
        else:
            mt = metrics(np.array([]), np.array([]), np.array([]), d0o, d1o)
            comp_15 = 0.0

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
    
    with open('dashboard_data_v2.json', 'w') as f:
        json.dump(dashboard_data, f)
        
    print("Saved dashboard_data_v2.json")

if __name__ == '__main__':
    main()
