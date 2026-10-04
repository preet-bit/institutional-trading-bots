import os, time, pickle, sys
import numpy as np, pandas as pd
from scipy import stats

from core import run_bt, to_sig, edge
from eurusd_pipeline import (
    dayoff, pick, trades_of, sel, metrics, daily, sharpe, maxdd
)
from strategies import X, dc_hi, dc_lo, SL_GRID, RR_GRID, MAX_HOLD

from futures_strategy_factory import get_500_strategies

# -------------------------------------------------------------
# Setup & Cost Model for ES Futures (Daily)
# -------------------------------------------------------------
# ES point value = $50. Commission + Slippage = ~0.60 points RT.
# 0.60 points at index value 4000 = 0.00015 (1.5 bps)
FUTURES_COST_RT = 0.00015
MAX_HOLD_DAILY = 40  # 40 days = ~8 weeks max hold

def load_futures(name='ES_daily'):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
    df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').drop_duplicates('t').reset_index(drop=True)
    df['nt'] = df['v']
    df['tb'] = df['v'] / 2.0  # Synthetic
    return df

def _exp_max_sr_val(N, avg_sr=0.2, std_sr=0.6):
    g = 0.5772156649
    return std_sr * ((1 - g) * stats.norm.ppf(1 - 1/N) + g * stats.norm.ppf(1 - 1/(N * np.e)))

def _dsr(sr_d, sr0, T, skew, kurt):
    den = np.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return float(stats.norm.cdf((sr_d - sr0) * np.sqrt(max(T - 1, 1)) / den))

def main():
    print("Loading ES Futures Daily data...")
    df = load_futures('ES_daily')
    x = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x.atr(14).values.astype(np.float64)
    
    START = df.t.min()
    END = df.t.max()
    T = pd.to_datetime(df.t.values)
    DAY = ((T.floor('D') - START).days).values

    print(f"Loaded {len(df)} bars from {START} to {END}")

    # Walk-forward windows (2 years train, 1 year test, 1 year roll for daily)
    WINS = []
    ts = pd.Timestamp('2003-01-01')
    while ts < END:
        te = min(ts + pd.DateOffset(months=12), END)
        WINS.append((ts - pd.DateOffset(months=24), ts, te))
        ts = ts + pd.DateOffset(months=12)

    d0o = (WINS[0][1] - START).days
    d1o = (END - START).days

    STRATS = get_500_strategies()
    total_N = len(STRATS)
    print(f"Evaluating {total_N} strategies × 9 grids = {total_N*9} combos on {len(WINS)} windows...")

    t0 = time.time()
    res = {}
    
    for st in STRATS:
        sid = st['id']
        for pi, p in enumerate(st['grid']):
            try:
                lg, sh = st['fn'](x, *p)
            except Exception as e_:
                lg = pd.Series(False, index=df.index); sh = pd.Series(False, index=df.index)
            sig = to_sig(lg, sh)
            sig[:100] = 0
            for sl in SL_GRID:
                for rr in RR_GRID:
                    res[(sid, pi, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD_DAILY)
    
    print(f"Backtests done in {time.time() - t0:.1f}s")

    # Walk-forward evaluation
    WF_ROWS = []
    OOS = {}
    
    for st in STRATS:
        sid = st['id']
        nets = []; rks = []; exds = []; grosses = []
        for wi, (a, b, c_win) in enumerate(WINS):
            da, db, dc = (a - START).days, (b - START).days, (c_win - START).days
            
            # Use `pick` from eurusd_pipeline (modified slightly to pass correct DAY and db/dc)
            best, breadth = pick(res, DAY, st, da, db, min_trades=5)
            
            row = dict(strategy_id=sid, strategy=st['name'], family=st['family'], window=wi + 1)
            
            if best is None:
                row.update(selected='NO TRADE')
                WF_ROWS.append(row)
                continue
                
            (pi, sl, rr), im = best
            tr = trades_of(res, DAY, (sid, pi, sl, rr), cost=FUTURES_COST_RT)
            m = sel(tr, db, dc)
            om = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], db, dc)
            
            row.update(
                selected=f"SL={sl} RR={rr}", 
                is_trades=im['trades'], is_sharpe=im['sharpe'],
                oos_trades=om['trades'], oos_sharpe=om['sharpe'], oos_ret_pct=om['total_ret_pct']
            )
            WF_ROWS.append(row)
            
            if m.sum() > 0:
                nets.append(tr['net'][m]); rks.append(tr['risk'][m])
                exds.append(tr['ex_day'][m]); grosses.append(tr['gross'][m])

        cat = lambda L: np.concatenate(L) if L else np.array([])
        OOS[sid] = dict(
            net=cat(nets), risk=cat(rks), exd=cat(exds).astype(int) if exds else np.array([], int)
        )

    # Compile Final Results
    rows = []
    for st in STRATS:
        sid = st['id']
        o_tr = OOS[sid]
        mt = metrics(o_tr['net'], o_tr['risk'], o_tr['exd'], d0o, d1o)
        dly = daily(o_tr['net'], o_tr['exd'], d0o, d1o) if len(o_tr['net']) else np.zeros(d1o - d0o)

        act = pd.DataFrame(WF_ROWS)
        act = act[(act.strategy_id == sid) & (act.oos_trades.fillna(0) > 0)]
        traded_win = len(act)
        prof_win = (act.oos_ret_pct > 0).sum() if traded_win else 0
        pct_prof = prof_win / traded_win if traded_win else 0.0
        
        # DSR calculation
        if len(o_tr['net']) > 5:
            skew_v = float(stats.skew(dly[dly != 0])) if (dly != 0).sum() > 3 else 0.0
            kurt_v = float(stats.kurtosis(dly[dly != 0])) if (dly != 0).sum() > 3 else 0.0
        else:
            skew_v = kurt_v = 0.0
            
        exp_sr = _exp_max_sr_val(500) # Penalty for 500 strategies
        dsr_prob = _dsr(mt['sharpe'], exp_sr, d1o - d0o, skew_v, kurt_v)
        
        gate_pass = (
            mt['trades'] >= 30 and 
            mt.get('profit_factor', 0.0) >= 1.05 and 
            pct_prof >= 0.50 and 
            dsr_prob >= 0.90
        )
        
        mt.update(
            strategy_id=sid, strategy=st['name'], family=st['family'],
            pct_windows_profitable=pct_prof, dsr_prob=dsr_prob,
            verdict='PASS' if gate_pass else 'FAIL'
        )
        rows.append(mt)

    EV = pd.DataFrame(rows)
    
    # Sort and Print top 15
    EV = EV.sort_values('sharpe', ascending=False)
    
    print("\n" + "=" * 80)
    print("TOP 15 OUT-OF-SAMPLE STRATEGIES (ES Daily)")
    print("=" * 80)
    cols = ['strategy_id', 'strategy', 'trades', 'win_rate', 'profit_factor', 'sharpe', 'pct_windows_profitable', 'dsr_prob', 'verdict']
    print(EV[cols].head(15).to_string(index=False))

    # Save to Excel
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'Futures_500_Strategies_ES.xlsx'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    EV.to_excel(out_path, index=False, sheet_name='Leaderboard')
    pd.DataFrame(WF_ROWS).to_excel(out_path, index=False, sheet_name='WalkForward')
    print(f"\nSaved full results to {out_path}")

if __name__ == '__main__':
    main()
