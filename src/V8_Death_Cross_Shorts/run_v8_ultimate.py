import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel
from core import run_bt, to_sig, edge
from strategies import X
import consistent_reversion

def load_global_data():
    dfs = load_all_data()
    for name in ['DAX_daily', 'Nikkei_daily']:
        df = pd.read_csv(f'../data/{name}.csv', sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        master_index = dfs['ES_daily'].set_index('t').index
        dfs[name] = df.sort_values('t').drop_duplicates('t').set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

def main():
    print("Building V8 Ultimate Engine (V7 Longs + Bear Rally Shorts)...")
    dfs = load_global_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    COST_RT = 0.00015
    
    with open('dashboard_data_v3_ml.json', 'r') as f:
        names = [s['name'] for s in json.load(f)[:5]]
        
    all_trades = []
    
    for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
        df = dfs[asset]
        x_asset = X(df)
        o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
        a14 = x_asset.atr(14).values.astype(np.float64)
        
        # 1. LONG ENSEMBLE TRADES
        for name in names:
            chrom = [name_to_genes[n] for n in name.split(' AND ')]
            bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, 2.0, 2.0, 40)
            m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
            
            ex_days = DAY[bt[1]][m]
            rets = bt[3][m] - COST_RT
            risks = bt[4][m]
            for i in range(len(ex_days)):
                r = rets[i] / (risks[i] if risks[i] != 0 else 1e-9)
                all_trades.append({'ex_day': ex_days[i], 'R': r, 'type': 'LONG'})
                
        # 2. SHORT ENGINE TRADES
        sma200 = df['c'].rolling(200).mean()
        rsi2 = consistent_reversion.rsi(df['c'], 2)
        short_sig = (df['c'] < sma200) & (rsi2 > 90)
        sig = to_sig(pd.Series(False, index=df.index), edge(short_sig))
        sig[:200] = 0
        
        bt_short = run_bt(o, h, l, c, sig, a14, sl_m=1.0, rr=3.0, max_hold=10)
        ms = sel(dict(ent_day=DAY[bt_short[0]], ex_day=DAY[bt_short[1]]), d0_test, d1_test)
        
        ex_days_s = DAY[bt_short[1]][ms]
        rets_s = bt_short[3][ms] - COST_RT
        risks_s = bt_short[4][ms]
        for i in range(len(ex_days_s)):
            r = rets_s[i] / (risks_s[i] if risks_s[i] != 0 else 1e-9)
            all_trades.append({'ex_day': ex_days_s[i], 'R': r, 'type': 'SHORT'})
            
    # Sort all trades chronologically by exit day
    all_trades.sort(key=lambda x: x['ex_day'])
    
    # Portfolio Simulation
    eq = 100.0
    peak = 100.0
    drawdowns = []
    
    # We maintain a separate Long-Only equity tracker to feed the V7 filter
    long_eq = 100.0
    running_long_eqs = []
    
    dates = []
    eq_curve = []
    
    for t in all_trades:
        if t['type'] == 'LONG':
            running_long_eqs.append(long_eq)
            if len(running_long_eqs) > 20:
                ma20 = np.mean(running_long_eqs[-20:])
                risk = 0.02 * 0.25 if long_eq < ma20 else 0.02
            else:
                risk = 0.02
                
            long_eq *= (1.0 + risk * t['R'])
            eq *= (1.0 + risk * t['R'])
            
        elif t['type'] == 'SHORT':
            # Shorts use a fixed aggressive 5% risk because they only trigger during verified crashes
            risk = 0.05 
            eq *= (1.0 + risk * t['R'])
            
        if eq > peak: peak = eq
        drawdowns.append((peak - eq) / peak)
        
        dates.append(pd.Timestamp('2000-09-18') + pd.to_timedelta(t['ex_day'], unit='D'))
        eq_curve.append(eq)
        
    # Analysis
    res_df = pd.DataFrame({'date': dates, 'eq': eq_curve})
    res_df['year'] = res_df.date.dt.year
    
    print("\n========================================================")
    print(" V8 ULTIMATE PORTFOLIO (V7 Longs + Bear Rally Shorts) ")
    print("========================================================")
    print("Year | Trades | Compounded Return")
    print("-----|--------|------------------")
    
    for y in sorted(res_df.year.unique()):
        y_data = res_df[res_df.year == y]
        y_trades = len(y_data)
        
        if y_data.empty: continue
        
        # Calculate year return based on equity at start vs end of year
        start_eq = eq_curve[y_data.index[0] - 1] if y_data.index[0] > 0 else 100.0
        end_eq = y_data.iloc[-1]['eq']
        y_ret = (end_eq / start_eq - 1) * 100
        print(f"{y} | {y_trades:6d} | {y_ret:15.1f}%")
        
    cagr = (eq / 100.0) ** (1 / 7.75) - 1.0
    
    print("--------------------------------------------------------")
    print(f"Total OOS Return (2019-2026): +{(eq/100.0 - 1.0)*100:,.1f}%")
    print(f"Global Compound Annual (CAGR): {cagr*100:.1f}%")
    print(f"Maximum Drawdown:             {max(drawdowns)*100:.1f}%")
    print("========================================================")

if __name__ == '__main__':
    main()
