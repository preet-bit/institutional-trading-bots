import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, json
import numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel, metrics
from strategies import X

def load_global_data():
    dfs = load_all_data() # Gets ES, NQ, Gold, Oil, Yields
    for name in ['DAX_daily', 'Nikkei_daily']:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        df = df.sort_values('t').drop_duplicates('t')
        
        master_index = dfs['ES_daily'].set_index('t').index
        # Align global indices to the ES calendar (ffill weekends/holidays)
        dfs[name] = df.set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

def main():
    print("Building V4 Global Ensemble (US + Europe + Asia)...")
    dfs = load_global_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    T = pd.to_datetime(dfs['ES_daily'].t.values)
    DAY = ((T.floor('D') - START).days).values
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    COST_RT = 0.00015; MAX_HOLD = 40; sl, rr = 2.0, 2.0
    
    with open('dashboard_data_v3_ml.json', 'r') as f:
        names = [s['name'] for s in json.load(f)[:5]]
        
    all_net, all_rk, all_exd = [], [], []
    
    assets = ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']
    for asset in assets:
        df = dfs[asset]
        x_asset = X(df)
        o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
        a14 = x_asset.atr(14).values.astype(np.float64)
        
        for name in names:
            chrom = [name_to_genes[n] for n in name.split(' AND ')]
            bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, sl, rr, MAX_HOLD)
            m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
            all_net.extend(bt[3][m]-COST_RT)
            all_rk.extend(bt[4][m])
            all_exd.extend(DAY[bt[1]][m])
            
    net_arr, rk_arr, exd_arr = np.array(all_net), np.array(all_rk), np.array(all_exd)
    idx_sort = np.argsort(exd_arr)
    net_arr, rk_arr, exd_arr = net_arr[idx_sort], rk_arr[idx_sort], exd_arr[idx_sort]
    R = net_arr / np.where(rk_arr == 0, 1e-9, rk_arr)
    
    # Simulate Global Portfolio at 2% risk
    eq_us_only = 68.8 # US Only was 6,784% return (68.8x multiple)
    
    eq = 100.0
    peak = 100.0
    drawdowns = []
    
    for r in R:
        eq *= (1.0 + 0.02 * r)
        if eq > peak: peak = eq
        drawdowns.append((peak - eq) / peak)
        
    comp_ret = (eq / 100.0 - 1.0) * 100
    cagr = (eq / 100.0) ** (1 / 7.75) - 1.0
    
    print("\n========================================================")
    print(" V4 GLOBAL ENSEMBLE (ES + NQ + DAX + NIKKEI) @ 2% Risk  ")
    print("========================================================")
    print(f"Total OOS Trades (2019-2026): {len(R)}")
    print(f"Total OOS Compounded Return:  +{comp_ret:,.1f}%")
    print(f"Global Compound Annual (CAGR): {cagr*100:.1f}%")
    print(f"Maximum Drawdown:             {max(drawdowns)*100:.1f}%")
    
if __name__ == '__main__':
    main()
