import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel
from strategies import X

def load_global_data():
    dfs = load_all_data()
    for name in ['DAX_daily', 'Nikkei_daily']:
        df = pd.read_csv(f'../data/{name}.csv', sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        master_index = dfs['ES_daily'].set_index('t').index
        dfs[name] = df.sort_values('t').drop_duplicates('t').set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

def main():
    print("Preparing Global Ensemble Trades for Prop Firm Monte Carlo...")
    dfs = load_global_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    COST_RT = 0.00015; MAX_HOLD = 40; sl, rr = 2.0, 2.0
    
    with open('dashboard_data_v3_ml.json', 'r') as f:
        names = [s['name'] for s in json.load(f)[:5]]
        
    all_net, all_rk, all_exd = [], [], []
    for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
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
    net_arr, rk_arr = net_arr[idx_sort], rk_arr[idx_sort]
    R = net_arr / np.where(rk_arr == 0, 1e-9, rk_arr)
    
    RISK_PCT = 0.0012 # 0.12% Risk ($60 on $50k)
    SIMULATIONS = 5000
    
    print(f"\nRunning {SIMULATIONS} Monte Carlo Simulations at 0.12% Risk...")
    np.random.seed(42)
    
    sim_cagrs = []
    sim_dds = []
    blown_accounts = 0
    
    for i in range(SIMULATIONS):
        sim_R = np.random.choice(R, size=len(R), replace=True)
        eq = 100.0
        peak = 100.0
        max_dd = 0.0
        
        for r in sim_R:
            eq *= (1.0 + RISK_PCT * r)
            if eq > peak: peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd: max_dd = dd
            
            # If drawdown hits 5%, account is blown.
            if dd >= 0.05:
                blown_accounts += 1
                break
                
        if max_dd < 0.05:
            cagr = (eq / 100.0) ** (1 / 7.75) - 1.0
            sim_cagrs.append(cagr * 100)
        sim_dds.append(max_dd * 100)
        
    print("\n========================================================")
    print(f" MONTE CARLO STRESS TEST: $50k Prop Firm Limit (5%)")
    print("========================================================")
    print(f"Total Simulations:             {SIMULATIONS}")
    print(f"Risk Per Trade:                0.12% ($60)")
    print(f"Mean Expected CAGR:            {np.mean(sim_cagrs):.2f}%")
    print(f"95th Percentile Worst DD:      {np.percentile(sim_dds, 95):.2f}%")
    print(f"99th Percentile Worst DD:      {np.percentile(sim_dds, 99):.2f}%")
    print("--------------------------------------------------------")
    print(f"Probability of Blowing Acct (>5% DD): {(blown_accounts/SIMULATIONS)*100:.2f}%")
    print("========================================================")

if __name__ == '__main__':
    main()
