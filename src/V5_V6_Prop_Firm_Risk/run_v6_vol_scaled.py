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
    print("Building V6 Engine: Volatility-Scaled Risk Parity...")
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
        
    all_net, all_rk, all_exd, all_atr_ratio = [], [], [], []
    
    for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
        df = dfs[asset]
        x_asset = X(df)
        o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
        a14 = x_asset.atr(14).values.astype(np.float64)
        
        # Calculate trailing 100-day ATR to establish a baseline "normal" volatility
        atr_series = pd.Series(a14)
        atr_100_mean = atr_series.rolling(100).mean().bfill().values
        # Volatility Ratio = Baseline Vol / Current Vol. 
        # If Current Vol is 2x normal, ratio is 0.5 (cut risk in half)
        # If Current Vol is 0.5x normal, ratio is 2.0 (double risk)
        vol_scaler = np.clip(atr_100_mean / (a14 + 1e-9), 0.25, 2.5) 
        
        for name in names:
            chrom = [name_to_genes[n] for n in name.split(' AND ')]
            bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, sl, rr, MAX_HOLD)
            m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
            
            all_net.extend(bt[3][m]-COST_RT)
            all_rk.extend(bt[4][m])
            all_exd.extend(DAY[bt[1]][m])
            # Store the volatility scaler at the time of entry
            all_atr_ratio.extend(vol_scaler[bt[0][m]])
            
    net_arr = np.array(all_net)
    rk_arr = np.array(all_rk)
    exd_arr = np.array(all_exd)
    scaler_arr = np.array(all_atr_ratio)
    
    idx_sort = np.argsort(exd_arr)
    net_arr = net_arr[idx_sort]
    rk_arr = rk_arr[idx_sort]
    scaler_arr = scaler_arr[idx_sort]
    
    # R is the unscaled risk-reward outcome of the trade (e.g. +2.0 or -1.0)
    R = net_arr / np.where(rk_arr == 0, 1e-9, rk_arr)
    
    # Base Risk is 2.0%. 
    # But now we multiply by the scaler.
    BASE_RISK = 0.02
    
    eq_fixed = 100.0; peak_fixed = 100.0; dd_fixed = []
    eq_scaled = 100.0; peak_scaled = 100.0; dd_scaled = []
    
    for i in range(len(R)):
        # 1. Fixed Risk (The Old Way)
        r_f = R[i] * BASE_RISK
        eq_fixed *= (1.0 + r_f)
        if eq_fixed > peak_fixed: peak_fixed = eq_fixed
        dd_fixed.append((peak_fixed - eq_fixed) / peak_fixed)
        
        # 2. Volatility Scaled Risk (The New Way)
        # Apply the scaler to the base risk.
        r_s = R[i] * (BASE_RISK * scaler_arr[i])
        eq_scaled *= (1.0 + r_s)
        if eq_scaled > peak_scaled: peak_scaled = eq_scaled
        dd_scaled.append((peak_scaled - eq_scaled) / peak_scaled)
        
    print("\n========================================================")
    print(" V6 VOLATILITY SCALING vs FIXED RISK (Global Ensemble) ")
    print("========================================================")
    
    ret_fixed = (eq_fixed / 100.0 - 1) * 100
    cagr_fixed = (eq_fixed / 100.0) ** (1/7.75) - 1.0
    maxdd_fixed = max(dd_fixed) * 100
    print(f"OLD METHOD (2.0% Fixed Risk)")
    print(f"Total Return: +{ret_fixed:,.1f}% | CAGR: {cagr_fixed*100:.1f}% | Max DD: -{maxdd_fixed:.1f}%\n")
    
    ret_scaled = (eq_scaled / 100.0 - 1) * 100
    cagr_scaled = (eq_scaled / 100.0) ** (1/7.75) - 1.0
    maxdd_scaled = max(dd_scaled) * 100
    print(f"NEW METHOD (V6 Volatility-Scaled Risk Parity)")
    print(f"Total Return: +{ret_scaled:,.1f}% | CAGR: {cagr_scaled*100:.1f}% | Max DD: -{maxdd_scaled:.1f}%")
    print("========================================================")
    
    # Calculate Calmar Ratio (CAGR / Max DD) to prove efficiency
    calmar_old = (cagr_fixed*100) / maxdd_fixed
    calmar_new = (cagr_scaled*100) / maxdd_scaled
    print(f"Old Calmar Ratio: {calmar_old:.2f}")
    print(f"New Calmar Ratio: {calmar_new:.2f}")

if __name__ == '__main__':
    main()
