import os, json
import numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel
from strategies import X

def load_crypto_data():
    dfs = load_all_data() # Gets ES, NQ, Gold, Oil, Yields
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'BTC_daily.csv'))
    df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
    df['t'] = pd.to_datetime(df['t'])
    
    # We will reindex BTC to the ES calendar so the macro indicators align perfectly.
    # We lose weekend BTC trades, but gain strict institutional intermarket filters.
    master_index = dfs['ES_daily'].set_index('t').index
    dfs['BTC_daily'] = df.set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

def main():
    print("Initializing The $3 Degen Crypto Engine (BTC-USD)...")
    dfs = load_crypto_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    T = pd.to_datetime(dfs['ES_daily'].t.values)
    DAY = ((T.floor('D') - START).days).values
    
    # We will test out of sample on the crypto boom from 2019 to 2026
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    COST_RT = 0.0010 # 0.1% cost for crypto taker fees
    MAX_HOLD = 40; sl, rr = 2.0, 2.0
    
    with open('dashboard_data_v3_ml.json', 'r') as f:
        names = [s['name'] for s in json.load(f)[:5]]
        
    all_net, all_rk, all_exd = [], [], []
    
    df = dfs['BTC_daily']
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
    
    dates = pd.Timestamp('2000-09-18') + pd.to_timedelta(exd_arr, unit='D')
    trades_df = pd.DataFrame({'date': dates, 'R': R})
    trades_df['year'] = trades_df.date.dt.year
    
    # The $3 Account Simulation
    eq = 3.00
    peak = 3.00
    drawdowns = []
    
    # Degen Risk: Risking 15% of the account on EVERY trade
    RISK = 0.15 
    
    print("\n========================================================")
    print(" DEGEN CRYPTO CHALLENGE: Turning $3 into something real ")
    print("========================================================")
    print(f"Algorithm: V4 Intermarket Ensemble (Filtered by Gold/Yields)")
    print(f"Risk per Trade: {RISK*100}% of total account")
    print("--------------------------------------------------------")
    print("Year | Trades | End of Year Balance")
    print("-----|--------|--------------------")
    
    for y in sorted(trades_df.year.unique()):
        y_trades = trades_df[trades_df.year == y]
        for r in y_trades.R: 
            eq *= (1.0 + RISK * r)
            if eq > peak: peak = eq
            drawdowns.append((peak - eq) / peak)
        print(f"{y} | {len(y_trades):6d} | ${eq:10.2f}")
        
    print("--------------------------------------------------------")
    print(f"Final Account Balance: ${eq:,.2f}")
    print(f"Total Multiplier:      {eq/3.0:,.1f}x")
    print(f"Maximum Drawdown:      -{max(drawdowns)*100:.1f}%")
    print("========================================================")
    
if __name__ == '__main__':
    main()
