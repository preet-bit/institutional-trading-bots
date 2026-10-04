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
    dfs = load_global_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
    d0_test = (pd.Timestamp('2019-01-01') - START).days; d1_test = (END - START).days
    
    with open('dashboard_data_v3_ml.json', 'r') as f: names = [s['name'] for s in json.load(f)[:5]]
    
    all_trades = []
    for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
        df = dfs[asset]; x_asset = X(df)
        o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
        a14 = x_asset.atr(14).values.astype(np.float64)
        for name in names:
            chrom = [name_to_genes[n] for n in name.split(' AND ')]
            bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, 2.0, 2.0, 40)
            m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
            ex_days = DAY[bt[1]][m]; rets = bt[3][m] - 0.00015; risks = bt[4][m]
            for i in range(len(ex_days)): 
                all_trades.append({'ex_day': ex_days[i], 'R': rets[i]/(risks[i] or 1e-9)})
                
    all_trades.sort(key=lambda x: x['ex_day'])
    
    eq = 100.0; running_eqs = []; dates = []; eq_curve = []
    for t in all_trades:
        running_eqs.append(eq)
        risk = 0.02 * 0.25 if (len(running_eqs) > 20 and eq < np.mean(running_eqs[-20:])) else 0.02
        eq *= (1.0 + risk * t['R'])
        dates.append(pd.Timestamp('2000-09-18') + pd.to_timedelta(t['ex_day'], unit='D'))
        eq_curve.append(eq)

    df_eq = pd.DataFrame({'date': dates, 'eq': eq_curve})
    
    # We must properly align it to a daily timeline
    daily_idx = pd.date_range(start='2019-01-01', end=df_eq.date.max(), freq='D')
    daily_eq = pd.Series(index=daily_idx, dtype=float)
    daily_eq.iloc[0] = 100.0
    for i, r in df_eq.iterrows():
        daily_eq.loc[r['date']] = r['eq']
    daily_eq = daily_eq.ffill()
    
    # Calculate monthly percentage change
    monthly_eq = daily_eq.resample('M').last()
    
    print("\n=====================================================================================================================")
    print(" V7 INSTITUTIONAL ALGORITHM - MONTHLY RETURN MATRIX (Long-Only + Equity Filter)")
    print("=====================================================================================================================")
    print("Year   |   Jan |   Feb |   Mar |   Apr |   May |   Jun |   Jul |   Aug |   Sep |   Oct |   Nov |   Dec |      YTD")
    print("-" * 117)
    
    prev_eq = 100.0
    for y in sorted(monthly_eq.index.year.unique()):
        row = f"{y:<6} | "
        ytd_eq = 1.0
        
        for m in range(1, 13):
            val = monthly_eq[(monthly_eq.index.year == y) & (monthly_eq.index.month == m)]
            if not val.empty:
                current_eq = val.iloc[0]
                ret = (current_eq / prev_eq - 1.0) * 100
                prev_eq = current_eq
                
                ytd_eq *= (1.0 + ret/100.0)
                row += f"{ret:>5.1f}% | "
            else:
                row += "    -  | "
                
        ytd = (ytd_eq - 1.0) * 100
        row += f"{ytd:>7.1f}%"
        print(row)
        
    print("=====================================================================================================================")
    
if __name__ == '__main__':
    main()
