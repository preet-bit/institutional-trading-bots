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
    master_index = dfs['ES_daily'].set_index('t').index
    
    # Process DAX and Nikkei
    for name in ['DAX_daily', 'Nikkei_daily']:
        df = pd.read_csv(f'../data/{name}.csv', sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        dfs[name] = df.sort_values('t').drop_duplicates('t').set_index('t').reindex(master_index).ffill().bfill().reset_index()
        
    # Align Yield Data for macro filter
    dfs['Yield_daily'] = dfs['Yield_daily'].set_index('t').reindex(master_index).ffill().bfill().reset_index()
    
    return dfs

def main():
    dfs = load_global_data()
    genes = build_genes(dfs)
    name_to_genes = {g.name: g for g in genes}
    START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
    DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
    
    # EXACTLY 10 YEARS OF DATA (2016-01-01 to Present)
    d0_test = (pd.Timestamp('2016-01-01') - START).days
    d1_test = (END - START).days

    with open('dashboard_data_v3_ml.json', 'r') as f: names = [s['name'] for s in json.load(f)[:5]]
    all_trades = []

    # Yield MACRO filter
    yield_c = dfs['Yield_daily']['c']
    yield_ema = yield_c.ewm(span=20, adjust=False).mean()
    yields_rising = yield_c > yield_ema

    print("Running 10-Year Validation (2016 - 2026)...")
    for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
        df = dfs[asset]; x_asset = X(df)
        o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
        a14 = x_asset.atr(14).values.astype(np.float64)
        
        # 1. LONGS
        for name in names:
            chrom = [name_to_genes[n] for n in name.split(' AND ')]
            bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, 2.0, 2.0, 40)
            m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
            ex_days = DAY[bt[1]][m]; rets = bt[3][m] - 0.00015; risks = bt[4][m]
            for i in range(len(ex_days)): 
                all_trades.append({'ex_day': ex_days[i], 'R': rets[i]/(risks[i] or 1e-9), 'type': 'LONG'})
                
        # 2. SHORTS (Macro Yield Filtered)
        sma200 = df['c'].rolling(200).mean()
        sma50 = df['c'].rolling(50).mean()
        rsi2 = consistent_reversion.rsi(df['c'], 2)
        
        # Short the Bear Rally ONLY if Yields are rising
        short_sig = (df['c'] < sma200) & (sma50 < sma200) & (rsi2 > 90) & yields_rising
        
        sig = to_sig(pd.Series(False, index=df.index), edge(short_sig))
        sig[:200] = 0
        bt_short = run_bt(o, h, l, c, sig, a14, sl_m=1.0, rr=3.0, max_hold=10)
        ms = sel(dict(ent_day=DAY[bt_short[0]], ex_day=DAY[bt_short[1]]), d0_test, d1_test)
        ex_days_s = DAY[bt_short[1]][ms]; rets_s = bt_short[3][ms] - 0.00015; risks_s = bt_short[4][ms]
        for i in range(len(ex_days_s)): 
            all_trades.append({'ex_day': ex_days_s[i], 'R': rets_s[i]/(risks_s[i] or 1e-9), 'type': 'SHORT'})

    all_trades.sort(key=lambda x: x['ex_day'])
    
    eq = 100.0; peak = 100.0; drawdowns = []
    long_eq = 100.0; running_long_eqs = []
    dates = []; eq_curve = []

    for t in all_trades:
        if t['type'] == 'LONG':
            running_long_eqs.append(long_eq)
            if len(running_long_eqs) > 20:
                ma20 = np.mean(running_long_eqs[-20:])
                risk = 0.02 * 0.25 if long_eq < ma20 else 0.02
            else: risk = 0.02
            long_eq *= (1.0 + risk * t['R']); eq *= (1.0 + risk * t['R'])
        elif t['type'] == 'SHORT':
            risk = 0.025 
            eq *= (1.0 + risk * t['R'])
            
        if eq > peak: peak = eq
        drawdowns.append((peak - eq) / peak)
        dates.append(pd.Timestamp('2000-09-18') + pd.to_timedelta(t['ex_day'], unit='D'))
        eq_curve.append(eq)

    res_df = pd.DataFrame({'date': dates, 'eq': eq_curve})
    res_df['year'] = res_df.date.dt.year

    print('\n========================================================')
    print(' V9 HOLY GRAIL - FULL 10-YEAR VALIDATION (2016-2026) ')
    print('========================================================')
    print('Year | Trades | Compounded Return')
    print('-----|--------|------------------')
    for y in sorted(res_df.year.unique()):
        y_data = res_df[res_df.year == y]
        if y_data.empty: continue
        start_eq = eq_curve[y_data.index[0] - 1] if y_data.index[0] > 0 else 100.0
        y_ret = (y_data.iloc[-1]['eq'] / start_eq - 1) * 100
        print(f'{y} | {len(y_data):6d} | {y_ret:15.1f}%')

    years_tested = (res_df.date.max() - pd.Timestamp('2016-01-01')).days / 365.25
    cagr = (eq / 100.0) ** (1 / years_tested) - 1.0
    print('--------------------------------------------------------')
    print(f'Total Return: +{(eq/100-1)*100:,.1f}% | CAGR: {cagr*100:.1f}%')
    print(f'Max Drawdown: {max(drawdowns)*100:.1f}%')
    print('========================================================')

if __name__ == '__main__':
    main()
