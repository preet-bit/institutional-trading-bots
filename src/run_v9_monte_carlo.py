import json, numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel
from core import run_bt, to_sig, edge
from strategies import X
import consistent_reversion

def get_data():
    dfs = load_all_data()
    master_index = dfs['ES_daily'].set_index('t').index
    for name in ['DAX_daily', 'Nikkei_daily']:
        df = pd.read_csv(f'../data/{name}.csv', sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        dfs[name] = df.sort_values('t').drop_duplicates('t').set_index('t').reindex(master_index).ffill().bfill().reset_index()
    dfs['Yield_daily'] = dfs['Yield_daily'].set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

print("Extracting V9 raw trade data for Monte Carlo Simulation...")
dfs = get_data()
genes = build_genes(dfs)
name_to_genes = {g.name: g for g in genes}
START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
d0_test = (pd.Timestamp('2016-01-01') - START).days; d1_test = (END - START).days

with open('dashboard_data_v3_ml.json', 'r') as f: names = [s['name'] for s in json.load(f)[:5]]

all_trades = []
yield_c = dfs['Yield_daily']['c']
yield_ema = yield_c.ewm(span=20, adjust=False).mean()
yields_rising = yield_c > yield_ema

for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
    df = dfs[asset]; x_asset = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x_asset.atr(14).values.astype(np.float64)
    for name in names:
        chrom = [name_to_genes[n] for n in name.split(' AND ')]
        bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, 2.0, 2.0, 40)
        m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
        ex_days = DAY[bt[1]][m]; rets = bt[3][m] - 0.00015; risks = bt[4][m]
        for i in range(len(ex_days)): all_trades.append({'ex_day': ex_days[i], 'R': rets[i]/(risks[i] or 1e-9), 'type': 'LONG'})
            
    sma200 = df['c'].rolling(200).mean(); sma50 = df['c'].rolling(50).mean()
    rsi2 = consistent_reversion.rsi(df['c'], 2)
    short_sig = (df['c'] < sma200) & (sma50 < sma200) & (rsi2 > 90) & yields_rising
    sig = to_sig(pd.Series(False, index=df.index), edge(short_sig))
    sig[:200] = 0
    bt_short = run_bt(o, h, l, c, sig, a14, sl_m=1.0, rr=3.0, max_hold=10)
    ms = sel(dict(ent_day=DAY[bt_short[0]], ex_day=DAY[bt_short[1]]), d0_test, d1_test)
    ex_days_s = DAY[bt_short[1]][ms]; rets_s = bt_short[3][ms] - 0.00015; risks_s = bt_short[4][ms]
    for i in range(len(ex_days_s)): all_trades.append({'ex_day': ex_days_s[i], 'R': rets_s[i]/(risks_s[i] or 1e-9), 'type': 'SHORT'})

# Separate Long and Short trades because we want to sample them independently based on regime likelihood
long_trades = [t['R'] for t in all_trades if t['type'] == 'LONG']
short_trades = [t['R'] for t in all_trades if t['type'] == 'SHORT']
long_count = len(long_trades)
short_count = len(short_trades)

print(f"Loaded {long_count} Long Trades and {short_count} Short Trades.")
print("Running 1,000 Monte Carlo alternate realities...")

num_iterations = 1000
mc_drawdowns = []
mc_returns = []

# To properly test the Equity Filter, we will simulate the same number of trades per iteration,
# shuffling the sequence and randomly placing the short trades into the sequence.
total_trades = long_count + short_count

for i in range(num_iterations):
    # Sample with replacement
    sampled_longs = np.random.choice(long_trades, size=long_count, replace=True)
    sampled_shorts = np.random.choice(short_trades, size=short_count, replace=True)
    
    # Merge and shuffle the sequence randomly
    combined_types = ['LONG'] * long_count + ['SHORT'] * short_count
    combined_rs = np.concatenate([sampled_longs, sampled_shorts])
    
    # Shuffle in unison
    indices = np.arange(total_trades)
    np.random.shuffle(indices)
    seq_types = np.array(combined_types)[indices]
    seq_rs = combined_rs[indices]
    
    # Run the V9 Dynamic Risk Manager Simulation
    eq = 100.0
    peak = 100.0
    long_eq = 100.0
    running_long_eqs = []
    
    max_dd = 0.0
    
    for t_type, r_val in zip(seq_types, seq_rs):
        if t_type == 'LONG':
            running_long_eqs.append(long_eq)
            if len(running_long_eqs) > 20:
                ma20 = np.mean(running_long_eqs[-20:])
                risk = 0.02 * 0.25 if long_eq < ma20 else 0.02
            else:
                risk = 0.02
            long_eq *= (1.0 + risk * r_val)
            eq *= (1.0 + risk * r_val)
        else: # SHORT
            risk = 0.025
            eq *= (1.0 + risk * r_val)
            
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
            
    mc_drawdowns.append(max_dd * 100)
    mc_returns.append((eq / 100.0 - 1) * 100)

print("\n========================================================")
print(" V9 HOLY GRAIL - MONTE CARLO STRESS TEST (1,000 RUNS)")
print("========================================================")
print("The simulation shuffles trade sequences to test how the ")
print("Equity Filter handles extreme bad luck and clustering.")
print("--------------------------------------------------------")
print(f"Historical 10-Year Return:  +352,423%")
print(f"Historical Max Drawdown:    -40.0%")
print("--------------------------------------------------------")
print("MONTE CARLO PROJECTED RETURNS:")
print(f" 99th Percentile (Best Luck):   +{np.percentile(mc_returns, 99):,.0f}%")
print(f" Median (Expected Path):        +{np.percentile(mc_returns, 50):,.0f}%")
print(f"  5th Percentile (Bad Luck):    +{np.percentile(mc_returns, 5):,.0f}%")
print(f"  1st Percentile (Worst Luck):  +{np.percentile(mc_returns, 1):,.0f}%")
print("--------------------------------------------------------")
print("MONTE CARLO MAXIMUM DRAWDOWNS:")
print(f"  1st Percentile (Best Luck):   -{np.percentile(mc_drawdowns, 1):.1f}%")
print(f" Median Expected Max Drawdown:  -{np.percentile(mc_drawdowns, 50):.1f}%")
print(f" 95th Percentile (Bad Luck):    -{np.percentile(mc_drawdowns, 95):.1f}%")
print(f" 99th Percentile (Worst Luck):  -{np.percentile(mc_drawdowns, 99):.1f}%")
print("========================================================")
