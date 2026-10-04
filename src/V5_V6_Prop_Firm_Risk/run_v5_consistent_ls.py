import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, time, sys, json, random
import numpy as np, pandas as pd
from scipy import stats
from core import run_bt, to_sig, edge
from eurusd_pipeline import sel, metrics
from strategies import X

random.seed(101)

# -------------------------------------------------------------
# 1. LOAD INTERMARKET DATA
# -------------------------------------------------------------
def load_all_data():
    dfs = {}
    for name in ['ES_daily', 'NQ_daily', 'Gold_daily', 'Oil_daily', 'Yield_daily']:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        df = df.sort_values('t').drop_duplicates('t')
        dfs[name] = df.set_index('t')
        
    master_index = dfs['ES_daily'].index
    aligned = {}
    for name, df in dfs.items():
        aligned[name] = df.reindex(master_index).ffill().bfill().reset_index()
    return aligned

# -------------------------------------------------------------
# 2. LONG/SHORT GENES
# -------------------------------------------------------------
class GeneLS:
    def __init__(self, name, long_fn, short_fn):
        self.name = name
        self.long_fn = long_fn
        self.short_fn = short_fn

def stoch_k(df, n=14):
    lo_n = df.l.rolling(n).min()
    hi_n = df.h.rolling(n).max()
    denom = (hi_n - lo_n).replace(0, np.nan)
    return 100.0 * (df.c - lo_n) / denom

def build_ls_genes(aligned_data):
    es = X(aligned_data['ES_daily'])
    gold = X(aligned_data['Gold_daily'])
    oil = X(aligned_data['Oil_daily'])
    yld = X(aligned_data['Yield_daily'])
    
    return [
        # Internal Equity Logic
        GeneLS("MacroTrend", lambda x: x.ema(50) > x.ema(200), lambda x: x.ema(50) < x.ema(200)),
        GeneLS("VolSpike", lambda x: x.v > x.vsma(20), lambda x: x.v > x.vsma(20)),
        GeneLS("ADX_Trend", lambda x: x.adx(14)[2] > 20, lambda x: x.adx(14)[2] > 20),
        GeneLS("Pullback", lambda x: x.rsi(14) < 35, lambda x: x.rsi(14) > 65),
        GeneLS("Donchian_BO", lambda x: x.c > x.h.rolling(20).max().shift(), lambda x: x.c < x.l.rolling(20).min().shift()),
        
        # Intermarket Logic (Gold) - Inverse correlation to risk
        GeneLS("Gold_Filter", lambda x: gold.c < gold.sma(50), lambda x: gold.c > gold.sma(50)),
        GeneLS("Gold_Momentum", lambda x: gold.c.pct_change(10) < 0, lambda x: gold.c.pct_change(10) > 0),
        
        # Intermarket Logic (Oil) - Inflation proxy
        GeneLS("Oil_Filter", lambda x: oil.c > oil.sma(50), lambda x: oil.c < oil.sma(50)),
        
        # Intermarket Logic (Yields) - Rate proxy
        GeneLS("Yields_Filter", lambda x: yld.c < yld.sma(20), lambda x: yld.c > yld.sma(20)),
        GeneLS("Yields_Momentum", lambda x: yld.c.pct_change(10) < 0, lambda x: yld.c.pct_change(10) > 0)
    ]

# -------------------------------------------------------------
# 3. GENETIC EVOLUTION (LONG/SHORT)
# -------------------------------------------------------------
def eval_chromosome_ls(chrom, x_asset, o, h, l, c, a14, sl, rr, MAX_HOLD):
    long_sig = pd.Series(True, index=x_asset.df.index)
    short_sig = pd.Series(True, index=x_asset.df.index)
    
    for g in chrom:
        long_sig = long_sig & g.long_fn(x_asset)
        short_sig = short_sig & g.short_fn(x_asset)
    
    sig = to_sig(edge(long_sig), edge(short_sig))
    sig[:100] = 0
    return run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)

def mutate(chrom, genes, prob=0.2):
    if random.random() < prob:
        idx = random.randint(0, len(chrom)-1)
        chrom[idx] = random.choice([g for g in genes if g not in chrom])
    return chrom

def crossover(p1, p2):
    split = len(p1) // 2
    c1 = p1[:split] + [g for g in p2 if g not in p1[:split]][:(len(p1)-split)]
    c2 = p2[:split] + [g for g in p1 if g not in p2[:split]][:(len(p2)-split)]
    return c1, c2

def main():
    aligned_data = load_all_data()
    genes = build_ls_genes(aligned_data)
    
    es_df = aligned_data['ES_daily']; nq_df = aligned_data['NQ_daily']
    x_es = X(es_df); x_nq = X(nq_df)
    o_e, h_e, l_e, c_e = [es_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_e = x_es.atr(14).values.astype(np.float64)
    o_n, h_n, l_n, c_n = [nq_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_n = x_nq.atr(14).values.astype(np.float64)
    
    START = es_df.t.min(); END = es_df.t.max()
    T = pd.to_datetime(es_df.t.values)
    DAY = ((T.floor('D') - START).days).values
    
    d0_train = (pd.Timestamp('2003-01-01') - START).days
    d1_train = (pd.Timestamp('2018-12-31') - START).days
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    
    POPULATION_SIZE = 150
    GENERATIONS = 10
    COST_RT = 0.00015
    MAX_HOLD = 40
    sl, rr = 2.0, 2.0 
    
    print("Evolving Long/Short Strategies for Absolute Consistency...")
    population = [random.sample(genes, random.choice([2, 3, 4])) for _ in range(POPULATION_SIZE)]
    seen_chroms = {}
    
    for gen in range(GENERATIONS):
        scored = []
        for chrom in population:
            name = " AND ".join(sorted([g.name for g in chrom]))
            if name in seen_chroms:
                scored.append((seen_chroms[name], chrom)); continue
                
            bt_es = eval_chromosome_ls(chrom, x_es, o_e, h_e, l_e, c_e, a14_e, sl, rr, MAX_HOLD)
            ei, xi, dr, g, rk = bt_es
            tr = dict(ent_day=DAY[ei], ex_day=DAY[xi], net=g - COST_RT, risk=rk)
            m = sel(tr, d0_train, d1_train, purge=True)
            
            if m.sum() < 20: # Require more trades for LS consistency
                fitness = -999.0
            else:
                mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], d0_train, d1_train)
                # Fitness: Optimize strictly for Sortino/Sharpe rather than absolute return to force consistency
                fitness = mt['sharpe'] * np.sqrt(mt['trades'])
                
            seen_chroms[name] = fitness
            scored.append((fitness, chrom))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in scored[:POPULATION_SIZE//4]]
        
        if gen == GENERATIONS - 1:
            population = [x[1] for x in scored]
            break
            
        new_pop = list(elites)
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1, genes))
            if len(new_pop) < POPULATION_SIZE: new_pop.append(mutate(c2, genes))
        population = new_pop

    print("Evolution complete. Re-evaluating OOS for Consistency Ensemble...")
    
    sorted_all = sorted([(f, n) for n, f in seen_chroms.items()], reverse=True)
    top_names = [x[1] for x in sorted_all[:5]] # Take Top 5 for the new ensemble
    name_to_genes = {g.name: g for g in genes}
    
    all_net, all_rk, all_exd = [], [], []
    for name in top_names:
        chrom = [name_to_genes[n] for n in name.split(' AND ')]
        
        bt_es = eval_chromosome_ls(chrom, x_es, o_e, h_e, l_e, c_e, a14_e, sl, rr, MAX_HOLD)
        m_es = sel(dict(ent_day=DAY[bt_es[0]], ex_day=DAY[bt_es[1]]), d0_test, d1_test)
        
        bt_nq = eval_chromosome_ls(chrom, x_nq, o_n, h_n, l_n, c_n, a14_n, sl, rr, MAX_HOLD)
        m_nq = sel(dict(ent_day=DAY[bt_nq[0]], ex_day=DAY[bt_nq[1]]), d0_test, d1_test)
        
        all_net.extend(bt_es[3][m_es]-COST_RT); all_rk.extend(bt_es[4][m_es]); all_exd.extend(DAY[bt_es[1]][m_es])
        all_net.extend(bt_nq[3][m_nq]-COST_RT); all_rk.extend(bt_nq[4][m_nq]); all_exd.extend(DAY[bt_nq[1]][m_nq])

    net_arr, rk_arr, exd_arr = np.array(all_net), np.array(all_rk), np.array(all_exd)
    idx_sort = np.argsort(exd_arr)
    net_arr, rk_arr, exd_arr = net_arr[idx_sort], rk_arr[idx_sort], exd_arr[idx_sort]
    
    R = net_arr / np.where(rk_arr == 0, 1e-9, rk_arr)
    dates = pd.Timestamp('2000-09-18') + pd.to_timedelta(exd_arr, unit='D')
    df = pd.DataFrame({'date': dates, 'R': R})
    df['year'] = df.date.dt.year

    print('\n======================================================')
    print('LONG/SHORT CONSISTENT ENSEMBLE (1.5% Risk Allocation)')
    print('======================================================')
    print('Year | Trades | Compounded Return')
    print('-----|--------|------------------')
    
    eq = 100.0
    for y in sorted(df.year.unique()):
        y_trades = df[df.year == y]
        start_eq = eq
        for r in y_trades.R: eq *= (1.0 + 0.015 * r)
        y_ret = (eq / start_eq - 1) * 100
        print(f'{y} | {len(y_trades):6d} | {y_ret:15.1f}%')

    eq = 100.0
    peak = 100.0
    drawdowns = []
    for r in df.R:
        eq *= (1.0 + 0.015 * r)
        if eq > peak: peak = eq
        drawdowns.append((peak - eq) / peak)
    print(f'\nMax Drawdown (1.5% Risk): {max(drawdowns)*100:.1f}%')

if __name__ == '__main__':
    main()
