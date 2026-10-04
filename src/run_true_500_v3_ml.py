import os, time, sys, json, random
import numpy as np, pandas as pd
from scipy import stats

from core import run_bt, to_sig, edge
from eurusd_pipeline import trades_of, sel, metrics
from strategies import X, SL_GRID, RR_GRID

random.seed(42)

# -------------------------------------------------------------
# 1. LOAD AND ALIGN INTERMARKET DATA
# -------------------------------------------------------------
def load_all_data():
    print("Loading and aligning intermarket data...")
    dfs = {}
    for name in ['ES_daily', 'NQ_daily', 'Gold_daily', 'Oil_daily', 'Yield_daily']:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', f'{name}.csv'))
        df = pd.read_csv(csv_path, sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        df = df.sort_values('t').drop_duplicates('t')
        dfs[name] = df.set_index('t')
        
    # Align to ES dates
    master_index = dfs['ES_daily'].index
    
    aligned = {}
    for name, df in dfs.items():
        # Reindex and forward fill missing days (holidays differ slightly across markets)
        aligned[name] = df.reindex(master_index).ffill().bfill().reset_index()
        
    return aligned

# -------------------------------------------------------------
# 2. GENETIC ALGORITHM INSTRUCTION SET (GENES)
# -------------------------------------------------------------
class Gene:
    def __init__(self, name, eval_fn):
        self.name = name
        self.eval_fn = eval_fn

def build_genes(aligned_data):
    es = X(aligned_data['ES_daily'])
    gold = X(aligned_data['Gold_daily'])
    oil = X(aligned_data['Oil_daily'])
    yld = X(aligned_data['Yield_daily'])
    
    genes = [
        # Internal Equity Logic
        Gene("ES_MacroBull", lambda x: x.ema(50) > x.ema(200)),
        Gene("ES_MacroBear", lambda x: x.ema(50) < x.ema(200)),
        Gene("ES_VolSpike", lambda x: x.v > x.vsma(20)),
        Gene("ES_ADX_Trend", lambda x: x.adx(14)[2] > 20),
        Gene("ES_Oversold", lambda x: x.rsi(14) < 30),
        Gene("ES_Overbought", lambda x: x.rsi(14) > 70),
        Gene("ES_Stoch_Oversold", lambda x: x.df.c < x.df.l.rolling(14).min() + (x.df.h.rolling(14).max()-x.df.l.rolling(14).min())*0.2),
        Gene("ES_Donchian_BO", lambda x: x.c > x.h.rolling(20).max().shift()),
        
        # Intermarket Logic (Gold)
        Gene("Gold_Rising", lambda x: gold.c > gold.sma(50)),
        Gene("Gold_Falling", lambda x: gold.c < gold.sma(50)),
        Gene("Gold_Spike", lambda x: gold.c.pct_change(5) > 0.03), # 3% move in 5 days
        
        # Intermarket Logic (Oil)
        Gene("Oil_Rising", lambda x: oil.c > oil.sma(50)),
        Gene("Oil_Falling", lambda x: oil.c < oil.sma(50)),
        Gene("Oil_Crush", lambda x: oil.c.pct_change(10) < -0.05), # 5% drop in 10 days
        
        # Intermarket Logic (Yields)
        Gene("Yields_Rising", lambda x: yld.c > yld.sma(20)),
        Gene("Yields_Falling", lambda x: yld.c < yld.sma(20)),
        Gene("Yield_Curve_Flatten proxy", lambda x: yld.c.pct_change(20) > 0.10) 
    ]
    return genes

# -------------------------------------------------------------
# 3. GENETIC EVOLUTION ENGINE
# -------------------------------------------------------------
def random_chromosome(genes, length=3):
    return random.sample(genes, length)

def crossover(p1, p2):
    # Take half from p1, half from p2
    split = len(p1) // 2
    child1 = p1[:split] + [g for g in p2 if g not in p1[:split]][:(len(p1)-split)]
    child2 = p2[:split] + [g for g in p1 if g not in p2[:split]][:(len(p2)-split)]
    return child1, child2

def mutate(chrom, genes, prob=0.2):
    if random.random() < prob:
        idx = random.randint(0, len(chrom)-1)
        new_gene = random.choice([g for g in genes if g not in chrom])
        chrom[idx] = new_gene
    return chrom

def eval_chromosome(chrom, x_asset, o, h, l, c, a14, sl, rr, MAX_HOLD):
    # Build signal
    long_sig = pd.Series(True, index=x_asset.df.index)
    for g in chrom:
        long_sig = long_sig & g.eval_fn(x_asset)
    
    # We only take longs for this intermarket engine to focus on structural equity premium + filters
    sig = to_sig(edge(long_sig), pd.Series(False, index=x_asset.df.index))
    sig[:100] = 0
    
    return run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)

# -------------------------------------------------------------
# 4. MAIN PIPELINE
# -------------------------------------------------------------
def main():
    aligned_data = load_all_data()
    genes = build_genes(aligned_data)
    
    es_df = aligned_data['ES_daily']
    nq_df = aligned_data['NQ_daily']
    x_es = X(es_df)
    x_nq = X(nq_df)
    
    o_e, h_e, l_e, c_e = [es_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_e = x_es.atr(14).values.astype(np.float64)
    o_n, h_n, l_n, c_n = [nq_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_n = x_nq.atr(14).values.astype(np.float64)
    
    # Define IS (Train) and OOS (Test) boundary. Let's use 2003-2018 as Train, 2019-2026 as Test
    START = es_df.t.min()
    END = es_df.t.max()
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
    sl, rr = 2.0, 2.0 # Fixed for GA speed
    
    print(f"Starting Genetic Evolution: {POPULATION_SIZE} Pop x {GENERATIONS} Gens")
    population = [random_chromosome(genes, random.choice([2, 3, 4])) for _ in range(POPULATION_SIZE)]
    
    seen_chroms = {}
    
    t0 = time.time()
    for gen in range(GENERATIONS):
        scored = []
        for chrom in population:
            name = " AND ".join(sorted([g.name for g in chrom]))
            if name in seen_chroms:
                scored.append((seen_chroms[name], chrom))
                continue
                
            # Eval on ES Training Data
            bt_res = eval_chromosome(chrom, x_es, o_e, h_e, l_e, c_e, a14_e, sl, rr, MAX_HOLD)
            ei, xi, dr, g, rk = bt_res
            tr = dict(ei=ei, xi=xi, ent_day=DAY[ei], ex_day=DAY[xi], net=g - COST_RT, gross=g, risk=rk, dir=dr)
            m = sel(tr, d0_train, d1_train, purge=True)
            
            if m.sum() < 10:
                fitness = -999.0
            else:
                mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], d0_train, d1_train)
                # Fitness function: Reward Sharpe, penalize low trades
                fitness = mt['sharpe'] * np.sqrt(mt['trades'])
                
            seen_chroms[name] = fitness
            scored.append((fitness, chrom))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in scored[:POPULATION_SIZE//4]] # Top 25%
        
        # Log best of generation
        print(f"Gen {gen+1:02d} | Best Fitness: {scored[0][0]:.2f} | Best Logic: {' AND '.join([g.name for g in scored[0][1]])}")
        
        if gen == GENERATIONS - 1:
            population = [x[1] for x in scored] # Keep all sorted for final eval
            break
            
        # Breed next generation
        new_pop = list(elites)
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(elites, 2)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1, genes))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate(c2, genes))
        population = new_pop

    print(f"Evolution complete in {time.time()-t0:.1f}s")
    
    # -------------------------------------------------------------
    # 5. OOS EVALUATION OF TOP 500 (Historically Unique across generations)
    # -------------------------------------------------------------
    print("Evaluating all historically unique strategies on OOS Portfolio (ES + NQ)...")
    # Take the top 500 strategies explored during evolution
    sorted_all = sorted([(f, n) for n, f in seen_chroms.items()], reverse=True)
    top_500_names = [x[1] for x in sorted_all[:500]]
    
    # Map name back to chrom
    name_to_chrom = {}
    for chrom in population: # current pop
        name = " AND ".join(sorted([g.name for g in chrom]))
        name_to_chrom[name] = chrom
    # We might not have all 500 in current population. Just reconstruct them from names.
    name_to_genes = {g.name: g for g in genes}
    
    dashboard_data = []
    
    for idx, name in enumerate(top_500_names):
        chrom = [name_to_genes[n] for n in name.split(' AND ')]
        
        # Eval ES
        bt_es = eval_chromosome(chrom, x_es, o_e, h_e, l_e, c_e, a14_e, sl, rr, MAX_HOLD)
        tr_es = dict(ei=bt_es[0], xi=bt_es[1], ent_day=DAY[bt_es[0]], ex_day=DAY[bt_es[1]], net=bt_es[3]-COST_RT, risk=bt_es[4])
        m_es = sel(tr_es, d0_test, d1_test)
        
        # Eval NQ
        bt_nq = eval_chromosome(chrom, x_nq, o_n, h_n, l_n, c_n, a14_n, sl, rr, MAX_HOLD)
        tr_nq = dict(ei=bt_nq[0], xi=bt_nq[1], ent_day=DAY[bt_nq[0]], ex_day=DAY[bt_nq[1]], net=bt_nq[3]-COST_RT, risk=bt_nq[4])
        m_nq = sel(tr_nq, d0_test, d1_test)
        
        all_net = np.concatenate([tr_es['net'][m_es], tr_nq['net'][m_nq]])
        all_rk = np.concatenate([tr_es['risk'][m_es], tr_nq['risk'][m_nq]])
        all_exd = np.concatenate([tr_es['ex_day'][m_es], tr_nq['ex_day'][m_nq]])
        
        if len(all_net) > 0:
            idx_sort = np.argsort(all_exd)
            all_net = all_net[idx_sort]; all_rk = all_rk[idx_sort]; all_exd = all_exd[idx_sort]
            mt = metrics(all_net, all_rk, all_exd.astype(int), d0_test, d1_test)
            
            # Compounding at 2% risk because we want REAL returns
            R = all_net / np.where(all_rk == 0, 1e-9, all_rk)
            eq_20 = 100.0
            for rv in R: eq_20 *= (1.0 + 0.02 * rv)
            comp_20 = eq_20 - 100.0
        else:
            mt = metrics(np.array([]), np.array([]), np.array([]), d0_test, d1_test)
            comp_20 = 0.0
            
        dashboard_data.append({
            'id': f"GA-{idx+1:03d}",
            'name': name,
            'trades': mt['trades'],
            'win_rate': round(mt['win_rate']*100, 1) if not np.isnan(mt['win_rate']) else 0,
            'pf': round(mt.get('profit_factor', 0), 2) if mt.get('profit_factor', 0) != np.inf else 0,
            'sharpe': round(mt['sharpe'], 2),
            'compounded': round(comp_20, 1)
        })
        
    dashboard_data.sort(key=lambda x: x['sharpe'], reverse=True)
    with open('dashboard_data_v3_ml.json', 'w') as f:
        json.dump(dashboard_data, f)
        
    print("Evolution and Walk-Forward completed. Saved dashboard_data_v3_ml.json")

if __name__ == '__main__':
    main()
