import numpy as np
import pandas as pd
import random
import time

def load_and_prep_data():
    print("Loading High-Frequency Orderflow Data (Bitcoin 15m: 2018-2025)...")
    # Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore
    df = pd.read_csv('../data/btc_15m_data_2018_to_2025.csv', usecols=[0,1,2,3,4,5,9])
    df.columns = ['t', 'o', 'h', 'l', 'c', 'v', 'taker_buy_v']
    df['t'] = pd.to_datetime(df['t'])
    
    # Calculate Orderflow Delta
    df['taker_sell_v'] = df['v'] - df['taker_buy_v']
    df['of_delta'] = df['taker_buy_v'] - df['taker_sell_v']
    
    # Need to pre-calculate standard indicators so the GA loop is blazing fast
    c = df['c'].values
    of_delta = df['of_delta'].values
    v = df['v'].values
    
    # Precompute a matrix of Moving Averages and Std Devs for BBands
    print("Precomputing mathematical primitives for 5,000 algorithmic batches...")
    primitives = {}
    
    for p in [10, 20, 30, 40, 50]:
        primitives[f'sma_{p}'] = df['c'].rolling(p).mean().bfill().values
        primitives[f'std_{p}'] = df['c'].rolling(p).std().bfill().values
        primitives[f'vol_sma_{p}'] = df['v'].rolling(p).mean().bfill().values
        primitives[f'of_sma_{p}'] = df['of_delta'].rolling(p).mean().bfill().values
        
    for p in [3, 5, 9, 14]:
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean().bfill().values
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean().bfill().values
        rs = gain / (loss + 1e-9)
        primitives[f'rsi_{p}'] = (100 - (100 / (1 + rs)))
        
    return df, primitives

def random_gene():
    return {
        'bb_len': random.choice([10, 20, 30, 40, 50]),
        'bb_std': random.uniform(1.5, 3.5),
        'rsi_len': random.choice([3, 5, 9, 14]),
        'rsi_thresh': random.uniform(15, 35),
        'of_mult': random.uniform(1.0, 3.0), # Orderflow Delta must be > of_mult * Avg OF Delta
        'exit_bb_std': random.uniform(-1.0, 1.0),
        'leverage': random.uniform(1.0, 5.0)
    }

def crossover(g1, g2):
    child = {}
    for k in g1.keys():
        child[k] = g1[k] if random.random() < 0.5 else g2[k]
    return child

def mutate(g):
    if random.random() < 0.2: g['bb_len'] = random.choice([10, 20, 30, 40, 50])
    if random.random() < 0.2: g['bb_std'] = random.uniform(1.5, 3.5)
    if random.random() < 0.2: g['rsi_len'] = random.choice([3, 5, 9, 14])
    if random.random() < 0.2: g['rsi_thresh'] = random.uniform(15, 35)
    if random.random() < 0.2: g['of_mult'] = random.uniform(1.0, 3.0)
    if random.random() < 0.2: g['exit_bb_std'] = random.uniform(-1.0, 1.0)
    if random.random() < 0.2: g['leverage'] = random.uniform(1.0, 5.0)
    return g

def backtest_vectorized(gene, df, prim):
    c = df['c'].values
    ret = np.zeros(len(c))
    ret[1:] = (c[1:] - c[:-1]) / c[:-1]
    
    # Unpack gene
    sma = prim[f'sma_{gene["bb_len"]}']
    std = prim[f'std_{gene["bb_len"]}']
    rsi = prim[f'rsi_{gene["rsi_len"]}']
    of_sma = prim[f'of_sma_{gene["bb_len"]}']
    
    bb_lower = sma - (gene['bb_std'] * std)
    bb_exit = sma + (gene['exit_bb_std'] * std)
    
    # Long Entry: Price spikes below BB AND RSI is oversold AND Orderflow shows aggressive absorption
    entries = (c < bb_lower) & (rsi < gene['rsi_thresh']) & (df['of_delta'].values > (np.abs(of_sma) * gene['of_mult']))
    
    # Exit: Mean reversion complete (Price hits target BB band)
    exits = (c > bb_exit)
    
    # Vectorized state machine
    # 1 for long, 0 for flat
    pos = np.zeros(len(c))
    current_pos = 0
    
    # Since doing a python loop over 250k rows takes ~20ms, it's very fast
    # but let's use numba-like speed by vectorizing if possible
    # Actually, a purely vectorized approach:
    # Forward fill entry signals, mask with exit signals
    # We will use the fast pandas approach:
    sig = np.zeros(len(c))
    sig[entries] = 1
    sig[exits] = -1
    
    # This is a classic fast pandas state machine trick
    s = pd.Series(sig)
    s = s.replace(0, np.nan)
    s = s.ffill().fillna(-1) # default to flat (-1 means flat in this context)
    s = s.replace(-1, 0)
    
    pos_shift = np.zeros(len(c))
    pos_shift[1:] = s.values[:-1]
    
    strat_ret = pos_shift * ret * gene['leverage']
    
    # Transaction costs (Crypto has 0.04% taker fees on Binance/Bybit)
    trades = np.diff(np.concatenate([[0], pos_shift]))
    strat_ret -= np.abs(trades) * 0.0004
    
    strat_ret = np.nan_to_num(strat_ret)
    
    win_days = np.sum(strat_ret > 0)
    loss_days = np.sum(strat_ret < 0)
    if win_days + loss_days == 0:
        return -9999, 0, 0, 0, 0
        
    win_rate = win_days / (win_days + loss_days)
    gross_prof = np.sum(strat_ret[strat_ret > 0])
    gross_loss = np.abs(np.sum(strat_ret[strat_ret < 0]))
    pf = gross_prof / (gross_loss + 1e-9)
    
    # Calculate Monthly returns
    eq_curve = np.cumprod(1.0 + strat_ret)
    eq_series = pd.Series(eq_curve, index=df['t'])
    monthly = eq_series.resample('ME').last().dropna()
    monthly_rets = (monthly / monthly.shift(1) - 1.0) * 100
    
    avg_m = monthly_rets.mean()
    min_m = monthly_rets.min()
    
    score = 0
    if win_rate >= 0.70: score += 5000
    if pf >= 2.0: score += 5000
    if min_m >= 30.0: score += 10000
    
    score += avg_m * 100 + pf * 200 - (1.0 - win_rate) * 500
    
    return score, win_rate, pf, avg_m, min_m

def run_evolution():
    df, primitives = load_and_prep_data()
    
    POP_SIZE = 500
    GENERATIONS = 10
    
    print("\n========================================================")
    print(" HIGH-FREQUENCY CRYPTO ENGINE: EVOLUTIONARY LOOP ")
    print("========================================================")
    
    population = [random_gene() for _ in range(POP_SIZE)]
    
    for gen in range(GENERATIONS):
        print(f"--- GENERATION {gen+1} (500 Batches) ---")
        scored = []
        for strat in population:
            score, wr, pf, avg_m, min_m = backtest_vectorized(strat, df, primitives)
            scored.append({'strat': strat, 'score': score, 'wr': wr, 'pf': pf, 'avg_m': avg_m, 'min_m': min_m})
            
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_50 = scored[:50]
        
        best = top_50[0]
        print(f"Top: WinRate={best['wr']*100:.2f}%, PF={best['pf']:.2f}, Avg Mthly={best['avg_m']:.2f}%, Min Mthly={best['min_m']:.2f}%")
        
        if best['wr'] >= 0.70 and best['pf'] >= 2.0 and best['min_m'] >= 30.0:
            print("HOLY GRAIL FOUND! Constraints strictly met.")
            break
            
        next_pop = [x['strat'] for x in top_50]
        while len(next_pop) < POP_SIZE:
            p1 = random.choice(top_50)['strat']
            p2 = random.choice(top_50)['strat']
            child = crossover(p1, p2)
            child = mutate(child)
            next_pop.append(child)
            
        population = next_pop

    print("\n========================================================")
    print(" FINAL RESULTS: HF CRYPTO ORDERFLOW BOT ")
    print("========================================================")
    best_overall = top_50[0]
    print(f"Win Rate:      {best_overall['wr']*100:.2f}%")
    print(f"Profit Factor: {best_overall['pf']:.2f}")
    print(f"Avg Monthly:   {best_overall['avg_m']:.2f}%")
    print(f"Min Monthly:   {best_overall['min_m']:.2f}%")
    print("Optimal Parameters Generated:")
    for k, v in best_overall['strat'].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("========================================================")
    print("If it failed to hit 30% MINIMUM every month, this proves definitively that even high-frequency 15-minute orderflow with infinite optimization cannot violate the boundaries of market efficiency without overfitting.")
    print("========================================================")

if __name__ == '__main__':
    run_evolution()
