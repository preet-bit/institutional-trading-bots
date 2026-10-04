import pandas as pd
import numpy as np
from itertools import product

# DOUBLE EFFORT PROTOCOL: STATISTICAL ARBITRAGE (MEAN REVERSION)
# Strategy: Z-Score divergence between highly correlated assets (ES and NQ)
# Why? Mean Reversion strategies are the ONLY way to mathematically achieve >70% Win Rates.

def run_500_batches():
    print("Loading Data for Statistical Arbitrage (ES vs NQ)...")
    es = pd.read_csv('../data/ES_daily.csv', sep='\t', header=None, names=['t','o','h','l','c','v'])
    nq = pd.read_csv('../data/NQ_daily.csv', sep='\t', header=None, names=['t','o','h','l','c','v'])
    
    es['t'] = pd.to_datetime(es['t'])
    nq['t'] = pd.to_datetime(nq['t'])
    
    es = es[es['t'] >= '2016-01-01'].set_index('t')['c']
    nq = nq[nq['t'] >= '2016-01-01'].set_index('t')['c']
    
    df = pd.concat([es, nq], axis=1).dropna()
    df.columns = ['ES', 'NQ']
    
    # Calculate returns
    df['ES_ret'] = df['ES'].pct_change()
    df['NQ_ret'] = df['NQ'].pct_change()
    
    # Parameter grid (Total Combinations ~ 500-1000)
    lookbacks = [10, 20, 30, 40, 50, 60]
    entry_z_scores = [1.5, 2.0, 2.5, 3.0]
    exit_z_scores = [0.0, 0.5, 1.0]
    stop_loss_z = [4.0, 5.0, 6.0]
    leverage_mults = [1.0, 3.0, 5.0, 10.0]
    
    best_score = -9999
    best_params = None
    best_metrics = None
    
    combinations = list(product(lookbacks, entry_z_scores, exit_z_scores, stop_loss_z, leverage_mults))
    
    print(f"Testing {len(combinations)} algorithmic batches...")
    
    for i, (lb, en_z, ex_z, sl_z, lev) in enumerate(combinations):
        # Calculate Rolling Hedge Ratio (Beta)
        rolling_cov = df['NQ_ret'].rolling(lb).cov(df['ES_ret'])
        rolling_var = df['ES_ret'].rolling(lb).var()
        hedge_ratio = (rolling_cov / rolling_var).fillna(1.0)
        
        # Calculate Spread
        spread = df['NQ'] - hedge_ratio * df['ES']
        
        # Z-Score of Spread
        spread_mean = spread.rolling(lb).mean()
        spread_std = spread.rolling(lb).std()
        z_score = ((spread - spread_mean) / spread_std).fillna(0)
        
        # Generate Signals
        long_spread = (z_score < -en_z)
        short_spread = (z_score > en_z)
        
        exit_long = (z_score > -ex_z) | (z_score < -sl_z) # Exit or stop loss
        exit_short = (z_score < ex_z) | (z_score > sl_z)
        
        pos = np.zeros(len(df))
        current_pos = 0
        
        long_spread_arr = long_spread.values
        short_spread_arr = short_spread.values
        exit_long_arr = exit_long.values
        exit_short_arr = exit_short.values
        
        for j in range(len(df)):
            if current_pos == 0:
                if long_spread_arr[j]: current_pos = 1
                elif short_spread_arr[j]: current_pos = -1
            elif current_pos == 1 and exit_long_arr[j]:
                current_pos = 0
            elif current_pos == -1 and exit_short_arr[j]:
                current_pos = 0
            pos[j] = current_pos
            
        pos_shift = np.zeros(len(df))
        pos_shift[1:] = pos[:-1]
        
        spread_ret = df['NQ_ret'].fillna(0).values - (hedge_ratio.fillna(1.0).values * df['ES_ret'].fillna(0).values)
        strat_ret = pos_shift * spread_ret * lev
        
        # Deduct costs
        trades = np.diff(np.concatenate([[0], pos_shift]))
        strat_ret -= np.abs(trades) * 0.0003 # Slippage
        strat_ret = np.nan_to_num(strat_ret)
        
        # Metrics
        win_days = np.sum(strat_ret > 0)
        loss_days = np.sum(strat_ret < 0)
        if win_days + loss_days == 0: continue
        
        win_rate = win_days / (win_days + loss_days)
        gross_prof = np.sum(strat_ret[strat_ret > 0])
        gross_loss = np.abs(np.sum(strat_ret[strat_ret < 0]))
        pf = gross_prof / (gross_loss + 1e-9)
        
        eq_curve = np.cumprod(1.0 + strat_ret)
        monthly_eq = pd.Series(eq_curve, index=df.index).resample('ME').last().dropna()
        monthly_rets = (monthly_eq / monthly_eq.shift(1) - 1.0) * 100
        avg_m = monthly_rets.mean()
        min_m = monthly_rets.min()
        
        score = 0
        if win_rate >= 0.70: score += 1000
        if pf >= 2.0: score += 1000
        if min_m >= 30.0: score += 5000
        score += avg_m * 10 + pf * 50 - (1.0 - win_rate)*100
        
        if score > best_score:
            best_score = score
            best_params = (lb, en_z, ex_z, sl_z, lev)
            best_metrics = (win_rate, pf, avg_m, min_m)
            
    print("\n========================================================")
    print(" DOUBLE EFFORT COMPLETE - STATISTICAL ARBITRAGE ")
    print("========================================================")
    print(f"Best Parameters: Lookback={best_params[0]}, EntryZ={best_params[1]}, ExitZ={best_params[2]}, StopZ={best_params[3]}, Leverage={best_params[4]}x")
    print(f"Win Rate:      {best_metrics[0]*100:.2f}%")
    print(f"Profit Factor: {best_metrics[1]:.2f}")
    print(f"Avg Monthly:   {best_metrics[2]:.2f}%")
    print(f"Min Monthly:   {best_metrics[3]:.2f}%")
    print("--------------------------------------------------------")
    if best_metrics[0] < 0.70 or best_metrics[1] < 2.0 or best_metrics[3] < 30.0:
        print("MANDATE FAILED: The constraints (>70% WR, >2.0 PF, >30% EVERY month) are mathematically impossible over a 10-year period without curve-fitting to the point of catastrophic ruin.")
    else:
        print("HOLY GRAIL PAIRS BOT FOUND.")
    print("========================================================")

if __name__ == '__main__':
    run_500_batches()
