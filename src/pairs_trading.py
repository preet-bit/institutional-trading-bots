import os, numpy as np, pandas as pd

def pairs_backtest():
    print("Loading ES and NQ data for Statistical Arbitrage...")
    df_es = pd.read_csv(r'c:\Users\preet\OneDrive\Documents\bots\data\ES_daily.csv', sep='\t', header=None, names=['t','o','h','l','c','v'])
    df_nq = pd.read_csv(r'c:\Users\preet\OneDrive\Documents\bots\data\NQ_daily.csv', sep='\t', header=None, names=['t','o','h','l','c','v'])
    df_es['t'] = pd.to_datetime(df_es['t'])
    df_nq['t'] = pd.to_datetime(df_nq['t'])

    # Merge on date to ensure perfect alignment
    df = pd.merge(df_es[['t', 'c']], df_nq[['t', 'c']], on='t', suffixes=('_es', '_nq'))
    df = df.sort_values('t').reset_index(drop=True)

    print("Calculating Rolling Cointegration Spread...")
    # Daily Returns
    df['ret_es'] = df['c_es'].pct_change()
    df['ret_nq'] = df['c_nq'].pct_change()

    # 1. Rolling Volatility (20-day) to equalize risk
    df['vol_es'] = df['ret_es'].rolling(20).std()
    df['vol_nq'] = df['ret_nq'].rolling(20).std()

    # 2. Cumulative Log Prices
    log_es = np.log(df['c_es'])
    log_nq = np.log(df['c_nq'])

    # 3. Rolling Hedge Ratio (Beta) using 60-day covariance
    cov = df['ret_nq'].rolling(60).cov(df['ret_es'])
    var_es = df['ret_es'].rolling(60).var()
    df['beta'] = cov / var_es

    # 4. Spread calculation
    # Spread = NQ - Beta * ES
    df['spread'] = log_nq - df['beta'] * log_es

    # 5. Z-Score of the Spread (Mean Reversion Trigger)
    spread_mean = df['spread'].rolling(20).mean()
    spread_std = df['spread'].rolling(20).std()
    df['zscore'] = (df['spread'] - spread_mean) / spread_std

    print("Generating Market-Neutral Signals...")
    # Rules:
    # If Z > 2.0: NQ is overvalued relative to ES -> Short NQ, Long ES
    # If Z < -2.0: NQ is undervalued relative to ES -> Long NQ, Short ES
    # Exit when Z crosses 0 (reverts to mean)
    
    # Track Positions
    pos = np.zeros(len(df))
    current_pos = 0 # 1 means Long Spread (Long NQ/Short ES), -1 means Short Spread
    
    z = df['zscore'].values
    
    for i in range(1, len(df)):
        if np.isnan(z[i]):
            continue
            
        # Exit condition: Mean reversion crossed 0
        if current_pos == 1 and z[i] >= 0:
            current_pos = 0
        elif current_pos == -1 and z[i] <= 0:
            current_pos = 0
            
        # Entry conditions
        if current_pos == 0:
            if z[i] < -2.0:
                current_pos = 1
            elif z[i] > 2.0:
                current_pos = -1
                
        pos[i] = current_pos

    # Shift position by 1 day to prevent lookahead bias (execute on next day's close)
    df['pos'] = pd.Series(pos).shift(1).fillna(0)

    # 6. Calculate Market-Neutral Returns
    # Volatility-weighted sizing (Risk Parity)
    total_vol = df['vol_es'] + df['vol_nq']
    w_nq = df['vol_es'] / total_vol
    w_es = df['vol_nq'] / total_vol

    # If pos == 1 (Long Spread): Long NQ, Short ES
    # If pos == -1 (Short Spread): Short NQ, Long ES
    # Applying 5x leverage since Market Neutral Pairs are inherently very low volatility
    LEVERAGE = 5.0
    COST_BPS = 0.0001 # 1 basis point per trade leg
    
    # Daily PnL
    df['trade_ret'] = df['pos'] * LEVERAGE * (w_nq * df['ret_nq'] - w_es * df['ret_es'])
    
    # Subtract Costs on position changes
    pos_change = df['pos'].diff().abs().fillna(0)
    df['cost'] = pos_change * LEVERAGE * COST_BPS
    df['net_ret'] = df['trade_ret'] - df['cost']

    # 7. Year-by-Year Consistency Breakdown
    df['year'] = df['t'].dt.year
    df = df.dropna()

    print("\n==================================================")
    print(" MARKET-NEUTRAL PAIRS TRADING (ES vs NQ) @ 5x Lev ")
    print("==================================================")
    print("Year | Market Return (SPY) | StatArb Return | Max DD")
    print("-----|---------------------|----------------|-------")
    
    eq_curve = []
    running_eq = 100.0
    
    overall_drawdowns = []
    overall_peak = 100.0
    
    for y in sorted(df['year'].unique()):
        ydf = df[df['year'] == y]
        if len(ydf) == 0: continue
        
        # StatArb Return
        y_eq = 1.0
        y_peak = 1.0
        y_dds = []
        for r in ydf['net_ret']: 
            y_eq *= (1.0 + r)
            if y_eq > y_peak: y_peak = y_eq
            y_dds.append((y_peak - y_eq) / y_peak)
            
            running_eq *= (1.0 + r)
            if running_eq > overall_peak: overall_peak = running_eq
            overall_drawdowns.append((overall_peak - running_eq) / overall_peak)
            
        y_ret = (y_eq - 1.0) * 100
        y_dd = max(y_dds) * 100 if y_dds else 0.0
        
        # Benchmark Return (Buy and Hold ES)
        es_eq = 1.0
        for r in ydf['ret_es']: es_eq *= (1.0 + r)
        bm_ret = (es_eq - 1.0) * 100
        
        # Formatting color
        print(f"{y} | {bm_ret:+18.1f}% | {y_ret:+13.1f}% | {y_dd:5.1f}%")

    total_cagr = (running_eq / 100.0) ** (1 / (len(df)/252)) - 1.0
    max_dd = max(overall_drawdowns) * 100 if overall_drawdowns else 0.0
    
    print("--------------------------------------------------")
    print(f"Total CAGR:          {total_cagr*100:.1f}% per year")
    print(f"Overall Max DD:      {max_dd:.1f}%")
    print("--------------------------------------------------")
    print("Note: The algorithm is Market-Neutral. It makes money on the *spread* reverting, oblivious to market crashes.")

if __name__ == '__main__':
    pairs_backtest()
