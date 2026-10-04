import pandas as pd, numpy as np

def rsi(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def run_consistent_model():
    print("Testing the Connors RSI(2) Deep Reversion Model on S&P 500 (ES)...")
    df = pd.read_csv(r'c:\Users\preet\OneDrive\Documents\bots\data\ES_daily.csv', sep='\t', header=None, names=['t','o','h','l','c','v'])
    df['t'] = pd.to_datetime(df['t'])
    df = df.sort_values('t').reset_index(drop=True)

    # Indicators
    df['sma200'] = df['c'].rolling(200).mean()
    df['sma5'] = df['c'].rolling(5).mean()
    df['rsi2'] = rsi(df['c'], 2)
    
    # Entry Rule: Price > SMA200 AND RSI(2) < 10
    # Exit Rule: Price > SMA5
    
    pos = 0
    trade_rets = []
    years = []
    entry_price = 0
    
    # Using 3x Leverage (Typical for short-term index swing trading)
    LEVERAGE = 3.0
    
    # Track equity
    df['year'] = df['t'].dt.year
    df['net_ret'] = 0.0
    
    for i in range(200, len(df)):
        if pos == 0:
            if df.loc[i, 'c'] > df.loc[i, 'sma200'] and df.loc[i, 'rsi2'] < 10:
                pos = 1
                entry_price = df.loc[i, 'c'] # Enter at close
        elif pos == 1:
            ret = (df.loc[i, 'c'] / df.loc[i-1, 'c']) - 1.0
            df.loc[i, 'net_ret'] = ret * LEVERAGE
            if df.loc[i, 'c'] > df.loc[i, 'sma5']:
                pos = 0 # Exit at close

    print("\n==================================================")
    print(" DEEP REVERSION MODEL (S&P 500) @ 3x Leverage ")
    print("==================================================")
    print("Year | Trades | Win Rate | Max DD | Compounded Return")
    print("-----|--------|----------|--------|------------------")
    
    overall_eq = 100.0
    
    for y in sorted(df['year'].unique()):
        if y < 2001: continue
        ydf = df[df['year'] == y]
        
        trades = ydf[ydf['net_ret'] != 0]
        if len(trades) == 0:
            print(f"{y} |      0 |      N/A |   0.0% |              0.0%")
            continue
            
        wins = len(trades[trades['net_ret'] > 0])
        win_rate = (wins / len(trades)) * 100
        
        y_eq = 1.0
        y_peak = 1.0
        y_dds = []
        
        for r in ydf['net_ret']:
            y_eq *= (1.0 + r)
            if y_eq > y_peak: y_peak = y_eq
            y_dds.append((y_peak - y_eq) / y_peak)
            overall_eq *= (1.0 + r)
            
        y_ret = (y_eq - 1.0) * 100
        y_dd = max(y_dds) * 100
        
        print(f"{y} | {len(trades):6d} |    {win_rate:5.1f}% |  {y_dd:5.1f}% |           {y_ret:+6.1f}%")

    total_cagr = (overall_eq / 100.0) ** (1 / (len(df)/252)) - 1.0
    print("--------------------------------------------------")
    print(f"Total CAGR:          {total_cagr*100:.1f}% per year")
    print("--------------------------------------------------")

if __name__ == '__main__':
    run_consistent_model()
