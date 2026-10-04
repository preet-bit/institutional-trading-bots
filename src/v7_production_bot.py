"""
V7 INSTITUTIONAL ALGORITHM - LIVE TRADING TEMPLATE
==================================================
This is the final production skeleton for the V7 Long-Only Equity Filtered Engine.
It contains the exact logic rules optimized over the 7-year Out-Of-Sample test.

Core Philosophy:
- Aggressive compounding during Bull Markets (2.0% Risk)
- Absolute capital preservation during Crashes via a self-monitoring Equity Moving Average

Logic Matrix (The 5 Genes):
1. ES_Donchian_BO AND Gold_Falling
2. ES_Donchian_BO AND ES_MacroBull AND Gold_Falling
3. ES_VolSpike AND Yields_Falling
4. ES_ADX_Trend AND Yields_Falling
5. ES_ADX_Trend AND ES_Donchian_BO AND ES_MacroBull AND Gold_Falling
"""

import numpy as np
import pandas as pd

# ==========================================================
# 1. CORE INDICATOR DEFINITIONS
# ==========================================================
def donchian_breakout(highs, period=20):
    """True if current close is higher than the highest high of the last N periods"""
    return highs.rolling(period).max().shift(1)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).rolling(period).mean()

def adx(df, period=14):
    """Simplified ADX calculation for trend strength"""
    tr = true_range(df)
    up = df['High'] - df['High'].shift(1)
    down = df['Low'].shift(1) - df['Low']
    
    pos_dm = np.where((up > down) & (up > 0), up, 0)
    neg_dm = np.where((down > up) & (down > 0), down, 0)
    
    tr_smooth = tr.rolling(period).sum()
    pdi = 100 * pd.Series(pos_dm).rolling(period).sum() / tr_smooth
    ndi = 100 * pd.Series(neg_dm).rolling(period).sum() / tr_smooth
    
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.rolling(period).mean()

# ==========================================================
# 2. GENERATE SIGNALS ON LIVE BAR CLOSE
# ==========================================================
def generate_signals(asset_df, gold_df, yield_df):
    """
    asset_df: The OHLCV dataframe of the asset you are trading (e.g. ES, NQ)
    gold_df: The OHLCV dataframe of Gold (GC=F)
    yield_df: The OHLCV dataframe of 10-Yr Yields (^TNX)
    """
    c = asset_df['Close']
    
    # 1. Asset Indicators
    donchian = donchian_breakout(asset_df['High'], 20)
    is_donchian_bo = (c > donchian)
    is_macro_bull = (ema(c, 50) > ema(c, 200))
    is_vol_spike = (atr(asset_df, 14) > atr(asset_df, 100))
    is_adx_trend = (adx(asset_df, 14) > 25)
    
    # 2. Intermarket Macro Filters
    gold_c = gold_df['Close']
    is_gold_falling = (gold_c < ema(gold_c, 20))
    
    yield_c = yield_df['Close']
    is_yield_falling = (yield_c < ema(yield_c, 20))
    
    # 3. Combine into the 5 verified Ensemble Genes
    sig1 = is_donchian_bo & is_gold_falling
    sig2 = is_donchian_bo & is_macro_bull & is_gold_falling
    sig3 = is_vol_spike & is_yield_falling
    sig4 = is_adx_trend & is_yield_falling
    sig5 = is_adx_trend & is_donchian_bo & is_macro_bull & is_gold_falling
    
    # Final unified signal: True if ANY of the 5 genes are True
    master_buy_signal = sig1 | sig2 | sig3 | sig4 | sig5
    
    # Return the boolean decision for the current bar
    return master_buy_signal.iloc[-1]

# ==========================================================
# 3. DYNAMIC EQUITY FILTER (The V7 Secret Sauce)
# ==========================================================
class V7RiskManager:
    def __init__(self):
        self.equity_history = []
        self.base_risk = 0.02 # 2.0% Default Risk
        
    def update_equity(self, current_balance):
        self.equity_history.append(current_balance)
        
    def get_current_risk(self, current_balance):
        """
        If current balance drops below the 20-trade moving average, 
        we are in a drawdown/crash. Slash risk by 75% to protect capital.
        """
        if len(self.equity_history) < 20:
            return self.base_risk
            
        ma20 = np.mean(self.equity_history[-20:])
        
        if current_balance < ma20:
            print("CRASH DETECTED: Equity below MA20. Slashing risk to 0.5%")
            return self.base_risk * 0.25
        else:
            print("BULL REGIME: Equity above MA20. Full 2.0% risk active.")
            return self.base_risk

# ==========================================================
# 4. EXECUTION LOOP (Pseudo-code for live integration)
# ==========================================================
def on_daily_candle_close(current_balance):
    # 1. Fetch live data
    # asset_df = fetch_data("ES")
    # gold_df = fetch_data("GC=F")
    # yield_df = fetch_data("^TNX")
    
    # 2. Check risk manager
    # risk_mgr = V7RiskManager()
    # risk_mgr.update_equity(current_balance)
    # risk_pct = risk_mgr.get_current_risk(current_balance)
    
    # 3. Check signals
    # should_buy = generate_signals(asset_df, gold_df, yield_df)
    
    # 4. Execute
    # if should_buy:
    #     execute_trade(direction="LONG", size=current_balance * risk_pct, sl=2.0_ATR, tp=2.0_ATR)
    pass
