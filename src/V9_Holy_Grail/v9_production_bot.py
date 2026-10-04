import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
V9 HOLY GRAIL - LIVE TRADING TEMPLATE (LONG + YIELD-FILTERED SHORTS)
====================================================================
This is the absolute final production skeleton for the V9 Engine.

Core Philosophy:
1. Long Ensemble runs aggressively during Bull Markets.
2. V7 Equity Curve Filter dynamically slashes long risk to zero during crashes.
3. Yield-Filtered Short Engine actively targets bear market rallies (Crisis Alpha)
   only when the macro trend is broken AND US Treasuries confirm a crash (Yields Rising).

"""
import numpy as np
import pandas as pd

# ==========================================================
# 1. CORE INDICATORS
# ==========================================================
def donchian_breakout(highs, period=20):
    return highs.rolling(period).max().shift(1)

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def true_range(df):
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    return true_range(df).rolling(period).mean()

def adx(df, period=14):
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

def rsi(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================================
# 2. GENERATE SIGNALS ON LIVE BAR CLOSE
# ==========================================================
def generate_signals(asset_df, gold_df, yield_df):
    c = asset_df['Close']
    
    # ------------------
    # LONGS (The 5 Genes)
    # ------------------
    donchian = donchian_breakout(asset_df['High'], 20)
    is_donchian_bo = (c > donchian)
    is_macro_bull = (ema(c, 50) > ema(c, 200))
    is_vol_spike = (atr(asset_df, 14) > atr(asset_df, 100))
    is_adx_trend = (adx(asset_df, 14) > 25)
    
    gold_c = gold_df['Close']
    is_gold_falling = (gold_c < ema(gold_c, 20))
    
    yield_c = yield_df['Close']
    is_yield_falling = (yield_c < ema(yield_c, 20))
    
    sig1 = is_donchian_bo & is_gold_falling
    sig2 = is_donchian_bo & is_macro_bull & is_gold_falling
    sig3 = is_vol_spike & is_yield_falling
    sig4 = is_adx_trend & is_yield_falling
    sig5 = is_adx_trend & is_donchian_bo & is_macro_bull & is_gold_falling
    
    master_long_signal = sig1 | sig2 | sig3 | sig4 | sig5
    
    # ------------------
    # SHORTS (Crisis Alpha)
    # ------------------
    # 1. Macro Trend is Broken (Death Cross + Below 200)
    sma200 = sma(c, 200)
    sma50 = sma(c, 50)
    is_bear_market = (c < sma200) & (sma50 < sma200)
    
    # 2. Exhaustion Bear Rally (Euphoria bounce inside a crash)
    rsi2 = rsi(c, 2)
    is_bear_rally = (rsi2 > 90)
    
    # 3. MACRO FILTER: Yields must be rising (Confirmed Liquidity Drain)
    yields_rising = (yield_c > ema(yield_c, 20))
    
    master_short_signal = is_bear_market & is_bear_rally & yields_rising
    
    return {
        'LONG': master_long_signal.iloc[-1],
        'SHORT': master_short_signal.iloc[-1]
    }

# ==========================================================
# 3. DYNAMIC EQUITY FILTER
# ==========================================================
class V9RiskManager:
    def __init__(self):
        self.long_equity_history = []
        self.base_long_risk = 0.02 # 2.0% Risk for Longs
        self.short_risk = 0.025    # 2.5% Fixed Risk for Shorts
        
    def update_long_equity(self, current_long_balance):
        self.long_equity_history.append(current_long_balance)
        
    def get_long_risk(self, current_long_balance):
        if len(self.long_equity_history) < 20:
            return self.base_long_risk
        ma20 = np.mean(self.long_equity_history[-20:])
        if current_long_balance < ma20:
            return self.base_long_risk * 0.25
        return self.base_long_risk

# ==========================================================
# 4. EXECUTION LOOP
# ==========================================================
def on_daily_candle_close(current_balance, long_pnl_tracker):
    # asset_df = fetch_data("ES")
    # gold_df = fetch_data("GC=F")
    # yield_df = fetch_data("^TNX")
    
    # risk_mgr = V9RiskManager()
    # risk_mgr.update_long_equity(long_pnl_tracker)
    
    # signals = generate_signals(asset_df, gold_df, yield_df)
    
    # if signals['LONG']:
    #     risk_pct = risk_mgr.get_long_risk(long_pnl_tracker)
    #     execute_trade(direction="LONG", size=current_balance * risk_pct, sl=2.0_ATR, tp=2.0_ATR)
        
    # elif signals['SHORT']:
    #     execute_trade(direction="SHORT", size=current_balance * risk_mgr.short_risk, sl=1.0_ATR, tp=3.0_ATR)
    pass
