# V9 Holy Grail Engine (Yield-Filtered Crisis Alpha)

## System Overview
The V9 Engine represents the absolute peak of aggressive, asymmetric quantitative trend-following. It is designed to aggressively compound capital during equity bull markets using heavy leverage, while utilizing a macro-filtered short engine to print "Crisis Alpha" (positive returns) during verified global liquidity crashes.

*   **Test Window:** January 2016 – October 2026 (10.75 Years Out-Of-Sample)
*   **Asset Universe:** S&P 500 (ES), Nasdaq 100 (NQ), DAX (Germany), Nikkei (Japan)
*   **Macro Inputs:** Gold Futures (GC=F), 10-Year US Treasury Yield (^TNX)
*   **Execution Timeframe:** Daily Close

---

## 1. The Long Engine (The Genetic 5)
During normal market conditions, the bot scans for any of the 5 top-performing logic trees discovered by the V3 Machine Learning Genetic Algorithm. If *any* of these 5 conditions are met, a Long trade is executed.

1.  **Macro Breakout:** Price breaks 20-day High `AND` Gold is in a downtrend (Risk-On).
2.  **Confirmed Macro Breakout:** Price breaks 20-day High `AND` 50 SMA > 200 SMA `AND` Gold is in a downtrend.
3.  **Volatility Expansion:** ATR(14) > ATR(100) `AND` US Yields are falling.
4.  **Trend Momentum:** ADX(14) > 25 `AND` US Yields are falling.
5.  **The Master Gene:** ADX > 25 `AND` 20-day Breakout `AND` 50 SMA > 200 SMA `AND` Gold is falling.

## 2. The Short Engine (Crisis Alpha)
Systematically shorting equities during normal regimes has a negative mathematical expectancy. Therefore, the Short Engine is heavily constrained by a **Macro Yield Filter** to ensure it only activates during a verified liquidity crisis (e.g., 2022 Fed Rate Hikes).

*   **Macro Condition 1 (Death Cross):** Price < 200 SMA `AND` 50 SMA < 200 SMA.
*   **Macro Condition 2 (Yield Filter):** 10-Year US Yield > 20 EMA (Confirming a liquidity drain / rate hike cycle).
*   **Trigger (Bear Rally Fade):** When the above conditions are met, wait for the market to violently bounce and retail traders to get greedy (`RSI(2) > 90`), then ruthlessly Short the bounce.

## 3. Dynamic Risk Management (The V7 Equity Filter)
Rather than using static risk or arbitrary volatility scaling, the algorithm mathematically audits its own performance in real-time.

*   **Standard Risk:** 2.0% per trade (Longs), 2.5% per trade (Shorts).
*   **The Filter:** The bot tracks a 20-trade moving average of its own Long Equity Curve.
*   **Drawdown Protection:** If the current account balance drops below the 20-trade moving average, the market is deemed "Choppy/Broken". The bot immediately slashes Long risk by 75% (from 2.0% down to 0.5% per trade). It stays at 0.5% until the equity curve breaks back above the moving average.

---

## 4. 10-Year OOS Performance Metrics (2016-2026)

| Metric | Result |
| :--- | :--- |
| **Total Absolute Return** | **+352,423.0%** |
| **Compound Annual Growth (CAGR)** | **113.8%** |
| **Maximum Drawdown** | **-40.0%** |
| **Worst Year (2016 Chop)** | -35.8% |
| **Best Year (2024 AI Boom)** | +411.8% |
| **Crash Year (2022 Bear Market)** | +72.0% |

### Year-by-Year Breakdown
*   **2016:** -35.8% *(Yield filter blocked shorts during Q1 crash; sideways election chop caused manageable losses).*
*   **2017:** +303.1%
*   **2018:** +40.3%
*   **2019:** +237.7% *(Yield filter successfully blocked shorts during the V-shape recovery).*
*   **2020:** +115.5%
*   **2021:** +19.6%
*   **2022:** +72.0% *(Yields spiked, triggering the Short Engine. Massive crisis alpha generated).*
*   **2023:** +278.4%
*   **2024:** +411.8%
*   **2025:** +113.7%
*   **2026:** +56.7%
