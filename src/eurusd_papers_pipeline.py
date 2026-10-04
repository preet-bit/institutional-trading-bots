import os, time, pickle, sys
import numpy as np, pandas as pd
from numba import njit
from scipy import stats

from core import run_bt, to_sig, cross_up, cross_dn, edge
from eurusd_pipeline import load_eurusd, build_windows, dayoff, pick, trades_of, sel, metrics, daily, sharpe, maxdd, START, END, FX_COST_RT, WINS, d0o, d1o, OOS_YEARS
from strategies import X, S, dc_hi, dc_lo, SL_GRID, RR_GRID, MAX_HOLD

# -------------------------------------------------------------
# 1. John Ehlers FRAMA (Gurrib & Elshareif 2016)
# -------------------------------------------------------------
@njit
def calc_frama_nb(h, l, c, n):
    length = len(c)
    out = np.empty(length, dtype=np.float64)
    out[0] = c[0]
    n2 = n // 2
    for i in range(1, length):
        if i < n:
            out[i] = c[i]
            continue
        max1 = -1e9; min1 = 1e9
        for j in range(i - n + 1, i - n2 + 1):
            if h[j] > max1: max1 = h[j]
            if l[j] < min1: min1 = l[j]
        n1 = (max1 - min1) / n2

        max2 = -1e9; min2 = 1e9
        for j in range(i - n2 + 1, i + 1):
            if h[j] > max2: max2 = h[j]
            if l[j] < min2: min2 = l[j]
        n2_val = (max2 - min2) / n2

        max3 = -1e9; min3 = 1e9
        for j in range(i - n + 1, i + 1):
            if h[j] > max3: max3 = h[j]
            if l[j] < min3: min3 = l[j]
        n3 = (max3 - min3) / n

        if n1 > 0 and n2_val > 0 and n3 > 0 and (n1 + n2_val) > 0:
            d = (np.log(n1 + n2_val) - np.log(n3)) / 0.6931471805599453
        else:
            d = 1.5
        if d < 1.0: d = 1.0
        elif d > 2.0: d = 2.0
        alpha = np.exp(-4.6 * (d - 1.0))
        if alpha < 0.01: alpha = 0.01
        elif alpha > 1.0: alpha = 1.0
        out[i] = alpha * c[i] + (1.0 - alpha) * out[i - 1]
    return out

# -------------------------------------------------------------
# 2. Multi-Scale Volatility Regime Detection (Chaudhary 2026)
# -------------------------------------------------------------
def get_volatility_regimes(df):
    ret = df.c.pct_change()
    vol_short = ret.rolling(12).std()
    vol_long = ret.rolling(150).std()
    vr = vol_short / vol_long.replace(0, np.nan)
    calm = vr < 0.85
    turbulent = (vr >= 0.85) & (vr <= 1.40)
    crisis = vr > 1.40
    return calm, turbulent, crisis

# -------------------------------------------------------------
# 3. Paper Strategy Formulations
# -------------------------------------------------------------
# P01: FRAMA Dual Cross (Gurrib & Elshareif 2016)
def p01_fn(x, n_fast, n_slow):
    f_fast = pd.Series(calc_frama_nb(x.h.values, x.l.values, x.c.values, n_fast), index=x.df.index)
    f_slow = pd.Series(calc_frama_nb(x.h.values, x.l.values, x.c.values, n_slow), index=x.df.index)
    return cross_up(f_fast, f_slow), cross_dn(f_fast, f_slow)

# P02: Turbulent Regime Trend Breakout (Chaudhary 2026)
def p02_fn(x, n, dc_n):
    _, turbulent, _ = get_volatility_regimes(x.df)
    hi = dc_hi(x, dc_n); lo = dc_lo(x, dc_n)
    return edge(x.c > hi) & turbulent, edge(x.c < lo) & turbulent

# P03: Calm Regime Mean Reversion (Wisniewska 2014 & Chaudhary 2026)
def p03_fn(x, n, k):
    calm, _, _ = get_volatility_regimes(x.df)
    mu = x.sma(n); sd = x.c.rolling(n).std()
    return cross_up(x.c, mu - k * sd) & calm, cross_dn(x.c, mu + k * sd) & calm

# P04: FRAMA Trend + Pullback (Guyard & Deriaz 2024 & Gurrib 2016)
def p04_fn(x, n_frama, rsi_n, thr):
    f = pd.Series(calc_frama_nb(x.h.values, x.l.values, x.c.values, n_frama), index=x.df.index)
    r = x.rsi(rsi_n)
    return cross_up(r, thr) & (x.c > f), cross_dn(r, 100 - thr) & (x.c < f)

# P05: Decomposable Friction Aware Channel Breakout (Saidd 2026)
def p05_fn(x, n, min_vol_ratio):
    a14 = x.atr(14); a72 = x.atr(72)
    vol_expansion = (a14 / a72.replace(0, np.nan)) > min_vol_ratio
    hi = dc_hi(x, n); lo = dc_lo(x, n)
    return edge(x.c > hi) & vol_expansion, edge(x.c < lo) & vol_expansion

PAPER_STRATS = [
    S('P01', 'FRAMA Dual Cross', 'Paper-Ehlers-Gurrib', 'Fractal adaptive moving average crossover',
      [(16, 32), (16, 48), (24, 48), (12, 36)], p01_fn),
    S('P02', 'MS-GARCH Turbulent Breakout', 'Paper-Chaudhary2026', 'Donchian breakout gated strictly in Turbulent volatility regime',
      [(20, 20), (20, 40), (40, 20), (40, 40)], p02_fn),
    S('P03', 'Calm Regime Mean-Reversion', 'Paper-Wisniewska-Chaudhary', 'Bollinger fade active strictly in Calm low-volatility regime',
      [(20, 2.0), (20, 2.5), (30, 2.0), (30, 2.5)], p03_fn),
    S('P04', 'FRAMA Trend + RSI Pullback', 'Paper-ML-Hybrid', 'Fractal adaptive trend with short-term RSI pullback entry',
      [(32, 7, 30), (32, 14, 35), (48, 7, 30), (48, 14, 35)], p04_fn),
    S('P05', 'Friction-Aware Expansion Breakout', 'Paper-Saidd2026', 'Channel breakout gated by ATR volatility expansion ratio',
      [(20, 1.1), (20, 1.25), (40, 1.1), (40, 1.25)], p05_fn),
]

def main():
    print("Loading EURUSD data for Paper Strategies evaluation...")
    df = load_eurusd()
    x = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x.atr(14).values.astype(np.float64)
    T = pd.to_datetime(df.t.values)
    DAY = ((T.floor('D') - START).days).values

    res = {}
    for st in PAPER_STRATS:
        for pi, p in enumerate(st['grid']):
            lg, sh = st['fn'](x, *p)
            sig = to_sig(lg, sh); sig[:150] = 0
            for sl in SL_GRID:
                for rr in RR_GRID:
                    res[(st['id'], pi, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)
    print("Backtests completed for 5 paper strategies.")

    # Walk-forward evaluation across 28 test windows
    WF_ROWS = []; OOS = {}
    for st in PAPER_STRATS:
        sid = st['id']; nets = []; rks = []; exds = []; ents = []; wid = []; grosses = []
        for wi, (a, b, c_win) in enumerate(WINS):
            da, db, dc = dayoff(a), dayoff(b), dayoff(c_win)
            best, breadth = pick(res, DAY, st, da, db, min_trades=40)
            row = dict(strategy_id=sid, strategy=st['name'], family=st['family'], window=wi + 1,
                       train=f"{a:%Y-%m-%d}..{b - pd.Timedelta(days=1):%Y-%m-%d}",
                       test=f"{b:%Y-%m-%d}..{c_win - pd.Timedelta(days=1):%Y-%m-%d}",
                       is_breadth_profitable=breadth)
            if best is None:
                row.update(selected='NO TRADE (nothing profitable in-sample)')
                WF_ROWS.append(row); continue
            (pi, sl, rr), im = best
            tr = trades_of(res, DAY, (sid, pi, sl, rr))
            m = sel(tr, db, dc)
            om = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], db, dc)
            row.update(selected=f"{st['grid'][pi]} SL={sl}xATR RR={rr}", sig_params=str(st['grid'][pi]),
                       param_idx=pi, sl_atr=sl, rr=rr, is_trades=im['trades'], is_win_rate=im['win_rate'],
                       is_sharpe=im['sharpe'], is_pf=im['profit_factor'], is_ret_pct=im['total_ret_pct'],
                       oos_trades=om['trades'], oos_win_rate=om['win_rate'], oos_pf=om['profit_factor'],
                       oos_ret_pct=om['total_ret_pct'], oos_exp_R=om['expectancy_R'], oos_sharpe=om['sharpe'])
            WF_ROWS.append(row)
            nets.append(tr['net'][m]); rks.append(tr['risk'][m]); exds.append(tr['ex_day'][m])
            ents.append(tr['ent_day'][m]); wid.append(np.full(m.sum(), wi))
            grosses.append(tr['gross'][m])
        cat = lambda L: np.concatenate(L) if L else np.array([])
        OOS[sid] = dict(net=cat(nets), gross=cat(grosses), risk=cat(rks),
                        exd=cat(exds).astype(int) if exds else np.array([], int),
                        entd=cat(ents), win=cat(wid))

    WF = pd.DataFrame(WF_ROWS)

    # Strategy evaluation + Compounding simulation
    rows = []
    for st in PAPER_STRATS:
        sid = st['id']; o_tr = OOS[sid]
        mt = metrics(o_tr['net'], o_tr['risk'], o_tr['exd'], d0o, d1o)
        dly = daily(o_tr['net'], o_tr['exd'], d0o, d1o) if len(o_tr['net']) else np.zeros(d1o - d0o)

        # Realistic FX Compounding:
        # Fixed fractional risk: 1.5% and 2.0% of account equity per trade
        net = o_tr['net']; rk = o_tr['risk']
        if len(net) > 0 and len(rk) > 0:
            R = net / rk
            # 1.5% risk compounded
            eq_15 = 100.0
            eq_curve_15 = [eq_15]
            for r_val in R:
                eq_15 *= (1.0 + 0.015 * r_val)
                eq_curve_15.append(eq_15)
            comp_15_ret = eq_15 - 100.0

            # 2.0% risk compounded
            eq_20 = 100.0
            for r_val in R:
                eq_20 *= (1.0 + 0.020 * r_val)
            comp_20_ret = eq_20 - 100.0

            # 3x Leverage (annualized)
            lev3_ret = dly.sum() * 3.0 * 100.0
        else:
            comp_15_ret = 0.0; comp_20_ret = 0.0; lev3_ret = 0.0

        mt.update(
            strategy_id=sid, strategy=st['name'], family=st['family'],
            unleveraged_1x_ret_pct=mt['total_ret_pct'],
            leverage_3x_ret_pct=lev3_ret,
            compounded_1_5pct_risk_pct=comp_15_ret,
            compounded_2_0pct_risk_pct=comp_20_ret
        )
        rows.append(mt)

    EV = pd.DataFrame(rows)
    print("\n=== PAPER STRATEGIES SUMMARY (UNLEVERAGED VS COMPOUNDED LEVERAGED) ===")
    cols_p = ['strategy_id', 'strategy', 'trades', 'win_rate', 'profit_factor', 'sharpe',
              'unleveraged_1x_ret_pct', 'leverage_3x_ret_pct', 'compounded_1_5pct_risk_pct', 'compounded_2_0pct_risk_pct']
    print(EV[cols_p].to_string())

if __name__ == '__main__':
    main()
