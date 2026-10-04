import os, time, pickle, sys
import numpy as np, pandas as pd
from scipy import stats

from core import load, run_bt, to_sig, COST_RT, FEE, SLIP
import strategies
from strategies import STRATS, SL_GRID, RR_GRID, MAX_HOLD, X
from hypotheses import HYPO_STRATS

# -------------------------------------------------------------
# 1. Resampling utilities
# -------------------------------------------------------------
def resample_df(df15, freq):
    """Resample 15m Binance klines to 1h or 4h."""
    df_s = df15.set_index('t')
    r = df_s.resample(freq, origin='start_day')
    agg = pd.DataFrame({
        'o': r['o'].first(),
        'h': r['h'].max(),
        'l': r['l'].min(),
        'c': r['c'].last(),
        'v': r['v'].sum(),
        'nt': r['nt'].sum(),
        'tb': r['tb'].sum(),
    }).dropna(subset=['o', 'c'])
    return agg.reset_index()

# Timeframe-aware adapter for strategies S19 and S50
def get_strategy_signals(st, x, p, freq='15m'):
    sid = st['id']
    if sid == 'S19':
        # Opening range breakout: scale n to timeframe
        n = p[0]
        if freq == '1h':
            n_eff = max(1, n // 4)
        elif freq == '4h':
            n_eff = max(1, n // 16)
        else:
            n_eff = n
        pos = x.df.groupby(x.day).cumcount()
        hi = x.h.where(pos < n_eff).groupby(x.day).cummax().groupby(x.day).transform('max')
        lo = x.l.where(pos < n_eff).groupby(x.day).cummin().groupby(x.day).transform('min')
        ok = pos >= n_eff
        from core import edge
        return edge((x.c > hi) & ok), edge((x.c < lo) & ok)
    elif sid == 'S50':
        # Session open momentum
        H, k = p
        if freq == '15m':
            at = (x.hour == (H - 1) % 24) & (x.minute == 45)
            r = x.c / x.c.shift(k) - 1
        elif freq == '1h':
            at = (x.hour == (H - 1) % 24)
            k_eff = max(2, k // 4)
            r = x.c / x.c.shift(k_eff) - 1
        elif freq == '4h':
            at = (x.hour == ((H - 4) % 24))
            k_eff = max(1, k // 16)
            r = x.c / x.c.shift(k_eff) - 1
        return at & (r > 0), at & (r < 0)
    else:
        return st['fn'](x, *p)

# -------------------------------------------------------------
# 2. Backtest runner for any timeframe / strategy set
# -------------------------------------------------------------
def run_dataset_backtests(df, strat_list, freq_name='1h', warm_bars=200):
    x = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x.atr(14).values.astype(np.float64)
    res = {}
    sig_counts = {}
    t0 = time.time()
    for st in strat_list:
        for pi, p in enumerate(st['grid']):
            lg, sh = get_strategy_signals(st, x, p, freq=freq_name)
            sig = to_sig(lg, sh)
            sig[:warm_bars] = 0
            sig_counts[(st['id'], pi)] = int((sig != 0).sum())
            for sl in SL_GRID:
                for rr in RR_GRID:
                    res[(st['id'], pi, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)
    print(f"[{freq_name}] Finished {len(strat_list)} strats in {time.time()-t0:.1f}s", flush=True)
    return dict(res=res, sig_counts=sig_counts, t=df.t.values, df=df)

# -------------------------------------------------------------
# 3. Walk-Forward Evaluator
# -------------------------------------------------------------
START = pd.Timestamp('2018-01-01')
END = pd.Timestamp('2026-06-16')
NDAYS = (END - START).days

def build_windows():
    wins = []
    ts = pd.Timestamp('2020-01-01')
    while ts < END:
        te = min(ts + pd.DateOffset(months=6), END)
        wins.append((ts - pd.DateOffset(months=24), ts, te))
        ts = ts + pd.DateOffset(months=6)
    return wins

WINS = build_windows()
OOS_START = WINS[0][1]
d0o = (OOS_START - START).days
d1o = NDAYS
OOS_YEARS = (d1o - d0o) / 365.25

def dayoff(ts):
    return (ts - START).days

def trades_of(RES, DAY, key, cost=COST_RT):
    ei, xi, dr, g, rk = RES[key]
    return dict(ei=ei, xi=xi, ent_day=DAY[ei], ex_day=DAY[xi], net=g - cost, gross=g, risk=rk, dir=dr)

def sel(tr, d0, d1, purge=False):
    m = (tr['ent_day'] >= d0) & (tr['ent_day'] < d1)
    if purge:
        m &= tr['ex_day'] < d1
    return m

def daily(net, exd, d0, d1):
    a = np.zeros(d1 - d0)
    np.add.at(a, np.clip(exd - d0, 0, d1 - d0 - 1), net)
    return a

def sharpe(a):
    s = a.std(ddof=1)
    return a.mean() / s * np.sqrt(365) if s > 0 else 0.0

def maxdd(cum):
    pk = np.maximum.accumulate(np.concatenate([[0], cum]))
    return (pk[1:] - cum).max() if len(cum) else 0.0

def metrics(net, risk, exd, d0, d1):
    n = len(net)
    out = dict(trades=n)
    if n == 0:
        return dict(trades=0, win_rate=np.nan, avg_win_R=np.nan, avg_loss_R=np.nan, payoff=np.nan,
                    breakeven_wr=np.nan, expectancy_R=np.nan, profit_factor=np.nan, total_ret_pct=0.0,
                    ann_ret_pct=0.0, sharpe=0.0, sortino=0.0, max_dd_pct=0.0, calmar=np.nan)
    R = net / risk
    w = net > 0
    aw = R[w].mean() if w.any() else 0.0
    al = -R[~w].mean() if (~w).any() else np.nan
    gp = net[w].sum()
    gl = -net[~w].sum()
    dly = daily(net, exd, d0, d1)
    yrs = (d1 - d0) / 365.25
    dn = dly[dly < 0]
    sortino = dly.mean() / np.sqrt((dn ** 2).sum() / len(dly)) * np.sqrt(365) if len(dn) else np.nan
    cum = np.cumsum(dly)
    mdd = maxdd(cum)
    ann = dly.sum() / yrs
    out.update(win_rate=w.mean(), avg_win_R=aw, avg_loss_R=al,
               payoff=aw / al if al and al > 0 else np.nan,
               breakeven_wr=1 / (1 + aw / al) if al and al > 0 and aw > 0 else np.nan,
               expectancy_R=R.mean(),
               profit_factor=gp / gl if gl > 0 else np.inf,
               total_ret_pct=dly.sum() * 100, ann_ret_pct=ann * 100,
               sharpe=sharpe(dly), sortino=sortino, max_dd_pct=mdd * 100,
               calmar=ann / mdd if mdd > 0 else np.nan)
    return out

def combos(st):
    return [(pi, sl, rr) for pi in range(len(st['grid'])) for sl in SL_GRID for rr in RR_GRID]

def pick(RES, DAY, st, d0, d1, min_trades=80):
    sid = st['id']
    best = None
    pos = 0
    tot = 0
    for (pi, sl, rr) in combos(st):
        tr = trades_of(RES, DAY, (sid, pi, sl, rr))
        m = sel(tr, d0, d1, purge=True)
        if m.sum() < min_trades:
            continue
        tot += 1
        mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], d0, d1)
        if mt['total_ret_pct'] > 0:
            pos += 1
        if mt['profit_factor'] > 1.05 and mt['total_ret_pct'] > 0:
            if best is None or mt['sharpe'] > best[1]['sharpe']:
                best = ((pi, sl, rr), mt)
    breadth = pos / tot if tot else np.nan
    return best, breadth

def exp_max_sr(sr_list, N=None):
    if N is None:
        N = len(sr_list)
    v = np.var(sr_list, ddof=1)
    g = 0.5772156649
    return np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e)))

def dsr(sr_d, sr0, T, skew, kurt):
    den = np.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return stats.norm.cdf((sr_d - sr0) * np.sqrt(T - 1) / den)

# -------------------------------------------------------------
# 4. Full Evaluation Engine for one Timeframe/Strategy set
# -------------------------------------------------------------
def evaluate_set(dataset_dict, strat_list, timeframe_label, min_in_sample_trades=80, global_sr0=None):
    RES = dataset_dict['res']
    df = dataset_dict['df']
    T = pd.to_datetime(df.t.values)
    DAY = ((T.floor('D') - START).days).values

    WF_ROWS = []
    OOS = {}
    for st in strat_list:
        sid = st['id']
        nets = []; rks = []; exds = []; ents = []; wid = []
        grosses = []
        for wi, (a, b, c) in enumerate(WINS):
            da, db, dc = dayoff(a), dayoff(b), dayoff(c)
            best, breadth = pick(RES, DAY, st, da, db, min_trades=min_in_sample_trades)
            row = dict(strategy_id=sid, strategy=st['name'], timeframe=timeframe_label, window=wi + 1,
                       train=f"{a:%Y-%m-%d}..{b - pd.Timedelta(days=1):%Y-%m-%d}",
                       test=f"{b:%Y-%m-%d}..{c - pd.Timedelta(days=1):%Y-%m-%d}",
                       is_breadth_profitable=breadth)
            if best is None:
                row.update(selected='NO TRADE (nothing profitable in-sample)')
                WF_ROWS.append(row)
                continue
            (pi, sl, rr), im = best
            tr = trades_of(RES, DAY, (sid, pi, sl, rr))
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

    # Strategy-level metrics
    rows = []
    DAILY = {}
    for st in strat_list:
        sid = st['id']
        o = OOS[sid]
        mt = metrics(o['net'], o['risk'], o['exd'], d0o, d1o)
        dly = daily(o['net'], o['exd'], d0o, d1o) if len(o['net']) else np.zeros(d1o - d0o)
        DAILY[sid] = dly
        mt.update(strategy_id=sid, strategy=st['name'], family=st.get('family', 'Custom'), timeframe=timeframe_label)
        rows.append(mt)
    EV = pd.DataFrame(rows)

    sr_daily = np.array([d.mean() / d.std(ddof=1) if d.std() > 0 else 0 for d in DAILY.values()])
    SR0_local = exp_max_sr(sr_daily, N=len(strat_list))
    SR0_effective = global_sr0 if global_sr0 is not None else SR0_local

    rng = np.random.default_rng(42)
    dsr_local_v = []; dsr_global_v = []; pv = []; mc_rows = []
    maker_rows = []

    for i, st in enumerate(strat_list):
        sid = st['id']; d = DAILY[sid]; o = OOS[sid]
        if d.std() == 0 or len(o['net']) < 5:
            dsr_local_v.append(np.nan); dsr_global_v.append(np.nan); pv.append(np.nan)
            mc_rows.append(dict(strategy_id=sid))
            maker_rows.append(dict(strategy_id=sid, maker_trades=0, maker_sharpe=0.0, maker_pf=np.nan,
                                   maker_ret_pct=0.0, maker_haircut_sharpe=0.0, maker_haircut_pf=np.nan,
                                   maker_haircut_ret_pct=0.0))
            continue
        sk = stats.skew(d); ku = stats.kurtosis(d, fisher=False)
        dsr_local_v.append(dsr(sr_daily[i], SR0_local, len(d), sk, ku))
        dsr_global_v.append(dsr(sr_daily[i], SR0_effective, len(d), sk, ku))

        # Bootstrap p-value & Monte Carlo drawdown
        net = o['net']; n = len(net)
        bs = rng.choice(net, size=(2000, n), replace=True)
        means = bs.mean(1); p = (means <= 0).mean()
        tot = bs.sum(1) * 100
        perm = np.argsort(rng.random((500, n)), axis=1); sh = net[perm]; cum = np.cumsum(sh, 1)
        dd = (np.maximum.accumulate(np.concatenate([np.zeros((500, 1)), cum], 1), 1)[:, 1:] - cum).max(1) * 100
        pv.append(p)
        mc_rows.append(dict(strategy_id=sid, mc_p_mean_le_0=p, mc_ret_p5_pct=np.percentile(tot, 5),
                            mc_ret_p50_pct=np.percentile(tot, 50), mc_ret_p95_pct=np.percentile(tot, 95),
                            mc_maxdd_median_pct=np.median(dd), mc_maxdd_p95_pct=np.percentile(dd, 95)))

        # Maker-fee sensitivity scenario:
        # Maker fee: 0.02% per side limit entry (0 slippage).
        # Exit: limit TP (0.02%) vs market SL (0.04% + 0.02% slip = 0.06%).
        # Round trip cost: 0.04% on winners (TP), 0.08% on losers (SL)
        gross = o['gross']
        # Winner gross > 0 => cost_rt = 0.0004; loser gross <= 0 => cost_rt = 0.0008
        cost_maker = np.where(gross > 0, 0.0004, 0.0008)
        net_maker = gross - cost_maker
        dl_maker = daily(net_maker, o['exd'], d0o, d1o)
        m_win = net_maker > 0
        gp_m = net_maker[m_win].sum(); gl_m = -net_maker[~m_win].sum()
        pf_m = gp_m / gl_m if gl_m > 0 else np.inf

        # Fill-probability haircut: 20% limit orders fail to fill (80% fill rate)
        # Non-filled trades return 0
        fill_mask = rng.random(n) < 0.80
        net_haircut = np.where(fill_mask, net_maker, 0.0)
        dl_haircut = daily(net_haircut, o['exd'], d0o, d1o)
        h_win = net_haircut > 0; h_loss = net_haircut < 0
        gp_h = net_haircut[h_win].sum(); gl_h = -net_haircut[h_loss].sum()
        pf_h = gp_h / gl_h if gl_h > 0 else np.inf

        maker_rows.append(dict(
            strategy_id=sid,
            maker_trades=n,
            maker_sharpe=sharpe(dl_maker),
            maker_pf=pf_m,
            maker_ret_pct=dl_maker.sum() * 100,
            maker_haircut_trades=int(fill_mask.sum()),
            maker_haircut_sharpe=sharpe(dl_haircut),
            maker_haircut_pf=pf_h,
            maker_haircut_ret_pct=dl_haircut.sum() * 100
        ))

    EV['dsr_local'] = dsr_local_v
    EV['dsr_global'] = dsr_global_v
    EV['dsr'] = dsr_local_v  # default reported
    EV['p_value'] = pv
    EV['p_holm_adj'] = np.minimum(1, np.array(pv) * len(strat_list))
    EV = EV.merge(pd.DataFrame(mc_rows), on='strategy_id', how='left')
    EV = EV.merge(pd.DataFrame(maker_rows), on='strategy_id', how='left')
    EV['sr0_ann'] = SR0_local * np.sqrt(365)
    EV['sr0_global_ann'] = SR0_effective * np.sqrt(365)

    # Window consistency
    act = WF[WF.oos_trades.fillna(0) > 0]
    if len(act):
        cons = act.groupby('strategy_id').agg(
            windows_traded=('window', 'count'),
            windows_profitable=('oos_ret_pct', lambda s: (s > 0).sum()),
            mean_is_sharpe=('is_sharpe', 'mean'),
            mean_is_breadth=('is_breadth_profitable', 'mean'),
            param_modal_share=('selected', lambda s: s.value_counts(normalize=True).iloc[0])
        ).reset_index()
        cons['pct_windows_profitable'] = cons.windows_profitable / cons.windows_traded
        nt = WF.groupby('strategy_id').apply(
            lambda g: g.selected.str.startswith('NO TRADE').sum(), include_groups=False
        ).rename('windows_no_trade').reset_index()
        EV = EV.merge(cons, on='strategy_id', how='left').merge(nt, on='strategy_id', how='left')
    else:
        EV['windows_traded'] = 0; EV['windows_profitable'] = 0; EV['pct_windows_profitable'] = np.nan
        EV['mean_is_sharpe'] = np.nan; EV['mean_is_breadth'] = np.nan; EV['param_modal_share'] = np.nan
        EV['windows_no_trade'] = len(WINS)

    EV['wfe'] = EV.sharpe / EV.mean_is_sharpe
    EV['edge_vs_breakeven_wr'] = EV.win_rate - EV.breakeven_wr

    # Full-sample best params
    fs = []
    for st in strat_list:
        sid = st['id']; best = None
        for (pi, sl, rr) in combos(st):
            tr = trades_of(RES, DAY, (sid, pi, sl, rr))
            if len(tr['net']) < (50 if timeframe_label == '4h' else 150): continue
            mt = metrics(tr['net'], tr['risk'], tr['ex_day'], 0, NDAYS)
            if best is None or mt['sharpe'] > best[1]['sharpe']:
                best = ((pi, sl, rr), mt)
        if best is not None:
            (pi, sl, rr), mt = best
            fs.append(dict(strategy_id=sid, fullsample_best=f"{st['grid'][pi]} SL={sl} RR={rr}",
                           fs_trades=mt['trades'], fs_win_rate=mt['win_rate'], fs_pf=mt['profit_factor'],
                           fs_sharpe=mt['sharpe'], fs_ret_pct=mt['total_ret_pct']))
        else:
            fs.append(dict(strategy_id=sid, fullsample_best='None (insufficient trades)',
                           fs_trades=0, fs_win_rate=np.nan, fs_pf=np.nan, fs_sharpe=0.0, fs_ret_pct=0.0))
    EV = EV.merge(pd.DataFrame(fs), on='strategy_id', how='left')

    # Gates (fixed in advance)
    # 1: >=100 OOS trades
    # 2: PF >= 1.10 and Sharpe >= 0.50
    # 3: >=60% profitable test windows
    # 4: DSR >= 0.95
    EV['g_trades'] = EV.trades >= 100
    EV['g_profit'] = (EV.profit_factor >= 1.10) & (EV.sharpe >= 0.50)
    EV['g_consistent'] = EV.pct_windows_profitable >= 0.60
    EV['g_dsr'] = EV.dsr_global >= 0.95
    EV['gates_passed'] = EV[['g_trades', 'g_profit', 'g_consistent', 'g_dsr']].sum(axis=1)

    EV['verdict'] = np.where(
        EV.trades == 0, 'FAIL (no in-sample edge found, never traded)',
        np.where(
            EV.gates_passed == 4, 'PASS',
            np.where(
                EV.g_trades & EV.g_profit, 'WATCH (profitable OOS, not statistically robust)',
                'FAIL'
            )
        )
    )

    # Cost sensitivity
    cs = []
    for st in strat_list:
        sid = st['id']; o = OOS[sid]; r = dict(strategy_id=sid, strategy=st['name'], timeframe=timeframe_label)
        for lab, mult in [('zero_cost', 0.0), ('half_cost', 0.5), ('base_cost', 1.0), ('double_cost', 2.0)]:
            if len(o['net']) == 0:
                r[f'sharpe_{lab}'] = 0; r[f'ret_{lab}_pct'] = 0; continue
            net = o['net'] + COST_RT - COST_RT * mult
            dl = daily(net, o['exd'], d0o, d1o)
            r[f'sharpe_{lab}'] = sharpe(dl); r[f'ret_{lab}_pct'] = dl.sum() * 100
        cs.append(r)
    CS = pd.DataFrame(cs)

    # Yearly returns
    yr = []
    ydays = {y: (dayoff(pd.Timestamp(f'{y}-01-01')), dayoff(min(pd.Timestamp(f'{y + 1}-01-01'), END))) for y in range(2020, 2027)}
    for st in strat_list:
        sid = st['id']; o = OOS[sid]; r = dict(strategy_id=sid, strategy=st['name'], timeframe=timeframe_label)
        for y, (a, b) in ydays.items():
            m = (o['exd'] >= a) & (o['exd'] < b) if len(o['net']) else np.array([], bool)
            r[str(y)] = o['net'][m].sum() * 100 if len(o['net']) else 0.0
        yr.append(r)
    YR = pd.DataFrame(yr)

    # Cost Diagnostics
    cd = []
    min_cd_trades = 20 if timeframe_label == '4h' else (60 if timeframe_label == '1h' else 200)
    for st in strat_list:
        sid = st['id']; cb_eval = 0; gross_bps = []; net_bps = []; stops = []; pos_g = 0; pos_n = 0
        for (pi, sl, rr) in combos(st):
            tr = trades_of(RES, DAY, (sid, pi, sl, rr))
            if len(tr['net']) < min_cd_trades: continue
            cb_eval += 1
            g = tr['net'] + COST_RT
            g_bps = g.mean() * 10000; n_bps = tr['net'].mean() * 10000
            gross_bps.append(g_bps); net_bps.append(n_bps)
            stops.extend(tr['risk'])
            if g_bps > 0: pos_g += 1
            if n_bps > 0: pos_n += 1
        cd.append(dict(strategy_id=sid, strategy=st['name'], timeframe=timeframe_label, combos_eval=float(cb_eval),
                       avg_gross_bps_per_trade=np.mean(gross_bps) if gross_bps else np.nan,
                       avg_net_bps_per_trade=np.mean(net_bps) if net_bps else np.nan,
                       cost_bps_rt=COST_RT * 10000,
                       median_stop_pct=np.median(stops) * 100 if len(stops) else np.nan,
                       pct_combos_gross_positive=pos_g / cb_eval if cb_eval else np.nan,
                       pct_combos_net_positive=pos_n / cb_eval if cb_eval else np.nan))
    CD = pd.DataFrame(cd)

    # Win rate vs RR
    wr = []
    for sl in SL_GRID:
        for rr in RR_GRID:
            n = 0; wg = 0; wn = 0; sg = 0.0; sn = 0.0
            for st in strat_list:
                for pi in range(len(st['grid'])):
                    ei, xi, dr, g, rk = RES[(st['id'], pi, sl, rr)]
                    n += len(g)
                    wg += (g > 0).sum()
                    wn += ((g - COST_RT) > 0).sum()
                    sg += (g / rk).sum()
                    sn += ((g - COST_RT) / rk).sum()
            wr.append(dict(sl_atr=sl, rr_target=rr, trades=n, win_rate_gross=wg / n if n else np.nan,
                           win_rate_net=wn / n if n else np.nan, theoretical_breakeven_wr=1 / (1 + rr),
                           expectancy_R_gross=sg / n if n else np.nan, expectancy_R_net=sn / n if n else np.nan))
    WRR = pd.DataFrame(wr)

    # Current parameters (fitted on last 24m)
    cur = []
    for st in strat_list:
        best, br = pick(RES, DAY, st, dayoff(END - pd.DateOffset(months=24)), NDAYS, min_trades=min_in_sample_trades)
        if best is None:
            cur.append(dict(strategy_id=st['id'], strategy=st['name'], timeframe=timeframe_label,
                            current_params='NO TRADE (nothing profitable in last 24m)'))
            continue
        (pi, sl, rr), im = best
        cur.append(dict(strategy_id=st['id'], strategy=st['name'], timeframe=timeframe_label,
                        current_params=f"{st['grid'][pi]} SL={sl}xATR RR={rr}", trades_24m=im['trades'],
                        win_rate_24m=im['win_rate'], pf_24m=im['profit_factor'], sharpe_24m=im['sharpe']))
    CUR = pd.DataFrame(cur)

    # Equal-weight portfolio
    allD = np.mean([DAILY[s['id']] for s in strat_list], axis=0) if len(strat_list) else np.zeros(d1o - d0o)

    return dict(EV=EV, WF=WF, CS=CS, YR=YR, CD=CD, WRR=WRR, CUR=CUR, DAILY=DAILY, allD=allD,
                OOS=OOS, SR0_local=SR0_local, SR0_effective=SR0_effective)

print("Pipeline module loaded successfully.")
