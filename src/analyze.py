import pickle, numpy as np, pandas as pd
from scipy import stats
from core import COST_RT
from strategies import STRATS, SL_GRID, RR_GRID

D = pickle.load(open('results.pkl', 'rb')); RES = D['res']; SIGC = D['sig_counts']
df = pd.read_pickle('df.pkl'); T = pd.to_datetime(df.t.values)
START = pd.Timestamp('2018-01-01'); END = pd.Timestamp('2026-06-16')      # exclusive end
NDAYS = (END - START).days
DAY = ((T.floor('D') - START).days).values                                    # day offset per bar
rng = np.random.default_rng(42)

# walk-forward windows: train 24m / test 6m / step 6m
TRAIN_M, TEST_M = 24, 6
wins = []
ts = pd.Timestamp('2020-01-01')
while ts < END:
    te = min(ts + pd.DateOffset(months=TEST_M), END)
    wins.append((ts - pd.DateOffset(months=TRAIN_M), ts, te)); ts = ts + pd.DateOffset(months=TEST_M)
OOS_START = wins[0][1]

def dayoff(ts): return (ts - START).days

def trades_of(key, cost=COST_RT):
    ei, xi, dr, g, rk = RES[key]
    return dict(ei=ei, xi=xi, ent_day=DAY[ei], ex_day=DAY[xi], net=g - cost, risk=rk, dir=dr)

def sel(tr, d0, d1, purge=False):
    """trades entering in [d0,d1) days; if purge, also require exit day < d1."""
    m = (tr['ent_day'] >= d0) & (tr['ent_day'] < d1)
    if purge: m &= tr['ex_day'] < d1
    return m

def daily(net, exd, d0, d1):
    a = np.zeros(d1 - d0); np.add.at(a, np.clip(exd - d0, 0, d1 - d0 - 1), net); return a

def sharpe(a): s = a.std(ddof=1); return a.mean() / s * np.sqrt(365) if s > 0 else 0.0

def maxdd(cum):
    pk = np.maximum.accumulate(np.concatenate([[0], cum])); return (pk[1:] - cum).max() if len(cum) else 0.0

def metrics(net, risk, exd, d0, d1):
    n = len(net); out = dict(trades=n)
    if n == 0:
        return dict(trades=0, win_rate=np.nan, avg_win_R=np.nan, avg_loss_R=np.nan, payoff=np.nan, breakeven_wr=np.nan,
                    expectancy_R=np.nan, profit_factor=np.nan, total_ret_pct=0.0, ann_ret_pct=0.0, sharpe=0.0, sortino=0.0,
                    max_dd_pct=0.0, calmar=np.nan, avg_bars_note=np.nan)
    R = net / risk; w = net > 0
    aw = R[w].mean() if w.any() else 0.0; al = -R[~w].mean() if (~w).any() else np.nan
    gp = net[w].sum(); gl = -net[~w].sum()
    dly = daily(net, exd, d0, d1); yrs = (d1 - d0) / 365.25
    dn = dly[dly < 0]; sortino = dly.mean() / np.sqrt((dn ** 2).sum() / len(dly)) * np.sqrt(365) if len(dn) else np.nan
    cum = np.cumsum(dly); mdd = maxdd(cum); ann = dly.sum() / yrs
    out.update(win_rate=w.mean(), avg_win_R=aw, avg_loss_R=al, payoff=aw / al if al and al > 0 else np.nan,
               breakeven_wr=1 / (1 + aw / al) if al and al > 0 and aw > 0 else np.nan, expectancy_R=R.mean(),
               profit_factor=gp / gl if gl > 0 else np.inf, total_ret_pct=dly.sum() * 100, ann_ret_pct=ann * 100,
               sharpe=sharpe(dly), sortino=sortino, max_dd_pct=mdd * 100, calmar=ann / mdd if mdd > 0 else np.nan)
    return out

# ---------------------------------------------------------------------------------
# precompute per-combo in-sample score for each window (Sharpe, trades, PF)
def combos(sid):
    st = next(s for s in STRATS if s['id'] == sid)
    return [(pi, sl, rr) for pi in range(len(st['grid'])) for sl in SL_GRID for rr in RR_GRID]

def pick(sid, d0, d1, min_trades=80):
    """In-sample selection on [d0,d1). Returns (best_combo, is_metrics, breadth) or None."""
    best = None; pos = 0; tot = 0
    for (pi, sl, rr) in combos(sid):
        tr = trades_of((sid, pi, sl, rr)); m = sel(tr, d0, d1, purge=True)
        if m.sum() < min_trades: continue
        tot += 1
        mt = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], d0, d1)
        if mt['total_ret_pct'] > 0: pos += 1
        if mt['profit_factor'] > 1.05 and mt['total_ret_pct'] > 0:
            if best is None or mt['sharpe'] > best[1]['sharpe']: best = ((pi, sl, rr), mt)
    breadth = pos / tot if tot else np.nan
    return best, breadth

WF_ROWS = []; OOS = {}      # OOS[sid] = dict(net, risk, exd, entd, win)
for st in STRATS:
    sid = st['id']; nets = []; rks = []; exds = []; ents = []; wid = []
    for wi, (a, b, c) in enumerate(wins):
        da, db, dc = dayoff(a), dayoff(b), dayoff(c)
        best, breadth = pick(sid, da, db)
        row = dict(strategy_id=sid, strategy=st['name'], window=wi + 1, train=f"{a:%Y-%m-%d}..{b - pd.Timedelta(days=1):%Y-%m-%d}",
                   test=f"{b:%Y-%m-%d}..{c - pd.Timedelta(days=1):%Y-%m-%d}", is_breadth_profitable=breadth)
        if best is None:
            row.update(selected='NO TRADE (nothing profitable in-sample)'); WF_ROWS.append(row); continue
        (pi, sl, rr), im = best; tr = trades_of((sid, pi, sl, rr)); m = sel(tr, db, dc)
        om = metrics(tr['net'][m], tr['risk'][m], tr['ex_day'][m], db, dc)
        row.update(selected=f"{st['grid'][pi]} SL={sl}xATR RR={rr}", sig_params=str(st['grid'][pi]), param_idx=pi, sl_atr=sl, rr=rr,
                   is_trades=im['trades'], is_win_rate=im['win_rate'], is_sharpe=im['sharpe'], is_pf=im['profit_factor'],
                   is_ret_pct=im['total_ret_pct'], oos_trades=om['trades'], oos_win_rate=om['win_rate'], oos_pf=om['profit_factor'],
                   oos_ret_pct=om['total_ret_pct'], oos_exp_R=om['expectancy_R'], oos_sharpe=om['sharpe'])
        WF_ROWS.append(row)
        nets.append(tr['net'][m]); rks.append(tr['risk'][m]); exds.append(tr['ex_day'][m]); ents.append(tr['ent_day'][m]); wid.append(np.full(m.sum(), wi))
    cat = lambda L: np.concatenate(L) if L else np.array([])
    OOS[sid] = dict(net=cat(nets), risk=cat(rks), exd=cat(exds).astype(int) if exds else np.array([], int), entd=cat(ents), win=cat(wid))
WF = pd.DataFrame(WF_ROWS)

d0o, d1o = dayoff(OOS_START), NDAYS
OOS_YEARS = (d1o - d0o) / 365.25

# ---------------------------------------------------------------------------------
# per-strategy OOS evaluation
def exp_max_sr(sr_list):
    """Expected max of N Sharpe estimates under the null (Bailey & Lopez de Prado)."""
    N = len(sr_list); v = np.var(sr_list, ddof=1); g = 0.5772156649
    return np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e)))

def dsr(sr_d, sr0, T, skew, kurt):
    den = np.sqrt(max(1e-12, 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2))
    return stats.norm.cdf((sr_d - sr0) * np.sqrt(T - 1) / den)

rows = []; DAILY = {}
for st in STRATS:
    sid = st['id']; o = OOS[sid]
    mt = metrics(o['net'], o['risk'], o['exd'], d0o, d1o)
    dly = daily(o['net'], o['exd'], d0o, d1o) if len(o['net']) else np.zeros(d1o - d0o); DAILY[sid] = dly
    mt.update(strategy_id=sid, strategy=st['name'], family=st['family'])
    rows.append(mt)
EV = pd.DataFrame(rows)
sr_daily = np.array([d.mean() / d.std(ddof=1) if d.std() > 0 else 0 for d in DAILY.values()])
SR0 = exp_max_sr(sr_daily)
dsr_v = []; pv = []; mc_rows = []
for i, st in enumerate(STRATS):
    sid = st['id']; d = DAILY[sid]; o = OOS[sid]
    if d.std() == 0 or len(o['net']) < 5:
        dsr_v.append(np.nan); pv.append(np.nan); mc_rows.append(dict(strategy_id=sid)); continue
    sk = stats.skew(d); ku = stats.kurtosis(d, fisher=False)
    dsr_v.append(dsr(sr_daily[i], SR0, len(d), sk, ku))
    net = o['net']; n = len(net)
    bs = rng.choice(net, size=(2000, n), replace=True)
    means = bs.mean(1); p = (means <= 0).mean()
    tot = bs.sum(1) * 100
    perm = np.argsort(rng.random((500, n)), axis=1); sh = net[perm]; cum = np.cumsum(sh, 1)
    dd = (np.maximum.accumulate(np.concatenate([np.zeros((500, 1)), cum], 1), 1)[:, 1:] - cum).max(1) * 100
    pv.append(p)
    mc_rows.append(dict(strategy_id=sid, mc_p_mean_le_0=p, mc_ret_p5_pct=np.percentile(tot, 5), mc_ret_p50_pct=np.percentile(tot, 50),
                        mc_ret_p95_pct=np.percentile(tot, 95), mc_maxdd_median_pct=np.median(dd), mc_maxdd_p95_pct=np.percentile(dd, 95)))
EV['dsr'] = dsr_v; EV['p_value'] = pv; EV['p_holm_adj'] = np.minimum(1, EV['p_value'] * len(STRATS))
EV = EV.merge(pd.DataFrame(mc_rows), on='strategy_id', how='left')
EV['sr0_ann'] = SR0 * np.sqrt(365)

# window-level consistency + IS vs OOS (walk-forward efficiency)
act = WF[WF.oos_trades.fillna(0) > 0]
cons = act.groupby('strategy_id').agg(windows_traded=('window', 'count'), windows_profitable=('oos_ret_pct', lambda s: (s > 0).sum()),
                                      mean_is_sharpe=('is_sharpe', 'mean'), mean_is_breadth=('is_breadth_profitable', 'mean'),
                                      param_modal_share=('selected', lambda s: s.value_counts(normalize=True).iloc[0])).reset_index()
cons['pct_windows_profitable'] = cons.windows_profitable / cons.windows_traded
nt = WF.groupby('strategy_id').apply(lambda g: g.selected.str.startswith('NO TRADE').sum(), include_groups=False).rename('windows_no_trade').reset_index()
EV = EV.merge(cons, on='strategy_id', how='left').merge(nt, on='strategy_id', how='left')
EV['wfe'] = EV.sharpe / EV.mean_is_sharpe            # OOS Sharpe / mean IS Sharpe
EV['edge_vs_breakeven_wr'] = EV.win_rate - EV.breakeven_wr

# full-sample "best params" (the optimistic, overfit number) for contrast
fs = []
for st in STRATS:
    sid = st['id']; best = None
    for (pi, sl, rr) in combos(sid):
        tr = trades_of((sid, pi, sl, rr))
        if len(tr['net']) < 200: continue
        mt = metrics(tr['net'], tr['risk'], tr['ex_day'], 0, NDAYS)
        if best is None or mt['sharpe'] > best[1]['sharpe']: best = ((pi, sl, rr), mt)
    (pi, sl, rr), mt = best
    fs.append(dict(strategy_id=sid, fullsample_best=f"{st['grid'][pi]} SL={sl} RR={rr}", fs_trades=mt['trades'], fs_win_rate=mt['win_rate'],
                   fs_pf=mt['profit_factor'], fs_sharpe=mt['sharpe'], fs_ret_pct=mt['total_ret_pct']))
EV = EV.merge(pd.DataFrame(fs), on='strategy_id', how='left')

# gates (fixed before looking at results)
EV['g_trades'] = EV.trades >= 100
EV['g_profit'] = (EV.profit_factor >= 1.10) & (EV.sharpe >= 0.5)
EV['g_consistent'] = EV.pct_windows_profitable >= 0.60
EV['g_dsr'] = EV.dsr >= 0.95
EV['gates_passed'] = EV[['g_trades', 'g_profit', 'g_consistent', 'g_dsr']].sum(axis=1)
EV['verdict'] = np.where(EV.gates_passed == 4, 'PASS', np.where((EV.g_trades & EV.g_profit), 'WATCH (profitable, not statistically robust)', 'FAIL'))

# cost sensitivity on the fixed OOS trade set
cs = []
for st in STRATS:
    sid = st['id']; o = OOS[sid]; r = dict(strategy_id=sid, strategy=st['name'])
    for lab, mult in [('zero_cost', 0.0), ('half_cost', 0.5), ('base_cost', 1.0), ('double_cost', 2.0)]:
        if len(o['net']) == 0: r[f'sharpe_{lab}'] = 0; r[f'ret_{lab}_pct'] = 0; continue
        net = o['net'] + COST_RT - COST_RT * mult; dl = daily(net, o['exd'], d0o, d1o)
        r[f'sharpe_{lab}'] = sharpe(dl); r[f'ret_{lab}_pct'] = dl.sum() * 100
    cs.append(r)
CS = pd.DataFrame(cs)

# yearly OOS returns
yr = []
ydays = {y: (dayoff(pd.Timestamp(f'{y}-01-01')), dayoff(min(pd.Timestamp(f'{y + 1}-01-01'), END))) for y in range(2020, 2027)}
for st in STRATS:
    sid = st['id']; o = OOS[sid]; r = dict(strategy_id=sid, strategy=st['name'])
    for y, (a, b) in ydays.items():
        m = (o['exd'] >= a) & (o['exd'] < b) if len(o['net']) else np.array([], bool)
        r[str(y)] = o['net'][m].sum() * 100 if len(o['net']) else 0.0
    yr.append(r)
YR = pd.DataFrame(yr)

# portfolio + buy & hold
allD = np.mean([DAILY[s['id']] for s in STRATS], axis=0)
passD = [DAILY[s] for s in EV[EV.verdict == 'PASS'].strategy_id]
watchD = [DAILY[s] for s in EV[EV.verdict.str.startswith('WATCH') | (EV.verdict == 'PASS')].strategy_id]
px = df.set_index(T).c.resample('D').last().ffill()
bh = px.pct_change().fillna(0).values[(START - px.index[0]).days:][d0o:d1o] if False else None
pxd = px.reindex(pd.date_range(START, END - pd.Timedelta(days=1))).ffill()
bh_ret = pxd.pct_change().fillna(0).values[d0o:d1o]
def port_stats(d, name):
    cum = np.cumsum(d); return dict(portfolio=name, sharpe=sharpe(d), total_ret_pct=d.sum() * 100, ann_ret_pct=d.sum() / OOS_YEARS * 100,
                                    max_dd_pct=maxdd(cum) * 100, n_strategies=None)
PORT = [port_stats(allD, 'Equal-weight ALL 50 strategies (no selection bias)'),
        port_stats(bh_ret, 'BTC buy & hold (same period)')]
PORT[0]['n_strategies'] = 50; PORT[1]['n_strategies'] = 0
if passD: PORT.append(port_stats(np.mean(passD, axis=0), 'Equal-weight PASS strategies (selected on OOS: optimistic)')); PORT[-1]['n_strategies'] = len(passD)
if len(watchD) and len(watchD) != len(passD): PORT.append(port_stats(np.mean(watchD, axis=0), 'Equal-weight PASS+WATCH (selected on OOS: optimistic)')); PORT[-1]['n_strategies'] = len(watchD)
PORT = pd.DataFrame(PORT)

# "current" parameters = fit on last 24 months (no OOS validation yet)
cur = []
for st in STRATS:
    best, br = pick(st['id'], dayoff(END - pd.DateOffset(months=24)), NDAYS, 80)
    if best is None: cur.append(dict(strategy_id=st['id'], strategy=st['name'], current_params='NO TRADE (nothing profitable in last 24m)')); continue
    (pi, sl, rr), im = best
    cur.append(dict(strategy_id=st['id'], strategy=st['name'], current_params=f"{st['grid'][pi]} SL={sl}xATR RR={rr}", trades_24m=im['trades'],
                    win_rate_24m=im['win_rate'], pf_24m=im['profit_factor'], sharpe_24m=im['sharpe']))
CUR = pd.DataFrame(cur)

DEFS = pd.DataFrame([dict(strategy_id=s['id'], name=s['name'], family=s['family'], logic=s['desc'], param_grid=str(s['grid']),
                          combos_tested=len(s['grid']) * len(SL_GRID) * len(RR_GRID)) for s in STRATS])

cd = []
for st in STRATS:
    sid = st['id']; cb_eval = 0; gross_bps = []; net_bps = []; stops = []; pos_g = 0; pos_n = 0
    for (pi, sl, rr) in combos(sid):
        tr = trades_of((sid, pi, sl, rr))
        if len(tr['net']) < 200: continue
        cb_eval += 1
        g = tr['net'] + COST_RT
        g_bps = g.mean() * 10000; n_bps = tr['net'].mean() * 10000
        gross_bps.append(g_bps); net_bps.append(n_bps)
        stops.extend(tr['risk'])
        if g_bps > 0: pos_g += 1
        if n_bps > 0: pos_n += 1
    cd.append(dict(strategy_id=sid, strategy=st['name'], combos_eval=float(cb_eval),
                   avg_gross_bps_per_trade=np.mean(gross_bps) if gross_bps else np.nan,
                   avg_net_bps_per_trade=np.mean(net_bps) if net_bps else np.nan,
                   cost_bps_rt=COST_RT * 10000,
                   median_stop_pct=np.median(stops) * 100 if len(stops) else np.nan,
                   pct_combos_gross_positive=pos_g / cb_eval if cb_eval else np.nan,
                   pct_combos_net_positive=pos_n / cb_eval if cb_eval else np.nan))
CD = pd.DataFrame(cd)

pickle.dump(dict(EV=EV, WF=WF, CS=CS, YR=YR, PORT=PORT, CUR=CUR, DEFS=DEFS, CD=CD, DAILY=DAILY, allD=allD, bh=bh_ret, OOS=OOS, wins=wins,
                 d0o=d0o, d1o=d1o, SR0=SR0), open('analysis.pkl', 'wb'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 80)
print(len(wins), 'windows; OOS', OOS_START.date(), '->', (END - pd.Timedelta(days=1)).date(), f'SR0(ann)={SR0 * np.sqrt(365):.2f}')
cols = ['strategy_id', 'strategy', 'trades', 'win_rate', 'payoff', 'expectancy_R', 'profit_factor', 'sharpe', 'max_dd_pct', 'ann_ret_pct', 'pct_windows_profitable', 'dsr', 'fs_sharpe', 'verdict']
print(EV.sort_values('sharpe', ascending=False)[cols].round(3).to_string(index=False))
print(PORT.round(2).to_string(index=False))
print(EV.verdict.value_counts())
