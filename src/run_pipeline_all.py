import os, time, pickle, sys
import numpy as np, pandas as pd

from core import load
from strategies import STRATS
from hypotheses import HYPO_STRATS
from multitimeframe_pipeline import (
    resample_df, run_dataset_backtests, evaluate_set, exp_max_sr,
    START, END, d0o, d1o, OOS_YEARS, sharpe, maxdd
)

def main():
    t_start = time.time()
    print("Step 1: Loading 15m baseline data...")
    df15 = load()
    print(f"15m bars: {len(df15)}, Range: {df15.t.min()} -> {df15.t.max()}")

    print("Step 2: Resampling to 1h and 4h...")
    df1h = resample_df(df15, '1h')
    df4h = resample_df(df15, '4h')
    print(f"1h bars: {len(df1h)}, Range: {df1h.t.min()} -> {df1h.t.max()}")
    print(f"4h bars: {len(df4h)}, Range: {df4h.t.min()} -> {df4h.t.max()}")

    # Save resampled dataframes
    df1h.to_pickle('df_1h.pkl')
    df4h.to_pickle('df_4h.pkl')

    # Load 15m results
    if os.path.exists('results.pkl'):
        print("Step 3: Loading pre-computed 15m results...")
        d15 = pickle.load(open('results.pkl', 'rb'))
        d15['df'] = df15
    else:
        print("Step 3: Running 15m backtests...")
        d15 = run_dataset_backtests(df15, STRATS, freq_name='15m', warm_bars=600)
        pickle.dump(d15, open('results.pkl', 'wb'))

    # Run 1h backtest
    print("Step 4: Running 1h backtests for 50 strategies...")
    d1h = run_dataset_backtests(df1h, STRATS, freq_name='1h', warm_bars=150)
    pickle.dump(d1h, open('results_1h.pkl', 'wb'))

    # Run 4h backtest
    print("Step 5: Running 4h backtests for 50 strategies...")
    d4h = run_dataset_backtests(df4h, STRATS, freq_name='4h', warm_bars=100)
    pickle.dump(d4h, open('results_4h.pkl', 'wb'))

    # Run Hypotheses (tested on 1h)
    print("Step 6: Running 8 New Hypotheses backtests (1h)...")
    dhypo = run_dataset_backtests(df1h, HYPO_STRATS, freq_name='1h', warm_bars=150)
    pickle.dump(dhypo, open('results_hypo.pkl', 'wb'))

    print("Step 7: First pass evaluation to compute Global SR0 across all 158 trials...")
    # Preliminary evaluation to gather daily returns of all strategies
    ev15_pre = evaluate_set(d15, STRATS, '15m', min_in_sample_trades=80)
    ev1h_pre = evaluate_set(d1h, STRATS, '1h', min_in_sample_trades=80)
    ev4h_pre = evaluate_set(d4h, STRATS, '4h', min_in_sample_trades=40) # scaled for 4h lower bar count
    evhypo_pre = evaluate_set(dhypo, HYPO_STRATS, '1h', min_in_sample_trades=80)

    # Gather all daily sharpes across the 158 strategies
    all_daily_sharpes = []
    for dset in [ev15_pre, ev1h_pre, ev4h_pre, evhypo_pre]:
        for d in dset['DAILY'].values():
            s = d.mean() / d.std(ddof=1) if d.std() > 0 else 0.0
            all_daily_sharpes.append(s)
    
    total_N = len(all_daily_sharpes)
    print(f"Total strategy trials across all timeframes: N = {total_N}")
    global_sr0 = exp_max_sr(np.array(all_daily_sharpes), N=total_N)
    print(f"Global SR0 (daily) = {global_sr0:.4f}, Annualised = {global_sr0 * np.sqrt(365):.2f}")

    print("Step 8: Final walk-forward evaluation with global DSR applied...")
    ev15 = evaluate_set(d15, STRATS, '15m', min_in_sample_trades=80, global_sr0=global_sr0)
    ev1h = evaluate_set(d1h, STRATS, '1h', min_in_sample_trades=80, global_sr0=global_sr0)
    ev4h = evaluate_set(d4h, STRATS, '4h', min_in_sample_trades=40, global_sr0=global_sr0)
    evhypo = evaluate_set(dhypo, HYPO_STRATS, '1h', min_in_sample_trades=80, global_sr0=global_sr0)

    # Buy and hold benchmark
    px = df15.set_index('t').c.resample('D').last().ffill()
    pxd = px.reindex(pd.date_range(START, END - pd.Timedelta(days=1))).ffill()
    bh_ret = pxd.pct_change().fillna(0).values[d0o:d1o]

    def port_stats(d, name, n_strats):
        cum = np.cumsum(d)
        return dict(portfolio=name, sharpe=sharpe(d), total_ret_pct=d.sum() * 100,
                    ann_ret_pct=d.sum() / OOS_YEARS * 100, max_dd_pct=maxdd(cum) * 100,
                    n_strategies=n_strats)

    port_rows = [
        port_stats(ev15['allD'], 'Equal-weight ALL 50 strategies (15m)', 50),
        port_stats(ev1h['allD'], 'Equal-weight ALL 50 strategies (1h)', 50),
        port_stats(ev4h['allD'], 'Equal-weight ALL 50 strategies (4h)', 50),
        port_stats(evhypo['allD'], 'Equal-weight 8 New Hypotheses (1h)', 8),
        port_stats(bh_ret, 'BTC Buy & Hold (Benchmark)', 0),
    ]
    PORT = pd.DataFrame(port_rows)

    out_bundle = dict(
        ev15=ev15, ev1h=ev1h, ev4h=ev4h, evhypo=evhypo,
        PORT=PORT, bh_ret=bh_ret, global_sr0=global_sr0, total_N=total_N,
        d0o=d0o, d1o=d1o, OOS_YEARS=OOS_YEARS
    )
    pickle.dump(out_bundle, open('multitimeframe_analysis.pkl', 'wb'))
    print(f"Step 9: Pipeline completed successfully in {time.time()-t_start:.1f}s!")

if __name__ == '__main__':
    main()
