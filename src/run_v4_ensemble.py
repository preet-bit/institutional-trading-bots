import os, time, sys, json
import numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel, metrics
from strategies import X

def main():
    print("Building V4 Ensembled Portfolio with Aggressive Risk Sizing...")
    
    # 1. Load the top 5 elite strategies from V3
    with open('dashboard_data_v3_ml.json', 'r') as f:
        data = json.load(f)
        
    top_5 = data[:5]
    names = [s['name'] for s in top_5]
    
    # 2. Re-initialize data and engine
    aligned_data = load_all_data()
    genes = build_genes(aligned_data)
    name_to_genes = {g.name: g for g in genes}
    
    es_df = aligned_data['ES_daily']; nq_df = aligned_data['NQ_daily']
    x_es = X(es_df); x_nq = X(nq_df)
    
    o_e, h_e, l_e, c_e = [es_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_e = x_es.atr(14).values.astype(np.float64)
    o_n, h_n, l_n, c_n = [nq_df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14_n = x_nq.atr(14).values.astype(np.float64)
    
    START = es_df.t.min(); END = es_df.t.max()
    T = pd.to_datetime(es_df.t.values)
    DAY = ((T.floor('D') - START).days).values
    
    d0_test = (pd.Timestamp('2019-01-01') - START).days
    d1_test = (END - START).days
    
    COST_RT = 0.00015
    MAX_HOLD = 40
    sl, rr = 2.0, 2.0
    
    all_net, all_rk, all_exd = [], [], []
    
    for name in names:
        chrom = [name_to_genes[n] for n in name.split(' AND ')]
        
        # ES
        bt_es = eval_chromosome(chrom, x_es, o_e, h_e, l_e, c_e, a14_e, sl, rr, MAX_HOLD)
        tr_es = dict(ei=bt_es[0], xi=bt_es[1], ex_day=DAY[bt_es[1]], net=bt_es[3]-COST_RT, risk=bt_es[4])
        m_es = sel(dict(ent_day=DAY[bt_es[0]], ex_day=DAY[bt_es[1]]), d0_test, d1_test)
        
        # NQ
        bt_nq = eval_chromosome(chrom, x_nq, o_n, h_n, l_n, c_n, a14_n, sl, rr, MAX_HOLD)
        tr_nq = dict(ei=bt_nq[0], xi=bt_nq[1], ex_day=DAY[bt_nq[1]], net=bt_nq[3]-COST_RT, risk=bt_nq[4])
        m_nq = sel(dict(ent_day=DAY[bt_nq[0]], ex_day=DAY[bt_nq[1]]), d0_test, d1_test)
        
        all_net.append(tr_es['net'][m_es]); all_net.append(tr_nq['net'][m_nq])
        all_rk.append(tr_es['risk'][m_es]); all_rk.append(tr_nq['risk'][m_nq])
        all_exd.append(tr_es['ex_day'][m_es]); all_exd.append(tr_nq['ex_day'][m_nq])
        
    net_arr = np.concatenate(all_net)
    rk_arr = np.concatenate(all_rk)
    exd_arr = np.concatenate(all_exd).astype(int)
    
    # Sort chronologically to simulate a real portfolio sequence
    idx_sort = np.argsort(exd_arr)
    net_arr, rk_arr, exd_arr = net_arr[idx_sort], rk_arr[idx_sort], exd_arr[idx_sort]
    
    mt = metrics(net_arr, rk_arr, exd_arr, d0_test, d1_test)
    
    # Simulating Aggressive but mathematically sound risk: 5% Fixed Fractional
    # (Since this is an ensemble, they will trade concurrently, mimicking a larger allocation)
    R = net_arr / np.where(rk_arr == 0, 1e-9, rk_arr)
    eq_50 = 100.0
    peak = 100.0
    drawdowns = []
    
    # Track equity curve over time
    eq_curve = []
    dates = pd.Timestamp('2000-09-18') + pd.to_timedelta(exd_arr, unit='D')
    
    for i, rv in enumerate(R):
        eq_50 *= (1.0 + 0.05 * rv)
        if eq_50 > peak: peak = eq_50
        dd = (peak - eq_50) / peak
        drawdowns.append(dd)
        eq_curve.append({'date': dates[i].strftime('%Y-%m-%d'), 'equity': round(eq_50, 2)})
        
    comp_50 = eq_50 - 100.0
    max_dd = max(drawdowns) * 100 if drawdowns else 0.0
    
    # Create an HTML report for this unified portfolio
    html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-gray-950 text-gray-200 antialiased p-8 font-sans">
  <div class="max-w-4xl mx-auto">
      <h1 class="text-4xl font-extrabold text-white tracking-tight mb-2">V4 Meta-Ensemble Portfolio</h1>
      <p class="text-gray-400 mb-8">Aggressive 5% Risk Allocation. Trading the Top 5 Intermarket Strategies concurrently across ES and NQ (2019-2026).</p>
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div class="bg-gray-900 border border-gray-800 p-6 rounded-xl">
            <p class="text-sm text-gray-500 uppercase font-bold">Total Return (OOS)</p>
            <p class="text-3xl font-black text-purple-400">+{comp_50:,.0f}%</p>
        </div>
        <div class="bg-gray-900 border border-gray-800 p-6 rounded-xl">
            <p class="text-sm text-gray-500 uppercase font-bold">Profit Factor</p>
            <p class="text-3xl font-black text-green-400">{mt.get('profit_factor', 0):.2f}</p>
        </div>
        <div class="bg-gray-900 border border-gray-800 p-6 rounded-xl">
            <p class="text-sm text-gray-500 uppercase font-bold">Win Rate</p>
            <p class="text-3xl font-black text-white">{mt['win_rate']*100:.1f}%</p>
        </div>
        <div class="bg-gray-900 border border-gray-800 p-6 rounded-xl">
            <p class="text-sm text-gray-500 uppercase font-bold">Max Drawdown</p>
            <p class="text-3xl font-black text-red-400">-{max_dd:.1f}%</p>
        </div>
      </div>
      
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-6 mb-8">
        <h2 class="text-xl font-bold text-white mb-4">The Meta-Ensemble Logic</h2>
        <ul class="space-y-3 text-sm text-gray-300 font-mono">
"""
    for n in names:
        html += f"<li>{n}</li>"
        
    html += f"""
        </ul>
      </div>
      <div class="bg-gray-900 border border-gray-800 rounded-xl p-6">
        <h2 class="text-xl font-bold text-white mb-4">Ensemble Metrics</h2>
        <p class="text-gray-400">Total OOS Trades: <span class="text-white font-bold">{len(R)}</span></p>
        <p class="text-gray-400">Sharpe Ratio: <span class="text-green-400 font-bold">{mt['sharpe']:.2f}</span></p>
        <p class="text-gray-400">Testing Period: <span class="text-white font-bold">2019 - 2026</span></p>
      </div>
  </div>
</body>
</html>
"""
    out_path = r"C:\Users\preet\.gemini\antigravity\brain\e31e97f2-6b70-4ac3-bead-de744f595286\v4_ensemble_dashboard.html"
    with open(out_path, 'w') as f:
        f.write(html)
        
    print(f"Created V4 Dashboard at {out_path}")
    print(f"V4 Ensemble Compounded Return: +{comp_50:,.1f}%")
    print(f"Max Drawdown: -{max_dd:.1f}%")

if __name__ == '__main__':
    main()
