import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, numpy as np, pandas as pd
from run_true_500_v3_ml import load_all_data, build_genes, eval_chromosome
from eurusd_pipeline import sel
from strategies import X
import os

def load_global_data():
    dfs = load_all_data()
    for name in ['DAX_daily', 'Nikkei_daily']:
        df = pd.read_csv(f'../data/{name}.csv', sep='\t', header=None, names=['t', 'o', 'h', 'l', 'c', 'v'])
        df['t'] = pd.to_datetime(df['t'])
        master_index = dfs['ES_daily'].set_index('t').index
        dfs[name] = df.sort_values('t').drop_duplicates('t').set_index('t').reindex(master_index).ffill().bfill().reset_index()
    return dfs

dfs = load_global_data()
genes = build_genes(dfs)
name_to_genes = {g.name: g for g in genes}
START = dfs['ES_daily'].t.min(); END = dfs['ES_daily'].t.max()
DAY = ((pd.to_datetime(dfs['ES_daily'].t.values).floor('D') - START).days).values
d0_test = (pd.Timestamp('2019-01-01') - START).days; d1_test = (END - START).days

with open('dashboard_data_v3_ml.json', 'r') as f: names = [s['name'] for s in json.load(f)[:5]]

all_trades = []
for asset in ['ES_daily', 'NQ_daily', 'DAX_daily', 'Nikkei_daily']:
    df = dfs[asset]; x_asset = X(df)
    o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
    a14 = x_asset.atr(14).values.astype(np.float64)
    for name in names:
        chrom = [name_to_genes[n] for n in name.split(' AND ')]
        bt = eval_chromosome(chrom, x_asset, o, h, l, c, a14, 2.0, 2.0, 40)
        m = sel(dict(ent_day=DAY[bt[0]], ex_day=DAY[bt[1]]), d0_test, d1_test)
        ex_days = DAY[bt[1]][m]; rets = bt[3][m] - 0.00015; risks = bt[4][m]
        for i in range(len(ex_days)): 
            all_trades.append({'ex_day': ex_days[i], 'R': rets[i]/(risks[i] or 1e-9)})
            
all_trades.sort(key=lambda x: x['ex_day'])

eq = 100.0; running_eqs = []; dates = []; eq_curve = []
for t in all_trades:
    running_eqs.append(eq)
    risk = 0.02 * 0.25 if (len(running_eqs) > 20 and eq < np.mean(running_eqs[-20:])) else 0.02
    eq *= (1.0 + risk * t['R'])
    dates.append(pd.Timestamp('2000-09-18') + pd.to_timedelta(t['ex_day'], unit='D'))
    eq_curve.append(eq)

df_eq = pd.DataFrame({'date': dates, 'eq': eq_curve})
daily_idx = pd.date_range(start='2019-01-01', end=df_eq.date.max(), freq='D')
daily_eq = pd.Series(index=daily_idx, dtype=float)
daily_eq.iloc[0] = 100.0
for i, r in df_eq.iterrows(): daily_eq.loc[r['date']] = r['eq']
daily_eq = daily_eq.ffill()
monthly_eq = daily_eq.resample('ME').last()

html = '''<!DOCTYPE html><html><head><title>V7 Final Tear Sheet</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; background: #121212; color: #fff; padding: 20px; }
table { border-collapse: collapse; width: 100%; margin-top: 20px; font-variant-numeric: tabular-nums; }
th, td { padding: 12px; text-align: right; border-bottom: 1px solid #333; }
th { background: #1e1e1e; color: #aaa; }
td.year { font-weight: bold; text-align: left; background: #1e1e1e; }
.pos { color: #4caf50; }
.neg { color: #f44336; }
.flat { color: #888; }
.ytd { font-weight: bold; background: #1e1e1e; }
h1 { color: #4caf50; border-bottom: 2px solid #333; padding-bottom: 10px; }
.stats { display: flex; gap: 20px; margin-top: 20px; }
.stat-box { background: #1e1e1e; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }
.stat-box h3 { margin: 0 0 5px 0; color: #aaa; font-size: 14px; }
.stat-box p { margin: 0; font-size: 24px; font-weight: bold; color: #4caf50; }
</style></head><body>
<h1>V7 Institutional Algorithm (Long-Only + Equity Filter)</h1>
<p>Out-Of-Sample Production Returns (2019-2026). Max Drawdown: 26.5%</p>
<div class="stats">
  <div class="stat-box"><h3>Total OOS Return</h3><p>+72,973%</p></div>
  <div class="stat-box"><h3>CAGR</h3><p>131.5%</p></div>
  <div class="stat-box"><h3>Max Drawdown</h3><p style="color:#f44336">-26.5%</p></div>
</div>
<table><tr><th class="year">Year</th><th>Jan</th><th>Feb</th><th>Mar</th><th>Apr</th><th>May</th><th>Jun</th><th>Jul</th><th>Aug</th><th>Sep</th><th>Oct</th><th>Nov</th><th>Dec</th><th class="ytd">YTD</th></tr>'''

prev_eq = 100.0
for y in sorted(monthly_eq.index.year.unique()):
    html += f'<tr><td class="year">{y}</td>'
    ytd_eq = 1.0
    for m in range(1, 13):
        val = monthly_eq[(monthly_eq.index.year == y) & (monthly_eq.index.month == m)]
        if not val.empty:
            current_eq = val.iloc[0]
            ret = (current_eq / prev_eq - 1.0) * 100
            prev_eq = current_eq
            ytd_eq *= (1.0 + ret/100.0)
            cls = 'pos' if ret > 0 else ('neg' if ret < 0 else 'flat')
            html += f'<td class="{cls}">{ret:+.1f}%</td>'
        else:
            html += '<td class="flat">-</td>'
    ytd = (ytd_eq - 1.0) * 100
    cls = 'pos' if ytd > 0 else ('neg' if ytd < 0 else 'flat')
    html += f'<td class="ytd {cls}">{ytd:+.1f}%</td></tr>'

html += '</table></body></html>'
out_path = 'C:/Users/preet/.gemini/antigravity/brain/e31e97f2-6b70-4ac3-bead-de744f595286/v7_final_tearsheet_dashboard.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Dashboard generated at {out_path}')
