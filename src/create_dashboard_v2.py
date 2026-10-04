import json
import os

def create_dashboard():
    json_path = r"c:\Users\preet\OneDrive\Documents\bots\src\dashboard_data_v2.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    html = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-8 h-screen overflow-hidden flex flex-col">
  <div class="mb-6 flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold tracking-tight">V2 Smart Strategies (ES + NQ Portfolio)</h1>
      <p class="text-[var(--muted-foreground)] mt-1">Simultaneous execution on S&P 500 and Nasdaq 100. Walk-forward OOS metrics net of $5/contract costs.</p>
    </div>
    <div class="bg-[var(--card)] px-4 py-2 rounded-lg border border-[var(--border)] text-sm">
      <span class="font-semibold">{len(data)}</span> Unique Logic Architectures
    </div>
  </div>

  <div class="bg-[var(--card)] border border-[var(--border)] rounded-xl shadow-sm flex-1 overflow-hidden flex flex-col">
    <div class="overflow-x-auto flex-1">
      <table class="w-full text-sm text-left">
        <thead class="text-xs text-[var(--muted-foreground)] uppercase bg-[var(--sidebar)] sticky top-0 border-b border-[var(--border)]">
          <tr>
            <th scope="col" class="px-6 py-4 font-medium">Rank</th>
            <th scope="col" class="px-6 py-4 font-medium">Strategy Logic (Unique Composition)</th>
            <th scope="col" class="px-6 py-4 font-medium text-right">Portfolio Trades</th>
            <th scope="col" class="px-6 py-4 font-medium text-right">Win Rate</th>
            <th scope="col" class="px-6 py-4 font-medium text-right">Profit Factor</th>
            <th scope="col" class="px-6 py-4 font-medium text-right">Compounded (1.5%)</th>
            <th scope="col" class="px-6 py-4 font-medium text-right">OOS Sharpe</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-[var(--border)] overflow-y-auto">
"""

    for idx, st in enumerate(data):
        if st['sharpe'] > 0.5: sr_color = "text-green-500 font-bold"
        elif st['sharpe'] > 0: sr_color = "text-green-400"
        elif st['sharpe'] < 0: sr_color = "text-red-400"
        else: sr_color = "text-[var(--foreground)]"
        
        if st['pf'] > 1.2: pf_color = "text-green-500 font-bold"
        elif st['pf'] >= 1.0: pf_color = "text-[var(--foreground)]"
        else: pf_color = "text-red-400"
        
        # Color coding for Compounded to highlight massive returns
        if st['compounded'] > 400: comp_color = "text-purple-500 font-bold text-lg"
        elif st['compounded'] > 100: comp_color = "text-green-500 font-bold text-base"
        elif st['compounded'] > 0: comp_color = "text-green-400"
        else: comp_color = "text-red-400"
        
        html += f"""
          <tr class="hover:bg-[var(--sidebar)] transition-colors">
            <td class="px-6 py-3 whitespace-nowrap text-[var(--muted-foreground)]">#{idx+1}</td>
            <td class="px-6 py-3 font-mono text-xs">{st['name']}</td>
            <td class="px-6 py-3 whitespace-nowrap text-right">{st['trades']}</td>
            <td class="px-6 py-3 whitespace-nowrap text-right">{st['win_rate']}%</td>
            <td class="px-6 py-3 whitespace-nowrap text-right {pf_color}">{st['pf']}</td>
            <td class="px-6 py-3 whitespace-nowrap text-right {comp_color}">{st['compounded']:,.1f}%</td>
            <td class="px-6 py-3 whitespace-nowrap text-right {sr_color}">{st['sharpe']}</td>
          </tr>
"""

    html += """
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""

    out_path = r"C:\Users\preet\.gemini\antigravity\brain\e31e97f2-6b70-4ac3-bead-de744f595286\strategies_dashboard_v2.html"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Created dashboard at {out_path}")

if __name__ == '__main__':
    create_dashboard()
