import json
import os

def create_dashboard():
    json_path = r"c:\Users\preet\OneDrive\Documents\bots\src\dashboard_data_v3_ml.json"
    
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
<body class="bg-gray-950 text-gray-200 antialiased p-8 h-screen overflow-hidden flex flex-col font-sans">
  <div class="mb-6 flex items-center justify-between">
    <div>
      <h1 class="text-3xl font-extrabold text-white tracking-tight">V3 Intermarket Genetic Engine (OOS Test Window: 2019-2026)</h1>
      <p class="text-gray-400 mt-1">Simultaneous execution on S&P 500 and Nasdaq 100. Filtered by Gold, Oil, and Bond Yields. Evaluated strictly Out-Of-Sample (net of $5 costs).</p>
    </div>
    <div class="bg-gray-900 px-4 py-2 rounded-lg border border-gray-800 text-sm">
      <span class="font-bold text-blue-400">{len(data)}</span> Elite Architectures Evolved
    </div>
  </div>

  <div class="bg-gray-900 border border-gray-800 rounded-xl shadow-2xl flex-1 overflow-hidden flex flex-col">
    <div class="overflow-x-auto flex-1">
      <table class="w-full text-sm text-left">
        <thead class="text-xs text-gray-500 uppercase bg-gray-950 sticky top-0 border-b border-gray-800 shadow-sm">
          <tr>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider">Rank</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider">Genetic Logic (Intermarket Composition)</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider text-right">OOS Trades</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider text-right">Win Rate</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider text-right">Profit Factor</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider text-right">Compounded (2.0% Risk)</th>
            <th scope="col" class="px-6 py-4 font-bold tracking-wider text-right">OOS Sharpe</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-800 overflow-y-auto">
"""

    for idx, st in enumerate(data):
        if st['sharpe'] > 1.0: sr_color = "text-purple-500 font-black"
        elif st['sharpe'] > 0.5: sr_color = "text-green-500 font-bold"
        elif st['sharpe'] > 0: sr_color = "text-green-400"
        elif st['sharpe'] < 0: sr_color = "text-red-400"
        else: sr_color = "text-gray-400"
        
        if st['pf'] > 1.5: pf_color = "text-green-500 font-bold"
        elif st['pf'] >= 1.0: pf_color = "text-gray-300"
        else: pf_color = "text-red-400"
        
        # Color coding for Compounded to highlight massive returns
        if st['compounded'] > 400: comp_color = "text-purple-500 font-black text-lg drop-shadow-md"
        elif st['compounded'] > 100: comp_color = "text-green-500 font-bold text-base"
        elif st['compounded'] > 0: comp_color = "text-green-400"
        else: comp_color = "text-red-400"
        
        html += f"""
          <tr class="hover:bg-gray-800 transition-colors">
            <td class="px-6 py-3 whitespace-nowrap text-gray-500 font-mono">#{idx+1}</td>
            <td class="px-6 py-3 font-mono text-xs text-blue-200">{st['name']}</td>
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

    out_path = r"C:\Users\preet\.gemini\antigravity\brain\e31e97f2-6b70-4ac3-bead-de744f595286\strategies_dashboard_v3_ml.html"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Created V3 ML dashboard at {out_path}")

if __name__ == '__main__':
    create_dashboard()
