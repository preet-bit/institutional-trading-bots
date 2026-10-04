import time, pickle, sys, numpy as np, pandas as pd
from core import *
from strategies import *

df = load()
print('bars', len(df), df.t.min(), df.t.max())
x = X(df)
o, h, l, c = [df[k].values.astype(np.float64) for k in ('o', 'h', 'l', 'c')]
a14 = x.atr(14).values.astype(np.float64)
WARM = 600   # ignore signals during indicator warm-up
res = {}     # (sid, pidx, sl, rr) -> (ei, xi, dir, gross, risk)
sig_counts = {}
t0 = time.time()
for st in STRATS:
    ts = time.time()
    for pi, p in enumerate(st['grid']):
        lg, sh = st['fn'](x, *p)
        sig = to_sig(lg, sh); sig[:WARM] = 0
        sig_counts[(st['id'], pi)] = int((sig != 0).sum())
        for sl in SL_GRID:
            for rr in RR_GRID:
                res[(st['id'], pi, sl, rr)] = run_bt(o, h, l, c, sig, a14, sl, rr, MAX_HOLD)
    print(st['id'], st['name'], f"{time.time()-ts:.1f}s", flush=True)
print('total', time.time() - t0)
pickle.dump(dict(res=res, sig_counts=sig_counts, t=df.t.values), open('results.pkl', 'wb'))
df.to_pickle('df.pkl')
