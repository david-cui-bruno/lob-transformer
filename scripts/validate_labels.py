"""Empirically validate label semantics: labels claimed UP must correspond to
rising future mid prices, DOWN to falling. Uses raw features + our label map.

Also validates alignment: the label at column t must describe the future
relative to timestamp t (not t+1 or t-1).
"""

import numpy as np

from lobt.data import HORIZONS, detect_segments, load_fi2010
from lobt.labels import DOWN, STATIONARY, UP

trx, tr_y, _, _ = load_fi2010("data")
segs = detect_segments(trx)

k = 50  # wide horizon: less stationary noise
y = tr_y[k]

for s, e in segs[:3]:
    x = trx[s:e]
    yy = y[s:e]
    # mid from best ask (col 0) and best bid (col 2), DecPre-normalized
    mid = (x[:, 0].astype(np.float64) + x[:, 2]) / 2
    n = len(mid)
    # future mean mid minus past mean mid, aligned at t (FI-2010 definition)
    rel = np.full(n, np.nan)
    for t in range(k, n - k):
        m_minus = mid[t - k + 1 : t + 1].mean()
        m_plus = mid[t + 1 : t + k + 1].mean()
        rel[t] = (m_plus - m_minus) / m_minus
    valid = ~np.isnan(rel)
    for cls, name in [(UP, "UP"), (STATIONARY, "STAT"), (DOWN, "DOWN")]:
        m = valid & (yy == cls)
        print(f"seg@{s}: {name}: mean rel change {np.nanmean(rel[m]):+.2e}  (n={m.sum()})")
    print()
