import numpy as np
from lobt.data import load_fi2010

trx, tr_y, tex, te_y = load_fi2010("data")


def segments(x, thresh=0.2):
    a = x[:, 0].astype(np.float64)
    b = np.where(np.abs(np.diff(a)) / a[:-1] > thresh)[0]
    edges = [0, *(b + 1), len(x)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


tr_seg = segments(trx)
te_seg = segments(tex)
print("train segments:", len(tr_seg))
for i, (s, e) in enumerate(tr_seg):
    print(f"  tr seg{i}: len={e-s} scale={trx[s:e,0].mean():.4f}")
print("test segments:", len(te_seg))
for i, (s, e) in enumerate(te_seg):
    print(f"  te seg{i} (day {i//5}, pos {i%5}): len={e-s} scale={tex[s:e,0].mean():.4f}")
