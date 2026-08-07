import json
import statistics

f1s = [
    json.load(open(f"results/transformer_s{i}/summary.json"))["test"]["k10"]["macro_f1"]
    for i in range(3)
]
m = statistics.mean(f1s)
s = statistics.stdev(f1s)
print(f"seeds: {f1s} mean {m:.4f} std {s:.4f}")
assert abs(m - 0.607) < 0.005, "README mean off"
