"""Aggregate results/*/summary.json into a markdown table."""

import json
import sys
from pathlib import Path

rows = []
for p in sorted(Path("results").glob("*/summary.json")):
    s = json.loads(p.read_text())
    rows.append(s)

if not rows:
    sys.exit("no summaries found")

horizons = [k for k in rows[0]["test"]]
print("| model | params | seed | " + " | ".join(f"test F1 {k}" for k in horizons) + " |")
print("|" + "---|" * (3 + len(horizons)))
for s in rows:
    name = Path(s["config"]["out"]).name
    cells = " | ".join(f"{s['test'][k]['macro_f1']:.4f}" for k in horizons)
    print(f"| {name} | {s['params']:,} | {s['seed']} | {cells} |")

print()
print("| model | " + " | ".join(f"val F1 {k}" for k in horizons) + " |")
print("|" + "---|" * (1 + len(horizons)))
for s in rows:
    name = Path(s["config"]["out"]).name
    cells = " | ".join(f"{s['val'][k]['macro_f1']:.4f}" for k in horizons)
    print(f"| {name} | {cells} |")
