#!/usr/bin/env python3
"""Verify every number claimed in README.md / docs/findings.md against the
committed results/*/summary.json artifacts. Exits nonzero on any mismatch.

Run: .venv/bin/python scripts/verify_claims.py  (or python3)
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "summary.json").read_text())


def k10(name: str) -> float:
    return load(name)["test"]["k10"]["macro_f1"]


checks: list[tuple[bool, str, object, object]] = []


def check(name: str, claim, actual) -> None:
    checks.append((claim == actual, name, claim, actual))


def main() -> None:
    tf15 = [k10(f"transformer_s{s}") for s in range(3)]
    tf60 = [k10(f"transformer_long_s{s}") for s in range(3)]
    cv60 = [k10(f"convtransformer_long_s{s}") for s in range(3)]

    # README results table
    check("logistic 0.267", 0.267, round(k10("logistic_s0"), 3))
    check("mlp 0.360", 0.360, round(k10("mlp_s0"), 3))
    check("tf 15ep 0.607±0.013",
          (0.607, 0.013),
          (round(statistics.mean(tf15), 3), round(statistics.stdev(tf15), 3)))
    check("tf 60ep 0.649±0.007",
          (0.649, 0.007),
          (round(statistics.mean(tf60), 3), round(statistics.stdev(tf60), 3)))
    check("conv 60ep 0.702±0.015",
          (0.702, 0.015),
          (round(statistics.mean(cv60), 3), round(statistics.stdev(cv60), 3)))
    check("stem delta +5.4 F1", 0.054,
          round(statistics.mean(cv60) - statistics.mean(tf60), 3))
    check("seed ranges non-overlapping", True, min(cv60) > max(tf60))
    check("conv range 0.685-0.712", (0.685, 0.712),
          (round(min(cv60), 3), round(max(cv60), 3)))
    check("tf range 0.643-0.656", (0.643, 0.656),
          (round(min(tf60), 3), round(max(tf60), 3)))

    # findings.md v1 + v2 tables
    check("v1 per-seed 0.609/0.619/0.592", [0.609, 0.619, 0.592],
          [round(x, 3) for x in tf15])
    check("tf60 per-seed 0.643/0.656/0.647", [0.643, 0.656, 0.647],
          [round(x, 3) for x in tf60])
    check("conv per-seed 0.712/0.685/0.711", [0.712, 0.685, 0.711],
          [round(x, 3) for x in cv60])
    check("undertrained conv 0.393", 0.393, round(k10("convtransformer_s0"), 3))
    check("conv k100 0.790/0.748/0.793", [0.790, 0.748, 0.793],
          [round(load(f"convtransformer_long_s{s}")["test"]["k100"]["macro_f1"], 3)
           for s in range(3)])

    # params
    check("tf params 210,063", 210063, load("transformer_long_s0")["params"])
    check("conv params 247,279", 247279, load("convtransformer_long_s0")["params"])

    fails = [c for c in checks if not c[0]]
    for ok, name, claim, actual in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  claimed={claim}  actual={actual}")
    print(f"\n{len(checks) - len(fails)}/{len(checks)} claims verified")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
