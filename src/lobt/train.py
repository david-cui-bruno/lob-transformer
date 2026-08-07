"""Training and evaluation for LOB direction prediction.

Usage:
    python -m lobt.train --model transformer --window 100 --out results/transformer_s0
    python -m lobt.train --model logistic --epochs 5 ...

Writes: <out>/metrics.jsonl (per-eval metrics), <out>/best.pt (val-best
checkpoint), <out>/summary.json (final val + test metrics, config, params).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import HORIZONS, load_fi2010, train_val_split
from .datasets import WindowedLOBDataset, class_weights
from .models import build_model, param_count

N_CLASSES = 3


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def macro_f1(conf: np.ndarray) -> float:
    """Macro F1 from a (C, C) confusion matrix (rows=true, cols=pred)."""
    f1s = []
    for c in range(conf.shape[0]):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    """Confusion matrices and metrics per horizon."""
    model.eval()
    confs = np.zeros((len(HORIZONS), N_CLASSES, N_CLASSES), dtype=np.int64)
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        logits = model(xb)  # (b, H, C)
        preds = logits.argmax(dim=-1).cpu().numpy()
        y = yb.numpy()
        for h in range(len(HORIZONS)):
            np.add.at(confs[h], (y[:, h], preds[:, h]), 1)
    out = {}
    for i, k in enumerate(HORIZONS):
        conf = confs[i]
        out[f"k{k}"] = {
            "macro_f1": round(macro_f1(conf), 4),
            "accuracy": round(float(np.trace(conf) / conf.sum()), 4),
            "confusion": conf.tolist(),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["logistic", "mlp", "transformer"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=2000, help="steps between evals")
    p.add_argument("--patience", type=int, default=5, help="evals without val improvement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select-horizon", type=int, default=10)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--shuffle-features", action="store_true",
                   help="ablation: fixed random permutation of the 40 features")
    p.add_argument("--max-steps", type=int, default=0, help="0 = unlimited")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()

    trx, tr_y, tex, te_y = load_fi2010(args.data_dir)
    trx, tr_y, vx, v_y = train_val_split(trx, tr_y, val_frac=0.1)

    if args.shuffle_features:
        perm = np.random.default_rng(1234).permutation(trx.shape[1])
        trx, vx, tex = trx[:, perm], vx[:, perm], tex[:, perm]

    ds_tr = WindowedLOBDataset(trx, tr_y, window=args.window, horizons=HORIZONS)
    ds_v = WindowedLOBDataset(vx, v_y, window=args.window, horizons=HORIZONS)
    ds_te = WindowedLOBDataset(tex, te_y, window=args.window, horizons=HORIZONS)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_v = DataLoader(ds_v, batch_size=1024)
    dl_te = DataLoader(ds_te, batch_size=1024)

    model = build_model(
        args.model, args.window, trx.shape[1], len(HORIZONS),
        **({"d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "d_ff": args.d_ff}
           if args.model == "transformer" else {}),
    ).to(dev)
    n_params = param_count(model)

    weights = torch.stack([class_weights(tr_y[k]) for k in HORIZONS]).to(dev)
    losses = [nn.CrossEntropyLoss(weight=weights[i]) for i in range(len(HORIZONS))]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.max_steps or args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / max(args.warmup_steps, 1),
                      0.5 * (1 + np.cos(np.pi * min(s / total_steps, 1.0)))),
    )

    sel = f"k{args.select_horizon}"
    best_val, best_state, bad_evals, step = -1.0, None, 0, 0
    t0 = time.time()
    metrics_f = (out_dir / "metrics.jsonl").open("w")
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        for xb, yb in dl_tr:
            model.train()
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            logits = model(xb)
            loss = sum(losses[h](logits[:, h], yb[:, h]) for h in range(len(HORIZONS)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % args.eval_every == 0 or (args.max_steps and step >= args.max_steps):
                val = evaluate(model, dl_v, dev)
                rec = {"step": step, "epoch": epoch, "loss": round(loss.item(), 4),
                       "elapsed_s": round(time.time() - t0, 1), "val": val}
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                vf1 = val[sel]["macro_f1"]
                print(f"JCODE_PROGRESS {json.dumps({'message': f'step {step} val {sel} F1 {vf1:.4f}', 'current': step, 'total': total_steps})}", flush=True)
                if vf1 > best_val:
                    best_val, bad_evals = vf1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_state, out_dir / "best.pt")
                else:
                    bad_evals += 1
                    if bad_evals >= args.patience:
                        print(f"early stop at step {step} (patience {args.patience})", flush=True)
                        stop = True
                        break
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, dl_v, dev)
    test = evaluate(model, dl_te, dev)
    summary = {
        "model": args.model, "params": n_params, "window": args.window,
        "seed": args.seed, "steps": step, "elapsed_s": round(time.time() - t0, 1),
        "config": vars(args), "val": final_val, "test": test,
        "train_windows": len(ds_tr), "val_windows": len(ds_v), "test_windows": len(ds_te),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    metrics_f.close()
    print(json.dumps({k: test[k]["macro_f1"] for k in test}, indent=2))
    print(f"done: {n_params} params, best val {sel} F1 {best_val:.4f}, "
          f"test {sel} F1 {test[sel]['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
