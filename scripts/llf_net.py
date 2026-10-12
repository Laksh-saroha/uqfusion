"""LLF-Net: learned late fusion as a set transformer over every candidate box of a frame (GPU).

Second arm of `docs/prereg-llf-2026-10-10.md` (Amendment 1). Same candidates, 34 features, credit
target, training rows and leave-one-run-out folds as `scripts/llf.py`; instead of scoring each box
from hand-built pair features, every VIS and IR candidate of the frame attends to all the others, so
duplicate suppression and cross-stream trust are learned jointly. Features standardised on the fold's
training rows and clipped to ±10; linear 34 -> 128, three pre-norm transformer encoder layers (4
heads, feed-forward 256, dropout 0.1), linear head, sigmoid; BCE on the [0, 1] credit; AdamW lr 3e-4,
weight decay 1e-4, 64 frames per batch, 15 epochs, cosine schedule, bf16 autocast, seed 0. Fixed; no
tuning, no early stopping. Not monotone in confidence.

    py -3.13 scripts/llf_net.py fit              # the four leave-one-run-out models
    py -3.13 scripts/llf.py eval --variants net  # scored by llf.py on draws 942-944
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "llf"
RUNS = ("pohang00", "pohang01", "pohang02", "pohang03")
CFG = dict(d=128, layers=3, heads=4, ff=256, drop=0.1, lr=3e-4, wd=1e-4, batch=64, epochs=15, seed=0)


def _net(n_in: int):
    import torch.nn as nn

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_in, CFG["d"])
            layer = nn.TransformerEncoderLayer(CFG["d"], CFG["heads"], CFG["ff"], CFG["drop"],
                                               batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, CFG["layers"], enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(CFG["d"]), nn.Linear(CFG["d"], 1))

        def forward(self, x, pad):
            return self.out(self.enc(self.inp(x), src_key_padding_mask=pad)).squeeze(-1)
    return SetNet()


def _sets(meta: np.ndarray) -> np.ndarray:
    """(n_sets, 2) [start, end) of each frame's candidate set; rows are contiguous per frame."""
    key = meta[:, [0, 1, 2, 5]]
    cut = np.flatnonzero((np.diff(key, axis=0) != 0).any(1)) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(meta)]])
    return np.column_stack([starts, ends])


def _batch_index(sets: np.ndarray, sel: np.ndarray):
    """Padded row-index matrix (B, L) with -1 for padding."""
    lens = sets[sel, 1] - sets[sel, 0]
    L = int(lens.max())
    idx = np.full((len(sel), L), -1, dtype=np.int64)
    for b, (s, e) in enumerate(sets[sel]):
        idx[b, : e - s] = np.arange(s, e)
    return idx


def cmd_fit(args) -> None:
    import torch
    torch.manual_seed(CFG["seed"])
    dev = torch.device("cuda")
    parts = [np.load(p) for p in sorted((OUT / "train").glob("*.npz"))]
    X = np.concatenate([p["X"] for p in parts]).astype(np.float32)
    y = np.concatenate([p["y"] for p in parts]).astype(np.float32)
    meta = np.concatenate([p["meta"] for p in parts])
    sets = _sets(meta)
    set_run = meta[sets[:, 0], 3]
    (OUT / "net").mkdir(parents=True, exist_ok=True)
    print(f"[net] {len(X):,} rows in {len(sets):,} frame sets", flush=True)
    for r, run in enumerate(RUNS):
        out = OUT / "net" / f"loro_{run}.pt"
        if out.is_file() and not args.force:
            continue
        t0 = time.time()
        tr_sets = np.flatnonzero(set_run != r)
        rows = np.concatenate([np.arange(s, e) for s, e in sets[tr_sets]])
        mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6
        Xg = torch.from_numpy(np.clip((X - mu) / sd, -10, 10)).to(dev)
        Xg = torch.cat([Xg, torch.zeros(1, X.shape[1], device=dev)])         # row -1 = padding
        yg = torch.cat([torch.from_numpy(y).to(dev), torch.zeros(1, device=dev)])
        torch.manual_seed(CFG["seed"])
        rng = np.random.default_rng(CFG["seed"])
        net = _net(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
        steps = CFG["epochs"] * math.ceil(len(tr_sets) / CFG["batch"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=steps,
                                                    pct_start=0.05, anneal_strategy="cos")
        lossf = torch.nn.BCEWithLogitsLoss(reduction="sum")
        net.train()
        for ep in range(CFG["epochs"]):
            perm = rng.permutation(tr_sets)
            tot, n = 0.0, 0
            for b in range(0, len(perm), CFG["batch"]):
                idx = torch.from_numpy(_batch_index(sets, perm[b:b + CFG["batch"]])).to(dev)
                pad = idx < 0
                xb, yb = Xg[idx], yg[idx]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = net(xb, pad)
                loss = lossf(logit.float()[~pad], yb[~pad]) / (~pad).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss) * int((~pad).sum())
                n += int((~pad).sum())
            print(f"[net] loro_{run} epoch {ep + 1}/{CFG['epochs']} train BCE {tot / n:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "n_in": X.shape[1], "cfg": CFG,
                    "train_sets": int(len(tr_sets))}, out)
        print(f"[net] wrote {out} ({time.time() - t0:.0f}s)", flush=True)


_LOADED: dict = {}


def predict_frames(Xs: list, names: list, device: str = "cuda") -> list:
    """Per-frame credit predictions; frame i scored by net `names[i]` (e.g. 'loro_pohang00')."""
    import torch
    if not torch.cuda.is_available():
        device = "cpu"
    out = [np.zeros(len(x), dtype=np.float64) for x in Xs]
    lens = np.array([len(x) for x in Xs])
    for name in sorted(set(names)):
        if name not in _LOADED:
            ck = torch.load(OUT / "net" / f"{name}.pt", map_location=device, weights_only=False)
            net = _net(ck["n_in"]).to(device)
            net.load_state_dict(ck["state"])
            net.eval()
            _LOADED[name] = (net, ck["mu"], ck["sd"])
        net, mu, sd = _LOADED[name]
        fr = [i for i, n in enumerate(names) if n == name and lens[i]]
        for b in range(0, len(fr), 256):
            chunk = fr[b:b + 256]
            L = int(lens[chunk].max())
            xb = np.zeros((len(chunk), L, len(mu)), dtype=np.float32)
            pad = np.ones((len(chunk), L), dtype=bool)
            for j, i in enumerate(chunk):
                xb[j, : lens[i]] = np.clip((Xs[i] - mu) / sd, -10, 10)
                pad[j, : lens[i]] = False
            with torch.no_grad():
                p = torch.sigmoid(net(torch.from_numpy(xb).to(device), torch.from_numpy(pad).to(device)).float())
            p = p.cpu().numpy()
            for j, i in enumerate(chunk):
                out[i] = p[j, : lens[i]].astype(np.float64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cmd_fit(args)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
