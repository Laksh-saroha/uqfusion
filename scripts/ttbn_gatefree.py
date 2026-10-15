"""Gate-free BatchNorm test-time adaptation, unregistered development screen (2026-10-15).

Can BN-mix (IG-BN's adaptation: N = 16, non-overlapping 8-frame blocks of the stream's frame list) run on
EVERY frame without hurting clean or night frames? Ungated BN-mix costs −0.0101 on clean day and −0.0195
on clean night (gate ablation, `docs/eval/ttbn/gate_ablation_dev_2026-10-14.md`). Three families, all
gradient-free and label-free at test time, each run on every block (no gate):

1. Domain-matched source. Population BN statistics of the frozen detector on training-split day frames
   and on night frames (pohang01). Each block is passed twice, BN-mixed (N = 16) toward the day source and
   toward the night source; the pass whose summed per-layer symmetric KL (block vs that source, channel
   Gaussians) is smaller is kept (`dom_mix_near`). `dom_mix_oracle` keeps the pass of the block's true
   domain (majority of its frames; a 4/4 tie counts as day). `dom_swap_*`: the same with N = inf (the
   domain statistics alone, no block statistics).
2. Shift-scaled prior. Per layer, D = mean channel symmetric KL of the block against the model's running
   statistics. `shift_exp`: N_eff = 16 exp(D0/D), D0 = median D over clean training blocks. `shift_hinge`:
   weight n/(16+n) * max(0, 1 − D0/D), D0 = 90th percentile over clean training blocks (no adaptation
   below a typical clean block's shift). `dom_shift_hinge`: the hinge toward `dom_mix_near`'s domain
   source, D0 = that domain's 90th percentile.
3. Shallow only. BN-mix (N = 16) on the first k BatchNorm layers (k = 2: stem; 11: through stride 4;
   21: through stride 8; 31: through stride 16), source statistics after.

`mix16` (ungated BN-mix, every layer) is rebuilt for reference; IG-BN on this draw is `mix16`'s records
where IG-BN's gate fires (a block's statistics depend only on its frames, as the gate ablation showed).
Training frames only for every constant; no pohang04 frame is read.

    py -3.13 scripts/ttbn_gatefree.py calib                     # source statistics + D calibration
    py -3.13 scripts/ttbn_gatefree.py build --draw 941
    py -3.13 scripts/ttbn_gatefree.py score --draw 941 && py -3.13 scripts/ttbn_gatefree.py report --draw 941
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import ttbn  # noqa: E402

OUT = ROOT / "runs/novel/ttbn_gatefree"
TRAIN_LIST = ROOT / "runs/derived/data_vis_train_stride2_p04out.txt"
HELD_LISTS = (ROOT / "runs/derived/vis_val_p04out.txt", ROOT / "runs/derived/vis_test_p04out.txt", ttbn.DEV_LIST)
NIGHT_RUN = "pohang01"
N_SRC = 1600            # training frames per domain for the source statistics
SRC_BATCH = 16
N_CAL = 120             # 8-frame training blocks per domain for the D calibration
N0 = ttbn.PRIOR_N
SHALLOW = (2, 11, 21, 31)
PASSES = ("mix16", "dmix_day", "dmix_night", "dswap_day", "dswap_night", "shift_exp", "shift_hinge",
          *(f"shallow{k}" for k in SHALLOW), "dom_shift_hinge")
ARMS = ("mix16", "dom_mix_near", "dom_mix_oracle", "dom_swap_near", "dom_swap_oracle", "shift_exp",
        "shift_hinge", "dom_shift_hinge", *(f"shallow{k}" for k in SHALLOW))
#: second quick set (2026-10-15, after the first subset run): shallow + hinge combinations
SHALLOW2 = (21, 31)
ARMS2 = ("mix16", "hinge50", *(f"sh{k}_hinge{q}" for k in SHALLOW2 for q in (90, 50)),
         *(f"dom_sh{k}_hinge" for k in SHALLOW2))
#: third set (2026-10-15): the two median-hinge shallow arms, full 60-cell build
ARMS3 = ("mix16", "sh21_hinge50", "sh31_hinge50")
#: registered (docs/prereg-gatefree-ttbn-2026-10-16.md): GF31 primary, GF21 secondary, development draws 942-944
ARMS4 = ("sh31_hinge50", "sh21_hinge50")
CFG: dict = {"c": None}
RUN = {"dir": OUT, "arms": ARMS}      # arms, diagnostics, scores; OUT/quick for a subset run (--every > 1)


# ------------------------------------------------------------------ the BatchNorm patch
def symkl(mt, vt, ms, vs, eps):
    vt, vs = vt + eps, vs + eps
    return 0.5 * ((vt / vs + vs / vt - 2.0) + (mt - ms) ** 2 * (1.0 / vt + 1.0 / vs)).mean()


def patch(model, j: int) -> int:
    """BatchNorm2d layer i of model j follows CFG['c'] (None: plain source statistics)."""
    import torch
    import torch.nn.functional as F
    i = 0
    for m in model.modules():
        if not isinstance(m, torch.nn.BatchNorm2d):
            continue

        def fwd(x, m=m, i=i):
            c = CFG["c"]
            if c is None:
                return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)
            xf = x.float()
            mu_t = xf.mean(dim=(0, 2, 3))
            var_t = xf.var(dim=(0, 2, 3), unbiased=False)
            n = x.shape[0]
            kind = c["kind"]
            if kind == "collect":                       # source statistics: normalise by the large random batch
                acc = c["acc"][j][i]
                acc[0] += mu_t.double() * n
                acc[1] += (var_t + mu_t ** 2).double() * n
                acc[2] += n
                return F.batch_norm(x, mu_t, var_t, m.weight, m.bias, False, 0.0, m.eps)
            if i >= c.get("kmax", 10 ** 9):
                return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)
            src = c["src"][j] if c.get("src") is not None else None
            mu_s, var_s = (src[0][i], src[1][i]) if src is not None else (m.running_mean, m.running_var)
            D = symkl(mu_t, var_t, mu_s, var_s, m.eps)
            c["d"][j][i] = D
            a0 = n / (c["N"] + n)
            if kind == "plain":                          # normalise with the source; record D only
                a = 0.0
            elif kind == "mix":
                a = a0
            elif kind == "exp":
                a = n / (c["N"] * torch.exp(torch.clamp(c["D0"][j][i] / D, max=30.0)) + n)
            elif kind == "hinge":
                a = a0 * torch.clamp(1.0 - c["D0"][j][i] / D, min=0.0)
            else:
                raise ValueError(kind)
            c["a"][j][i] = a
            if isinstance(a, float) and a == 0.0:
                return F.batch_norm(x, mu_s, var_s, m.weight, m.bias, False, 0.0, m.eps)
            mu = (1 - a) * mu_s + a * mu_t
            var = (1 - a) * var_s + a * var_t
            return F.batch_norm(x, mu, var, m.weight, m.bias, False, 0.0, m.eps)
        m.forward = fwd
        i += 1
    return i


def _preds():
    from uqfusion.uq.infer import UQPredictor
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    nb = [patch(p.model, j) for j, p in enumerate(preds)]
    assert len(set(nb)) == 1
    return preds, nb[0]


def _tensor(preds, ims):
    import torch
    p0 = preds[0]
    ts = [torch.from_numpy(np.ascontiguousarray(p0._letterbox(image=im)[..., ::-1].transpose(2, 0, 1))).float()
          .div_(255.0) for im in ims]
    return torch.stack(ts).to(p0.device)


def _forward(preds, ims, t, decode=True):
    """`p04_heldout._multi_batch` on a prebuilt letterbox tensor (its code, unchanged otherwise)."""
    import torch
    from ultralytics.utils import ops
    outs = []
    for p in preds:
        with torch.no_grad():
            out = p.model(t)
        if not decode:
            continue
        y = out[0] if isinstance(out, (tuple, list)) else out
        recs = []
        for im0, yi in zip(ims, y):
            det = yi[yi[:, 4] >= p.conf]
            h0, w0 = im0.shape[:2]
            boxes = det[:, :4].clone()
            if boxes.shape[0]:
                boxes = ops.scale_boxes(t.shape[2:], boxes, im0.shape)
            gain = min(t.shape[2] / h0, t.shape[3] / w0)
            det_c = det.cpu()
            recs.append({"boxes_xyxy": boxes.cpu().numpy(), "conf": det_c[:, 4].numpy(),
                         "cls": det_c[:, 5].numpy().astype(int), "image_hw": (h0, w0),
                         "sigma_ltrb": (det_c[:, 6:10] / gain).numpy()})
        outs.append(recs)
    return outs


def _run(preds, ims, t, nl, decode=True, **cfg):
    """One pass under `cfg`; returns records and per-model D (sum over layers), mean weight per layer."""
    import torch
    dev = preds[0].device
    c = {"d": [torch.zeros(nl, device=dev) for _ in preds], "a": [torch.zeros(nl, device=dev) for _ in preds],
         "N": N0, **cfg}
    CFG["c"] = c
    try:
        recs = _forward(preds, ims, t, decode)
    finally:
        CFG["c"] = None
    return recs, torch.stack(c["d"]).cpu().numpy(), torch.stack(c["a"]).cpu().numpy()


# ------------------------------------------------------------------ calibration (training frames only)
def _runcam(p: str) -> tuple[str, str, int]:
    r, cam, num = re.search(r"(pohang\d+)_([LR])_(\d+)", p).groups()
    return r, cam, int(num)


def _cal_blocks(rng) -> dict:
    """8 consecutive raw frames (gaps of at most 2 frame numbers) starting at a training-list frame, none of
    them in the val, test or development lists; per domain."""
    train = [ln.strip() for ln in open(TRAIN_LIST) if ln.strip()]
    held = {Path(ln.strip()).name for f in HELD_LISTS for ln in open(f) if ln.strip()}
    by_dir: dict = {}
    out = {"day": [], "night": []}
    for dom in ("day", "night"):
        cand = [p for p in train if (_runcam(p)[0] == NIGHT_RUN) == (dom == "night")]
        for s in rng.permutation(len(cand)):
            p = Path(cand[s])
            if p.parent not in by_dir:
                fs = sorted((x for x in os.listdir(p.parent) if re.search(r"pohang\d+_[LR]_\d+\.png$", x)),
                            key=lambda x: (_runcam(x)[1], _runcam(x)[2]))
                by_dir[p.parent] = (fs, {x: n for n, x in enumerate(fs)})
            fs, pos = by_dir[p.parent]
            a = pos[p.name]
            blk = fs[a:a + ttbn.BATCH]
            if len(blk) < ttbn.BATCH or any(_runcam(x)[1] != _runcam(p.name)[1] for x in blk):
                continue
            nums = [_runcam(x)[2] for x in blk]
            if max(np.diff(nums)) > 2 or any(x in held for x in blk):
                continue
            out[dom].append([str(p.parent / x) for x in blk])
            if len(out[dom]) == N_CAL:
                break
    return out


def cmd_calib(args) -> int:
    import torch
    ttbn._low()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    train = [ln.strip() for ln in open(TRAIN_LIST) if ln.strip()]
    sets = {dom: [train[i] for i in rng.permutation([i for i, p in enumerate(train)
                                                      if (_runcam(p)[0] == NIGHT_RUN) == (dom == "night")])[:N_SRC]]
            for dom in ("day", "night")}
    blocks = _cal_blocks(rng)
    preds, nl = _preds()
    dev = preds[0].device
    t0 = time.time()
    src = {}
    for dom, lst in sets.items():
        acc = [[[torch.zeros(m.num_features, dtype=torch.float64, device=dev),
                 torch.zeros(m.num_features, dtype=torch.float64, device=dev), 0]
                for m in p.model.modules() if isinstance(m, torch.nn.BatchNorm2d)] for p in preds]
        chunks = [list(range(a, min(a + SRC_BATCH, len(lst)))) for a in range(0, len(lst), SRC_BATCH)]
        for ch in ttbn._frames({"kind": None, "sev": 0, "seed": 0, "images": lst}, chunks, args.workers):
            ims = [im for _, im in ch]
            CFG["c"] = {"kind": "collect", "acc": acc}
            try:
                _forward(preds, ims, _tensor(preds, ims), decode=False)
            finally:
                CFG["c"] = None
        src[dom] = []
        for j in range(len(preds)):
            mu = [(a[0] / a[2]).float().contiguous() for a in acc[j]]
            var = [(a[1] / a[2] - (a[0] / a[2]) ** 2).clamp_min(0).float().contiguous() for a in acc[j]]
            src[dom].append((mu, var))
        print(f"[calib] {dom} source statistics from {len(lst)} frames ({time.time() - t0:.0f}s)", flush=True)
    # D on clean training blocks: against the running statistics, and against each domain's statistics
    D = {k: {"day": [], "night": []} for k in ("run", "swap_day", "swap_night", "mix_day", "mix_night")}
    for dom, bl in blocks.items():
        flat = [p for b in bl for p in b]
        chunks = [list(range(a, a + ttbn.BATCH)) for a in range(0, len(flat), ttbn.BATCH)]
        for ch in ttbn._frames({"kind": None, "sev": 0, "seed": 0, "images": flat}, chunks, args.workers):
            ims = [im for _, im in ch]
            t = _tensor(preds, ims)
            D["run"][dom].append(_run(preds, ims, t, nl, False, kind="plain")[1])
            for sd in ("day", "night"):
                D[f"swap_{sd}"][dom].append(_run(preds, ims, t, nl, False, kind="plain", src=src[sd])[1])
                D[f"mix_{sd}"][dom].append(_run(preds, ims, t, nl, False, kind="mix", src=src[sd])[1])
        print(f"[calib] {dom}: {len(bl)} blocks ({time.time() - t0:.0f}s)", flush=True)
    D = {k: {d: np.stack(v) for d, v in x.items()} for k, x in D.items()}       # (blocks, models, layers)
    allrun = np.concatenate([D["run"]["day"], D["run"]["night"]])
    cal = {"D0_exp": np.median(allrun, axis=0), "D0_hinge": np.quantile(allrun, 0.9, axis=0),
           "D0_day": np.quantile(D["swap_day"]["day"], 0.9, axis=0),
           "D0_night": np.quantile(D["swap_night"]["night"], 0.9, axis=0)}
    sel = {}
    for kind in ("swap", "mix"):
        for dom in ("day", "night"):
            dd, dn = D[f"{kind}_day"][dom].sum(-1), D[f"{kind}_night"][dom].sum(-1)      # (blocks, models)
            sel[f"{kind}|{dom}"] = float(((dn < dd) == (dom == "night")).mean())
    torch.save({"src": {d: [([x.cpu() for x in mu], [x.cpu() for x in v]) for mu, v in s] for d, s in src.items()},
                "cal": cal, "D": D, "blocks": blocks, "sets": sets, "selector_accuracy": sel},
               OUT / "calib.pt")
    print("[calib] nearest-domain selector accuracy on training blocks:", json.dumps(sel), flush=True)
    print("[calib] median D per layer (run, first 12):", np.round(cal["D0_exp"].mean(0)[:12], 4), flush=True)
    return 0


def _load_calib(dev):
    import torch
    z = torch.load(OUT / "calib.pt", weights_only=False)
    src = {d: [([x.to(dev) for x in mu], [x.to(dev) for x in v]) for mu, v in s] for d, s in z["src"].items()}
    cal = {k: [torch.as_tensor(v[j], dtype=torch.float32, device=dev) for j in range(v.shape[0])]
           for k, v in z["cal"].items()}
    return src, cal


# ------------------------------------------------------------------ build (all frames, every arm)
def arm_path(draw: int, k: int, arm: str, cond: str) -> Path:
    where = "clean" if (cond == "clean" and RUN.get("clean_shared")) else ttbn.sub(draw)
    return RUN["dir"] / "arms" / f"seed{k}" / where / f"{arm}_{cond}.pkl"


def cmd_build(args) -> int:
    from uqfusion.eval.corruptions_v2 import CODE_REV
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    night = np.array([_runcam(p)[0] == NIGHT_RUN for p in lst])
    conds = [c for c in (args.conds or ttbn.DEV_CONDS)
             if not all(arm_path(args.draw, k, a, c).is_file() for k in ttbn.SEEDS for a in RUN["arms"])]
    print(f"[build] draw {args.draw}: {len(lst)} frames, conditions {conds}", flush=True)
    if not conds:
        return 0
    preds, nl = _preds()
    src, cal = _load_calib(preds[0].device)
    nm = len(preds)
    for c in conds:
        kc = ttbn.DEV_CONDS[c]
        chunks = [list(range(a, min(a + ttbn.BATCH, len(lst)))) for a in range(0, len(lst), ttbn.BATCH)][::args.every]
        recs = {a: [[None] * len(lst) for _ in range(nm)] for a in ARMS}
        diag = {"dsum": {p: [] for p in PASSES}, "amean": {p: [] for p in PASSES}, "near_mix": [], "near_swap": [],
                "oracle_night": [], "block_start": []}
        spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": args.draw, "images": lst}
        t0 = time.time()
        if args.set >= 2:
            _build2(preds, nl, src, cal, lst, night, c, kc, chunks, args)
            continue
        for nb, ch in enumerate(ttbn._frames(spec, chunks, args.workers), 1):
            idx = [i for i, _ in ch]
            ims = [im for _, im in ch]
            t = _tensor(preds, ims)
            on = bool(night[idx].sum() * 2 > len(idx))             # majority; a 4/4 tie counts as day
            out = {}

            def go(name, **cfg):
                r, d, a = _run(preds, ims, t, nl, **cfg)
                out[name] = r
                diag["dsum"][name].append(d.sum(-1).tolist())
                diag["amean"][name].append(a.mean(-1).tolist())

            go("mix16", kind="mix")
            go("dmix_day", kind="mix", src=src["day"])
            go("dmix_night", kind="mix", src=src["night"])
            go("dswap_day", kind="plain", src=src["day"])
            go("dswap_night", kind="plain", src=src["night"])
            go("shift_exp", kind="exp", D0=cal["D0_exp"])
            go("shift_hinge", kind="hinge", D0=cal["D0_hinge"])
            for k in SHALLOW:
                go(f"shallow{k}", kind="mix", kmax=k)
            near_mix = [bool(diag["dsum"]["dmix_night"][-1][j] < diag["dsum"]["dmix_day"][-1][j]) for j in range(nm)]
            near_swap = [bool(diag["dsum"]["dswap_night"][-1][j] < diag["dsum"]["dswap_day"][-1][j]) for j in range(nm)]
            go("dom_shift_hinge", kind="hinge", src=[src["night" if near_mix[j] else "day"][j] for j in range(nm)],
               D0=[cal["D0_night" if near_mix[j] else "D0_day"][j] for j in range(nm)])
            diag["near_mix"].append(near_mix)
            diag["near_swap"].append(near_swap)
            diag["oracle_night"].append(on)
            diag["block_start"].append(idx[0])
            pick = {"mix16": lambda j: out["mix16"][j],
                    "dom_mix_near": lambda j: out["dmix_night" if near_mix[j] else "dmix_day"][j],
                    "dom_mix_oracle": lambda j: out["dmix_night" if on else "dmix_day"][j],
                    "dom_swap_near": lambda j: out["dswap_night" if near_swap[j] else "dswap_day"][j],
                    "dom_swap_oracle": lambda j: out["dswap_night" if on else "dswap_day"][j],
                    **{a: (lambda j, a=a: out[a][j]) for a in ("shift_exp", "shift_hinge", "dom_shift_hinge",
                                                                *(f"shallow{k}" for k in SHALLOW))}}
            for a in ARMS:
                for j in range(nm):
                    for i, r in zip(idx, pick[a](j)):
                        recs[a][j][i] = {**r, "image_path": lst[i]}
            if nb % 25 == 0:
                print(f"[build] {c}/{args.draw} {nb}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
        for a in ARMS:
            for j, k in enumerate(ttbn.SEEDS):
                o = arm_path(args.draw, k, a, c)
                o.parent.mkdir(parents=True, exist_ok=True)
                meta = {"arm": a, "cond": c, "draw": args.draw, "prior_N": N0, "batch": ttbn.BATCH,
                        "blocks": "non-overlapping 8-frame blocks of the frame list, every block (no gate)",
                        "precision": "fp32", "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                        "corrupt_seed": args.draw if kc else None, "corrupt_version": "v2" if kc else None,
                        "corrupt_code": CODE_REV if kc else None,
                        "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
                with open(o.with_suffix(".tmp"), "wb") as fh:
                    pickle.dump({"meta": meta, "records": recs[a][j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(o.with_suffix(".tmp"), o)
        dp = RUN["dir"] / "diag" / ttbn.sub(args.draw) / f"{c}.json"
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps(diag), encoding="utf-8")
        print(f"[build] {c}/{args.draw} done: {len(chunks)} blocks, {time.time() - t0:.0f}s", flush=True)
    return 0



def _build2(preds, nl, src, cal, lst, night, c, kc, chunks, args) -> None:
    """Second quick set: hinge (D0 = 90th percentile or median of clean training blocks) on the first k
    BatchNorm layers, the running statistics after; and the nearest-domain hinge on the first k layers."""
    from uqfusion.eval.corruptions_v2 import CODE_REV
    nm = len(preds)
    arms = RUN["arms"]
    recs = {a: [[None] * len(lst) for _ in range(nm)] for a in arms}
    diag = {"amean": {a: [] for a in arms}, "near_mix": [], "oracle_night": [], "block_start": []}
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": args.draw, "images": lst}
    t0 = time.time()
    for nb, ch in enumerate(ttbn._frames(spec, chunks, args.workers), 1):
        idx = [i for i, _ in ch]
        ims = [im for _, im in ch]
        t = _tensor(preds, ims)
        out = {}

        def go(name, **cfg):
            if name not in arms:
                return None
            r, d, a = _run(preds, ims, t, nl, **cfg)
            out[name] = r
            diag["amean"][name].append(a.mean(-1).tolist())
            return d

        go("mix16", kind="mix")
        go("hinge50", kind="hinge", D0=cal["D0_exp"])
        for k in SHALLOW2:
            go(f"sh{k}_hinge90", kind="hinge", D0=cal["D0_hinge"], kmax=k)
            go(f"sh{k}_hinge50", kind="hinge", D0=cal["D0_exp"], kmax=k)
        # the nearest-domain choice, as `dom_mix_near` makes it (two full BN-mix passes, outputs not kept)
        near = [False] * nm
        if any(a.startswith("dom_") for a in arms):
            dd = _run(preds, ims, t, nl, False, kind="mix", src=src["day"])[1].sum(-1)
            dn = _run(preds, ims, t, nl, False, kind="mix", src=src["night"])[1].sum(-1)
            near = [bool(dn[j] < dd[j]) for j in range(nm)]
        for k in SHALLOW2:
            go(f"dom_sh{k}_hinge", kind="hinge", kmax=k, src=[src["night" if near[j] else "day"][j] for j in range(nm)],
               D0=[cal["D0_night" if near[j] else "D0_day"][j] for j in range(nm)])
        diag["near_mix"].append(near)
        diag["oracle_night"].append(bool(night[idx].sum() * 2 > len(idx)))
        diag["block_start"].append(idx[0])
        for a in arms:
            for j in range(nm):
                for i, r in zip(idx, out[a][j]):
                    recs[a][j][i] = {**r, "image_path": lst[i]}
        if nb % 25 == 0:
            print(f"[build2] {c}/{args.draw} {nb}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    for a in arms:
        for j, k in enumerate(ttbn.SEEDS):
            o = arm_path(args.draw, k, a, c)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"arm": a, "set": args.set, "cond": c, "draw": args.draw, "prior_N": N0, "batch": ttbn.BATCH,
                    "blocks": (f"every {args.every}th 8-frame block of the frame list (subset run), no gate" if args.every > 1 else "every 8-frame block of the frame list, no gate"),
                    "precision": "fp32", "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": args.draw if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[a][j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
    dp = RUN["dir"] / "diag" / ttbn.sub(args.draw) / f"{c}.json"
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_text(json.dumps(diag), encoding="utf-8")
    print(f"[build2] {c}/{args.draw} done: {len(chunks)} blocks, {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ scoring
def _arm(draw, k, a, c):
    with open(arm_path(draw, k, a, c), "rb") as fh:
        return pickle.load(fh)["records"]


def _set_run(d: str, arms, shared=False) -> None:
    RUN["dir"], RUN["arms"], RUN["clean_shared"] = Path(d), tuple(arms), shared


def _job(spec):
    import dataclasses
    import seed_ensemble as se
    from gate_ablation import select
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond, k = spec
    ttbn._low()
    se._memo_distances()
    d = f"{se.STATS}/brightness/{ttbn.sub(draw)}"
    conds = tuple(c for c in ttbn.DEV_CONDS if arm_path(draw, k, RUN["arms"][0], c).is_file())
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{ttbn.sub(draw)}", bright_dir=d,
                       structure_dir=f"{se.STATS}/structure/{ttbn.sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=conds, ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    u = dataclasses.replace(ctx, veto=None)
    mix0 = _arm(draw, k, RUN["arms"][0], "clean")
    ones = np.array([r is not None for r in mix0])           # every frame, or the subset built
    sl = {s: np.intersect1d(idx, np.flatnonzero(ones)) for s, idx in sl.items()}
    res = {}
    for c in conds:
        vis = ctx.vis_by_cond[c]
        g = ttbn.gate_ctx(ctx, c)
        o0 = run_systems(u, c, compute_metrics=False)
        parts = {"vis": frame_parts(vis, ctx.gts), "union": frame_parts(o0["fused_gated"], ctx.gts)}
        if RUN.get("clean_shared"):     # registered run: IG-BN = the gate ablation's complete N4 streams, N4's gate
            from gate_ablation import full_stream
            mix = full_stream("dev", k, c, None if c == "clean" else draw)
        else:
            mix = _arm(draw, k, "mix16", c)
        streams = {"igbn": select(mix, vis, g & ones), **{a: select(_arm(draw, k, a, c), vis, ones) for a in RUN["arms"]}}
        for a, vg in streams.items():
            parts[f"vis_{a}"] = frame_parts(vg, ctx.gts)
            parts[f"union_{a}"] = (parts["union"] if a == "igbn" and not g.any() else
                                   frame_parts(run_systems(u, c, compute_metrics=False, vis_records=vg)["fused_gated"],
                                               ctx.gts))
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate"] = float(g[idx].mean())
    return spec, res


def cmd_score(args) -> int:
    out_p = RUN["dir"] / ("score.json" if args.set == 4 else f"score_{args.draw}.json")
    allres = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    draws = ttbn.DEV_DRAWS if args.set == 4 else (args.draw,)
    specs = [(d, ic, k) for d in draws for ic in (ttbn.DEV_IR_CONDS if args.every == 1 else (None,)) for k in ttbn.SEEDS
             if f"{d}|{ic}|{k}" not in allres]
    print(f"[score] {len(specs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1, initializer=_set_run,
                             initargs=(str(RUN['dir']), RUN['arms'], bool(RUN.get('clean_shared')))) as ex:
        for n, (spec, res) in enumerate(ex.map(_job, specs), 1):
            allres["|".join(map(str, spec))] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(allres), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[score] {n}/{len(specs)} {spec} ({time.time() - t0:.0f}s)", flush=True)
    return 0


NAMES = {"igbn": "IG-BN (gated mix16, reference)", "mix16": "ungated BN-mix N=16 (reference)",
         "dom_mix_near": "C1 domain source, nearest", "dom_mix_oracle": "C1 domain source, oracle",
         "dom_swap_near": "C1' domain statistics alone, nearest", "dom_swap_oracle": "C1' domain statistics alone, oracle",
         "shift_exp": "C2 shift-scaled, N·exp(D0/D)", "shift_hinge": "C2 shift-scaled, hinge (1−D0/D)+",
         "dom_shift_hinge": "C1+C2 nearest domain, hinge", "hinge50": "C2 hinge, D0 = median",
         **{f"sh{k}_hinge{q}": f"C3+C2 first {k} BN, hinge D0 = {'p90' if q == 90 else 'median'}" for k in SHALLOW2 for q in (90, 50)},
         **{f"dom_sh{k}_hinge": f"C1+C2+C3 nearest domain, hinge, first {k} BN" for k in SHALLOW2}, **{f"shallow{k}": f"C3 shallow, first {k} BN" for k in SHALLOW}}
DEGRADED = ("fog", "fog_s1", "lowlight", "glare")
KEY = ("clean/clean/day", "clean/clean/night", "lowlight/clean/night", "lowlight/noise_s2/night", "lowlight/clean/day",
       "fog_s1/clean/day", "fog/clean/day", "glare/clean/day")


def cmd_report(args) -> int:
    res = json.loads((RUN["dir"] / f"score_{args.draw}.json").read_text(encoding="utf-8"))
    arms = ("igbn",) + RUN["arms"]
    ks = {key.split("|")[1] for key in res}
    ics = [ic for ic in ttbn.DEV_IR_CONDS if str(ic) in ks]
    cs = [c for c in ttbn.DEV_CONDS if f"{c}|day" in next(iter(res.values()))]
    cells = [(c, ic, s) for c in cs for ic in ics for s in ("day", "night")]
    tab, summ = {}, {}
    for a in arms:
        tab[a] = {}
        for c, ic, s in cells:
            rs = [res[f"{args.draw}|{ic}|{k}"][f"{c}|{s}"] for k in ttbn.SEEDS]
            d = ttbn._cell_stats([r[f"union_{a}"] - r["union"] for r in rs])
            dg = ttbn._cell_stats([r[f"union_{a}"] - r["union_igbn"] for r in rs])
            tab[a][f"{c}/{ic or 'clean'}/{s}"] = {"d": d, "d_igbn": dg, "union": float(np.mean([r["union"] for r in rs]))}
        v = tab[a]
        worse = [k for k, x in v.items() if x["d"][0] <= -0.0014 and x["d"][2] < 0]
        better = [k for k, x in v.items() if x["d"][0] >= 0.0060 and x["d"][1] > 0]
        deg = [k for k in v if k.split("/")[0] in DEGRADED and k.endswith("/day")]
        bg = [k for k in deg if v[k]["d_igbn"][0] >= 0.0060 and v[k]["d_igbn"][1] > 0]
        wg = [k for k in v if v[k]["d_igbn"][0] <= -0.0014 and v[k]["d_igbn"][2] < 0]
        worst = min(v, key=lambda k: v[k]["d"][0])
        summ[a] = {"better": better, "worse": worse, "worst": [worst, v[worst]["d"]],
                   "better_than_igbn_degraded_day": bg, "worse_than_igbn_any": wg,
                   "mean_vs_igbn_degraded_day": float(np.mean([v[k]["d_igbn"][0] for k in deg])),
                   "clean_vis_min": min(v[k]["d"][0] for k in v if k.startswith("clean/")),
                   "night_min": min(v[k]["d"][0] for k in v if k.endswith("/night"))}
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"
    L = [f"## Screen summary, draw {args.draw} (unregistered)", "",
         "| arm | better | worse | worst cell (Δ vs `union`) | min over clean-VIS cells | min over night cells | "
         "better / worse than IG-BN (degraded day / any) | mean Δ vs IG-BN, degraded day |",
         "|---|---:|---:|---|---:|---:|---:|---:|"]
    for a in arms:
        s = summ[a]
        L.append(f"| {NAMES[a]} | {len(s['better'])} | {len(s['worse'])} | {s['worst'][0]} {f(s['worst'][1])} | "
                 f"{s['clean_vis_min']:+.4f} | {s['night_min']:+.4f} | {len(s['better_than_igbn_degraded_day'])} / "
                 f"{len(s['worse_than_igbn_any'])} | {s['mean_vs_igbn_degraded_day']:+.4f} |")
    key = [k for k in KEY if k in tab["mix16"]]
    L += ["", "### Key cells, Δ vs `union` [95%]", "", "| arm | " + " | ".join(key) + " |",
          "|---|" + "---|" * len(key)]
    for a in arms:
        L.append(f"| {NAMES[a]} | " + " | ".join(f(tab[a][k]["d"]) for k in key) + " |")
    L += ["", "### Per cell, Δ vs `union` (mean over five systems)", "",
          "| cell | union | " + " | ".join(arms) + " |", "|---|---:|" + "---:|" * len(arms)]
    for k in tab["mix16"]:
        L.append(f"| {k} | {tab['mix16'][k]['union']:.4f} | " + " | ".join(f"{tab[a][k]['d'][0]:+.4f}" for a in arms) + " |")
    # selector and weight diagnostics
    L += ["", "### Diagnostics (build): nearest-domain picks night; mean adaptation weight per layer", ""]
    hdr = False
    for c in cs:
        p = RUN["dir"] / "diag" / ttbn.sub(args.draw) / f"{c}.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        on = np.array(d["oracle_night"])
        nm = np.array(d["near_mix"], float).mean(1)
        qs = [q for q in d["amean"] if q != "mix16" and d["amean"][q]]
        if not hdr:
            L += ["| VIS cond | slice | blocks | picks night | " + " | ".join(qs) + " |", "|---|---|---:|---:|" + "---:|" * len(qs)]
            hdr = True
        for s_, msk in (("day", ~on), ("night", on)):
            L.append(f"| {c} | {s_} | {int(msk.sum())} | {nm[msk].mean():.3f} | " +
                     " | ".join(f"{np.array(d['amean'][q])[msk].mean():.4f}" for q in qs) + " |")
    o = RUN["dir"] / f"report_{args.draw}.md"
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"summary": summ, "cells": tab}, indent=1), encoding="utf-8")
    print("\n".join(L[:len(arms) + 4]))
    return 0


P2_CELLS = tuple(f"fog_s1/{ic}/day" for ic in ("clean", "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")) + \
    tuple(f"lowlight/{ic}/day" for ic in ("fog_s2", "fog_s3", "noise_s2", "noise_s3"))


def cmd_verdict(args) -> int:
    """`docs/prereg-gatefree-ttbn-2026-10-16.md`: per system, mean over draws 942-944; t-interval over systems."""
    res = json.loads((RUN["dir"] / "score.json").read_text(encoding="utf-8"))
    n4 = json.loads((ttbn.DEV_OUT / "score.json").read_text(encoding="utf-8"))
    repro = max(abs(res[key][cs]["union_igbn"] - n4[key][cs]["union_ttbn"]) for key in res for cs in res[key])
    cells = [(c, ic, s) for c in ttbn.DEV_CONDS for ic in ttbn.DEV_IR_CONDS for s in ("day", "night")]
    tab = {}
    for a in RUN["arms"] + ("igbn",):
        tab[a] = {}
        for c, ic, s in cells:
            pk = {"u": [], "g": []}
            for k in ttbn.SEEDS:
                rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                pk["u"].append(np.mean([r[f"union_{a}"] - r["union"] for r in rs]))
                pk["g"].append(np.mean([r[f"union_{a}"] - r["union_igbn"] for r in rs]))
            tab[a][f"{c}/{ic or 'clean'}/{s}"] = {"d": ttbn._cell_stats(pk["u"]), "d_igbn": ttbn._cell_stats(pk["g"]),
                                                   "union": float(np.mean([res[f"{d}|{ic}|{k}"][f"{c}|{s}"]["union"]
                                                                           for d in ttbn.DEV_DRAWS for k in ttbn.SEEDS]))}
    worse = lambda x: x[0] <= -0.0014 and x[2] < 0
    better = lambda x: x[0] >= 0.0060 and x[1] > 0
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"
    out, L = {}, ["# Gate-free BN adaptation (GF31 / GF21), development verdict", "",
                  "`docs/prereg-gatefree-ttbn-2026-10-16.md`. Draws VIS 942–944 / IR 952–954, five systems. Δ per system = "
                  "mean over the three draws; cell = mean over systems, 95% t-interval (df 4). Worse: Δ ≤ −0.0014 with "
                  "interval below zero; better: Δ ≥ +0.0060 with interval above zero.", "",
                  f"IG-BN reproduces N4's registered `union_ttbn` scores: max |diff| {repro:.1e}.", ""]
    for a in RUN["arms"]:
        v = tab[a]
        w = [k for k in v if worse(v[k]["d"])]
        b = [k for k in v if better(v[k]["d"])]
        p2 = [k for k in P2_CELLS if better(v[k]["d_igbn"])]
        night = [k for k in v if k.endswith("/night") and k.split("/")[0] in ("clean", "lowlight")]
        p3a = all(-0.0050 <= v[k]["d"][0] <= 0.0050 for k in night)
        p3b = not any(worse(v[k]["d"]) for k in night)
        p1, p1b = not w, bool(b)
        bg = [k for k in v if better(v[k]["d_igbn"])]
        wg = [k for k in v if worse(v[k]["d_igbn"])]
        worst = min(v, key=lambda k: v[k]["d"][0])
        out[a] = {"P1": p1, "P1b": p1b, "verdict": "PASSES" if (p1 and p1b) else "FAILS", "worse": w, "better": b,
                  "P2": len(p2) == len(P2_CELLS), "P2_better": p2, "P3a": p3a, "P3b": p3b, "P3": p3a and p3b,
                  "better_than_igbn": bg, "worse_than_igbn": wg, "worst": [worst, v[worst]["d"]]}
        L += [f"## {'GF31 (primary)' if a == 'sh31_hinge50' else 'GF21 (secondary, descriptive)'}: development {out[a]['verdict']}", "",
              f"* P1 (no cell worse than `union`): {'PASS' if p1 else 'FAIL (' + ', '.join(w) + ')'}.",
              f"* P1b (a cell better): {'PASS' if p1b else 'FAIL'} ({len(b)} cells).",
              f"* P2 (better than IG-BN in the ten cells): {'HOLDS' if out[a]['P2'] else 'FAILS'} ({len(p2)}/10"
              f"{'; not better: ' + ', '.join(k for k in P2_CELLS if k not in p2) if len(p2) < 10 else ''}).",
              f"* P3 (clean/low-light night within ±0.005 and none worse): {'HOLDS' if out[a]['P3'] else 'FAILS'} "
              f"(a {'yes' if p3a else 'no'}, b {'yes' if p3b else 'no'}); night range "
              f"{min(v[k]['d'][0] for k in night):+.4f} to {max(v[k]['d'][0] for k in night):+.4f}.",
              f"* Worst cell: {worst} {f(v[worst]['d'])}.",
              f"* Against IG-BN (all 60 cells): {len(bg)} better, {len(wg)} worse"
              f"{' (' + ', '.join(f'{k} {f(v[k]['d_igbn'])}' for k in wg) + ')' if wg else ''}.", ""]
    L += ["## Per cell: Δ vs `union` and vs IG-BN [95%]", "",
          "| cell | union | IG-BN vs union | GF31 vs union | GF31 vs IG-BN | GF21 vs union | GF21 vs IG-BN |",
          "|---|---:|---|---|---|---|---|"]
    for k in tab["igbn"]:
        L.append(f"| {k} | {tab['igbn'][k]['union']:.4f} | {f(tab['igbn'][k]['d'])} | {f(tab['sh31_hinge50'][k]['d'])} | "
                 f"{f(tab['sh31_hinge50'][k]['d_igbn'])} | {f(tab['sh21_hinge50'][k]['d'])} | {f(tab['sh21_hinge50'][k]['d_igbn'])} |")
    o = ROOT / args.out
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"repro_max_abs_diff": repro, "arms": out, "cells": tab}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L[:30]))
    return 0


# ------------------------------------------------------------------ pohang04 (GF31 only; only after a development PASS)
GF = "sh31_hinge50"
GF_K = 31
P04_DIR = OUT / "p04"
P04_PRE = OUT / "p04_pre"


def p04_path(k: int, cond: str, draw, fp16: bool) -> Path:
    where = "clean" if cond == "clean" else ttbn.sub(draw)
    return P04_DIR / f"seed{k}" / where / f"{GF}_{cond}{'_fp16' if fp16 else ''}.pkl"


def _gf_stream(preds, nl, cal, lst, spec, chunks, fp16, workers, tag):
    """GF31 records for the given blocks (the registered arithmetic; fp16 autocast if asked)."""
    import p04_heldout as ph
    recs = [[None] * len(lst) for _ in preds]
    t0 = time.time()
    for nb, ch in enumerate(ttbn._frames(spec, chunks, workers), 1):
        ims = [im for _, im in ch]
        with ph.amp(fp16):
            r, _, _ = _run(preds, ims, _tensor(preds, ims), nl, kind="hinge", D0=cal["D0_exp"], kmax=GF_K)
        for j in range(len(preds)):
            for (i, _), x in zip(ch, r[j]):
                recs[j][i] = {**(ph._f32(x) if fp16 else x), "image_path": lst[i]}
        if nb % 200 == 0:
            print(f"[{tag}] {nb}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    return recs


def _gf_preds(fp16: bool):
    import p04_heldout as ph
    preds, nl = _preds()
    if fp16:
        ph.setup_fp16(preds)
    src, cal = _load_calib(preds[0].device)
    return preds, nl, cal


def cmd_check_fp16(args) -> int:
    """Development fp16 check for GF31 (as N4's): draw 942, VIS fog / fog_s1 / lowlight, IR clean, day, five
    systems; `union` with GF31 records in fp16 against the registered fp32 build; passes iff each cell moves by at most
    0.0010 ship AP."""
    import dataclasses
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    conds = ("fog", "fog_s1", "lowlight")
    out_d = OUT / "fp16_check"
    preds, nl, cal = _gf_preds(True)
    chunks = [list(range(a, min(a + ttbn.BATCH, len(lst)))) for a in range(0, len(lst), ttbn.BATCH)]
    for c in conds:
        if all((out_d / f"seed{k}_{c}.pkl").is_file() for k in ttbn.SEEDS):
            continue
        kc = ttbn.DEV_CONDS[c]
        recs = _gf_stream(preds, nl, cal, lst, {"kind": kc[0], "sev": kc[1], "seed": 942, "images": lst}, chunks,
                          True, args.workers, f"fp16 {c}")
        out_d.mkdir(parents=True, exist_ok=True)
        for j, k in enumerate(ttbn.SEEDS):
            with open(out_d / f"seed{k}_{c}.pkl", "wb") as fh:
                pickle.dump({"records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    del preds
    import seed_ensemble as se
    from gate_ablation import select
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    ref = json.loads((OUT / "verdict" / "score.json").read_text(encoding="utf-8"))
    se._memo_distances()
    d = f"{se.STATS}/brightness/{ttbn.sub(942)}"
    diffs = {c: [] for c in conds}
    for k in ttbn.SEEDS:
        ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{ttbn.sub(942)}", bright_dir=d,
                           structure_dir=f"{se.STATS}/structure/{ttbn.sub(942)}",
                           ir_bright=f"{d}/gauss_ir_paired_clean.json", conditions=conds, ir_condition=None,
                           verbose=False)
        day = np.flatnonzero(~np.isin(ctx.runs, NIGHT_RUNS))
        u = dataclasses.replace(ctx, veto=None)
        for c in conds:
            with open(out_d / f"seed{k}_{c}.pkl", "rb") as fh:
                rr = pickle.load(fh)["records"]
            vg = select(rr, ctx.vis_by_cond[c], np.ones(len(rr), bool))
            v = ship_ap(frame_parts(run_systems(u, c, compute_metrics=False, vis_records=vg)["fused_gated"], ctx.gts), day)
            diffs[c].append(v - ref[f"942|None|{k}"][f"{c}|day"][f"union_{GF}"])
    cells = {c: float(np.mean(v)) for c, v in diffs.items()}
    ok = all(abs(v) <= 0.0010 for v in cells.values())
    (OUT / "fp16_check.json").write_text(json.dumps({"pass": ok, "tolerance": 0.0010, "diff_mean_over_systems": cells,
                                                     "per_system": diffs}, indent=1), encoding="utf-8")
    print(json.dumps(cells), "PASS" if ok else "FAIL: pohang04 is built in fp32", flush=True)
    return 0


def _p04_fp16() -> bool:
    p = OUT / "fp16_check.json"
    return bool(json.loads(p.read_text(encoding="utf-8")).get("pass")) if p.is_file() else False


def cmd_build_p04(args) -> int:
    import p04_heldout as ph
    from gate_ablation import streams
    ver = json.loads((ROOT / "docs/eval/ttbn/gatefree_dev_2026-10-16.json").read_text(encoding="utf-8"))
    if ver["arms"][GF]["verdict"] != "PASSES":
        raise SystemExit("GF31 did not pass development: pohang04 is not scored")
    if not (OUT / "fp16_check.json").is_file():
        raise SystemExit("run check-fp16 first (it decides pohang04's precision)")
    fp16 = _p04_fp16()
    lst = ph.images("vis")
    todo = [(c, d) for c, d in streams("p04") if not all(p04_path(k, c, d, fp16).is_file() for k in ttbn.SEEDS)]
    print(f"[p04] {len(lst)} frames, {len(todo)} streams ({'fp16' if fp16 else 'fp32'})", flush=True)
    if not todo:
        return 0
    preds, nl, cal = _gf_preds(fp16)
    chunks = [list(range(a, min(a + ttbn.BATCH, len(lst)))) for a in range(0, len(lst), ttbn.BATCH)]
    for c, d in todo:
        kc = ttbn.P04_CONDS[c]
        t0 = time.time()
        recs = _gf_stream(preds, nl, cal, lst, {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0,
                                                "seed": d or 0, "images": lst}, chunks, fp16, args.workers, f"p04 {c}/{d}")
        for j, k in enumerate(ttbn.SEEDS):
            o = p04_path(k, c, d, fp16)
            o.parent.mkdir(parents=True, exist_ok=True)
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": {"arm": GF, "cond": c, "draw": d, "fp16": fp16, "blocks": "every 8-frame block, no gate",
                                      "prereg": "docs/prereg-gatefree-p04-2026-10-16.md"}, "records": recs[j]}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[p04] {c}/{d} done: {len(chunks)} blocks, {time.time() - t0:.0f}s", flush=True)
    return 0


def _p04_job(job):
    import dataclasses
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    import p04_heldout_score as psc
    import seed_ensemble as se
    from gate_ablation import select
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc, fp16 = job
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
    d = P04_PRE / f"union_{GF}" / tag / key
    if (d / "obs.json").is_file():
        return job, json.loads((d / "obs.json").read_text())["ap"]
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    vis = ctx.vis_by_cond[vc]
    draw = None if (s == "clean" or vc == "clean") else int(s[4:7])
    with open(p04_path(k, vc, draw, fp16), "rb") as fh:
        rr = pickle.load(fh)["records"]
    recs = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False,
                       vis_records=select(rr, vis, np.ones(len(vis), bool)))["fused_gated"]
    d.mkdir(parents=True, exist_ok=True)
    return job, psc.save_pre(frame_parts(recs, ctx.gts), d)


def cmd_score_p04(args) -> int:
    import p04_heldout_score as psc
    from gate_ablation import _p04_cells
    fp16 = _p04_fp16()
    jobs = [(k, s, ic, vc, fp16) for vc, ic in _p04_cells() for s in psc.subs_of((vc, ic)) for k in ttbn.SEEDS]
    print(f"[p04 score] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, v) in enumerate(ex.map(_p04_job, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                print(f"[p04 score] {n}/{len(jobs)} {j[:4]} {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_boot_p04(args) -> int:
    """The gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1), the gate ablation's GPU path."""
    import gate_ablation as ga
    ga.PRE = P04_PRE
    return ga.cmd_boot_p04(args)


def cmd_report_p04(args) -> int:
    import gate_ablation as ga
    import p04_heldout_score as psc
    gates = json.loads(ga.P04_GATES.read_text(encoding="utf-8"))
    cells, L = {}, []
    for c in ga._p04_cells():
        tag = psc.cell_tag(c)
        vals, reps = [], []
        for s in psc.subs_of(c):
            vv, rr = [], []
            for k in ttbn.SEEDS:
                d = P04_PRE / f"union_{GF}" / tag / f"seed{k}_{s}"
                vv.append(json.loads((d / "obs.json").read_text())["ap"])
                rr.append(np.load(d / "reps.npy"))
            vals.append(np.mean(vv))
            reps.append(np.mean(rr, axis=0))
        gf = (float(np.mean(vals)), np.mean(reps, axis=0))
        ref = {a: ga._series(a, c, gates) for a in ("union", "n4", "ungated")}
        cells[tag] = {"union": ref["union"][0], "gf31": gf[0]}
        for a, x in ref.items():
            cells[tag][f"vs_{a}"] = [gf[0] - x[0], *psc.ci(gf[1] - x[1])]
    rule_w = lambda v: v[0] <= -0.0014 and v[2] < 0
    rule_b = lambda v: v[0] >= 0.0060 and v[1] > 0
    cnt = {a: {"better": [t for t, v in cells.items() if rule_b(v[f"vs_{a}"])],
               "worse": [t for t, v in cells.items() if rule_w(v[f"vs_{a}"])]} for a in ("union", "n4", "ungated")}
    ver = "REPLICATES" if (not cnt["union"]["worse"] and cnt["union"]["better"]) else "DOES NOT REPLICATE"
    f = lambda v: f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]"
    L = ["# Gate-free BN adaptation (GF31) on the held-out run (pohang04)", "",
         "`docs/prereg-gatefree-p04-2026-10-16.md` (thirteenth scoring). Nine registered cells; ship AP against the VIS "
         "ground truth, mean over the five systems and the four draws of a corrupted cell. Δ paired; 95% interval from "
         "the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1). Worse: Δ ≤ −0.0014 with interval below "
         f"zero; better: Δ ≥ +0.0060 with interval above zero. Precision: {'fp16' if _p04_fp16() else 'fp32'}.", "",
         f"**Q1 (vs `union`): {ver}** ({len(cnt['union']['better'])} better, {len(cnt['union']['worse'])} worse"
         f"{': ' + ', '.join(cnt['union']['worse']) if cnt['union']['worse'] else ''}).",
         f"Against IG-BN: {len(cnt['n4']['better'])} better, {len(cnt['n4']['worse'])} worse. Against ungated BN-mix: "
         f"{len(cnt['ungated']['better'])} better, {len(cnt['ungated']['worse'])} worse.", "",
         "| cell (VIS/IR) | union | GF31 | Δ vs union | Δ vs IG-BN | Δ vs ungated |", "|---|---:|---:|---|---|---|"]
    for t, v in cells.items():
        L.append(f"| {t.replace('__', '/')} | {v['union']:.4f} | {v['gf31']:.4f} | {f(v['vs_union'])} | {f(v['vs_n4'])} | "
                 f"{f(v['vs_ungated'])} |")
    o = ROOT / args.out
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"Q1": ver, "counts": cnt, "cells": cells}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calib")
    c.add_argument("--workers", type=int, default=4)
    b = sp.add_parser("build")
    b.add_argument("--draw", type=int, default=941)
    b.add_argument("--conds", nargs="+", default=None)
    b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score")
    s.add_argument("--draw", type=int, default=941)
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report")
    r.add_argument("--draw", type=int, default=941)
    for name in ("check-fp16", "build-p04", "score-p04", "boot-p04"):
        q = sp.add_parser(name)
        q.add_argument("--workers", type=int, default=4)
    q = sp.add_parser("report-p04")
    q.add_argument("--out", default="docs/eval/ttbn/gatefree_p04_2026-10-16.md")
    v = sp.add_parser("verdict")
    v.add_argument("--out", default="docs/eval/ttbn/gatefree_dev_2026-10-16.md")
    v.add_argument("--draw", type=int, default=942)
    for x in (b, s, r, v):
        x.add_argument("--set", type=int, default=1, choices=[1, 2, 3, 4], help="2: shallow + hinge arms (OUT/quick2); 3: the two median-hinge shallow arms (OUT/full3)")
        x.add_argument("--every", type=int, default=1, help="run every k-th block only (subset run, OUT/quick)")
    args = ap.parse_args()
    if getattr(args, "every", 1) > 1:
        RUN["dir"] = OUT / "quick"
    if getattr(args, "set", 1) == 2:
        RUN["dir"], RUN["arms"] = OUT / "quick2", ARMS2
    if getattr(args, "set", 1) == 3:
        RUN["dir"], RUN["arms"] = OUT / ("full3" if args.every == 1 else "quick3"), ARMS3
    if getattr(args, "set", 1) == 4:
        RUN["dir"], RUN["arms"], RUN["clean_shared"] = OUT / "verdict", ARMS4, True
    return {"calib": cmd_calib, "build": cmd_build, "score": cmd_score, "report": cmd_report, "verdict": cmd_verdict, "check-fp16": cmd_check_fp16, "build-p04": cmd_build_p04,
            "score-p04": cmd_score_p04, "boot-p04": cmd_boot_p04, "report-p04": cmd_report_p04}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
