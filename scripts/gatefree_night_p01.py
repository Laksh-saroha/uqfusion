"""Shallow BN-mix (GF31) and ungated BN-mix on the pohang01 night test (`docs/prereg-gatefree-night-p01-2026-10-16.md`).

The night test's 1,109 frame pairs (`p01_nighttest.py`), VIS and IR clean, the five Phase 3 systems, fused by
`union`. Two adapted VIS streams, both on every non-overlapping 8-frame block of the pair list, fp32, as on
development:
* GF31: the registered arithmetic of `ttbn_gatefree.py` (first 31 BatchNorm layers, median hinge, frozen
  `calib.pt`);
* mix16: ungated BN-mix on every layer (N = 16), the gate ablation's "none" arm.
IG-BN and the clock gate adapt no frame here (IR clean: IG-BN's gate fires on none of these pairs; the sun is
below the horizon on all), so each equals `union`. Scored with the night test's own replicates
(`runs/night_p01test/boot_weights.npy`), paired against its stored `union`.

    py -3.13 scripts/gatefree_night_p01.py build
    py -3.13 scripts/gatefree_night_p01.py score
    py -3.13 scripts/p01_nighttest.py boot          # replicates for the new presorts (same weights)
    py -3.13 scripts/gatefree_night_p01.py report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import p01_nighttest as nt  # noqa: E402
import ttbn  # noqa: E402
import ttbn_gatefree as gf  # noqa: E402

OUT = nt.OUT / "gatefree"
ARMS = {"gf31": dict(kind="hinge", kmax=gf.GF_K), "mix16": dict(kind="mix")}
PREREG = "docs/prereg-gatefree-night-p01-2026-10-16.md"


def rec_path(k: int, arm: str) -> Path:
    return OUT / f"seed{k}" / f"{arm}_clean.pkl"


def cmd_build(args) -> int:
    lst = nt.read_list(nt.LISTS["vis"])
    todo = [a for a in ARMS if not all(rec_path(k, a).is_file() for k in ttbn.SEEDS)]
    print(f"[build] {len(lst)} frames, arms {todo}", flush=True)
    if not todo:
        return 0
    preds, nl, cal = gf._gf_preds(False)
    chunks = [list(range(a, min(a + ttbn.BATCH, len(lst)))) for a in range(0, len(lst), ttbn.BATCH)]
    spec = {"kind": None, "sev": 0, "seed": 0, "images": lst}
    for arm in todo:
        cfg = dict(ARMS[arm])
        if cfg["kind"] == "hinge":
            cfg["D0"] = cal["D0_exp"]
        recs = [[None] * len(lst) for _ in preds]
        t0 = time.time()
        for ch in ttbn._frames(spec, chunks, args.workers):
            ims = [im for _, im in ch]
            r, _, _ = gf._run(preds, ims, gf._tensor(preds, ims), nl, **cfg)
            for j in range(len(preds)):
                for (i, _), x in zip(ch, r[j]):
                    recs[j][i] = {**x, "image_path": lst[i]}
        for j, k in enumerate(ttbn.SEEDS):
            o = rec_path(k, arm)
            o.parent.mkdir(parents=True, exist_ok=True)
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": {"arm": arm, "cfg": {a: v for a, v in cfg.items() if a != "D0"}, "fp16": False,
                                      "blocks": "every 8-frame block of the pair list, no gate", "prereg": PREREG},
                             "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[build] {arm}: {len(chunks)} blocks in {time.time() - t0:.0f}s", flush=True)
    return 0


def _score_job(k: int) -> dict:
    import holdout_p04_look as look
    import seed_ensemble as se
    from gate_ablation import select
    from uqfusion.eval.ctx import run_systems
    nt._low()
    se._memo_distances()
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    ctx = look.system_context(nt._substrate(), k, "clean", ["clean"], None, dev)
    vis = ctx.vis_by_cond["clean"]
    u = dataclasses.replace(ctx, veto=None)
    info = {"k": k}
    for arm in ARMS:
        with open(rec_path(k, arm), "rb") as fh:
            rr = pickle.load(fh)["records"]
        assert [Path(r["image_path"]).name for r in rr] == [Path(r["image_path"]).name for r in vis]
        av = select(rr, vis, np.ones(len(vis), bool))
        out = run_systems(u, "clean", compute_metrics=False, vis_records=av)
        info[f"{arm}/vis"] = nt._save(f"{arm}/vis", f"seed{k}", av, ctx.gts)
        info[f"union_{arm}/fused"] = nt._save(f"union_{arm}/fused", f"seed{k}", out["fused_gated"], ctx.gts)
    return info


def cmd_score(args) -> int:
    obs_p = OUT / "observed.json"
    obs = json.loads(obs_p.read_text(encoding="utf-8")) if obs_p.is_file() else {}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for r in ex.map(_score_job, ttbn.SEEDS):
            obs[f"seed{r['k']}"] = r
            obs_p.parent.mkdir(parents=True, exist_ok=True)
            obs_p.write_text(json.dumps(obs, indent=1), encoding="utf-8")
            print(f"[score] system {r['k']}: {r}", flush=True)
    return 0


def cmd_report(args) -> int:
    un, uv = nt.series("union/fused"), nt.series("single/vis")
    rule_w = lambda d, lo, hi: d <= -0.0014 and hi < 0
    rule_b = lambda d, lo, hi: d >= 0.0060 and lo > 0
    res, rows = {}, []
    for arm in ARMS:
        f, v = nt.series(f"union_{arm}/fused"), nt.series(f"{arm}/vis")
        d, (lo, hi) = f[0] - un[0], nt.ci(f[1] - un[1])
        dv, (vlo, vhi) = v[0] - uv[0], nt.ci(v[1] - uv[1])
        res[arm] = {"ap": f[0], "vs_union": [d, lo, hi], "worse": rule_w(d, lo, hi), "better": rule_b(d, lo, hi),
                    "vis": v[0], "vis_vs_plain": [dv, vlo, vhi], "per_system": f[2]}
        rows.append(f"| {arm} | {f[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                    f"{'worse' if res[arm]['worse'] else 'better' if res[arm]['better'] else 'neither'} | {v[0]:.4f} | "
                    f"{dv:+.4f} [{vlo:+.4f}, {vhi:+.4f}] | {', '.join(f'{x:.4f}' for x in f[2])} |")
    g, m = res["gf31"], res["mix16"]
    res["N1"] = "FAIL" if g["worse"] else "PASS"
    res["N2"] = "HOLDS" if m["worse"] else "FAILS"
    res["N3"] = "HOLDS" if -0.0050 <= g["vs_union"][0] <= 0.0050 else "FAILS"
    L = ["# Shallow BN-mix and ungated BN-mix on the pohang01 night test", "",
         f"`{PREREG}`. The night test's {len(nt.read_list(nt.LISTS['vis']))} frame pairs, VIS and IR clean; ship AP "
         "against the VIS labels, mean over the five Phase 3 systems; Δ paired against the stored `union`; 95% interval "
         "from the night test's 1,000 moving-block replicates (L = 20, seed 1). Worse: Δ ≤ −0.0014 with interval below "
         "zero; better: Δ ≥ +0.0060 with interval above zero. IG-BN and the clock gate equal `union` here.", "",
         f"**N1 (shallow BN-mix not worse than `union`): {res['N1']}.**  ",
         f"**N2 (ungated BN-mix worse than `union`): {res['N2']}.**  ",
         f"**N3 (shallow BN-mix within ±0.005 of `union`, stated expectation): {res['N3']}.**", "",
         f"`union`: {un[0]:.4f} (VIS stream {uv[0]:.4f}).", "",
         "| arm | `union` + arm | Δ vs `union` [95%] | rule | VIS stream | VIS Δ vs plain [95%] | per system |",
         "|---|---:|---:|---|---:|---:|---|", *rows]
    o = ROOT / args.out
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--workers", type=int, default=3)
    s = sp.add_parser("score")
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report")
    r.add_argument("--out", default="docs/eval/ttbn/gatefree_night_p01_2026-10-16.md")
    args = ap.parse_args()
    return {"build": cmd_build, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
