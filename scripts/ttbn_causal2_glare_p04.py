"""N4c2g on pohang04's glare cell (P-C2G, 2026-10-14).

Registered in `docs/prereg-ttbn-causal-glare-p04-2026-10-14.md`. N4c2g is N4c2 (`ttbn_causal2.py`: each gated
block normalised with BN-mix of the source statistics and N4's statistics of the previous block) under N4g's
gate (`ttbn_glare_p04.py`: the glare term, VIS mean > 117.92 and sd > 62.31, beside q_vis < 1, under the same
two thermal terms). One cell: glare/clean, draws 941-944, five systems, on P-G's plain glare caches.

    py -3.13 scripts/ttbn_causal2_glare_p04.py build      # GPU, N4c2's sparse passes, fp16
    py -3.13 scripts/ttbn_causal2_glare_p04.py score      # presorts union_ttbnc2g/fused, ttbnc2g/vis
    py -3.13 scripts/p04_heldout_score.py boot            # replicates of the new presorts
    py -3.13 scripts/ttbn_causal2_glare_p04.py report
"""

from __future__ import annotations

import argparse
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

import ttbn  # noqa: E402
import ttbn_causal2 as c2  # noqa: E402
import ttbn_glare_p04 as tg  # noqa: E402
import ttbn_variants as tv  # noqa: E402

OUT = ROOT / "runs/holdout_p04/ttbn_causal2_glare"
CELL = tg.NEW_CELL
PRESORTS = ("union_ttbnc2g/fused", "ttbnc2g/vis")


def out_path(k: int, draw: int) -> Path:
    return OUT / f"seed{k}" / ttbn.sub(draw) / f"{tg.STEM}_bn16_fp16.pkl"


def cmd_build(args) -> int:
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    chk = json.loads(c2.SPARSE_CHECK.read_text(encoding="utf-8"))
    if not chk.get("pass"):
        raise SystemExit("the build needs N4c2's passed sparse check")
    ttbn._low()
    lst = ph.images("vis")
    todo = [d for d in (args.draws or tg.p04_draws()) if not all(out_path(k, d).is_file() for k in ttbn.SEEDS)]
    if not todo:
        print("[c2g] nothing to build", flush=True)
        return 0
    preds = c2._predictors(0)
    print(f"[c2g] BN layers {[c2.patch_bn_delayed(p.model) for p in preds]}; {len(lst)} frames, draws {todo}",
          flush=True)
    kind, sev = tg.GLARE
    for d in todo:
        need = tg.needed_g("glare", d)
        assert len(need) == len(lst)
        plan = c2._plan(need, lst)
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        spec = {"kind": kind, "sev": sev, "seed": d, "images": lst}
        sec = c2._run_plan(preds, plan, spec, args.workers, ph, recs, lst, f"glare/{d}") if plan else 0.0
        miss = [i for i in np.flatnonzero(need) if recs[0][i] is None]
        if miss:
            raise SystemExit(f"glare/{d}: {len(miss)} needed frames without a record")
        for j, k in enumerate(ttbn.SEEDS):
            o = out_path(k, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"variant": "N4c2g", "target": "p04", "cond": "glare", "draw": d, "prior_N": ttbn.PRIOR_N,
                    "batch": ttbn.BATCH, "needed_frames": int(need.sum()), "passes": len(plan),
                    "output_blocks": sum(p[2] for p in plan), "recording_passes": sum(p[3] for p in plan),
                    "gate": "N4g (q_vis < 1 OR glare term) AND IR day AND ir_d2 bound",
                    "blocks": "N4c's per-run grid; block b normalised with N4's statistics of block b-1 "
                              "(closed-loop record pass); a run's first block at source; only used passes run",
                    "precision": "fp16 autocast, channels_last, fused statistics",
                    "corrupt": kind, "severity": sev, "corrupt_seed": d, "corrupt_version": "v2",
                    "corrupt_code": CODE_REV,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[c2g] glare/{d} done: {sum(p[2] for p in plan)} output blocks, {len(plan)} passes, "
              f"{int(need.sum())} needed frames, {sec:.0f}s", flush=True)
    return 0


def _score_job(job):
    import dataclasses
    import holdout_p04_look as look
    import p04_heldout_score as psc
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s = job
    vc, ic = CELL
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag(CELL), f"seed{k}_{s}"
    draw = int(s[4:7])
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    ctx = look.system_context(tg._substrate(), k, s, sorted({vc, "clean"}), ic, dev)
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    gg = ((np.asarray(ctx.q_vis_by_cond[vc]) < 1.0)
          | tv.glare_term(ttbn._p04_stats("vis", vc, draw, "brightness"))) & irok
    if not np.array_equal(gg, tg.gate_g_p04(vc, ic, s)):
        raise SystemExit(f"context gate differs from the frame-statistics gate ({k}, {s})")
    vis = ctx.vis_by_cond[vc]
    vg = tv._stream(out_path(k, draw), vis, gg)
    out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False, vis_records=vg)
    for name, recs in (("union_ttbnc2g/fused", out["fused_gated"]), ("ttbnc2g/vis", vg)):
        d = psc.PRE / name / tag / key
        if not (d / "obs.json").is_file():
            psc.save_pre(frame_parts(recs, ctx.gts), d)
    return job, {"gate": float(gg.mean()), "n_gated": int(gg.sum())}


def cmd_score(args) -> int:
    import p04_heldout_score as psc
    gp = OUT / "gates.json"
    gates = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else {}
    tag = psc.cell_tag(CELL)
    jobs = [(k, s) for s in psc.subs_of(CELL) for k in ttbn.SEEDS
            if not (f"{k}|{s}" in gates and all((psc.PRE / n / tag / f"seed{k}_{s}" / "obs.json").is_file()
                                                for n in PRESORTS))]
    print(f"[score] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_score_job, jobs), 1):
            gates[f"{j[0]}|{j[1]}"] = info
            tmp = gp.with_suffix(".tmp")
            tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
            os.replace(tmp, gp)
            print(f"[score] {n}/{len(jobs)} {j} gate {info['gate']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report(args) -> int:
    import p04_heldout_score as psc
    names = ("union_ttbnc2g/fused", "union_ttbng/fused", "union/fused", "shipped/fused", "ttbnc2g/vis",
             "single/vis", "single/ir")
    S = {n: psc.series(n, CELL) for n in names}
    if any(v is None for v in S.values()):
        raise SystemExit(f"presorts or replicates missing: {[n for n, v in S.items() if v is None]}")
    gate = float(np.mean([tg.gate_g_p04(CELL[0], CELL[1], s).mean() for s in psc.subs_of(CELL)]))
    a = S["union_ttbnc2g/fused"]

    def diff(b):
        return a[0] - b[0], psc.ci(a[1] - b[1])

    d, (lo, hi) = diff(S["union/fused"])
    dg, (glo, ghi) = diff(S["union_ttbng/fused"])
    ds, (slo, shi) = diff(S["shipped/fused"])
    mx = np.maximum(S["ttbnc2g/vis"][1], S["single/ir"][1])
    cl, (cllo, clhi) = a[0] - max(S["ttbnc2g/vis"][0], S["single/ir"][0]), psc.ci(a[1] - mx)
    worse = d <= -0.0014 and hi < 0
    better = d >= 0.0060 and lo > 0
    v = "REPLICATES" if (better and not worse) else \
        f"DOES NOT REPLICATE ({'worse' if worse else 'not worse'}; {'gains' if better else 'does not gain'})"
    L = ["# N4c2g (IG-BN-causal+glare) on pohang04's glare cell", "",
         "`docs/prereg-ttbn-causal-glare-p04-2026-10-14.md` (P-C2G), the twelfth scoring of pohang04. Ship AP "
         "against the VIS ground truth, mean over the five systems and the four draws (VIS v2 glare severity 2, "
         "draws 941–944, IR clean). Δ = `union_ttbnc2g` − `union`, paired; 95% interval from the gate-free "
         "scoring's 1,000 moving-block replicates (L = 20, seed 1). N4c2g: N4c2's causal BN-mix (previous "
         "block's N4 statistics, N = 16, blocks of 8, fp16) under N4g's gate.", "",
         f"**P-C2G: {v}.**", "",
         "| cell (VIS/IR) | gate | union | Δ vs union [95%] | Δ vs shipped [95%] | N4c2g − N4g [95%] | "
         "claim union_ttbnc2g [95%] |", "|---|---:|---:|---:|---:|---:|---:|",
         f"| glare/clean | {gate:.3f} | {S['union/fused'][0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
         f"{ds:+.4f} [{slo:+.4f}, {shi:+.4f}] | {dg:+.4f} [{glo:+.4f}, {ghi:+.4f}] | "
         f"{cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |", "",
         "N4c2g − N4g is descriptive (`union_ttbng` is P-G's presort): the price of causality on glare. "
         "The claim is fused − max(VIS, IR) with VIS the N4c2g stream.", "",
         "## glare/clean, descriptive", "", "| arm | ship AP |", "|---|---:|"] + \
        [f"| `{n}` | {x[0]:.4f} |" for n, x in S.items()]
    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    L += ["", f"Gate check: the context gate equalled the frame-statistics gate in all {len(gates)} jobs; "
              f"gated frames per job {min(g['n_gated'] for g in gates.values())}–"
              f"{max(g['n_gated'] for g in gates.values())}."]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps(
        {"P-C2G": v, "gate": gate, "delta": d, "ci": [lo, hi], "minus_n4g": dg, "ci_minus_n4g": [glo, ghi],
         "delta_vs_shipped": ds, "ci_vs_shipped": [slo, shi], "claim": [cl, cllo, clhi],
         "values": {n: x[0] for n, x in S.items()}}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--draws", type=int, nargs="+", default=None, help="only these draws (parallel builds)")
    s = sp.add_parser("score")
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_causal2_glare_p04_2026-10-14.md")
    args = ap.parse_args()
    return {"build": cmd_build, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
