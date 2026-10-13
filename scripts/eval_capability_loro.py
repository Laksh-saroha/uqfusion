"""Leave-one-run-out refit of the capability prior (TODO-improvements §D.2).

Under `preset="crossmodal26m"` the constant WBF weights are each stream's clean
mAP over `capability_sel="fit"`, which is the 1,200 clean day paired frames of
pohang00/02/03 -- the same frames every day cell is scored on. "Run-disjoint" in
the earlier records meant only that the night run is excluded. PAPER_DRAFT2 §9
item 20 discloses this and says no LORO refit was run. This is that refit.

For each day run r: fit the prior on the OTHER two day runs' clean frames, apply
the shipped ÷`cap_ir_scale`, and score the shipped system on run r's frames for
the four default conditions. The comparison arm is the shipped in-sample prior on
the SAME frames, so every delta is paired. A pooled "LORO system" stitches the
three folds frame by frame (each frame scored under the prior that never saw its
run) and is compared to the shipped system over all day frames and over TEST.

Stitching is exact, not an approximation: WBF, the veto (no hysteresis under
`crossmodal`) and the support term are per-frame, and `frame_parts` is per-frame,
so frame i's TP/confidence list depends only on frame i's inputs and the two
weights. The pooled AP then re-ranks across frames as any AP does.

The prior is swapped with `dataclasses.replace` on ONE context, as
`reprice_constants_draw_avg.py` does. The fold whose training runs are exactly
`TEST_RUNS` (held-out run pohang00) has a named selector, so it is checked against
a genuine `load_context(capability_sel="test")` -- prior to 1e-12 and fused AP on
all four conditions to 1e-12 -- before anything is reported.

Intervals: paired moving-block bootstrap within run, L = 20
(`uqfusion.eval.blockboot`). Reported for the macro (ship + buoy) and for ship
(class 0) separately. Nothing scores pohang04: the paired manifest has no
pohang04 frames, and the script asserts it.

Usage:
    py -3.13 scripts/eval_capability_loro.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import frame_parts                   # noqa: E402
from uqfusion.eval.blockboot import block_bootstrap_delta         # noqa: E402
from uqfusion.eval.ctx import (DEFAULT_CONDITIONS, FIT_RUNS,       # noqa: E402
                               NIGHT_RUNS, TEST_RUNS, capability_prior,
                               load_context, run_systems)
from uqfusion.eval.identity import system_identity                # noqa: E402
from uqfusion.eval.matching import map50_95                       # noqa: E402
from uqfusion.uq.fusion import apply_homography                   # noqa: E402

PRESET = "crossmodal26m"
CACHE = "runs/cache_m"
BLOCK_LEN = 20
SEED = 0
SHIP, BUOY = 0, 1
#: PAPER_DRAFT2 §5.3 paired 2-sigma floor for a macro delta on the informative cells.
FLOOR_LO, FLOOR_HI = 0.0014, 0.0031
DAY_RUNS = FIT_RUNS                     # pohang00/02/03; pohang01 is all night


def per_class_prior(ctx, sel) -> dict:
    """Ship/buoy AP behind each stream's prior, for the record (same inputs as
    `capability_prior`: clean VIS, clean IR after the preset's IR NMS, mapped)."""
    vis = [ctx.vis_by_cond["clean"][i] for i in sel]
    ir = [{**ctx.ir_clean[i], "boxes_xyxy": apply_homography(
        np.asarray(ctx.ir_clean[i]["boxes_xyxy"]).reshape(-1, 4), ctx.h_frames[i])}
        for i in sel]
    gts = [ctx.gts[i] for i in sel]
    out = {}
    for name, recs in (("vis", vis), ("ir", ir)):
        pc = map50_95(recs, gts)["per_class"]
        out[name] = {int(c): float(v["ap50_95"]) for c, v in pc.items()}
    return out


def fold_context(ctx, train_sel: np.ndarray):
    """The shipped context with only the prior refitted on `train_sel`."""
    cv, ci = capability_prior(ctx.vis_by_cond["clean"], ctx.ir_clean, ctx.gts,
                              ctx.h_frames, train_sel)
    ci = ci / float(ctx.cap_ir_scale)
    return dataclasses.replace(ctx, cap_vis=cv, cap_ir=ci,
                               cap_fit_frames=np.asarray(train_sel))


def verify_against_load(ctx, fold_ctx_test, conds) -> dict:
    """Fold 'held-out pohang00' (train = TEST_RUNS) vs a genuine load."""
    real = load_context(preset=PRESET, cache_dir=CACHE, conditions=conds,
                        capability_sel="test", verbose=False)
    dv = abs(real.cap_vis - fold_ctx_test.cap_vis)
    di = abs(real.cap_ir - fold_ctx_test.cap_ir)
    assert dv <= 1e-12 and di <= 1e-12, f"prior mismatch vis {dv:.3e} ir {di:.3e}"
    worst = 0.0
    for c in conds:
        a = run_systems(real, c)["gated_fusion"]["map50_95"]
        b = run_systems(fold_ctx_test, c)["gated_fusion"]["map50_95"]
        worst = max(worst, abs(a - b))
    assert worst <= 1e-12, f"replace vs load fused AP mismatch {worst:.3e}"
    # Power: the check above is only worth something if the swap moves the output.
    moved = abs(run_systems(ctx, "clean")["gated_fusion"]["map50_95"]
                - run_systems(fold_ctx_test, "clean")["gated_fusion"]["map50_95"])
    assert moved > 0.0, "fold prior does not change the fused output -- swap is a no-op"
    print(f"[verify] replace == load on fold pohang00 (prior {max(dv, di):.1e}, "
          f"AP over {len(conds)} conditions {worst:.1e}); swap moves clean AP "
          f"{moved:.1e}", flush=True)
    return {"d_prior": max(dv, di), "d_ap": worst, "moved_clean": moved,
            "cap_vis": real.cap_vis, "cap_ir": real.cap_ir}


def boot(pa, pb, paths, sel, n_boot, cls):
    r = block_bootstrap_delta(pa, pb, paths, BLOCK_LEN, sel=sel, n_boot=n_boot,
                              seed=SEED, cls=cls)
    return {k: r[k] for k in ("a", "b", "delta", "ci_lo", "ci_hi", "se",
                              "sign_flip_fraction", "spans_zero", "n_frames",
                              "n_undefined", "runs")}


def floor_word(d: float, lo: float = FLOOR_LO, hi: float = FLOOR_HI) -> str:
    a = abs(d)
    if a < lo:
        return "below"
    if a <= hi:
        return "within"
    return "ABOVE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="docs/eval/capability_loro_2026-10-08.md")
    ap.add_argument("--ship-floor", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="ship-AP paired floor, if measured; default judges ship "
                         "deltas against the macro floor")
    args = ap.parse_args()
    s_lo, s_hi = args.ship_floor or (FLOOR_LO, FLOOR_HI)
    t0 = time.time()
    conds = tuple(DEFAULT_CONDITIONS)

    ctx = load_context(preset=PRESET, cache_dir=CACHE, conditions=conds, verbose=False)
    assert ctx.inputs["capability_sel"] == "fit", ctx.inputs["capability_sel"]
    assert "pohang04" not in set(ctx.runs.tolist()), "pohang04 must never be scored"
    w_ship = ctx.cap_vis / (ctx.cap_vis + ctx.cap_ir)
    print(f"[ctx] in-sample prior VIS {ctx.cap_vis:.6f} IR {ctx.cap_ir:.6f} "
          f"(÷{ctx.cap_ir_scale:g}) w_vis {w_ship:.6f}", flush=True)

    fit_sel = ctx.sel("fit")
    folds = {}
    for r in DAY_RUNS:
        train = np.flatnonzero(np.isin(ctx.runs, [x for x in DAY_RUNS if x != r]))
        assert not np.intersect1d(train, ctx.sel(r)).size
        fc = fold_context(ctx, train)
        folds[r] = {"ctx": fc, "train": train,
                    "w_vis": fc.cap_vis / (fc.cap_vis + fc.cap_ir),
                    "per_class": per_class_prior(ctx, train)}
        print(f"[fold {r}] train {len(train)} frames  VIS {fc.cap_vis:.6f} "
              f"IR {fc.cap_ir:.6f}  w_vis {folds[r]['w_vis']:.6f}", flush=True)
    insample_pc = per_class_prior(ctx, fit_sel)

    verify = verify_against_load(ctx, folds["pohang00"]["ctx"], conds)

    # ---- score --------------------------------------------------------------
    paths = [r["image_path"] for r in ctx.vis_by_cond["clean"]]
    day = ctx.sel("day")
    test = ctx.sel("test")
    res = {"fold": {}, "pooled": {}}
    w_obs = {}
    for c in conds:
        base_out = run_systems(ctx, c)
        base = frame_parts(base_out["fused_gated"], ctx.gts)
        fold_parts = {}
        for r in DAY_RUNS:
            fo = run_systems(folds[r]["ctx"], c)
            fold_parts[r] = frame_parts(fo["fused_gated"], ctx.gts)
            s = ctx.sel(r)
            w_obs[(r, c)] = (float(np.mean(np.asarray(base_out["w_vis_gated"])[s])),
                             float(np.mean(np.asarray(fo["w_vis_gated"])[s])))
            for tag, cls in (("macro", None), ("ship", SHIP)):
                res["fold"][(r, c, tag)] = boot(fold_parts[r], base, paths, s,
                                                args.n_boot, cls)
        # stitched LORO system: every day frame under the prior that never saw its run
        stitched = list(base)
        for r in DAY_RUNS:
            for i in ctx.sel(r):
                stitched[i] = fold_parts[r][i]
        for split, s in (("day", day), ("test", test)):
            for tag, cls in (("macro", None), ("ship", SHIP)):
                res["pooled"][(split, c, tag)] = boot(stitched, base, paths, s,
                                                      args.n_boot, cls)
        print(f"[cond] {c:9s} day macro "
              f"{res['pooled'][('day', c, 'macro')]['delta']:+.6f}  ship "
              f"{res['pooled'][('day', c, 'ship')]['delta']:+.6f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- report -------------------------------------------------------------
    rel0 = ctx.cap_ir / ctx.cap_vis
    gt_rows = []
    for r in DAY_RUNS:
        cls = np.concatenate([np.asarray(ctx.gts[i]["cls"]).ravel()
                              for i in ctx.sel(r)]).astype(int)
        gt_rows.append([r, str(len(ctx.sel(r))), str(int((cls == SHIP).sum())),
                        str(int((cls == BUOY).sum()))])
    prior_rows = [["in-sample (shipped)", "pohang00+02+03", str(len(fit_sel)),
                   fmt(ctx.cap_vis), fmt(ctx.cap_ir, 6),
                   f"{ctx.cap_vis / ctx.cap_ir:.1f}x", "1.00x", fmt(w_ship, 6),
                   fmt(insample_pc["vis"].get(SHIP, np.nan)),
                   fmt(insample_pc["vis"].get(BUOY, np.nan)),
                   fmt(insample_pc["ir"].get(SHIP, np.nan)),
                   fmt(insample_pc["ir"].get(BUOY, np.nan))]]
    for r in DAY_RUNS:
        f = folds[r]
        fc = f["ctx"]
        tr = "+".join(x.replace("pohang", "") for x in DAY_RUNS if x != r)
        prior_rows.append([f"LORO, held out {r}", f"pohang{tr}", str(len(f["train"])),
                           fmt(fc.cap_vis), fmt(fc.cap_ir, 6),
                           f"{fc.cap_vis / fc.cap_ir:.1f}x",
                           f"{(fc.cap_ir / fc.cap_vis) / rel0:.2f}x", fmt(f["w_vis"], 6),
                           fmt(f["per_class"]["vis"].get(SHIP, np.nan)),
                           fmt(f["per_class"]["vis"].get(BUOY, np.nan)),
                           fmt(f["per_class"]["ir"].get(SHIP, np.nan)),
                           fmt(f["per_class"]["ir"].get(BUOY, np.nan))])

    def row(key, rec, label, tag):
        lo, hi = (s_lo, s_hi) if tag == "ship" else (FLOOR_LO, FLOOR_HI)
        return [label, key, fmt(rec["b"]), fmt(rec["a"]), sgn(rec["delta"]),
                f"[{sgn(rec['ci_lo'])}, {sgn(rec['ci_hi'])}]", fmt(rec["se"]),
                "yes" if rec["spans_zero"] else "**no**",
                floor_word(rec["delta"], lo, hi)]

    hdr = ["scored on", "cell", "in-sample", "LORO", "delta", "95% block CI",
           "se", "CI spans 0", "|delta| vs floor"]
    pooled_rows = {tag: [] for tag in ("macro", "ship")}
    fold_rows = {tag: [] for tag in ("macro", "ship")}
    for tag in ("macro", "ship"):
        for split, lab in (("day", "all day (1,200)"), ("test", "TEST pohang02+03")):
            for c in conds:
                pooled_rows[tag].append(row(f"{c}/day", res["pooled"][(split, c, tag)],
                                            lab, tag))
        for r in DAY_RUNS:
            for c in conds:
                fold_rows[tag].append(row(f"{c}/day", res["fold"][(r, c, tag)],
                                          f"{r} ({len(ctx.sel(r))})", tag))

    w_rows = [[r, c, fmt(w_obs[(r, c)][0], 6), fmt(w_obs[(r, c)][1], 6)]
              for r in DAY_RUNS for c in conds]

    all_d = {tag: [res["pooled"][("day", c, tag)]["delta"] for c in conds]
             for tag in ("macro", "ship")}
    max_fold = {tag: max(abs(res["fold"][(r, c, tag)]["delta"])
                         for r in DAY_RUNS for c in conds) for tag in ("macro", "ship")}
    excl = {tag: [f"{k[0]} {k[1]}" for k, v in res["pooled"].items()
                  if k[2] == tag and not v["spans_zero"]] for tag in ("macro", "ship")}
    excl_f = {tag: [f"{k[0]} {k[1]}" for k, v in res["fold"].items()
                    if k[2] == tag and not v["spans_zero"]] for tag in ("macro", "ship")}

    secs = [
        "Preset `crossmodal26m`, `runs/cache_m`, conditions "
        f"{', '.join(conds)} (VIS stream; IR clean). The shipped prior is fitted with "
        "`capability_sel=\"fit\"` = the 1,200 clean day paired frames, the same frames "
        "every day cell below is scored on. Each LORO fold refits it on the other two "
        "day runs and keeps everything else, including the ÷"
        f"{ctx.cap_ir_scale:g} `cap_ir_scale`, fixed. Deltas are **LORO − in-sample**, "
        f"paired, with a moving-block bootstrap within run (L = {BLOCK_LEN}, "
        f"{args.n_boot} resamples, seed {SEED}). Night (pohang01) is not re-scored: "
        "its prior was already fitted without it.",

        "## 1. Fitted priors\n\n"
        "`IR` is after the ÷4. `IR wt vs shipped` is the fold's IR/VIS weight ratio "
        "divided by the shipped one, i.e. the multiplier on IR's relative weight -- "
        "the axis `runs/eval/reprice_constants.md` priced at ×2 and ×0.5. Per-class "
        "columns are the raw clean AP50-95 behind each prior (before the ÷4); IR's "
        "buoy AP is 0 because the IR detector is single-class, which is why IR's "
        "macro prior is half its ship AP.\n\n"
        + md_table(["prior", "fit runs", "frames", "VIS", "IR", "VIS/IR",
                    "IR wt vs shipped", "w_vis", "VIS ship", "VIS buoy", "IR ship",
                    "IR buoy"],
                   prior_rows, align="llr" + "r" * 9)
        + "\n\nGround truth on the paired day frames, per run. A run with no buoy "
        "boxes scores macro == ship (the missing class is dropped, not scored 0), "
        "and a fold that trains on few buoys gets a noisy VIS buoy term in its "
        "prior:\n\n"
        + md_table(["run", "frames", "ship boxes", "buoy boxes"], gt_rows)
        + f"\n\nVerification: the fold that holds out pohang00 trains on exactly "
        f"`TEST_RUNS`, so it was rebuilt with a genuine "
        f"`load_context(capability_sel=\"test\")`: prior equal to "
        f"{verify['d_prior']:.1e}, fused AP equal to {verify['d_ap']:.1e} on all four "
        "conditions. The `dataclasses.replace` path is the real system. The swap is "
        f"not a no-op: it moves the all-frame clean AP by {verify['moved_clean']:.1e}.",

        "## How to read the floor column\n\n"
        f"`|delta| vs floor`: `below` = under {FLOOR_LO} (the paper's convention "
        "reports this as *not resolved*), `within` = inside the "
        f"{FLOOR_LO}–{FLOOR_HI} band, `ABOVE` = over {FLOOR_HI}. Ship deltas are "
        + (f"judged against a measured ship floor of {s_lo}–{s_hi}."
           if args.ship_floor else
           "judged against the macro floor. The ship-AP floor, measured separately "
           "(`docs/eval/delta_noise_floor_ship_2026-10-08.md`), is 0.0008–0.0024 on the "
           "informative cells; pass `--ship-floor` to judge against it.")
        + " The block CI is this arm's own scene-resampling interval on one "
        "corruption draw (`runs/cache_m`). It omits draw variance, which the floor "
        "includes (draw sd 0.0008 on fog/clean), so a CI that excludes 0 says the "
        "sign is stable over scenes in these recordings, not that the effect is "
        "larger than the floor.",

        "## 2. Pooled: the LORO system against the shipped one\n\n"
        "Each day frame scored under the prior that never saw its run, stitched into "
        "one prediction set, then pooled. This is the number to put next to a "
        "published day cell.\n\n### Macro (ship + buoy)\n\n"
        + md_table(hdr, pooled_rows["macro"], align="ll" + "r" * 7)
        + "\n\n### Ship (class 0)\n\n"
        + md_table(hdr, pooled_rows["ship"], align="ll" + "r" * 7),

        "## 3. Per fold: held-out run only\n\n"
        "Smaller samples (836 / 247 / 117 frames), so wider intervals; a held-out run "
        "is where an in-sample fit would show if it were doing work.\n\n"
        "### Macro (ship + buoy)\n\n"
        + md_table(hdr, fold_rows["macro"], align="ll" + "r" * 7)
        + "\n\n### Ship (class 0)\n\n"
        + md_table(hdr, fold_rows["ship"], align="ll" + "r" * 7),

        "## 4. Observed w_vis on the scored frames\n\n"
        "Mean `w_vis_gated` from `run_systems`, held-out run only. Constant within a "
        "fold, because the weights are the prior alone.\n\n"
        + md_table(["held-out run", "condition", "in-sample", "LORO"], w_rows),

        "## 5. Reading\n\n"
        f"* Largest pooled day |delta|: macro {max(abs(x) for x in all_d['macro']):.4f}, "
        f"ship {max(abs(x) for x in all_d['ship']):.4f}. Largest single-fold |delta|: "
        f"macro {max_fold['macro']:.4f}, ship {max_fold['ship']:.4f}. Macro floor "
        f"(PAPER_DRAFT2 §5.3, paired 2σ on informative cells): {FLOOR_LO}–{FLOOR_HI}.\n"
        f"* Pooled cells whose block CI excludes 0: macro "
        f"{', '.join(excl['macro']) or 'none'}; ship {', '.join(excl['ship']) or 'none'}.\n"
        f"* Fold cells whose block CI excludes 0: macro "
        f"{', '.join(excl_f['macro']) or 'none'}; ship {', '.join(excl_f['ship']) or 'none'}.\n"
        "* Cells at `within` or `ABOVE`: "
        + (", ".join(f"{k[0]} {k[1]} {k[2]} {v['delta']:+.4f}"
                     for part in ("pooled", "fold") for k, v in res[part].items()
                     if floor_word(v["delta"], *((s_lo, s_hi) if k[2] == "ship"
                                                 else (FLOOR_LO, FLOOR_HI))) != "below")
           or "none") + ".",

        f"---\n\n_Generated by `scripts/eval_capability_loro.py` in "
        f"{time.time() - t0:.0f}s._",
    ]
    out = write_md(args.out, "Capability prior: leave-one-run-out refit", secs,
                   identity=system_identity(ctx, block_len=BLOCK_LEN, n_boot=args.n_boot,
                                            seed=SEED, cache=CACHE))
    js = {
        "in_sample": {"cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir, "w_vis": w_ship,
                      "per_class_raw": insample_pc},
        "folds": {r: {"cap_vis": f["ctx"].cap_vis, "cap_ir": f["ctx"].cap_ir,
                      "w_vis": f["w_vis"], "n_train": int(len(f["train"])),
                      "per_class_raw": f["per_class"]} for r, f in folds.items()},
        "verify": verify,
        "fold": {"|".join(k): v for k, v in res["fold"].items()},
        "pooled": {"|".join(k): v for k, v in res["pooled"].items()},
        "w_vis_observed": {f"{r}|{c}": v for (r, c), v in w_obs.items()},
        "block_len": BLOCK_LEN, "n_boot": args.n_boot, "seed": SEED,
        "floor_macro": [FLOOR_LO, FLOOR_HI], "floor_ship": [s_lo, s_hi],
        "ship_floor_measured": bool(args.ship_floor),
        "gt_per_run": {r[0]: {"frames": int(r[1]), "ship": int(r[2]), "buoy": int(r[3])}
                       for r in gt_rows},
    }
    out.with_suffix(".json").write_text(json.dumps(js, indent=2, default=str),
                                         encoding="utf-8")
    print(f"[done] {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
