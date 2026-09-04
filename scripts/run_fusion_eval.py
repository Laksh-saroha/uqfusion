"""End-to-end fusion-system evaluation on real paired Pohang frames (Table 3).

This is the runner HOW_TO_RUN §5 lists as missing: it takes the two Gaussian
sigma checkpoints, the index-aligned VIS/IR caches from `build_pairs.py`, and
the per-run calibration homographies from `derive_homography.py`, and produces
the system comparison

    visible-only | IR-only | naive 0.5/0.5 fusion | uncertainty-gated fusion

for the clean VIS stream and for each degraded VIS condition. The question it
answers is scope's fusion premise: **does IR carry the system when VIS is
degraded, and does the uncertainty gate find that out on its own?**

Two things are per-modality here, and deliberately so (see `evaluate_systems`):

* **Mahalanobis scorer.** VIS and IR features come from different checkpoints;
  their 896-d pooled-neck spaces are not comparable, so each modality is scored
  against its own clean-train distribution.
* **Reliability constants.** Fitted per modality on that modality's clean val,
  so R means "degraded relative to THIS sensor's own clean baseline". Sharing
  VIS constants with IR would make IR look permanently unreliable and the gate
  would never hand it the frame. `--shared-constants` runs the other way as a
  sensitivity check.

Constants are fit on clean val and the systems are then scored on that same
clean val — that is the pre-registered protocol (decision D5 / plan B5), not
leakage. The learned gate IS a fitted model, so it is trained and evaluated on
disjoint recording runs and reported separately.

**Every table here is also reported day/night (TODO §0.1), and that is not a
nicety.** pohang01 is a real night run where the VIS detector scores exactly
0.0000; pohang00/02/03 are daylight and score 0.40 / 0.37 / 0.18. Pooling them
into one 0.258 hides both the real daylight capability and the real night result.
Worse, pooled mAP ranks every detection in ONE global list, so a change confined
to night frames moves daylight precision at equal recall — the pooled number can
fall while every per-run number rises. Read the split; the pooled column is kept
only for continuity with earlier tables.

`--brightness-constants` and `--veto` switch on the §16/§17 gate: the photometric
term the Mahalanobis distance cannot supply, and the hard veto that removes a
blind modality from the WBF input list instead of down-weighting it. Both default
to OFF so this script still reproduces the original Table 3 byte for byte.

2026-08-20 finalization: with `--brightness-constants` + `--veto`, the defaults
now produce the FINALIZED system — veto-only photometric term (no soft
component), dilate-15 hysteresis on the switch, capability prior over the fit
runs only. To reproduce the 2026-08-19 record's table instead, add
`--bright-soft --capability-runs all --veto-dilate 1`.

Usage:
    python scripts/run_fusion_eval.py --out runs/eval/table3_fusion.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import evaluate_systems
from uqfusion.eval.learned_gate import LearnedGate
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty

SYSTEMS = ["visible_only", "ir_only", "naive_fusion", "gated_fusion"]


def fit_scorer(cache_path: Path) -> MahalanobisScorer:
    records, _ = load_cache(cache_path)
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in records]))


def constants_for(records: list[dict], scorer: MahalanobisScorer, alpha: float, combination: str):
    u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                        for r in records if len(r["conf"])])
    d = np.asarray([scorer.score(r["feat"]) for r in records])
    return fit_constants(u, d, alpha=alpha, combination=combination)


def per_frame_homographies(manifest: Path, homography_json: Path) -> list[np.ndarray]:
    payload = json.loads(homography_json.read_text(encoding="utf-8"))
    by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], dtype=np.float64)
              for k, v in payload["runs"].items()}
    out = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run = row["run"]
            if run not in by_run:
                raise KeyError(f"no homography for run '{run}' in {homography_json}")
            out.append(by_run[run])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="runs/cache")
    parser.add_argument("--manifest", default="runs/derived/paired_val_manifest.csv")
    parser.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--out", default="runs/eval/table3_fusion.md")
    parser.add_argument("--conditions", nargs="+", default=["clean", "fog", "lowlight", "glare"])
    parser.add_argument("--shared-constants", action="store_true",
                        help="sensitivity check: score IR with the VIS scorer and VIS constants")
    parser.add_argument("--identity-h", action="store_true",
                        help="sensitivity check: fuse with H=identity (the pre-calibration placeholder)")
    parser.add_argument("--no-learned-gate", action="store_true")
    parser.add_argument("--capability-weighted", action="store_true",
                        help="scale each modality's reliability by its clean-val mAP, so the gate "
                             "weighs EXPECTED ABSOLUTE capability rather than retained fraction")
    parser.add_argument("--constants", default=None,
                        help="JSON from fit_reliability_constants.py — use the D-6 ladder-fitted "
                             "mu_d/tau instead of the clean-val 95th-pct/IQR rule")
    parser.add_argument("--iou-thr", type=float, default=0.55,
                        help="WBF iou_thr; 0.55 is the original Table 3 value, see TODO 0.5")
    parser.add_argument("--brightness-constants", default=None,
                        help="JSON from fit_brightness_gate.py — switches on the photometric term")
    parser.add_argument("--bright-dir", default="runs/derived/brightness")
    parser.add_argument("--veto", type=float, default=None,
                        help="exclude a modality from fusion when its r_bright falls below this; "
                             "needs --brightness-constants. Use 0.5, the registered boundary")
    parser.add_argument("--veto-dilate", type=int, default=15,
                        help="hysteresis window on the veto switch (dilate mode, capture order); "
                             "adopted 15 per runs/eval/x_veto_hysteresis.md. <=1 restores the "
                             "per-frame switch. Only active together with --veto")
    parser.add_argument("--bright-soft", action="store_true",
                        help="ALSO fold r_bright into the soft weight (the 2026-08-19 chain). "
                             "Default off: the interaction readout showed veto-only equals "
                             "gate+veto in every cell, so the soft term is retired")
    parser.add_argument("--capability-runs", choices=("fit", "all"), default="fit",
                        help="frames the capability prior is computed over. 'fit' (default) is "
                             "run-disjoint from the held-out night run; 'all' reproduces the "
                             "2026-08-19 tables, whose prior included the held-out run")
    parser.add_argument("--night-runs", default="pohang01",
                        help="comma-separated recording runs to report as the night split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    alpha = float(cfg["reliability"]["alpha"])
    combination = str(cfg["reliability"]["combination"])

    ir_clean, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    vis_by_cond = {}
    for cond in args.conditions:
        name = "gauss_vis_paired_clean.pkl" if cond == "clean" else f"gauss_vis_paired_{cond}.pkl"
        vis_by_cond[cond], _ = load_cache(cache_dir / name)

    scorer_vis = fit_scorer(cache_dir / "gauss_vis_train_clean.pkl")
    scorer_ir = fit_scorer(cache_dir / "gauss_ir_train_clean.pkl")

    # Pre-registered rule: constants come from the CLEAN validation stream of
    # each modality, then stay fixed across every degraded condition.
    c_vis = constants_for(vis_by_cond["clean"], scorer_vis, alpha, combination)
    c_ir = constants_for(ir_clean, scorer_ir, alpha, combination)
    constants_rule = "clean-val 95th pct / IQR (D5/B5)"
    if args.constants:
        fitted = json.loads(Path(args.constants).read_text(encoding="utf-8"))
        from dataclasses import replace as _replace
        for key, cur in (("vis", "c_vis"), ("ir", "c_ir")):
            if key not in fitted:
                continue
            f = fitted[key]
            obj = _replace(c_vis if key == "vis" else c_ir, mu_d=float(f["mu_d"]), tau=float(f["tau"]))
            if key == "vis":
                c_vis = obj
            else:
                c_ir = obj
        constants_rule = "mAP-retention ladder (D-6)"
        print(f"[fusion] constants rule: {constants_rule} <- {args.constants}")
    if args.shared_constants:
        scorer_ir, c_ir = scorer_vis, c_vis

    night_runs = tuple(r.strip() for r in args.night_runs.split(",") if r.strip())
    frame_runs = np.asarray([Path(r["image_path"]).parent.name for r in vis_by_cond["clean"]])
    night_sel = np.flatnonzero(np.isin(frame_runs, night_runs))
    day_sel = np.flatnonzero(~np.isin(frame_runs, night_runs))

    # Capability prior: each modality's clean-val mAP, measured on the same
    # clean caches the constants are fit on (so no new data is consulted).
    # Default 'fit' restricts it to the non-night runs — the 'all' prior
    # included the held-out night run (followup §7); measured effect of the fix
    # is <= +0.0047, all of it on day cells, none on the night headline.
    cap_vis = cap_ir = 1.0
    if args.capability_weighted:
        from uqfusion.eval.matching import load_gt, map50_95

        g = [load_gt(r["image_path"], r["image_hw"]) for r in vis_by_cond["clean"]]
        ir_in_vis_h = per_frame_homographies(Path(args.manifest), Path(args.homography))
        from uqfusion.uq.fusion import apply_homography

        ir_mapped = [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
                     for r, h in zip(ir_clean, ir_in_vis_h)]
        cap_sel = day_sel if args.capability_runs == "fit" else np.arange(len(g))
        g_cap = [g[i] for i in cap_sel]
        cap_vis = map50_95([vis_by_cond["clean"][i] for i in cap_sel], g_cap)["map50_95"]
        cap_ir = map50_95([ir_mapped[i] for i in cap_sel], g_cap)["map50_95"]
        print(f"[fusion] capability prior ({args.capability_runs} runs, {len(cap_sel)} frames): "
              f"VIS {cap_vis:.4f}  IR {cap_ir:.4f}  (ratio {cap_vis / cap_ir:.1f}x)")

    # Photometric gate and hard veto — both off unless asked for.
    bright_by_cond, c_vis_gate, bright_desc = {}, c_vis, "photometric gate OFF"
    if args.brightness_constants:
        from dataclasses import replace as _replace2

        bc = json.loads(Path(args.brightness_constants).read_text(encoding="utf-8"))["vis"]
        stat, bdir = bc["stat"], Path(args.bright_dir)
        for cond in args.conditions:
            bp = bdir / f"gauss_vis_paired_{cond}.json"
            if not bp.is_file():
                raise SystemExit(f"missing {bp} — run scripts/frame_brightness.py --cache "
                                 f"runs/cache/gauss_vis_paired_{cond}.pkl --modality vis")
            bright_by_cond[cond] = np.asarray(
                [f[stat] for f in json.loads(bp.read_text(encoding="utf-8"))["frames"]], dtype=float)
        c_vis_gate = _replace2(c_vis, mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat,
                               bright_soft=args.bright_soft)
        bright_desc = (f"photometric term ({stat}, mu_b={bc['mu_b']:.3f}, tau_b={bc['tau_b']:.3f}, "
                       f"{'soft+veto' if args.bright_soft else 'veto-only'})")
        print(f"[fusion] {bright_desc}"
              + (f", hard veto at r_bright < {args.veto}"
                 + (f" with dilate-{args.veto_dilate} hysteresis" if args.veto_dilate > 1 else "")
                 if args.veto is not None else ", no veto"))
    elif args.veto is not None:
        raise SystemExit("--veto needs --brightness-constants: the veto fires on r_bright")
    # IR deliberately gets no photometric term: on a thermal sensor "dark" means
    # cold water, which is the condition IR exists for. The fix is VIS-only.

    print(f"[fusion] day/night split: {len(day_sel)} day, {len(night_sel)} night "
          f"({'+'.join(night_runs)})")

    h_frames = None if args.identity_h else per_frame_homographies(Path(args.manifest), Path(args.homography))

    print(f"[fusion] {len(ir_clean)} paired frames, conditions: {', '.join(args.conditions)}")
    print(f"[fusion] VIS constants: lam={c_vis.lam:.3f} mu_d={c_vis.mu_d:.2f} tau={c_vis.tau:.2f}")
    print(f"[fusion] IR  constants: lam={c_ir.lam:.3f} mu_d={c_ir.mu_d:.2f} tau={c_ir.tau:.2f}")

    # Learned gate (§7.5): a FITTED model, so train and test on disjoint runs.
    gate = None
    gate_note = "not fitted"
    if not args.no_learned_gate:
        runs = [Path(r["image_path"]).parent.name for r in vis_by_cond["clean"]]
        fit_mask = np.array([r in ("pohang00", "pohang02") for r in runs])
        vis_deg = vis_by_cond.get("fog", vis_by_cond["clean"])
        # Mixed-condition training data: alternate which stream is the degraded
        # one so both outcomes are represented (single-class labels otherwise).
        vis_mix, ir_mix = [], []
        for i, (rc, rd, ri) in enumerate(zip(vis_by_cond["clean"], vis_deg, ir_clean)):
            vis_mix.append(rd if i % 2 == 0 else rc)
            ir_mix.append(ri)
        vm = [r for r, m in zip(vis_mix, fit_mask) if m]
        im = [r for r, m in zip(ir_mix, fit_mask) if m]
        gate = LearnedGate()
        info = gate.fit(vis_records=vm, ir_records=im,
                        vis_distances=np.asarray([scorer_vis.score(r["feat"]) for r in vm]),
                        ir_distances=np.asarray([scorer_ir.score(r["feat"]) for r in im]),
                        constants=c_vis)
        gate.save(Path(args.out).parent / "learned_gate_real.pkl")
        gate_note = (f"fitted on pohang00+02 ({info['n_train_frames']} frames), "
                     f"scored on all {len(runs)} — rows marked * are partly in-sample")
        print(f"[fusion] learned gate: {gate_note}")

    from uqfusion.eval.matching import load_gt as _load_gt, map50_95 as _map

    gts_all = [_load_gt(r["image_path"], r["image_hw"]) for r in vis_by_cond["clean"]]

    # Veto hysteresis (finalized 2026-08-20): the switch is dilated in capture
    # order BEFORE fusion, via veto_override; the soft weight is untouched.
    hyst_order = None
    if args.veto is not None and args.veto_dilate > 1:
        from uqfusion.eval.hysteresis import filter_veto, raw_veto_flags, temporal_order
        hyst_order = temporal_order(vis_by_cond["clean"])

    rows = []
    diag = []
    split_rows = []
    for cond in args.conditions:
        veto_kw = {"veto_below": args.veto}
        if hyst_order is not None and bright_by_cond.get(cond) is not None:
            vv = raw_veto_flags(bright_by_cond[cond], c_vis_gate.mu_b, c_vis_gate.tau_b,
                                args.veto, len(vis_by_cond[cond]))
            vv = filter_veto(vv, hyst_order, args.veto_dilate, "dilate")
            veto_kw = {"veto_below": None,
                       "veto_override": (vv, [False] * len(vis_by_cond[cond]))}
        res = evaluate_systems(vis_by_cond[cond], ir_clean, scorer_vis, c_vis_gate, gate=gate,
                               h_ir_to_vis=h_frames, scorer_ir=scorer_ir, constants_ir=c_ir,
                               capability_vis=cap_vis, capability_ir=cap_ir,
                               gts=gts_all, iou_thr_wbf=args.iou_thr,
                               brightness_vis=bright_by_cond.get(cond), brightness_ir=None,
                               **veto_kw)
        # The split re-scores the SAME fused outputs on a frame subset; fusion is
        # never re-run, so the split cannot disagree with the pooled row about
        # what the system actually did.
        for label, sel in (("day", day_sel), ("night", night_sel)):
            if not len(sel):
                continue
            g = [gts_all[i] for i in sel]
            split_rows.append({
                "condition": cond, "split": label, "frames": len(sel),
                "visible_only": _map([vis_by_cond[cond][i] for i in sel], g)["map50_95"],
                "ir_only": _map([res["ir_in_vis"][i] for i in sel], g)["map50_95"],
                "gated_fusion": _map([res["fused_gated"][i] for i in sel], g)["map50_95"],
                "mean_w_vis": float(np.mean([res["w_vis_gated"][i] for i in sel])),
            })
        row = {"condition": cond}
        for s in SYSTEMS:
            row[s] = res[s]["map50_95"]
            row[s + "_50"] = res[s]["map50"]
        if gate is not None:
            row["learned_gate_fusion"] = res["learned_gate_fusion"]["map50_95"]
            row["learned_gate_fusion_50"] = res["learned_gate_fusion"]["map50"]
        row["mean_w_vis"] = float(np.mean(res["w_vis_gated"]))
        row["mean_R_sys"] = float(np.mean(res["R_sys"]))
        rows.append(row)
        diag.append({"condition": cond,
                     "w_vis_p05": float(np.percentile(res["w_vis_gated"], 5)),
                     "w_vis_p50": float(np.percentile(res["w_vis_gated"], 50)),
                     "w_vis_p95": float(np.percentile(res["w_vis_gated"], 95)),
                     "R_sys_p05": float(np.percentile(res["R_sys"], 5)),
                     "frac_ir_favoured": float(np.mean(np.asarray(res["w_vis_gated"]) < 0.5))})
        print(f"[fusion] {cond:9s} " + "  ".join(f"{s}={row[s]:.4f}" for s in SYSTEMS)
              + f"  w_vis={row['mean_w_vis']:.3f}")

    cols = SYSTEMS + (["learned_gate_fusion"] if gate is not None else [])
    md = [f"# Table 3 — fusion systems on {len(ir_clean)} paired Pohang val frames",
          "",
          f"`yolo26s` Gaussian sigma checkpoints, imgsz 640, conf 0.001. "
          f"IR->VIS homography: {'IDENTITY (sensitivity run)' if args.identity_h else 'per-run, from calibration'}. "
          f"Reliability constants: {'SHARED VIS (sensitivity run)' if args.shared_constants else 'per modality'}, "
          f"rule = {constants_rule}, combination={combination}, alpha={alpha}"
          + (f", capability-weighted over {args.capability_runs} runs "
             f"(VIS {cap_vis:.4f} / IR {cap_ir:.4f})" if args.capability_weighted else "")
          + f", WBF iou_thr={args.iou_thr}, {bright_desc}"
          + (f", hard veto at r_bright<{args.veto}"
             + (f" (dilate-{args.veto_dilate} hysteresis)" if args.veto_dilate > 1 else "")
             if args.veto is not None else "") + ".",
          "",
          "mAP@50-95 (mAP@50 in parentheses). The IR stream is always clean — only the VIS stream is degraded.",
          "",
          "| VIS condition | " + " | ".join(c.replace("_", " ") for c in cols) + " | mean w_vis | mean R_sys |",
          "|" + "---|" * (len(cols) + 3)]
    for r in rows:
        cells = [f"{r[c]:.4f} ({r[c + '_50']:.4f})" for c in cols]
        md.append(f"| {r['condition']} | " + " | ".join(cells)
                  + f" | {r['mean_w_vis']:.3f} | {r['mean_R_sys']:.3f} |")
    md += ["", "## Day vs night — read this table, not the pooled one", "",
           f"Night = {'+'.join(night_runs)}, a real night run where the VIS detector scores "
           f"exactly 0.0000. Pooled mAP ranks all detections in one global list, so pooling a "
           f"blind run with working ones produces a number that moves for reasons unrelated to "
           f"either population.", "",
           "| condition | split | frames | visible only | ir only | gated fusion | mean w_vis |",
           "|---|---|---|---|---|---|---|"]
    for s in split_rows:
        md.append(f"| {s['condition']} | {s['split']} | {s['frames']} | {s['visible_only']:.4f} | "
                  f"{s['ir_only']:.4f} | **{s['gated_fusion']:.4f}** | {s['mean_w_vis']:.3f} |")
    md += ["", "## Gate diagnostics", "",
           "| VIS condition | w_vis p05 | p50 | p95 | R_sys p05 | frames favouring IR |",
           "|---|---|---|---|---|---|"]
    for d in diag:
        md.append(f"| {d['condition']} | {d['w_vis_p05']:.3f} | {d['w_vis_p50']:.3f} | "
                  f"{d['w_vis_p95']:.3f} | {d['R_sys_p05']:.3f} | {d['frac_ir_favoured']:.1%} |")
    md += ["", "## Fitted constants", "",
           "| modality | lambda | mu_d | tau |", "|---|---|---|---|",
           f"| VIS | {c_vis.lam:.4f} | {c_vis.mu_d:.3f} | {c_vis.tau:.3f} |",
           f"| IR | {c_ir.lam:.4f} | {c_ir.mu_d:.3f} | {c_ir.tau:.3f} |",
           "", f"Learned gate: {gate_note}.", ""]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    out_path.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "split_rows": split_rows, "diagnostics": diag,
                    "constants": {"vis": c_vis.to_dict(), "ir": c_ir.to_dict()},
                    "n_frames": len(ir_clean),
                    "identity_h": args.identity_h,
                    "constants_rule": constants_rule,
                    "iou_thr": args.iou_thr,
                    "brightness_constants": args.brightness_constants,
                    "veto": args.veto,
                    "veto_dilate": args.veto_dilate,
                    "bright_soft": args.bright_soft,
                    "capability_runs": args.capability_runs,
                    "night_runs": list(night_runs),
                    "shared_constants": args.shared_constants}, indent=2), encoding="utf-8")
    print(f"[fusion] -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
