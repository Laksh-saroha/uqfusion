"""Does a better IR detector actually move the fusion night result?

§9.3 of the experiment record sets the premise: on the night run `ir_only` is
0.0810 and gated fusion is 0.0813, so **no further gate work is worth anything
there — the only way night moves is raising `ir_only`.** The screen raised it:
`p2feat` is +9% ship AP on IR, measured at two settings.

Nothing about that reaches fusion by itself. The fusion evaluation runs off cached
predictions from `runs/phase2/gauss_ir_seed0/weights/best.pt`; a better checkpoint
changes nothing until its predictions are cached and the table re-run. This script
does exactly that and nothing else:

  1. rebuild the two IR caches from the new checkpoint, using the SAME image lists,
     imgsz and conf the originals were built with (read off the existing caches'
     metadata rather than assumed)
  2. hardlink the VIS caches unchanged, so VIS is provably identical
  3. re-run the adopted configuration on the new cache directory
  4. report `ir_only` and gated fusion, before and after, with paired intervals

The VIS side being byte-identical is what makes this a clean single-variable test:
any movement is the IR detector.

Usage:
    python scripts/eval_ir_upgrade_fusion.py --weights runs/screen3/s3_p2feat_640_b10/weights/best.pt --tag p2feat
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402

BASE_CACHE = ROOT / "runs" / "cache"
VIS_CACHES = ("gauss_vis_paired_clean.pkl", "gauss_vis_paired_fog.pkl",
              "gauss_vis_paired_lowlight.pkl", "gauss_vis_paired_glare.pkl",
              "gauss_vis_train_clean.pkl")
IR_CACHES = ("gauss_ir_paired_clean.pkl", "gauss_ir_train_clean.pkl")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_ir_cache(weights: Path, name: str, out_dir: Path, ir_list: str | None) -> None:
    """Rebuild one IR cache with the new weights, matching the original's build args."""
    _, meta = load_cache(BASE_CACHE / name)
    images_list = ir_list or meta["images_list"]
    cmd = [sys.executable, str(ROOT / "scripts" / "build_cache.py"),
           "--source", "gaussian", "--weights", str(weights),
           "--conf", str(meta["conf"]), "--imgsz", str(meta["imgsz"]),
           "--images-list", images_list, "--out", str(out_dir / name)]
    print(f"[upgrade] {name}: {images_list} @ imgsz {meta['imgsz']} conf {meta['conf']}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"build_cache failed for {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True, help="cache dir suffix, e.g. p2feat")
    ap.add_argument("--ir-images-list", default=None,
                    help="override the IR image list (needed when the pixels changed, e.g. CLAHE)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    weights = ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")
    out_md = ROOT / (args.out or f"runs/eval/x_ir_upgrade_{args.tag}.md")
    new_dir = ROOT / "runs" / f"cache_{args.tag}"
    new_dir.mkdir(parents=True, exist_ok=True)

    for n in VIS_CACHES:
        link_or_copy(BASE_CACHE / n, new_dir / n)
    print(f"[upgrade] VIS caches hardlinked unchanged into {new_dir.name}")
    for n in IR_CACHES:
        if not (new_dir / n).is_file():
            build_ir_cache(weights, n, new_dir, args.ir_images_list)
    print(f"[upgrade] caches ready ({time.time() - t0:.0f}s)", flush=True)

    ctx_old = load_context()
    ctx_new = load_context(cache_dir=f"runs/cache_{args.tag}", verbose=False)
    splits = {"day": ctx_old.sel("day"), "night": ctx_old.sel("night")}
    print(f"[upgrade] capability prior: old IR {ctx_old.cap_ir:.4f} -> new IR {ctx_new.cap_ir:.4f}")

    rows, boots = [], {}
    for cond in ctx_old.conditions:
        ro = run_systems(ctx_old, cond)
        rn = run_systems(ctx_new, cond)
        po_g = frame_parts(ro["fused_gated"], ctx_old.gts)
        pn_g = frame_parts(rn["fused_gated"], ctx_old.gts)
        po_i = frame_parts(ro["ir_in_vis"], ctx_old.gts)
        pn_i = frame_parts(rn["ir_in_vis"], ctx_old.gts)
        for sname, sel in splits.items():
            rows.append({
                "condition": cond, "split": sname,
                "ir_only_old": ap_from_parts(po_i, sel)["map50_95"],
                "ir_only_new": ap_from_parts(pn_i, sel)["map50_95"],
                "gated_old": ap_from_parts(po_g, sel)["map50_95"],
                "gated_new": ap_from_parts(pn_g, sel)["map50_95"],
            })
            boots[("gated", cond, sname)] = bootstrap_delta(pn_g, po_g, sel, n_boot=args.n_boot)
            boots[("ir_only", cond, sname)] = bootstrap_delta(pn_i, po_i, sel, n_boot=args.n_boot)
            # the claim §9.3 cares about: does gated fusion still beat ir_only once
            # ir_only itself has been raised?
            boots[("headline", cond, sname)] = bootstrap_delta(pn_g, pn_i, sel, n_boot=args.n_boot)
        print(f"[upgrade] {cond:9s} " + "  ".join(
            f"{r['split']}: ir {r['ir_only_old']:.4f}->{r['ir_only_new']:.4f} "
            f"gated {r['gated_old']:.4f}->{r['gated_new']:.4f}"
            for r in rows if r["condition"] == cond), flush=True)

    L = [f"# Fusion with an upgraded IR detector — `{args.tag}`", "",
         f"IR checkpoint: `{args.weights}`. VIS caches are hardlinks to the originals, so "
         f"the VIS stream is byte-identical and any movement below is the IR detector. "
         f"Image lists, imgsz and conf were read from the original caches' metadata.",
         "",
         f"Capability prior: IR {ctx_old.cap_ir:.4f} -> {ctx_new.cap_ir:.4f} "
         f"(VIS unchanged at {ctx_old.cap_vis:.4f}).",
         "",
         "## 1. `ir_only` — the ceiling §9.3 says is the only thing that can move night", "",
         "| condition | split | before | after | delta | 95% CI | sign flips |",
         "|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        b = boots[("ir_only", r["condition"], r["split"])]
        L.append(f"| {r['condition']} | {r['split']} | {r['ir_only_old']:.4f} | {r['ir_only_new']:.4f} | "
                 f"{b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")

    L += ["", "## 2. Gated fusion", "",
          "| condition | split | before | after | delta | 95% CI | sign flips |",
          "|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        b = boots[("gated", r["condition"], r["split"])]
        L.append(f"| {r['condition']} | {r['split']} | {r['gated_old']:.4f} | {r['gated_new']:.4f} | "
                 f"{b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")

    L += ["", "## 3. Does gated fusion still beat `ir_only` after the upgrade?", "",
          "The paper's claim is gated > `ir_only` on night. Raising `ir_only` raises the bar "
          "it has to clear, so this is the number that decides whether the claim survives a "
          "better IR detector.",
          "",
          "| condition | split | gated | ir_only | delta | 95% CI | verdict |",
          "|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        b = boots[("headline", r["condition"], r["split"])]
        verdict = "spans zero" if b["spans_zero"] else ("gated wins" if b["delta"] > 0 else "ir_only wins")
        L.append(f"| {r['condition']} | {r['split']} | {b['a']:.4f} | {b['b']:.4f} | "
                 f"{b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {verdict} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps({
        "weights": str(args.weights), "tag": args.tag, "rows": rows,
        "cap_ir_old": ctx_old.cap_ir, "cap_ir_new": ctx_new.cap_ir,
        "bootstrap": {"|".join(map(str, k)): v for k, v in boots.items()}}, indent=2), encoding="utf-8")
    print(f"[upgrade] wrote {out_md} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
