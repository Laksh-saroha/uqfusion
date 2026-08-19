"""Gate-level ablation sweep over cached predictions (Phase 3; plan B5-3/B5-4).

Consumes two INDEX-ALIGNED caches (VIS stream, IR stream) built from the paired
frame lists that `scripts/build_pairs.py` emits, and sweeps combination rule x α
— pure CPU, no model re-run.

The VIS and IR caches must be index-aligned: `evaluate_systems` pairs them
element by element. Passing an arbitrary VIS val cache and an arbitrary IR val
cache (different splits, different lengths, different instants) is NOT valid and
now fails loudly instead of silently pairing unrelated frames.

Each modality gets its own Mahalanobis scorer and its own reliability constants,
because the two streams come from different checkpoints — see
`uqfusion.eval.fusion_eval.evaluate_systems` for why sharing them makes IR look
permanently unreliable.

Usage:
    python scripts/ablate_gate.py \
        --vis-cache runs/cache/gauss_vis_paired_fog.pkl \
        --ir-cache  runs/cache/gauss_ir_paired_clean.pkl \
        --vis-clean-cache runs/cache/gauss_vis_paired_clean.pkl \
        --vis-fit-cache runs/cache/gauss_vis_train_clean.pkl \
        --ir-fit-cache  runs/cache/gauss_ir_train_clean.pkl \
        --homography runs/derived/homography_ir_to_vis.json \
        --manifest runs/derived/paired_val_manifest.csv \
        --out runs/eval/gate_ablation.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import ablate_gate_rules, format_ablation_table
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty


def _scorer(path: str) -> MahalanobisScorer:
    records, _ = load_cache(path)
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in records]))


def _constants(records, scorer):
    u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                        for r in records if len(r["conf"])])
    d = np.asarray([scorer.score(r["feat"]) for r in records])
    return fit_constants(u, d)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--vis-cache", required=True, help="VIS stream under the condition being gated")
    parser.add_argument("--ir-cache", required=True, help="IR stream, index-aligned with --vis-cache")
    parser.add_argument("--vis-clean-cache", required=True, help="clean VIS val cache — fits VIS constants")
    parser.add_argument("--ir-clean-cache", default=None,
                        help="clean IR val cache — fits IR constants (default: --ir-cache)")
    parser.add_argument("--vis-fit-cache", required=True, help="clean VIS train cache — Mahalanobis fit")
    parser.add_argument("--ir-fit-cache", default=None,
                        help="clean IR train cache — Mahalanobis fit (default: share the VIS scorer)")
    parser.add_argument("--homography", default=None, help="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--manifest", default=None, help="paired manifest, for the per-run homography lookup")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_md = Path(args.out) if args.out else Path(cfg["paths"]["outputs_root"]) / "eval" / "gate_ablation.md"

    vis_records, _ = load_cache(args.vis_cache)
    ir_records, _ = load_cache(args.ir_cache)
    if len(vis_records) != len(ir_records):
        raise SystemExit(
            f"caches are not index-aligned: {len(vis_records)} VIS frames vs {len(ir_records)} IR frames.\n"
            "Build both from the paired lists (scripts/build_pairs.py) — a raw VIS val cache and a raw "
            "IR val cache observe different instants and cannot be fused."
        )
    clean_vis, _ = load_cache(args.vis_clean_cache)
    clean_ir, _ = load_cache(args.ir_clean_cache) if args.ir_clean_cache else (ir_records, None)

    scorer_vis = _scorer(args.vis_fit_cache)
    scorer_ir = _scorer(args.ir_fit_cache) if args.ir_fit_cache else scorer_vis
    c_vis = _constants(clean_vis, scorer_vis)
    c_ir = _constants(clean_ir, scorer_ir)
    print(f"[ablate] VIS constants: lam={c_vis.lam:.3f}, mu_d={c_vis.mu_d:.2f}, tau={c_vis.tau:.2f}")
    print(f"[ablate] IR  constants: lam={c_ir.lam:.3f}, mu_d={c_ir.mu_d:.2f}, tau={c_ir.tau:.2f}")

    h_frames = None
    if args.homography and args.manifest:
        from run_fusion_eval import per_frame_homographies

        h_frames = per_frame_homographies(Path(args.manifest), Path(args.homography))
        print(f"[ablate] per-run homographies loaded for {len(h_frames)} frames")

    rows = ablate_gate_rules(vis_records, ir_records, scorer_vis, c_vis, h_ir_to_vis=h_frames,
                             scorer_ir=scorer_ir, base_constants_ir=c_ir)
    table = format_ablation_table(rows)
    print(table)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print(f"[ablate] written -> {out_md}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))  # allow `from run_fusion_eval import ...`
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
