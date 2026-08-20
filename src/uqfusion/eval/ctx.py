"""Shared loader for the fusion evaluation context — one place, one config.

Every analysis that asks "what does the adopted system do?" needs the same eight
things loaded the same way: the paired caches, two Mahalanobis scorers, two sets
of reliability constants, the photometric constants, the per-run homographies,
the GT, the day/night split, and the capability prior. `run_fusion_eval.py`
builds all of that inline, which is fine for one script and a liability for nine:
§6.2 of `docs/fusion-gate-experiment-record.md` records a bug where the headline
table used the D5/B5 constants while every analysis used the D-6 ladder fit, and
gated glare read 0.0641 instead of 0.2058. That happened because the setup was
duplicated. This module exists so it cannot happen again.

`load_context()` reproduces the ADOPTED configuration by default. As of the
2026-08-20 finalization (`docs/architecture-final-2026-08-20.md`) that is:

    capability-weighted WBF, iou_thr 0.85, sigma-weighted available
    D-6 ladder constants (runs/eval/reliability_constants.json)
    photometric term VETO-ONLY (bright_soft=False; followup §3: the soft term
        is redundant — `veto_only` equals gate+veto in every cell)
    hard veto at r_bright < 0.5 with DILATE-15 hysteresis in capture order
        (followup §4: fog/night +0.0020 CI [+0.0007, +0.0028], guard unmoved)
    capability prior computed RUN-DISJOINT over the fit runs (followup §7:
        the all-frames prior included the held-out night run)

To reproduce the 2026-08-19 record exactly instead, pass
`capability_sel="all", bright_soft=True, veto_filter=None`.

The learned gate is deliberately NOT loaded: it is a fitted model reported
separately, and none of the follow-up tests ablate it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import evaluate_systems
from uqfusion.eval.hysteresis import ADOPTED_VETO_FILTER, filter_veto, raw_veto_flags, temporal_order
from uqfusion.eval.matching import load_gt
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import ReliabilityConstants, fit_constants, per_box_uncertainty

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONDITIONS = ("clean", "fog", "lowlight", "glare")
NIGHT_RUNS = ("pohang01",)
#: Runs the gate was allowed to see. pohang01 is held out of every fit (§7).
FIT_RUNS = ("pohang00", "pohang02", "pohang03")


def fit_scorer(cache_path: Path) -> MahalanobisScorer:
    records, _ = load_cache(cache_path)
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in records]))


def constants_for(records, scorer, alpha, combination) -> ReliabilityConstants:
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


@dataclass
class FusionContext:
    vis_by_cond: dict[str, list[dict]]
    ir_clean: list[dict]
    scorer_vis: MahalanobisScorer
    scorer_ir: MahalanobisScorer
    c_vis: ReliabilityConstants          # photometric constants already merged in
    c_ir: ReliabilityConstants
    bright_by_cond: dict[str, np.ndarray]
    h_frames: list
    gts: list[dict]
    runs: np.ndarray
    cap_vis: float
    cap_ir: float
    conditions: tuple[str, ...]
    iou_thr: float
    veto: float | None
    veto_filter: tuple[str, int] | None = None   # (mode, k) hysteresis on the switch
    cap_note: str = ""
    _sel: dict = field(default_factory=dict)
    _order: dict | None = None

    @property
    def order(self) -> dict[str, np.ndarray]:
        """Capture order per run, computed once (needed by the veto filter)."""
        if self._order is None:
            self._order = temporal_order(self.vis_by_cond["clean"])
        return self._order

    # ---- frame selectors -------------------------------------------------
    def sel(self, name: str) -> np.ndarray:
        """Frame indices for 'all', 'day', 'night', 'fit', or a run id."""
        if name in self._sel:
            return self._sel[name]
        if name == "all":
            v = np.arange(len(self.runs))
        elif name == "night":
            v = np.flatnonzero(np.isin(self.runs, NIGHT_RUNS))
        elif name == "day":
            v = np.flatnonzero(~np.isin(self.runs, NIGHT_RUNS))
        elif name == "fit":
            v = np.flatnonzero(np.isin(self.runs, FIT_RUNS))
        else:
            v = np.flatnonzero(self.runs == name)
            if not len(v):
                raise KeyError(f"no frames for selector {name!r}")
        self._sel[name] = v
        return v

    @property
    def run_ids(self) -> list[str]:
        return sorted(set(self.runs.tolist()))

    def n(self) -> int:
        return len(self.gts)


def capability_prior(vis_clean, ir_clean, gts, h_frames, sel=None) -> tuple[float, float]:
    """Each modality's clean mAP on THIS evaluation (boxes in the VIS frame, VIS GT).

    §6.1: it must not be read off `reliability_constants.json`, whose `map_clean`
    is IR's score on the IR ladder against IR GT — a different task on different
    frames, 3.3x higher.

    `sel` restricts the frames the prior is computed over. The adopted table uses
    every clean frame, which includes the held-out night run; passing
    `ctx.sel("fit")` gives the run-disjoint alternative.
    """
    from uqfusion.eval.matching import map50_95
    from uqfusion.uq.fusion import apply_homography

    ir_mapped = [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
                 for r, h in zip(ir_clean, h_frames)]
    if sel is not None:
        sel = np.asarray(sel)
        vis_clean = [vis_clean[i] for i in sel]
        ir_mapped = [ir_mapped[i] for i in sel]
        gts = [gts[i] for i in sel]
    return (float(map50_95(vis_clean, gts)["map50_95"]),
            float(map50_95(ir_mapped, gts)["map50_95"]))


def load_context(
    conditions=DEFAULT_CONDITIONS,
    cache_dir="runs/cache",
    manifest="runs/derived/paired_val_manifest.csv",
    homography="runs/derived/homography_ir_to_vis.json",
    constants="runs/eval/reliability_constants.json",
    brightness_constants="runs/eval/brightness_constants.json",
    bright_dir="runs/derived/brightness",
    iou_thr: float = 0.85,
    veto: float | None = 0.5,
    capability_sel: str | None = "fit",
    bright_soft: bool = False,
    veto_filter: tuple[str, int] | None = ADOPTED_VETO_FILTER,
    config=None,
    verbose: bool = True,
) -> FusionContext:
    cfg = load_config(config)
    cache_dir = ROOT / cache_dir
    alpha = float(cfg["reliability"]["alpha"])
    combination = str(cfg["reliability"]["combination"])

    ir_clean, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    vis_by_cond = {}
    for cond in conditions:
        name = "gauss_vis_paired_clean.pkl" if cond == "clean" else f"gauss_vis_paired_{cond}.pkl"
        vis_by_cond[cond], _ = load_cache(cache_dir / name)

    scorer_vis = fit_scorer(cache_dir / "gauss_vis_train_clean.pkl")
    scorer_ir = fit_scorer(cache_dir / "gauss_ir_train_clean.pkl")

    c_vis = constants_for(vis_by_cond["clean"], scorer_vis, alpha, combination)
    c_ir = constants_for(ir_clean, scorer_ir, alpha, combination)
    # The D-6 ladder fit. Not optional: without it the D5/B5 fallback silently
    # changes gated glare from 0.2058 to 0.0641 (§6.2).
    fitted = json.loads((ROOT / constants).read_text(encoding="utf-8"))
    for key in ("vis", "ir"):
        if key not in fitted:
            continue
        f = fitted[key]
        obj = replace(c_vis if key == "vis" else c_ir, mu_d=float(f["mu_d"]), tau=float(f["tau"]))
        if key == "vis":
            c_vis = obj
        else:
            c_ir = obj

    bright_by_cond = {}
    if brightness_constants:
        bc = json.loads((ROOT / brightness_constants).read_text(encoding="utf-8"))["vis"]
        stat = bc["stat"]
        for cond in conditions:
            bp = ROOT / bright_dir / f"gauss_vis_paired_{cond}.json"
            if not bp.is_file():
                raise SystemExit(f"missing {bp} — run scripts/frame_brightness.py first")
            bright_by_cond[cond] = np.asarray(
                [f[stat] for f in json.loads(bp.read_text(encoding="utf-8"))["frames"]], dtype=float)
        c_vis = replace(c_vis, mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat,
                        bright_soft=bright_soft)

    h_frames = per_frame_homographies(ROOT / manifest, ROOT / homography)
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_by_cond["clean"]]
    runs = np.asarray([Path(r["image_path"]).parent.name for r in vis_by_cond["clean"]])

    ctx = FusionContext(
        vis_by_cond=vis_by_cond, ir_clean=ir_clean, scorer_vis=scorer_vis, scorer_ir=scorer_ir,
        c_vis=c_vis, c_ir=c_ir, bright_by_cond=bright_by_cond, h_frames=h_frames, gts=gts,
        runs=runs, cap_vis=1.0, cap_ir=1.0, conditions=tuple(conditions),
        iou_thr=iou_thr, veto=veto, veto_filter=veto_filter)

    if capability_sel:
        sel = None if capability_sel == "all" else ctx.sel(capability_sel)
        ctx.cap_vis, ctx.cap_ir = capability_prior(
            vis_by_cond["clean"], ir_clean, gts, h_frames, sel)
        ctx.cap_note = f"capability prior over {capability_sel} frames"

    if verbose:
        print(f"[ctx] {len(ir_clean)} paired frames | conditions {', '.join(conditions)}")
        print(f"[ctx] VIS mu_d={c_vis.mu_d:.2f} tau={c_vis.tau:.2f} mu_b={c_vis.mu_b} tau_b={c_vis.tau_b}")
        print(f"[ctx] capability prior: VIS {ctx.cap_vis:.4f}  IR {ctx.cap_ir:.4f} "
              f"(ratio {ctx.cap_vis / max(ctx.cap_ir, 1e-9):.1f}x)  [{capability_sel}]")
        print(f"[ctx] iou_thr={iou_thr} veto={veto} filter={veto_filter} "
              f"bright_soft={bright_soft}  day {len(ctx.sel('day'))} / night {len(ctx.sel('night'))}")
    return ctx


def run_systems(ctx: FusionContext, condition: str, **overrides) -> dict:
    """`evaluate_systems` under the adopted configuration, for one condition.

    When the context carries a `veto_filter`, the hysteresis is applied here:
    the instantaneous flags are computed from brightness alone (identical to
    the fitted path — `raw_veto_flags`), dilated in capture order, and handed
    to `evaluate_systems` as `veto_override`. Passing your own `veto_override`
    or `veto_below` in `overrides` bypasses the filter entirely, so ablation
    scripts that study the raw switch keep meaning what they say.
    """
    kw = dict(
        h_ir_to_vis=ctx.h_frames, scorer_ir=ctx.scorer_ir, constants_ir=ctx.c_ir,
        capability_vis=ctx.cap_vis, capability_ir=ctx.cap_ir, gts=ctx.gts,
        iou_thr_wbf=ctx.iou_thr, brightness_vis=ctx.bright_by_cond.get(condition),
        brightness_ir=None, veto_below=ctx.veto,
    )
    kw.update(overrides)
    vis = kw.pop("vis_records", ctx.vis_by_cond[condition])
    ir = kw.pop("ir_records", ctx.ir_clean)
    if (ctx.veto_filter is not None and ctx.veto is not None
            and "veto_override" not in overrides and "veto_below" not in overrides
            and kw.get("veto_on", "r_bright") == "r_bright"
            and ctx.c_vis.mu_b is not None):
        mode, k = ctx.veto_filter
        vv = raw_veto_flags(kw["brightness_vis"], ctx.c_vis.mu_b, ctx.c_vis.tau_b or 1e-9,
                            ctx.veto, len(vis))
        vv = filter_veto(vv, ctx.order, k, mode)
        kw["veto_override"] = (vv, [False] * len(vis))   # IR is never vetoed by design
        kw["veto_below"] = None
    return evaluate_systems(vis, ir, ctx.scorer_vis, ctx.c_vis, **kw)


def run_table(ctx: FusionContext, **overrides) -> dict[str, dict]:
    """The full condition sweep. Returns {condition: evaluate_systems result}."""
    return {cond: run_systems(ctx, cond, **overrides) for cond in ctx.conditions}
