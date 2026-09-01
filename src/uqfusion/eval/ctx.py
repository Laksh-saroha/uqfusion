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
from uqfusion.eval.hysteresis import (ADOPTED_VEIL_FILTER, ADOPTED_VETO_FILTER, filter_veto,
                                      raw_veto_flags, temporal_order)
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
    struct_by_cond: dict[str, np.ndarray]   # lap_var per frame; {} disables the veil term
    h_frames: list
    gts: list[dict]
    runs: np.ndarray
    cap_vis: float
    cap_ir: float
    conditions: tuple[str, ...]
    iou_thr: float
    veto: float | None
    veto_filter: tuple[str, int] | None = None   # (mode, k) hysteresis on the switch
    tau_lap: float | None = None                # veil veto threshold; None disables it
    veil_filter: tuple[str, int] | None = ADOPTED_VEIL_FILTER   # denoise, not dilate
    # --- the 2026-09-01 cross-modal gate (preset "crossmodal"); see load_context.
    gini_by_cond: dict[str, np.ndarray] = field(default_factory=dict)
    ir_night: np.ndarray | None = None
    ir_d2: np.ndarray | None = None       # IR novelty score per frame
    ir_bound: float = 0.0                 # merge bound: above this, IR leaves the fusion
    ir_bound_switch: float = 0.0          # authority bound (tighter): above this, IR may not veto
    q_vis_by_cond: dict[str, np.ndarray] = field(default_factory=dict)   # VIS absolute health
    struct_const: dict = field(default_factory=dict)
    veto_rule: str = "photometric+veil"          # | "gini+ir_night"
    single_passthrough: bool = False
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
    veil_filter: tuple[str, int] | None = ADOPTED_VEIL_FILTER,
    tau_lap: float | None = None,
    preset: str = "adopted",
    structure_dir="runs/derived/structure",
    structure_constants="runs/eval/structure_constants.json",
    ir_bright="runs/derived/brightness/gauss_ir_paired_clean.json",
    config=None,
    verbose: bool = True,
) -> FusionContext:
    """`preset="adopted"` (default) reproduces the 2026-08-20/09-01 system exactly.

    `preset="crossmodal"` is the 2026-09-01 replacement, measured in
    `docs/crossmodal-gate-2026-09-01.md`. Three changes, each independently
    measured:

      weights   capability prior alone (`mu_d -> inf`, `lam -> 0`). The Mahalanobis
                soft term is not inert, it is the MECHANISM of the lowlight/day
                loss: lowlight drives VIS's D to ~248 against a mu_d of 71, so
                r_frame_vis collapses to ~0.009 while clean IR keeps ~0.82, and
                mean w_vis lands at 0.320 on a cell where VIS scores 0.0346 and IR
                0.0177. The gate hands the frame to the weaker sensor.

      veto      `grad_gini < gini_thr` OR `ir_p05 > ir_night_thr`, no hysteresis on
                either. The photometric axis is dropped, because it asks VIS "are
                you dark?" and cannot tell a dark world from a dark sensor:
                lowlight/day has p05 = 0, DARKER than the real night run, on frames
                where the detector still works. IR settles it -- on lowlight/day
                the IR frame is ordinary daylight, since the corruption hit VIS
                alone.

      fusion    `single_passthrough`, so a frame with one surviving stream returns
                that stream instead of passing it through single-list WBF, which
                clips, merges at iou_thr and re-scores (worth -0.0011 on a vetoed
                day cell and +0.0003 on a vetoed night one).

    Measured against the adopted system, ship AP, n_boot 1000: lowlight/day
    +0.0215 [+0.0184, +0.0246], glare/day +0.0032, fog/day +0.0011, clean/day
    +0.0003, and -0.0003 (CI spans zero) on each of the four night cells. Every one
    of the eight cells lands at or above max(VIS, IR).
    """
    if preset not in ("adopted", "crossmodal"):
        raise ValueError(f"unknown preset {preset!r}")
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

    bright_by_cond, struct_by_cond = {}, {}
    if brightness_constants:
        bc = json.loads((ROOT / brightness_constants).read_text(encoding="utf-8"))["vis"]
        stat = bc["stat"]
        # Resolved BEFORE the loop, not after: the loop below only reads `lap_var`
        # when a threshold exists, so loading tau_lap afterwards left struct_by_cond
        # empty and the veil term silently inert -- an eval that ran to completion and
        # reproduced the photometric-only numbers exactly, which is the worst way for
        # a bug to present. The `[ctx]` line now prints the veil state for that reason.
        if tau_lap is None:
            tau_lap = bc.get("tau_lap")
        for cond in conditions:
            bp = ROOT / bright_dir / f"gauss_vis_paired_{cond}.json"
            if not bp.is_file():
                raise SystemExit(f"missing {bp} — run scripts/frame_brightness.py first")
            frames = json.loads(bp.read_text(encoding="utf-8"))["frames"]
            bright_by_cond[cond] = np.asarray([f[stat] for f in frames], dtype=float)
            # The veil term is opt-in on the data: brightness files written before
            # 2026-09-01 carry no `lap_var`, and a missing column must reproduce the
            # photometric-only gate exactly rather than silently veto nothing at a
            # threshold it cannot evaluate.
            if tau_lap is not None and all("lap_var" in f for f in frames):
                struct_by_cond[cond] = np.asarray([f["lap_var"] for f in frames], dtype=float)
        c_vis = replace(c_vis, mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat,
                        bright_soft=bright_soft)

    h_frames = per_frame_homographies(ROOT / manifest, ROOT / homography)
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_by_cond["clean"]]
    runs = np.asarray([Path(r["image_path"]).parent.name for r in vis_by_cond["clean"]])

    gini_by_cond, ir_night_flag, sconst = {}, None, {}
    ir_d2, ir_bound, ir_bound_switch, q_vis_by_cond = None, 0.0, 0.0, {}
    if preset == "crossmodal":
        sp = ROOT / structure_constants
        if not sp.is_file():
            raise SystemExit(f"missing {sp} — run scripts/fit_structure_gate.py first")
        sconst = json.loads(sp.read_text(encoding="utf-8"))
        gini_thr = float(sconst["axes"]["grad_gini"]["threshold"])
        for cond in conditions:
            fp = ROOT / structure_dir / f"gauss_vis_paired_{cond}.json"
            if not fp.is_file():
                raise SystemExit(f"missing {fp} — run scripts/frame_structure.py first")
            g = np.asarray([f["grad_gini"] for f in
                            json.loads(fp.read_text(encoding="utf-8"))["frames"]], dtype=float)
            gini_by_cond[cond] = g
        ir_thr = float(sconst["axes"]["ir_p05"]["threshold"])
        ir_p05 = np.asarray([f["p05"] for f in json.loads(
            (ROOT / ir_bright).read_text(encoding="utf-8"))["frames"]], dtype=float)
        # IR SELF-CHECK. `runs/eval/ir_night_robustness.md`: applied to a FOGGED IR
        # sensor the bare `ir_p05` test misreads 75-96% of day frames as night, and a
        # false night vetoes a VIS stream scoring 0.3683 in favour of one scoring
        # 0.0177. IR may hold the night switch only while IR itself looks like IR.
        # `lap_over_var` is the statistic that can say so: it is stable across clean
        # day and clean night (medians 0.394 / 0.388, so it does not fire on the very
        # frames the switch is for) and leaves its clean band on 100% of fogged
        # frames. With it, IR-fog false nights go 96% -> 0%.
        hm = sconst["axes"].get("ir_health")
        band = sconst["axes"].get("ir_lap_over_var", {}).get("band")
        isp = ROOT / structure_dir / "gauss_ir_paired_clean.json"
        if (hm is not None or band is not None) and not isp.is_file():
            raise SystemExit(f"missing {isp} — run scripts/frame_structure.py "
                             f"--cache runs/cache/gauss_ir_paired_clean.pkl --modality ir")
        if hm is not None:
            # The multivariate check supersedes the single-axis band: a corruption
            # moves the JOINT distribution of frame statistics even when it moves no
            # one of them past its own extreme. Measured on the corruption probe,
            # per-axis `lap_over_var` catches IR glare on 12-34% of frames and this
            # catches 64-75%, at the same 0.0% on clean.
            ifr = json.loads(isp.read_text(encoding="utf-8"))["frames"]
            X = np.stack([[f[k] for k in hm["keys"]] for f in ifr]).astype(float)
            for j, k in enumerate(hm["keys"]):
                if k in hm["log1p_keys"]:
                    X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
            Z = (X - np.asarray(hm["mean"])) / np.asarray(hm["std"])
            ir_d2 = np.einsum("ij,jk,ik->i", Z, np.asarray(hm["precision"]), Z)
            ir_bound = float(hm["bound"])
            # The AUTHORITY bound is tighter than the MERGE bound, because the two
            # decisions fail in opposite directions. Over-restricting authority just
            # disables the night veto (worth at most the -0.0022..-0.0054 `no_veto`
            # costs a night cell); under-restricting lets a glare-corrupted IR veto a
            # working VIS, worth -0.35. Dropping IR from the MERGE has the reverse
            # shape -- it costs real AP on a clean day frame -- so that one keeps the
            # strict maximum. Measured: p99 flags 0.0% of clean NIGHT frames, so it
            # never disarms the switch on the frames the switch exists for.
            ir_bound_switch = float(hm.get("bound_switch", hm["bound"]))
        elif band is not None:
            iv = np.asarray([f["lap_over_var"] for f in
                             json.loads(isp.read_text(encoding="utf-8"))["frames"]], dtype=float)
            ir_d2 = np.maximum(float(band[0]) - iv, iv - float(band[1]))
            ir_bound = ir_bound_switch = 0.0
        else:
            ir_d2, ir_bound, ir_bound_switch = np.zeros(len(ir_p05)), 1.0, 1.0
        ir_ok = ir_d2 <= ir_bound_switch
        ir_night_flag = (ir_p05 > ir_thr) & ir_ok
        # VIS absolute health, per frame, for R_sys and the abstain. Same instrument
        # as the IR side: a Mahalanobis novelty score, because a RATIO of a raw
        # statistic to its own threshold has no dynamic range -- `grad_gini / thr`
        # bottoms out near 0.80 on a fully fogged frame, so the first version of this
        # reported VIS healthy on fog and the abstain released zero vetoes across 76
        # corruption pairs. Fitted on clean FIT-RUN frames only; night is not pooled
        # in, because on the VIS side night IS a failure and should score novel.
        vh = sconst["axes"].get("vis_health")
        if vh is not None:
            vmu, vsd = np.asarray(vh["mean"]), np.asarray(vh["std"])
            vprec, vbound = np.asarray(vh["precision"]), float(vh["bound"])
            for cond in conditions:
                fr = json.loads((ROOT / structure_dir /
                                 f"gauss_vis_paired_{cond}.json").read_text(
                                     encoding="utf-8"))["frames"]
                Xv = np.stack([[f[k] for k in vh["keys"]] for f in fr]).astype(float)
                for j, k in enumerate(vh["keys"]):
                    if k in vh["log1p_keys"]:
                        Xv[:, j] = np.log1p(np.clip(Xv[:, j], 0, None))
                Zv = (Xv - vmu) / vsd
                q_vis_by_cond[cond] = np.clip(
                    vbound / np.maximum(np.einsum("ij,jk,ik->i", Zv, vprec, Zv), 1e-12), 0, 1)

        # The capability prior alone: R == 1 for both streams, so the per-frame
        # weight is constant and the whole soft gate is switched off. That is the
        # measured configuration, not a simplification of it.
        c_vis = replace(c_vis, mu_d=1e9, lam=0.0)
        c_ir = replace(c_ir, mu_d=1e9, lam=0.0)

    ctx = FusionContext(
        vis_by_cond=vis_by_cond, ir_clean=ir_clean, scorer_vis=scorer_vis, scorer_ir=scorer_ir,
        c_vis=c_vis, c_ir=c_ir, bright_by_cond=bright_by_cond, struct_by_cond=struct_by_cond,
        h_frames=h_frames, gts=gts,
        runs=runs, cap_vis=1.0, cap_ir=1.0, conditions=tuple(conditions),
        iou_thr=iou_thr, veto=veto, veto_filter=veto_filter, veil_filter=veil_filter,
        tau_lap=tau_lap, gini_by_cond=gini_by_cond, ir_night=ir_night_flag,
        struct_const=sconst,
        veto_rule="gini+ir_night" if preset == "crossmodal" else "photometric+veil",
        single_passthrough=(preset == "crossmodal"),
        ir_d2=(ir_d2 if preset == "crossmodal" else None),
        ir_bound=(ir_bound if preset == "crossmodal" else 0.0),
        ir_bound_switch=(ir_bound_switch if preset == "crossmodal" else 0.0),
        q_vis_by_cond=q_vis_by_cond)

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
        if ctx.tau_lap is not None and ctx.struct_by_cond:
            print(f"[ctx] veil term ACTIVE: lap_var < {ctx.tau_lap:.1f}, filter={veil_filter}, "
                  f"conditions {sorted(ctx.struct_by_cond)}")
        else:
            why = ("no tau_lap in brightness_constants.json" if ctx.tau_lap is None
                   else "no lap_var column in the brightness files")
            print(f"[ctx] veil term OFF ({why}) - photometric-only gate")
        # Same reasoning as the veil line above: a preset that silently failed to
        # load its data would reproduce the other preset's numbers exactly, which
        # is the hardest kind of bug to notice (§5 of the veil record).
        if ctx.veto_rule == "gini+ir_night":
            print(f"[ctx] preset CROSSMODAL: veto = grad_gini < "
                  f"{ctx.struct_const['axes']['grad_gini']['threshold']:.4f} OR ir_p05 > "
                  f"{ctx.struct_const['axes']['ir_p05']['threshold']:.1f}; no hysteresis; "
                  f"capability-only weights; single_passthrough=True")
            print(f"[ctx] preset CROSSMODAL: IR-night fires on "
                  f"{ctx.ir_night.mean():.1%} of frames, gini-veil conditions "
                  f"{sorted(ctx.gini_by_cond)}")
        else:
            print("[ctx] preset ADOPTED: photometric OR veil veto, full soft weights")
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
    kw.setdefault("single_passthrough", ctx.single_passthrough)
    kw.update(overrides)
    vis = kw.pop("vis_records", ctx.vis_by_cond[condition])
    ir = kw.pop("ir_records", ctx.ir_clean)
    if (ctx.veto_rule == "gini+ir_night" and ctx.veto is not None
            and "veto_override" not in overrides and "veto_below" not in overrides):
        # Two axes, neither filtered, and neither is about VIS brightness.
        #
        # `grad_gini` fires on 100% of both fog cells and 0.0% of the other six --
        # cleanly enough that the majority-15 the `lap_var` veil term needed has
        # nothing left to denoise. `ir_p05` fires on 100% of all four night cells
        # and 0.0% of all four day cells, which is what removes the need for the
        # dilate-15 the photometric axis needed: the flickering that hysteresis
        # existed to repair was a property of reading darkness off the DEGRADED
        # sensor, and this reads it off the intact one.
        n = len(vis)
        vv = np.zeros(n, dtype=bool)
        g = ctx.gini_by_cond.get(condition)
        if g is not None:
            vv |= g < float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
        if ctx.ir_night is not None:
            night = ctx.ir_night
            # TWO-OF-TWO. The night arm requires BOTH sensors to agree that it is
            # dark, and that asymmetry is the safety property: the expensive error
            # is vetoing a WORKING VIS, and requiring VIS to look dark too makes
            # that impossible unless VIS really is dark. Measured across all 19 IR
            # corruption arms, the conjunction drops false-veto of a healthy day
            # VIS to 0.0% -- including IR glare, which the self-check alone leaves
            # at 27%.
            #
            # It does NOT reintroduce §7.2. VIS brightness is no longer being asked
            # to distinguish a dark world from a dark sensor; it only has to confirm
            # a call IR has already made, and on lowlight/day IR correctly says day.
            #
            # It costs the fog/night veto (fog lifts VIS p05 above mu_b on 69% of
            # night frames -- the effect dilate-15 existed to repair), which is why
            # the veil axis is OR-ed and not AND-ed: gini covers both fog cells at
            # 100% on its own.
            b = ctx.bright_by_cond.get(condition)
            if b is not None and ctx.c_vis.mu_b is not None:
                night = night & (b < float(ctx.c_vis.mu_b))
            vv |= night

        # ---- R_sys, and the abstain it exists for --------------------------
        # scope §7.4 introduced R_sys as "has every modality failed?", and under the
        # capability-only weights it degenerates to a constant 1 -- R is identically
        # 1 for both streams, so `max(R_vis, R_ir)` carries no information at all.
        # It is rebuilt here from the axes the gate actually uses, as a per-frame
        # ABSOLUTE health for each sensor:
        #
        #   q_vis = min(veil ratio, photometric sigmoid)   -- min, not product: either
        #           alarm firing is sufficient grounds to distrust the stream, which is
        #           the same convention `compute_reliability` uses for r_bright.
        #   q_ir  = bound / novelty, capped at 1           -- a ratio to the novelty
        #           bound, the same softening the veil axis takes from its own bound.
        #
        # Both are ratios or sigmoids of constants already fitted; no new constant is
        # introduced. R_sys = max(q_vis, q_ir): "is at least one sensor healthy?"
        q_vis = ctx.q_vis_by_cond.get(condition)
        if q_vis is None:
            q_vis = np.ones(n)
        q_ir = np.ones(n)
        if ctx.ir_d2 is not None and ctx.ir_bound > 0:
            q_ir = np.clip(ctx.ir_bound / np.maximum(ctx.ir_d2, 1e-12), 0, 1)
        r_sys = np.maximum(q_vis, q_ir)
        healthy_vis, healthy_ir = q_vis >= 0.5, q_ir >= 0.5   # the pre-registered midpoint

        # A broken IR should not merely be barred from holding the switch (which the
        # `ir_ok` term in `ir_night` already does) -- it should leave the merge, for
        # the same reason a broken VIS does. Symmetric, and gated on VIS being
        # healthy so the frame is never left with nothing.
        vi = (~healthy_ir) & healthy_vis

        # ABSTAIN is REPORTED, not acted on, and that is a measured decision rather
        # than a cautious one. The obvious design -- when neither sensor can vouch
        # for itself, release the switch and keep both -- was implemented and
        # measured across 76 (IR corruption x VIS condition) pairs. It prevented
        # **0** bad vetoes and lost **2,095** correct ones: the frames where both
        # streams are flagged are overwhelmingly VIS-fogged AND IR-broken, and
        # releasing the veil veto there merely adds fog-VIS junk on top of the
        # broken-IR junk. Nothing is gained by declining to choose when both options
        # are bad; the veto was already picking the less bad one.
        #
        # The residual the release was meant to fix is not reachable this way either.
        # It is a MISSED DETECTION -- IR glare that slips under the authority bound,
        # so `healthy_ir` is True and the abstain cannot fire on those frames at all.
        #
        # So abstain keeps the role scope §7.4 actually gave it: a per-frame flag
        # saying "no modality is reliable here", for a downstream consumer or a
        # risk-coverage curve. It does not touch the merge.
        abstain = (~healthy_vis) & (~healthy_ir)
        kw["veto_override"] = (vv.tolist(), vi.tolist())
        kw["veto_below"] = None
        res = evaluate_systems(vis, ir, ctx.scorer_vis, ctx.c_vis, **kw)
        # Reported alongside the systems, because a gate that can abstain has to say
        # how often it did -- an abstain rate is a result, not a diagnostic.
        res["R_sys_gate"] = r_sys.tolist()
        res["q_vis"], res["q_ir"] = q_vis.tolist(), q_ir.tolist()
        res["abstain"] = abstain.tolist()
        return res
    elif (ctx.veto_filter is not None and ctx.veto is not None
            and "veto_override" not in overrides and "veto_below" not in overrides
            and kw.get("veto_on", "r_bright") == "r_bright"
            and ctx.c_vis.mu_b is not None):
        # The two veto axes are filtered SEPARATELY and OR-ed after, never OR-ed and
        # then filtered together -- see hysteresis.ADOPTED_VEIL_FILTER for the
        # measurement that forces this. Dilating the combined switch costs the
        # glare/day guard cell 4.8% of its frames.
        mode, k = ctx.veto_filter
        n = len(vis)
        vv = np.asarray(filter_veto(
            raw_veto_flags(kw["brightness_vis"], ctx.c_vis.mu_b, ctx.c_vis.tau_b or 1e-9,
                           ctx.veto, n), ctx.order, k, mode), dtype=bool)
        struct = ctx.struct_by_cond.get(condition)
        if struct is not None and ctx.tau_lap is not None:
            veil = raw_veto_flags(None, 0.0, 1.0, ctx.veto, n,
                                  structure=struct, tau_lap=ctx.tau_lap)
            if ctx.veil_filter is not None:
                vmode, vk = ctx.veil_filter
                veil = filter_veto(veil, ctx.order, vk, vmode)
            vv = vv | np.asarray(veil, dtype=bool)
        vv = vv.tolist()
        kw["veto_override"] = (vv, [False] * len(vis))   # IR is never vetoed by design
        kw["veto_below"] = None
    return evaluate_systems(vis, ir, ctx.scorer_vis, ctx.c_vis, **kw)


def run_table(ctx: FusionContext, **overrides) -> dict[str, dict]:
    """The full condition sweep. Returns {condition: evaluate_systems result}."""
    return {cond: run_systems(ctx, cond, **overrides) for cond in ctx.conditions}
