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

#: A RUN-DISJOINT SPLIT OF THE DAY FRAMES, which the project has never had.
#:
#: `FIT_RUNS` excludes only pohang01, and pohang01 is entirely night -- so the
#: "day" report set and the "fit-run day" selection set are the SAME 1200 frames.
#: Every constant tuned on fit-run day frames has therefore been reported on the
#: frames it was tuned on, and the only genuinely held-out data in the whole
#: benchmark is a run where VIS scores exactly 0.0000 and nothing about the fusion
#: of two streams can be tested at all.
#:
#: Splitting by RUN and not by frame is the point: consecutive frames of one
#: canal transit are near-duplicates, so a random frame split would leak. pohang00
#: (836 day frames) tunes; pohang02 + pohang03 (247 + 117 = 364) are held out. The
#: split is uneven because the runs are, and the smaller side is the held-out one
#: deliberately -- a constant that only survives on the run it was fitted to should
#: fail here.
TUNE_RUNS = ("pohang00",)
TEST_RUNS = ("pohang02", "pohang03")


def fit_scorer(cache_path: Path) -> MahalanobisScorer:
    records, _ = load_cache(cache_path)
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in records]))


def constants_for(records, scorer, alpha, combination) -> ReliabilityConstants:
    parts = [per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
             for r in records if len(r["conf"])]
    if not parts:
        # A stream can be destroyed badly enough to emit nothing at all (IR under
        # noise s2 is exactly that, on all 2,232 frames). That is a legitimate
        # condition to evaluate, not a reason to abort: lam is a clean-data
        # calibration constant and a stream with no boxes has no r_box to scale.
        raise ValueError("no detections to fit reliability constants on")
    u = np.concatenate(parts)
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
    veto_keep_cls: tuple[int, ...] = ()   # classes a veto may not delete (IR is nc=1)
    consensus_beta: float = 1.0           # 1.0 == stock WBF's agreement bonus
    consensus_distinct: bool = False      # count streams, not cluster members
    support_iou: float = 0.0              # loose cross-stream confirmation...
    support_gamma: float = 0.0            # ...worth (1 + gamma) on the score; 0 = off
    order_records: list | None = None    # the stream capture order is read from
    struct_const: dict = field(default_factory=dict)
    veil_requires_night: bool = False   # AND the veil axis with the night arm
    veto_rule: str = "photometric+veil"          # | "gini+ir_night"
    single_passthrough: bool = False
    cap_note: str = ""
    _sel: dict = field(default_factory=dict)
    _order: dict | None = None

    @property
    def order(self) -> dict[str, np.ndarray]:
        """Capture order per run, computed once (needed by the veto filter)."""
        if self._order is None:
            # Not `vis_by_cond["clean"]`: a caller may be evaluating a single
            # non-clean condition, and capture order is a property of the frame
            # list, which every condition shares.
            self._order = temporal_order(
                self.order_records
                if self.order_records is not None
                else next(iter(self.vis_by_cond.values())))
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
    ir_condition: str | None = None,
    ir_nms: float | None = None,
    cap_ir_scale: float | None = None,
    structure_dir="runs/derived/structure",
    structure_constants="runs/eval/structure_constants.json",
    ir_bright=None,
    veto_keep_cls=(),
    config=None,
    verbose: bool = True,
) -> FusionContext:
    """`preset="adopted"` (default) reproduces the 2026-08-20/09-01 system exactly.

    `preset="crossmodal26m"` is `crossmodal` plus the two repairs the full-scale
    yolo26m / yolo26m-p2feat detectors force, measured in
    `docs/levers-and-the-26m-swap-2026-09-01.md`: the veil axis becomes conditional
    on the night arm, and a cross-modal SUPPORT term replaces the cross-modal
    merging the geometry does not allow. Use it with `cache_dir="runs/cache_m"`.

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
    if preset not in ("adopted", "crossmodal", "crossmodal26m"):
        raise ValueError(f"unknown preset {preset!r}")
    # `crossmodal26m` is `crossmodal` with the two repairs the full-scale detectors
    # force. It is a separate preset and not a change to `crossmodal` because every
    # published number was measured under the latter and must stay reproducible.
    v2 = preset == "crossmodal26m"
    if v2:
        preset = "crossmodal"
    cfg = load_config(config)
    cache_dir = ROOT / cache_dir
    alpha = float(cfg["reliability"]["alpha"])
    combination = str(cfg["reliability"]["combination"])

    ir_clean, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    # `ir_condition` degrades the IR STREAM while leaving everything fitted on IR
    # fitted on the clean one. That split is the whole point: the capability prior,
    # the IR health model and the night threshold are all calibration, and
    # calibration does not get to see the damage it is supposed to detect. Only the
    # records the system consumes at inference change.
    ir_prior_records = ir_clean
    if ir_condition:
        ir_clean, _ = load_cache(cache_dir / f"gauss_ir_paired_{ir_condition}.pkl")
        if len(ir_clean) != len(ir_prior_records):
            raise SystemExit(f"IR condition {ir_condition!r} has {len(ir_clean)} frames, "
                             f"clean has {len(ir_prior_records)} — the paired caches must "
                             f"stay index-aligned")
    if ir_nms is None and preset == "crossmodal":
        ir_nms = 0.70
    if ir_nms:
        # DETECTOR-SIDE duplicate suppression on the IR stream, at an IoU chosen on
        # the clean fit runs (`probe_ir_dedup.py`): +0.0003 [+0.0003, +0.0005] there
        # and +0.0019 [+0.0015, +0.0020] on the held-out night run, with the sign
        # agreeing across the two -- the check that separates a real effect from a
        # threshold fitted to one run's scene texture. Greedy NMS keeps the
        # highest-scoring box of a cluster; the WBF-style variant that averages the
        # cluster was BETTER at night and WORSE on the selection set, so it is
        # rejected by the pre-stated protocol rather than by preference.
        #
        # Read the gain honestly: it lifts the `ir_only` BASELINE by exactly the same
        # amount, so it does not widen the gap the fusion is judged on. It is a
        # better IR stream, not better fusion -- which matters because, with
        # `single_passthrough`, five of the eight cells simply ARE this stream.
        from uqfusion.eval.irdedup import nms_records
        ir_clean = nms_records(ir_clean, float(ir_nms))
        ir_prior_records = (ir_clean if not ir_condition
                            else nms_records(ir_prior_records, float(ir_nms)))
    vis_by_cond = {}
    for cond in conditions:
        name = "gauss_vis_paired_clean.pkl" if cond == "clean" else f"gauss_vis_paired_{cond}.pkl"
        vis_by_cond[cond], _ = load_cache(cache_dir / name)

    # The CLEAN VIS stream, always loaded, whatever conditions were asked for. Same
    # rule as the IR side: `c_vis` and the capability prior are calibration, and
    # calibration is fitted on clean data by definition -- so a caller evaluating
    # only `blur_s3` must still get constants fitted on `clean`, not on blur.
    vis_prior_records = vis_by_cond.get("clean")
    if vis_prior_records is None:
        vis_prior_records, _ = load_cache(cache_dir / "gauss_vis_paired_clean.pkl")

    scorer_vis = fit_scorer(cache_dir / "gauss_vis_train_clean.pkl")
    scorer_ir = fit_scorer(cache_dir / "gauss_ir_train_clean.pkl")

    c_vis = constants_for(vis_prior_records, scorer_vis, alpha, combination)
    # Fitted on the CLEAN IR stream, always. These are calibration constants, and
    # calibration does not get to see the damage it exists to measure -- the same
    # rule the capability prior and the health models follow. Before this was
    # explicit, an `ir_condition` silently refitted lam on the corrupted stream,
    # and IR under noise s2 (zero detections on all 2,232 frames) turned that into
    # a crash rather than a wrong number, which is the lucky version.
    c_ir = constants_for(ir_prior_records, scorer_ir, alpha, combination)
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
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_prior_records]
    runs = np.asarray([Path(r["image_path"]).parent.name for r in vis_prior_records])

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
        # The gate reads the statistics of the IR frames it is actually being given.
        ir_tag = ir_condition or "clean"
        irb = ROOT / (ir_bright or f"runs/derived/brightness/gauss_ir_paired_{ir_tag}.json")
        if not irb.is_file():
            raise SystemExit(f"missing {irb} — run scripts/frame_brightness.py "
                             f"--cache runs/cache/gauss_ir_paired_{ir_tag}.pkl --modality ir")
        ir_p05 = np.asarray([f["p05"] for f in json.loads(
            irb.read_text(encoding="utf-8"))["frames"]], dtype=float)
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
        isp = ROOT / structure_dir / f"gauss_ir_paired_{ir_tag}.json"
        if (hm is not None or band is not None) and not isp.is_file():
            raise SystemExit(f"missing {isp} — run scripts/frame_structure.py "
                             f"--cache runs/cache/gauss_ir_paired_{ir_tag}.pkl --modality ir")
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

    # Which classes may a veto NOT delete? The ones the surviving stream cannot
    # produce. Derived from the caches rather than hardcoded, because it is a fact
    # about the checkpoints and it changes: the phase2 IR model is nc=2 and can
    # supply buoys, so the exempt set there is EMPTY and every 26s number stays
    # bit-identical; the full-scale IR model is nc=1 per D28/A-1 and cannot, so
    # the exempt set is {buoy} and a VIS veto currently zeroes buoy AP outright.
    # "auto" therefore means the same rule on both, not the same behaviour.
    #
    # OFF by default, including under `crossmodal`: every published table was
    # measured with the veto deleting the whole stream, and a default that
    # silently changed them would make the two incomparable. Pass "auto" to
    # enable it; scripts/eval_class_selective_veto.py measures what it buys.
    if veto_keep_cls == "auto":
        vis_cls = set(int(c) for r in vis_prior_records for c in np.asarray(r["cls"]).ravel())
        ir_cls = set(int(c) for r in ir_prior_records for c in np.asarray(r["cls"]).ravel())
        veto_keep_cls = tuple(sorted(vis_cls - ir_cls))
    veto_keep_cls = tuple(int(c) for c in (veto_keep_cls or ()))

    if v2:
        # (1) The veil axis becomes conditional on the night arm. It fires on 100%
        # of fog frames and hands them to IR, which was right against yolo26s (VIS
        # 0.0020 on fogged day frames vs IR 0.0181) and is a -0.0632 regression
        # against yolo26m (VIS 0.0824, 4x the sensor the veto prefers). Deleting the
        # axis instead costs fog/night -0.0347, because fog lifts VIS p05 above mu_b
        # on 69% of night frames and the veil axis is what covers that. Conditional
        # keeps both: fog/day +0.0716 [+0.0657, +0.0784], fog/night +0.0000 exactly.
        veil_requires_night = True
        # (2) Cross-modal SUPPORT, since cross-modal MERGING is unavailable: at
        # iou_thr 0.85 only 0.05% of VIS boxes have an IR partner. A score bonus at
        # IoU 0.30 that never moves a coordinate is +0.0078 on the tune runs and
        # +0.0033 on the held-out ones, where 0.55 wins the tune runs and loses the
        # held-out ones in all six variants it appears in.
        support_iou, support_gamma = 0.30, 0.5
    else:
        veil_requires_night, support_iou, support_gamma = False, 0.0, 0.0

    ctx = FusionContext(
        vis_by_cond=vis_by_cond, ir_clean=ir_clean, scorer_vis=scorer_vis, scorer_ir=scorer_ir,
        c_vis=c_vis, c_ir=c_ir, bright_by_cond=bright_by_cond, struct_by_cond=struct_by_cond,
        h_frames=h_frames, gts=gts,
        runs=runs, cap_vis=1.0, cap_ir=1.0, conditions=tuple(conditions),
        iou_thr=iou_thr, veto=veto, veto_filter=veto_filter, veil_filter=veil_filter,
        tau_lap=tau_lap, gini_by_cond=gini_by_cond, ir_night=ir_night_flag,
        struct_const=sconst,
        order_records=vis_prior_records,
        veto_rule="gini+ir_night" if preset == "crossmodal" else "photometric+veil",
        single_passthrough=(preset == "crossmodal"),
        ir_d2=(ir_d2 if preset == "crossmodal" else None),
        ir_bound=(ir_bound if preset == "crossmodal" else 0.0),
        ir_bound_switch=(ir_bound_switch if preset == "crossmodal" else 0.0),
        q_vis_by_cond=q_vis_by_cond, veto_keep_cls=veto_keep_cls,
        veil_requires_night=veil_requires_night,
        support_iou=support_iou, support_gamma=support_gamma)

    if capability_sel:
        sel = None if capability_sel == "all" else ctx.sel(capability_sel)
        ctx.cap_vis, ctx.cap_ir = capability_prior(
            vis_prior_records, ir_prior_records, gts, h_frames, sel)
        # The fitted prior is each stream's clean mAP, which answers "how good is
        # this sensor?" -- not "how should its boxes rank against the other's". The
        # two differ, and the sweep shows it: IR's weight wants to be ~4x smaller
        # than its clean capability implies. At the saturating end IR contributes
        # essentially no detections yet fusion still beats VIS alone, so what the
        # weight is really buying on a day frame is WBF's consensus boost --
        # re-ranking VIS's boxes by agreeing with them.
        #
        # This was left at 1.0 as long as the benchmark could not PRICE it: the
        # ratio only matters where VIS is unvetoed, and on the original eight cells
        # every such cell had VIS ~36x ahead, so raising it was free. The extended
        # grid supplies the missing cells and does charge for it -- blur_s3/clean,
        # rain_s2/clean and blur_s3/glare_s2 all lose a little -- while the
        # corrupted-IR cells gain more, and the worst cell goes -0.0015 -> +0.0000.
        # 4.0 is the SMALLEST multiplier reaching that best worst-cell value; the
        # tie-break toward the fitted prior is deliberate, since larger values only
        # deepen the losses on the cells that now do the pricing.
        if cap_ir_scale is None:
            cap_ir_scale = 4.0 if preset == "crossmodal" else 1.0
        ctx.cap_ir = ctx.cap_ir / float(cap_ir_scale)
        ctx.cap_note = f"capability prior over {capability_sel} frames"

    if verbose:
        print(f"[ctx] {len(ir_clean)} paired frames | VIS conditions "
              f"{', '.join(conditions)} | IR stream {ir_condition or 'clean'}"
              + (f" (NMS {ir_nms})" if ir_nms else ""))
        print(f"[ctx] VIS mu_d={c_vis.mu_d:.2f} tau={c_vis.tau:.2f} mu_b={c_vis.mu_b} tau_b={c_vis.tau_b}")
        print(f"[ctx] capability prior: VIS {ctx.cap_vis:.4f}  IR {ctx.cap_ir:.4f} "
              f"(ratio {ctx.cap_vis / max(ctx.cap_ir, 1e-9):.1f}x, IR scaled "
              f"1/{cap_ir_scale:g})  [{capability_sel}]")
        print(f"[ctx] veto-exempt classes (IR cannot supply): "
              f"{list(ctx.veto_keep_cls) or 'none — IR covers every VIS class'}")
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
            gt = ctx.struct_const["axes"]["grad_gini"]["threshold"]
            it = ctx.struct_const["axes"]["ir_p05"]["threshold"]
            rule = (f"ir_p05 > {it:.1f} AND (VIS dark OR grad_gini < {gt:.4f})"
                    if ctx.veil_requires_night
                    else f"grad_gini < {gt:.4f} OR (ir_p05 > {it:.1f} AND VIS dark)")
            print(f"[ctx] preset CROSSMODAL{'26m' if ctx.veil_requires_night else ''}: "
                  f"veto = {rule}; no hysteresis; capability-only weights; "
                  f"single_passthrough=True"
                  + (f"; support IoU {ctx.support_iou:g} gamma {ctx.support_gamma:g}"
                     if ctx.support_gamma else ""))
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
    kw.setdefault("veto_keep_cls", ctx.veto_keep_cls)
    kw.setdefault("consensus_beta", ctx.consensus_beta)
    kw.setdefault("consensus_distinct", ctx.consensus_distinct)
    kw.setdefault("support_iou", ctx.support_iou)
    kw.setdefault("support_gamma", ctx.support_gamma)
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
        veil = np.zeros(n, dtype=bool)
        g = ctx.gini_by_cond.get(condition)
        if g is not None:
            veil = g < float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
        # `veil_requires_night` is the repair for a veto that was calibrated against
        # a detector this system no longer uses. The veil axis fires on 100% of fog
        # frames and hands them to IR, and under yolo26s that was right: VIS scored
        # 0.0020 on fogged day frames against IR's 0.0181. Under the full-scale
        # yolo26m it is a -0.0632 regression, because VIS now scores 0.0824 there --
        # 41x better under fog, and 4x better than the sensor the veto prefers.
        #
        # No image statistic can detect that, because what changed is not in the
        # image. The veil axis measures the fog correctly; the CLAIM attached to it,
        # "a fogged VIS cannot see", is what stopped being true.
        #
        # Deleting the axis is not the fix either: it takes fog/night from 0.0850 to
        # 0.0503, because fog lifts VIS p05 above mu_b on 69% of night frames and
        # the veil axis is what covers the night arm's resulting blind spot.
        #
        # So the axis is kept and made CONDITIONAL on the other sensor agreeing it is
        # dark: veto when IR says night AND (VIS looks dark OR the frame is veiled).
        # Fog at night is still caught by both paths; fog in daylight is caught by
        # neither, which is now the correct answer.
        if not ctx.veil_requires_night:
            vv |= veil
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
            dark = np.ones(n, dtype=bool)
            if b is not None and ctx.c_vis.mu_b is not None:
                dark = b < float(ctx.c_vis.mu_b)
            vv |= night & (dark | veil if ctx.veil_requires_night else dark)

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
