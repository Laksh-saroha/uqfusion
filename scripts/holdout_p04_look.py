"""Phase 3 §7.2 — THE SINGLE LOOK at pohang04. Run once, after the freeze commit, never again.

Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendments 8 and 9 as written. Every
rule below is transcribed from there; if this file and the amendments disagree, the
amendments win and the look does not run until the file is fixed **before** the freeze.

Refusals, all checked before a single pohang04 AP is computed
-------------------------------------------------------------
1. **Freeze.** HEAD's subject starts with `FREEZE`, no tracked file is modified, and this
   script, the development reference and the step-1 manifest are committed unchanged.
2. **One look.** `runs/holdout_p04/LOOK_TAKEN.json` must not exist; it is created
   (exclusively) immediately before scoring. A crash after that point does not license a
   re-run — that decision has to be written down first (§7.2).
3. **Ground truth (A8.3).** pohang04 VIS label hash must be `c06611a684f4`.
4. **Inputs.** Step-1 manifest equals its committed copy, the pair table hash matches, the
   composition has zero night frames (A9.4; a night cell is not implemented and the look
   refuses rather than silently pooling), all 190 caches pass the builder's meta check, all
   76 frame-statistic files carry the right corruption / severity / seed / frame count and
   were measured under the `.venv` arithmetic (numpy 2.4.6, opencv 5.0.0).
6. **Frozen bytes.** Every file listed in the committed `holdout_p04_freeze_manifest.json`
   (checkpoints, calibration, caches, statistics, substrates) hashes to its frozen value.
5. **Reference reproduces.** Each system's development `clean/clean` AP on both groups is
   recomputed here and must equal `docs/eval/holdout_p04_devref_2026-09-14.json` exactly —
   which also proves this interpreter does the same arithmetic the reference was built with.

How a pohang04 system is assembled (A9.1)
-----------------------------------------
`load_context(preset="crossmodal26m", role="final", capability_sel=None)` over the pohang04
caches and statistics, then the **development** context's calibration is injected: the
capability prior (`cap_vis`, `cap_ir`, `cap_ir_scale`, fitted on development `fit` frames)
and `c_vis`/`c_ir`. The two contexts' `c_vis`/`c_ir` must already be equal — the only
data-fitted fields are overwritten by the preset — and the script asserts it, so no
calibration value is ever fitted on pohang04.

What is computed
----------------
* Cells: `gate_snms_draw_avg.CELLS`, all 11. `clean/clean` once per system; the other ten
  per system x draw (VIS 941-944, IR 951-954).
* Statistic: ship AP (class 0), mean over the five systems; for corrupted cells, mean over
  draws of that. Per-seed and per-draw values reported.
* Interval: moving-block bootstrap L = 20, n_boot = 1000, **seed 1**, one resample scoring
  every system (and every draw) on the same frames.
* **Verdict, `clean/clean` only (A9.3):** `D = AP_ref − AP_p04`, replicates
  `ref_rep[t] − p04_rep[t]` against the 1,000 committed seed-0 reference replicates.
  HOLDOUT-GAP iff `D ≥ 0.0060` and the 2.5th percentile of the replicates is > 0; else
  NO-GAP. "Above development" (no outcome label) iff `AP_p04 − AP_high ≥ 0.0060` with its
  replicate CI above zero. Counts at 0.0014 / 0.0031 / 0.0060 / 0.0100 reported.
* The other ten cells: value and interval only. No verdict, no label (A9.2).

`--selftest` runs the same assembly and bootstrap code on DEVELOPMENT data only
(`runs/cache_m`, shipped corruption seeds) and requires it to reproduce Stage 1's
committed `clean/A/tune` value exactly. It never opens a pohang04 cache.

Usage:
    python scripts/holdout_p04_look.py --selftest
    python scripts/holdout_p04_look.py --out docs/eval/holdout_p04_look_<date>.md   # ONCE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from _ideas_common import write_md                                                # noqa: E402
from gate_snms_draw_avg import CELLS, CORRUPTED                                   # noqa: E402
from uqfusion.eval.apmetrics import _score, ap_from_parts, frame_parts, presort   # noqa: E402
from uqfusion.eval.blockboot import block_resample, run_ids, run_slices           # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems                           # noqa: E402
from uqfusion.eval.identity import system_identity                                # noqa: E402

PRESET = "crossmodal26m"
SEEDS = (0, 1, 2, 3, 4)
VIS_DRAWS = (941, 942, 943, 944)
BLOCK_LEN, N_BOOT, BOOT_SEED = 20, 1000, 1
SHIP = 0
FLOOR = 0.0060
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100)
LABEL_HASH = "c06611a684f4"
STATS_ENV = {"numpy": "2.4.6", "opencv": "5.0.0"}

HP = ROOT / "runs" / "holdout_p04"
DEVREF = ROOT / "docs/eval/holdout_p04_devref_2026-09-14.json"
STEP1_COPY = ROOT / "docs/eval/holdout_p04_step1_manifest_2026-09-14.json"
MARKER = HP / "LOOK_TAKEN.json"
# `runs/` is git-ignored and the caches in it are deterministically rebuildable, so MARKER
# alone does not survive a rebuild of the tree: restore `runs/`, check out the freeze commit,
# and all 316 hashes match again with no marker present. MIRROR is the tracked copy, and the
# refusal below reads both. It is written after the numbers, not before, because its purpose
# is durability across a rebuild, not crash-safety -- MARKER already provides that.
MIRROR = ROOT / "docs/eval/holdout_p04_LOOK_TAKEN.json"
VERDICT_CELL = ("clean", None)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def sha256_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


# ------------------------------------------------------------------ refusals (look only)
def check_freeze() -> str:
    head = git("log", "-1", "--format=%H %s").strip()
    if not head.split(" ", 1)[1].startswith("FREEZE"):
        raise SystemExit(f"REFUSED: HEAD is not a freeze commit ({head[:80]})")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("REFUSED: tracked files are modified; the look runs on the freeze tree only")
    for p in (Path(__file__).resolve(), DEVREF, STEP1_COPY):
        if not git("ls-files", rel(p)).strip():
            raise SystemExit(f"REFUSED: {rel(p)} is not committed")
    return head.split(" ", 1)[0]


FREEZE_MANIFEST = ROOT / "docs/eval/holdout_p04_freeze_manifest.json"


def check_freeze_manifest() -> None:
    """Every file the look reads, hashed at freeze time (holdout_p04_freeze.py), must be
    byte-identical now. `runs/` is git-ignored, so this -- not the commit -- is what freezes
    the checkpoints, calibration JSONs, caches and statistics."""
    if not git("ls-files", rel(FREEZE_MANIFEST)).strip():
        raise SystemExit("REFUSED: the freeze manifest is not committed")
    man = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if man.get("draft") or not man.get("ready"):
        raise SystemExit("REFUSED: the committed freeze manifest is a draft")
    import holdout_p04_freeze as fz
    bad = []
    for group, entries in man["files"].items():
        for key, want in entries.items():
            p = ROOT / key
            if want is None or not p.is_file() or fz.sha256(p) != want["sha256"]:
                bad.append(key)
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} frozen files changed or missing, e.g. {bad[0]}")


def check_labels() -> None:
    fs = sorted((ROOT / "Pohang_dataset/visible/labels/pohang04").glob("*.txt"))
    h = hashlib.sha256()
    for f in fs:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    if h.hexdigest()[:12] != LABEL_HASH:
        raise SystemExit(f"REFUSED: pohang04 label hash {h.hexdigest()[:12]} != {LABEL_HASH} (A8.3)")


def check_inputs() -> int:
    live = HP / "step1_manifest.json"
    if sha256_bytes(live) != sha256_bytes(STEP1_COPY):
        raise SystemExit("REFUSED: step-1 manifest differs from its committed copy")
    man = json.loads(live.read_text(encoding="utf-8"))
    if sha256_bytes(ROOT / "Pohang_dataset/paired/pohang04_pairs.csv") != man["pairs"]["sha256"]:
        raise SystemExit("REFUSED: pair table changed since step 1")
    if man["composition"]["n_night"] != 0:
        raise SystemExit("REFUSED: night frames exist; A9.4's night cell is not implemented here")
    import holdout_p04_build_caches as b
    lv, li, n = b.write_lists(write=False)
    bad = [str(j["out"]) for j in b.jobs(lv, li) if not j["out"].is_file() or b.meta_ok(j["out"], j, n)]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} of 190 caches missing or failing meta, e.g. {bad[0]}")
    import holdout_p04_frame_stats as fsm
    for s in fsm.streams():
        for d in ("brightness", "structure"):
            p = fsm.OUT / d / s["sub"] / f"{s['stem']}.json"
            if not p.is_file():
                raise SystemExit(f"REFUSED: missing {rel(p)}")
            hd = json.loads(p.read_text(encoding="utf-8"))
            want = (s["kind"], s["sev"], s["seed"], n, STATS_ENV["numpy"], STATS_ENV["opencv"])
            got = (hd["corrupt"], hd["severity"], hd["corrupt_seed"], hd["n_frames"], hd.get("numpy"), hd.get("opencv"))
            if got != want:
                raise SystemExit(f"REFUSED: {rel(p)} is {got}, expected {want}")
    return n


# ------------------------------------------------------------------ substrates
class Substrate:
    """Where one system's caches and statistics live, per draw. `sub` is 'clean' or a draw."""

    def __init__(self, name, seeds, subs, manifest, frames_sel):
        self.name, self.seeds, self.subs, self.manifest, self.frames_sel = name, seeds, subs, manifest, frames_sel

    def cache_dir(self, seed, sub): raise NotImplementedError
    def bright_dir(self, sub): raise NotImplementedError
    def struct_dir(self, sub): raise NotImplementedError


class Pohang04(Substrate):
    """Hard-linked staging, so each (system, draw) is one directory `load_context` can read."""

    def __init__(self):
        super().__init__("pohang04", SEEDS, ["clean"] + [f"draw{v}_{v + 10}" for v in VIS_DRAWS],
                         rel(HP / "p04_pairs_manifest.csv"), None)
        self.stage = HP / "stage"

    @staticmethod
    def _link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if os.path.samefile(src, dst):
                return
            dst.unlink()
        os.link(src, dst)

    def cache_dir(self, seed, sub):
        d = self.stage / f"seed{seed}" / sub
        for m in ("vis", "ir"):
            self._link(ROOT / f"runs/cache_p3/seed{seed}/gauss_{m}_train_clean.pkl", d / f"gauss_{m}_train_clean.pkl")
            self._link(HP / f"caches/seed{seed}/clean/gauss_{m}_paired_clean.pkl", d / f"gauss_{m}_paired_clean.pkl")
        if sub != "clean":
            for _m, stem, _k, _s in CORRUPTED:
                self._link(HP / f"caches/seed{seed}/{sub}/{stem}.pkl", d / f"{stem}.pkl")
        return rel(d)

    def _derived(self, kind, sub):
        d = self.stage / "derived" / sub / kind
        for m in ("vis", "ir"):
            self._link(HP / f"derived/{kind}/clean/gauss_{m}_paired_clean.json", d / f"gauss_{m}_paired_clean.json")
        if sub != "clean":
            for _m, stem, _k, _s in CORRUPTED:
                self._link(HP / f"derived/{kind}/{sub}/{stem}.json", d / f"{stem}.json")
        return rel(d)

    def bright_dir(self, sub): return self._derived("brightness", sub)
    def struct_dir(self, sub): return self._derived("structure", sub)


class DevSelftest(Substrate):
    """Development data only: the shipped caches and statistics, one 'system', one 'draw'."""

    def __init__(self):
        super().__init__("selftest (development)", (0,), ["clean", "shipped"],
                         "runs/derived/paired_val_manifest.csv", "tune")

    def cache_dir(self, seed, sub): return "runs/cache_m"
    def bright_dir(self, sub): return "runs/derived/brightness"
    def struct_dir(self, sub): return "runs/derived/structure"


# ------------------------------------------------------------------ assembly
def dev_context(seed: int, cache_dir: str):
    return load_context(preset=PRESET, cache_dir=cache_dir, conditions=("clean",), verbose=False)


def system_context(sub_: Substrate, seed: int, sub: str, vis_conds, ir_cond, dev):
    tag = ir_cond or "clean"
    ctx = load_context(preset=PRESET, cache_dir=sub_.cache_dir(seed, sub), manifest=sub_.manifest,
                       bright_dir=sub_.bright_dir(sub), structure_dir=sub_.struct_dir(sub),
                       ir_bright=f"{sub_.bright_dir(sub)}/gauss_ir_paired_{tag}.json",
                       conditions=tuple(vis_conds), ir_condition=ir_cond,
                       role="final", capability_sel=None, verbose=False)
    if ctx.c_vis != dev.c_vis or ctx.c_ir != dev.c_ir:
        raise SystemExit("calibration constants differ between the development and scored contexts -- "
                         "something was fitted on the scored frames")
    ctx.cap_vis, ctx.cap_ir, ctx.cap_ir_scale = dev.cap_vis, dev.cap_ir, dev.cap_ir_scale
    ctx.cap_note = "capability prior injected from the development context (fit frames)"
    ctx.inputs["capability_sel"] = "injected:development-fit"
    return ctx


def score_substrate(sub_: Substrate, dev_ctx_by_seed) -> dict:
    """{(vis_cond, ir_cond): {(seed, sub): frame_parts}} plus frame paths."""
    parts, paths = {}, None
    for seed in sub_.seeds:
        dev = dev_ctx_by_seed[seed]
        for sub in sub_.subs:
            cells = [VERDICT_CELL] if sub == "clean" else [c for c in CELLS if c != VERDICT_CELL]
            by_ir = {}
            for vc, ic in cells:
                by_ir.setdefault(ic, []).append(vc)
            for ic, vcs in by_ir.items():
                ctx = system_context(sub_, seed, sub, sorted(set(vcs) | {"clean"}), ic, dev)
                p = [r["image_path"] for r in ctx.vis_by_cond["clean"]]
                if paths is None:
                    paths = p
                elif p != paths:
                    raise SystemExit("frame order differs between systems")
                for vc in vcs:
                    out = run_systems(ctx, vc)
                    parts.setdefault((vc, ic), {})[(seed, sub)] = frame_parts(out["fused_gated"], out["gts"])
            print(f"[score] {sub_.name}: system {seed} {sub} done", flush=True)
    return {"parts": parts, "paths": paths}


def boot_cell(cell_parts: dict, paths, sel, n_boot: int) -> dict:
    """Seed-mean (and draw-mean) ship AP with a shared block resample. No printing."""
    idx = np.arange(len(paths)) if sel is None else np.asarray(sel)
    per_run = run_slices(run_ids([paths[i] for i in idx]))
    keys = sorted(cell_parts)
    pre = {k: presort(cell_parts[k], idx) for k in keys}
    n = pre[keys[0]]["n_frames"]
    subs = sorted({s for _, s in keys})

    def agg(vals):
        return float(np.mean([np.mean([vals[(sd, s2)] for sd, s2 in keys if s2 == s]) for s in subs]))

    def stat(w):
        return agg({k: _score(pre[k], w, SHIP) for k in keys})

    # Observed values through `ap_from_parts`, exactly as the development reference and
    # Stage 1 computed theirs; `_score` only for the replicates, exactly as the reference
    # computed its replicates. Mixing the two paths would compare numbers that can differ
    # in the last bits for reasons unrelated to pohang04.
    per = {}
    for k in keys:
        e = ap_from_parts(cell_parts[k], sel=idx)["per_class"].get(SHIP)
        per[k] = float(e["ap50_95"]) if e else float("nan")
    obs = agg(per)
    rng = np.random.default_rng(BOOT_SEED)
    reps = np.empty(n_boot)
    for t in range(n_boot):
        reps[t] = stat(np.bincount(block_resample(rng, per_run, BLOCK_LEN), minlength=n))
    kept = reps[np.isfinite(reps)]
    lo, hi = np.percentile(kept, [2.5, 97.5])
    return {"value": obs, "ci_lo": float(lo), "ci_hi": float(hi), "n_effective": int(kept.size),
            "per_system_draw": {f"seed{sd}/{s}": v for (sd, s), v in per.items()},
            "replicates": reps.tolist()}


def verdict(p04: dict, ref: dict) -> dict:
    ref_rep = np.asarray(ref["reference"]["replicates"], dtype=float)
    p_rep = np.asarray(p04["replicates"], dtype=float)
    d_obs = ref["reference"]["AP_ref"] - p04["value"]
    d_rep = ref_rep - p_rep
    d_rep = d_rep[np.isfinite(d_rep)]
    lo, hi = np.percentile(d_rep, [2.5, 97.5])
    high_g = max(ref["groups"], key=lambda g: ref["groups"][g]["seed_mean"])
    up_obs = p04["value"] - ref["groups"][high_g]["seed_mean"]
    up_rep = p_rep - np.asarray(ref["groups"][high_g]["replicates"], dtype=float)
    up_rep = up_rep[np.isfinite(up_rep)]
    up_lo = float(np.percentile(up_rep, 2.5))
    return {
        "AP_ref": ref["reference"]["AP_ref"], "ref_group": ref["reference"]["group"], "AP_p04": p04["value"],
        "D": d_obs, "D_ci": [float(lo), float(hi)],
        "outcome": "HOLDOUT-GAP" if (d_obs >= FLOOR and lo > 0) else "NO-GAP",
        "gap_by_floor": {str(f): bool(d_obs >= f and lo > 0) for f in FLOORS},
        "above_development": bool(up_obs >= FLOOR and up_lo > 0), "high_group": high_g,
        "U": up_obs, "U_ci_lo": up_lo,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    t0 = time.time()

    if args.selftest:
        sub_ = DevSelftest()
        dev = {0: dev_context(0, "runs/cache_m")}
        sc = score_substrate(sub_, dev)
        tune = dev[0].sel("tune")
        got = boot_cell(sc["parts"][VERDICT_CELL], sc["paths"], tune, 20)["value"]
        want = json.loads((ROOT / "runs/eval/stage1_crossing_2026-09-14.json").read_text(encoding="utf-8"))["ap"]["clean/A/tune"]
        corrupt = {f"{vc}/{ic or 'clean'}": boot_cell(sc["parts"][(vc, ic)], sc["paths"], tune, 20)["value"]
                   for vc, ic in CELLS if (vc, ic) != VERDICT_CELL}
        ok = got == want
        print(f"[selftest] clean/clean pohang00 via injection path {got!r} vs Stage 1 {want!r}: "
              f"{'EXACT' if ok else 'MISMATCH'}; {len(corrupt)} corrupted cells scored without error")
        return 0 if ok else 1

    if not args.out:
        raise SystemExit("--out is required for the look")
    head = check_freeze()
    for taken in (MARKER, MIRROR):
        if taken.exists():
            raise SystemExit(f"REFUSED: {rel(taken)} exists -- the single look has already been taken")
    check_freeze_manifest()
    check_labels()
    n = check_inputs()
    ref = json.loads(DEVREF.read_text(encoding="utf-8"))
    dev_ctx = {}
    for k in SEEDS:
        dev_ctx[k] = dev_context(k, f"runs/cache_p3/seed{k}")
        out = run_systems(dev_ctx[k], "clean")
        parts = frame_parts(out["fused_gated"], out["gts"])
        for g, runs in (("pohang00", ("pohang00",)), ("pohang02+pohang03", ("pohang02", "pohang03"))):
            e = ap_from_parts(parts, sel=np.flatnonzero(np.isin(dev_ctx[k].runs, runs)))["per_class"].get(SHIP)
            if float(e["ap50_95"]) != ref["groups"][g]["per_seed_ap"][k]:
                raise SystemExit(f"REFUSED: development reference does not reproduce (seed {k}, {g})")
    print(f"[look] all refusals cleared ({time.time() - t0:.0f}s); taking the single look", flush=True)

    fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"head": head, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "pid": os.getpid(), "n_pairs": n, "numbers_written": False}).encode())
    os.close(fd)

    sc = score_substrate(Pohang04(), dev_ctx)
    cells = {}
    for c in CELLS:
        cells[c] = boot_cell(sc["parts"][c], sc["paths"], None, args.boot)
        print(f"[look] bootstrap {c[0]}/{c[1] or 'clean'} done ({time.time() - t0:.0f}s)", flush=True)
    v = verdict(cells[VERDICT_CELL], ref)

    L = [f"The single look (§7.2), under Amendments 8 and 9, at freeze commit `{head}`. "
         f"{n} pohang04 day pairs; fused ship AP; five Phase 3 systems; block bootstrap L={BLOCK_LEN}, "
         f"n_boot={args.boot}, seed {BOOT_SEED}. **Not to be re-run.**",
         f"## Verdict (`clean/clean`, A9.3) — **{v['outcome']}**\n\n"
         f"| | value |\n|---|---|\n| AP_ref ({v['ref_group']}) | {v['AP_ref']:.4f} |\n"
         f"| AP_p04 (seed mean) | {v['AP_p04']:.4f} |\n| D = AP_ref − AP_p04 | {v['D']:+.4f} |\n"
         f"| D 95% CI (unpaired, replicate-wise) | [{v['D_ci'][0]:+.4f}, {v['D_ci'][1]:+.4f}] |\n\n"
         "Gap at each floor: " + ", ".join(f"{f}: {'yes' if g else 'no'}" for f, g in v["gap_by_floor"].items())
         + f" — verdict taken at {FLOOR}.\n\n"
         + (f"pohang04 is above the higher development group ({v['high_group']}) by {v['U']:+.4f}, CI lower "
            f"bound {v['U_ci_lo']:+.4f} — reported, no outcome label." if v["above_development"] else ""),
         "## All 11 cells — values and intervals (only `clean/clean` carries a verdict)\n\n"
         "| cell (VIS/IR) | seed-mean AP | 95% CI |\n|---|---:|---|\n"
         + "\n".join(f"| {vc}/{ic or 'clean'} | {cells[(vc, ic)]['value']:.4f} | "
                     f"[{cells[(vc, ic)]['ci_lo']:.4f}, {cells[(vc, ic)]['ci_hi']:.4f}] |" for vc, ic in CELLS),
         "## Per system (and draw)\n\n" + "\n".join(
             f"- **{vc}/{ic or 'clean'}**: " + ", ".join(f"{k} {x:.4f}" for k, x in sorted(cells[(vc, ic)]["per_system_draw"].items()))
             for vc, ic in CELLS)]
    ident = system_identity(dev_ctx[0], preset=PRESET, seeds=list(SEEDS), draws=list(VIS_DRAWS),
                            block_len=BLOCK_LEN, n_boot=args.boot, bootstrap_seed=BOOT_SEED, freeze_commit=head)
    write_md(Path(args.out), "pohang04 — the single look", L, identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"verdict": v, "cells": {f"{vc}/{ic or 'clean'}": {k: x for k, x in c.items() if k != "replicates"}
                                 for (vc, ic), c in cells.items()},
         "p04_clean_replicates": cells[VERDICT_CELL]["replicates"], "freeze_commit": head}, indent=1), encoding="utf-8")
    m = json.loads(MARKER.read_text(encoding="utf-8"))
    m.update(numbers_written=True, finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), out=args.out)
    MARKER.write_text(json.dumps(m, indent=2), encoding="utf-8")
    MIRROR.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"[look] {v['outcome']} -- written to {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
