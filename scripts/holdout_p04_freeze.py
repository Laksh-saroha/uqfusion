"""Phase 3 §7.2 — prepare the FREEZE commit for the pohang04 single look.

§7.2: *"pohang04 is scored exactly once, after ... every constant, threshold, checkpoint,
correspondence rule and preset is frozen and committed."* Most of what the look consumes
lives under the git-ignored `runs/` tree — checkpoints, calibration JSONs, 190 caches, 76
statistic files — so committing code alone freezes nothing. This writes a manifest that
**hashes every file the look reads**, commits alongside the code, and is re-verified by
`holdout_p04_look.py` before it scores a frame.

Readiness gate (refuses to write the real manifest unless all hold):
  * all 190 pohang04 caches pass the builder's meta check, and all 76 statistic files match
    their stream (the look script's own `check_inputs`);
  * the pohang04 label hash is `c06611a684f4` (A8.3);
  * the look selftest log ends in `EXACT` (the injection path reproduces Stage 1).

`--draft` writes the same manifest to a scratch path with missing entries marked, and never
refuses, so readiness can be inspected while the caches build.

Usage:
    python scripts/holdout_p04_freeze.py --draft --out <scratch>/freeze_draft.json
    python scripts/holdout_p04_freeze.py              # writes docs/eval/holdout_p04_freeze_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

MANIFEST = ROOT / "docs/eval/holdout_p04_freeze_manifest.json"
SEEDS = (0, 1, 2, 3, 4)
SELFTEST_LOG = ROOT / "runs/holdout_p04/look_selftest.log"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory() -> dict[str, list[Path]]:
    import holdout_p04_build_caches as b
    import holdout_p04_frame_stats as fs
    groups: dict[str, list[Path]] = {}
    groups["checkpoints"] = [ROOT / f"runs/phase3_stage2/p3_{m}_seed{k}/weights/best.pt"
                             for k in SEEDS for m in ("vis", "ir")]
    groups["mahalanobis_references"] = [ROOT / f"runs/cache_p3/seed{k}/gauss_{m}_train_clean.pkl"
                                        for k in SEEDS for m in ("vis", "ir")]
    groups["development_caches"] = [ROOT / f"runs/cache_p3/seed{k}/gauss_{m}_paired_clean.pkl"
                                    for k in SEEDS for m in ("vis", "ir")]
    lv, li = b.HP / "p04_pairs_vis.txt", b.HP / "p04_pairs_ir.txt"
    groups["pohang04_caches"] = [j["out"] for j in b.jobs(lv, li)]
    groups["pohang04_frame_stats"] = [fs.OUT / d / s["sub"] / f"{s['stem']}.json"
                                      for s in fs.streams() for d in ("brightness", "structure")]
    groups["calibration_and_geometry"] = [ROOT / p for p in (
        "runs/eval/reliability_constants.json", "runs/eval/brightness_constants.json",
        "runs/eval/structure_constants.json", "runs/derived/homography_ir_to_vis.json", "config.yaml")]
    groups["development_substrate"] = [ROOT / p for p in (
        "runs/derived/paired_val_vis.txt", "runs/derived/paired_val_ir.txt",
        "runs/derived/paired_val_manifest.csv",
        "runs/derived/brightness/gauss_vis_paired_clean.json", "runs/derived/brightness/gauss_ir_paired_clean.json",
        "runs/derived/structure/gauss_vis_paired_clean.json", "runs/derived/structure/gauss_ir_paired_clean.json",
        "runs/eval/stage1_crossing_2026-09-14.json")]
    groups["pohang04_substrate"] = [lv, li, b.HP / "p04_pairs_manifest.csv", b.HP / "step1_manifest.json",
                                    ROOT / "Pohang_dataset/paired/pohang04_pairs.csv"]
    groups["committed_reference"] = [ROOT / "docs/eval/holdout_p04_devref_2026-09-14.json",
                                     ROOT / "docs/eval/holdout_p04_step1_manifest_2026-09-14.json"]
    return groups


def environment() -> dict:
    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None
    venv = subprocess.run([str(ROOT / ".venv/Scripts/python.exe"), "-c",
                           "import numpy, cv2; print(numpy.__version__, cv2.__version__)"],
                          capture_output=True, text=True)
    return {"scoring_interpreter": sys.executable,
            "scoring": {p: ver(p) for p in ("numpy", "torch", "ultralytics", "opencv-python-headless", "opencv-python")},
            "frame_stats_venv": venv.stdout.strip() or venv.stderr.strip()[-200:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ready, reasons = True, []
    if not args.draft:
        import holdout_p04_look as look
        look.check_labels()
        look.check_inputs()
        tail = SELFTEST_LOG.read_text(encoding="utf-8").strip().splitlines()[-1:] if SELFTEST_LOG.is_file() else []
        if not tail or "EXACT" not in tail[0]:
            raise SystemExit(f"NOT READY: look selftest has not passed ({tail})")

    files, missing = {}, []
    for group, paths in inventory().items():
        files[group] = {}
        for p in paths:
            key = str(p.relative_to(ROOT)).replace("\\", "/")
            if p.is_file():
                files[group][key] = {"sha256": sha256(p), "bytes": p.stat().st_size}
            else:
                files[group][key] = None
                missing.append(key)
    if missing:
        ready = False
        reasons.append(f"{len(missing)} files missing")
        if not args.draft:
            raise SystemExit(f"NOT READY: {len(missing)} files missing, e.g. {missing[0]}")

    step1 = json.loads((ROOT / "runs/holdout_p04/step1_manifest.json").read_text(encoding="utf-8"))
    out = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft": bool(args.draft), "ready": ready, "not_ready_because": reasons,
        "prereg": "docs/prereg-phase3-retrain-2026-09-10.md, Amendments 8 and 9",
        "git_head_before_freeze": subprocess.run(["git", "log", "-1", "--format=%H"], cwd=ROOT,
                                                 capture_output=True, text=True).stdout.strip(),
        "pohang04_label_hash": "c06611a684f4",
        "pohang04_ir_converted_sha256": step1["converted"]["sha256"],
        "rules": {"preset": "crossmodal26m", "systems": "VIS seed k + IR seed k, k=0..4",
                  "draws": {"vis": [941, 942, 943, 944], "ir": [951, 952, 953, 954]},
                  "verdict_cell": "clean/clean", "floor": 0.0060, "floors_reported": [0.0014, 0.0031, 0.0060, 0.0100],
                  "bootstrap": {"block_len": 20, "n_boot": 1000, "reference_seed": 0, "pohang04_seed": 1},
                  "AP_ref": json.loads((ROOT / "docs/eval/holdout_p04_devref_2026-09-14.json").read_text(
                      encoding="utf-8"))["reference"]},
        "environment": environment(),
        "files": files,
    }
    out["rules"]["AP_ref"] = {k: v for k, v in out["rules"]["AP_ref"].items() if k != "replicates"}
    n_files = sum(len(v) for v in files.values())
    dest = Path(args.out) if args.out else MANIFEST
    if not args.draft and dest.exists():
        raise SystemExit(f"{dest} already exists; a freeze manifest is written once")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[freeze] {'DRAFT ' if args.draft else ''}{n_files} files inventoried, {len(missing)} missing "
          f"-> {dest}")
    if not args.draft:
        print("[freeze] next: git add docs/eval/holdout_p04_freeze_manifest.json scripts/ docs/ && "
              "git commit -m 'FREEZE: pohang04 single look'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
