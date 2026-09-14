"""Phase 3 §7, pre-freeze — per-frame brightness and structure statistics for pohang04.

The shipped gate (`crossmodal26m`) reads image statistics, not only detections: VIS
`p05`/`lap_var` (brightness), VIS `grad_gini` and the VIS health vector (structure), IR
`p05` (brightness) and the IR health vector (structure). `load_context` refuses to run
without them, so they must exist before the freeze. **They are pixel arithmetic — no
detector, no label, no score.**

They depend on the image and its corruption only, never on the checkpoint, so each of the
38 streams (2 clean + 9 corrupted x 4 draws, Amendment 9 §A9.2) is measured ONCE and
serves all five Phase 3 systems.

Equivalence is proven, not assumed: before touching pohang04 this reproduces the first
`--verify-n` frames of three existing development files bit-exactly through the same
code path — VIS fog (seed 1) brightness and structure, and IR glare_s2 (seed 7)
structure. It calls `frame_brightness.frame_stats` and `frame_structure.frame_stats`
directly, with the same `cv2.imread` -> `make_corruption(kind, sev, seed)(im, index)` ->
gray -> content-rows crop as those scripts and `build_cache` use. One image read and one
corruption replay serve both statistics.

Output: `runs/holdout_p04/derived/{brightness,structure}/{clean|draw{v}_{v+10}}/<stem>.json`,
in the exact payload format of the development files. Written to `.tmp`, then renamed; an
existing file is skipped only when its corruption, severity, seed and frame count match.

Usage:
    python scripts/holdout_p04_frame_stats.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                                     # noqa: E402

import frame_brightness as fb                                  # noqa: E402
import frame_structure as fs                                   # noqa: E402
from gate_snms_draw_avg import CORRUPTED                       # noqa: E402
from uqfusion.eval.corruptions import make_corruption         # noqa: E402

fs.cv2 = cv2      # frame_structure imports cv2 inside main(); its frame_stats needs it too
HP = ROOT / "runs" / "holdout_p04"
OUT = HP / "derived"
VIS_DRAWS = (941, 942, 943, 944)
LISTS = {"vis": HP / "p04_pairs_vis.txt", "ir": HP / "p04_pairs_ir.txt"}


def read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def measure(paths: list[str], modality: str, kind, sev, seed, n: int | None = None):
    lo, hi = fb.content_rows(modality)
    if (lo, hi) != fs.content_rows(modality):
        raise SystemExit("frame_brightness and frame_structure disagree on content rows")
    tf = make_corruption(kind, sev, seed) if kind else None
    br, st = [], []
    for i, p in enumerate(paths[:n] if n else paths):
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        if tf is not None:
            im = tf(im, i)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]
        tag = {"image_path": p, "run": Path(p).parent.name}
        br.append({**fb.frame_stats(g), **tag})
        st.append({**fs.frame_stats(g), **tag})
    return (lo, hi), br, st


def verify(n: int) -> None:
    pv = read_list(ROOT / "runs/derived/paired_val_vis.txt")
    pi = read_list(ROOT / "runs/derived/paired_val_ir.txt")
    checks = [("brightness", "gauss_vis_paired_fog", pv, "vis"),
              ("structure", "gauss_vis_paired_fog", pv, "vis"),
              ("structure", "gauss_ir_paired_glare_s2", pi, "ir")]
    for kind_dir, stem, paths, mod in checks:
        ref = json.loads((ROOT / "runs/derived" / kind_dir / f"{stem}.json").read_text(encoding="utf-8"))
        _, br, st = measure(paths, mod, ref["corrupt"], ref["severity"], ref["corrupt_seed"], n)
        got = br if kind_dir == "brightness" else st
        for i in range(n):
            a, b = got[i], ref["frames"][i]
            diff = [k for k in b if k not in ("image_path",) and a.get(k) != b[k]]
            if Path(a["image_path"]).name != Path(b["image_path"]).name or diff:
                raise SystemExit(f"[verify] FAIL {kind_dir}/{stem} frame {i}: {diff[:5]}")
        print(f"[verify] {kind_dir}/{stem}: {n}/{n} frames bit-exact", flush=True)


def streams() -> list[dict]:
    out = [{"mod": m, "stem": f"gauss_{m}_paired_clean", "kind": None, "sev": None, "seed": None,
            "sub": "clean"} for m in ("vis", "ir")]
    for v in VIS_DRAWS:
        for mod, stem, kind, sev in CORRUPTED:
            out.append({"mod": mod, "stem": stem, "kind": kind, "sev": sev,
                        "seed": v + 10 if mod == "ir" else v, "sub": f"draw{v}_{v + 10}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-n", type=int, default=40)
    args = ap.parse_args()
    verify(args.verify_n)
    lists = {m: read_list(p) for m, p in LISTS.items()}
    if len(lists["vis"]) != len(lists["ir"]):
        raise SystemExit("pair lists differ in length")
    todo = streams()
    t0 = time.time()
    for j, s in enumerate(todo, 1):
        outs = {d: OUT / d / s["sub"] / f"{s['stem']}.json" for d in ("brightness", "structure")}
        def ok(p: Path) -> bool:
            if not p.is_file():
                return False
            h = json.loads(p.read_text(encoding="utf-8"))
            return (h["corrupt"], h["severity"], h["corrupt_seed"], h["n_frames"]) == \
                (s["kind"], s["sev"], s["seed"], len(lists[s["mod"]]))
        if all(ok(p) for p in outs.values()):
            print(f"[{j}/{len(todo)}] {s['sub']}/{s['stem']} skip (verified)", flush=True)
            continue
        t = time.time()
        rows, br, st = measure(lists[s["mod"]], s["mod"], s["kind"], s["sev"], s["seed"])
        for d, frames in (("brightness", br), ("structure", st)):
            p = outs[d]
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"cache": f"runs/holdout_p04/caches/seed*/{s['sub']}/{s['stem']}.pkl",
                       "modality": s["mod"], "content_rows": list(rows), "corrupt": s["kind"],
                       "severity": s["sev"], "corrupt_seed": s["seed"], "n_frames": len(frames),
                       "frames": frames}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, p)
        print(f"[{j}/{len(todo)}] {s['sub']}/{s['stem']} {len(br)} frames in {time.time() - t:.0f}s "
              f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
    print("[done] all streams verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
