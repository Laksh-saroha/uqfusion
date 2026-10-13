"""Measured frame intervals per run and camera, from the dataset's own timestamp files.

Chung et al. (2023) give both the stereo and the thermal camera as 10 Hz, and note that some
thermal intervals run long because of the camera's thermal calibration. This measures what the
files on disk actually contain: `Pohang_dataset/meta/<run>/timestamps/{stereo,ir}.txt`, one
`<unix seconds> <frame index>` row per frame.

pohang04 is not read. Nothing here needs it, and it is the spent held-out run.

Usage:
    python scripts/measure_frame_intervals.py --out docs/eval/frame_intervals_2026-10-08.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "Pohang_dataset" / "meta"
RUNS = ("pohang00", "pohang01", "pohang02", "pohang03")
STREAMS = (("stereo", "VIS (stereo, left)"), ("ir", "IR"))
LONG_MS = 150.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lines = [
        "# Frame intervals, pohang00–03",
        "",
        "Written by `scripts/measure_frame_intervals.py` from `Pohang_dataset/meta/<run>/timestamps/`. "
        f"`long` counts intervals over {LONG_MS:.0f} ms. pohang04 is not read.",
        "",
        "| run | camera | frames | median ms | p01 ms | p99 ms | max ms | long | long % | rate Hz |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        for key, label in STREAMS:
            t = np.loadtxt(META / run / "timestamps" / f"{key}.txt", usecols=0, dtype=np.float64)
            d = np.diff(t) * 1000.0
            n_long = int((d > LONG_MS).sum())
            row = (f"| {run} | {label} | {len(t)} | {np.median(d):.1f} | {np.percentile(d, 1):.1f} | "
                   f"{np.percentile(d, 99):.1f} | {d.max():.1f} | {n_long} | {100 * n_long / len(d):.2f} | "
                   f"{len(t) / (t[-1] - t[0]):.3f} |")
            lines.append(row)
            print(row)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
