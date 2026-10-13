"""Phase 3 §7 step 1 — bring pohang04 thermal into the tree, build its pairs, measure day/night.

Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendment 2 §A2.2, Amendment 8 §A8.4
and Amendment 9 §A9.4 / A9.5 item 1. **Nothing here runs a detector or reads a score.**
It reads pixels, filenames, timestamps and GPS fixes, and writes one manifest.

Stages, each refusing to continue on a failed check:

  a. Source provenance -- sha256 over the 16-bit frame set (sorted names, basename then
     bytes), frame count, `timestamp.txt` digest; the count must equal
     `meta/pohang04/timestamps/ir.txt`.
  b. Conversion path re-verified -- the exact 16->8-bit transform of
     `D:/Datasets/Pohang/_scripts/convert_ir_to_8bit.py` (per-frame min-max, +0.5, uint8)
     followed by `prepare_pohang.letterbox_image` (640x640, pad 114) must reproduce
     pohang00-03 tree frames from their D: sources BIT-EXACTLY on a seeded sample. If the
     path cannot reproduce the development runs, pohang04 would test the conversion.
  c. Convert + letterbox every pohang04 frame into `Pohang_dataset/infrared/images/pohang04/`,
     recording degenerate (zero-variance) frames, the output content hash, and each
     frame's content-row p05 -- the statistic `frame_brightness.py` feeds the IR night flag.
  d. Pairs by A8.4: `meta/pohang04/pairs.csv` rows with in_tolerance == 1, a left stereo
     label present, and the IR frame present in (c). Written in the pohang00-03 format.
  e. Day/night by A9.4: solar elevation (NOAA algorithm, geometric, no refraction) at the
     stereo capture time and the nearest GPS fix; day iff elevation > 0 deg. Cross-check:
     `ir_p05 > 41.5` (structure_constants.json), disagreements counted, never reconciled.
     The solar code is sanity-checked on pohang00 (day) and pohang01 (night) first.

Usage:
    python scripts/holdout_p04_step1.py
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "Pohang_dataset"
RAW = Path("D:/Datasets/Pohang")
SRC = RAW / "pohang04" / "infrared" / "images"
DST = DS / "infrared" / "images" / "pohang04"
META = DS / "meta" / "pohang04"
PAIRS_OUT = DS / "paired" / "pohang04_pairs.csv"
OUT_DIR = ROOT / "runs" / "holdout_p04"

TARGET, PAD = 640, 114                    # prepare_pohang.py
IR_CONTENT = (64, 576)                    # frame_brightness.content_rows("ir")
IR_NIGHT_THR = 41.5                       # structure_constants.json axes.ir_p05.threshold
VERIFY_PER_RUN, VERIFY_SEED = 15, 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_8bit(a: np.ndarray) -> tuple[np.ndarray, bool]:
    """convert_ir_to_8bit.py, verbatim arithmetic."""
    if a.dtype != np.uint16:
        raise SystemExit(f"unexpected dtype {a.dtype}")
    lo, hi = a.min(), a.max()
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8), True
    return ((a.astype(np.float32) - lo) / (hi - lo) * 255 + 0.5).astype(np.uint8), False


def letterbox(a8: np.ndarray) -> np.ndarray:
    """prepare_pohang.letterbox_image, verbatim, returning the array."""
    im = Image.fromarray(a8)
    w, h = im.size
    scale = min(TARGET / w, TARGET / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = im.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (w, h) else im
    canvas = Image.new("L", (TARGET, TARGET), PAD)
    canvas.paste(resized, ((TARGET - nw) // 2, (TARGET - nh) // 2))
    return np.array(canvas)


def solar_elevation(epoch: float, lat: float, lon: float) -> float:
    """NOAA solar position (spreadsheet algorithm), geometric elevation in degrees, UTC."""
    rad, deg = np.radians, np.degrees
    jd = epoch / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    L = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    C = (np.sin(rad(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + np.sin(rad(2 * M)) * (0.019993 - 0.000101 * jc) + np.sin(rad(3 * M)) * 0.000289)
    omega = 125.04 - 1934.136 * jc
    app_long = L + C - 0.00569 - 0.00478 * np.sin(rad(omega))
    mean_obl = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obl = mean_obl + 0.00256 * np.cos(rad(omega))
    decl = deg(np.arcsin(np.sin(rad(obl)) * np.sin(rad(app_long))))
    y = np.tan(rad(obl / 2)) ** 2
    eq_time = 4 * deg(y * np.sin(2 * rad(L)) - 2 * e * np.sin(rad(M))
                      + 4 * e * y * np.sin(rad(M)) * np.cos(2 * rad(L))
                      - 0.5 * y * y * np.sin(4 * rad(L)) - 1.25 * e * e * np.sin(2 * rad(M)))
    tst = ((epoch % 86400) / 60.0 + eq_time + 4 * lon) % 1440
    ha = tst / 4 - 180 if tst / 4 >= 0 else tst / 4 + 180
    cz = (np.sin(rad(lat)) * np.sin(rad(decl))
          + np.cos(rad(lat)) * np.cos(rad(decl)) * np.cos(rad(ha)))
    return float(90.0 - deg(np.arccos(np.clip(cz, -1, 1))))


def load_ts(p: Path) -> dict[str, float]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[1]] = float(parts[0])
    return out


def load_gps(p: Path) -> tuple[list[float], list[tuple[float, float]]]:
    """Valid fixes only. Column 7 is the NMEA fix quality; 0 means no fix, and those rows
    carry lat/lon 0.000000 (8 such rows in pohang04). Found 2026-09-14 because they put 27
    mid-afternoon frames at the Gulf of Guinea, solar elevation -12.2 deg, and the rule
    called them night. An implementation defect in reading the fix, not a change to A9.4."""
    ts, pos = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) < 8 or f[7] == "0" or (float(f[2]) == 0.0 and float(f[4]) == 0.0):
            continue
        lat = float(f[2]) * (1 if f[3] == "N" else -1)
        lon = float(f[4]) * (1 if f[5] == "E" else -1)
        ts.append(float(f[0]))
        pos.append((lat, lon))
    return ts, pos


def nearest(ts: list[float], t: float) -> int:
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return 0
    if i == len(ts):
        return len(ts) - 1
    return i if ts[i] - t < t - ts[i - 1] else i - 1


def run_elevation(run: str) -> tuple[float, float]:
    """Elevation at the first stereo frame of a run -- the solar code's sanity check."""
    m = DS / "meta" / run
    st = load_ts(m / "timestamps" / "stereo.txt")
    gt, gp = load_gps(m / "navigation" / "gps.txt")
    t0 = min(st.values())
    lat, lon = gp[nearest(gt, t0)]
    return t0, solar_elevation(t0, lat, lon)


def recompute_composition() -> int:
    """Re-run stage (e) alone on the already-built pairs and converted frames, keeping the
    superseded result in the manifest. Stages a-d refuse to overwrite, by design."""
    mp = OUT_DIR / "step1_manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8"))
    with open(PAIRS_OUT, newline="", encoding="utf-8") as fh:
        rows = [(r["stereo_L_file"].split("_")[-1][:-4], r["ir_file"].split("_")[-1][:-4], r["dt_ms"])
                for r in csv.DictReader(fh)]
    if sha256_file(PAIRS_OUT) != man["pairs"]["sha256"]:
        raise SystemExit("pair table changed since step 1")
    p05 = {i: float(np.percentile(np.array(Image.open(DST / f"pohang04_{i}.png"))[IR_CONTENT[0]:IR_CONTENT[1]]
                                  .astype(np.float32), 5)) for _, i, _ in rows}
    man.setdefault("composition_superseded", []).append(
        {**man["composition"], "superseded_because": "GPS no-fix rows (quality 0, lat/lon 0,0) were used as positions"})
    man["composition"] = composition(rows, p05)
    mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
    c = man["composition"]
    print(f"[e] composition: {c['n_day']} day / {c['n_night']} night; elevation "
          f"{c['elevation_deg']['min']:.1f}..{c['elevation_deg']['max']:.1f}; IR flag night "
          f"{c['crosscheck_ir_p05']['n_ir_night']}, disagreements {c['crosscheck_ir_p05']['n_disagree']}")
    return 0


def composition(rows, p05) -> dict:
    sanity = {r: dict(zip(("epoch", "elevation_deg"), run_elevation(r))) for r in ("pohang00", "pohang01")}
    if not (sanity["pohang00"]["elevation_deg"] > 0 > sanity["pohang01"]["elevation_deg"]):
        raise SystemExit("[e] FAIL: solar code does not call pohang00 day and pohang01 night")
    st = load_ts(META / "timestamps" / "stereo.txt")
    gt, gp = load_gps(META / "navigation" / "gps.txt")
    elev, gps_gap, ir_night = [], [], []
    for s, i, _ in rows:
        t = st[s]
        j = nearest(gt, t)
        gps_gap.append(abs(gt[j] - t))
        elev.append(solar_elevation(t, *gp[j]))
        ir_night.append(p05[i] > IR_NIGHT_THR)
    elev, ir_night = np.asarray(elev), np.asarray(ir_night)
    day = elev > 0
    return {
        "rule": "A9.4: solar elevation > 0 deg (geometric, NOAA) at stereo capture time, nearest valid GPS fix",
        "n_pairs": int(len(rows)), "n_day": int(day.sum()), "n_night": int((~day).sum()),
        "elevation_deg": {"min": float(elev.min()), "max": float(elev.max())},
        "capture_utc": {"first": datetime.fromtimestamp(min(st[s] for s, _, _ in rows), timezone.utc).isoformat(),
                        "last": datetime.fromtimestamp(max(st[s] for s, _, _ in rows), timezone.utc).isoformat()},
        "gps_gap_s_max": float(max(gps_gap)), "sanity": sanity,
        "crosscheck_ir_p05": {"threshold": IR_NIGHT_THR, "n_ir_night": int(ir_night.sum()),
                              "n_disagree": int((ir_night == day).sum()),
                              "n_ir_night_on_solar_day": int((ir_night & day).sum()),
                              "note": "disagree = IR flag night on a solar-day frame, or day on a solar-night frame"},
    }


def main() -> int:
    if "--composition-only" in sys.argv:
        return recompute_composition()
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    man: dict = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "script": "scripts/holdout_p04_step1.py"}

    # ---- a. source provenance -------------------------------------------------------
    src_files = sorted(SRC.glob("*.png"))
    h = hashlib.sha256()
    for f in src_files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    n_meta_ir = len(load_ts(META / "timestamps" / "ir.txt"))
    man["source"] = {
        "dir": str(SRC), "n_frames": len(src_files), "sha256": h.hexdigest(),
        "timestamp_txt_sha256": sha256_file(SRC.parent / "timestamp.txt"),
        "meta_ir_txt_sha256": sha256_file(META / "timestamps" / "ir.txt"),
        "meta_ir_txt_lines": n_meta_ir,
    }
    print(f"[a] {len(src_files)} source frames, sha256 {h.hexdigest()[:12]}; meta ir.txt {n_meta_ir}", flush=True)
    if len(src_files) != n_meta_ir:
        raise SystemExit("[a] FAIL: source frame count != meta ir.txt lines")

    # ---- b. conversion path reproduces the development runs ------------------------
    rng = random.Random(VERIFY_SEED)
    checked = 0
    for run in ("pohang00", "pohang01", "pohang02", "pohang03"):
        tree = sorted((DS / "infrared" / "images" / run).glob("*.png"))
        for f in rng.sample(tree, VERIFY_PER_RUN):
            raw = RAW / run / "infrared" / "images" / f"{f.stem.split('_')[1]}.png"
            got = letterbox(to_8bit(np.array(Image.open(raw)))[0])
            if not np.array_equal(got, np.array(Image.open(f))):
                raise SystemExit(f"[b] FAIL: conversion path does not reproduce {f}")
            checked += 1
    man["conversion"] = {
        "rule": "per-frame min-max to uint8 ((x-lo)/(hi-lo)*255+0.5), zero-variance -> all 0; "
                "then letterbox 640x512 -> 640x640, pad 114, no resize",
        "convert_script": str(RAW / "_scripts" / "convert_ir_to_8bit.py"),
        "convert_script_sha256": sha256_file(RAW / "_scripts" / "convert_ir_to_8bit.py"),
        "letterbox_script_sha256": sha256_file(ROOT / "scripts" / "prepare_pohang.py"),
        "reproduced_dev_frames_bit_exact": checked,
    }
    print(f"[b] conversion path reproduced {checked}/{checked} pohang00-03 frames bit-exactly", flush=True)

    # ---- c. convert pohang04 --------------------------------------------------------
    if DST.exists() and any(DST.iterdir()):
        raise SystemExit(f"[c] FAIL: {DST} already holds files; refusing to overwrite")
    DST.mkdir(parents=True)
    degenerate, p05 = [], {}
    hd = hashlib.sha256()
    for k, f in enumerate(src_files):
        a8, degen = to_8bit(np.array(Image.open(f)))
        if degen:
            degenerate.append(f.name)
        lb = letterbox(a8)
        name = f"pohang04_{f.stem}.png"
        Image.fromarray(lb).save(DST / name)
        hd.update(name.encode())
        hd.update((DST / name).read_bytes())
        p05[f.stem] = float(np.percentile(lb[IR_CONTENT[0]:IR_CONTENT[1]].astype(np.float32), 5))
        if (k + 1) % 2000 == 0:
            print(f"[c] {k + 1}/{len(src_files)} ({time.time() - t_start:.0f}s)", flush=True)
    man["converted"] = {"dir": str(DST), "n_frames": len(src_files), "sha256": hd.hexdigest(),
                        "n_degenerate": len(degenerate), "degenerate": degenerate}
    print(f"[c] wrote {len(src_files)} frames, {len(degenerate)} degenerate, sha256 {hd.hexdigest()[:12]}", flush=True)

    # ---- d. pairs -------------------------------------------------------------------
    existing = PAIRS_OUT.read_text(encoding="utf-8").splitlines() if PAIRS_OUT.is_file() else []
    if len(existing) > 1:
        raise SystemExit(f"[d] FAIL: {PAIRS_OUT} already has rows; refusing to overwrite")
    lbl_dir = DS / "visible" / "labels" / "pohang04"
    n_rows = n_tol = n_lbl = 0
    rows = []
    with open(META / "pairs.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            if row["in_tolerance"] != "1":
                continue
            n_tol += 1
            s, i = row["stereo_frame"], row["ir_frame"]
            if not (lbl_dir / f"pohang04_L_{s}.txt").is_file():
                continue
            n_lbl += 1
            if not (DST / f"pohang04_{i}.png").is_file():
                continue
            rows.append((s, i, row["dt_ms"]))
    with open(PAIRS_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "stereo_L_file", "stereo_R_file", "ir_file", "dt_ms"])
        for s, i, dt in rows:
            w.writerow(["pohang04", f"pohang04_L_{s}.png", f"pohang04_R_{s}.png", f"pohang04_{i}.png", dt])
    abs_dt = sorted(abs(float(dt)) for _, _, dt in rows)
    man["pairs"] = {"file": str(PAIRS_OUT), "sha256": sha256_file(PAIRS_OUT),
                    "meta_rows": n_rows, "in_tolerance": n_tol, "with_left_label": n_lbl,
                    "emitted": len(rows),
                    "abs_dt_ms_median": abs_dt[len(abs_dt) // 2] if abs_dt else None,
                    "abs_dt_ms_max": abs_dt[-1] if abs_dt else None,
                    "rule": "A8.4: in_tolerance==1 AND left stereo label present AND IR frame converted"}
    print(f"[d] pairs: {n_rows} meta rows -> {n_tol} in tolerance -> {n_lbl} with L label -> {len(rows)} emitted", flush=True)

    # ---- e. day/night ---------------------------------------------------------------
    sanity = {r: dict(zip(("epoch", "elevation_deg"), run_elevation(r))) for r in ("pohang00", "pohang01")}
    print(f"[e] solar sanity: pohang00 {sanity['pohang00']['elevation_deg']:+.1f} deg, "
          f"pohang01 {sanity['pohang01']['elevation_deg']:+.1f} deg", flush=True)
    if not (sanity["pohang00"]["elevation_deg"] > 0 > sanity["pohang01"]["elevation_deg"]):
        raise SystemExit("[e] FAIL: solar code does not call pohang00 day and pohang01 night")
    st = load_ts(META / "timestamps" / "stereo.txt")
    gt, gp = load_gps(META / "navigation" / "gps.txt")
    elev, gps_gap, ir_night = [], [], []
    for s, i, _ in rows:
        t = st[s]
        j = nearest(gt, t)
        gps_gap.append(abs(gt[j] - t))
        elev.append(solar_elevation(t, *gp[j]))
        ir_night.append(p05[i] > IR_NIGHT_THR)
    elev = np.asarray(elev)
    day = elev > 0
    ir_night = np.asarray(ir_night)
    man["composition"] = {
        "rule": "A9.4: solar elevation > 0 deg (geometric, NOAA) at stereo capture time, nearest GPS fix",
        "n_pairs": int(len(rows)), "n_day": int(day.sum()), "n_night": int((~day).sum()),
        "elevation_deg": {"min": float(elev.min()), "max": float(elev.max())} if len(elev) else None,
        "capture_utc": {"first": datetime.fromtimestamp(min(st[s] for s, _, _ in rows), timezone.utc).isoformat(),
                        "last": datetime.fromtimestamp(max(st[s] for s, _, _ in rows), timezone.utc).isoformat()},
        "gps_gap_s_max": float(max(gps_gap)) if gps_gap else None,
        "sanity": sanity,
        "crosscheck_ir_p05": {"threshold": IR_NIGHT_THR, "n_ir_night": int(ir_night.sum()),
                              "n_disagree": int((ir_night == day).sum()),
                              "note": "disagree = IR flag says night on a solar-day frame, or day on a solar-night frame"},
    }
    print(f"[e] composition: {int(day.sum())} day / {int((~day).sum())} night; elevation "
          f"{elev.min():.1f}..{elev.max():.1f} deg; IR flag night {int(ir_night.sum())}, "
          f"disagreements {int((ir_night == day).sum())}", flush=True)

    man["seconds"] = round(time.time() - t_start, 1)
    (OUT_DIR / "step1_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[done] {OUT_DIR / 'step1_manifest.json'} in {man['seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
