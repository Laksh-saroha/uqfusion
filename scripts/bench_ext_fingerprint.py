"""Is the local VIS dataset the one dgxanode01 trained the ep25 benchmark on?

`bench_ext_local.py` extends the server's benchmark runs on this machine. That only continues
those runs if the data is the same data. The server left three records, and this reproduces each
one from the local tree:

  F1  split lists. The benchmark's stride-4 train list is the server's own file (snapshot). The
      local val/test lists must hash to the sha256 the server recorded
      (`handoff/server_data_snapshot.txt`) after CRLF->LF.
  F2  images. `fp_server.txt` (`vis_fingerprint.py`, 2026-08-22) holds a path|size hash over
      stride-2 train + val (stride-4 is a subset).
  F3  labels as of 2026-08-22, from the same file. That predates the night-label restore, so it is
      checked against the pre-restore state rebuilt in memory: the 17,502 files `cut_dark`
      emptied (`runs/visfilter/visfilter_manifest.json`) read as empty. The local tree stores
      tens of thousands of label files with CRLF where the server has LF. The raw hash is taken
      after CRLF->LF, which changes no box: Ultralytics splits on whitespace. The normalised hash
      needs no conversion.
  F4  the restore. `_label_stage/` holds the pohang01 train labels the server was restored to
      (2026-09-04). Each staged file must be byte-identical to the local file, and the manifest's
      own hash must reproduce over the local directory.
  F5  what the benchmark actually read. Every Ultralytics train scan in the server's benchmark
      logs, base runs and extensions alike, reports "24070 images, N backgrounds". The label
      states give three different N on this list: original 86, filtered 4,461, restored 387.

`--quick` is the per-run gate. It recomputes only the pinned hashes of the files an extension
reads (stride-4 train + val: image path|size, labels CRLF->LF, restored state) and compares them
with PINNED, which records the values a full pass produced.

Read-only on the dataset.

Usage:
    python scripts/bench_ext_fingerprint.py                 # full proof, writes --out
    python scripts/bench_ext_fingerprint.py --quick         # gate: exit 1 on any drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "server_dgxanode01" / "workspace"
SRV_VIS = "/workspace/pohang/visible"
LOC_VIS = (ROOT / "Pohang_dataset" / "visible").as_posix()
SRV_STRIDE4 = SNAP / "derived" / "data_vis_train_stride4.txt"
SRV_STRIDE2 = SNAP / "derived" / "data_vis_train_stride2.txt"
LOC_STRIDE4 = ROOT / "runs" / "derived" / "data_vis_train_stride4.txt"
LOC_VAL = ROOT / "Pohang_dataset" / "visible" / "val.txt"
LOC_TEST = ROOT / "Pohang_dataset" / "visible" / "test.txt"
FP_SERVER = SNAP / "fp_server.txt"
DATA_SNAPSHOT = SNAP / "uqfusion" / "runs" / "queue_vis_benchmark_ep25" / "handoff" / "server_data_snapshot.txt"
QUEUE_LOGS = SNAP / "uqfusion" / "runs" / "queue_vis_benchmark_ep25"
STAGE = SNAP / "_label_stage"
VISFILTER = ROOT / "runs" / "visfilter" / "visfilter_manifest.json"
BACKGROUNDS = {"original": 86, "filtered": 4461, "restored": 387}

# Values the full pass produced on 2026-09-27 (all of F1-F5 passing). --quick compares against these.
PINNED = {
    "n_files": 35422,
    "img_sha": "e099b6e17ede0cdde9abc5eee4dd75128bc55e755751ac0a3a66ed91eb6481d4",
    "lbl_lf_sha": "9f4154effe719f8301b1e8442e00cfdb79e760a7b24d96cba07855a90032a7bc",
}


def srv_to_loc(p: str) -> str:
    return p.replace(SRV_VIS, LOC_VIS)


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def label_of(img: str) -> str:
    return os.path.splitext(img.replace("/images/", "/labels/"))[0] + ".txt"


def list_paths(txt: Path) -> list[str]:
    """Image paths from a split list, as forward-slash local paths."""
    out = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        ln = ln.strip().replace("\\", "/")
        if not ln:
            continue
        if ln.startswith("./"):
            ln = txt.parent.as_posix() + "/" + ln[2:]
        out.append(srv_to_loc(ln))
    return out


def canon(p: str) -> str:
    return p[p.find("/visible/") + 1:]


def fp_server() -> dict:
    out = {}
    for ln in FP_SERVER.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0] in ("n_images", "bytes", "img_sha", "lbl_raw", "lbl_norm"):
            out[parts[0]] = parts[1]
    return out


def recorded_list_shas() -> dict:
    out = {}
    for m in re.finditer(r"^(\S+) (\d+) lines sha256=([0-9a-f]{64})", DATA_SNAPSHOT.read_text(encoding="utf-8"), re.M):
        out[Path(m.group(1)).name] = m.group(3)
    return out


def pinned_hashes(paths: list[str]) -> dict:
    """Image path|size and label (CRLF->LF) hashes over `paths`, keyed canonically, restored state."""
    hi, hl = hashlib.sha256(), hashlib.sha256()
    keys = {canon(p): p for p in paths}
    for k in sorted(keys):
        p = keys[k]
        hi.update(f"{k}|{os.path.getsize(p)}\n".encode())
        hl.update(f"{k}|{hashlib.sha256(lf(open(label_of(p), 'rb').read())).hexdigest()}\n".encode())
    return {"n_files": len(keys), "img_sha": hi.hexdigest(), "lbl_lf_sha": hl.hexdigest()}


def quick_paths() -> list[str]:
    return list_paths(LOC_STRIDE4) + list_paths(LOC_VAL)


def full() -> dict:
    res, ok = {}, True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        res[name] = {"pass": bool(passed), "detail": detail}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    # F1 -- lists
    want = recorded_list_shas()
    s4 = SRV_STRIDE4.read_bytes()
    check("F1 stride4 list (server file)", hashlib.sha256(s4).hexdigest() == want["data_vis_train_stride4.txt"],
          f"{len(s4.splitlines())} lines")
    back = "".join(p.replace(LOC_VIS, SRV_VIS) + "\n" for p in list_paths(LOC_STRIDE4)).encode()
    check("F1 stride4 list (local copy maps back)", back == s4, LOC_STRIDE4.relative_to(ROOT).as_posix())
    for f, key in ((LOC_VAL, "val.txt"), (LOC_TEST, "test.txt")):
        h = hashlib.sha256(lf(f.read_bytes())).hexdigest()
        check(f"F1 {key}", h == want[key], f"{h[:12]} vs server {want[key][:12]}")

    # F2/F3 -- vis_fingerprint.py over stride-2 train + val
    fp = fp_server()
    man = json.loads(VISFILTER.read_text(encoding="utf-8"))
    emptied = {os.path.normcase(str(Path(k).resolve())) for k in man["dropped"]}
    keys = {}
    for p in list_paths(SRV_STRIDE2) + list_paths(LOC_VAL):
        keys[canon(p)] = p
    hi, hr, hn = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n = total = crlf = 0
    for k in sorted(keys):
        p = keys[k]
        sz = os.path.getsize(p)
        n += 1
        total += sz
        hi.update(f"{k}|{sz}\n".encode())
        lp = label_of(p)
        raw = b"" if os.path.normcase(str(Path(lp).resolve())) in emptied else open(lp, "rb").read()
        crlf += b"\r\n" in raw
        hr.update(f"{k}|{hashlib.sha256(lf(raw)).hexdigest()}\n".encode())
        norm = "\n".join(" ".join(ln.split()) for ln in raw.decode("utf-8", "replace").splitlines() if ln.strip())
        hn.update(f"{k}|{norm}\n".encode())
    check("F2 images n/bytes", (str(n), str(total)) == (fp["n_images"], fp["bytes"]), f"{n} files, {total} bytes")
    check("F2 img_sha", hi.hexdigest() == fp["img_sha"], hi.hexdigest()[:12])
    check("F3 lbl_raw (pre-restore, CRLF->LF)", hr.hexdigest() == fp["lbl_raw"],
          f"{hr.hexdigest()[:12]}; {crlf} local files carry CRLF")
    check("F3 lbl_norm (pre-restore)", hn.hexdigest() == fp["lbl_norm"], hn.hexdigest()[:12])

    # F4 -- the server restore
    sm = json.loads((STAGE / "manifest.json").read_text(encoding="utf-8"))
    loc_dir = Path(LOC_VIS) / "labels" / sm["run"]
    diff = [nm for nm in sm["files"] if (STAGE / nm).read_bytes() != (loc_dir / nm).read_bytes()]
    check("F4 staged files byte-identical to local", not diff, f"{len(sm['files']) - len(diff)}/{len(sm['files'])}")
    h = hashlib.sha256()
    for nm in sm["files"]:
        h.update(f"{sm['run']}/{nm}\n".encode())
        h.update((loc_dir / nm).read_bytes())
        h.update(b"\x00")
    check("F4 staged manifest hash over local dir", h.hexdigest()[:12] == sm["pohang01_train_label_hash"],
          f"{h.hexdigest()[:12]} ({sm['n_boxes']} boxes staged)")

    # F5 -- what the server benchmark read
    seen = {}
    for log in QUEUE_LOGS.rglob("*.log"):
        txt = log.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
        txt = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", txt)                  # Ultralytics colours "train: "
        for m in re.finditer(r"train: Scanning .*? (\d+) images, (\d+) backgrounds", txt):
            if m.group(1) == "24070":
                seen[int(m.group(2))] = seen.get(int(m.group(2)), 0) + 1
    s4_labels = [label_of(p) for p in list_paths(LOC_STRIDE4)]
    counts = {"restored": sum(os.path.getsize(q) == 0 for q in s4_labels),
              "filtered": sum(os.path.getsize(q) == 0 or os.path.normcase(str(Path(q).resolve())) in emptied
                              for q in s4_labels)}
    orig = 0
    for q in s4_labels:
        b = q + ".pre_visfilter"
        orig += os.path.getsize(b if os.path.isfile(b) else q) == 0
    counts["original"] = orig
    check("F5 server scans all report the restored count", set(seen) == {BACKGROUNDS["restored"]},
          f"server scans {seen}; local stride-4 backgrounds {counts}")
    check("F5 local restored tree gives that count", counts == BACKGROUNDS, "original/filtered/restored distinct")

    pin = pinned_hashes(quick_paths())
    print(f"[info] pinned values for --quick: {pin}", flush=True)
    return {"pass": bool(ok), "checks": res, "pinned": pin}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="per-run gate against PINNED")
    ap.add_argument("--out", default="runs/vis_benchmark_stride4_ep25_ext/fingerprint_full.json")
    args = ap.parse_args()
    if args.quick:
        if None in PINNED.values():
            print("[FAIL] PINNED is empty: run the full check first and pin its values", flush=True)
            return 1
        got = pinned_hashes(quick_paths())
        bad = {k: (got[k], PINNED[k]) for k in PINNED if got[k] != PINNED[k]}
        print(f"[{'FAIL' if bad else 'PASS'}] quick gate over {got['n_files']} stride-4 train + val files"
              + (f": {bad}" if bad else ""), flush=True)
        return 1 if bad else 0
    r = full()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(f"{'ALL PASS' if r['pass'] else 'FAILED'} -> {out.relative_to(ROOT).as_posix()}", flush=True)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
