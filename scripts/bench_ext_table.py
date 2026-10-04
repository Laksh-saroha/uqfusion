"""Table 1 for the patience-20 VIS benchmark: the ep25 grid plus its extensions, mean ± sd over seeds.

Reads only what the runs already logged, so nothing is evaluated and nothing is scored:

  * base run   `server_dgxanode01/.../vis_benchmark_stride4_ep25/<run>/results.csv` (25 epochs, dgxanode01)
  * extension  the server's `<run>_ext/` for the 10 runs it finished, else this machine's
               `runs/vis_benchmark_stride4_ep25_ext/<run>_ext/`, counted only once `ext_done.json` says done

A run's score is the row at its best mAP50-95 epoch over base + extension, earliest maximum, which is
the checkpoint EarlyStopping keeps (fitness in Ultralytics 8.4.90 is mAP50-95 alone). Precision, recall
and mAP50 come from that same row. The best epoch is selected on the val split it is reported on, as in
the Phase 1 table.

Runs still training or queued are left out and listed; a family with fewer finished seeds than it has
runs is shown below the ranking, not in it. Re-run at any time; the table says how many runs it holds.

Usage (from the repo root):
    python scripts/bench_ext_table.py
    python scripts/bench_ext_table.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ext_local import (BASE, EXCLUDED, K, OUT, PATIENCE, SERVER_PROGRESS, family, replay,  # noqa: E402
                             server_done)

COLS = {"precision": "metrics/precision(B)", "recall": "metrics/recall(B)", "map50": "metrics/mAP50(B)",
        "map50_95": K}
RUN_FIELDS = ["run_id", "family", "status", "host", "batch", "workers", "base_epochs", "ext_epochs",
              "best_epoch", "base_map50_95", "precision", "recall", "map50", "map50_95", "improved",
              "stop", "replay", "rows_contiguous", "note"]


def rows_of(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [{"epoch": int(float(r["epoch"])), **{k: float(r[c]) for k, c in COLS.items()}}
            for r in csv.DictReader(open(path, encoding="utf-8"))]


def args_of(run_dir: Path) -> dict:
    p = run_dir / "args.yaml"
    out = {}
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            if k in ("batch", "workers"):
                out[k] = int(v.strip())
    return out


def contiguous(rows: list[dict]) -> bool:
    return [r["epoch"] for r in rows] == list(range(1, len(rows) + 1))


def server_lengths() -> dict[str, int]:
    """{run_id: extension epochs} for the runs the server finished, from its progress file."""
    return {r["run_id"]: int(r["current_epoch"]) for r in csv.DictReader(open(SERVER_PROGRESS, encoding="utf-8"))
            if r["run_id"] in server_done()}


def one_run(rid: str, on_server: dict[str, int]) -> dict:
    base = rows_of(BASE / rid / "results.csv")
    b_best = max(base, key=lambda r: r["map50_95"])               # max() keeps the first maximum
    rec = {"run_id": rid, "family": family(rid), "base_epochs": len(base), "base_map50_95": b_best["map50_95"]}

    if rid in on_server:
        ext_dir = BASE / f"{rid}_ext"
        ext = rows_of(ext_dir / "results.csv")
        gap = base[-1]["epoch"] - b_best["epoch"]
        n_run = on_server[rid]                                    # the server's own count; results.csv can be short
        stop_at, _, be = replay(b_best["map50_95"], gap, [(r["epoch"], r["map50_95"]) for r in ext])
        ok = stop_at == n_run
        if stop_at is None and be + PATIENCE == n_run and len(ext) < n_run:
            # Rows after the best are missing, and the stopper fired exactly PATIENCE epochs after it, so none
            # of the missing rows was a new best: the best and the stop both stand.
            ok = True
            rec["note"] = (f"results.csv ends at ext epoch {len(ext)}, the run at {n_run} (last.pt); "
                           f"stop = best {be} + {PATIENCE}, so the missing rows hold no new best")
        rec.update(status="done", host="dgxanode01", **args_of(ext_dir), stop="patience" if ok else "epochs",
                   replay="OK" if ok else "MISMATCH")
        ext_n = n_run
    else:
        ext_dir = OUT / f"{rid}_ext"
        done = json.loads((ext_dir / "ext_done.json").read_text()) if (ext_dir / "ext_done.json").is_file() else {}
        ext = rows_of(ext_dir / "results.csv")
        if done.get("status") != "done":
            rec.update(status="training" if ext else "queued", ext_epochs=len(ext))
            return rec
        rec.update(status="done", host=done.get("host", ""), batch=done.get("batch"), workers=done.get("workers"),
                   stop=done.get("stop", ""), replay=done.get("replay", ""))
        ext_n = int(done.get("ext_epochs", len(ext)))

    best = b_best
    for r in ext:                                                 # strict >: the earliest maximum stands
        if r["map50_95"] > best["map50_95"]:
            best = {**r, "epoch": len(base) + r["epoch"]}
    rec.update(ext_epochs=ext_n, best_epoch=best["epoch"], improved=best["map50_95"] > b_best["map50_95"],
               rows_contiguous=contiguous(base) and contiguous(ext) and len(ext) == ext_n,
               **{k: best[k] for k in COLS})
    if not rec["rows_contiguous"] and "note" not in rec:
        rec["note"] = "results.csv skips an epoch; the best is taken over the rows present"
    return rec


def ms(xs: list[float], nd: int = 4) -> str:
    if not xs:
        return "—"
    return f"{mean(xs):.{nd}f} ± {(stdev(xs) if len(xs) > 1 else 0.0):.{nd}f}"


def ranks(values: dict[str, float]) -> dict[str, float]:
    """Average ranks, 1 = highest."""
    order = sorted(values, key=lambda k: -values[k])
    out, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            out[k] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    ra, rb = ranks(a), ranks(b)
    keys = list(a)
    ma, mb = mean(ra[k] for k in keys), mean(rb[k] for k in keys)
    num = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    den = (sum((ra[k] - ma) ** 2 for k in keys) * sum((rb[k] - mb) ** 2 for k in keys)) ** 0.5
    return num / den if den else float("nan")


def build(out_dir: Path) -> str:
    on_server = server_lengths()
    rids = sorted(p.name for p in BASE.iterdir() if p.is_dir() and not p.name.endswith("_ext") and p.name not in EXCLUDED)
    runs = [one_run(r, on_server) for r in rids]
    done = [r for r in runs if r["status"] == "done"]

    fams: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        fams[r["family"]].append(r)
    complete = {f: rs for f, rs in fams.items() if all(r["status"] == "done" for r in rs)}
    partial = {f: rs for f, rs in fams.items() if f not in complete}

    m_ext = {f: mean(r["map50_95"] for r in rs) for f, rs in complete.items()}
    m_base = {f: mean(r["base_map50_95"] for r in rs) for f, rs in complete.items()}
    sd_ext = {f: stdev([r["map50_95"] for r in rs]) if len(rs) > 1 else 0.0 for f, rs in complete.items()}
    order = sorted(complete, key=lambda f: -m_ext[f])
    base_rank = ranks(m_base)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table1_ext_runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        for r in runs:
            w.writerow({k: (f"{r[k]:.5f}" if isinstance(r.get(k), float) else r.get(k, "")) for k in RUN_FIELDS})

    def fam_row(i, f, rs):
        d = [r for r in rs if r["status"] == "done"]
        batches = sorted({r["batch"] for r in d if r.get("batch")}, reverse=True)
        hosts = sorted({r["host"] for r in d})
        gain = [r["map50_95"] - r["base_map50_95"] for r in d]
        return (f"| {i} | `{f}` | {len(d)}/{len(rs)} | {ms([r['map50_95'] for r in d])} | {ms([r['map50'] for r in d], 3)} "
                f"| {ms([r['precision'] for r in d], 3)} | {ms([r['recall'] for r in d], 3)} "
                f"| {ms([r['base_map50_95'] for r in d])} | {mean(gain):+.4f} | {sum(r['improved'] for r in d)}/{len(d)} "
                f"| {mean(r['best_epoch'] for r in d):.0f} | {'/'.join(map(str, batches))} | {', '.join(hosts)} |")

    head = ("| # | Family | Seeds | mAP@50–95 | mAP@50 | Precision | Recall | ep25 mAP@50–95 | Δ vs ep25 | Improved "
            "| Best epoch | Batch | Host |\n|---:|---|:-:|---|---|---|---|---|---:|:-:|---:|---|---|")
    L = [f"# Table 1 — VIS benchmark at patience {PATIENCE} (ep25 grid + extensions)", ""]
    pend = [r["run_id"] for r in runs if r["status"] != "done"]
    if pend:
        L += [f"**PARTIAL: {len(done)} of {len(runs)} runs finished; {len(complete)} of {len(fams)} families complete.** "
              f"Re-run `scripts/bench_ext_table.py` when the queue ends.", ""]
    else:
        L += [f"All {len(runs)} runs finished; {len(fams)} families.", ""]

    L += ["## Ranking (families with every seed finished)", "", head]
    L += [fam_row(i, f, complete[f]) for i, f in enumerate(order, 1)]

    if len(order) >= 2:
        lead = order[0]
        tier = [f for f in order[1:] if m_ext[lead] - m_ext[f] < max(sd_ext[lead], sd_ext[f])]
        top5_base = sorted(complete, key=lambda f: -m_base[f])[:5]
        L += ["", "## Reading", "",
              f"* **Leader `{lead}`** {m_ext[lead]:.4f}; next `{order[1]}` at {m_ext[order[1]]:.4f}, a margin of "
              f"{m_ext[lead] - m_ext[order[1]]:.4f} against seed sds of {sd_ext[lead]:.4f} and {sd_ext[order[1]]:.4f}.",
              f"* **Not separable from the leader** (gap below the larger of the two seed sds, the Phase 1 "
              f"convention; a screen, not a test): {', '.join(f'`{f}`' for f in tier) or 'none'}.",
              f"* **Top-5 spread** {m_ext[order[0]] - m_ext[order[min(4, len(order) - 1)]]:.4f}; median seed sd "
              f"across families {sorted(sd_ext.values())[len(sd_ext) // 2]:.4f}.",
              f"* **Ranking vs ep25:** Spearman ρ = {spearman(m_ext, m_base):.3f} over {len(complete)} families; "
              f"top 5 at ep25 {', '.join(f'`{f}`' for f in top5_base)}; at patience {PATIENCE} "
              f"{', '.join(f'`{f}`' for f in order[:5])}.",
              f"* **Moved ≥ 3 places:** " + (", ".join(f"`{f}` {base_rank[f]:.0f}→{i}" for i, f in enumerate(order, 1)
                                                if abs(base_rank[f] - i) >= 3) or "none") + ".",
              f"* **Runs that beat their ep25 best:** {sum(r['improved'] for r in done)} of {len(done)}."]

    if partial:
        L += ["", "## Families not yet complete (not ranked)", "", head]
        L += [fam_row("—", f, rs) if any(r["status"] == "done" for r in rs) else
              f"| — | `{f}` | 0/{len(rs)} | | | | | | | | | | |" for f, rs in sorted(partial.items())]
        L += ["", "Pending: " + ", ".join(f"`{r['run_id'].replace('vis_bench_', '')}` ({r['status']})"
                                          for r in runs if r["status"] != "done") + "."]

    bad = [r for r in done if r.get("replay") != "OK" or (not r.get("rows_contiguous", True) and "note" not in r)]
    gaps = [r for r in done if r.get("note")]
    L += ["", "## Notes", "",
          "* **Score:** best-epoch row over base + extension, earliest maximum, the checkpoint EarlyStopping keeps. "
          "The epoch is chosen on the same val split it is reported on (as in Phase 1), so absolute values carry a "
          "small selection optimism shared by every family. *Best epoch* counts from base epoch 1.",
          "* **Data:** `data_vis_stride4.yaml`, restored labels (train ledger `8ed69b5974ed`). The val split spans "
          "pohang00–04, pohang01 night included. These are the training-time val logs, not a new evaluation and "
          "not a holdout claim.",
          "* **Recipe deviations, local runs** (authorized 2026-09-27): batch planned per family, the largest of "
          "16/8/4/2 under 90% of the 12 GB card, with nbs=64 kept, so the effective batch is 64 throughout; "
          "loader workers 6 (server 2). Batch and host are per family above, per run in `table1_ext_runs.csv`.",
          f"* **Excluded:** {', '.join(f'`{k}` ({v})' for k, v in EXCLUDED.items())}. `yolov8s` reports seeds 0, 1, 3.",
          "* **Replay / rows:** " + ("every finished run replays its stopper exactly; rows are contiguous except where noted below."
                                     if not bad else "; ".join(f"`{r['run_id']}` replay={r.get('replay')} "
                                                               f"contiguous={r.get('rows_contiguous')}" for r in bad) + ".")]
    L += [f"* `{r['run_id'].replace('vis_bench_', '')}`: {r['note']}." for r in gaps]
    md = "\n".join(L) + "\n"
    (out_dir / "table1_ext.md").write_text(md, encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT), help="writes table1_ext.md and table1_ext_runs.csv here")
    print(build(Path(ap.parse_args().out_dir)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
