"""R-A5: which adopted decisions survive the corrected evaluator and wider intervals?

The deliverable the rest of the backlog waits on. Two things changed underneath every
number this project has published:

* **R-A1** measured the AP-convention gap. On deltas it is at most **0.00029**, because
  both arms of a comparison share the convention. Small, but not zero, and it is a
  *systematic* offset on the point estimate rather than noise.
* **R-A3** measured the interval defect. The iid frame bootstrap on 10 Hz video is
  **~1.95x too narrow**, as a lower bound. This is the one with teeth.

What this script does, and what it does not
-------------------------------------------
It **re-adjudicates** recorded decisions: it takes each decision's stored delta and
interval, widens the interval by the measured R-A3 factor, adds the R-A1 convention
bound as a systematic band on the point estimate, and reports whether the decision still
holds. That is not the same as a **re-score**, which would re-run each decision's own
bootstrap with `blockboot` over its own cached predictions.

The difference matters and is not hidden: the 1.95x factor was measured on VIS UQ-arm
mAP deltas over `paired_val_vis.txt`. Transporting it to fusion cells assumes the
dependence structure is comparable -- same frames, same four runs, same 10 Hz cadence,
so it is a reasonable transport, but it is an assumption and every row here is labelled
as such. A full re-score is mechanical now that `blockboot` exists and is the honest
next step for any row this table puts in doubt.

Usage:
    python scripts/change_impact_table.py --out runs/eval/change_impact.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import ROOT, fmt, md_table, write_md  # noqa: E402

# R-A3 median se inflation at the largest defensible block (L=20). A LOWER bound: the
# curve was still rising there and this dataset's shortest run prevents measuring where
# it levels off. runs/eval/interval_block_sensitivity_v3.md
INFLATION = 1.95

# R-A1 worst delta disagreement between the local AP and official COCO AP across every
# arm pair and subset. runs/eval/ap_convention_parity.md
CONVENTION = 0.00028501


def readjudicate(delta: float, lo: float, hi: float) -> dict:
    """Widen a recorded interval by the measured factors and re-read the verdict."""
    was = not (lo <= 0.0 <= hi)
    ilo = delta - INFLATION * (delta - lo)
    ihi = delta + INFLATION * (hi - delta)
    # the convention gap is a systematic offset on the point estimate, not extra noise,
    # so it widens the band rather than being added in quadrature.
    clo, chi = ilo - CONVENTION, ihi + CONVENTION
    now = not (clo <= 0.0 <= chi)
    if was and now:
        verdict = "SURVIVES"
    elif was and not now:
        verdict = "**INDETERMINATE**"
    elif not was and now:
        verdict = "**INVERTS**"          # would be alarming; recorded if it happens
    else:
        verdict = "already spanned zero"
    return {"was_significant": was, "ci_inflated": (ilo, ihi),
            "ci_full": (clo, chi), "still_significant": now, "verdict": verdict}


def rows_from_final_system(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in d.get("vs_visible", []):
        for pre, name in (("", "macro"), ("ship_", "ship")):
            ci = r.get(f"{pre}ci")
            if ci is None:
                continue
            out.append({"source": label, "family": "gated vs visible_only",
                        "item": f"{r['condition']} / {name}",
                        "delta": r[f"{pre}delta"], "lo": ci[0], "hi": ci[1]})
    for r in d.get("ablation", []):
        for pre, name in (("", "macro"), ("ship_", "ship")):
            ci = r.get(f"{pre}ci")
            if ci is None:
                continue
            out.append({"source": label, "family": f"D27 ablation: {r['variant']}",
                        "item": f"{r['condition']}/{r['split']} / {name}",
                        "delta": r[f"{pre}delta"], "lo": ci[0], "hi": ci[1]})
    return out


def rows_from_veil(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for key, split in (("boot_day", "day"), ("boot_night", "night")):
        b = d.get(key)
        if not b or b.get("se", 0) == 0:
            continue
        out.append({"source": "veil_veto_repair_26m", "family": "veil veto repair",
                    "item": split, "delta": b["delta"],
                    "lo": b["ci_lo"], "hi": b["ci_hi"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/change_impact.md")
    args = ap.parse_args()

    ev = ROOT / "runs/eval"
    items: list[dict] = []
    items += rows_from_final_system(ev / "final_system_crossmodal.json",
                                    "final_system_crossmodal (shipped preset)")
    items += rows_from_final_system(ev / "final_system.json",
                                    "final_system (D27, 2026-08-20)")
    items += rows_from_veil(ev / "veil_veto_repair_26m.json")

    for it in items:
        it.update(readjudicate(it["delta"], it["lo"], it["hi"]))

    was_sig = [i for i in items if i["was_significant"]]
    lost = [i for i in was_sig if not i["still_significant"]]

    rows = [[i["source"].split(" (")[0], i["family"], i["item"], fmt(i["delta"], 6),
             f"[{fmt(i['lo'], 6)}, {fmt(i['hi'], 6)}]",
             f"[{fmt(i['ci_full'][0], 6)}, {fmt(i['ci_full'][1], 6)}]", i["verdict"]]
            for i in items if i["was_significant"]]
    rows.sort(key=lambda r: (r[-1] != "**INDETERMINATE**", r[1], r[2]))

    # The source column is load-bearing, not decoration: the same cell name appears
    # under two different systems (the shipped crossmodal preset and the 2026-08-20 D27
    # system) with different magnitudes, and without it two rows look like a duplicate.
    lost_rows = [[i["source"].split(" (")[0], i["family"], i["item"], fmt(i["delta"], 6),
                  f"[{fmt(i['lo'], 6)}, {fmt(i['hi'], 6)}]",
                  f"[{fmt(i['ci_full'][0], 6)}, {fmt(i['ci_full'][1], 6)}]"]
                 for i in lost]

    # Per-claim roll-up, so the reader gets "which claims survive" rather than 53 rows.
    fam: dict[str, list[dict]] = {}
    for i in was_sig:
        fam.setdefault(i["family"], []).append(i)
    fam_rows = []
    for f, its in sorted(fam.items()):
        keep = [x for x in its if x["still_significant"]]
        strongest = min((x["delta"] for x in keep), key=abs, default=float("nan"))
        fam_rows.append([
            f, str(len(its)), str(len(keep)), str(len(its) - len(keep)),
            ("**all lost**" if not keep else
             "all hold" if len(keep) == len(its) else
             f"holds on the larger cells (smallest surviving |delta| {abs(strongest):.6f})")])

    secs = [
        "Produced by `scripts/change_impact_table.py` for R-A5 "
        "(`docs/TODO-2026-09-09-architecture-review.md`). Two measurements from earlier "
        "in workstream A are applied to every recorded decision that was defended by an "
        "interval excluding zero.",

        f"**The two corrections.** R-A1: the AP-convention gap survives into a delta at "
        f"most **{CONVENTION:.5f}** "
        f"(`runs/eval/ap_convention_parity.md`), applied here as a systematic band on the "
        f"point estimate. R-A3: the iid frame bootstrap on 10 Hz video is **{INFLATION}× "
        f"too narrow** as a lower bound "
        f"(`runs/eval/interval_block_sensitivity_v3.md`), applied by widening each "
        f"recorded interval about its own delta. The second dominates.",

        f"## Result: {len(lost)} of {len(was_sig)} significant findings become "
        f"indeterminate\n\n"
        + (md_table(["source", "family", "item", "delta", "recorded 95% CI",
                     "corrected band"], lost_rows) if lost_rows else
           "_None — every recorded interval that excluded zero still excludes it._")
        + "\n\nThese are not refuted. They are **no longer supported by the interval "
        "that was used to defend them**, which is a different and weaker statement: the "
        "effect may well be real, but this data and this resampling scheme cannot "
        "establish it at 95%.",

        "## Per-claim roll-up — does the claim survive, or only some of its cells?\n\n"
        + md_table(["claim", "cells defended by an interval", "still supported",
                    "now indeterminate", "reading"], fam_rows)
        + "\n\nA claim can lose cells and still stand: what breaks first is always the "
        "smallest effect, and several of these families were defended across a range of "
        "magnitudes. Read this table before the row-level one.",

        "## Every decision that was defended by a non-zero-spanning interval\n\n"
        + md_table(["source", "family", "item", "delta", "recorded 95% CI",
                    "corrected band", "verdict"], rows),

        "## Decisions this table cannot re-adjudicate\n\n"
        "Three of the five families R-A5 names were not decided by a bootstrap interval, "
        "so widening one says nothing about them. They need their own treatment:\n\n"
        "* **The inherited-constants reprice** (`runs/eval/reprice_constants.md`) uses a "
        "*worse-somewhere* rule over per-cell deltas with a margin built from a draw SD "
        "and a bootstrap SD. The bootstrap half of that margin is understated by the "
        "R-A3 factor, so the margins are too tight, but the rule is a sign count and not "
        "an interval — re-deciding it means re-running it, not rescaling it. Its three "
        "verdicts (`cap_ir_scale`, `iou_thr`, veil repair all STAND) are **not** "
        "revisited here.\n"
        "* **The soft-NMS reject** (`runs/eval/snms_gate_draw_avg.md`) is a draw-averaged "
        "sign count decided at −1.03e-5 on 4 draws. R-A3 does not touch it: its noise is "
        "corruption-draw noise, not frame-resampling noise. It was already unsound for "
        "the reason `project-gate-magnitude-floor` records — a measured margin with no "
        "absolute floor degenerates to a sign test on ~1e-5.\n"
        "* **The crossmodal gate cells** are partly covered above through "
        "`final_system_crossmodal`, but the tuning sweeps "
        "(`runs/eval/crossmodal_tuning_*.json`) selected constants by comparing cell "
        "means without intervals at all. Selection under a too-narrow interval is a "
        "different failure from adoption under one, and is R-B2's territory.",

        "## Re-adjudicated, not re-scored — the limit of this table\n\n"
        f"Every row here takes a **recorded** delta and interval and widens it. The "
        f"{INFLATION}× factor was measured on VIS UQ-arm mAP deltas over "
        "`paired_val_vis.txt`; transporting it to fusion cells assumes a comparable "
        "dependence structure. That is reasonable — same frames, same four runs, same "
        "10 Hz cadence — but it is an **assumption, not a measurement**, and a full "
        "re-score would re-run each decision's own bootstrap through "
        "`blockboot.block_bootstrap_delta` on its own caches.\n\n"
        "That is now mechanical rather than hard, and it is the right next step for any "
        "row this table puts in doubt. It is also the only way to settle rows where the "
        "corrected band lands close to zero, since the transported factor is a lower "
        "bound and the true inflation for those cells is unknown.",
    ]
    out = write_md(args.out, "Change-impact table — adopted decisions under the corrected metrics", secs)
    Path(str(out).replace(".md", ".json")).write_text(json.dumps(
        {"inflation": INFLATION, "convention": CONVENTION,
         "n_significant": len(was_sig), "n_lost": len(lost),
         "items": items}, indent=1, default=float), encoding="utf-8")
    print(f"[summary] {len(lost)} of {len(was_sig)} significant findings become indeterminate")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
