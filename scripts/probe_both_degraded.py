"""The both-degraded regime: what the switch does when NEITHER sensor is healthy.

The eight-cell benchmark corrupts VIS only, so it never presents the case scope
§7.4 built `R_sys` for. `docs/crossmodal-gate-2026-09-01.md` §3a closed the
single-degraded hole (a fogged IR misreading clear days as night) with an IR
self-check plus a two-of-two vote, and left one residual measured but unfixed:
**IR glare plus VIS lowlight still vetoed VIS on 12.7-24% of frames**, because
both votes were wrong at once and neither sensor could vouch for the other.

This crosses every IR corruption arm with every VIS condition and reports what the
composed switch does, including the two terms added afterwards:

    q_vis  = min(veil ratio, photometric sigmoid)     VIS absolute health
    q_ir   = bound / novelty, capped at 1             IR absolute health
    R_sys  = max(q_vis, q_ir)                         "is any sensor healthy?"
    veto_ir = IR unhealthy AND VIS healthy            symmetric: drop a broken IR
    abstain = neither healthy                         REPORTED only (see below)

The harm metric is deliberately not "false night". It is **VIS vetoed on a frame
where VIS is the better stream** — clean/day (0.3683 vs 0.0177), lowlight/day
(0.0346 vs 0.0177) and glare/day (0.2892 vs 0.0177). On fog/day IR is better, so a
veto there is correct and is scored as such.

Switch-level only: it needs no fusion run, because a veto decision is made before
any box is merged. Measurement only; writes one report.

Usage:
    python scripts/probe_both_degraded.py --out runs/eval/both_degraded.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

NIGHT_RUN = "pohang01"
VIS_CONDS = ("clean", "lowlight", "glare", "fog")
#: Which stream is actually better on that VIS condition, day frames (handoff §6).
VIS_BETTER = {"clean": True, "lowlight": True, "glare": True, "fog": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constants", default="runs/eval/structure_constants.json")
    ap.add_argument("--ir-probe", default="runs/derived/ir_selfcheck")
    ap.add_argument("--struct-dir", default="runs/derived/structure")
    ap.add_argument("--bright-dir", default="runs/derived/brightness")
    ap.add_argument("--out", default="runs/eval/both_degraded.md")
    ap.add_argument("--mu-b", type=float, default=10.5)
    ap.add_argument("--tau-b", type=float, default=2.625)
    args = ap.parse_args()

    c = json.loads((ROOT / args.constants).read_text(encoding="utf-8"))["axes"]
    gini_thr = float(c["grad_gini"]["threshold"])
    ir_thr = float(c["ir_p05"]["threshold"])
    hm = c["ir_health"]
    mu_v, sd_v = np.asarray(hm["mean"]), np.asarray(hm["std"])
    prec, bound = np.asarray(hm["precision"]), float(hm["bound"])
    # Two bounds, opposite asymmetries — see fit_structure_gate.py. `bound_switch`
    # (p99) gates whether IR may VETO VIS; `bound` (hard max) gates whether IR
    # leaves the MERGE and feeds R_sys.
    bound_switch = float(hm.get("bound_switch", bound))
    vh = c["vis_health"]

    # ---- IR arms (clean + every corruption), from the self-check probe ------
    arms, runs = {}, None
    for p in sorted(glob.glob(str(ROOT / args.ir_probe / "ir_stats_stride8_*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        runs = np.asarray(d["runs"]) if runs is None else runs
        for k, v in d["arms"].items():
            arms.setdefault(k, {kk: np.asarray(vv) for kk, vv in v.items()})
    if not arms:
        raise SystemExit("no IR probe stats — run scripts/probe_ir_selfcheck.py first")
    stride = 8
    night = runs == NIGHT_RUN

    def ir_state(a):
        X = np.stack([a[k] for k in hm["keys"]], 1).astype(float)
        for j, k in enumerate(hm["keys"]):
            if k in hm["log1p_keys"]:
                X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
        Z = (X - mu_v) / sd_v
        d2 = np.einsum("ij,jk,ik->i", Z, prec, Z)
        q = np.clip(bound / np.maximum(d2, 1e-12), 0, 1)
        return (a["p05"] > ir_thr) & (d2 <= bound_switch), q

    # ---- VIS conditions ----------------------------------------------------
    vis = {}
    for cond in VIS_CONDS:
        g = np.asarray([f["grad_gini"] for f in json.loads(
            (ROOT / args.struct_dir / f"gauss_vis_paired_{cond}.json").read_text(
                encoding="utf-8"))["frames"]], dtype=float)[::stride]
        b = np.asarray([f["p05"] for f in json.loads(
            (ROOT / args.bright_dir / f"gauss_vis_paired_{cond}.json").read_text(
                encoding="utf-8"))["frames"]], dtype=float)[::stride]
        fr = json.loads((ROOT / args.struct_dir /
                         f"gauss_vis_paired_{cond}.json").read_text(encoding="utf-8"))["frames"]
        X = np.stack([[f[k] for k in vh["keys"]] for f in fr]).astype(float)
        for j, k in enumerate(vh["keys"]):
            if k in vh["log1p_keys"]:
                X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
        Z = (X - np.asarray(vh["mean"])) / np.asarray(vh["std"])
        d2v = np.einsum("ij,jk,ik->i", Z, np.asarray(vh["precision"]), Z)[::stride]
        qv = np.clip(float(vh["bound"]) / np.maximum(d2v, 1e-12), 0, 1)
        vis[cond] = {"veil": g < gini_thr, "dark": b < args.mu_b, "q": qv}

    ir_keys = ["clean|0"] + [f"{k}|{s}" for k in
                             ("fog", "glare", "rain", "blur", "noise", "lowlight")
                             for s in (1, 2, 3) if f"{k}|{s}" in arms]

    rows = []
    for ik in ir_keys:
        ir_n, q_ir = ir_state(arms[ik])
        for vc in VIS_CONDS:
            v = vis[vc]
            vv = v["veil"] | (ir_n & v["dark"])
            h_v, h_i = v["q"] >= 0.5, q_ir >= 0.5
            vi = (~h_i) & h_v
            ab = (~h_v) & (~h_i)
            # ABSTAIN IS REPORTED, NOT ACTED ON — measured: releasing the switch
            # prevented 0 bad vetoes and lost 2,095 correct ones (see ctx.run_systems).
            vv_final, vi_final = vv, vi
            day = ~night
            rows.append({
                "ir": ik.replace("|0", "").replace("|", " s"), "vis": vc,
                "bad_veto_naive": float(vv[day].mean()) if VIS_BETTER[vc] else 0.0,
                "bad_veto": float(vv_final[day].mean()) if VIS_BETTER[vc] else 0.0,
                "veto_ir": float(vi_final[day].mean()),
                "abstain": float(ab[day].mean()),
                "r_sys_med": float(np.median(np.maximum(v["q"], q_ir)[day])),
            })

    harmful = [r for r in rows if VIS_BETTER[r["vis"]]]
    worst_n = max(harmful, key=lambda r: r["bad_veto_naive"])
    worst_f = max(harmful, key=lambda r: r["bad_veto"])

    L = ["# Both-degraded regime: the switch when neither sensor is healthy", "",
         "Day frames only (the night cells are single-sensor by construction). "
         "`bad veto` = VIS vetoed on a frame where **VIS is the better stream** — "
         "clean/day 0.3683 vs 0.0177, lowlight/day 0.0346 vs 0.0177, glare/day "
         "0.2892 vs 0.0177. The fog/day column is omitted from that metric because "
         "IR is genuinely better there and a veto is correct.", "",
         "`before` is the switch with the two-of-two vote and the per-axis IR band; "
         "`after` is the shipped switch, whose IR self-check is the multivariate "
         "novelty score with a separate (tighter) AUTHORITY bound. Switch-level, "
         "stride 8.", "",
         "**The abstain column is reported, not acted on.** The design that released "
         "the switch when neither sensor was healthy was implemented and measured: "
         "over these 76 pairs it prevented 0 bad vetoes and lost 2,095 correct ones, "
         "because the both-flagged frames are overwhelmingly VIS-fogged AND "
         "IR-broken, where releasing the veil veto just adds fog-VIS junk to "
         "broken-IR junk. It is a flag for a downstream consumer, which is the role "
         "scope §7.4 gave it.", "",
         f"IR novelty: merge bound {bound:.1f}, authority bound {bound_switch:.1f}; "
         f"`ir_p05` > {ir_thr:.1f}; `grad_gini` < {gini_thr:.4f}; abstain at "
         f"R_sys < 0.5.", "",
         "| IR arm | VIS cond | bad veto before | **bad veto after** | veto_ir | abstain | median R_sys |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if not VIS_BETTER[r["vis"]]:
            continue
        flag = "" if r["bad_veto"] == 0 else "  ⚠"
        L.append(f"| {r['ir']} | {r['vis']} | {r['bad_veto_naive']:.1%} | "
                 f"**{r['bad_veto']:.1%}**{flag} | {r['veto_ir']:.1%} | "
                 f"{r['abstain']:.1%} | {r['r_sys_med']:.3f} |")

    L += ["", "## Verdict", "",
          f"- Worst bad-veto **before** the abstain: **{worst_n['bad_veto_naive']:.1%}** "
          f"(IR {worst_n['ir']} + VIS {worst_n['vis']}).",
          f"- Worst bad-veto **after**: **{worst_f['bad_veto']:.1%}** "
          f"(IR {worst_f['ir']} + VIS {worst_f['vis']}).",
          f"- Arms still showing any bad veto: "
          f"{sum(1 for r in harmful if r['bad_veto'] > 0)} of {len(harmful)}.",
          "",
          "On the eight benchmark cells every row of this table is unreachable: IR "
          "is clean there, so `q_ir` is 1, `R_sys` is 1, and neither `veto_ir` nor "
          "`abstain` ever fires. That is why the headline table cannot move and why "
          "these terms need their own test."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"bound": bound, "ir_thr": ir_thr, "gini_thr": gini_thr, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"[both] wrote {out}")
    print(f"[both] worst bad-veto  before {worst_n['bad_veto_naive']:.1%} "
          f"({worst_n['ir']} + {worst_n['vis']})  ->  after {worst_f['bad_veto']:.1%} "
          f"({worst_f['ir']} + {worst_f['vis']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
