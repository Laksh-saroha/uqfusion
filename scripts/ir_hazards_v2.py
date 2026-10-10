"""Table 6 under corruption v2: is the IR night switch safe when IR itself is corrupted?

`probe_ir_night_robustness.py` (raw rule) and `probe_ir_selfcheck.py` + `probe_both_degraded.py`
(hardened switch) measured this with the v1 corruptions, which put visible-light filters on
the thermal frames: a white-disc fog layer plus a 21-px blur, a lens flare, RGB rain streaks,
a brightness clip, per-channel colour noise. v2 (`docs/eval/corruption_v2/README.md`) models
only what a thermal sensor can suffer: Koschmieder fog on the frame's metric depth with a
weaker LWIR extinction (assumption, `IR_BETA_RATIO`) and the IR sky as airlight, temporal
+ column fixed-pattern noise, and blur; it refuses lowlight, glare and rain for IR.

Switch-level, no detector: the night switch is decided on frame statistics before any box is
merged. Per IR arm (clean + {blur, fog, noise} x severity 1-3, corruption seed 7 as the v1
probes, all 2,232 paired frames, stride 1) this reports

1. the RAW rule `ir_p05 > thr` (Table 6): false night on day frames, missed night on
   pohang01, and whether a threshold refitted on that arm's own fit-run day frames could
   still separate day from night;
2. the HARDENED IR vote of the shipped system, `ir_p05 > thr AND d2 <= bound_switch`
   (multivariate IR health, authority bound), and how often the health score flags the arm;
3. the shipped `crossmodal26m` veto replayed per frame from `ctx.py`
   (`night & (dark | veil)` plus the weak-IR fallback), against VIS conditions clean,
   lowlight, glare (VIS the better stream on day frames, so any day veto is harmful) and fog,
   with the VIS statistics of `runs/derived_m_v2` (v2 VIS corruptions, seed 1). The IR side of
   the replay is checked against `load_context`'s own arrays on clean IR first.

    py -3.13 scripts/ir_hazards_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uqfusion.eval.parallel_frames import corrupted_frames, default_workers   # noqa: E402

NIGHT_RUN = "pohang01"
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
KINDS = ("blur", "fog", "noise")
NOT_MODELLED = ("glare", "lowlight", "rain")
VIS_CONDS = ("clean", "lowlight", "glare", "fog")
VIS_BETTER = {"clean": True, "lowlight": True, "glare": True, "fog": False}
LIST = ROOT / "runs/derived/paired_val_ir.txt"
V1 = ROOT / "runs/eval"


def arm_stats(images, kind, sev, seed, workers) -> dict[str, np.ndarray]:
    spec = {"corrupt": kind, "severity": sev, "corrupt_seed": seed, "corrupt_version": "v2",
            "modality": "ir", "stats": "ir"}
    rows = [st[1] for _i, _im, st in corrupted_frames(images, spec, workers=workers)]
    return {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0] if k not in ("image_path", "run")}


def ir_d2(a: dict, hm: dict) -> np.ndarray:
    X = np.stack([a[k] for k in hm["keys"]], 1).astype(float)
    for j, k in enumerate(hm["keys"]):
        if k in hm["log1p_keys"]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = (X - np.asarray(hm["mean"])) / np.asarray(hm["std"])
    return np.einsum("ij,jk,ik->i", Z, np.asarray(hm["precision"]), Z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="docs/eval/corruption_v2/ir_hazards_v2.md")
    args = ap.parse_args()
    from uqfusion.eval.ctx import load_context

    images = [ln.strip() for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    runs = np.asarray([Path(p).parent.name for p in images])
    night, fit = runs == NIGHT_RUN, np.isin(runs, FIT_RUNS)
    day = ~night
    workers = args.workers or default_workers()

    ctx = load_context(preset="crossmodal26m", cache_dir="runs/cache_m_v2",
                       bright_dir="runs/derived_m_v2/brightness", structure_dir="runs/derived_m_v2/structure",
                       conditions=VIS_CONDS, verbose=False)
    if [str(Path(r["image_path"])) for r in ctx.ir_clean] != [str(Path(p)) for p in images]:
        raise SystemExit("IR list order differs from the context's")
    c = ctx.struct_const["axes"]
    thr, hm = float(c["ir_p05"]["threshold"]), c["ir_health"]
    bound, bound_sw = float(hm["bound"]), float(hm.get("bound_switch", hm["bound"]))
    lov_thr, gini_thr = float(c["lap_over_var"]["threshold"]), float(c["grad_gini"]["threshold"])
    mu_b = float(ctx.c_vis.mu_b)

    t0 = time.time()
    arms = {"clean": arm_stats(images, None, 0, args.seed, workers)}
    # The replay's IR side must be the context's IR side on clean IR, bit for bit.
    d2c = ir_d2(arms["clean"], hm)
    if not (np.array_equal(arms["clean"]["p05"] > thr, ctx.ir_night_raw)
            and np.array_equal(d2c <= bound_sw, ctx.ir_ok)):
        raise SystemExit("clean-IR replay disagrees with load_context (ir_night_raw / ir_ok)")
    print(f"[irhaz] clean done, replay matches load_context ({time.time() - t0:.0f}s)", flush=True)
    for kind in KINDS:
        for s in (1, 2, 3):
            arms[f"{kind} s{s}"] = arm_stats(images, kind, s, args.seed, workers)
            print(f"[irhaz] {kind} s{s} done ({time.time() - t0:.0f}s)", flush=True)

    vis = {}
    for vc in VIS_CONDS:
        b = ctx.bright_by_cond[vc]
        veil = ctx.gini_by_cond[vc] < gini_thr
        lov = ctx.struct_lov_by_cond.get(vc)
        vis[vc] = {"dark": b < mu_b, "veil": veil,
                   "concentrated": (lov > lov_thr) if lov is not None else np.zeros(len(b), bool)}

    rows, veto_rows = [], []
    for name, a in arms.items():
        p05, d2 = a["p05"], ir_d2(a, hm)
        raw, ok = p05 > thr, d2 <= bound_sw
        hard = raw & ok
        refit = float(p05[fit & day].max())
        rows.append({"arm": name, "day_p05_med": float(np.median(p05[day])), "night_p05_med": float(np.median(p05[night])),
                     "false_night_raw": float(raw[day].mean()), "missed_night_raw": float((~raw[night]).mean()),
                     "separable_after_refit": bool(p05[night].min() > refit),
                     "false_night_hardened": float(hard[day].mean()), "missed_night_hardened": float((~hard[night]).mean()),
                     "health_flag_day": float((~ok[day]).mean()), "health_flag_night": float((~ok[night]).mean()),
                     "merge_flag_day": float((d2[day] > bound).mean())})
        for vc in VIS_CONDS:
            v = vis[vc]
            vv = hard & (v["dark"] | v["veil"]) if ctx.veil_requires_night else (v["veil"] | (hard & v["dark"]))
            if ctx.night_weak_fallback:
                vv = vv | (raw & ~ok & (v["concentrated"] | (v["dark"] & v["veil"])))
            veto_rows.append({"ir": name, "vis": vc, "veto_day": float(vv[day].mean()),
                              "harmful": VIS_BETTER[vc], "veto_night": float(vv[night].mean())})

    v1 = {}
    for kind in ("blur", "fog", "glare", "lowlight", "noise", "rain"):
        p = V1 / f"ir_night_robustness_{kind}.json"
        if p.is_file():
            for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                if r["severity"]:
                    v1[f"{r['corruption']} s{r['severity']}"] = r["false_night"]

    L = ["# IR night switch under IR corruption, corruption v2 (Table 6)", "",
         f"All {len(images)} paired IR frames ({int(day.sum())} day / {int(night.sum())} night, pohang01), "
         f"stride 1, corruption seed {args.seed} (as the v1 probes), v2 IR corruptions on content rows. "
         f"Shipped constants: `ir_p05` > {thr:.1f}; IR health authority bound {bound_sw:.1f}, merge bound "
         f"{bound:.1f}; VIS `mu_b` {mu_b}; `grad_gini` < {gini_thr:.4f}; `lap_over_var` > {lov_thr:.3f}. "
         "v2 does not model IR " + ", ".join(NOT_MODELLED) + " (a thermal sensor does not see a "
         "visible-light flare, an exposure cut or RGB rain streaks), so those v1 rows have no v2 counterpart.", "",
         "## 1. Raw rule and hardened vote", "",
         "| IR arm | day p05 med | night p05 med | **false night, raw** | v1 raw | missed night, raw | "
         "separable after refit | **false night, hardened** | missed night, hardened | health flag day / night |",
         "|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['day_p05_med']:.1f} | {r['night_p05_med']:.1f} | **{r['false_night_raw']:.1%}** | "
                 f"{('%.1f%%' % (100 * v1[r['arm']])) if r['arm'] in v1 else '—'} | {r['missed_night_raw']:.1%} | "
                 f"{'yes' if r['separable_after_refit'] else 'no'} | **{r['false_night_hardened']:.1%}** | "
                 f"{r['missed_night_hardened']:.1%} | {r['health_flag_day']:.1%} / {r['health_flag_night']:.1%} |")
    worst_raw = max(rows[1:], key=lambda r: r["false_night_raw"])
    worst_h = max(rows[1:], key=lambda r: r["false_night_hardened"])
    harm = [r for r in veto_rows if r["harmful"] and r["ir"] != "clean"]
    worst_v = max(harm, key=lambda r: r["veto_day"])
    L += ["", f"Worst raw false night: **{worst_raw['false_night_raw']:.1%}** ({worst_raw['arm']}). "
          f"Worst hardened false night: **{worst_h['false_night_hardened']:.1%}** ({worst_h['arm']}).", "",
          "## 2. The shipped veto, replayed (day frames; VIS the better stream except fog)", "",
          "`crossmodal26m`: veto = IR night (hardened) AND (VIS dark OR veil), OR the weak-IR fallback "
          "(raw IR night, IR disarmed, VIS concentrated-highlight or dark-and-veiled). A day veto on "
          "clean, lowlight or glare VIS throws away the better stream.", "",
          "| IR arm | VIS clean | VIS lowlight | VIS glare | VIS fog (veto correct) |", "|---|---:|---:|---:|---:|"]
    for name in arms:
        cells = {r["vis"]: r["veto_day"] for r in veto_rows if r["ir"] == name}
        L.append(f"| {name} | " + " | ".join(f"{cells[vc]:.1%}" for vc in VIS_CONDS) + " |")
    L += ["", f"Worst harmful day veto over the corrupted IR arms: **{worst_v['veto_day']:.1%}** "
          f"(IR {worst_v['ir']} + VIS {worst_v['vis']}).", "",
          "The fallback and both IR bounds were fitted with data that includes pohang01, the only night "
          "run (§4.2, §9), so the night columns are in-sample."]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"corrupt_code": __import__("uqfusion.eval.corruptions_v2", fromlist=["CODE_REV"]).CODE_REV,
         "seed": args.seed, "n": len(images), "thr": thr, "bound_switch": bound_sw, "bound": bound, "mu_b": mu_b,
         "rows": rows, "veto_rows": veto_rows, "v1_false_night_raw": v1, "elapsed_s": time.time() - t0}, indent=1),
        encoding="utf-8")
    print(f"[irhaz] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
