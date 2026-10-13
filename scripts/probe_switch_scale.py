"""Descriptive, draw 941: does switching fusion outputs on RANDOM frames lower AP below both pure policies?

Per frame, output VIS alone (raw scores) on a random fraction f of frames and the union elsewhere.
A frame-level choice made at random carries no information, so any loss below the f-weighted
interpolation of the two pure policies comes from pooling two score scales into one ranking.
`tiered` keeps one scale: the same random frames emit VIS first and IR as the tail (lex).
"""
import dataclasses
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path[:0] = [r"A:\Uncertain\src", r"A:\Uncertain\scripts"]
import numpy as np  # noqa: E402

import gate_zoo as gz  # noqa: E402

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def job(spec):
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, run_systems
    from v2_sensitivity import ship_ap
    ic, k = spec
    ctx = gz.load(k, 941, ic, conds=("clean",))
    night = np.isin(ctx.runs, NIGHT_RUNS)
    day = np.flatnonzero(~night)
    irv = gz.ir_in_vis(ctx)
    vrec = ctx.vis_by_cond["clean"]
    un = run_systems(dataclasses.replace(ctx, veto=None), "clean", compute_metrics=False)["fused_gated"]
    # how the union rescales an unpartnered VIS box: ratio of union score to raw VIS score
    ratios = []
    for u, v in zip(un[:300], vrec[:300]):
        uc, vc = np.sort(np.asarray(u["conf"], float))[::-1][:5], np.sort(np.asarray(v["conf"], float))[::-1][:5]
        n = min(len(uc), len(vc))
        if n:
            ratios.append(np.median(uc[:n] / np.maximum(vc[:n], 1e-9)))
    pv, pu = frame_parts(vrec, ctx.gts), frame_parts(un, ctx.gts)
    out = {"ratio_median": float(np.median(ratios)) if ratios else float("nan")}
    rng = np.random.default_rng(1000 + k)
    u01 = rng.random(len(vrec))
    for f in FRACS:
        sw = u01 < f
        mixed = [pv[i] if sw[i] else pu[i] for i in range(len(vrec))]
        lexm = frame_parts(gz.tiered([vrec[i] if sw[i] else un[i] for i in range(len(vrec))],
                                     [irv[i] if sw[i] else None for i in range(len(vrec))]), ctx.gts)
        out[f] = {"switch": ship_ap(mixed, day), "switch_tiered": ship_ap(lexm, day)}
    out["vis"], out["union"] = ship_ap(pv, day), ship_ap(pu, day)
    return spec, out


if __name__ == "__main__":
    specs = [(ic, k) for ic in (None, "fog_s3") for k in gz.SEEDS]
    res = {}
    with ProcessPoolExecutor(5) as ex:
        for spec, r in ex.map(job, specs):
            res[spec] = r
    for ic in (None, "fog_s3"):
        m = lambda key, sub=None: np.mean([res[(ic, k)][key][sub] if sub else res[(ic, k)][key] for k in gz.SEEDS])  # noqa: E731
        print(f"\n== VIS clean, IR {ic or 'clean'}, day: VIS {m('vis'):.4f}, union {m('union'):.4f}, "
              f"union/VIS score ratio (top-5 median) {m('ratio_median'):.3f}")
        print("frac VIS-only   switch   linear-interp   switch_tiered")
        for f in FRACS:
            lin = f * m("vis") + (1 - f) * m("union")
            print(f"{f:4.2f}           {m(f, 'switch'):.4f}   {lin:.4f}          {m(f, 'switch_tiered'):.4f}")
