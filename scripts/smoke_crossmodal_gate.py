"""Structural checks on the `crossmodal` preset (2026-09-01).

The failure this file exists to prevent is the one recorded in
`docs/veil-veto-and-rule-sweep-2026-09-01.md` §5: a gate feature that failed to
load its data ran to completion and reproduced the OTHER configuration's numbers
exactly, with all seven smoke checks green, because every check exercised the
switch logic and none touched the assembly path where the bug lived. So these
checks are deliberately about ASSEMBLY -- what `load_context` actually built --
and not about the arithmetic of a threshold.

Checks (A-D need only the constants file; E-H load the caches):

  A  the fitted thresholds are the ones the record quotes, and the IR night margin
     is positive -- i.e. the held-out night run really does sit above a bound
     fitted on daylight fit runs alone.
  B  `preset="adopted"` leaves the new fields empty and the old rule in place.
  C  `preset="crossmodal"` populates BOTH axes; a half-loaded gate is refused.
  D  an unknown preset raises rather than silently falling back.
  E  the crossmodal veto rate per cell is exactly what the record states:
     day cells 0%, night cells 100%, both fog cells 100%.
  F  the two axes are INDEPENDENT in the way claimed -- gini fires only on fog,
     ir_night only on night -- so the OR is not one axis doing all the work.
  G  the crossmodal weights are constant across frames (capability prior alone).
  H  `preset="adopted"` still produces the adopted veto rates, unchanged: the new
     preset must not have moved the old one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

CONST = ROOT / "runs" / "eval" / "structure_constants.json"
# From runs/eval/architecture_v3.md and fit_structure_gate.py's own table.
EXPECT_VETO = {"clean/day": 0.0, "clean/night": 1.0, "fog/day": 1.0, "fog/night": 1.0,
               "lowlight/day": 0.0, "lowlight/night": 1.0, "glare/day": 0.0,
               "glare/night": 1.0}
ADOPTED_VETO = {"clean/day": 0.0, "clean/night": 1.0, "fog/day": 1.0, "fog/night": 1.0,
                "lowlight/day": 1.0, "lowlight/night": 1.0, "glare/day": 0.0,
                "glare/night": 1.0}


def main() -> int:
    fails = []

    # ---- A ---------------------------------------------------------------
    if not CONST.is_file():
        print(f"[smoke] MISSING {CONST} — run scripts/fit_structure_gate.py")
        return 1
    c = json.loads(CONST.read_text(encoding="utf-8"))["axes"]
    for key, want, above in (("grad_gini", 0.4826, False), ("ir_p05", 34.0, True)):
        got = c[key]["threshold"]
        if abs(got - want) > 5e-4 or c[key]["fires_above"] != above:
            fails.append(f"A {key}: threshold {got} / above={c[key]['fires_above']}, "
                         f"record says {want} / above={above}")
    if c["ir_p05"]["margin"] <= 0:
        fails.append(f"A ir_p05 margin {c['ir_p05']['margin']} <= 0 — the held-out "
                     f"night run is NOT separated by a bound fitted on daylight")
    print(f"[smoke] A thresholds gini<{c['grad_gini']['threshold']:.4f} "
          f"ir_p05>{c['ir_p05']['threshold']:.1f}, night margin "
          f"{c['ir_p05']['margin']:+.0f}  {'FAIL' if fails else 'OK'}")

    # ---- D ---------------------------------------------------------------
    try:
        load_context(preset="nonsense", verbose=False)
        fails.append("D unknown preset did not raise")
    except ValueError:
        pass
    except SystemExit:
        pass
    print("[smoke] D unknown preset raises  OK")

    # ---- B, C, E, F, G, H ------------------------------------------------
    ad = load_context(verbose=False)
    if ad.veto_rule != "photometric+veil" or ad.gini_by_cond or ad.ir_night is not None:
        fails.append("B preset='adopted' picked up crossmodal state")
    if ad.single_passthrough:
        fails.append("B preset='adopted' turned on single_passthrough")
    print(f"[smoke] B adopted preset clean: rule={ad.veto_rule} "
          f"passthrough={ad.single_passthrough}  "
          f"{'FAIL' if any(f.startswith('B') for f in fails) else 'OK'}")

    cm = load_context(preset="crossmodal", verbose=False)
    if not cm.gini_by_cond or cm.ir_night is None:
        fails.append("C crossmodal preset half-loaded")
    if cm.c_vis.mu_d < 1e8 or cm.c_vis.lam != 0.0:
        fails.append("C crossmodal preset did not switch to capability-only weights")
    print(f"[smoke] C crossmodal loaded: {len(cm.gini_by_cond)} gini conditions, "
          f"ir_night on {cm.ir_night.mean():.1%} of frames  "
          f"{'FAIL' if any(f.startswith('C') for f in fails) else 'OK'}")

    night = np.isin(cm.runs, NIGHT_RUNS)
    splits = {"day": ~night, "night": night}
    gini_thr = float(c["grad_gini"]["threshold"])
    for cond in cm.conditions:
        g = cm.gini_by_cond[cond] < gini_thr
        for s, sel in splits.items():
            cell = f"{cond}/{s}"
            got = float((g | cm.ir_night)[sel].mean())
            if abs(got - EXPECT_VETO[cell]) > 1e-9:
                fails.append(f"E {cell}: veto {got:.1%}, record says {EXPECT_VETO[cell]:.0%}")
            # F: each axis fires only where it is supposed to.
            gr, ir = float(g[sel].mean()), float(cm.ir_night[sel].mean())
            want_g = 1.0 if cond == "fog" else 0.0
            want_i = 1.0 if s == "night" else 0.0
            if abs(gr - want_g) > 1e-9 or abs(ir - want_i) > 1e-9:
                fails.append(f"F {cell}: gini {gr:.1%} (want {want_g:.0%}), "
                             f"ir_night {ir:.1%} (want {want_i:.0%})")
    print(f"[smoke] E veto rates match the record on all 8 cells  "
          f"{'FAIL' if any(f.startswith('E') for f in fails) else 'OK'}")
    print(f"[smoke] F gini fires on fog only, ir_night on night only  "
          f"{'FAIL' if any(f.startswith('F') for f in fails) else 'OK'}")

    res = run_systems(cm, "clean")
    w = np.asarray(res["w_vis_gated"])
    if w.std() > 1e-12:
        fails.append(f"G crossmodal w_vis is not constant (std {w.std():.2e}) — the "
                     f"soft gate is still contributing")
    print(f"[smoke] G crossmodal w_vis constant at {w[0]:.4f}  "
          f"{'FAIL' if any(f.startswith('G') for f in fails) else 'OK'}")

    for cond in ad.conditions:
        r = run_systems(ad, cond)
        v = np.asarray(r["veto_vis"])
        for s, sel in splits.items():
            cell = f"{cond}/{s}"
            got = float(v[sel].mean())
            if abs(got - ADOPTED_VETO[cell]) > 1e-9:
                fails.append(f"H adopted {cell}: veto {got:.1%}, "
                             f"was {ADOPTED_VETO[cell]:.0%}")
    print(f"[smoke] H adopted preset veto rates unmoved  "
          f"{'FAIL' if any(f.startswith('H') for f in fails) else 'OK'}")

    for f in fails:
        print(f"[smoke] FAIL {f}")
    if fails:
        print("\nCROSSMODAL GATE SMOKE FAILED")
        return 1
    print("\nCROSSMODAL GATE SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
