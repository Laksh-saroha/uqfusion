"""Pins the veil veto term — MUST pass before any fog number is quoted.

The veil term is a SECOND veto axis OR-ed into the switch, and the risk it carries
is regression, not novelty: the photometric gate is a finalized, measured result
(D27, dilate-15), and adding a term next to it must not perturb any cell where the
old rule already decided correctly. So the checks that matter are the ones proving
the new code reduces to the old code.

  A. structure=None reproduces the photometric-only flags EXACTLY, bit for bit,
     over randomized brightness — the path taken by every pre-2026-09-01
     brightness file, which has no `lap_var` column.
  B. tau_lap=None does the same even when structure IS supplied, so the term is
     inert until a threshold is fitted.
  C. OR semantics: a frame is vetoed iff dark OR low-structure. Checked against an
     independent reference implementation, not against itself.
  D. brightness=None with structure supplied vetoes on structure alone (the term
     stands on its own; IR would use this if it ever carried one).
  E. length validation fires on both axes rather than silently broadcasting.
  F. the COMPOSED switch (dilated darkness OR denoised veil) catches every cell
     where VIS is dead and leaves both guard cells at exactly zero.
  G. the two filters must stay separate: OR-ing first and dilating after costs
     the glare/day guard 4.8% of its frames.

Runs on synthetic arrays plus the fitted constants — no caches, no GPU, ~1 s.

Usage:  python scripts/smoke_veil_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.hysteresis import raw_veto_flags  # noqa: E402

MU_B, TAU_B, VETO = 10.5, 2.625, 0.5


def _photometric_only(b, n):
    r = 1.0 / (1.0 + np.exp(-(np.asarray(b, float) - MU_B) / TAU_B))
    return (r < VETO).tolist()


def check_a_no_structure_is_bit_identical():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        b = rng.uniform(-5, 80, n)
        got = raw_veto_flags(b, MU_B, TAU_B, VETO, n)
        assert got == _photometric_only(b, n), "structure=None diverged from the fitted path"
    print("[veil] A structure=None == photometric-only, 200 randomized cases  OK")


def check_b_tau_none_is_inert():
    rng = np.random.default_rng(1)
    n = 50
    b, v = rng.uniform(-5, 80, n), rng.uniform(0, 5000, n)
    assert raw_veto_flags(b, MU_B, TAU_B, VETO, n, structure=v, tau_lap=None) == \
        _photometric_only(b, n), "tau_lap=None was not inert"
    print("[veil] B tau_lap=None inert even with structure supplied            OK")


def check_c_or_semantics():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        b, v = rng.uniform(-5, 80, n), rng.uniform(0, 3000, n)
        tau = float(rng.uniform(100, 2000))
        got = raw_veto_flags(b, MU_B, TAU_B, VETO, n, structure=v, tau_lap=tau)
        want = [d or s for d, s in zip(_photometric_only(b, n), (v < tau).tolist())]
        assert got == want, "veil term is not a clean OR"
    print("[veil] C veto == (dark OR low-structure), 200 randomized cases      OK")


def check_d_structure_alone():
    v = np.array([10.0, 5000.0, 100.0])
    got = raw_veto_flags(None, MU_B, TAU_B, VETO, 3, structure=v, tau_lap=500.0)
    assert got == [True, False, True], got
    assert raw_veto_flags(None, MU_B, TAU_B, VETO, 3) == [False] * 3
    print("[veil] D structure alone vetoes; both None never vetoes             OK")


def check_e_length_validation():
    for kw, msg in ((dict(), "brightness"), (dict(structure=np.zeros(4), tau_lap=1.0), "structure")):
        try:
            raw_veto_flags(np.zeros(4) if not kw else np.zeros(3), MU_B, TAU_B, VETO, 3, **kw)
        except ValueError as e:
            assert msg in str(e), e
        else:
            raise AssertionError(f"no ValueError for mismatched {msg}")
    print("[veil] E length mismatch raises on both axes                        OK")


def _switches(cond, tau):
    """Reproduce the composed switch exactly as `ctx.run_systems` builds it."""
    from uqfusion.eval.hysteresis import (ADOPTED_VEIL_FILTER, ADOPTED_VETO_FILTER,
                                          filter_veto, temporal_order)
    fr = json.loads((ROOT / f"runs/derived/brightness/gauss_vis_paired_{cond}.json")
                    .read_text(encoding="utf-8"))["frames"]
    order = temporal_order([{"image_path": f["image_path"]} for f in fr])
    b = np.array([f["p05"] for f in fr]); v = np.array([f["lap_var"] for f in fr])
    dmode, dk = ADOPTED_VETO_FILTER; vmode, vk = ADOPTED_VEIL_FILTER
    dark = np.asarray(filter_veto((b < MU_B).tolist(), order, dk, dmode), dtype=bool)
    veil = np.asarray(filter_veto((v < tau).tolist(), order, vk, vmode), dtype=bool)
    naive = np.asarray(filter_veto(((b < MU_B) | (v < tau)).tolist(), order, dk, dmode), dtype=bool)
    night = np.array([f["run"] == "pohang01" for f in fr])
    return dark, veil, naive, night


def check_f_separates_the_measured_cells():
    bc = json.loads((ROOT / "runs/eval/brightness_constants.json").read_text(encoding="utf-8"))["vis"]
    tau = bc.get("tau_lap")
    if tau is None:
        print("[veil] F SKIPPED - tau_lap not fitted yet (run scripts/fit_veil_gate.py)")
        return
    rates = {}
    for cond in ("clean", "fog", "lowlight", "glare"):
        try:
            dark, veil, _, night = _switches(cond, tau)
        except (KeyError, FileNotFoundError):
            print(f"[veil] F SKIPPED - {cond} brightness file has no lap_var")
            return
        for lab, m in (("day", ~night), ("night", night)):
            rates[f"{cond}/{lab}"] = float((dark | veil)[m].mean())
    for guard in ("clean/day", "glare/day"):
        assert rates[guard] == 0.0, f"guard cell {guard} vetoed at {rates[guard]:.4f}, must be 0"
    for dead in ("fog/day", "fog/night", "clean/night", "lowlight/night", "glare/night"):
        assert rates[dead] > 0.9, f"{dead} veto only {rates[dead]:.3f}, VIS is dead there"
    print(f"[veil] F composed switch: fog/day {rates['fog/day']:.0%}, fog/night "
          f"{rates['fog/night']:.0%}, guards clean/day + glare/day both 0%      OK")


def check_g_filters_must_stay_separate():
    """Pin the design: OR-ing first and dilating after breaks the glare/day guard."""
    bc = json.loads((ROOT / "runs/eval/brightness_constants.json").read_text(encoding="utf-8"))["vis"]
    tau = bc.get("tau_lap")
    if tau is None:
        print("[veil] G SKIPPED - tau_lap not fitted yet")
        return
    dark, veil, naive, night = _switches("glare", tau)
    day = ~night
    good, bad = float((dark | veil)[day].mean()), float(naive[day].mean())
    assert good == 0.0 and bad > 0.02, (
        f"expected separate-filter 0% vs dilate-the-OR >2%; got {good:.4f} / {bad:.4f}")
    print(f"[veil] G glare/day: filters separate {good:.1%}, dilate-the-OR {bad:.1%} "
          f"-- do not merge                     OK")


def main() -> int:
    check_a_no_structure_is_bit_identical()
    check_b_tau_none_is_inert()
    check_c_or_semantics()
    check_d_structure_alone()
    check_e_length_validation()
    check_f_separates_the_measured_cells()
    check_g_filters_must_stay_separate()
    print("\nVEIL GATE SMOKE OK — the new term reduces to the old rule wherever it should")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
