"""Corruption v2: physically modelled, calibrated, content-only (2026-10-09, revised 2026-10-10).

Why v1 is replaced (measured in docs/eval/corruption_inspect_2026-10-09/ and
docs/eval/corruption_v2/): albumentations 2.0.8 `RandomFog` ends with a fixed
min(h, w)//30 = 21-px Gaussian blur, so v1 fog is a blur whose strength does not
change with severity (fine detail falls to ~1.3% of clean at every severity, and the
median VIS target is 12 px tall); `RandomBrightnessContrast` is ADDITIVE
(x - beta*255, then clip; its docstring says multiplicative), so v1 low-light erases
dark pixels instead of dimming them; `RandomSunFlare`'s default ROI is the top half
of the 640x640 canvas, which puts the flare in the letterbox pad on 45% of frames;
and every v1 filter also rewrites the grey pad.

The rule (revision 2). A synthetic corruption must produce what the real camera plus the
dataset's processing chain would have produced, in physical order:

    atmosphere (fog) -> optics (glare veil, glint, blur) -> sensor noise
        -> exposure control / ISP / the IR per-frame stretch -> quantisation

The frames on disk are already at the end of that chain (VIS: the camera's 8-bit output,
resized 2048x1080 -> 640x338 and letterboxed; IR: 16-bit raw, per-frame min-max stretched
to 8 bit, letterboxed without resize). So each condition is undone to the stage it acts on,
applied there, and the later stages are re-applied:

* IR. Min-max is affine, so a corruption of the raw signal equals the same corruption of
  the 8-bit content followed by a fresh per-frame min-max over the whole IR frame (the
  content rows are the whole native frame). Every IR condition ends with that re-stretch.
  Revision 1 skipped it for fog, which kept a fogged frame at ~38% of its range where the
  real pipeline would have stretched it back to 0-255 (~2.6x too much contrast loss).
* Sensor noise is added AFTER the atmosphere and the optics. Fog multiplies the clean
  frame by t, which attenuates the clean frame's own sensor noise as well; the missing part
  is restored (IR: the frame's own noise, Immerkaer's estimator; VIS: the camera's noise
  level function measured on TRAIN at this 640-px scale).
* Exposure control. Across 5,777 TRAIN day frames the linear frame mean is the most stable
  statistic (sd of log 0.083, against 0.17-0.52 for median, trimmed, centre-weighted and
  log-mean metering), so the camera runs an average-metering AE across scenes. How strongly
  it reacts to radiance a corruption adds is AE_STRENGTH (0 = not at all, 1 = a converged
  mean-holding AE), fitted to the one real sun-entry sequence in TRAIN on the radiance
  profile around the glow (scripts/calibrate_glare_ae.py): best fit 0 -- within the ~4 s of
  the event the camera did not darken the rest of the frame; 1 is rejected (fit error 1.33
  against 0.18). The model is kept (never brightens; at night the camera is at its exposure
  limit, so it reacts only above the day set point) with sensitivity rows at 0.5 and 1.

Every condition runs on the CONTENT ROWS only (the pad stays 114) and is seeded per frame
exactly like v1 (`seed * 100003 + index`):

* fog       Koschmieder scattering, I = J t + A (1 - t), t = exp(-beta d), in linear
            radiance. d is the frame's metric depth map: Depth Anything V2 disparity
            scaled to metres by a robust fit to the flat-sea range of the frame's own
            attitude (scripts/build_depth_maps.py, frame_geometry.py). beta = 3.912 / V,
            V the meteorological visibility drawn per frame from the severity's band; a
            smooth +-20% patchiness field; A = the frame's own sky radiance just above the
            horizon. Then noise restoration, AE (VIS), quantisation; IR: then the stretch.
            Thermal: beta_IR = IR_BETA_RATIO * beta (assumed), A = the IR sky at the horizon
            on the frame's grey levels.
* lowlight  Exposure scaling. For a power-law camera response this is a multiplication in
            the stored domain, so nothing is clipped. Each frame is scaled to a target mean
            drawn from the real night distribution (train pohang01: median content mean
            6.8), then the night noise level function measured on the same frames is added
            and the result quantised. A frame already darker than the target is not
            brightened. VIS only.
* glare     A bright source placed by geometry: the sun 1-6 deg above the horizon by day, a
            lamp on the shore line at night (decided by frame brightness). Veiling glare with
            the spread I0 (1 + r^2/r0^2)^-1.5 (fitted by day and by night separately; both
            land on 1.5, a lens tail falling as r^-3) and a glint path on the water below,
            added in linear radiance; then the AE reaction; severity sets the source intensity.
* noise     VIS: shot noise as in ImageNet-C (Poisson, c = 60/25/12 photons at full scale);
            an ImageNet-C-style severity, not this camera's noise. IR: grey temporal noise
            plus column fixed-pattern noise (the two microbolometer terms), then the stretch.
* blur      Motion blur as v1 (ImageNet-C-style), on the content rows only; IR: then the stretch.
* rain      Albumentations RandomRain as v1 (ImageNet-C-style), content rows only. VIS only.

Not supported, by design: lowlight, glare and rain on thermal frames. LWIR does not depend
on scene illumination, does not produce coloured lens flare, and rain is not a visible streak
in LWIR; v1 applied visible-light filters there. Asking for them raises.

Constants: `DEFAULTS` below, each marked MEASURED (TRAIN split, scripts/calibrate_*.py,
docs/eval/corruption_v2_calibration/) or ASSUMED. Any can be overridden per transform
(`params=`) for a sensitivity row; the override is recorded in the transform's spec.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from uqfusion.eval import frame_geometry as fg

VERSION = "v2"
#: Hash of this file: stamped into every cache built with it, so a cache built by an older
#: revision of the v2 code can never pass a builder's "already built" check.
CODE_REV = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
CORRUPTIONS_V2 = ("fog", "rain", "lowlight", "glare", "blur", "noise")
SUPPORTED = {"vis": set(CORRUPTIONS_V2), "ir": {"fog", "blur", "noise"}}
GAMMA = 2.2                  # ASSUMED: camera response approximated as a 2.2 power law
_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)
#: noise level function bins (grey levels) and their centres
NLF_BINS = np.array([0, 2, 4, 8, 16, 32, 64, 128, 256], np.float32)
NLF_CENTRES = np.sqrt(np.maximum(NLF_BINS[:-1], 0.5) * NLF_BINS[1:]).astype(np.float32)

DEFAULTS = {
    # fog ------------------------------------------------------------------------------
    #: ASSUMED (definition): meteorological visibility band per severity, metres, log-uniform.
    #: s1 at the fog threshold (V < 1 km), s3 dense fog (V < 200 m). MEASURED context: the
    #: median VIS target is at 92 m (train boxes, 5-95%: 62-198 m).
    "FOG_V": {1: (600.0, 1000.0), 2: (250.0, 500.0), 3: (100.0, 200.0)},
    "FOG_PATCH": 0.20,       # ASSUMED: sd of the log patchiness multiplier on beta
    #: ASSUMED: LWIR extinction relative to visible in fog. Literature: ~0.3 (radiation fog,
    #: small droplets) to ~1 (advection fog). Sensitivity rows at 0.3 and 1.0.
    "IR_BETA_RATIO": 0.5,
    # lowlight -------------------------------------------------------------------------
    #: MEASURED: target content mean (stored grey) per severity; s2 = q25-q75 core of train
    #: pohang01 frame means (median 6.8); s1 two stops brighter (dusk), s3 two stops darker.
    "LOWLIGHT_TARGET": {1: (16.0, 32.0), 2: (4.0, 12.0), 3: (1.5, 3.0)},
    #: MEASURED: night noise level function on grey, var = a + b * mu (train pohang01, flat pixels)
    "NIGHT_NLF": (0.23, 0.0028),
    #: MEASURED: per-channel noise variance / grey noise variance (train, flat pixels).
    #: 2.24 if channels were independent; demosaicing + resize correlate them.
    "CHAN_NOISE_FACTOR": {"day": 2.25, "night": 1.83},
    #: MEASURED: camera noise sigma per NLF bin, grey levels at 640 px (train day / night)
    "DAY_NLF_SIGMA": (0.25, 0.25, 0.26, 0.42, 0.54, 0.70, 0.67, 0.49),
    "NIGHT_NLF_SIGMA": (0.47, 0.49, 0.49, 0.49, 0.52, 0.74, 0.75, 0.88),
    # glare / exposure -----------------------------------------------------------------
    #: (peak linear intensity, core radius px) of the source per severity. The spread is
    #: I0 (1 + r^2 / r0^2)^-beta: beta = 1 is a Lorentzian, whose r^-2 tail lifted the dark floor
    #: of every night frame (5th-percentile grey 16 at the s2 lamp against 7 for real lights of
    #: the same size, 100% of frames above the veto's `dark` threshold against 0%).
    #: SUN s2 FITTED (calibrate_sun_tail.py --fine, jointly with SUN_BETA and AE_STRENGTH) to the
    #: one real sun-in-frame event in TRAIN (pohang00 L 18884-19000: clipped area 0 -> 12.9%,
    #: radiance x1.3-1.7 within 180 px of the glow, x1.0 at 250-350 px), source pinned at the
    #: real glow centre; error 0.056 (Lorentzian best 0.177). s1/s3 = s2 x (0.4, 2.67) in
    #: intensity and x (0.7, 1.43) in core radius (ASSUMED spacing, the revision-1 ratios). A
    #: sun behind thin cloud: a direct sun would be brighter, so s2 is a measured case.
    "SUN": {1: (5.6, 28.0), 2: (14.0, 40.0), 3: (37.4, 57.2)},
    "SUN_BETA": 1.5,         # FITTED (above); grid 1-3
    #: LAMP FITTED (calibrate_lamp_tail.py) to real TRAIN night lights (pohang01): I0 solves the
    #: clipped-blob area at q90 / q99 / max of the largest real light (100 / 927 / 1,748 px);
    #: beta and r0 minimise the log error of the glow's ring means (15-240 px) and the dark
    #: floor's ring 5th percentiles (30-800 px) against the 101 real lights of 600-1300 px.
    #: s2: 5th-percentile grey 5.0 (real 7.0), above 10.5 on 3% of frames (real 0%).
    #: Core radius ladder s1/s3 = s2 x (2/3, 4/3) ASSUMED.
    "LAMP": {1: (2.592, 8.0), 2: (7.172, 12.0), 3: (7.693, 16.0)},
    "LAMP_BETA": 1.5,        # FITTED independently of SUN_BETA, and equal to it: one lens tail (r^-3)
    "I0_SCALE": 1.0,         # multiplier on SUN/LAMP intensity (and glint), for sensitivity rows
    "HALO_FRAC": 0.0,        # FITTED (revision 2): a wide (200-px) halo made the far field too bright
    "HALO_PX": 200.0,
    #: Intensity scale of the glint path on the water, kept at the revision-2 Lorentzian
    #: intensities so that refitting the veil's shape (beta moves the peak I0 needed for the
    #: same clipped core) does not change the reflection (ASSUMED; the glint was never fitted).
    "LAMP_GLINT": {1: 2.586, 2: 6.212, 3: 6.212},
    "SUN_GLINT": {1: 8.0, 2: 20.0, 3: 53.0},
    #: FITTED with SUN s2 (calibrate_sun_tail.py --fine): fraction of a converged average-
    #: metering AE's log-gain the camera applies to an in-frame source, 4-6 s after it enters.
    #: Fit error by strength: 0 -> 0.097, 0.25 -> 0.065, 0.375 -> 0.056 (best), 0.5 -> 0.074,
    #: 0.75 -> 0.217; 1 rejected (1.50, coarse grid). Applied to fog as well (one camera);
    #: sensitivity rows AE 0 and AE 1 (persistent fog could let the AE converge fully).
    "AE_STRENGTH": 0.375,
    #: MEASURED: median linear content mean of TRAIN day frames (the AE's set point)
    "AE_DAY_TARGET": 0.2161,
    "NIGHT_MEAN": 50.0,      # MEASURED: content mean below which a frame is night (0 errors on paired val / pohang04)
    "SRC_XY": None,          # calibration only: pin the glare source at (x, y) content px instead of drawing it
    # noise / blur / rain (ImageNet-C-style severities, ASSUMED) ------------------------
    "SHOT_C": {1: 60.0, 2: 25.0, 3: 12.0},
    "IR_NOISE": {1: (4.0, 2.0), 2: (8.0, 4.0), 3: (16.0, 8.0)},   # (temporal sd, column sd), grey
    "BLUR_LIMIT": {1: (5, 9), 2: (9, 15), 3: (15, 25)},
    "RAIN_BLUR": {1: 3, 2: 5, 3: 7},
}
FOG_FAR_M = 2000.0           # sky and far shore (matches build_depth_maps.FAR_M)
DEPTH_ROOT = fg.DATA_ROOT.parent / "runs" / "derived" / "depth_v2"
SUN_BGR = np.array([0.90, 0.97, 1.0], np.float32)
LAMP_BGR = np.array([0.45, 0.75, 1.0], np.float32)

# Back-compat names (read by inspection scripts).
FOG_V, IR_BETA_RATIO = DEFAULTS["FOG_V"], DEFAULTS["IR_BETA_RATIO"]
LOWLIGHT_TARGET, NIGHT_NLF = DEFAULTS["LOWLIGHT_TARGET"], DEFAULTS["NIGHT_NLF"]
SUN, LAMP, SHOT_C, IR_NOISE = DEFAULTS["SUN"], DEFAULTS["LAMP"], DEFAULTS["SHOT_C"], DEFAULTS["IR_NOISE"]
NIGHT_MEAN = DEFAULTS["NIGHT_MEAN"]


def _to_lin(im: np.ndarray) -> np.ndarray:
    return _LIN[im]


def _to_srgb_f(x: np.ndarray) -> np.ndarray:
    """Linear radiance -> stored grey levels, float (not yet quantised)."""
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / GAMMA) * 255.0


def _quant(y: np.ndarray) -> np.ndarray:
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def _to_srgb(x: np.ndarray) -> np.ndarray:
    return _quant(_to_srgb_f(x))


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: tuple[int, int] = (4, 7)) -> np.ndarray:
    import cv2

    z = rng.standard_normal(cells).astype(np.float32)
    f = cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC)
    return (f - f.mean()) / (f.std() + 1e-6)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _noise_sd(g: np.ndarray) -> float:
    """Sensor-noise sigma of one grey frame: Immerkaer's (1996) Laplacian-difference residual,
    taken robustly (1.4826 x median |r|) over FLAT pixels only, as the TRAIN noise level
    functions were measured. Immerkaer's plain mean |r| also counts edges and texture as
    noise; on textured frames that over-states it several times. Falls back to the plain
    estimator when fewer than 2,000 pixels are flat."""
    import cv2

    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6.0
    gf = g.astype(np.float32)
    r = cv2.filter2D(gf, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    mu = cv2.blur(gf, (5, 5))
    flat = np.hypot(cv2.Sobel(mu, cv2.CV_32F, 1, 0), cv2.Sobel(mu, cv2.CV_32F, 0, 1)) < 4.0 + 0.05 * mu
    flat[:1], flat[-1:], flat[:, :1], flat[:, -1:] = False, False, False, False
    if flat.sum() >= 2000:
        return float(1.4826 * np.median(np.abs(r[flat])))
    return float(np.sqrt(np.pi / 2) * np.abs(r[1:-1, 1:-1]).mean())


def _restretch(F: np.ndarray) -> np.ndarray:
    """The dataset's IR conversion: per-frame min-max to uint8 ((x-lo)/(hi-lo)*255+0.5).
    The IR content rows are the whole native frame, so this is the whole-frame rule."""
    lo, hi = float(F.min()), float(F.max())
    g = np.zeros_like(F, dtype=np.float32) if hi - lo < 1e-9 else (F - lo) / (hi - lo)
    return np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _ir_out(g: np.ndarray) -> np.ndarray:
    return np.repeat(g[..., None], 3, axis=2)


def _nlf_sigma(y: np.ndarray, night: bool, P: dict) -> np.ndarray:
    tab = np.asarray(P["NIGHT_NLF_SIGMA" if night else "DAY_NLF_SIGMA"], np.float32)
    return np.interp(y, NLF_CENTRES, tab).astype(np.float32)


def _restore_noise(y_new: np.ndarray, y0: np.ndarray, k: np.ndarray, night: bool, rng, P: dict) -> np.ndarray:
    """Sensor noise after the atmosphere/optics, in the stored domain.

    The camera's stored noise at level y is the measured NLF sigma(y) (per channel: times the
    measured cross-channel factor). A transform with local slope k (d y_new / d y0) carried
    the clean frame's noise through as k * sigma(y0); the part a real exposure would have and
    the transform removed is added back: var = sigma(y_new)^2 - k^2 sigma(y0)^2, if positive.
    """
    f = P["CHAN_NOISE_FACTOR"]["night" if night else "day"]
    var = f * (_nlf_sigma(y_new, night, P) ** 2 - (k ** 2) * _nlf_sigma(y0, night, P) ** 2)
    sd = np.sqrt(np.clip(var, 0.0, None))
    return y_new + rng.standard_normal(y_new.shape, dtype=np.float32) * sd


def _ae_gain(J: np.ndarray, target: float) -> float:
    """Gain g with mean(clip(g * J)) == target (metered on a 2x-strided grid); never > 1."""
    x = J[::2, ::2]
    if float(np.clip(x, 0, 1).mean()) <= target:
        return 1.0
    lo, hi = 1e-4, 1.0
    for _ in range(40):                        # mean(clip(g*J)) is monotone in g
        m = np.sqrt(lo * hi)
        if float(np.clip(m * x, 0, 1).mean()) > target:
            hi = m
        else:
            lo = m
    return lo


def _exposure(J: np.ndarray, J0: np.ndarray, night: bool, P: dict) -> float:
    """The camera's exposure reaction to radiance a corruption added. Day: the clean frame is
    taken as AE-converged, so its own mean is the set point. Night: the camera is at its
    exposure limit, so only a frame brighter than the day set point is pulled down."""
    m0 = float(np.clip(J0[::2, ::2], 0, 1).mean())
    target = max(m0, P["AE_DAY_TARGET"]) if night else m0
    return float(_ae_gain(J, target) ** P["AE_STRENGTH"])


class _Nominal:
    """Geometry when a frame has no navigation record: level camera, calibrated horizon row."""

    HORIZON = {"vis": 176.0, "ir": 281.0}     # train medians, content-relative rows

    def __init__(self, modality: str):
        self.modality = modality
        self.run = None

    def _grid(self):
        lo, hi = fg.content_rows(self.modality)
        f = {"vis": 1814.506 * fg.letterbox("vis")[0], "ir": 786.631}[self.modality]
        v = np.arange(hi - lo, dtype=np.float64)[:, None] - self.HORIZON[self.modality]
        return np.repeat(np.sin(np.arctan2(v, f)), 640, axis=1)

    def down(self):
        return self._grid()

    def range_m(self, far_m, h_cam=fg.H_CAM_M):
        d = self.down()
        with np.errstate(divide="ignore"):
            return np.minimum(np.where(d > 0, h_cam / np.maximum(d, 1e-12), np.inf), far_m)

    def horizon_rows(self):
        return np.full(640, self.HORIZON[self.modality])


def geometry(path, modality: str):
    if path is None:
        return _Nominal(modality)
    try:
        return fg.FrameGeometry(path, modality)
    except (KeyError, FileNotFoundError, OSError):
        return _Nominal(modality)


def _sky_band(gray: np.ndarray, hz: np.ndarray) -> np.ndarray:
    """Boolean mask of the band 4-40 rows above the horizon (falls back to the top rows)."""
    h, w = gray.shape
    rows = np.arange(h)[:, None]
    m = (rows < hz[None, :] - 4) & (rows >= hz[None, :] - 40)
    if m.sum() < 500:
        m = np.zeros_like(m)
        m[: max(8, h // 10)] = True
    return m


def _airlight(c: np.ndarray, gray: np.ndarray, hz: np.ndarray, modality: str) -> np.ndarray:
    band = _sky_band(gray, hz)
    g = gray[band]
    if modality == "ir":
        return np.full(3, np.median(g) / 255.0, np.float32)      # thermal: grey levels as radiance
    lo, hi = np.percentile(g, [80, 95])
    sel = band & (gray >= lo) & (gray <= hi)
    a = c[sel].reshape(-1, 3).mean(0) if sel.any() else c[band].reshape(-1, 3).mean(0)
    return _to_lin(np.clip(a + 0.5, 0, 255).astype(np.uint8))


def depth_path(image_path, modality: str):
    p = Path(image_path)
    return DEPTH_ROOT / modality / p.parent.name / f"{p.stem}.npy"


def load_depth(image_path, modality: str, shape: tuple[int, int]) -> np.ndarray:
    if image_path is None:
        raise ValueError("v2 fog needs the frame list (`images=`) to find each frame's depth map")
    p = depth_path(image_path, modality)
    if not p.is_file():
        raise FileNotFoundError(f"{p}: no depth map for {image_path}. Build it with "
                                f".venv_depth/Scripts/python.exe scripts/build_depth_maps.py "
                                f"--list <the frame list> --modality {modality}")
    d = np.load(p).astype(np.float32)
    if d.shape != shape:
        raise ValueError(f"{p}: depth map {d.shape} does not match the content rows {shape}")
    return d


def fog(c, sev, rng, geo, modality, P, path=None):
    import cv2

    h, w = c.shape[:2]
    V = _loguniform(rng, *P["FOG_V"][sev])
    beta = 3.912 / V * (P["IR_BETA_RATIO"] if modality == "ir" else 1.0)
    d = load_depth(path, modality, (h, w))
    patch = np.exp(P["FOG_PATCH"] * _smooth_field(rng, h, w))
    t = np.exp(-beta * d * patch)[..., None]
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    A = _airlight(c, gray, geo.horizon_rows(), modality)
    if modality == "ir":
        # 8-bit content is an affine image of the raw signal, so Koschmieder here + the
        # re-stretch == Koschmieder on raw (A maps the same way). The frame's own sensor
        # noise enters after the atmosphere: the part t removed is restored.
        J = c[..., 0].astype(np.float32) / 255.0                  # IR channels are replicated
        tt = t[..., 0]
        F = J * tt + A[0] * (1 - tt)
        s0 = _noise_sd(c[..., 0]) / 255.0
        F = F + rng.standard_normal(F.shape, dtype=np.float32) * s0 * np.sqrt(np.clip(1 - tt * tt, 0, None))
        lo, hi = float(F.min()), float(F.max())
        return _ir_out(_restretch(F)), {"V_m": V, "beta": beta, "noise_sd": float(s0 * 255),
                                        "stretch": float(1 / max(hi - lo, 1e-9))}
    night = float(gray.mean()) < P["NIGHT_MEAN"]
    J0 = _to_lin(c)
    I = J0 * t + A * (1 - t)
    g = _exposure(I, J0, night, P)
    y0 = c.astype(np.float32)
    y = _to_srgb_f(g * I)
    k = g * t * np.power(np.maximum(y0, 0.5) / np.maximum(y, 0.5), GAMMA - 1.0)
    y = _restore_noise(y, y0, k, night, rng, P)
    return _quant(y), {"V_m": V, "beta": beta, "A": A.tolist(), "ae_gain": g, "night": night}


def lowlight(c, sev, rng, geo, modality, P):
    import cv2

    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    target = _loguniform(rng, *P["LOWLIGHT_TARGET"][sev])
    a = min(1.0, target / max(float(gray.mean()), 1e-3))
    x = c.astype(np.float32) * a
    mu = np.maximum(x, 0.0)
    a_nlf, b_nlf = P["NIGHT_NLF"]
    var = (a_nlf + b_nlf * mu) * P["CHAN_NOISE_FACTOR"]["night"]
    x = x + rng.standard_normal(x.shape, dtype=np.float32) * np.sqrt(var)
    return _quant(x), {"target_mean": target, "gain": a}


def glare(c, sev, rng, geo, modality, P):
    import cv2

    h, w = c.shape[:2]
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    night = float(gray.mean()) < P["NIGHT_MEAN"]
    hz = geo.horizon_rows()
    xs = float(rng.uniform(0.15, 0.85) * w)
    hz_s = float(np.interp(xs, np.arange(w), hz))
    f = 1814.506 * fg.letterbox("vis")[0]
    elev = rng.uniform(0.0, 1.0) if night else rng.uniform(1.0, 6.0)
    ys = float(np.clip(hz_s - f * np.tan(np.radians(elev)), 4, h - 4))
    if P["SRC_XY"] is not None:
        xs, ys = (float(z) for z in P["SRC_XY"])
    I0, r0 = (P["LAMP"] if night else P["SUN"])[sev]
    I0 = I0 * P["I0_SCALE"]
    col = LAMP_BGR if night else SUN_BGR
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (u - xs) ** 2 + (v - ys) ** 2
    H = P["HALO_PX"]
    beta = P["LAMP_BETA"] if night else P["SUN_BETA"]
    core = r0 * r0 / (r2 + r0 * r0)
    veil = I0 * (core if beta == 1.0 else core ** beta) + P["HALO_FRAC"] * I0 * (H * H / (r2 + H * H))
    # glint path: a column under the source on the water, widening toward the camera,
    # modulated by the water's own ripple texture (bright facets reflect the source)
    below = np.clip(v - hz[None, :], 0, None)
    width = 3.0 + 0.35 * below
    band = np.exp(-0.5 * ((u - xs) / width) ** 2) * (below > 0)
    gf = gray.astype(np.float32)
    hp = gf - cv2.GaussianBlur(gf, (0, 0), 3)
    sparkle = np.clip(hp / (hp.std() + 1e-6), 0, None) ** 2
    Ig = (P["LAMP_GLINT"] if night else P["SUN_GLINT"])[sev] * P["I0_SCALE"]
    glint = (0.25 if night else 0.15) * Ig * band * (0.3 + sparkle)
    J0 = _to_lin(c)
    J = J0 + (veil + glint)[..., None] * col
    g = _exposure(J, J0, night, P)
    y0 = c.astype(np.float32)
    y = _to_srgb_f(g * J)
    k = g * np.power(np.maximum(y0, 0.5) / np.maximum(y, 0.5), GAMMA - 1.0)
    y = _restore_noise(y, y0, k, night, rng, P)
    return _quant(y), {"night": night, "src_xy": [xs, ys], "elev_deg": float(elev), "I0": I0, "ae_gain": g}


def noise(c, sev, rng, geo, modality, P):
    if modality == "ir":
        st, sc = P["IR_NOISE"][sev]
        h, w = c.shape[:2]
        n = rng.standard_normal((h, w), dtype=np.float32) * st + rng.standard_normal((1, w), dtype=np.float32) * sc
        return _ir_out(_restretch(c[..., 0].astype(np.float32) + n)), {"sd_t": st, "sd_col": sc}
    k = P["SHOT_C"][sev]
    x = rng.poisson(c.astype(np.float64) / 255.0 * k) / k * 255.0
    return np.clip(x + 0.5, 0, 255).astype(np.uint8), {"photons": k}


def _blur_noise_restore(c, r, comp_seeded, modality, rng, P):
    """Motion blur happens before the sensor reads out, so it must not smooth the frame's own
    sensor noise. The blur kernel's white-noise variance factor rho is measured exactly by
    running the same seeded transform on a noise field; the removed share (1 - rho) of the
    frame's noise is put back (VIS: the measured NLF; IR: the frame's own noise, then the
    IR stretch)."""
    # uint8 probe on purpose: albumentations draws a different kernel for float32 input
    # (measured: a float32 noise probe gave rho 0.034 where the uint8 impulse kernel has
    # sum k^2 = 0.098; a uint8 uniform-noise probe gives 0.0998).
    z = np.random.default_rng(0).integers(0, 256, c.shape).astype(np.uint8)
    zb = comp_seeded(image=z)["image"].astype(np.float32)
    zf = z.astype(np.float32)
    rho = float(np.clip(zb[8:-8, 8:-8].var() / max(zf[8:-8, 8:-8].var(), 1e-12), 0.0, 1.0))
    if modality == "ir":
        s0 = _noise_sd(c[..., 0]) / 255.0
        F = r[..., 0].astype(np.float32) / 255.0
        F = F + rng.standard_normal(F.shape, dtype=np.float32) * s0 * np.sqrt(1.0 - rho)
        return _ir_out(_restretch(F)), {"rho": rho, "noise_sd": s0 * 255}
    night = float(c.mean()) < P["NIGHT_MEAN"]
    y = r.astype(np.float32)
    f = P["CHAN_NOISE_FACTOR"]["night" if night else "day"]
    sd = np.sqrt(f * (1.0 - rho)) * _nlf_sigma(y, night, P)
    return _quant(y + rng.standard_normal(y.shape, dtype=np.float32) * sd), {"rho": rho}


def _albu(name, sev, seed_i, P):
    import albumentations as A

    if name == "blur":
        t = A.MotionBlur(blur_limit=tuple(P["BLUR_LIMIT"][sev]), p=1.0)
    else:
        t = A.RandomRain(blur_value=P["RAIN_BLUR"][sev], brightness_coefficient=0.8, p=1.0)
    comp = A.Compose([t], p=1.0)
    comp.set_random_seed(seed_i)
    return comp


FUNCS = {"fog": fog, "lowlight": lowlight, "glare": glare, "noise": noise}


def make_corruption_v2(name: str, severity: int = 2, seed: int = 0, modality: str = "vis",
                       images: list | None = None, params: dict | None = None):
    """transform(im_bgr, index) -> corrupted copy; deterministic per (seed, index).

    `images` is the ordered frame list the index refers to; fog and glare read the
    frame's navigation record (and fog its depth map) through it. Without it glare falls
    back to a level camera at the calibrated horizon row and says so in
    `transform.last_params["geometry"]`. `params` overrides entries of `DEFAULTS`
    (sensitivity rows); it is recorded in `transform.spec`.
    """
    if name not in CORRUPTIONS_V2:
        raise ValueError(f"unknown corruption '{name}' -- options: {CORRUPTIONS_V2}")
    if name not in SUPPORTED[modality]:
        raise ValueError(f"corruption v2 does not model '{name}' on {modality} frames "
                         f"(supported: {sorted(SUPPORTED[modality])}); see corruptions_v2 docstring")
    bad = set(params or {}) - set(DEFAULTS)
    if bad:
        raise ValueError(f"unknown corruption v2 params {sorted(bad)}")
    P = {**DEFAULTS, **(params or {})}
    for key in ("FOG_V", "LOWLIGHT_TARGET", "SUN", "LAMP", "LAMP_GLINT", "SUN_GLINT", "SHOT_C", "IR_NOISE", "BLUR_LIMIT", "RAIN_BLUR"):
        P[key] = {int(k): v for k, v in P[key].items()}          # JSON round trips stringify keys
    s = int(np.clip(severity, 1, 3))
    lo, hi = fg.content_rows(modality)

    def transform(im_bgr: np.ndarray, index: int) -> np.ndarray:
        if im_bgr.shape[:2] != (fg.CANVAS, fg.CANVAS):
            raise ValueError(f"expected a {fg.CANVAS}x{fg.CANVAS} letterboxed frame, got {im_bgr.shape}")
        seed_i = seed * 100003 + index
        out = im_bgr.copy()
        c = np.ascontiguousarray(im_bgr[lo:hi])
        if name == "rain":                                  # ImageNet-C-style, VIS only
            out[lo:hi] = _albu(name, s, seed_i, P)(image=c)["image"]
            transform.last_params = {}
            return out
        if name == "blur":
            r = _albu(name, s, seed_i, P)(image=c)["image"]
            out[lo:hi], prm = _blur_noise_restore(c, r, _albu(name, s, seed_i, P), modality,
                                                  np.random.default_rng(seed_i), P)
            transform.last_params = prm
            return out
        path = images[index] if images is not None else None
        geo = geometry(path, modality) if name in ("fog", "glare") else None
        rng = np.random.default_rng(seed_i)
        if name == "fog":
            out[lo:hi], prm = fog(c, s, rng, geo, modality, P, path=path)
        else:
            out[lo:hi], prm = FUNCS[name](c, s, rng, geo, modality, P)
        if geo is not None:
            prm["geometry"] = "nominal" if isinstance(geo, _Nominal) else "ahrs"
        transform.last_params = prm
        return out

    transform.last_params = {}
    transform.spec = {"corrupt": name, "severity": s, "corrupt_seed": seed, "corrupt_version": VERSION,
                      "corrupt_code": CODE_REV, "modality": modality, "content_rows": [lo, hi]}
    if params:
        transform.spec["corrupt_params"] = dict(params)
    return transform
