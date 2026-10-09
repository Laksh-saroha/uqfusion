"""Build the TMLR and IEEE JOE drafts from PAPER_DRAFT2.md, and fill every venue draft's references.

Both long drafts keep Draft 2's section numbering, so every "§x.y" in the copied text stays valid:
condensed subsections keep their heading in the main body and move their full text to an appendix.
Handwritten text (front matter, condensed summaries, conclusions) lives in this file; every number
in it is copied from Draft 2. Re-run after Draft 2 changes:

    py -3.13 venues/build_long_drafts.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VEN = ROOT / "venues"
D2 = (ROOT / "PAPER_DRAFT2.md").read_text(encoding="utf-8").replace("\r\n", "\n")


# ---------------------------------------------------------------- parse Draft 2
def parse(text):
    """Ordered list of (heading, body) for every '## ' / '### ' heading after the title."""
    lines = text.split("\n")
    heads = [i for i, ln in enumerate(lines) if re.match(r"#{2,3} ", ln)]
    out = []
    for k, i in enumerate(heads):
        j = heads[k + 1] if k + 1 < len(heads) else len(lines)
        out.append((lines[i], "\n".join(lines[i + 1:j]).strip("\n")))
    return out


SEC = parse(D2)
BODY = {h: b for h, b in SEC}
REFS = BODY["## References"]


def body(head):
    assert head in BODY, head
    return BODY[head]


def strip_rule(b):
    """Drop a trailing '---' separator that belongs to the section break."""
    return re.sub(r"\n*---\s*$", "", b).strip("\n")


def split_paras(b):
    return [p for p in re.split(r"\n{2,}", b.strip("\n")) if p.strip()]


def move(head, starts):
    """Split a subsection into (kept, moved) paragraphs; moved = those starting with any of `starts`."""
    kept, moved = [], []
    for p in split_paras(strip_rule(body(head))):
        (moved if any(p.lstrip().startswith(s) for s in starts) else kept).append(p)
    assert len(moved) == len(starts), (head, len(moved), len(starts))
    return "\n\n".join(kept), "\n\n".join(moved)


def section(head, text):
    return f"{head}\n\n{text.strip()}\n"


def fix_paths(s, appendix_for_10):
    s = s.replace("](docs/figures/", "](../docs/figures/")
    return s.replace("the laptop of §10", f"the laptop of {appendix_for_10}")


# ---------------------------------------------------------------- references
def fill_refs(path):
    """Rewrite the '## References' section with the Draft 2 entries the text cites (idempotent)."""
    text = path.read_text(encoding="utf-8")
    i = text.index("## References\n") + len("## References\n")
    j = text.find("\n---\n", i)
    j = len(text) if j < 0 else j
    text = text[:i] + "\n<!-- REFS -->\n" + text[j:]
    main = text[:i]
    entries = [e.strip() for e in REFS.split("\n\n") if e.strip()]
    used = []
    for e in entries:
        m = re.match(r"([^,]+),.*?\((\d{4})\)", e)
        sur, yr = m.group(1).strip(), m.group(2)
        pat = re.escape(sur) + r"(?: et al\.| and [A-ZÀ-ÿ][\w\-]+)?,? \(?" + yr
        if re.search(pat, main):
            used.append(e)
    # every author-year citation in the text must resolve to an entry
    cites = set(re.findall(r"([A-ZÀ-ÿ][\w\-]+)(?: et al\.| and [A-ZÀ-ÿ][\w\-]+)?,? \(?((?:19|20)\d\d)\)?", main))
    keys = {(re.match(r"([^,]+),", e).group(1).strip(), re.search(r"\((\d{4})\)", e).group(1)) for e in entries}
    surnames = {k[0] for k in keys}
    missing = sorted(c for c in cites if c[0] in surnames and c not in keys)
    path.write_text(text.replace("<!-- REFS -->", "\n\n".join(used)), encoding="utf-8")
    print(f"{path.name}: {len(used)} references" + (f"; UNRESOLVED {missing}" if missing else ""))


# ---------------------------------------------------------------- shared condensed text
C_3_2 = ("Counts were re-derived from disk: 158,319 images (127,309 VIS, 31,010 IR), 1,183,736 boxes and 28,388 "
         "paired frames. Pairing follows the dataset's own timestamp table, not frame ordinals; 16,544 of the "
         "pairs have different VIS and IR indices. Per-run counts are in Appendix {A}.1.")
C_3_3 = ("VIS frames are letterboxed from 2048×1080 into a 640×640 canvas, so 47 percent of every stored VIS frame "
         "is pad (which matters in §3.5). IR frames are min–max normalized per frame to 8 bits, which can mask "
         "thermal crossover and removes cross-frame radiometric comparability. Details, and a rejected "
         "percentile-clip export, are in Appendix {A}.2.")
C_3_4 = ("An interleaved K-block split with guard bands gives 80/10/10 train, validation and test per run, using "
         "the same ordinals for both modalities so that VIS–IR pairs never straddle a split. Sizes (Table S) are "
         "in Appendix {A}.3.")
C_3_5 = ("A train-only filter (2026-07-15) removed every box in night-run frames whose content-median luminance "
         "was below 100, on the reasoning that such boxes were copied from the thermal annotations. A padding "
         "constant bug made it cut whole frames: it dropped 132,688 boxes, of which only 38,135 individually fail "
         "intensity, gradient and contrast tests. A pre-registered restore (2026-09-02) put back all but those "
         "38,135, a net gain of 94,553 boxes. §6.6 reports the result. The label accounting (Table D) and two "
         "provenance anomalies are in Appendix {A}.4.")
C_5_6 = ("Six claims about the uncertainty metrics were tested and hold (Appendix {B}): D-ECE, the confidence-only "
         "form of the detection calibration error of Küppers et al. (2020), conditions on confidence only; AUSE "
         "(Ilg et al., 2018) and AURC (Geifman et al., 2019) are ranking-only; NLL and interval-ECE are computed on "
         "true positives and published with their true-positive share. One defect is disclosed rather than "
         "repaired: WBF can emit fused confidences above 1.0 (maximum 1.7532, on 0.0641 percent of detections).")
C_5_8 = ("A one-frame shift in cache pairing moves gated fusion by −0.000968, below the noise floor; only a "
         "content identity check, now run on every cache load, catches it. Before Phase 3, minimum detectable "
         "effects with five seeds were computed (VIS 0.01291, IR 0.01010); neither reaches the 0.0060 floor, so "
         "the retrained-versus-deployed comparison was cut in advance and is not reported (Appendix {B}).")
C_6_1 = ("Ninety-three training runs, 31 YOLO variants with three seeds each from six families (YOLOv8, Jocher et "
         "al., 2023; YOLOv9, C.-Y. Wang et al., 2024; YOLOv10, A. Wang et al., 2024; YOLO11, Jocher and Qiu, 2024; "
         "YOLO12, Tian et al., 2025; YOLO26, Jocher et al., 2026), were trained on the restored labels to a "
         "patience-20 stop. On the training library's own validation mAP over ship and buoy (not local AP), the "
         "top three, yolo26x (0.2666 ± 0.0040), yolo26l (0.2664 ± 0.0070) and yolo26m (0.2626 ± 0.0109), differ by "
         "less than the larger seed sd of every pair: a YOLO26 m/l/x tier with no resolved order. yolo26m was "
         "chosen earlier, on the Phase 1 grid, under a rule fixed in advance; it stays inside the tier and runs at "
         "57.0 FPS per detector against 30.7 for yolo26x (fp32, GPU clock pinned at 1500 MHz so that timings are "
         "consistent, detector `predict()` only). The full table "
         "(Table 1), the Phase 1 selection record (Table 1b), the throughput of every variant (Figure 2), the "
         "disclosures and the IR architecture ladder are in "
         "Appendix {C}.")


def condensed(text, **letters):
    return text.format(**letters)


# ---------------------------------------------------------------- TMLR
def rework(head, replace=None, moved=()):
    """Split a subsection's paragraphs. `replace` maps a paragraph start to handwritten text, a function of the
    original paragraph, or None to drop it; `moved` lists starts whose original paragraph goes to an appendix.
    A start in both is replaced in the main body and moved in full. Every start must match exactly once.
    Returns (kept text, {start: original paragraph})."""
    replace = replace or {}
    keys = set(replace) | set(moved)
    kept, mv, seen = [], {}, []
    for p in split_paras(strip_rule(body(head))):
        hits = [k for k in keys if p.lstrip().startswith(k)]
        assert len(hits) <= 1, (head, hits)
        k = hits[0] if hits else None
        if k is None:
            kept.append(p)
            continue
        seen.append(k)
        if k in moved:
            mv[k] = p
        if k in replace:
            r = replace[k]
            if r is not None:
                kept.append(r(p) if callable(r) else r)
    assert sorted(seen) == sorted(keys), (head, sorted(keys - set(seen)))
    return "\n\n".join(kept), mv


def pick(mv, *starts):
    return "\n\n".join(mv[s] for s in starts)


def replace_line(text, start, new):
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith(start)]
    assert len(idx) == 1, (start, len(idx))
    lines[idx[0]] = new
    return "\n".join(lines)


T_ABSTRACT = (
    "We pre-registered a test of whether predicted uncertainty should decide how a two-stream visible–infrared "
    "maritime detector fuses its streams. On the Pohang Canal dataset with PoLaRIS boxes, independent YOLO26 "
    "detectors with single-pass Gaussian variance heads feed a decision layer, and the registered comparison is "
    "real predicted σ against the same σ shuffled onto the wrong boxes, on ship AP, with a 0.0060 floor and "
    "block-bootstrap intervals. As a coordinate weight σ passes on zero of four conditions; as a score re-ranker it "
    "passes on three, so it is informative. It is not useful: the signal lies within each detector's own boxes, the "
    "σ-scored system is below the system with no σ on six of eight cells, and a learned within-detector variant "
    "failed its pre-registered replication on five retrained detectors (positive on all five, beyond the floor on "
    "two). The recorded runs had scored a ship-and-buoy macro instead of the registered metric, and the macro read "
    "NULL on both paths; we argue that the scored quantity belongs in the registration. The mechanism that shipped "
    "instead, an image-statistic veto with union aggregation, beats the visible stream by day on five retrained "
    "systems (+0.0059 to +0.0107 AP) but discards a working visible stream at night (−0.1847) once a label artifact "
    "behind its night arm was corrected. Paired noise floors, dependence-aware intervals and magnitude floors moved "
    "20 of 74 earlier findings to indeterminate, and one logged look at an untouched run returns a gap of +0.0216 "
    "that excludes neither zero nor 0.05.")
T_C1 = (
    "1. **A pre-registered test with a split answer.** As a coordinate weight in the fusion, real σ does not beat "
    "the same σ shuffled onto the wrong boxes on any of four conditions at a 0.0060 AP floor (R-D1), and letting σ "
    "arbitrate relaxed cross-modal correspondences does not rescue it (Stage 1, S1-NULL). As a score re-ranker, real "
    "σ beats shuffled σ on three of four conditions on the registered metric, ship AP, so the uncertainty is "
    "informative. It is not useful: the re-ranking is within each detector's own boxes, and the σ-scored system is "
    "below the shipped system without σ on six of eight cells (§6.4). Learned jointly with confidence, σ's "
    "within-detector gain is positive on all five retrained detectors but fails its pre-registered replication at "
    "the floor (§6.8).")
T_2_4 = (
    "Table R (Appendix A) positions this work against Gaussian YOLOv3, UA-CMDet, Zhao et al. (2024), RDSC-YOLOv4 "
    "and YOLOv7-Sea on uncertainty target, inference-time adaptation, calibration, registration and compute; every "
    "cell for another work comes from a direct read of its full text. Of the six systems it lists, this work is the "
    "only one that evaluates calibration. The two VIS–IR systems use uncertainty only as training-loss weights, "
    "removed at inference, and the maritime detectors output none.")
T_3_6 = (
    "pohang04 (26,188 VIS images, no IR) is the held-out run, but its visible labels are **not unseen**. Earlier "
    "VIS detectors, including the VIS ensemble and MC-Dropout arms, trained on 9,841 of its frames, and two VIS-only "
    "probes scored a validation list that included 2,343 of them (§6.8). No fusion score was ever computed on it, "
    "and the ten Phase 3 checkpoints and the Mahalanobis references built from them never saw it; the held-out claim "
    "of §7 rests on those facts and no wider one. Removing pohang04 from the Phase 3 lists raises the validation "
    "night share from 18.2 percent to 23.0 percent, so Phase 3 numbers are not comparable to earlier pooled "
    "validation numbers. List sizes and a contamination audit of the earlier Mahalanobis references are in "
    "Appendix B.5.")
T_4_2_GAUSS = (
    "**Gaussian head.** A fresh log-variance branch is attached to the live detection head of a loaded model, with "
    "no fork of the training library, and trained with a fourth loss term, beta-NLL (Seitzer et al., 2022), after a "
    "warm-up during which the NLL weight is zero. The σ branch reads detached features and the NLL sees a detached "
    "mean, so the deterministic detector is intended to train identically to the baseline by construction; §9 "
    "reports that this parity is not yet demonstrated. The port to YOLO26's end-to-end head is in Appendix C.")
T_5_3 = (
    "Deltas between systems are always paired on the same frames and the same corruption draw, which tightens the "
    "standard deviation of a delta by 3–16× on the nine informative cells. The combined draw-plus-bootstrap "
    "two-sigma floor on a paired delta is 0.0014–0.0031 AP on the macro over ship and buoy and 0.0008–0.0024 on ship "
    "AP, re-measured with the same arm, cells, draws and resamples; the maximum sets the magnitude floor in §5.4. "
    "Buoys carry 74–75 percent of the macro's variance while making up 5.3 percent of day ground-truth boxes. A delta "
    "below the floor is reported as \"not resolved,\" never as \"no effect.\" Sources and the two zero-information "
    "cells are in Appendix D.1.")
T_5_5_CLASS = (
    "**Class set.** Ship is the primary class. It is the only class both detectors emit, since the IR detector is "
    "single-class (§4.2), and the Phase 3 pre-registration fixed fused ship AP as its quantity before any Phase 3 "
    "number existed. Tables report ship AP except Table 1 and Table 4c (the macro over ship and buoy), Table 2 (both "
    "classes per stream) and two rows of Table L, each of which says so; the constants re-price of Appendix G is on "
    "the macro its registration scored. Rows recorded on the macro were re-scored on ship from cached detections, "
    "each after its unchanged macro path reproduced the record. The macro is not a safe stand-in for ship AP: a VIS "
    "veto deletes every buoy, because the surviving IR stream cannot supply one; buoys carry 74–75 percent of the "
    "macro's variance (§5.3); and a single-class stream scored on the macro reads exactly half its ship AP. R-D1 "
    "shows the cost: its macro read NULL where its registered ship AP reads POSITIVE (§6.4). The full accounting is "
    "in Appendix D.2.")
T_6_2_CAP = (
    "**Table 2. Per-stream uncertainty calibration, three arms, day slice (1,200 of the 2,232 paired validation "
    "frames), no fusion. Lower is better on every column except the two AP columns, which are local AP50-95 per "
    "class; best arm per stream and column in bold.** One checkpoint per arm, trained on restored labels. The IR "
    "detector is single-class (§4.2) and has no buoy AP. Ties in the MC-Dropout and ensemble uncertainties move AUSE "
    "and AURC by up to 6×10⁻⁴ and change no ordering. Sources, checkpoints and the tie analysis are in Appendix F.1.")
T_6_2_SCOPE = (
    "**Scope.** These are development-data numbers from single checkpoints that are neither Phase 3 checkpoints nor "
    "fusion inputs; R-D1 (§6.4) tests the σ head only. The VIS MC-Dropout and ensemble arms trained on 9,841 "
    "pohang04 frames (§3.6), so no three-arm number may appear on the held-out run. For those two arms the registered "
    "estimand is disagreement ranking, not predictive likelihood, so their NLL is undefined. Night is SUSPECT on both "
    "streams (Appendix F.2).")
T_6_2_TAIL = (
    "Figure 4 (Appendix F.3) shows the full lift screen. Day-only is the primary basis because the registered screen "
    "placed night in its SUSPECT band on both streams; that rationale, one registered rule amended after it fired, "
    "and why the arms are not ranked on mAP (Figure 5) are in Appendix F.4.")
T_6_3_INTRO = (
    "The decision layer of §4.2 is the sixth rewrite of the gate; each rewrite was forced by a measured failure of "
    "the previous one. The history (Table 3a, Figure 6), the pre-restore measurements behind it and its three general "
    "lessons are in Appendix G.1; two of the lessons recur in §8. Here we re-measure the shipped rule on the five "
    "Phase 3 systems.")
T_FIG6 = (
    "**Figure 7. The shipped rule on the five Phase 3 systems (Table 3b), ship AP (local AP), seed mean.** By day "
    "the fused output sits just above VIS alone on every cell. At night VIS is vetoed on every frame, so the fused "
    "output equals IR alone: right where VIS fails (fog, low light), wrong where it still works (clean, glare). "
    "Sources: `docs/eval/p3_night_check_2026-09-27.json`, `docs/eval/p3_corrupt_cells_2026-09-27.json`.")
T_6_4_DECOMP = (
    "**The gain is re-ranking within a stream, not fusion.** We decomposed the score-path deltas by emptying the IR "
    "stream, descriptively and after the verdict (Appendix H.2). By day, VIS re-ranking alone gives +0.0160 on clean "
    "and +0.0153 on glare under both presets, and adding the IR stream moves these by −0.0040 to +0.0035, with no "
    "consistent sign. Wherever VIS is vetoed, on every night frame and on fog under `crossmodal`, the delta is IR "
    "re-ranking IR boxes. Lowlight fails by day because real σ does not re-rank VIS boxes there (+0.0006, CI spans "
    "zero).")
T_6_5_PRE = (
    "The coordinate-path null of R-D1 was measured at a merge threshold (0.85) where almost nothing merges, and two "
    "earlier negative results on correspondence were each measured with σ inert (Appendix H.3). Phase 3 Stage 1 "
    "crossed the two: threshold {0.85, 0.55} × σ-weighting {off, live}. Cells A (shipped), B (shipped threshold, σ "
    "live) and C (relaxed, σ off) were known. Cell D, relaxed with σ live, was the experiment: it had to be "
    "non-inferior to A within 0.0060 on at least three of four conditions on TUNE.")
T_6_5_POST = (
    "One of four: verdict S1-NULL, robust at every floor from 0.0014 to 0.0100. Low-light passes because relaxing "
    "correspondence barely costs anything there (C − A = −0.0013), not because live σ recovered anything: σ changed "
    "the fused output on 753–836 of 836 clean frames, yet the interaction terms B − A and D − C sit inside "
    "[−0.0002, +0.0004] everywhere, with every CI spanning zero. TEST, pre-declared not to override TUNE, gives two "
    "of four. The correspondence question is closed, and the fusion is documented as union aggregation, not "
    "consensus.")
T_6_6_INHERIT = (
    "**The retrained system inherits the rule and pays for it.** The five Phase 3 VIS detectors trained on the "
    "restored labels see at night (0.2535 seed-mean ship AP on the night run), but the frozen rule still drops VIS on "
    "every night frame (Figure 9), at the costs in Table 3b. The rule was correct for the detector it was written against and "
    "is wrong for the detector it ships with. Nothing in the image changed; what changed is the claim the rule makes "
    "about the detector.")
T_6_6_ARM = (
    "**Removing the night arm is not the fix.** Three pre-registered attempts, run before Phase 3 on an earlier "
    "night-trained VIS checkpoint, tried to re-price the arm, and none was adopted (Appendix G.2): removing the arm "
    "left two VIS-degraded night cells below IR alone, replacing darkness with a VIS health test left five night "
    "cells below max(VIS, IR), and widening the trigger broke day safety. The variable that should gate VIS at night "
    "is VIS health, not darkness, but the one health instrument that separated them (AUROC 0.9921) was selected on "
    "the only night run, so no held-out night exists to confirm it. We report the night cost of the frozen rule as a "
    "defect of the shipped system, not a tuned repair.")
T_6_7 = (
    "The night vote trusts IR, which was uncorrupted in every benchmark cell, so we attacked the frozen rule "
    "`ir_p05 > 41.5` with six IR hazards at three severities (Appendix I, Table 6). A false night on a clear day "
    "vetoes a working VIS stream. The raw rule misreads up to 94.8 percent of clear days as night under IR fog and "
    "19–27 percent under IR glare. Two votes, an IR self-check, the multivariate health score and an authority bound "
    "took the false-night rate on 19 IR-corruption arms to 0 percent at zero benchmark cost; that figure is "
    "in-sample (§9), and the both-degraded worst-case false-veto rate fell from 24 percent to 1.3 percent. An "
    "abstain signal was demoted to an advisory flag after it prevented zero bad vetoes and lost 2,095 correct ones.")
T_6_8_LEAD = (
    "Table L (Appendix J) lists the fusion and post-processing levers that were tested and not adopted, as paired "
    "ship-AP deltas on the frames each row names. Most are inert or negative; three results from it bear on the "
    "argument of this paper.")
T_7_1_LOOK = (
    "The look is mechanically single-shot: the scoring script refuses to run unless the repository is at a clean "
    "FREEZE commit, a 316-file hash manifest verifies and the development reference reproduces exactly, and it "
    "writes a `LOOK_TAKEN` marker before scoring begins, so a crash mid-look still counts as the look having been "
    "taken (Appendix K.1). The exposure is logged in the project's ledger.")
T_7_2_DESC = (
    "The ten descriptive cells (Table 7, Appendix K.2) carry no pass or fail language. They show only what the "
    "development cells already showed: the fused output tracks the VIS stream. IR-side corruption leaves it within "
    "±0.0014 of clean, and VIS-side corruption moves it by up to 0.2568.")
T_LIM = [
    "1. **One held-out run.** No untouched test set existed before pohang04; pohang02 and pohang03 were declared "
    "TEST after the fact and fail a selection-bias test. pohang04 is now spent, and its look is inconclusive (§7.2).",
    "2. **The shipped night rule is wrong for the shipped detector** (§6.3, §6.6). It is reported, not repaired.",
    "3. **Three checkpoint generations** (yolo26s, pre-restore yolo26m, Phase 3), each named, never pooled.",
    "4. **In-sample constants.** The IR night threshold, the IR health model and the capability prior were "
    "fitted on frames that the evaluation also scores.",
    "5. **Pohang only**, adverse conditions simulated, and night a single run with no between-run interval.",
    "6. **pohang04 has VIS ground truth only**, and only the fused output was scored there.",
    "7. **Registration residual** of 3–6 px median with drift up to 10 px; no time-varying homography.",
    "8. **Parity of the σ-attached detector** with its baseline is claimed neither as bit-identity nor as "
    "non-inferiority.",
    "9. **The 1.95 interval factor is a lower bound**, measured on VIS uncertainty-arm deltas.",
    "10. **R-D1** was scored on the wrong metric before being re-scored on the registered one, α was never tuned, "
    "Table 3a and Table L were re-scored on ship AP after the fact, and the within-detector σ gain is "
    "unresolved in size (§6.4, §6.8).",
    "",
    "The full list of 24 items is in Appendix L.",
]
T_CONCL = (
    "A pre-registered test asked whether predicted uncertainty should decide how two sensors' detections are "
    "fused. It should not: σ carries information about a detector's own boxes and none about which sensor to "
    "believe, and spending it at the registered strength costs accuracy. Two methodological points generalize "
    "beyond this system. A shuffled control establishes that a signal exists; only the comparison against no "
    "signal establishes that using it helps, and a registration should name both. And the scored quantity is "
    "part of the registration: scoring a macro that included a class one stream cannot detect reversed one of "
    "two verdicts. The sensor-selection rule that shipped instead shows the cost of not re-pricing a decision "
    "rule when the detector beneath it changes.")
TABLE_4C = "| Path | Condition | `crossmodal` | `crossmodal26m` |\n|---|---|---|---|\n| coordinate (S1 − S3) | clean | −0.000566"


def build_tmlr():
    """Main body cut toward 15 typeset pages; appendices A–M follow the order of the sections they come from."""
    L = dict(A="B", B="D", C="E")
    title = D2.split("\n", 1)[0]  # Draft 2's title line
    note = ("> Venue draft for TMLR, derived from `PAPER_DRAFT2.md` at commit `d9260ef` (2026-10-09) by "
            "`venues/build_long_drafts.py`. TMLR has no hard page limit, but a main body over 12 pages gets a "
            "longer review. The main body is cut toward 15 typeset pages: condensed subsections keep their headings, "
            "and their full text, four tables and one figure move to appendices A–M, in the order of the sections "
            "they come from. Every number is from Draft 2. See `venues/README.md`.")
    out = [title, "", "**Anonymous authors** (TMLR review is double-blind; restore the author block for the camera-ready).",
           "", note, "", "---", "", "## Abstract", "", T_ABSTRACT, "", "---", ""]
    app = {}  # appendix letter -> list of blocks

    def keep(h):
        out.append(section(h, strip_rule(body(h))) if strip_rule(body(h)) else h + "\n")

    def put(letter, *blocks):
        app.setdefault(letter, []).extend(b.rstrip("\n") + "\n" for b in blocks if b)

    intro = replace_line(strip_rule(body("## 1. Introduction")), "1. **A pre-registered test with a split answer.**", T_C1)
    out.append(section("## 1. Introduction", intro))
    for h in ["## 2. Related work", "### 2.1 Uncertainty in single-stage detectors",
              "### 2.2 Visible–infrared fusion with uncertainty", "### 2.3 Maritime detectors and datasets"]:
        keep(h)
    out.append(section("### 2.4 Comparison axes", T_2_4))
    put("A", "## Appendix A. Comparison axes (full text of §2.4)\n", strip_rule(body("### 2.4 Comparison axes")))

    out.append("## 3. Dataset\n")
    keep("### 3.1 Pohang Canal and PoLaRIS")
    put("B", "## Appendix B. Dataset details (full text of §3.2–§3.6)\n")
    for h, c, a in [("### 3.2 Verified counts", C_3_2, "### B.1 Verified counts"),
                    ("### 3.3 Preprocessing", C_3_3, "### B.2 Preprocessing"),
                    ("### 3.4 Splits", C_3_4, "### B.3 Splits"),
                    ("### 3.5 The night-box filter and its reversal", C_3_5, "### B.4 The night-box filter and its reversal")]:
        out.append(section(h, condensed(c, **L)))
        put("B", section(a, strip_rule(body(h))))
    out.append(section("### 3.6 Holdout and contamination", T_3_6))
    put("B", section("### B.5 Holdout and contamination", strip_rule(body("### 3.6 Holdout and contamination"))))

    out.append(section("## 4. Method", strip_rule(body("## 4. Method"))))
    keep("### 4.1 As designed")
    kept, mv = rework("### 4.2 As shipped (preset `crossmodal26m`)",
                      replace={"**Gaussian head.**": T_4_2_GAUSS}, moved=["**Gaussian head.**"])
    out.append(section("### 4.2 As shipped (preset `crossmodal26m`)", kept))
    put("C", "## Appendix C. The Gaussian head on an end-to-end detector (from §4.2)\n", pick(mv, "**Gaussian head.**"))

    out.append("## 5. Experimental protocol and statistics\n")
    keep("### 5.1 Benchmark cells and substrates")
    keep("### 5.2 Tune and test discipline")
    out.append(section("### 5.3 Noise floor", T_5_3))
    keep("### 5.4 Dependence-aware intervals")
    kept, mv = rework("### 5.5 AP convention", replace={"**Class set.**": T_5_5_CLASS}, moved=["**Class set.**"])
    out.append(section("### 5.5 AP convention", kept))
    put("D", "## Appendix D. Protocol details (from §5.3, §5.5, §5.6 and §5.8)\n",
        section("### D.1 Noise floor", strip_rule(body("### 5.3 Noise floor"))),
        section("### D.2 Class set", pick(mv, "**Class set.**")))
    out.append(section("### 5.6 Metric contracts", condensed(C_5_6, **L)))
    put("D", section("### D.3 Metric contracts", strip_rule(body("### 5.6 Metric contracts"))))
    keep("### 5.7 Pre-registrations and decision rules")
    out.append(section("### 5.8 Identity checks and power", condensed(C_5_8, **L)))
    put("D", section("### D.4 Identity checks and power", strip_rule(body("### 5.8 Identity checks and power"))))

    out.append("## 6. Results\n")
    out.append(section("### 6.1 Backbone benchmark is a negative result", condensed(C_6_1, **L)))
    put("E", "## Appendix E. Backbone benchmark (full text of §6.1)\n",
        section("### E.1 Tables 1 and 1b", strip_rule(body("### 6.1 Backbone benchmark is a negative result"))))

    s62 = ["**Table 2.", "**Scope and caveats.**", "![lift_screen]", "**Figure 4.", "**Why day-only is primary.**",
           "One registered rule was amended", "We do not rank uncertainty methods", "![checkpoint_selection]",
           "**Figure 5."]
    kept, mv = rework("### 6.2 Per-modality uncertainty calibration",
                      replace={"**Table 2.": T_6_2_CAP, "**Scope and caveats.**": T_6_2_SCOPE}, moved=s62)
    out.append(section("### 6.2 Per-modality uncertainty calibration", kept + "\n\n" + T_6_2_TAIL))

    s63 = ["The decision layer in §4.2 is the sixth rewrite.", "**Table 3a.", "| Stage (date)",
           "Ship AP; each gap is against", "Under the shipped preset on the pre-restore",
           "Three findings from the rewrite history", "![gate_history]", "**Figure 6."]
    kept, mv63 = rework("### 6.3 Fusion robustness of the sensor-selection baseline",
                        replace={"**Figure 7.": T_FIG6}, moved=s63)
    out.append(section("### 6.3 Fusion robustness of the sensor-selection baseline", T_6_3_INTRO + "\n\n" + kept))

    s64 = ["**Table 4c.", TABLE_4C, "Two macro results", "**The gain is re-ranking"]
    kept, mv64 = rework("### 6.4 Uncertainty is informative but does not improve the fusion (R-D1)",
                        replace={"**The gain is re-ranking": T_6_4_DECOMP, "**Table 4c.": None, TABLE_4C: None,
                                 "Two macro results": None,
                                 "**Why the macro hid it.**": lambda p: p + " The recorded macro table (Table 4c) "
                                 "and two macro results that do not survive on ship AP are in Appendix H.1."},
                        moved=s64)
    out.append(section("### 6.4 Uncertainty is informative but does not improve the fusion (R-D1)", kept))

    s65 = ["The coordinate-path null of R-D1", "One of four: verdict"]
    kept, mv65 = rework("### 6.5 Relaxing correspondence does not rescue the coordinate path (Stage 1)",
                        replace={s65[0]: T_6_5_PRE, s65[1]: T_6_5_POST}, moved=s65)
    out.append(section("### 6.5 Relaxing correspondence does not rescue the coordinate path (Stage 1)", kept))

    s66 = ["**Removing the night arm", "The measured lesson from", "A related repair"]
    kept, mv66 = rework("### 6.6 Night visible blindness was a label artifact",
                        replace={"**The retrained system inherits": T_6_6_INHERIT, s66[0]: T_6_6_ARM,
                                 s66[1]: None, s66[2]: None}, moved=s66)
    out.append(section("### 6.6 Night visible blindness was a label artifact", kept))

    out.append(section("### 6.7 Is the IR night switch safe when IR is corrupted?", T_6_7))

    s68 = ["**Table L.", "| Lever | Result", "The soft-NMS rejection"]
    kept, mv68 = rework("### 6.8 Levers that are inert or negative",
                        replace={k: None for k in s68}, moved=s68)
    out.append(section("### 6.8 Levers that are inert or negative", T_6_8_LEAD + "\n\n" + kept))

    out.append(section("## 7. Held-out evaluation: the single pohang04 look",
                       strip_rule(body("## 7. Held-out evaluation: the single pohang04 look"))))
    kept, mv71 = rework("### 7.1 Protocol (fixed before the look)",
                        replace={"The look is mechanically": T_7_1_LOOK}, moved=["The look is mechanically"])
    out.append(section("### 7.1 Protocol (fixed before the look)", kept))
    s72 = ["**Table 7.", "| Cell (VIS / IR)"]
    kept, mv72 = rework("### 7.2 Result", replace={s72[0]: None, s72[1]: None, "The ten descriptive cells": T_7_2_DESC},
                        moved=s72)
    out.append(section("### 7.2 Result", kept))

    s8 = ["**Redundancy is worth", "**Synthetic ladders", "**Checkpoint selection is noisier"]
    kept, mv8 = rework("## 8. Discussion", replace={k: None for k in s8}, moved=s8)
    out.append(section("## 8. Discussion", kept))
    out.append(section("## 9. Limitations", "\n".join(T_LIM)))
    out.append(section("## 10. Conclusion", T_CONCL))
    out.append("## References\n\n<!-- REFS -->\n\n---\n")

    put("F", "## Appendix F. Calibration details (from §6.2)\n",
        section("### F.1 Table 2: sources, checkpoints and ties", pick(mv, "**Table 2.")),
        section("### F.2 Scope and caveats", pick(mv, "**Scope and caveats.**")),
        section("### F.3 Signal lift screen", pick(mv, "![lift_screen]", "**Figure 4.")),
        section("### F.4 Day-only basis, amendment and checkpoint selection",
                pick(mv, "**Why day-only is primary.**", "One registered rule was amended",
                     "We do not rank uncertainty methods", "![checkpoint_selection]", "**Figure 5.",)
                + "\n\n" + mv8["**Checkpoint selection is noisier"]))
    put("G", "## Appendix G. Gate history and night-arm re-pricing (from §6.3 and §6.6)\n",
        section("### G.1 Gate rewrite history", pick(mv63, *s63) + "\n\n" + mv8["**Synthetic ladders"]),
        section("### G.2 Night-arm re-pricing", pick(mv66, *s66)))
    put("H", "## Appendix H. Uncertainty-mechanism details (from §6.4 and §6.5)\n",
        section("### H.1 R-D1 as recorded, on the macro", pick(mv64, "**Table 4c.", TABLE_4C, "Two macro results")),
        section("### H.2 Score-path decomposition", pick(mv64, "**The gain is re-ranking")),
        section("### H.3 Stage 1 in full", pick(mv65, *s65)))
    put("I", section("## Appendix I. IR night-switch safety (full text of §6.7)",
                     strip_rule(body("### 6.7 Is the IR night switch safe when IR is corrupted?"))))
    put("J", "## Appendix J. Levers tested and not adopted (from §6.8)\n",
        section("### J.1 Table L", pick(mv68, "**Table L.", "| Lever | Result")),
        section("### J.2 Notes", pick(mv68, "The soft-NMS rejection") + "\n\n" + mv8["**Redundancy is worth"]))
    put("K", "## Appendix K. Held-out look details (from §7)\n",
        section("### K.1 Single-shot mechanics", pick(mv71, "The look is mechanically")),
        section("### K.2 All eleven cells", pick(mv72, *s72)))
    put("L", section("## Appendix L. Limitations in full (from §9)", strip_rule(body("## 9. Limitations"))))
    put("M", section("## Appendix M. Reproducibility and implementation notes",
                     strip_rule(body("## 10. Reproducibility and implementation notes"))))
    assert sorted(app) == list("ABCDEFGHIJKLM"), sorted(app)
    appendix = [blk for k in sorted(app) for blk in app[k]]
    text = "\n".join(out) + "\n" + "\n".join(appendix)
    # relative pointers whose target moved to an appendix
    for old, new in [("are not a ranking (next paragraph)", "are not a ranking (Appendix F.4)")]:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    # double-blind: the public repository link identifies the author
    text = text.replace("(github.com/Laksh-saroha/uqfusion, branch `fusion-uq-phase3`)",
                        "(an anonymized repository, linked for review)")
    assert not re.search(r"Saroha|Laksh-saroha|Thapar|Mandia", text, re.I)
    p = VEN / "PAPER_TMLR.md"
    p.write_text(fix_paths(text, "Appendix M"), encoding="utf-8")
    return p


# ---------------------------------------------------------------- IEEE JOE
def build_joe():
    L = dict(A="A", B="B", C="A")
    title = ("# Predicted Uncertainty and Sensor Selection for Visible–Infrared Ship Detection at Sea: "
             "A Pre-Registered Evaluation on the Pohang Canal Dataset")
    note = ("> Venue draft for the IEEE Journal of Oceanic Engineering, derived from `PAPER_DRAFT2.md` at commit "
            "`d9260ef` (2026-10-09) by `venues/build_long_drafts.py`. JOE sets no fixed length; this version keeps "
            "Draft 2's numbering, reframes the front and back matter for a maritime-engineering audience, and moves "
            "the backbone benchmark, the metric contracts, the checkpoint-selection analysis and the implementation "
            "notes to appendices. Every number is from Draft 2. See `venues/README.md`.")
    abstract = (
        "Visible cameras on vessels degrade in fog, glare and darkness, and thermal cameras lose targets at "
        "thermal crossover; neither failure announces itself. We built and evaluated a two-stream ship detector "
        "on the Pohang Canal dataset in which each sensor has its own YOLO26 detector with a predicted per-box "
        "variance, and asked, under pre-registered decision rules, whether that variance can decide which sensor "
        "to trust. It cannot. As a fusion weight it changes nothing on any of four conditions. As a re-ranker of "
        "detection scores it carries real information, but only about each detector's own boxes, and using it "
        "lowers ship AP against using none on six of eight cells; a learned variant failed a pre-registered "
        "replication on five retrained detectors. The cameras are not co-registered (3–6 px residual), and at the "
        "merge threshold used only 0.05 percent of visible boxes have a thermal partner, so fusion is in effect "
        "the concatenation of the two detectors' outputs. What works by day is a hard veto on image statistics, "
        "dropping the visible stream when the thermal camera reports night and the visible frame is dark or "
        "veiled, followed by the union of the surviving detections: on five retrained systems it beats the visible "
        "stream alone on every daylight condition (+0.0059 to +0.0107 AP). The same rule, written while a "
        "labelling artifact made the visible detector blind at night, discards a working visible stream once the "
        "detector is retrained (−0.1847 AP on clear nights). We report the evaluation protocol in full, including "
        "paired noise floors, intervals that respect 10 Hz autocorrelation and a single pre-registered look at an "
        "untouched run (fused ship AP 0.2682), and draw design lessons for maritime perception systems.")
    index = ("*Index Terms*—Maritime object detection, infrared imaging, sensor fusion, uncertainty estimation, "
             "autonomous navigation, evaluation methodology, pre-registration.")
    out = [title, "", "**Laksh Saroha**, Department of Electronics and Communication Engineering, Thapar Institute of "
           "Engineering and Technology, Patiala.", "", note, "", "---", "", "## Abstract", "", abstract, "", index,
           "", "---", ""]
    appendix = []

    intro_paras = split_paras(strip_rule(body("## 1. Introduction")))
    joe_open = (
        "Autonomous and assisted navigation in harbours and coastal waters depends on detecting other vessels "
        "early and reliably. Electro-optical sensing is the least expensive way to do it, and a visible camera "
        "paired with a long-wave infrared camera is a common configuration, because the two fail in different "
        "conditions. Visible imagery degrades in fog, haze, glare, rain and darkness. Thermal imagery sees through "
        "darkness and glare but loses a vessel at thermal crossover, when hull and water reach the same "
        "temperature. The engineering problem is that neither failure announces itself: a detector that has "
        "stopped seeing emits fewer boxes, and a naive fusion rule reads fewer boxes as fewer targets, not as a "
        "blind sensor. This paper reports a two-stream detector built for that problem on the Pohang Canal dataset "
        "(Chung et al., 2023), recorded along a 7.5 km route from a canal through inner and outer port to "
        "near-coastal water, with PoLaRIS ship and buoy annotations (Choi et al., 2025), and asks whether each "
        "detector's own predicted uncertainty can tell the system, frame by frame, which sensor to trust.")
    out.append(section("## 1. Introduction", "\n\n".join([joe_open] + intro_paras[1:])))
    for h in ["## 2. Related work", "### 2.1 Uncertainty in single-stage detectors",
              "### 2.2 Visible–infrared fusion with uncertainty", "### 2.3 Maritime detectors and datasets",
              "### 2.4 Comparison axes", "## 3. Dataset", "### 3.1 Pohang Canal and PoLaRIS", "### 3.2 Verified counts",
              "### 3.3 Preprocessing", "### 3.4 Splits", "### 3.5 The night-box filter and its reversal",
              "### 3.6 Holdout and contamination", "## 4. Method", "### 4.1 As designed",
              "### 4.2 As shipped (preset `crossmodal26m`)", "## 5. Experimental protocol and statistics",
              "### 5.1 Benchmark cells and substrates", "### 5.2 Tune and test discipline", "### 5.3 Noise floor",
              "### 5.4 Dependence-aware intervals", "### 5.5 AP convention"]:
        b = strip_rule(body(h))
        out.append(section(h, b) if b else h + "\n")
    appendix.append("## Appendix A. Metric contracts and backbone benchmark (full text of §5.6 and §6.1)\n")
    out.append(section("### 5.6 Metric contracts", condensed(C_5_6, **dict(L, B="A.1"))))
    appendix.append(section("### A.1 Metric contracts", strip_rule(body("### 5.6 Metric contracts"))))
    out.append(section("### 5.7 Pre-registrations and decision rules",
                       strip_rule(body("### 5.7 Pre-registrations and decision rules"))))
    out.append(section("### 5.8 Identity checks and power", strip_rule(body("### 5.8 Identity checks and power"))))
    out.append("## 6. Results\n")
    out.append(section("### 6.1 Backbone benchmark is a negative result", condensed(C_6_1, **dict(L, C="A.2"))))
    appendix.append(section("### A.2 Tables 1 and 1b", strip_rule(body("### 6.1 Backbone benchmark is a negative result"))))
    kept, moved = move("### 6.2 Per-modality uncertainty calibration",
                       ["We do not rank uncertainty methods", "![checkpoint_selection]", "**Figure 5."])
    kept += ("\n\nWhy the arms are not ranked on mAP, including the checkpoint-selection analysis of Figure 5, is in "
             "Appendix B.")
    out.append(section("### 6.2 Per-modality uncertainty calibration", kept))
    appendix.append(section("## Appendix B. Checkpoint selection (from §6.2)", moved))
    for h in ["### 6.3 Fusion robustness of the sensor-selection baseline",
              "### 6.4 Uncertainty is informative but does not improve the fusion (R-D1)",
              "### 6.5 Relaxing correspondence does not rescue the coordinate path (Stage 1)",
              "### 6.6 Night visible blindness was a label artifact",
              "### 6.7 Is the IR night switch safe when IR is corrupted?",
              "### 6.8 Levers that are inert or negative",
              "## 7. Held-out evaluation: the single pohang04 look", "### 7.1 Protocol (fixed before the look)",
              "### 7.2 Result"]:
        b = strip_rule(body(h))
        out.append(section(h, b) if b else h + "\n")
    implications = (
        "**Implications for maritime perception systems.** Five practical points follow for teams fielding "
        "visible–thermal detection on vessels.\n"
        "* *Measure registration before designing fusion.* With timestamp pairing and a 3–6 px residual, "
        "decision-level fusion degenerated to concatenation; cross-modal agreement cannot serve as evidence when "
        "it almost never occurs (§4.2, §6.5).\n"
        "* *Treat a sensor-selection rule as calibrated to one detector.* Any retraining, including on corrected "
        "labels, changes the rule's premise; re-measure the rule on every condition before deployment (§6.3, §6.6).\n"
        "* *Gate on the health of the stream to be dropped, not on a scene condition that happens to coincide with "
        "it.* Darkness stood in for \"the visible detector fails\" only while a labelling artifact made it true.\n"
        "* *Harden any cross-sensor vote against the voting sensor's own failures.* The thermal night vote misread "
        "up to 94.8 percent of clear days as night under thermal fog until it had to pass a health check (§6.7).\n"
        "* *Budget the evaluation for small effects.* Paired comparisons, intervals that respect frame "
        "autocorrelation and a magnitude floor are needed, because fusion changes of a few thousandths of AP are "
        "otherwise indistinguishable from resampling noise (§5.3, §5.4).")
    out.append(section("## 8. Discussion", strip_rule(body("## 8. Discussion")) + "\n\n" + implications))
    out.append(section("## 9. Limitations", strip_rule(body("## 9. Limitations"))))
    concl = (
        "We set out to let predicted uncertainty choose between a visible and a thermal camera at sea and found, "
        "under pre-registered tests, that it cannot: the variance a detector predicts for its own boxes is "
        "informative about those boxes and silent about which sensor has failed. The system that works by day is "
        "a hard veto on image statistics with the union of the survivors, and its night behaviour shows how such a "
        "rule fails when the detector beneath it changes. On the Pohang Canal data, with simulated adverse weather "
        "and a single night run, these results are development evidence that one untouched run cannot extend. "
        "More held-out runs, field-collected adverse conditions and a co-registered camera rig are the next "
        "requirements.")
    out.append(section("## 10. Conclusion", concl))
    appendix.append(section("## Appendix C. Reproducibility and implementation notes",
                            strip_rule(body("## 10. Reproducibility and implementation notes"))))
    out.append("## References\n\n<!-- REFS -->\n\n---\n")
    text = "\n".join(out) + "\n" + "\n".join(appendix)
    p = VEN / "PAPER_JOE.md"
    p.write_text(fix_paths(text, "Appendix C"), encoding="utf-8")
    return p


if __name__ == "__main__":
    for p in (build_tmlr(), build_joe()):
        fill_refs(p)
    for name in ("PAPER_MACVI.md", "PAPER_PBVS.md"):
        fill_refs(VEN / name)
