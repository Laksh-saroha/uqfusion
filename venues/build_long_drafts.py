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
         "57.0 FPS per detector against 30.7 for yolo26x (detector `predict()` only). The full table (Table 1), "
         "the Phase 1 selection record (Table 1b), the disclosures and the IR architecture ladder are in "
         "Appendix {C}.")


def condensed(text, **letters):
    return text.format(**letters)


# ---------------------------------------------------------------- TMLR
def build_tmlr():
    L = dict(A="A", B="B", C="C")
    title = D2.split("\n", 1)[0]  # Draft 2's title line
    note = ("> Venue draft for TMLR, derived from `PAPER_DRAFT2.md` at commit `d9260ef` (2026-10-09) by "
            "`venues/build_long_drafts.py`. TMLR has no hard page limit, but a main body over 12 pages gets a "
            "longer review; condensed subsections keep their headings and move their full text to the "
            "appendices. Every number is from Draft 2. See `venues/README.md`.")
    abstract = (
        "We pre-registered a test of whether predicted uncertainty should decide how a two-stream "
        "visible–infrared maritime detector fuses its streams, and we report the answer together with the "
        "evaluation machinery that produced it. On the Pohang Canal dataset with PoLaRIS boxes, independent "
        "YOLO26 detectors with single-pass Gaussian variance heads feed a decision layer. The registered "
        "comparison is real predicted σ against the same σ shuffled onto the wrong boxes, on ship AP, with a "
        "0.0060 floor and block-bootstrap intervals. As a coordinate weight σ passes on zero of four conditions; "
        "as a score re-ranker it passes on three, so it is informative. It is not useful: the signal lies within "
        "each detector's own boxes, the σ-scored system is below the system with no σ on six of eight cells, and "
        "a learned within-detector variant failed its pre-registered replication on five retrained detectors "
        "(positive on all five, beyond the floor on two). The recorded runs had scored a ship-and-buoy macro "
        "instead of the registered metric, and the macro read NULL on both paths; we report the deviation and "
        "argue that the scored quantity belongs in the registration. The mechanism that shipped instead, an "
        "image-statistic veto with union aggregation, beats the visible stream by day on five retrained systems "
        "(+0.0059 to +0.0107 AP) but discards a working visible stream at night (−0.1847) once a label artifact "
        "behind its night arm was corrected. The evaluation protocol (paired noise floors, dependence-aware "
        "intervals that widen frame-level ones 1.9–1.99×, magnitude floors, a declared class set and one logged "
        "look at an untouched run) moved 20 of 74 earlier findings to indeterminate, and the held-out look "
        "returns a gap of +0.0216 that excludes neither zero nor 0.05.")
    out = [title, "", "**Anonymous authors** (TMLR review is double-blind; restore the author block for the camera-ready).",
           "", note, "", "---", "", "## Abstract", "", abstract, "", "---", ""]
    appendix = []

    def keep(h):
        out.append(section(h, strip_rule(body(h))))

    def cond(h, text, app_head):
        out.append(section(h, text))
        appendix.append(section(app_head, strip_rule(body(h))))

    out.append(section("## 1. Introduction", strip_rule(body("## 1. Introduction"))))
    for h in ["## 2. Related work", "### 2.1 Uncertainty in single-stage detectors",
              "### 2.2 Visible–infrared fusion with uncertainty", "### 2.3 Maritime detectors and datasets",
              "### 2.4 Comparison axes"]:
        keep(h) if body(h).strip() else out.append(h + "\n")
    out.append("## 3. Dataset\n")
    keep("### 3.1 Pohang Canal and PoLaRIS")
    appendix.append("## Appendix A. Dataset details (full text of §3.2–§3.5)\n")
    cond("### 3.2 Verified counts", condensed(C_3_2, **L), "### A.1 Verified counts")
    cond("### 3.3 Preprocessing", condensed(C_3_3, **L), "### A.2 Preprocessing")
    cond("### 3.4 Splits", condensed(C_3_4, **L), "### A.3 Splits")
    cond("### 3.5 The night-box filter and its reversal", condensed(C_3_5, **L),
         "### A.4 The night-box filter and its reversal")
    keep("### 3.6 Holdout and contamination")
    out.append(section("## 4. Method", strip_rule(body("## 4. Method"))))
    keep("### 4.1 As designed")
    keep("### 4.2 As shipped (preset `crossmodal26m`)")
    out.append("## 5. Experimental protocol and statistics\n")
    for h in ["### 5.1 Benchmark cells and substrates", "### 5.2 Tune and test discipline", "### 5.3 Noise floor",
              "### 5.4 Dependence-aware intervals", "### 5.5 AP convention"]:
        keep(h)
    appendix.append("## Appendix B. Metric contracts, identity checks and power (full text of §5.6 and §5.8)\n")
    cond("### 5.6 Metric contracts", condensed(C_5_6, **L), "### B.1 Metric contracts")
    keep("### 5.7 Pre-registrations and decision rules")
    cond("### 5.8 Identity checks and power", condensed(C_5_8, **L), "### B.2 Identity checks and power")
    out.append("## 6. Results\n")
    appendix.append("## Appendix C. Backbone benchmark (full text of §6.1)\n")
    cond("### 6.1 Backbone benchmark is a negative result", condensed(C_6_1, **L), "### C.1 Tables 1 and 1b")

    # 6.2: move the day-only rationale, the amendment and the checkpoint-selection paragraph + Figure 4
    kept, moved = move("### 6.2 Per-modality uncertainty calibration",
                       ["**Why day-only is primary.**", "One registered rule was amended",
                        "We do not rank uncertainty methods", "![checkpoint_selection]", "**Figure 4."])
    kept += ("\n\nDay-only is the primary basis because the registered screen placed night in its SUSPECT band on "
             "both streams. That rationale, one registered rule amended after it fired, and why the arms are not "
             "ranked on mAP (Figure 4) are in Appendix D.")
    out.append(section("### 6.2 Per-modality uncertainty calibration", kept))
    appendix.append(section("## Appendix D. Calibration: basis, amendment and checkpoint selection (from §6.2)", moved))

    # 6.3: the gate history goes to Appendix E; Table 3b stays
    kept, moved = move("### 6.3 Fusion robustness of the sensor-selection baseline",
                       ["The decision layer in §4.2 is the sixth rewrite.", "**Table 3a.", "| Stage (date)",
                        "Ship AP; each gap is against", "Under the shipped preset on the pre-restore",
                        "Three findings from the rewrite history", "![gate_history]", "**Figure 5."])
    intro = ("The decision layer of §4.2 is the sixth rewrite of the gate; each rewrite was forced by a measured "
             "failure of the previous one. The history (Table 3a, Figure 5), the pre-restore measurements behind "
             "it and its three general lessons are in Appendix E; two of the lessons recur in §8. Here we "
             "re-measure the shipped rule on the five Phase 3 systems.")
    out.append(section("### 6.3 Fusion robustness of the sensor-selection baseline", intro + "\n\n" + kept))
    appendix.append(section("## Appendix E. Gate rewrite history (from §6.3)", moved))

    keep("### 6.4 Uncertainty is informative but does not improve the fusion (R-D1)")
    keep("### 6.5 Relaxing correspondence does not rescue the coordinate path (Stage 1)")
    keep("### 6.6 Night visible blindness was a label artifact")
    c67 = ("The night vote trusts IR, which was uncorrupted in every benchmark cell, so we attacked the frozen rule "
           "`ir_p05 > 41.5` with six IR hazards at three severities (Appendix F, Table 6). A false night on a clear "
           "day vetoes a working VIS stream. The raw rule misreads up to 94.8 percent of clear days as night under "
           "IR fog and 19–27 percent under IR glare. Two votes, an IR self-check, the multivariate health score and "
           "an authority bound took the false-night rate on 19 IR-corruption arms to 0 percent at zero benchmark "
           "cost; that figure is in-sample (§9), and the both-degraded worst-case false-veto rate fell from 24 "
           "percent to 1.3 percent. An abstain signal was demoted to an advisory flag after it prevented zero bad "
           "vetoes and lost 2,095 correct ones.")
    cond("### 6.7 Is the IR night switch safe when IR is corrupted?", c67,
         "## Appendix F. IR night-switch safety (full text of §6.7)")
    keep("### 6.8 Levers that are inert or negative")
    out.append(section("## 7. Held-out evaluation: the single pohang04 look",
                       strip_rule(body("## 7. Held-out evaluation: the single pohang04 look"))))
    keep("### 7.1 Protocol (fixed before the look)")
    keep("### 7.2 Result")
    keep("## 8. Discussion")
    lim = [
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
        "The full list of 24 items is in Appendix G.",
    ]
    out.append(section("## 9. Limitations", "\n".join(lim)))
    concl = (
        "A pre-registered test asked whether predicted uncertainty should decide how two sensors' detections are "
        "fused. It should not: σ carries information about a detector's own boxes and none about which sensor to "
        "believe, and spending it at the registered strength costs accuracy. Two methodological points generalize "
        "beyond this system. A shuffled control establishes that a signal exists; only the comparison against no "
        "signal establishes that using it helps, and a registration should name both. And the scored quantity is "
        "part of the registration: scoring a macro that included a class one stream cannot detect reversed one of "
        "two verdicts. The sensor-selection rule that shipped instead shows the cost of not re-pricing a decision "
        "rule when the detector beneath it changes.")
    out.append(section("## 10. Conclusion", concl))
    appendix.append(section("## Appendix G. Limitations in full (from §9)", strip_rule(body("## 9. Limitations"))))
    appendix.append(section("## Appendix H. Reproducibility and implementation notes",
                            strip_rule(body("## 10. Reproducibility and implementation notes"))))
    out.append("## References\n\n<!-- REFS -->\n\n---\n")
    text = "\n".join(out) + "\n" + "\n".join(appendix)
    # double-blind: the public repository link identifies the author
    text = text.replace("(github.com/Laksh-saroha/uqfusion, branch `fusion-uq-phase3`)",
                        "(an anonymized repository, linked for review)")
    assert not re.search(r"Saroha|Laksh-saroha|Thapar|Mandia", text, re.I)
    p = VEN / "PAPER_TMLR.md"
    p.write_text(fix_paths(text, "Appendix H"), encoding="utf-8")
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
                       ["We do not rank uncertainty methods", "![checkpoint_selection]", "**Figure 4."])
    kept += ("\n\nWhy the arms are not ranked on mAP, including the checkpoint-selection analysis of Figure 4, is in "
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
