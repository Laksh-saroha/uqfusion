"""Two-page summary of the project for a non-specialist reader (2026-10-09), built with reportlab.

Every number is copied from PAPER_DRAFT2.md. Uses the Windows Segoe UI fonts.

    py -3.13 scripts/what_we_did_simply_pdf.py docs/what-we-did-simply.pdf
"""
import math
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                ListFlowable, ListItem, PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = sys.argv[1]
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("UI", F + "segoeui.ttf"))
pdfmetrics.registerFont(TTFont("UI-B", F + "segoeuib.ttf"))
pdfmetrics.registerFont(TTFont("UI-I", F + "segoeuii.ttf"))
pdfmetrics.registerFont(TTFont("UI-SB", F + "seguisb.ttf"))
pdfmetrics.registerFontFamily("UI", normal="UI", bold="UI-B", italic="UI-I", boldItalic="UI-B")

INK = HexColor("#1d2433")
MUTED = HexColor("#5b6475")
ACCENT = HexColor("#1f5fa8")
TINT = HexColor("#eaf1fa")
VIS = HexColor("#d98a1c")
IR = HexColor("#b8324a")
GOOD = HexColor("#2e7d4f")
BAD = HexColor("#b03a2e")
GRID = HexColor("#d5dbe5")

body = ParagraphStyle("body", fontName="UI", fontSize=10.4, leading=14.2, textColor=INK, spaceAfter=3)
small = ParagraphStyle("small", parent=body, fontSize=9.2, leading=12.6, textColor=MUTED)
h = ParagraphStyle("h", fontName="UI-SB", fontSize=13, leading=16, textColor=ACCENT,
                   spaceBefore=8, spaceAfter=3)
h2 = ParagraphStyle("h2", fontName="UI-B", fontSize=10.8, leading=14.5, textColor=INK,
                    spaceBefore=5, spaceAfter=2)
title = ParagraphStyle("t", fontName="UI-B", fontSize=24, leading=28, textColor=INK)
sub = ParagraphStyle("s", fontName="UI", fontSize=11.5, leading=15, textColor=MUTED, spaceBefore=2)
item = ParagraphStyle("item", parent=body, spaceAfter=2)
cell = ParagraphStyle("cell", parent=body, fontSize=9.6, leading=12.8, spaceAfter=0)
cellh = ParagraphStyle("cellh", parent=cell, fontName="UI-SB", textColor=white)

W, H = A4
M = 44


def bullets(texts, style=item):
    return ListFlowable([ListItem(Paragraph(t, style), leftIndent=14) for t in texts],
                        bulletType="bullet", bulletFontName="UI", bulletFontSize=8,
                        bulletColor=ACCENT, leftIndent=14, bulletOffsetY=-1)


def node(d, x, y, w, hgt, lines, fill, stroke, color=INK, sub_color=MUTED):
    d.add(Rect(x, y, w, hgt, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    lead = 10.5
    top = y + hgt / 2 + (len(lines) - 1) * lead / 2 - 3
    for i, t in enumerate(lines):
        d.add(String(x + w / 2, top - i * lead, t, textAnchor="middle",
                     fontName="UI-SB" if i == 0 else "UI", fontSize=8.6 if i == 0 else 7.8,
                     fillColor=color if i == 0 else sub_color))


def arrow(d, x1, y1, x2, y2, color=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.1))
    a, s = math.atan2(y2 - y1, x2 - x1), 5
    p1 = (x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45))
    p2 = (x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.5))


def diagram():
    dw, dh = W - 2 * M, 92
    d = Drawing(dw, dh)
    bh = 34
    y_top, y_bot = dh - bh - 4, 6
    cw, ex, ew, dx, dwid, px, pw = 88, 104, 100, 222, 124, 364, 72
    ox = px + pw + 14
    node(d, 0, y_top, cw, bh, ["Visible camera", "colour video"], HexColor("#fdf3e3"), VIS)
    node(d, 0, y_bot, cw, bh, ["Thermal camera", "infrared video"], HexColor("#fbe8ec"), IR)
    node(d, ex, y_top, ew, bh, ["Detector", "boxes + uncertainty"], white, VIS)
    node(d, ex, y_bot, ew, bh, ["Detector", "boxes + uncertainty"], white, IR)
    arrow(d, cw, y_top + bh / 2, ex - 1, y_top + bh / 2)
    arrow(d, cw, y_bot + bh / 2, ex - 1, y_bot + bh / 2)
    node(d, dx, (dh - 58) / 2, dwid, 58, ["Decision layer", "chooses which camera’s", "detections to keep"],
         TINT, ACCENT, color=ACCENT)
    arrow(d, ex + ew, y_top + bh / 2, dx - 1, dh / 2 + 10)
    arrow(d, ex + ew, y_bot + bh / 2, dx - 1, dh / 2 - 10)
    node(d, px, (dh - bh) / 2, pw, bh, ["Pooling", "union of survivors"], white, ACCENT)
    arrow(d, dx + dwid, dh / 2, px - 1, dh / 2)
    node(d, ox, (dh - bh) / 2, dw - ox, bh, ["Output", "ships and buoys"], INK, INK,
         color=white, sub_color=HexColor("#c9d2e0"))
    arrow(d, px + pw, dh / 2, ox - 1, dh / 2)
    return d


def boxed(flows, fill=TINT, pad=9):
    t = Table([[flows]], colWidths=[W - 2 * M])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill),
                           ("LEFTPADDING", (0, 0), (-1, -1), pad + 3),
                           ("RIGHTPADDING", (0, 0), (-1, -1), pad + 3),
                           ("TOPPADDING", (0, 0), (-1, -1), pad - 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 2),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT)]))
    return t


def tried_table():
    no = "<font color='#b03a2e'><b>{}</b></font>"
    rows = [
        ["Approach", "Outcome", "Detail"],
        ["Matching the same object across both cameras", no.format("Not useful"),
         "The cameras are not precisely aligned, so only 0.05% of visible-camera boxes found a "
         "thermal partner. In practice, fusion amounts to pooling both cameras’ detections."],
        ["Selecting the best detector architecture (31 designs, 93 training runs)",
         "<font color='#5b6475'><b>No clear winner</b></font>",
         "The top three were statistically indistinguishable."],
        ["Refinements: uncertainty-weighted box merging, score recalibration, softer duplicate "
         "removal", no.format("Not adopted"),
         "None produced a clear improvement."],
    ]
    data = [[Paragraph(c, cellh if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[190, 86, W - 2 * M - 276])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 1), (-1, -1), 0.6, GRID),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f6f8fb")]),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]))
    return t


story = [
    Paragraph("What We Did, Simply", title),
    Paragraph("Uncertainty-aware visible–infrared ship detection: what we tested, what held up, "
              "and what did not", sub),
    Paragraph("Laksh Saroha · UG Research Fellowship, Thapar Institute · 9 October 2026", small),
    Spacer(1, 10),
    boxed([Paragraph("<b>Summary</b>", body), bullets([
        "We built a ship and buoy detector that combines a visible camera with a thermal infrared camera.",
        "The central question was whether each detector’s predicted uncertainty could decide, frame "
        "by frame, which camera to trust.",
        "The uncertainty proved informative, but no way of using it made the combined system more accurate.",
        "A simple image-brightness rule worked better in daylight. At night it failed once the detector "
        "was retrained on corrected labels.",
    ])]),

    Paragraph("The problem", h),
    Paragraph("Maritime systems often pair a visible camera with a thermal infrared camera, because the two "
              "fail under different conditions. Visible imagery degrades in fog, glare and darkness. Thermal "
              "imagery loses vessels when they reach the same temperature as the surrounding water.", body),
    Paragraph("Neither failure is signalled. A degraded detector simply returns fewer detections, and a "
              "combining rule cannot tell that apart from an empty scene.", body),

    Paragraph("The question", h),
    Paragraph("Modern detectors can report, alongside each bounding box, an estimate of how uncertain that "
              "box is. We asked whether this predicted uncertainty is reliable enough to decide, frame by "
              "frame, how much weight each camera should receive.", body),

    Paragraph("What we built", h),
    bullets([
        "<b>Data:</b> the Pohang Canal dataset, real video recorded from a vessel in South Korea. It holds "
        "158,319 images and 1.18 million annotated ships and buoys across five recordings.",
        "<b>Detectors:</b> one YOLO detector per camera, each extended with an output that estimates the "
        "uncertainty of every box.",
        "<b>Fusion:</b> a decision layer that selects which camera’s detections to keep, then pools "
        "the survivors.",
    ]),
    Spacer(1, 6),
    diagram(),

    Paragraph("How we kept the results honest", h),
    bullets([
        "<b>Pre-registration:</b> the success criteria for each major test were written down before the "
        "test was run.",
        "<b>A held-out recording:</b> one of the five recordings was kept untouched and evaluated exactly "
        "once, at the end.",
        "<b>Measured noise:</b> we measured how much scores vary by chance, so small differences were not "
        "mistaken for improvements.",
    ]),

    Paragraph("Limitations", h),
    Paragraph("Fog, glare and low light were simulated on clean images rather than recorded. All data comes "
              "from a single location, and only one recording was made at night.", body),

    PageBreak(),
    Paragraph("What we found", h),
    Paragraph("Accuracy is reported as average precision (<b>AP</b>), the standard detection score, from 0 "
              "(nothing found) to 1 (perfect).", small),

    Paragraph("1. Uncertainty is informative, but it does not improve the system", h2),
    Paragraph("The test compared the real uncertainty values with the same values shuffled randomly across "
              "boxes. If the real values carry information, they should beat the shuffled ones.", body),
    bullets([
        "As a weight on box positions, real uncertainty beat the shuffled control in 0 of 4 conditions.",
        "As a way to re-rank which boxes to keep, it beat the control in 3 of 4, meeting the "
        "pre-registered threshold.",
        "However, a system that ignored uncertainty entirely scored higher in 6 of 8 conditions.",
    ]),
    Paragraph("A further variant learned to combine uncertainty with the detector’s confidence. It gave "
              "a small gain on all five retrained detectors, but only two cleared the margin fixed in "
              "advance, so we do not claim it.", body),

    Paragraph("2. A simple rule worked, until the detector changed", h2),
    Paragraph("The rule: <i>at night, if the image is dark or foggy, discard the visible camera.</i>", body),
    bullets([
        "In daylight it outperformed the visible camera alone in all 4 conditions (+0.006 to +0.011 AP).",
        "At night it appeared justified, because the visible detector scored 0 at night.",
    ]),
    Paragraph("That zero turned out to be an artefact. A label-filtering step had a bug and removed many "
              "ship annotations from the night training images, so the visible detector never learned to "
              "detect at night. After the labels were restored and the detector retrained, it scores "
              "0.25 AP at night. The unchanged rule still discards it, and night accuracy falls to the "
              "thermal camera’s 0.07.", body),
    Paragraph("<b>Lesson:</b> a rule for choosing between cameras is really a statement about the detectors "
              "behind them, and it must be re-checked whenever a detector is retrained.", body),

    Paragraph("3. Many small improvements were within the noise", h2),
    Paragraph("Differences below roughly 0.003 AP cannot be distinguished from chance. Consecutive video "
              "frames are nearly identical, so standard error bars were at least 1.9 times too narrow. "
              "Re-examining 74 earlier findings with these corrections reclassified 20 as inconclusive.",
              body),

    Paragraph("4. The held-out test", h2),
    Paragraph("On the untouched recording, the system scored 0.27 AP, which is 0.02 below its score on the "
              "development data. The uncertainty around that gap is too wide to say whether it is real.",
              body),

    KeepTogether([Paragraph("Other approaches we tested", h), tried_table()]),

    Paragraph("Where it stands", h),
    bullets([
        "The work is written up as a research paper that reports the negative results alongside the "
        "positive ones.",
        "Title: <i>Predicted Uncertainty Is Informative but Does Not Improve Visible–Infrared "
        "Maritime Detection</i>.",
        "Target venue: Transactions on Machine Learning Research (TMLR). The formatted LaTeX version "
        "compiles.",
        "Remaining: decide whether to shorten the main text, add acknowledgements, and submit.",
    ]),
]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=40, bottomMargin=34,
                        title="What We Did, Simply", author="Laksh Saroha",
                        subject="Summary of the visible-infrared maritime detection project")
doc.build(story)
