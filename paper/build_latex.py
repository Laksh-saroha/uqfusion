"""Convert PAPER.md into a TMLR LaTeX submission in paper/, or with --ieee into an IEEE journal submission in
paper/ieee/.

Writes main.tex, references.bib (from the paper's own reference list) and figures/ (copies of docs/figures/*.pdf).
The TMLR style files (tmlr.sty, tmlr.bst, fancyhdr.sty, natbib.sty) are not in the repository: compile on
Overleaf from the TMLR template, or drop the files from the TMLR style repository into paper/.

The IEEE version (IEEEtran, two columns, numbered citations; IEEEtran.cls and IEEEtranN.bst ship with TeX Live)
takes its author, abstract and index terms from paper/ieee/front.md, drops the TMLR broader-impact section,
restores the code link, and moves Appendices A-K into a separate supplement.tex. The two documents cross-reference
each other through the xr package, so compile main, then supplement, then main again.

Tables and figures are renumbered in order of appearance; every "Table X", "Figure X", "§x.y" and "Appendix X.n"
in the text becomes a \\ref, and the build fails if any target is undefined. Re-run after the Markdown changes:

    py -3.13 paper/build_latex.py
    py -3.13 paper/build_latex.py --ieee
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IEEE = "--ieee" in sys.argv[1:]
OUT = ROOT / "paper" / ("ieee" if IEEE else "")
MD = (ROOT / "PAPER.md").read_text(encoding="utf-8").replace("\r\n", "\n")
CODE_ANON = "in a public repository, withheld from this review copy for anonymity and linked in the camera-ready"
CODE_OPEN = "in the public repository (github.com/Laksh-saroha/uqfusion, branch `fusion-uq-phase3`)"
if IEEE:  # IEEE review is single-blind: the code link stays, and TMLR's broader-impact section goes
    assert MD.count(CODE_ANON) == 2  # Appendix J and the availability statement
    MD = MD.replace(CODE_ANON, CODE_OPEN)
    i, j = MD.index("\n## 9. Broader impact statement\n"), MD.index("\n## References\n")
    MD = MD[:i] + "\n" + MD[j:]

FIG_WIDTH = {  # fraction of \linewidth, from each figure's aspect ratio
    "fig_n4_gate": 1.0, "fig_tta_heldout": 1.0, "fig_decision_layer": 1.0, "fig_phase3_cells": 0.7,
    "fig_night_restore": 0.6, "fig_detections": 1.0, "fig_uq_calibration": 0.9, "fig_throughput": 1.0, "fig_tta_detections": 1.0,
}
TABLE_NOTES = ("†", "Ship AP; each gap is against")  # paragraphs that annotate the table just above them
STATE = {"two_col": False}  # True while converting the IEEE main body, whose floats choose one or two columns

UNI = [  # applied after escaping, longest first
    ("10⁻⁴", "10$^{-4}$"), ("R̄", r"$\bar{R}$"), ("−", "$-$"), ("±", r"$\pm$"), ("×", r"$\times$"),
    ("σ", r"$\sigma$"), ("α", r"$\alpha$"), ("λ", r"$\lambda$"), ("γ", r"$\gamma$"), ("μ", r"$\mu$"),
    ("τ", r"$\tau$"), ("ρ", r"$\rho$"), ("Σ", r"$\Sigma$"), ("Δ", r"$\Delta$"), ("≥", r"$\geq$"), ("∧", r"$\wedge$"),
    ("≤", r"$\leq$"), ("≈", r"$\approx$"), ("·", r"$\cdot$"), ("÷", r"$\div$"), ("²", r"$^{2}$"),
    ("¼", r"$\frac{1}{4}$"), ("′", r"$'$"), ("—", "---"), ("–", "--"), ("†", r"\textdagger{}"),
    ("‡", r"\textdaggerdbl{}"), ("§", r"\S{}"),
]
MATH = {  # the BN-mix formulas (§4.3) and the Appendix D.1 formulas, which the Markdown carries as code spans
    "μ = (N μ_src + n μ_b) / (N + n)": r"$\mu = (N\mu_{\mathrm{src}} + n\mu_b)/(N+n)$",
    "σ² = (N σ²_src + n σ²_b) / (N + n)": r"$\sigma^2 = (N\sigma^2_{\mathrm{src}} + n\sigma^2_b)/(N+n)$",
    "μ_b": r"$\mu_b$", "σ²_b": r"$\sigma^2_b$",
    "u_i = ¼ Σ_t σ_{i,t} / s_{i,t}": r"$u_i = \frac{1}{4}\sum_t \sigma_{i,t}/s_{i,t}$",
    "U_box,m = Σ_i c_i u_i / Σ_i c_i": r"$U_{\mathrm{box},m} = \sum_i c_i u_i / \sum_i c_i$",
    "O_m = sigmoid((d_m − μ_d) / τ)": r"$O_m = \mathrm{sigmoid}((d_m - \mu_d)/\tau)$",
    "R_m = r_frame,m · r_box,m": r"$R_m = r_{\mathrm{frame},m} \cdot r_{\mathrm{box},m}$",
    "r_box,m = exp(−λ U_box,m)": r"$r_{\mathrm{box},m} = \exp(-\lambda U_{\mathrm{box},m})$",
    "r_frame,m = 1 − O_m": r"$r_{\mathrm{frame},m} = 1 - O_m$",
    "r_frame,m": r"$r_{\mathrm{frame},m}$",
    "R̄_m(t) = α R_m(t) + (1 − α) R̄_m(t−1)": r"$\bar{R}_m(t) = \alpha R_m(t) + (1 - \alpha)\,\bar{R}_m(t-1)$",
    "w_m = R̄_m / (R̄_vis + R̄_ir)": r"$w_m = \bar{R}_m / (\bar{R}_{\mathrm{vis}} + \bar{R}_{\mathrm{ir}})$",
}
BIB_ACCENTS = [("ü", r'{\"u}'), ("Ç", r"{\c{C}}"), ("ç", r"{\c{c}}"), ("Ö", r'{\"O}'), ("á", r"{\'a}"),
               ("ø", r"{\o}"), ("č", r"{\v{c}}"), ("š", r"{\v{s}}"), ("Ž", r"{\v{Z}}"), ("–", "--")]


# ---------------------------------------------------------------- references -> BibTeX
def split_entries(md):
    i = md.index("## References\n")
    j = md.index("\n---\n", i)
    return [e.strip() for e in md[i + len("## References\n"):j].split("\n\n") if e.strip()]


def parse_ref(e):
    m = re.match(r"(?P<auth>.+?) \((?P<year>\d{4})\)\. (?P<rest>.*)$", e, re.S)
    auth, year, rest = m["auth"], m["year"], m["rest"]
    parts = re.split(r",\s+(?:and\s+)?", auth.rstrip("."))
    parts = [p[4:] if p.startswith("and ") else p for p in parts]
    assert len(parts) % 2 == 0, e
    pairs = [(parts[k], parts[k + 1] if re.search(r"[a-z]$", parts[k + 1]) else parts[k + 1].rstrip(".") + ".")
             for k in range(0, len(parts), 2)]
    f = {"author": " and ".join(f"{s}, {g}" for s, g in pairs), "year": year}
    if rest.startswith("*"):  # software: *Title* (note). url. licence.
        t = re.match(r"\*(.+?)\* \((.+?)\)\. (\S+)\. (.+)$", rest)
        f.update(title=t[1], note=f"{t[2][0].upper()}{t[2][1:]}. {t[4].rstrip('.')}",
                 howpublished=r"\url{" + t[3] + "}")
        typ = "misc"
    else:
        t = re.match(r"(.+?)\. (In \*|\*|arXiv)(.*)$", rest, re.S)
        f["title"], tail = t[1], t[2] + t[3]
        if tail.startswith("arXiv"):
            typ, f["note"] = "misc", tail.rstrip(".")
        else:
            notes = []
            for pat, key in [(r"doi:(\S+?)\.?(?=\s|$)", "doi"), (r"pp\. ([\d–]+)", "pages")]:
                mm = re.search(pat, tail)
                if mm:
                    f[key] = mm[1]
                    tail = tail.replace(mm[0], "")
            for mm in re.finditer(r"(?:Preprint )?arXiv:\S+?(?=\.?(?:\s|$))", tail):
                notes.append(mm[0])
            tail = re.sub(r"(?:Preprint )?arXiv:\S+?(?=\.?(?:\s|$))\.?", "", tail)
            if tail.startswith("In *"):
                typ = "inproceedings"
                b = re.match(r"In \*(.+?)\*(.*)$", tail, re.S)
                book, extra = b[1], b[2]
                pm = re.match(r"\s*(\(.+?\))", extra)
                if pm:
                    book += " " + pm[1]
                    extra = extra[pm.end():]
                f["booktitle"] = book
                if "Lecture Notes in Computer Science" in extra:
                    f["series"] = "Lecture Notes in Computer Science"
                    extra = extra.replace("Lecture Notes in Computer Science", "")
                pm = re.search(r"PMLR (\d+)", extra)
                if pm:
                    f["series"], f["volume"] = "Proceedings of Machine Learning Research", pm[1]
                    extra = extra.replace(pm[0], "")
            else:
                typ = "article"
                a = re.match(r"\*(.+?)\*, (\d+)(?:\(([^)]+)\))?, ([\d–]+)(.*)$", tail, re.S)
                f.update(journal=a[1], volume=a[2], pages=a[4])
                if a[3]:
                    f["number"] = a[3]
                extra = a[5]
            leftover = re.sub(r"[\s.,]+", " ", extra).strip()
            assert not leftover, (e, leftover)
            if notes:
                f["note"] = "; ".join(notes)
    first_word = re.sub(r"[^a-z0-9]", "", f["title"].split()[0].lower()) or "x"
    sur = pairs[0][0]
    key = re.sub(r"[^a-z]", "", sur.lower().translate(str.maketrans("üçöáøčšž", "ucoaocsz"))) + year + first_word
    return dict(key=key, type=typ, fields=f, surnames=[s for s, _ in pairs], given=pairs[0][1], year=year)


def bib_value(k, v):
    if k in ("howpublished", "doi"):
        return v
    for a, b in BIB_ACCENTS:
        v = v.replace(a, b)
    v = v.replace("&", r"\&")
    v = re.sub(r"\*(.+?)\*", r"\\emph{\1}", v)  # Markdown italics, e.g. a journal named in a note
    return "{" + v + "}" if k == "title" else v


def write_bib(refs):
    out = []
    for r in refs:
        body = ",\n".join(f"  {k} = {{{bib_value(k, v)}}}" for k, v in r["fields"].items())
        out.append(f"@{r['type']}{{{r['key']},\n{body}\n}}")
    (OUT / "references.bib").write_text("\n\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- inline conversion
def code(s, breakable=True):
    s = escape(s)
    if breakable:
        s = s.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
        if IEEE:  # narrower columns: dated file names also break at their hyphens
            s = s.replace("-", r"-\allowbreak{}")
    return r"\texttt{" + s + "}"


def escape(s):
    out = []
    for ch in s:
        out.append({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(ch, ch))
    s = "".join(out)
    for a, b in UNI:
        s = s.replace(a, b)
    return s.replace(r"\textasciicircum{}$\alpha$", r"$^{\alpha}$")  # (median σ / σ)^α


LABEL = r"(?:\d+[a-z]?|[RLSD])\b"


class Converter:
    def __init__(self, refs):
        self.refs = refs
        self.used_labels = set()   # ("tab"|"fig"|"sec"|"app", label) referenced
        self.cite_pats = []
        by = {}
        for r in refs:
            by.setdefault((r["surnames"][0], r["year"]), []).append(r)
        for r in refs:
            s, y = r["surnames"][0], r["year"]
            ini = re.escape(r["given"]) + r"\s" if len(by[(s, y)]) > 1 else r"(?:[A-Z][\w.-]*\s)?"
            if len(r["surnames"]) == 2:
                who = re.escape(s) + r" and " + re.escape(r["surnames"][1])
            elif len(r["surnames"]) > 2:
                who = re.escape(s) + r" et al\."
            else:
                who = re.escape(s)
            self.cite_pats.append((re.compile(ini + who + r" \(" + y + r"\)"), r"\citet{" + r["key"] + "}"))
            self.cite_pats.append((re.compile(ini + who + r", " + y + r"(?![\d])"), r"\citealp{" + r["key"] + "}"))

    def inline(self, s, heading=False):
        toks = []

        def T(x):
            toks.append(x)
            return f"\x00{len(toks) - 1}\x00"

        s = re.sub(r"`([^`]+)`", lambda m: T(MATH.get(m[1]) or code(m[1], breakable=not heading)), s)
        for pat, rep in self.cite_pats:
            s = pat.sub(lambda m, rep=rep: T(rep), s)
        left = re.findall(r"[A-Z][\w-]+(?: et al\.| and [A-Z][\w-]+)?,? \(?(?:19|20)\d\d\b", s)
        assert not [x for x in left if any(x.startswith(r["surnames"][0]) for r in self.refs)], left
        if heading:
            s = re.sub(r"§(\d+(?:\.\d+)?)", lambda m: T(r"\S" + m[1]), s)
        else:
            def sec(m):
                self.used_labels.add(("sec", m[1]))
                return T((r"Section~" if IEEE else r"\S") + r"\ref{sec:" + m[1] + "}")
            s = re.sub(r"§(\d+(?:\.\d+)?)", sec, s)

            def app(m):
                self.used_labels.add(("app", m[1]))
                return T(r"Appendix~\ref{app:" + m[1] + "}")
            s = re.sub(r"Appendix ([A-M](?:\.\d+)?)\b", app, s)

            def tabfig(m):
                kind = "tab" if m[1].startswith("Table") else "fig"
                labs = m[2]

                def one(mm):
                    self.used_labels.add((kind, mm[0]))
                    return r"\ref{" + kind + ":" + mm[0] + "}"
                word = {"Figure": "Fig.", "Figures": "Figs."}.get(m[1], m[1]) if IEEE else m[1]
                return T(word + "~" + re.sub(LABEL, one, labs))
            s = re.sub(r"\b(Tables?|Figures?) (" + LABEL + r"(?:,? (?:and )?" + LABEL + r")*)", tabfig, s)
        s = escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s, flags=re.S)
        s = re.sub(r"(?<![\w*\\])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", r"\\emph{\1}", s, flags=re.S)
        assert "*" not in s, s[:200]
        assert s.count('"') % 2 == 0, s[:200]
        s = re.sub(r'"([^"]*)"', r"``\1''", s)
        while "\x00" in s:
            s = re.sub(r"\x00(\d+)\x00", lambda m: toks[int(m[1])], s)
        if IEEE:  # numbered citations: "(A, 2020; B, 2021)" -> "[1], [2]", an inline "A, 2020" -> "[1]"
            s = s.replace(r"\citealp{", r"\cite{")
            s = re.sub(r"\((\\cite\{[^}]+\}(?:; \\cite\{[^}]+\})*)\)",
                       lambda m: r"\cite{" + ",".join(re.findall(r"\\cite\{([^}]+)\}", m[1])) + "}", s)
            return s
        return re.sub(r"\(\\citealp\{([^}]+)\}\)", r"\\citep{\1}", s)


# ---------------------------------------------------------------- blocks
def paragraphs(text):
    return [p.strip("\n") for p in re.split(r"\n{2,}", text) if p.strip()]


def is_table(p):
    return all(ln.startswith("|") for ln in p.split("\n"))


def is_list_line(ln):
    return bool(re.match(r"\s*(\* |- |\d+\. )", ln))


def table_tex(cv, p, caption, label, notes):
    rows = [ln.strip().strip("|").split("|") for ln in p.split("\n")]
    head, align, body = rows[0], rows[1], rows[2:]
    n = len(head)
    assert all(len(r) == n for r in body), (label, [len(r) for r in body])
    width = [max(len(r[c].strip()) for r in [head] + body) for c in range(n)]
    total = sum(width)
    size = r"\scriptsize" if n >= 7 and total > 200 else (r"\footnotesize" if n >= 6 or total > 110 else r"\small")
    cols = []
    for c in range(n):
        a = align[c].strip()
        if width[c] > 28 and total > 90:
            cols.append(r">{\raggedright\arraybackslash}X")
        else:
            cols.append("r" if a.endswith(":") and not a.startswith(":") else "l")
    use_x = any("X" in c for c in cols)
    env, spec = ("tabularx", r"{\linewidth}") if use_x else ("tabular", "")
    # In a two-column IEEE page a column holds about 55 characters of \footnotesize; wider tables span both.
    fenv = "table*" if STATE["two_col"] and (total + 2 * n > 55 or use_x) else "table"
    if caption is None:  # an uncaptioned table stays in the text flow
        lines = [r"\begin{center}", size]
        if fenv == "table*":
            lines = [r"\begin{table*}[!t]", r"\centering", size]
    else:
        lines = [rf"\begin{{{fenv}}}[{'!t' if fenv == 'table*' else 'htbp'}]", r"\centering", size,
                 r"\caption{" + caption + "}", r"\label{tab:" + label + "}"]
    if n >= 7:  # narrow columns: tighten the gutters so long words fit
        lines.append(r"\setlength{\tabcolsep}{3pt}")
    box = not use_x  # a plain tabular may be wider than the text; adjustbox scales it down only if it is
    if box:
        lines.append(r"\begin{adjustbox}{max width=\linewidth}")
    lines += [rf"\begin{{{env}}}{spec}{{{''.join(cols)}}}", r"\toprule",
             " & ".join(cv.inline(c.strip()) for c in head) + r" \\", r"\midrule"]
    for r in body:
        cells = [cv.inline(c.strip()) for c in r]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", rf"\end{{{env}}}"]
    if box:
        lines.append(r"\end{adjustbox}")
    for nt in notes:
        lines.append(r"\par\smallskip\raggedright\footnotesize " + cv.inline(nt))
    if caption is None and fenv != "table*":
        lines.append(r"\end{center}")
    else:
        lines.append(rf"\end{{{fenv}}}")
    return "\n".join(lines)


def figure_tex(cv, img, caption, label):
    stem = Path(re.match(r"!\[[^\]]*\]\(([^)]+)\)", img)[1]).stem
    w = FIG_WIDTH[stem]
    env = "figure"
    if STATE["two_col"]:  # a full-width figure spans both columns; a narrower one fills one column
        env, w = ("figure*", w) if w >= 0.9 else ("figure", 1.0)
    return "\n".join([rf"\begin{{{env}}}[{'!t' if env == 'figure*' else 'htbp'}]", r"\centering",
                      rf"\includegraphics[width={w}\linewidth]{{figures/{stem}.pdf}}",
                      r"\caption{" + caption + "}", r"\label{fig:" + label + "}", rf"\end{{{env}}}"])


def list_tex(cv, lines):
    """Markdown lists with one level of nesting. An indented item is nested; an indented line that is not an
    item continues the item above it, or, after a nested list, the outer item."""
    items = []  # [level, env | "cont", text]
    for ln in lines:
        m = re.match(r"(\s*)(\* |- |(\d+)\. )(.*)$", ln)
        if m:
            items.append([0 if not m[1] else 1, "enumerate" if m[3] else "itemize", m[4], int(m[3] or 1)])
            continue
        assert items and ln.startswith("  "), ln
        indent = len(ln) - len(ln.lstrip(" "))
        if items[-1][0] == 1 and indent < 4:
            items.append([0, "cont", ln.strip(), 0])
        else:
            items[-1][2] += " " + ln.strip()
    out, stack = [], []
    for lvl, env, text, num in items:
        if env == "cont":
            while len(stack) > 1:
                out.append(rf"\end{{{stack.pop()}}}")
            out.append(cv.inline(text))
            continue
        while len(stack) > lvl + 1:
            out.append(rf"\end{{{stack.pop()}}}")
        if len(stack) == lvl + 1 and stack[-1] != env:
            out.append(rf"\end{{{stack.pop()}}}")
        while len(stack) < lvl + 1:
            out.append(rf"\begin{{{env}}}")
            stack.append(env)
            if env == "enumerate" and num > 1:  # a list that continues Markdown numbering, e.g. §9 item 25
                out.append(rf"\setcounter{{{'enumi' if lvl == 0 else 'enumii'}}}{{{num - 1}}}")
        item = cv.inline(text)
        out.append(r"\item " + ("{}" + item if item.startswith("[") else item))
    while stack:
        out.append(rf"\end{{{stack.pop()}}}")
    return out


def caption_parts(p, kind):
    m = re.match(r"\*\*" + kind + r" (\w+)\. (.*?)\*\*(.*)$", p, re.S)
    return m[1], "**" + m[2] + "**" + m[3]


def ieee_table_caption(p):
    """IEEE sets a table caption as a short small-caps title: the bold lead becomes the caption and the rest of
    the Markdown caption a note under the table."""
    m = re.match(r"\*\*Table (\w+)\. (.*?)\*\*(.*)$", p, re.S)
    return m[1], m[2].rstrip("."), [m[3].strip()] if m[3].strip() else []


def convert(md, cv):
    title = md.split("\n", 1)[0][2:].strip()
    i_abs = md.index("## Abstract\n")
    abstract = paragraphs(md[i_abs + len("## Abstract\n"):md.index("\n---\n", i_abs)])
    i_ref = md.index("## References\n")
    main = md[md.index("\n---\n", i_abs) + 5:i_ref]
    appx = md[md.index("\n---\n", i_ref) + 5:]
    tex, defined = [], set()
    counters = {"sec": 0, "sub": 0, "app": 0}

    def blocks(text, appendix):
        ps = paragraphs(text)
        k = 0
        while k < len(ps):
            p = ps[k]
            if p == "---" or p.startswith(">") or p.startswith("**Anonymous authors**"):
                k += 1
                continue
            m = re.match(r"^(#{2,3}) (.+)$", p)
            if m and "\n" not in p:
                level, text_ = m[1], m[2]
                if not appendix:
                    mm = re.match(r"(\d+(?:\.\d+)?)\.? (.+)$", text_)
                    num, name = mm[1], mm[2]
                    if level == "##":
                        counters["sec"] += 1
                        counters["sub"] = 0
                        assert num == str(counters["sec"]), (num, counters)
                        tex.append(r"\section{" + cv.inline(name, heading=True) + r"}\label{sec:" + num + "}")
                    else:
                        counters["sub"] += 1
                        assert num == f"{counters['sec']}.{counters['sub']}", (num, counters)
                        tex.append(r"\subsection{" + cv.inline(name, heading=True) + r"}\label{sec:" + num + "}")
                    defined.add(("sec", num))
                else:
                    mm = re.match(r"(?:Appendix )?([A-M](?:\.\d+)?)\.? (.+)$", text_)
                    num, name = mm[1], mm[2]
                    if level == "##":
                        assert num == "ABCDEFGHIJKLM"[counters["app"]], num
                        counters["app"] += 1
                        counters["sub"] = 0
                        tex.append(r"\FloatBarrier")  # keep each appendix's floats inside it
                        tex.append(r"\section{" + cv.inline(name, heading=True) + r"}\label{app:" + num + "}")
                    else:
                        counters["sub"] += 1
                        assert num == f"{'ABCDEFGHIJKLM'[counters['app'] - 1]}.{counters['sub']}", num
                        tex.append(r"\subsection{" + cv.inline(name, heading=True) + r"}\label{app:" + num + "}")
                    defined.add(("app", num))
                k += 1
                continue
            if p.startswith("!["):
                lab, cap = caption_parts(ps[k + 1], "Figure")
                tex.append(figure_tex(cv, p, cv.inline(cap), lab))
                defined.add(("fig", lab))
                k += 2
                continue
            if p.startswith("**Table ") and k + 1 < len(ps) and is_table(ps[k + 1]):
                if IEEE:
                    lab, cap, notes = ieee_table_caption(p)
                else:
                    (lab, cap), notes = caption_parts(p, "Table"), []
                j = k + 2
                while j < len(ps) and ps[j].startswith(TABLE_NOTES):
                    notes.append(ps[j])
                    j += 1
                tex.append(table_tex(cv, ps[k + 1], cv.inline(cap), lab, notes))
                defined.add(("tab", lab))
                k = j
                continue
            if is_table(p):
                tex.append(table_tex(cv, p, None, None, []))
                tex.append("")
                k += 1
                continue
            lines = p.split("\n")
            if any(is_list_line(ln) for ln in lines):
                lead = []
                while lines and not is_list_line(lines[0]):
                    lead.append(lines.pop(0))
                assert lines and is_list_line(lines[0]), p[:120]
                assert all(is_list_line(ln) or ln.startswith("  ") for ln in lines), p[:120]
                if lead:
                    tex.append(cv.inline(" ".join(lead)))
                tex.extend(list_tex(cv, lines))
            else:
                tex.append(cv.inline(" ".join(lines)))
            tex.append("")
            k += 1

    STATE["two_col"] = IEEE
    blocks(main, appendix=False)
    in_main = {f"{kind}:{lab}" for kind, lab in defined}
    tex.append(r"\FloatBarrier")  # main-body floats stay before the references
    if IEEE:  # the appendices go to supplement.tex, which has its own reference list
        tex.append(r"\bibliographystyle{IEEEtranN}")
        tex.append(r"\bibliography{references}")
        main_tex, tex = "\n".join(tex), []
    else:
        tex.append(r"\bibliography{references}")
        tex.append(r"\bibliographystyle{tmlr}")
        tex.append("")
        tex.append(r"\appendix")
    STATE["two_col"] = False
    counters["sub"] = 0
    blocks(appx, appendix=True)
    undefined = sorted(cv.used_labels - defined)
    assert not undefined, undefined
    if IEEE:  # a label defined in the other document is imported by xr under that document's prefix
        def cross(text, here_is_main):
            def one(m):
                other = (m[1] not in in_main) if here_is_main else (m[1] in in_main)
                return r"\ref{" + (("S-" if here_is_main else "M-") if other else "") + m[1] + "}"
            return re.sub(r"\\ref\{([^}]+)\}", one, text)
        return title, [cv.inline(a) for a in abstract], cross(main_tex, True), cross("\n".join(tex), False)
    return title, [cv.inline(a) for a in abstract], "\n".join(tex)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
% Camera-ready: \usepackage[accepted]{tmlr}. Preprint, de-anonymized: \usepackage[preprint]{tmlr}.

\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{placeins}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\usepackage{url}

% Generated by paper/build_latex.py from PAPER.md; edit the Markdown, not this file.

\title{%s}

% Anonymous for review; tmlr.sty hides the author block in submission mode. Restore for the camera-ready.
\author{\name Anonymous authors \email anonymous@example.org \\
      \addr Anonymous institution}

\def\month{MM}
\def\year{YYYY}
\def\openreview{\url{https://openreview.net/forum?id=XXXX}}

\begin{document}

\maketitle

\begin{abstract}
%s
\end{abstract}

%s

\end{document}
"""


IEEE_PACKAGES = r"""\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{adjustbox}
\usepackage{placeins}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage{xr}  % labels and citations of the other document, under its prefix
\emergencystretch=2em  % long file names in \texttt

% Generated by paper/build_latex.py --ieee from PAPER.md and paper/ieee/front.md; edit those, not this file.
"""

IEEE_MAIN = r"""\documentclass[journal]{IEEEtran}
\PACKAGES
\externaldocument[S-]{supplement}  % Appendices A-K; compile main, supplement, main

\begin{document}

\title{\TITLE}

\author{\AUTHOR%
\thanks{\AFFIL}%
\thanks{Appendices A--K, cited in the text, are provided as supplementary material.}}

\markboth{IEEE Journal of Oceanic Engineering}{\SHORT}

\maketitle

\begin{abstract}
\ABSTRACT
\end{abstract}

\begin{IEEEkeywords}
\KEYWORDS
\end{IEEEkeywords}

\BODY

\end{document}
"""

IEEE_SUPP = r"""\documentclass[journal,onecolumn]{IEEEtran}
\PACKAGES
\externaldocument[M-]{main}  % section, table and figure numbers of the paper

% Appendices keep their letters (Appendix D.2); tables and figures are numbered S1, S2, ...
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thesection}{\Alph{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\makeatletter
\def\thesectiondis{Appendix \Alph{section}:}
\def\thesubsectiondis{\thesection.\arabic{subsection}}
\makeatother

\begin{document}

\title{Supplementary Material for\\``\TITLE''}
\author{\AUTHOR}
\markboth{IEEE Journal of Oceanic Engineering}{\SHORT: supplementary material}
\maketitle

This supplement holds Appendices A--K of the paper. Section numbers in Roman numerals, tables numbered I, II and Figs.~1 and~2 refer to the paper; tables and figures numbered S1, S2, \ldots{} are in this supplement.

\BODY

\FloatBarrier
\bibliographystyle{IEEEtranN}
\bibliography{references}

\end{document}
"""


def front_matter():
    text = (OUT / "front.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    parts = dict(re.findall(r"^## (.+?)\n\n(.+?)(?=\n\n## |\s*$)", text, re.S | re.M))
    return {k: parts[k].strip() for k in ("Author", "Affiliation", "Abstract", "Index Terms")}


def fill(template, values):
    for k, v in values.items():
        template = template.replace("\\" + k, v.replace("\\", "\x03"))
    return template.replace("\x03", "\\")


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    refs = [parse_ref(e) for e in split_entries(MD)]
    assert len({r["key"] for r in refs}) == len(refs)
    write_bib(refs)
    cv = Converter(refs)
    if IEEE:
        title, _, body, appx = convert(MD, cv)
        fm = front_matter()
        assert len(fm["Abstract"].split()) <= 250, len(fm["Abstract"].split())
        vals = {"PACKAGES": IEEE_PACKAGES, "TITLE": cv.inline(title, heading=True), "AUTHOR": fm["Author"],
                "AFFIL": cv.inline(fm["Affiliation"]), "SHORT": "Saroha: Context-gated test-time adaptation",
                "ABSTRACT": cv.inline(fm["Abstract"]), "KEYWORDS": cv.inline(fm["Index Terms"].rstrip("."))}
        tex = fill(IEEE_MAIN, {**vals, "BODY": body})
        (OUT / "main.tex").write_text(tex, encoding="utf-8")
        (OUT / "supplement.tex").write_text(fill(IEEE_SUPP, {**vals, "BODY": appx}), encoding="utf-8")
        tex += appx
    else:
        title, abstract, body = convert(MD, cv)
        tex = PREAMBLE.replace("%s", "\x01", 3)
        for part in (cv.inline(title, heading=True), "\n\n".join(abstract), body):
            tex = tex.replace("\x01", part, 1)
        (OUT / "main.tex").write_text(tex, encoding="utf-8")
    for stem in FIG_WIDTH:
        shutil.copyfile(ROOT / "docs" / "figures" / f"{stem}.pdf", OUT / "figures" / f"{stem}.pdf")
    cited = {k for g in re.findall(r"\\cite(?:t|alp|p)?\{([^}]+)\}", tex) for k in g.split(",")}
    unused = sorted({r["key"] for r in refs} - cited)
    print(f"main.tex: {len(tex.split())} words of LaTeX, {len(refs)} bib entries, "
          f"{len(cv.used_labels)} cross-references resolved" + (f"; uncited {unused}" if unused else ""))


if __name__ == "__main__":
    main()
