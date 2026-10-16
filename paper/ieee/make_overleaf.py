"""Standalone Overleaf projects for the IEEE paper: main-overleaf.zip and supplement-overleaf.zip.

The two documents normally cross-reference each other through `xr` (labels S-* and M-*), which needs both compiled
side by side. Overleaf compiles one document per project, so:
* main: the main text cites nothing in the supplement; the `xr` lines are removed (asserted).
* supplement: its references to the paper (\\ref{M-...}) are written in as fixed text, read from main.aux of a fresh
  compile. Regenerate the zip whenever the paper's section, table or figure numbering changes.

    py -3.13 paper/ieee/make_overleaf.py <out dir> [--tectonic PATH]

Run `py -3.13 paper/build_latex.py --ieee` first. Tectonic defaults to $TECTONIC.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent
XR_MAIN = ("\\usepackage{xr}  % labels and citations of the other document, under its prefix\n",
           "\\externaldocument[S-]{supplement}  % Appendices A-J; compile main, supplement, main\n\n")
XR_SUPP = ("\\usepackage{xr}  % labels and citations of the other document, under its prefix\n",
           "\\externaldocument[M-]{main}  % section, table and figure numbers of the paper\n")


def group(s: str, i: int) -> tuple[str, int]:
    """Balanced {...} starting at s[i] == '{'; returns (content, index after it)."""
    assert s[i] == "{"
    depth, j = 0, i
    while True:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1


def labels_of(aux: str) -> dict[str, str]:
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{", aux):
        first, _ = group(aux, m.end())  # {{number}{page}...}: the number is the first inner group
        out[m[1]] = group(first, 0)[0] if first.startswith("{") else first
    return out


def strip(t: str, lines: tuple[str, ...]) -> str:
    for line in lines:
        assert t.count(line) == 1, f"expected the line {line!r} once"
        t = t.replace(line, "")
    return t


def figures(t: str) -> list[str]:
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", t)))


def write_zip(path: Path, tex_name: str, tex: str) -> list[str]:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(tex_name, tex)
        z.write(SRC / "references.bib", "references.bib")
        for f in figures(tex):
            z.write(SRC / "figures" / f, f"figures/{f}")
        return z.namelist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tectonic", default=os.environ.get("TECTONIC"))
    a = ap.parse_args()
    assert a.tectonic, "pass --tectonic or set TECTONIC"
    a.out.mkdir(parents=True, exist_ok=True)

    main_tex = strip((SRC / "main.tex").read_text(encoding="utf-8"), XR_MAIN)
    assert "{S-" not in main_tex, "the main text references the supplement; it cannot stand alone"
    print(a.out / "main-overleaf.zip", write_zip(a.out / "main-overleaf.zip", "main.tex", main_tex))

    with tempfile.TemporaryDirectory() as tmp:  # fresh main.aux for the paper's numbers
        w = Path(tmp)
        for f in ("main.tex", "supplement.tex", "references.bib"):
            shutil.copy2(SRC / f, w / f)
        shutil.copytree(SRC / "figures", w / "figures")
        for f in ("main", "supplement", "main"):
            subprocess.run([a.tectonic, "--keep-intermediates", f"{f}.tex"], cwd=w, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        labels = labels_of((w / "main.aux").read_text(encoding="utf-8"))

    supp = strip((SRC / "supplement.tex").read_text(encoding="utf-8"), XR_SUPP)
    n = len(re.findall(r"\\ref\{M-", supp))

    def lit(m: re.Match) -> str:
        assert m[1] in labels, f"main.aux has no label {m[1]}"
        return labels[m[1]]

    supp = re.sub(r"\\ref\{M-([^}]+)\}", lit, supp)
    print(f"supplement: {n} references to the paper written in")
    print(a.out / "supplement-overleaf.zip", write_zip(a.out / "supplement-overleaf.zip", "supplement.tex", supp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
