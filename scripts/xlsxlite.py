"""A minimal, dependency-free .xlsx writer.

`scripts/queue_xlsx_report.py` runs unattended for days on a training box whose
egress is unreliable (`pypi.ngc.nvidia.com` does not resolve there; one IR
benchmark run was already lost to a TLS failure fetching pretrained weights).
Adding `openpyxl` to that box would put a pip install on the critical path of a
monitoring tool whose whole job is to keep working when other things do not, so
the writer is stdlib-only — the same rule the dashboard and the divergence
watcher already follow.

An .xlsx is a zip of XML parts. This writes the five that Excel, LibreOffice and
pandas all require, and nothing else: no shared string table (strings are
inlined), no themes, no charts. Supported per sheet: a bold header row, frozen
panes, an autofilter, per-column widths and number formats.

    write_xlsx("out.xlsx", [
        {"name": "Runs",
         "columns": [{"header": "run", "width": 24},
                     {"header": "mAP50-95", "width": 12, "fmt": "num5"}],
         "rows": [["vis_bench_yolov8x_seed0", 0.25552]]},
    ])
"""

from __future__ import annotations

import math
import os
import re
import zipfile
from pathlib import Path

# The index into this tuple IS the cellXfs style index used by `_cell`, so the
# order is load-bearing. numFmtId 164+ is the custom range; 0 is General.
_FORMATS = (
    ("general", None),
    ("bold", None),      # header style: a font, not a number format
    ("num5", "0.00000"),
    ("num3", "0.000"),
    ("num2", "0.00"),
    ("num1", "0.0"),
    ("int", "#,##0"),
    ("pct1", "0.0%"),
)
_FMT_INDEX = {name: i for i, (name, _) in enumerate(_FORMATS)}
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _esc(text) -> str:
    text = _ILLEGAL.sub("", str(text))
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))


def col_letter(idx: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA."""
    out = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        out = chr(65 + rem) + out
    return out


def _cell(ref: str, value, style: int) -> str:
    if value is None or value == "":
        return ""
    s = f' s="{style}"' if style else ""
    if isinstance(value, bool):
        value = "yes" if value else "no"
    elif isinstance(value, (int, float)):
        # NaN/inf have no XML representation and Excel rejects the whole file
        # rather than showing one empty cell, so they become blanks here.
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return f'<c r="{ref}"{s}><v>{value!r}</v></c>'
    return (f'<c r="{ref}"{s} t="inlineStr"><is>'
            f'<t xml:space="preserve">{_esc(value)}</t></is></c>')


def _sheet_xml(sheet: dict) -> str:
    cols = sheet.get("columns") or []
    rows = sheet.get("rows") or []
    styles = [_FMT_INDEX.get(c.get("fmt") or "general", 0) for c in cols]

    views = ""
    freeze = sheet.get("freeze")
    if freeze:
        letters = re.sub(r"[0-9]", "", freeze)
        row = int(re.sub(r"[A-Z]", "", freeze) or 1)
        col = 0 if letters in ("", "A") else len(letters)
        views = ('<sheetViews><sheetView workbookViewId="0">'
                 f'<pane xSplit="{col}" ySplit="{row - 1}" topLeftCell="{freeze}" '
                 'activePane="bottomRight" state="frozen"/></sheetView></sheetViews>')

    colspec = ""
    if cols:
        colspec = "<cols>" + "".join(
            f'<col min="{i + 1}" max="{i + 1}" width="{c.get("width", 14)}" customWidth="1"/>'
            for i, c in enumerate(cols)) + "</cols>"

    body = []
    if cols:
        cells = "".join(_cell(f"{col_letter(i)}1", c.get("header", ""), 1)
                        for i, c in enumerate(cols))
        body.append(f'<row r="1">{cells}</row>')
    for r, values in enumerate(rows, start=2):
        cells = "".join(_cell(f"{col_letter(i)}{r}", v, styles[i] if i < len(styles) else 0)
                        for i, v in enumerate(values))
        body.append(f'<row r="{r}">{cells}</row>')

    autofilter = ""
    if sheet.get("autofilter") and cols and rows:
        autofilter = f'<autoFilter ref="A1:{col_letter(len(cols) - 1)}{len(rows) + 1}"/>'

    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{views}{colspec}<sheetData>{"".join(body)}</sheetData>{autofilter}</worksheet>')


def _styles_xml() -> str:
    numfmts = [(164 + i, fmt) for i, fmt in
               enumerate(f for _, f in _FORMATS if f)]
    nf = "".join(f'<numFmt numFmtId="{i}" formatCode="{_esc(f)}"/>' for i, f in numfmts)
    xfs, n = [], 164
    for name, fmt in _FORMATS:
        if fmt:
            xfs.append(f'<xf numFmtId="{n}" fontId="0" fillId="0" borderId="0" '
                       'applyNumberFormat="1"/>')
            n += 1
        elif name == "bold":
            xfs.append('<xf numFmtId="0" fontId="1" fillId="0" borderId="0" applyFont="1"/>')
        else:
            xfs.append('<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<numFmts count="{len(numfmts)}">{nf}</numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(xfs)}">{"".join(xfs)}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>')


def _safe_name(name: str, taken: set) -> str:
    clean = re.sub(r"[\[\]:*?/\\]", "-", str(name))[:31] or "Sheet"
    base, i = clean, 2
    while clean in taken:
        clean = f"{base[:28]}_{i}"
        i += 1
    taken.add(clean)
    return clean


def write_xlsx(path, sheets: list) -> Path:
    """Write `sheets` to `path` atomically. Returns the path written."""
    path = Path(path)
    if not sheets:
        raise ValueError("write_xlsx: no sheets")
    taken = set()
    names = [_safe_name(s.get("name", f"Sheet{i + 1}"), taken) for i, s in enumerate(sheets)]

    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
          '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(len(sheets)):
        ct.append(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" '
                  'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    ct.append("</Types>")

    wb_sheets = "".join(f'<sheet name="{_esc(n)}" sheetId="{i + 1}" r:id="rId{i + 1}"/>'
                        for i, n in enumerate(names))
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{wb_sheets}</sheets></workbook>')

    rels = "".join(
        f'<Relationship Id="rId{i + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i + 1}.xml"/>' for i in range(len(sheets)))
    rels += (f'<Relationship Id="rId{len(sheets) + 1}" '
             'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
             'Target="styles.xml"/>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
               f'{rels}</Relationships>')

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" '
                 'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                 'Target="xl/workbook.xml"/></Relationships>')

    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/styles.xml", _styles_xml())
        for i, sheet in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet_xml(sheet))
    # os.replace is atomic on POSIX; on Windows it raises if a reader (Excel) is
    # holding the destination open, which the caller retries rather than losing
    # the report.
    os.replace(tmp, path)
    return path
