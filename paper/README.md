# The paper (TMLR)

[`PAPER.md`](../PAPER.md) is the source. This folder is generated from it. An IEEE journal version of the same paper (JOE, two columns, appendices as a supplement) is built into [`ieee/`](ieee/README.md) by `build_latex.py --ieee`.

* `main.tex`, `references.bib` and `figures/` by `build_latex.py` (tables and figures renumbered in order of appearance; every "Table X", "Figure X", "§x.y" and "Appendix X.n" becomes a `\ref`, every author–year citation a natbib cite; the build fails on any undefined target or unmapped citation);
* `main.pdf`, compiled with Tectonic 0.17.0 against the official TMLR style files (`tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` from github.com/JmlrOrg/tmlr-style-file, not in this repository).

Edit the Markdown, then:

```
py -3.13 scripts/paper_figures.py          # only if a figure's source changed
py -3.13 paper/build_latex.py
tectonic --keep-logs main.tex              # in a scratch folder holding main.tex, references.bib, figures/ and the style files
```

**State on 2026-10-14** (revised after a simulated JOE review: claims scoped, variants renamed IG-BN, Table 1 and Fig. 2 moved to the appendices, related work extended, availability statement added; then the registered gate ablation and its feature-gap amendment added as §5.1's closing paragraph, Appendix G.5 and Table 21; then the retitle and a short §3.5 backbone benchmark with an eight-row Table 1, the full benchmark staying in Appendix H; then Figure 2, IG-BN's detections on one held-out frame, in §5.2). 57 pages: main body pages 1–13 (one page over TMLR's normal-review mark of 12), references from page 14, Appendices A–K after them. No errors, overfull boxes or undefined references. 51 references, 21 tables, 9 figures.

**Before submission.**
1. The review copy is anonymous: the author block is hidden and the code link is not given. Restore both for the camera-ready (`\usepackage[accepted]{tmlr}`), and add Acknowledgements, which the review copy omits.
2. TMLR asks that the submission not be under review elsewhere; neither the IEEE version in `ieee/` nor the archived venue drafts may be submitted in parallel.
3. Posting the same PDF to arXiv is allowed by TMLR.
