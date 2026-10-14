# The paper, IEEE version (IEEE Journal of Oceanic Engineering)

The same paper as [`../README.md`](../README.md), set for an IEEE journal: generated from [`PAPER.md`](../../PAPER.md) and [`front.md`](front.md) by `py -3.13 paper/build_latex.py --ieee`. Nothing in the text or the numbers differs from the TMLR version except:

* **Front matter** from `front.md`: the author block (IEEE review is single-blind, so it is not anonymous), an abstract cut to IEEE's 250-word limit (248 words, same numbers), and Index Terms.
* **Code link restored** in Appendix J (the TMLR review copy withholds it).
* **TMLR's broader-impact section is dropped**; JOE asks for none. Its caveat (simulated corruptions, one held-out run) stays in Section VII and the abstract.
* **Layout:** `IEEEtran` journal class, two columns, numbered citations (`IEEEtranN`), "Fig." and "Section IV-B" references, table captions as short titles with the rest of the Markdown caption as a note under the table.
* **Appendices A–K are a separate `supplement.pdf`** (one column, tables and figures numbered S1, S2, …, its own reference list). The two documents cross-reference each other through the `xr` package (`S-` and `M-` label prefixes), so "Appendix D.2" in the paper and "Section V-B" or "Table I" in the supplement resolve to the right numbers.

Files: `main.tex`, `supplement.tex`, `references.bib`, `figures/` (generated), `main.pdf` (10 pages: body pages 1–8, references from page 8; page 10 holds the last references since the 2026-10-14 revision added seven) and `supplement.pdf` (30 pages). Compile with Tectonic in a scratch folder holding the four generated items, in the order main, supplement, main, supplement; `IEEEtran.cls` and `IEEEtranN.bst` come from the TeX Live bundle. No errors, overfull boxes, undefined references or multiply defined citations on 2026-10-14 (rebuilt after the simulated-review revision; keep Tectonic's intermediates, `--keep-intermediates`, so the cross-document references resolve).

**Before submission to JOE.**
1. **One venue at a time.** This and the TMLR version are the same paper; submit to one journal only.
2. **Corresponding-author e-mail and ORCID.** JOE requires an ORCID for the corresponding author; add the e-mail to the affiliation line in `front.md`. It is left out because the repository is public.
3. **Acknowledgements and author biography** (with photo) are needed for the final version, not for review.
4. **Page charges.** JOE requests a voluntary page charge after acceptance (US$110 per page on the OES submission page as checked on 2026-10-12; confirm the current amount). Keeping the appendices in the supplement keeps the article at 10 pages (9 before the N4Fg update; trimming the reference list would recover one).
5. Upload `supplement.pdf` as supplementary material for review.
