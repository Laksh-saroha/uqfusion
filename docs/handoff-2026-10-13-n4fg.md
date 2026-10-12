# Handoff — N4Fg (N4F with N4g's glare term), queued unattended at 11:49 PM on 10-12

**Final, 2:00 AM on 10-13: done.** Development PASSES (glare/clean/day +0.0179 [+0.0141, +0.0207]; no cell worse).
**P-FG REPLICATES**: glare/clean N4Fg 0.1967 against `full_tta` 0.1656, +0.0311 [+0.0286, +0.0332]; N4F (N4's gate)
equals `full_tta` there; no other cell worse than N4F. Paper updated (Table 2 row, Table 3 glare row, §4.4, §5.1,
§5.2, Fig. 2 and its caption, nine scorings in the abstract, §3.4, §7 and Appendix G, IEEE front matter) and both
versions rebuilt clean (`f100bcf`). The IEEE main PDF is now 10 pages (references spill onto page 10). The
recommendation paragraph (§7) is unchanged: whether to name N4Fg there is Laksh's call. The PC shut down after this.
Everything below is the plan as queued.

Registration: `docs/prereg-ttbn-fulltta-glare-2026-10-12.md` (11:44 PM; Amendment 1 at 11:47 PM: pohang04 runs
whatever the development verdict, integrity checks still stop the queue). Code: `scripts/ttbn_fulltta_glare.py`.

## What is running

* `runs/novel/ttbn_fulltta_glare/dev_chain.sh` (log `dev_chain.log` and per-step logs there): development build,
  fp16 check on draw 945, score, bootstrap, report.
* `runs/holdout_p04/ttbn_fulltta_glare/queue.sh` (log `queue.log`), launched detached: waits for development
  (resumes it if it died), logs the fp16 check and the development verdict in the ledger, commits and pushes; then
  builds pohang04 (plain flipped glare views, ~40 min; N4Fg's flipped adapted blocks, ~1.5–2 h), scores, bootstraps,
  writes `docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.md`, logs P-FG in the ledger, regenerates Fig. 2
  (`fig_tta_heldout`, which now draws the N4Fg glare row from the report's JSON), commits and pushes.
* Expected end: about 3:30 AM on 10-13. On any failure the queue writes a "queue stopped" ledger entry, commits and
  pushes, and stops; every step resumes where it stopped when rerun.

## Left for the next session (not automated: the wording depends on the result)

1. Read `queue.log`, the two reports and the ledger entries; check the commits.
2. PAPER.md: Table 3's glare row (`full_tta` and N4Fg − `full_tta` in place of "not scored") and its caption; the
   N4g/N4F text in §4.4 and §5.2; the Fig. 2 caption ("N4F was not scored on glare"); the scoring count "eight" →
   "nine" in the abstract, §3 (Registration), §7 and Appendix G (also lists P-FG); `paper/ieee/front.md`. If P-FG
   does not replicate, or development failed, say so in the same places and keep the row.
3. Rebuild: `py -3.13 paper/build_latex.py`, `py -3.13 paper/build_latex.py --ieee`, then Tectonic as in
   `paper/README.md` and `paper/ieee/README.md`. Commit and push (no Co-Authored-By trailers).
