#!/usr/bin/env bash
# N4Fg queue (docs/prereg-ttbn-fulltta-glare-2026-10-12.md, Amendment 1): waits for the development chain, then
# builds and scores pohang04 glare/clean whatever the development verdict; integrity checks stop it.
# Logs each step in the exposure ledger, regenerates Fig. 2, commits and pushes the results.
cd /a/Uncertain
D=runs/novel/ttbn_fulltta_glare
P=runs/holdout_p04/ttbn_fulltta_glare
LED=docs/exposure-ledger-2026-09-09.md
t() { date +'%I:%M %p'; }
led() { printf '%s\n' "$@" >> $LED; }
fail() {
  echo "[queue] FAILED: $1 $(t)"
  led "  * **$(t) — N4Fg queue stopped: $1.** See \`$2\`. Nothing after this step ran."
  git add $LED docs/eval/ttbn/ttbn_fulltta_glare_* 2>/dev/null
  git commit -q -m "N4Fg queue stopped: $1" && git push -q origin fusion-uq-phase3
  exit 1
}
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 KMP_BLOCKTIME=0
echo "[queue] waiting for the development chain $(t)"
dev_running() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*ttbn_fulltta_glare.py*-dev*' -or \$_.CommandLine -like '*ttbn_fulltta_glare.py check-fp16*' }).Count" | tr -d '\r'; }
idle=0
until grep -q "DEV DONE\|stopping" $D/dev_chain.log || grep -qs "^\[chain\] FAILED" $D/*.log; do
  sleep 30
  if [ "$(dev_running)" = "0" ]; then idle=$((idle + 1)); else idle=0; fi
  if [ $idle -ge 4 ]; then              # two minutes with no development step running: the chain died; resume it
    echo "[queue] development chain not running; resuming it $(t)"
    led "  * **$(t) — N4Fg development chain resumed by the queue** (its session ended; every step resumes where it stopped)."
    bash $D/dev_chain.sh > $D/dev_chain.log 2>&1
    idle=0
  fi
done
# fp16 check outcome to the ledger (registration step 1)
if [ -f $D/fp16_check.json ]; then
  led "  * **$(date -r $D/fp16_check.json +'%I:%M %p') — N4Fg development fp16 check:** $(tail -1 $D/check_fp16.log | cut -c9-). File \`$D/fp16_check.json\`."
fi
grep -q "DEV DONE" $D/dev_chain.log || fail "development chain ($(grep -hs "FAILED\|stopping" $D/*.log | tail -1))" "$D/dev_chain.log"
DV=$(py -3.13 -c "import json;print(json.load(open('docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.json'))['verdict'])")
[ "$DV" = "PASSES" ] || [ "$DV" = "FAILS" ] || fail "development integrity check ($DV)" "docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md"
led "* **$(date +%F) ($(t)) — N4Fg development: $DV** (\`docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md\`; registered in" \
    "  \`docs/prereg-ttbn-fulltta-glare-2026-10-12.md\`). Draws 945–947, third use. $(sed -n 's/^\*\*N4Fg [A-Z]* development\.\*\* //p' docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md)" \
    "  * pohang04 starts now whatever the verdict (Amendment 1)."
git add $LED docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.* scripts/ttbn_fulltta_glare.py scripts/paper_figures.py
git commit -q -m "N4Fg development: $DV" && git push -q origin fusion-uq-phase3
echo "[queue] development $DV; pohang04 build $(t)"

led "  * **$(t) — P-FG: pohang04 builds started** (plain flipped glare views, then N4Fg's flipped adapted blocks)."
py -3.13 -u scripts/ttbn_fulltta_glare.py build-p04-flip --workers 8 > $P/build_flip.log 2>&1 || fail "pohang04 flipped glare build" "$P/build_flip.log"
echo "[queue] flip done $(t)"
py -3.13 -u scripts/ttbn_fulltta_glare.py build-p04-adapted --workers 6 > $P/build_adapted.log 2>&1 || fail "pohang04 adapted build" "$P/build_adapted.log"
echo "[queue] adapted done $(t)"
export UQF_FAST_WBF=1 UQF_FAST_DIST=1 UQF_FAST_IO=1
led "  * **$(t) — P-FG: pohang04 builds done; scoring started** (\`UQF_FAST_*\`)."
py -3.13 -u scripts/ttbn_fulltta_glare.py score-p04 --workers 2 > $P/score.log 2>&1 || fail "pohang04 scoring" "$P/score.log"
echo "[queue] score done $(t)"
py -3.13 -u scripts/p04_heldout_score.py boot 2>&1 | grep -v "Warn\|fidx_g" > $P/boot.log
[ ${PIPESTATUS[0]} -eq 0 ] || fail "pohang04 bootstrap" "$P/boot.log"
echo "[queue] boot done $(t)"
py -3.13 -u scripts/ttbn_fulltta_glare.py report-p04 > $P/report.log 2>&1 || fail "pohang04 report" "$P/report.log"
R=docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.md
V=$(sed -n 's/^\*\*P-FG: \(.*\)\.\*\*$/\1/p' $R)
ROW=$(grep "^| \*\*N4Fg" $R | sed 's/\*\*//g')
led "* **$(date +%F) ($(t)) — N4Fg on pohang04: P-FG $V** (\`$R\`; registered in" \
    "  \`docs/prereg-ttbn-fulltta-glare-2026-10-12.md\`, Amendment 1). Ninth scoring of pohang04. Development: $DV." \
    "  * glare/clean: $ROW" \
    "  * $(grep '^| N4F (' $R | sed 's/\*\*//g')" \
    "  * Other cells (gate-equality check against P-FT): see the report's second table."
py -3.13 scripts/paper_figures.py tta_heldout > $P/figure.log 2>&1 || echo "[queue] figure FAILED (see $P/figure.log)"
git add $LED $R docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.json docs/figures/fig_tta_heldout.*
git commit -q -m "N4Fg on pohang04 glare/clean: P-FG $V" && git push -q origin fusion-uq-phase3
echo "[queue] ALL DONE: P-FG $V $(t)"
