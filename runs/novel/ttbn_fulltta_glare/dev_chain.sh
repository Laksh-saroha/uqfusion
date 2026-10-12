#!/usr/bin/env bash
# N4Fg development: build, fp16 check, score, bootstrap, report (docs/prereg-ttbn-fulltta-glare-2026-10-12.md)
cd /a/Uncertain
L=runs/novel/ttbn_fulltta_glare
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 KMP_BLOCKTIME=0
step() { echo "[chain] $1 start $(date +'%I:%M %p')"; shift; "$@" || { echo "[chain] FAILED $(date +'%I:%M %p')"; exit 1; }; }
step build py -3.13 -u scripts/ttbn_fulltta_glare.py build-dev --workers 6 > $L/build_dev.log 2>&1
step check py -3.13 -u scripts/ttbn_fulltta_glare.py check-fp16 --workers 6 > $L/check_fp16.log 2>&1
echo "[chain] check: $(tail -1 $L/check_fp16.log)"
grep -q "PASS: fp16" $L/check_fp16.log || { echo "[chain] fp16 check did not pass; stopping $(date +'%I:%M %p')"; exit 1; }
export UQF_FAST_WBF=1 UQF_FAST_DIST=1 UQF_FAST_IO=1
step score py -3.13 -u scripts/ttbn_fulltta_glare.py score-dev --workers 3 > $L/score_dev.log 2>&1
step boot py -3.13 -u scripts/ttbn_fulltta_glare.py boot-dev > $L/boot_dev.log 2>&1
step report py -3.13 -u scripts/ttbn_fulltta_glare.py report-dev > $L/report_dev.log 2>&1
echo "[chain] DEV DONE $(date +'%I:%M %p')"
