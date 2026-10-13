Benchmark extension (ep25 -> patience 20), portable package
=============================================================

What this is
  The dgxanode01 ep25 VIS benchmark runs are being trained on to patience 20 (warm start from each
  base run's last.pt, patience carried over). This package holds the runs delegated to this machine,
  the exact data they read (stride-4 train + val, restored labels), the pinned environment and the
  same runner (scripts\bench_ext_local.py) the main laptop uses. package_manifest.json lists the runs.

Start
  1. Plug the stick into the training laptop and double-click START.bat on it. It offers to unpack
     the package (payload.tar, ~9.3 GB) into C:\Training: answer Y, or type another folder. Do unpack
     locally: the stick is far too slow to train from (each epoch reads ~9 GB).
     From then on start it with C:\Training\START.bat.
  2. The first run then:
       - finds Python 3.13 (offers to install it with winget if missing)
       - creates %LOCALAPPDATA%\uqfusion_bench_ext\venv with the pinned versions
         (torch 2.7.1+cu118, ultralytics 8.4.90, ...; ~3 GB download)
       - checks the GPU, sha256-verifies every packaged file, runs the data fingerprint gate
       - opens the dashboard (http://127.0.0.1:8771) and starts the queue
  3. Keep the laptop plugged in, lid open. The queue stops Windows from sleeping while it runs.

Stop / resume
  - Dashboard "Pause": stops after the current epoch's checkpoint. "Resume" continues it.
  - Closing the START.bat window also stops it (the epoch in progress is lost).
  - START.bat again resumes: finished runs are skipped, an interrupted run continues from its last
    epoch with its patience intact. Moving the folder (another drive letter) is fine.

Check without training
  powershell -ExecutionPolicy Bypass -File setup_and_run.ps1 -DryRun

When it has finished
  Bring back the whole folder  runs\vis_benchmark_stride4_ep25_ext  (the vis_bench_*_ext run folders,
  progress.csv, queue.log, logs\). Each finished run has an ext_done.json recording the host, batch,
  replay check and data gate.

Differences from the main laptop
  - Data gate is bench_ext_fingerprint.py --quick only (image path|size and label content of every
    file a run reads). The label-hash ledger needs the full repo and train tree, so it is not packaged.
    ext_done.json records which gate ran.
  - Batch: the largest of 16/8/4/2 that fits this GPU is probed once per model family
    (batch_plan.json). nbs 64 keeps the effective batch at 64.
