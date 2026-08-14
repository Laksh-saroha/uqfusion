# yolo26x on a rented JarvisLabs L4 — runbook

The Phase 1 tail needs `yolo26x` × seeds 0,1,2 (~19 h/seed on the laptop's 4080).
Those three runs sit behind `yolo26l` in the laptop queue. This directory rents an
**L4 Ada 24 GB spot** instance and runs *only* `yolo26x` there, in parallel, then
merges the rows back into the same experiment.

Nothing here touches the laptop grid: different variants, different results CSV,
its own log and health file.

## The rule everything else serves

A row may join `runs/benchmark/benchmark_results_tail.csv` only if it was produced
with **`ultralytics==8.4.90`**, on split **`682dbe9f0f05`**, with **`--classes 0`**.
8.4.7 scores the same weights ~0.034 mAP higher (docs/phase1-handoff-2026-08-11.md
§1), so a row from any other version is not comparable and is worth nothing.

Three independent checks enforce it: `build_payload.py` refuses to write a payload
whose fingerprint differs, `remote_verify.py` re-checks it on the instance before
training, and `watch_l4.py` re-checks every row it pulls back.

## Files

| file | runs where | what it does |
|---|---|---|
| `build_payload.py` | laptop | rewrites the split lists to Linux paths, verifies the fingerprint, cuts the 16.4 GB upload into 16 groups |
| `sync_to_instance.sh` | laptop (Git Bash) | code → env setup → weights → data, resumable per group, ends with the on-instance verification |
| `remote_setup.sh` | instance | libGL, `ultralytics==8.4.90` under a torch constraint, CUDA preflight |
| `remote_run_26x.sh` | instance | the grid with a 40-attempt retry loop, `flock`ed against a second trainer |
| `remote_verify.py` | instance | manifest byte-for-byte, fingerprint, labels |
| `watch_l4.py` | laptop | polls, pulls each finished seed, resumes after a spot pause, **pauses the box when the grid finishes**, hard budget cap |
| `netguard.py` | laptop | upload phase only: if the uplink dies, keeps retrying the pause until the network returns |
| `pause_instance.py` | laptop | stop the meter and record why, used by the automation's halt paths |

## Cost and time — measured, 2026-08-14

Prices are INR/hour, both + 50 GB storage ₹0.61: L4 spot **₹28.15/h**, L4
on-demand **₹41.92/h**. The run switched to on-demand on 2026-08-14 for the reason
in "Spot reclaim" below. Measured on this box, `26x` at batch 16, imgsz 640,
workers 8, over 8 completed epochs:

```
45.1 min/epoch   (3009 iters at ~1.2-1.3 it/s, batch 16) = 0.752 h
epoch 1 mAP50-95 0.2777, best 0.2991 at epoch 5
16.4 GB upload   149 min  (~1.9 MB/s uplink)  ≈ ₹70
```

So **₹31.5 per epoch on-demand**, ₹21.2 on spot — but only if spot banks the epoch.

Total depends entirely on when `patience=20` fires, which nothing predicts — the
run ends 20 epochs after its best:

| early stop at | per seed | on-demand/seed | 3 seeds |
|---|---|---|---|
| ~epoch 30 (what `26m` and `26l` did) | 22.6 h | ₹945 | ₹2,840 |
| ~epoch 45 | 33.8 h | ₹1,420 | ₹4,250 |
| never (100 epochs) | 75.2 h | ₹3,150 | ₹9,450 |

`--max-hours` on the supervisor is the ceiling that makes this bounded; it pauses
the instance at the cap wherever the runs have got to.

**L4 is the right buy per rupee** even though it is slow: the A100 40 GB spot
discount is only 12% (₹74.52), so at ~2.7x the speed it costs about the same per
completed run. Pick A100 only when wall-clock beats spend. It runs at roughly half
the laptop 4080's throughput for this model (17.6 img/s vs 36).

## Sequence

```bash
# 0. once: credentials and a key on the account
jl setup                                   # paste the API key from jarvislabs.ai/settings/api-keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
jl ssh-key add ~/.ssh/id_ed25519.pub --name laptop

# 1. laptop-side payload (no GPU running, ~1 min)
python jarvislabs/build_payload.py         # must print fingerprint 682dbe9f0f05

# 2. rent
jl create --gpu L4 --spot --storage 100 --name uq-26x --yes --json

# 3. code + 16.4 GB data, resumable; re-run it if the link drops
bash jarvislabs/sync_to_instance.sh <machine_id>

# 4+5. launch the grid and supervise it, one command
jarvislabs\.venv-jl\Scripts\python.exe jarvislabs\watch_l4.py --machine-id <id> --launch
```

Launch the supervisor **detached**, never with `Start-Process` — a console teardown
propagates `CTRL_CLOSE_EVENT` to the whole process group and killed both safety
layers of the laptop grid once already (handoff trap 2):

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c A:\Uncertain\jarvislabs\watch_l4.cmd'
  CurrentDirectory = 'A:\Uncertain' }
```

## Checking on it, and on the money

```bash
A:\Uncertain\jarvislabs\status.cmd                    # snapshot: progress, ETA, spend, runway
type A:\Uncertain\runs\health_l4.log                  # one line per 5 min
findstr /V " OK " A:\Uncertain\runs\health_l4.log     # only problems; empty == all clear
```

`status.cmd` only reads, so it is safe to run at any time and as often as you like.
It answers "is more money needed" directly:

```
instance 12345  Running  L4 spot
seeds finished : 1/3  ['0']
active run     : ship_yolo26x_seed1  epoch 14/100  4.8it/s  gpu 98 % 14322 MiB
curve          : 13 epochs, 21.4 min/epoch, best mAP50-95 0.2914 @ epoch 11
eta this run   : 6.4 h if patience fires now, 31.0 h if it runs to 100
eta remaining  : + 1 seed(s) x up to 36 h
spend          : 4.82 so far, burn 0.19/h, balance 21.40
runway         : 112.6 h at the current rate
```

The ETA is a range on purpose: `patience=20` ends a run 20 epochs after its best
epoch, and nothing predicts when that lands. `26l` seed 0 stopped near epoch 28.

Every `health_l4.log` line carries `cost=`, `burn=/h`, `bal=` and `runway=`, so the
spend history is in the same file as the training history. Burn is measured from
the instance's own cost counter, not assumed from a price list, so it is right
whatever the account bills in.

**The balance guard is the one that protects the data.** If the balance would run
out within `--min-runway-hours` (default 2 h), the supervisor pauses the instance
while there is still credit to resume with, and says so in the log. Top up, then:

```bash
jl resume <machine_id> --spot --yes
```

and relaunch the wrapper (step 4 of the sequence) — the grid picks up from
`weights/last.pt`.

## When it needs a human, it stops the meter

Anything requiring judgement pauses the instance instead of polling on at ₹28/h.
Pause keeps the disk, so the cost of a false alarm is one `jl resume`. The log line
starts with `STOP`, and `jarvislabs/l4_state.json` records `halted_reason`:

| trigger | why it is not retryable |
|---|---|
| upload verification failed | wrong or truncated data — every hour after this produces nothing |
| grid would not reach a training iteration in 3 launch attempts | a real defect, not a flake |
| wrapper exhausted its 40 attempts (`runs/GRID_FAILED`) | another poll will not fix it |
| trainer would not stay up across 3 relaunches | same |
| a finished row carries the wrong split / ultralytics / class filter | the row cannot join the CSV, so more of them are worthless |
| balance would run out within 2 h | pause now, while there is still credit to resume with |

After fixing the cause (this is also the top-up path):

```bash
A:\Uncertain\jarvislabs\resume_after_topup.cmd <machine_id>
```

It resumes as **spot**, relaunches the grid — which continues from `weights/last.pt`,
losing at most the epoch that was in flight — and supervises it. The supervisor
refuses to start on a halted instance without `--clear-halt` (that wrapper passes
it), because otherwise it would see "paused and unfinished" and resume it straight
back into the same failure.

### Why launch lives in the supervisor

It used to be a bash script with its own probe regex, and on the first real launch
that regex missed a perfectly healthy epoch-1 run. Because its halt predicate was
"confirmation absent", three misses would have **paused a training run that was at
100% GPU**. The predicate is now asymmetric and lives in one tested place:

- **confirm** on a live iteration counter *or* a busy GPU with a trainer process;
- **halt** only on positively observed idleness — no trainer AND an idle GPU, seen
  four polls running, over a connection that is working;
- a failed probe counts for **nothing**, because a broken probe looks exactly like a
  dead trainer, and only one of those is worth throwing away GPU hours over.

## If the laptop's internet drops mid-upload

Every safeguard here pauses the instance by calling the JarvisLabs API — over the
same uplink that just died. So the single pause attempt in `autostart_v2.sh` fails
and logs "could not pause", and a rented GPU bills for an empty box.

`netguard.py` exists for exactly that. Launch it detached alongside the upload:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c A:\Uncertain\jarvislabs\netguard.cmd'
  CurrentDirectory = 'A:\Uncertain' }
```

It watches for "the upload is dead **and** training never started" — a sync log that
stops growing for 12 min (a healthy group lands every ~10) or an explicit failure —
then retries the pause every 30 s until it lands, whether that takes five minutes or
all night. It stands down by itself once training is confirmed, because an outage
during training costs nothing: the grid keeps running on the instance whether or not
the laptop can see it, and the supervisor reconnects afterwards.

**What it cannot do:** stop the meter *during* the outage. Nothing laptop-side can —
the pause needs the network. Waste is bounded by the outage length (₹28/h), and by
the balance itself in the worst case. Closing that gap needs a dead-man's switch
running *on the instance*, which means storing an API key there — an account-owner
decision, not something the automation should do on its own.

## What an interruption actually costs

| event | cost | who handles it |
|---|---|---|
| spot pause by JarvisLabs | ≤ 1 epoch (~25-35 min); disk survives | supervisor resumes with `--spot`, relaunches, grid.py continues from `last.pt` |
| trainer crash / OOM kill | ≤ 1 epoch | `remote_run_26x.sh` retries 40 times |
| ssh or laptop drops | nothing — training is detached under `setsid nohup` | supervisor reconnects next poll |
| instance lost outright | back to the last `last.pt` pulled (≤ `--ckpt-hours`, default 6 h) | re-create, re-sync, drop the checkpoint into the run dir |
| balance hits zero | nothing, if the guard fired first | supervisor pauses at < 2 h runway |

Ultralytics writes `weights/last.pt` **every epoch** and `grid.py` resumes from it,
which is why every row above is bounded by one epoch rather than by a whole run.
Finished seeds are pulled to `runs/benchmark/l4/` as they land, so a finished row
never depends on the instance still existing.

## When it finishes

`watch_l4.py` pauses the instance itself and stops. Storage still bills while
paused (~$0.014/h for 100 GB), so once the rows are merged:

```bash
jl destroy <machine_id> --yes
```

Merging: the L4 rows land in `runs/benchmark/benchmark_results_l4_26x.csv` with the
same schema. Append them to `benchmark_results_tail.csv` only after confirming
`ultralytics_version == 8.4.90`, `split_fingerprint == 682dbe9f0f05`, `classes == 0`
— and record in the experimental record that these three rows ran at batch 16 on an
L4, while the laptop rows ran at batch 8 (batch is already non-uniform across the
CSV: 7 server rows used 24, and the measured batch effect is smaller than the
seed-0 duplicate spread).

## Spot reclaim: two things learned the hard way, 2026-08-14

**1. A resume hands back a FRESH CONTAINER. Only `/home` survives.**
The dataset, run dirs and checkpoints live on the rbd volume and came back intact —
but `site-packages` is under `/root/miniconda3`, so `ultralytics==8.4.90` was simply
gone, and the grid died with `ModuleNotFoundError` on a box that otherwise had
everything. `remote_run_26x.sh` now checks for ultralytics 8.4.90 on every start and
re-runs `remote_setup.sh` if it is missing (~90 s). Anything else that ever needs
installing must go in that script, never typed into a shell by hand.

**2. Reclaim frequency has to be compared against epoch length.**
Ultralytics writes `last.pt` **at the end of an epoch**. An epoch of `26x` here is
45 min. On 2026-08-14 two reclaims landed ~30 min apart, the second only ~10 min
after the resume — which means *no epoch can finish between reclaims*, and every
rented hour banks exactly nothing while still being billed. That is not a case for
retrying harder; it is a spend decision:

| | rate | with ₹247 | reclaim risk |
|---|---|---|---|
| L4 spot | ₹27.54/h | 8.8 h | may bank 0 epochs if reclaims stay this frequent |
| L4 on-demand | ₹41.31/h | 5.9 h | none — ~7 epochs banked, guaranteed |

**3. `pgrep -f run_benchmark.py` inside the probe matched the probe.**
`pgrep -f` scans whole command lines, and the probe's own `bash -c` line contains
the pattern — so `ALIVE` was never 0 and "is the trainer alive?" always answered
yes. On 2026-08-14 the supervisor started against a box whose grid had crash-looped
to a stop, read a stale epoch line out of the log, and reported "training already
running at epoch 9/100" while the GPU sat at 0%. The pattern is now `'[r]un_benchmark.py'`.
Live evidence means GPU utilisation and a process count that excludes the asker.

The rule of thumb: **spot is only cheaper if the mean time between reclaims exceeds
one epoch.** Below that, on-demand is cheaper per *banked epoch* even at 1.5x the
rate. Shorter epochs (smaller model, fewer images) shift the balance back to spot.

## Known traps, carried over

- **`resume` drops `--spot`** unless you pass it again — the instance comes back
  on-demand at 2.3x. `watch_l4.py` always passes it.
- **Batch cannot change on resume.** Ultralytics reloads it from the checkpoint, so
  an OOM at epoch 20 means restarting that seed, not lowering `BATCH`. Batch 16 is
  the largest measured-safe value: `26x` peaked 15.31 GB at batch 16 on the 12 GB
  laptop card (which paged rather than OOMed); 24 GB has real headroom. If it ever
  does OOM, delete `runs/benchmark/runs/ship_yolo26x_seed<N>/` and relaunch with
  `BATCH=8` (the laptop's known-good 7.9 GB setting).
- **`/dev/shm` in containers** is often 64 MB and silently kills dataloader workers.
  `remote_setup.sh` prints it; if it is small, relaunch with `WORKERS=2`.
- **Never `pip install torch`** on the instance. The template's build matches its
  driver; a PyPI wheel does not, and every run then dies with "driver too old".
- **Don't pull `*.cache` files back** — stale Ultralytics label caches silently
  reuse old labels.
