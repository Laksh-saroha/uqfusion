"""Unattended supervisor for the rented-L4 yolo26x runs.

Four jobs, in priority order:

1. **Never lose progress.** Ultralytics writes `weights/last.pt` every epoch and
   grid.py resumes from it, so a spot pause costs at most the epoch in flight
   (~30 min). This resumes the instance when JarvisLabs pauses it — re-passing
   `--spot`, since a plain resume comes back on-demand at 2.3x — relaunches the
   wrapper, and periodically pulls `last.pt` back to the laptop so even losing the
   whole box costs hours, not days.
2. **Never run out of money mid-run.** A zero balance is the one failure that can
   take the disk with it. Every poll records cost and balance, derives the burn
   rate from the cost delta (currency-agnostic — it is whatever the account bills
   in), and pauses the instance while there is still credit left to resume with.
3. **Stop paying for nothing.** When the grid writes runs/GRID_DONE the results are
   pulled and the instance paused within one poll, plus a hard --max-hours cap.
4. **Leave a trail.** One line per poll in runs/health_l4.log, same grep-friendly
   shape as runs/health.log:  `grep -v " OK " runs/health_l4.log`

    python jarvislabs/watch_l4.py --machine-id 12345      # supervise (loops)
    python jarvislabs/watch_l4.py --once                  # one snapshot, human-readable

Run it with the venv that has the SDK: jarvislabs/.venv-jl/Scripts/python.exe
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "jarvislabs" / "l4_state.json"
HEALTH = ROOT / "runs" / "health_l4.log"
LOCAL_CSV = ROOT / "runs" / "benchmark" / "benchmark_results_l4_26x.csv"
LOCAL_PULL = ROOT / "runs" / "benchmark" / "l4"

RROOT = "/home/uqfusion"
REMOTE_CSV = f"{RROOT}/runs/benchmark/benchmark_results_l4_26x.csv"
REMOTE_RUNS = f"{RROOT}/runs/benchmark/runs"
EXPECTED_FINGERPRINT = "682dbe9f0f05"
EXPECTED_ULTRALYTICS = "8.4.90"
SEEDS = (0, 1, 2)

ANSI_PAT = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_PAT = re.compile(r"^\s*(\d+)/(\d+)\s")
RATE_PAT = re.compile(r"([\d.]+)it/s")
ATTEMPT_PAT = re.compile(r"\[wrapper\] attempt (\d+)")

LAUNCH = (
    f"cd {RROOT} && BATCH=${{BATCH:-16}} WORKERS=${{WORKERS:-8}} "
    f"setsid nohup bash jarvislabs/remote_run_26x.sh >> {RROOT}/runs/wrapper_l4.log 2>&1 "
    "< /dev/null & sleep 2; echo launched"
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def log(line: str) -> None:
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEALTH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.is_file() else {}

    def get(self, k, default=None):
        return self.d.get(k, default)

    def set(self, **kw):
        self.d.update(kw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# instance access: SDK for the control plane, ssh/scp for the data plane
# --------------------------------------------------------------------------
class Box:
    def __init__(self, machine_id: int):
        from jarvislabs import Client

        self.client = Client()
        self.machine_id = int(machine_id)
        self._ssh: list[str] | None = None

    def get(self):
        return self.client.instances.get(self.machine_id)

    def balance(self) -> float:
        b = self.client.account.balance()
        return float(b.balance) + float(getattr(b, "grants", 0.0) or 0.0)

    def pause(self) -> None:
        self.client.instances.pause(self.machine_id)

    def resume(self, spot: bool = True) -> int:
        inst = self.client.instances.resume(self.machine_id, is_spot=spot)
        self.machine_id = int(inst.machine_id)
        self._ssh = None
        return self.machine_id

    # --- ssh -------------------------------------------------------------
    def ssh_parts(self, refresh: bool = False) -> list[str]:
        if self._ssh and not refresh:
            return self._ssh
        inst = self.get()
        if not inst.ssh_command:
            raise RuntimeError(f"instance {self.machine_id} exposes no ssh command (status {inst.status})")
        parts = shlex.split(inst.ssh_command)
        host = parts[-1]
        self._ssh = parts[:-1] + [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", host,
        ]
        return self._ssh

    def sh(self, command: str, timeout: int = 180) -> subprocess.CompletedProcess:
        # encoding/errors are not optional here: the log carries ultralytics'
        # UTF-8 progress bars, and text=True alone decodes with the Windows
        # locale codec (cp1252), which raises on those bytes, kills subprocess'
        # reader thread and hands back stdout=None with returncode 0 — a probe
        # that looks like it worked and tells you nothing.
        return subprocess.run([*self.ssh_parts(), command], capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)

    def fetch(self, remote: str, local: Path, timeout: int = 1800) -> bool:
        parts = self.ssh_parts()
        host = parts[-1]
        scp = ["scp"]
        i = 1
        while i < len(parts) - 1:
            tok = parts[i]
            if tok == "-p":
                scp += ["-P", parts[i + 1]]
                i += 2
            elif tok in ("-o", "-i", "-F", "-J"):
                scp += [tok, parts[i + 1]]
                i += 2
            else:
                scp.append(tok)
                i += 1
        local.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run([*scp, f"{host}:{remote}", str(local)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode == 0


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_progress(tail: str) -> tuple[str, str, int]:
    text = ANSI_PAT.sub("", tail.replace("\r", "\n"))
    lines = text.splitlines()
    attempts = [int(m.group(1)) for m in (ATTEMPT_PAT.search(l) for l in lines) if m]
    epoch = rate = "?"
    for l in reversed(lines):
        m = EPOCH_PAT.match(l)
        if m:
            epoch = f"{m.group(1)}/{m.group(2)}"
            r = RATE_PAT.search(l)
            rate = f"{r.group(1)}it/s" if r else "?"
            break
    return epoch, rate, (max(attempts) if attempts else 0)


def csv_rows(path_or_text) -> list[dict]:
    if isinstance(path_or_text, Path):
        if not path_or_text.is_file():
            return []
        text = path_or_text.read_text(encoding="utf-8", errors="replace")
    else:
        text = path_or_text
    return list(csv.DictReader(io.StringIO(text)))


def curve_stats(text: str) -> dict:
    """epochs done, seconds/epoch and best mAP50-95 from an ultralytics results.csv.

    Read while the trainer is writing it, so it must tolerate a torn file: a row
    with more values than the header lands under DictReader's `None` restkey, and
    `None.strip()` took a whole poll down the first time epoch 1 finished. Progress
    metrics are a convenience — they never get to break the supervision loop.
    """
    # Drop blank lines first. The probe's section marker leaves a leading newline,
    # and DictReader takes that empty line as the header — then every real row goes
    # into the None restkey and the whole curve silently reads as nan.
    text = "\n".join(line for line in (text or "").splitlines() if line.strip())
    try:
        rows = csv_rows(text)
    except csv.Error:
        return {}
    if not rows:
        return {}
    key_t = next((k for k in rows[0] if k and k.strip() == "time"), None)
    key_m = next((k for k in rows[0] if k and "mAP50-95" in k), None)
    out = {"epochs": len(rows)}
    if key_t:
        try:
            out["sec_per_epoch"] = float(rows[-1][key_t]) / len(rows)
        except (TypeError, ValueError):
            pass
    if key_m:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key_m]))
            except (TypeError, ValueError):
                pass
        if vals:
            out["best_map"] = max(vals)
            out["best_epoch"] = vals.index(max(vals)) + 1
    return out


def check_rows(rows: list[dict]) -> str | None:
    """A row on a different split, ultralytics or class filter cannot join the
    experiment's CSV — better to know at row 1 than at merge time."""
    for r in rows:
        if r.get("split_fingerprint") != EXPECTED_FINGERPRINT:
            return f"seed {r.get('seed')} fingerprint {r.get('split_fingerprint')}"
        if r.get("ultralytics_version") != EXPECTED_ULTRALYTICS:
            return f"seed {r.get('seed')} ultralytics {r.get('ultralytics_version')}"
        if (r.get("classes") or "") != "0":
            return f"seed {r.get('seed')} classes {r.get('classes')}"
    return None


# Sections are separated by ';', never '&&', and the script ends in `exit 0`.
# Chained with '&&' this probe reported "ssh failed" for the whole of epoch 1,
# because results.csv does not exist until the first epoch ends: `cat` returned
# non-zero and took the rest of the chain — including the log tail — with it.
# A probe must fail only when the connection fails, not when one optional file
# it looks for is legitimately absent.
PROBE = (
    f"cd {RROOT} 2>/dev/null; "
    "echo MARK_DONE=$( [ -f runs/GRID_DONE ] && echo 1 || echo 0 ); "
    "echo MARK_FAIL=$( [ -f runs/GRID_FAILED ] && echo 1 || echo 0 ); "
    # The bracket around the first letter is load-bearing: `pgrep -f` scans full
    # command lines, and this probe's own `bash -c` line contains the pattern, so
    # the unbracketed form matches itself and ALIVE is never 0 — a dead trainer
    # then reads as alive, which is exactly the case the watchdog exists for.
    "echo ALIVE=$(pgrep -fc '[r]un_benchmark.py' || echo 0); "
    "echo GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader "
    "2>/dev/null | tr ',' ' ' | tr '\\n' ' '); "
    f"D=$(ls -1dt {REMOTE_RUNS}/ship_yolo26x_seed*/ 2>/dev/null | grep -v _val | head -1); "
    "echo ACTIVE=$D; "
    "echo ---CURVE---; "
    "[ -n \"$D\" ] && cat \"$D\"results.csv 2>/dev/null; "
    "echo ---LOG---; "
    "tail -c 200000 runs/grid_l4.log 2>/dev/null; "
    "exit 0"
)


def probe_fields(out: str) -> dict:
    head = out.split("---CURVE---", 1)[0]
    fields = {}
    for line in head.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    curve_text = ""
    log_text = ""
    if "---CURVE---" in out:
        rest = out.split("---CURVE---", 1)[1]
        curve_text, _, log_text = rest.partition("---LOG---")
    return {"f": fields, "curve": curve_text, "log": log_text}


# --------------------------------------------------------------------------
def pull_results(box: Box, state: State) -> int:
    """Bring the CSV back and, for each finished seed, that run's artifacts. Runs
    every poll: a row that exists locally survives losing the instance."""
    box.fetch(REMOTE_CSV, LOCAL_CSV, timeout=300)
    rows = csv_rows(LOCAL_CSV)
    pulled = set(state.get("pulled", []))
    for row in rows:
        name = f"ship_yolo26x_seed{row.get('seed')}"
        if name in pulled:
            continue
        ok = True
        for rel in ("results.csv", "args.yaml", "weights/best.pt"):
            ok &= box.fetch(f"{REMOTE_RUNS}/{name}/{rel}", LOCAL_PULL / name / rel)
        if ok:
            pulled.add(name)
            state.set(pulled=sorted(pulled))
            log(f"{now()}  INFO  pulled artifacts for {name}")
    return len(rows)


def pull_checkpoint(box: Box, state: State, active: str) -> None:
    """Periodic `last.pt` copy of the run in flight. Spot pauses keep the disk, so
    this is insurance against losing the instance outright — it turns 'start that
    seed again' into 'resume it from the last checkpoint we saw'."""
    name = Path(active.rstrip("/")).name if active else ""
    if not name:
        return
    if box.fetch(f"{REMOTE_RUNS}/{name}/weights/last.pt", LOCAL_PULL / name / "weights" / "last.pt"):
        state.set(last_ckpt_at=time.time(), last_ckpt_run=name)
        log(f"{now()}  INFO  checkpoint pulled: {name}/weights/last.pt")


def burn_rate(state: State, cost: float) -> float:
    """Account-currency per hour, measured from the instance's own cost counter."""
    samples = state.get("cost_samples", [])
    # A resume starts a NEW machine whose cost counter restarts at 0. Left alone,
    # that negative delta drives the measured rate to 0, which silently disables
    # the low-balance guard — the one safeguard that must never switch itself off.
    # Treat any drop as a new billing epoch and start the window again.
    if samples and cost < samples[-1][1]:
        samples = []
    samples.append([time.time(), cost])
    samples = samples[-40:]
    state.set(cost_samples=samples)
    if len(samples) < 2:
        return 0.0
    dt = samples[-1][0] - samples[0][0]
    dc = samples[-1][1] - samples[0][1]
    return (dc / dt * 3600) if dt > 60 and dc >= 0 else 0.0


def snapshot(box: Box, state: State) -> dict:
    inst = box.get()
    snap = {"machine_id": box.machine_id, "status": inst.status, "cost": float(inst.cost or 0),
            "gpu": inst.gpu_type, "spot": inst.is_spot, "balance": None, "rows": 0}
    try:
        snap["balance"] = box.balance()
    except Exception as exc:  # noqa: BLE001 - balance is advisory, never fatal
        snap["balance_error"] = repr(exc)
    if inst.status != "Running":
        return snap
    p = box.sh(PROBE)
    if p.returncode != 0:
        snap["probe_error"] = p.stderr.strip()[:200]
        return snap
    parsed = probe_fields(p.stdout)
    snap.update(parsed["f"])
    snap["curve"] = curve_stats(parsed["curve"])
    epoch, rate, attempt = parse_progress(parsed["log"])
    snap.update(epoch=epoch, it_rate=rate, attempt=attempt)
    return snap


def sync_report() -> str | None:
    """Upload progress, or None once the payload is verified on the instance.

    Reads the sync's own log rather than the instance: tar reports each finished
    group with a byte count and a rate, which is exactly the ETA input, and it
    works even when the box is unreachable.
    """
    log_path = ROOT / "runs" / "sync_l4.log"
    manifest = ROOT / "jarvislabs" / "payload" / "manifest"
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "[verify] OK" in text:
        return None

    total_groups = len(list(manifest.glob("group_*.txt"))) or 16
    total_bytes = 0
    expect = manifest / "expect.tsv"
    if expect.is_file():
        for line in expect.read_text(encoding="utf-8").splitlines():
            if line.startswith("TOTAL\t"):
                total_bytes = int(line.split("\t")[2])

    done_bytes = [int(m) for m in re.findall(r"Total bytes written: (\d+)", text)]
    sent = sum(done_bytes)
    rates = re.findall(r"([\d.]+)MiB/s", text)
    rate = float(rates[-1]) if rates else 0.0
    active = re.findall(r"(group_\d+)\s+(\d+) files", text)
    current = active[-1][0] if active else "?"

    pct = (sent / total_bytes * 100) if total_bytes else 0
    eta = ((total_bytes - sent) / (rate * 1024 ** 2) / 3600) if rate else float("inf")
    lines = [
        "phase          : UPLOADING (training has not started yet)",
        f"groups         : {len(done_bytes)}/{total_groups} complete, now sending {current}",
        f"transferred    : {sent / 1e9:.2f} / {total_bytes / 1e9:.2f} GB  ({pct:.0f}%)",
        f"rate           : {rate:.2f} MiB/s   eta {eta:.1f} h",
    ]
    if "[verify] FAIL" in text or "giving up on group" in text:
        lines.append("STATUS         : FAILED — read runs/sync_l4.log")
    return "\n".join(lines)


def report(snap: dict, state: State) -> str:
    """Human-readable answer to 'where is it and will the money last'."""
    rows = csv_rows(LOCAL_CSV)
    c = snap.get("curve") or {}
    lines = [
        f"instance {snap['machine_id']}  {snap['status']}  {snap.get('gpu')}"
        f"{' spot' if snap.get('spot') else ''}",
        f"seeds finished : {len(rows)}/3  {[r.get('seed') for r in rows]}",
        f"active run     : {Path(str(snap.get('ACTIVE', '')).rstrip('/')).name or '-'}"
        f"  epoch {snap.get('epoch', '?')}  {snap.get('it_rate', '?')}  gpu {snap.get('GPU', '?')}",
    ]
    if c:
        spe = c.get("sec_per_epoch", 0)
        lines.append(f"curve          : {c.get('epochs')} epochs, {spe / 60:.1f} min/epoch, "
                     f"best mAP50-95 {c.get('best_map', float('nan')):.4f} @ epoch {c.get('best_epoch')}")
        if spe:
            # Early stopping ends a run 20 epochs after its best, so the honest
            # answer is a range: patience-limited at one end, 100 epochs at the other.
            done = c.get("epochs", 0)
            soon = max(0, c.get("best_epoch", done) + 20 - done) * spe / 3600
            late = max(0, 100 - done) * spe / 3600
            per_run = 100 * spe / 3600
            left_runs = max(0, 3 - len(rows) - 1)
            lines.append(f"eta this run   : {soon:.1f} h if patience fires now, {late:.1f} h if it runs to 100")
            lines.append(f"eta remaining  : + {left_runs} seed(s) x up to {per_run:.0f} h")
    bal, cost = snap.get("balance"), snap.get("cost", 0)
    rate = state.get("burn_rate", 0.0)
    lines.append(f"spend          : {cost:.2f} so far, burn {rate:.3f}/h, balance {bal if bal is None else f'{bal:.2f}'}")
    if bal is not None and rate > 0:
        lines.append(f"runway         : {bal / rate:.1f} h at the current rate")
    return "\n".join(lines)


def halt(box: Box, state: State, reason: str, body: str = "") -> bool:
    """Stop the meter and wait for a human.

    Anything that needs judgement — a grid that will not start, rows that are not
    comparable — ends here rather than in a poll loop that keeps a rented GPU
    warm. Pause keeps the disk: the payload and every checkpoint survive, so the
    cost of being wrong about this is one `jl resume`, while the cost of not
    doing it is the whole balance.
    """
    log(f"{now()}  STOP  {reason} | {body}")
    try:
        box.fetch(f"{RROOT}/runs/grid_l4.log", ROOT / "runs" / "grid_l4.log", timeout=900)
    except Exception:  # noqa: BLE001 - the log is evidence, not a precondition
        pass
    try:
        box.pause()
        log(f"{now()}  STOP  instance {box.machine_id} paused. Nothing is lost — after fixing it: "
            f"jl resume {box.machine_id} --spot --yes, then rerun jarvislabs/autostart_v2.sh")
    except Exception as exc:  # noqa: BLE001
        log(f"{now()}  FAIL  COULD NOT PAUSE instance {box.machine_id} ({exc!r}) — it is still billing, "
            f"pause it by hand at jarvislabs.ai")
    state.set(halted=True, halted_reason=reason)
    return True


def poll(box: Box, state: State, args) -> bool:
    """One sample. Returns True when the watch is over."""
    started = state.get("started_at", time.time())
    hours = (time.time() - started) / 3600

    inst = box.get()
    status = str(inst.status)
    cost = float(inst.cost or 0)
    rate = burn_rate(state, cost)
    state.set(burn_rate=rate)
    try:
        bal = box.balance()
    except Exception:  # noqa: BLE001
        bal = None
    runway = (bal / rate) if (bal is not None and rate > 0) else float("inf")
    money = f"cost={cost:.2f} burn={rate:.3f}/h bal={'?' if bal is None else f'{bal:.2f}'} runway={runway:.1f}h"

    if status != "Running":
        if state.get("finished"):
            log(f"{now()}  OK    finished; instance {box.machine_id} is {status} | {money}")
            return True
        log(f"{now()}  WARN  instance {box.machine_id} is {status} — resuming (spot reclaim?) | {money}")
        new_id = box.resume(spot=not args.ondemand_resume)
        state.set(machine_id=new_id)
        time.sleep(20)
        out = box.sh(LAUNCH)
        log(f"{now()}  INFO  resumed as {new_id}, wrapper relaunched: {out.stdout.strip()[:60]}")
        return False

    # Pause while there is still credit left to resume with: a balance that hits
    # zero mid-run is the one failure that can cost the disk.
    if bal is not None and rate > 0 and runway < args.min_runway_hours:
        log(f"{now()}  FAIL  runway {runway:.1f} h < {args.min_runway_hours} h — pausing to protect "
            f"the disk. Top up, then: jl resume {box.machine_id} --spot | {money}")
        box.pause()
        state.set(paused_for_funds=True)
        return True

    p = box.sh(PROBE)
    if p.returncode != 0:
        log(f"{now()}  WARN  ssh probe failed: {p.stderr.strip()[:120]} | {money}")
        box.ssh_parts(refresh=True)
        return False

    parsed = probe_fields(p.stdout)
    f = parsed["f"]
    done = f.get("MARK_DONE") == "1"
    failed = f.get("MARK_FAIL") == "1"
    alive = f.get("ALIVE", "0") not in ("0", "")
    active = f.get("ACTIVE", "")
    epoch, it_rate, attempt = parse_progress(parsed["log"])
    c = curve_stats(parsed["curve"])

    n_rows = pull_results(box, state)
    bad = check_rows(csv_rows(LOCAL_CSV))

    if time.time() - float(state.get("last_ckpt_at", 0)) > args.ckpt_hours * 3600 and active:
        pull_checkpoint(box, state, active)

    body = (f"id={box.machine_id} rows={n_rows}/3 run={Path(active.rstrip('/')).name or '?'} "
            f"ep={epoch} rate={it_rate} gpu={f.get('GPU', '?')} "
            f"epochs={c.get('epochs', 0)} attempt={attempt} up={hours:.1f}h {money}")

    # A row on the wrong split/version is worthless and every further hour spent
    # producing more of them is wasted money — this one always needs a human.
    if bad:
        return halt(box, state, f"incomparable row: {bad}", body)

    if done or n_rows >= len(SEEDS):
        state.set(finished=True)
        log(f"{now()}  OK    grid complete — pausing instance | {body}")
        box.fetch(f"{RROOT}/runs/grid_l4.log", ROOT / "runs" / "grid_l4.log")
        box.pause()
        log(f"{now()}  OK    instance {box.machine_id} paused. Storage still bills while paused — "
            f"destroy it once the rows are merged.")
        return True

    if failed:
        # 40 consecutive failed attempts is not something another poll fixes.
        return halt(box, state, "wrapper gave up (runs/GRID_FAILED)", body)

    if not alive:
        strikes = int(state.get("relaunch_strikes", 0)) + 1
        state.set(relaunch_strikes=strikes)
        if strikes > args.max_relaunches:
            return halt(box, state,
                        f"trainer would not stay up after {strikes} relaunches", body)
        log(f"{now()}  WARN  no trainer process — relaunching ({strikes}/{args.max_relaunches}) | {body}")
        box.sh(LAUNCH)
        return False
    if state.get("relaunch_strikes"):
        state.set(relaunch_strikes=0)  # it is training again; forget the strikes

    if hours > args.max_hours:
        log(f"{now()}  FAIL  budget cap {args.max_hours} h reached — pausing | {body}")
        box.pause()
        return True

    log(f"{now()}  OK    {body}")
    return False


def ensure_training(box: Box, state: State, args) -> bool:
    """Get the grid running before supervising it. True if it is training.

    The predicate for giving up is deliberately asymmetric. Confirmation is a live
    iteration counter — but *absence* of one is never enough to pause a paid run,
    because a broken probe looks exactly like a dead trainer. This halts only on
    positively observed idleness: no trainer process AND an idle GPU, seen
    repeatedly, over a working connection. An earlier bash version got this
    backwards and would have paused a perfectly healthy epoch-1 run because its
    regex missed.
    """
    for attempt in range(1, args.launch_attempts + 1):
        p = box.sh(PROBE)
        if p.returncode == 0:
            f = probe_fields(p.stdout)["f"]
            epoch, _, _ = parse_progress(probe_fields(p.stdout)["log"])
            if f.get("ALIVE", "0") not in ("0", "") and epoch != "?":
                log(f"{now()}  OK    training already running at epoch {epoch}")
                return True

        log(f"{now()}  INFO  launch attempt {attempt}/{args.launch_attempts}")
        box.sh(LAUNCH)

        # 59k labels are scanned and cached before the first frame; allow ~20 min.
        idle_strikes = 0
        for _ in range(40):
            time.sleep(30)
            p = box.sh(PROBE)
            if p.returncode != 0:
                log(f"{now()}  WARN  probe failed while confirming launch — not counted "
                    f"against the run: {p.stderr.strip()[:100]}")
                box.ssh_parts(refresh=True)
                continue
            parsed = probe_fields(p.stdout)
            f, epoch = parsed["f"], parse_progress(parsed["log"])[0]
            alive = f.get("ALIVE", "0") not in ("0", "")
            gpu_busy = not f.get("GPU", "").strip().startswith("0 %")
            if alive and (epoch != "?" or gpu_busy):
                log(f"{now()}  OK    training confirmed: epoch {epoch}, gpu {f.get('GPU')}")
                state.set(relaunch_strikes=0)
                return True
            if not alive and not gpu_busy:
                idle_strikes += 1
                if idle_strikes >= 4:  # 2 min of positively idle box
                    log(f"{now()}  WARN  box is idle (no trainer, gpu {f.get('GPU')}) on attempt {attempt}")
                    break
            if re.search(r"CUDA out of memory|Traceback|AssertionError|RuntimeError",
                         parsed["log"] or ""):
                log(f"{now()}  WARN  error signature in the training log on attempt {attempt}")
                break

    halt(box, state, f"grid would not start after {args.launch_attempts} attempts "
                     f"(no trainer process, idle GPU)")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=int, default=None)
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true", help="print one human-readable snapshot and exit")
    ap.add_argument("--max-hours", type=float, default=130.0,
                    help="hard cap: pause the instance after this many hours of watching")
    ap.add_argument("--min-runway-hours", type=float, default=2.0,
                    help="pause the instance when the balance would run out within this many hours")
    ap.add_argument("--ckpt-hours", type=float, default=6.0,
                    help="how often to pull the in-flight run's last.pt back to the laptop")
    ap.add_argument("--ondemand-resume", action="store_true",
                    help="if spot capacity is gone, resume on-demand (2.3x price) instead of waiting")
    ap.add_argument("--max-relaunches", type=int, default=3,
                    help="pause the instance after this many failed wrapper relaunches")
    ap.add_argument("--clear-halt", action="store_true",
                    help="resume supervising an instance a previous halt stopped (after fixing the cause)")
    ap.add_argument("--launch", action="store_true",
                    help="start the grid (if it is not already training) before supervising — "
                         "use after a resume, or as the first launch once the upload verifies")
    ap.add_argument("--launch-attempts", type=int, default=3)
    args = ap.parse_args()

    state = State(STATE)
    mid = args.machine_id or state.get("machine_id")
    if not mid:
        print("no machine id: pass --machine-id (it is remembered afterwards)", file=sys.stderr)
        return 2
    state.set(machine_id=int(mid))
    if not state.get("started_at"):
        state.set(started_at=time.time())

    # A halted instance is paused on purpose. Restarting the supervisor would see
    # "Paused and unfinished" and resume it straight back into the same failure.
    if state.get("halted") and not (args.clear_halt or args.once):
        print(f"instance {mid} was halted: {state.get('halted_reason')}\n"
              f"fix the cause, then rerun with --clear-halt", file=sys.stderr)
        return 3
    if args.clear_halt:
        state.set(halted=False, halted_reason=None, relaunch_strikes=0)

    box = Box(int(mid))

    if args.once:
        # During the upload the instance has nothing to report yet, and probing it
        # would just print an empty run — show the transfer instead.
        up = sync_report()
        if up:
            inst = box.get()
            print(f"instance {box.machine_id}  {inst.status}  {inst.gpu_type}"
                  f"{' spot' if inst.is_spot else ''}   cost {inst.cost:.2f}")
            print(up)
            return 0
        print(report(snapshot(box, state), state))
        return 0

    log(f"{now()}  INFO  watching machine {mid}: interval {args.interval}s, cap {args.max_hours}h, "
        f"pause below {args.min_runway_hours}h of runway")

    if args.launch and not ensure_training(box, state, args):
        return 1
    while True:
        try:
            if poll(box, state, args):
                return 0
            state.set(machine_id=box.machine_id)
        except Exception as exc:  # noqa: BLE001 - a watchdog must not die on its own bug
            log(f"{now()}  WARN  watchdog error: {exc!r}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
