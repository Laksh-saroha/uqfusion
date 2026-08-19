"""A small local dashboard for the training queue.

Launch it yourself, leave it open in a browser tab:

    python scripts/dashboard.py                 # http://127.0.0.1:8770
    python scripts/dashboard.py --port 9000 --open

It shows what is left to run, how far the current run has got, and gives you a
Pause / Resume button. It has no dependencies beyond the standard library and no
authority of its own: it reads the queue files that `scripts/run_queue.py` writes
and, for pause/resume, flips one boolean in `runs/queue/control.json`. The runner
picks that up at its next checkpoint boundary. Closing the dashboard, or never
opening it, changes nothing about a run in progress.

Bound to 127.0.0.1 — it is a local instrument panel, not a service.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = ROOT / "runs" / "queue"
QUEUE_JSON = QUEUE_DIR / "queue.json"
STATE_JSON = QUEUE_DIR / "state.json"
LIVE_JSON = QUEUE_DIR / "live.json"
CONTROL_JSON = QUEUE_DIR / "control.json"

TERMINAL = {"done", "failed", "skipped"}


def set_queue_dir(path: str | Path) -> None:
    """Watch a different queue directory (mirrors run_queue.py --queue-dir)."""
    global QUEUE_DIR, QUEUE_JSON, STATE_JSON, LIVE_JSON, CONTROL_JSON
    QUEUE_DIR = Path(path)
    QUEUE_JSON = QUEUE_DIR / "queue.json"
    STATE_JSON = QUEUE_DIR / "state.json"
    LIVE_JSON = QUEUE_DIR / "live.json"
    CONTROL_JSON = QUEUE_DIR / "control.json"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_control(paused: bool) -> bool:
    """Flip the pause flag. Same Windows rename hazard as the runner's writes: if
    the runner has control.json open when we rename onto it, `replace` raises
    WinError 5. Retry, then write in place — the runner tolerates a torn read by
    keeping its last good value."""
    ctl = read_json(CONTROL_JSON, {}) or {}
    ctl["paused"] = paused
    ctl["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    ctl["source"] = "dashboard"
    payload = json.dumps(ctl, indent=2)
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CONTROL_JSON.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(CONTROL_JSON)
                return True
            except PermissionError:
                time.sleep(0.05)
        CONTROL_JSON.write_text(payload, encoding="utf-8")
        tmp.unlink(missing_ok=True)
        return True
    except OSError as exc:
        print(f"dashboard: could not write control.json: {exc}")
        return False


def snapshot() -> dict:
    queue = read_json(QUEUE_JSON, {"runs": [], "defaults": {}})
    state = read_json(STATE_JSON, {"runs": {}})
    live = read_json(LIVE_JSON, {})
    ctl = read_json(CONTROL_JSON, {"paused": False})
    defaults = queue.get("defaults", {})
    epochs_cfg = int(defaults.get("epochs", 0) or 0)
    patience = int(defaults.get("patience", 0) or 0)

    rows, remaining = [], 0
    for spec in queue.get("runs", []):
        rs = state.get("runs", {}).get(spec["id"], {})
        status = rs.get("status", "pending")
        done_ep = rs.get("epochs_done")
        best_ep = rs.get("best_epoch")
        # Early stopping usually fires well before epochs_cfg; the honest ceiling
        # for "when does this run end" is best_epoch + patience, capped.
        stops_by = min(epochs_cfg, best_ep + patience) if (best_ep is not None and patience) else None
        if status not in TERMINAL:
            remaining += 1
        rows.append({
            "id": spec["id"],
            "sigma": bool(spec.get("sigma", defaults.get("sigma", True))),
            "data": Path(str(spec.get("data", ""))).name,
            "status": status,
            "epochs_done": done_ep,
            "epochs_cfg": epochs_cfg,
            "best_epoch": best_ep,
            "patience_gap": rs.get("patience_gap"),
            "stops_by": stops_by,
            "map50_95": rs.get("map50_95"),
            "epoch_time_s": rs.get("epoch_time_s"),
            "started": rs.get("started"),
            "finished": rs.get("finished"),
            "error": rs.get("error"),
            "note": rs.get("note"),
        })

    return {
        "queue_status": state.get("queue_status", "not started"),
        "paused": bool(ctl.get("paused")),
        "updated": state.get("updated"),
        "pid": state.get("pid"),
        "defaults": defaults,
        "remaining": remaining,
        "runs": rows,
        "live": live,
        "server_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>uqfusion queue</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#16191d; --muted:#6b7280; --line:#e3e6ea;
          --run:#1d4ed8; --done:#15803d; --fail:#b91c1c; --pause:#b45309; --bar:#1d4ed8; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#111418; --card:#181c22; --ink:#e8eaed; --muted:#9aa3ae; --line:#272c33;
            --run:#60a5fa; --done:#4ade80; --fail:#f87171; --pause:#fbbf24; --bar:#60a5fa; }
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:22px; background:var(--bg); color:var(--ink);
         font:14px/1.45 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
  h1 { font-size:16px; margin:0 0 2px; font-weight:650; }
  .sub { color:var(--muted); font-size:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:14px; }
  .top { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  .top .grow { flex:1; min-width:200px; }
  button { font:inherit; font-weight:600; padding:9px 18px; border-radius:8px;
           border:1px solid var(--line); cursor:pointer; background:var(--card); color:var(--ink); }
  button.primary { background:var(--bar); border-color:var(--bar); color:#fff; }
  button:disabled { opacity:.45; cursor:default; }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; }
  th { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
       font-weight:600; border-bottom:1px solid var(--line); }
  tr:last-child td { border-bottom:none; }
  .st { font-weight:650; }
  .st.running { color:var(--run); } .st.done { color:var(--done); }
  .st.failed, .st.interrupted { color:var(--fail); } .st.paused { color:var(--pause); }
  .st.pending, .st.skipped { color:var(--muted); }
  .bar { height:7px; border-radius:4px; background:var(--line); overflow:hidden; margin-top:6px; }
  .bar > i { display:block; height:100%; background:var(--bar); transition:width .4s; }
  .kv { display:flex; gap:26px; flex-wrap:wrap; margin-top:10px; }
  .kv div span { display:block; color:var(--muted); font-size:11px; text-transform:uppercase;
                 letter-spacing:.05em; }
  .kv div b { font-size:17px; font-weight:650; font-variant-numeric:tabular-nums; }
  .err { color:var(--fail); font-size:12px; }
  .muted { color:var(--muted); }
</style></head><body>
<div class="card top">
  <div class="grow">
    <h1>uqfusion &mdash; training queue</h1>
    <div class="sub" id="hdr">loading&hellip;</div>
  </div>
  <button id="btn" class="primary" onclick="toggle()">&hellip;</button>
</div>

<div class="card" id="livecard" style="display:none">
  <div style="font-weight:650" id="liverun"></div>
  <div class="sub" id="liveep"></div>
  <div class="bar"><i id="epbar" style="width:0%"></i></div>
  <div class="kv" id="livekv"></div>
</div>

<div class="card">
  <table><thead><tr>
    <th>run</th><th>arm</th><th>data</th><th>status</th><th>epochs</th>
    <th>best</th><th>stops by</th><th>mAP50-95</th><th>min/epoch</th>
  </tr></thead><tbody id="rows"></tbody></table>
</div>
<div class="sub" id="foot"></div>

<script>
let paused = false;
function fmt(v, d) { return (v === null || v === undefined || v === '') ? (d || '\\u2013') : v; }
function hms(s) {
  if (!s && s !== 0) return '\\u2013';
  s = Math.round(s);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
  return h ? h+'h '+m+'m' : m+'m '+(s%60)+'s';
}
async function toggle() {
  const btn = document.getElementById('btn');
  btn.disabled = true;
  await fetch(paused ? '/api/resume' : '/api/pause', {method:'POST'});
  await tick();
  btn.disabled = false;
}
async function tick() {
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch (e) { document.getElementById('foot').textContent = 'dashboard lost the queue files'; return; }

  paused = s.paused;
  const d = s.defaults || {};
  document.getElementById('hdr').innerHTML =
    '<b>' + s.queue_status + '</b>' + (s.paused ? ' \\u2014 <b>pause requested</b>' : '') +
    ' &middot; ' + s.remaining + ' run(s) remaining &middot; ' +
    fmt(d.variant) + ' &middot; imgsz ' + fmt(d.imgsz) + ' &middot; batch ' + fmt(d.batch) +
    ' &middot; workers ' + fmt(d.workers) + ' &middot; epochs ' + fmt(d.epochs) +
    ' &middot; patience ' + fmt(d.patience);
  const btn = document.getElementById('btn');
  btn.textContent = paused ? 'Resume' : 'Pause';
  btn.className = paused ? 'primary' : '';

  const L = s.live || {};
  const card = document.getElementById('livecard');
  if (L.run_id) {
    card.style.display = '';
    document.getElementById('liverun').textContent = L.run_id;
    document.getElementById('liveep').textContent =
      'epoch ' + L.epoch + ' / ' + L.epochs + '  \\u00b7  batch ' + L.batch_i + ' / ' + L.batch_n +
      (L.pause_pending ? '  \\u00b7  will stop after this epoch' : '');
    document.getElementById('epbar').style.width = (100 * L.batch_i / Math.max(1, L.batch_n)) + '%';
    document.getElementById('livekv').innerHTML =
      '<div><span>throughput</span><b>' + fmt(L.img_s) + ' img/s</b></div>' +
      '<div><span>epoch eta</span><b>' + hms(L.epoch_eta_s) + '</b></div>' +
      '<div><span>gpu reserved</span><b>' + fmt(L.gpu_reserved_gb) + ' GB</b></div>';
  } else { card.style.display = 'none'; }

  document.getElementById('rows').innerHTML = s.runs.map(r =>
    '<tr><td><b>' + r.id + '</b>' +
      (r.error ? '<div class="err">' + r.error + '</div>' : '') +
      (r.note ? '<div class="sub">' + r.note + '</div>' : '') + '</td>' +
    '<td>' + (r.sigma ? '&sigma;' : 'parity') + '</td>' +
    '<td class="muted">' + r.data + '</td>' +
    '<td class="st ' + r.status + '">' + r.status + '</td>' +
    '<td>' + fmt(r.epochs_done, '0') + ' / ' + fmt(r.epochs_cfg) + '</td>' +
    '<td>' + fmt(r.best_epoch) + '</td>' +
    '<td>' + fmt(r.stops_by) + '</td>' +
    '<td>' + fmt(r.map50_95) + '</td>' +
    '<td>' + (r.epoch_time_s ? (r.epoch_time_s/60).toFixed(1) : '\\u2013') + '</td></tr>').join('');

  document.getElementById('foot').textContent =
    'queue updated ' + fmt(s.updated) + (s.pid ? '  \\u00b7  runner pid ' + s.pid : '') +
    '  \\u00b7  "stops by" = best epoch + patience, the earliest guaranteed end';
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/state"):
            self._send(200, json.dumps(snapshot()).encode(), "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/pause"):
            write_control(True)
        elif self.path.startswith("/api/resume"):
            write_control(False)
        else:
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, json.dumps({"ok": True}).encode(), "application/json")

    def log_message(self, *a):  # keep the console clean; the runner owns stdout
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true", help="open a browser tab")
    parser.add_argument("--queue-dir", default=None, help="default runs/queue")
    args = parser.parse_args()
    if args.queue_dir:
        set_queue_dir(args.queue_dir)

    if not QUEUE_JSON.is_file():
        print(f"note: no queue yet at {QUEUE_JSON} — "
              f"run `python scripts/run_queue.py init` (the dashboard will fill in once it exists).")

    url = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: {url}   (Ctrl-C to stop; stopping it does not stop training)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\ndashboard stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
