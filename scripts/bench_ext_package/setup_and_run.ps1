# Benchmark extension, portable package: install the pinned environment, check the data, start the
# dashboard and the queue. Safe to run again at any time: a finished run is skipped and an interrupted
# run resumes from its last epoch with its patience intact.
#
#   START.bat                       (double-click; same as the line below)
#   powershell -ExecutionPolicy Bypass -File setup_and_run.ps1 [-DryRun] [-NoDashboard] [-Port 8771]
#
# -DryRun does everything except train: unpacking, environment, GPU check, data verification, run list.
# From a removable drive it first offers to unpack into C:\Training (-CopyTo <dir> skips the question,
# -NoCopy unpacks and runs on the drive itself).
param([switch]$DryRun, [switch]$NoDashboard, [int]$Port = 8771, [string]$CopyTo = "", [switch]$NoCopy)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Out = Join-Path $Root "runs\vis_benchmark_stride4_ep25_ext"
$EnvDir = Join-Path $env:LOCALAPPDATA "uqfusion_bench_ext"
$Venv = Join-Path $EnvDir "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Req = Join-Path $Root "requirements.txt"
$TorchIndex = "https://download.pytorch.org/whl/cu118"

function Say($msg) { Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host "[setup] FAILED: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------------- 0. unpack
# The data ships as payload.tar: the stick this was built on writes ~70k small files at ~1.3 MB/s but one
# sequential file at ~4 MB/s. It is unpacked once per package build, and from a removable drive into a
# local folder (default C:\Training), because every epoch reads ~9 GB of images. Training output is
# never copied, so launching from the stick again cannot overwrite progress made in the local copy.
$Stamp = (Get-Content (Join-Path $Root "package_manifest.json") -Raw | ConvertFrom-Json).built_at
$Local = $Root
$vol = Get-Volume -DriveLetter $Root.Substring(0, 1) -ErrorAction SilentlyContinue
if ($CopyTo) { $Local = $CopyTo }
elseif (-not $NoCopy -and $vol -and $vol.DriveType -eq "Removable") {
    $Local = "$env:SystemDrive\Training"
    Say "This package is on a removable drive ($($Root.Substring(0, 2))), too slow to train from."
    $a = Read-Host "Unpack it to $Local and run from there? [Y = yes / N = run on the drive / or type another folder]"
    if ($a -match "^[Nn]$") { $Local = $Root } elseif ($a -and $a -notmatch "^[Yy]$") { $Local = $a }
}
$Local = [IO.Path]::GetFullPath($Local).TrimEnd("\")
$here = $Local -eq [IO.Path]::GetFullPath($Root).TrimEnd("\")
$unpacked = Join-Path $Local ".unpacked"
if (-not ((Test-Path $unpacked) -and ((Get-Content $unpacked -Raw).Trim() -eq $Stamp))) {
    $payload = Join-Path $Root "payload.tar"
    if (-not (Test-Path $payload)) { Fail "payload.tar not found next to this script" }
    $tar = Join-Path $env:SystemRoot "System32\tar.exe"          # not Git's GNU tar, which reads C: as a host
    if (-not (Test-Path $tar)) { Fail "$tar not found (Windows 10 1803 or later has it)" }
    New-Item -ItemType Directory -Force $Local | Out-Null
    $need = (Get-Item $payload).Length + 1GB
    $free = (Get-PSDrive $Local.Substring(0, 1)).Free
    if ($free -lt $need) { Fail "$([math]::Round($free / 1GB, 1)) GB free on $($Local.Substring(0, 2)); need $([math]::Round($need / 1GB, 1)) GB" }
    if (-not $here) {
        robocopy $Root $Local /E /R:2 /W:2 /NFL /NDL /NP /NJH /XF payload.tar .verified.json .unpacked `
            /XD (Join-Path $Root "runs") (Join-Path $Root "Pohang_dataset") (Join-Path $Root "server_dgxanode01")
        if ($LASTEXITCODE -ge 8) { Fail "robocopy to $Local (code $LASTEXITCODE)" }
    }
    Say "Unpacking the data into $Local (one time, a few minutes)"
    & $tar -xf $payload -C $Local
    if ($LASTEXITCODE -ne 0) { Fail "tar -xf payload.tar" }
    Remove-Item (Join-Path $Local ".verified.json") -ErrorAction SilentlyContinue      # verify the new copy
    Set-Content -Path $unpacked -Value $Stamp -Encoding ascii
}
if (-not $here) {
    Say "Running from $Local. Next time start it with $Local\START.bat"
    $fwd = @("-NoCopy", "-Port", "$Port")
    if ($DryRun) { $fwd += "-DryRun" }
    if ($NoDashboard) { $fwd += "-NoDashboard" }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Local "setup_and_run.ps1") @fwd
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------------- 1. Python 3.13
function Find-Python313 {
    $ErrorActionPreference = "Continue"      # a missing runtime writes to stderr; that is an answer, not an error
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $p = & py -3.13 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe", "$env:ProgramFiles\Python313\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
        $v = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -eq "3.13") { return $cmd.Source }
    }
    return $null
}

$Base = Find-Python313
if (-not $Base) {
    Say "Python 3.13 was not found."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "install Python 3.13 (64-bit) from https://www.python.org/downloads/ and run this again"
    }
    $a = Read-Host "Install Python 3.13 for this user with winget now? [Y/n]"
    if ($a -and $a -notmatch "^[Yy]") { Fail "Python 3.13 is required" }
    & winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
    $Base = Find-Python313
    if (-not $Base) { Fail "Python 3.13 still not found after winget; open a new window and run START.bat again" }
}
Say "Python: $Base"

# ---------------------------------------------------------------------------------- 2. GPU
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { Fail "nvidia-smi not found: install the NVIDIA driver first" }
$gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
Say "GPU: $gpu"

# ---------------------------------------------------------------------------------- 3. pinned environment
$reqHash = (Get-FileHash $Req -Algorithm SHA256).Hash
$marker = Join-Path $EnvDir "installed.txt"
$installed = (Test-Path $Py) -and (Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $reqHash)
if (-not $installed) {
    Say "Creating the environment in $Venv (one time, ~3 GB download)"
    New-Item -ItemType Directory -Force $EnvDir | Out-Null
    if (-not (Test-Path $Py)) {
        & $Base -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Fail "python -m venv" }
    }
    & $Py -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade" }
    & $Py -m pip install -r $Req --extra-index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt" }
    Set-Content -Path $marker -Value $reqHash -Encoding ascii
}
$check = @"
import sys, torch, torchvision, ultralytics, numpy, cv2
want = {'torch': '2.7.1+cu118', 'torchvision': '0.22.1+cu118', 'ultralytics': '8.4.90', 'numpy': '2.1.3', 'cv2': '4.10.0'}
got = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'ultralytics': ultralytics.__version__,
       'numpy': numpy.__version__, 'cv2': cv2.__version__}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
print('versions', got)
if bad: sys.exit('version mismatch (got, want): %s' % bad)
if not torch.cuda.is_available(): sys.exit('torch sees no CUDA GPU')
p = torch.cuda.get_device_properties(0)
print('cuda', torch.version.cuda, p.name, round(p.total_memory / 2**30, 1), 'GB')
"@
$checkPy = Join-Path $EnvDir "check_env.py"
Set-Content -Path $checkPy -Value $check -Encoding ascii
& $Py $checkPy
if ($LASTEXITCODE -ne 0) {
    Remove-Item $marker -ErrorAction SilentlyContinue
    Fail "environment check (run again to reinstall)"
}

# ---------------------------------------------------------------------------------- 4. data
# Full sha256 verification once per location (after the copy or a move), then the per-run gate
# (bench_ext_fingerprint.py --quick, run by the queue) checks the data before every run.
$verified = $false
$vf = Join-Path $Root ".verified.json"
if (Test-Path $vf) { $verified = ((Get-Content $vf -Raw | ConvertFrom-Json).root -eq $Root) }
if ($verified) { & $Py scripts\prepare_data.py } else {
    Say "Verifying every packaged file against package_manifest.json (one time, a few minutes)"
    & $Py scripts\prepare_data.py --verify
}
if ($LASTEXITCODE -ne 0) {
    Remove-Item (Join-Path $Root ".unpacked") -ErrorAction SilentlyContinue       # the next launch unpacks again
    Fail "data verification: files differ from the manifest. Run START.bat from the stick again to re-unpack"
}
& $Py -u scripts\bench_ext_fingerprint.py --quick
if ($LASTEXITCODE -ne 0) { Fail "fingerprint gate: the data differs from the benchmark's" }
& $Py scripts\bench_ext_local.py --list
if ($LASTEXITCODE -ne 0) { Fail "run list" }

if ($DryRun) { Say "Dry run: everything checks out; not training."; exit 0 }

# ---------------------------------------------------------------------------------- 5. dashboard + queue
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine }
$running = $procs | Where-Object { $_.CommandLine -match "bench_ext_local\.py\s+--queue" -and $_.ExecutablePath -eq $Py }
if ($running) { Fail "a queue is already running (pid $($running.ProcessId -join ', ')); use the dashboard" }

$url = "http://127.0.0.1:$Port"
if (-not $NoDashboard) {
    $dash = $procs | Where-Object { $_.CommandLine -match "dashboard\.py" -and $_.CommandLine -match "--port $Port" }
    if ($dash) { Start-Process $url } else {
        Start-Process -FilePath $Py -WorkingDirectory $Root -WindowStyle Minimized -ArgumentList `
            "-u", "scripts\dashboard.py", "--queue-dir", "runs\vis_benchmark_stride4_ep25_ext", "--port", "$Port", "--open"
    }
    Say "Dashboard: $url  (Pause there stops after the current epoch; Resume continues it)"
}

Say "Starting the queue. Keep this window open and the laptop plugged in with the lid open."
Say "Closing this window stops training; run START.bat again to resume where it stopped."
Say "Per-run training logs: $Out\logs"
& $Py -u scripts\bench_ext_local.py --queue
$rc = $LASTEXITCODE
if ($rc -eq 0) {
    Say "Queue finished. Bring back the folder:  $Out"
} else {
    Say "Queue stopped with code $rc; see $Out\queue.log"
}
exit $rc
