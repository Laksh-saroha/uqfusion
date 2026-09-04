"""Configuration loading for uqfusion.

One YAML file (repo-root ``config.yaml`` by default) holds every
machine-specific path and hyperparameter. Relative paths are resolved
against the config file's own directory, so a cloned repo works from any
location with at most an edit to the ``paths:`` block.

Resolution order for which file to load:
    explicit argument > $UQFUSION_CONFIG > <repo root>/config.yaml

Smoke test:
    python -m uqfusion.config [path/to/config.yaml]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

ENV_VAR = "UQFUSION_CONFIG"

# Sections that must exist for any pipeline code to run.
REQUIRED_SECTIONS = ("paths", "datasets", "benchmark", "smoke", "reliability")


def _repo_default() -> Path:
    # src/uqfusion/config.py -> repo root is two levels up from this file's dir
    return Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the project config and resolve all paths to absolute strings."""
    cfg_path = Path(path or os.environ.get(ENV_VAR) or _repo_default()).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"config {cfg_path} is missing required sections: {missing}")

    base = cfg_path.parent

    def _absolute(value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else (base / p).resolve())

    paths = cfg["paths"]
    for key, value in paths.items():
        paths[key] = _absolute(str(value))

    # Expand "{data_root}"-style references inside dataset entries, then absolutize.
    for entry in cfg["datasets"].values():
        for key, value in entry.items():
            if not isinstance(value, str):
                continue  # null yamls (e.g. mit_marine before onboarding) stay null
            for name, root in paths.items():
                value = value.replace("{" + name + "}", root)
            entry[key] = _absolute(value)

    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_gpu_python(cfg: dict[str, Any] | None = None, require_cuda: bool = True) -> str:
    """The interpreter GPU work must be launched with — never bare `sys.executable`.

    The dev `.venv` is the torch+CPU build; a chain script that inherits it
    trains at a ~20x penalty (followup-analysis-2026-08-20.md §9). `gpu_python`
    in config.yaml names the CUDA interpreter for this machine; null falls back
    to `sys.executable` (correct on the server). With `require_cuda`, the
    interpreter is probed for `torch.cuda.is_available()` before being returned,
    so the failure is a loud refusal instead of a silent 20x slowdown.
    """
    import subprocess

    cfg = cfg or load_config()
    py = cfg.get("gpu_python") or sys.executable
    py = str(Path(py))
    if not Path(py).is_file():
        raise FileNotFoundError(f"gpu_python interpreter not found: {py} — fix config.yaml")
    if require_cuda:
        probe = subprocess.run(
            [py, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)"],
            capture_output=True, text=True, timeout=120)
        if probe.returncode == 3:
            raise RuntimeError(f"{py} has torch WITHOUT CUDA — refusing to launch GPU work on it")
        if probe.returncode != 0:
            raise RuntimeError(f"could not probe {py} for CUDA torch: {probe.stderr.strip()[:500]}")
    return py


def resolve_data_yaml(cfg: dict[str, Any], data: str) -> str:
    """Map a CLI --data value to a dataset yaml path.

    'vis' / 'ir' are aliases for the Pohang per-modality yamls (scope §5.1);
    anything else is taken as an explicit path (e.g. a derived stride yaml).
    """
    aliases = {
        "vis": cfg["datasets"]["pohang"]["vis_yaml"],
        "ir": cfg["datasets"]["pohang"]["ir_yaml"],
    }
    resolved = aliases.get(data, data)
    if resolved is None:
        raise ValueError(f"dataset alias '{data}' resolves to null in config.yaml")
    if not Path(resolved).is_file():
        raise FileNotFoundError(
            f"dataset yaml not found: {resolved} — is the data present on this machine "
            "and laid out per dataset_requirement.md?"
        )
    return str(resolved)


def main(argv: list[str] | None = None) -> int:
    """Structural smoke test: load, resolve, report. Fails on malformed config;
    warns (does not fail) when data folders are absent, since the datasets only
    exist on the training server."""
    argv = sys.argv[1:] if argv is None else argv
    cfg = load_config(argv[0] if argv else None)

    print(f"config OK: {cfg['_config_path']}")
    for key, value in cfg["paths"].items():
        tag = "" if Path(value).exists() else "   [missing on this machine — expected off the training server]"
        print(f"  paths.{key} = {value}{tag}")
    for ds_name, entry in cfg["datasets"].items():
        for key, value in entry.items():
            if value is None:
                print(f"  datasets.{ds_name}.{key} = null (not onboarded yet)")
                continue
            tag = "" if Path(value).exists() else "   [missing on this machine]"
            print(f"  datasets.{ds_name}.{key} = {value}{tag}")
    b = cfg["benchmark"]
    print(f"  benchmark: {len(b['variants'])} variants x {len(b['seeds'])} seeds, "
          f"imgsz={b['imgsz']}, epochs={b['epochs']}, batch={b['batch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
