"""
Centralized path resolution for the recorder package.

Handles finding configs and tools in both development mode
(running from repo clone) and installed mode (pip install).
"""
from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def tool_root() -> Path:
    """Return the calib_record_tool root directory.

    In dev mode (repo clone), returns the actual calib_record_tool/ dir.
    After pip install, returns the current working directory.
    """
    dev = _dev_root()
    return dev if dev else Path.cwd()


def _dev_root() -> Path | None:
    """Return calib_record_tool/ root if running from a repo clone."""
    # __file__ = .../calib_record_tool/src/recorder/paths.py
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs" / "default.yaml").exists():
        return candidate
    return None


def resolve_default_config() -> Path:
    """Find the default.yaml config file.

    Search order:
      1. Package data (recorder/configs/default.yaml) — works after pip install
      2. Repo layout (calib_record_tool/configs/default.yaml) — works in dev
    """
    # Try package data first (works after pip install from wheel)
    try:
        pkg_configs = resources.files("recorder") / "configs" / "default.yaml"
        p = Path(str(pkg_configs))
        if p.exists():
            return p
    except (TypeError, FileNotFoundError, AttributeError):
        pass

    # Fallback: repo layout
    dev = _dev_root()
    if dev:
        p = dev / "configs" / "default.yaml"
        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find default.yaml config. "
        "Pass --config explicitly or run from the repo directory."
    )


def resolve_ffmpeg() -> str:
    """Find ffmpeg executable.

    Search order:
      1. System PATH (preferred — works cross-platform)
      2. Repo-local bin/ directory (development fallback)
    """
    # System PATH first
    which = shutil.which("ffmpeg")
    if which:
        return which

    # Dev fallback: bin/ffmpeg/ffmpeg.exe
    dev = _dev_root()
    if dev:
        local = dev / "bin" / "ffmpeg" / "ffmpeg.exe"
        if local.exists():
            return str(local)

    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg and ensure it is on your PATH.\n"
        "  Windows:  winget install ffmpeg  OR  choco install ffmpeg\n"
        "  Linux:    sudo apt install ffmpeg\n"
        "  macOS:    brew install ffmpeg\n"
        "  Conda:    conda install -c conda-forge ffmpeg"
    )
