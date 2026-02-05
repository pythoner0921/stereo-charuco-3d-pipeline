from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ── Check dependencies early with clear messages ──
try:
    import caliscope.core  # noqa: F401
except ImportError:
    raise ImportError(
        "\n\ncaliscope is not installed (or wrong version).\n"
        "Install the pinned version:\n\n"
        "  pip install git+https://github.com/mprib/caliscope.git@8dc0cd4e\n\n"
        "PyPI caliscope is outdated — you must install from GitHub.\n"
    )

import cv2
if not hasattr(cv2, "aruco"):
    raise ImportError(
        "\n\ncv2.aruco module is missing.\n"
        "This happens because ultralytics installs opencv-python (no aruco).\n"
        "Fix:\n\n"
        "  pip uninstall opencv-python opencv-python-headless -y\n"
        "  pip install opencv-contrib-python>=4.8.0.74\n"
    )

from .config import RecorderConfig
from .ffmpeg import FFmpegRunner, python_executable

# Calibration UI exports
from .calibration_ui import CalibrationUI, CalibConfig

# Pipeline exports
from .auto_calibrate import AutoCalibConfig, run_auto_calibration, run_auto_reconstruction
from .pipeline_ui import PipelineUI

# Project management exports
from .project_manager import ProjectManagerUI, ProjectContext

# Smart recorder exports
from .smart_recorder import SmartRecorder, SmartRecorderConfig


EventCallback = Callable[[dict], None]


@dataclass
class PipelineResult:
    returncode: int
    script_path: str
    session_hint: str
    cmd: list[str]


def _project_root() -> Path:
    """Find the calib_record_tool root directory.

    Works in dev mode (running from repo) and after pip install
    (falls back to cwd).
    """
    from .paths import _dev_root
    dev = _dev_root()
    if dev:
        return dev
    return Path.cwd()


def run_pipeline(cfg: RecorderConfig, on_event: Optional[EventCallback] = None) -> PipelineResult:
    """
    Stable API entrypoint. Runs the existing scripts/record_and_split.py pipeline.

    on_event receives dicts like:
      {"stage": "run", "message": "..."}
      {"stage": "log", "message": "..."}
      {"stage": "done", "message": "..."}
      {"stage": "error", "message": "..."}
    """
    root = _project_root()
    script = root / "scripts" / "record_and_split.py"
    if not script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script}")

    runner = FFmpegRunner()

    cmd = [python_executable(), str(script)] + cfg.to_args()

    def emit(stage: str, message: str) -> None:
        if on_event:
            on_event({"stage": stage, "message": message})

    emit("run", "Starting pipeline...")
    emit("run", f"CMD: {' '.join(cmd)}")

    # Stream output for UI
    rc = runner.run_stream(cmd, on_log=lambda line: emit("log", line), cwd=str(root))

    if rc == 0:
        emit("done", "Pipeline finished successfully.")
    else:
        emit("error", f"Pipeline failed. returncode={rc}")

    # session_hint：你现在脚本里 session 目录可能由 outroot 或默认规则决定，这里先给一个“提示”
    session_hint = cfg.outroot if cfg.outroot else str(Path(cfg.outdir).resolve())

    return PipelineResult(
        returncode=rc,
        script_path=str(script),
        session_hint=session_hint,
        cmd=cmd,
    )
