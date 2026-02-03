from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


LogCallback = Callable[[str], None]


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    cmd: list[str]


class FFmpegRunner:
    """
    Minimal runner for subprocess-based tools (ffmpeg, python scripts, etc.)
    - supports live log streaming
    - supports cancellation
    """

    def __init__(self, executable: Optional[str] = None):
        self.executable = executable
        self._proc: Optional[subprocess.Popen] = None

    @classmethod
    def resolve_ffmpeg(cls) -> str:
        """
        Resolve ffmpeg path:
        1) System PATH (cross-platform)
        2) Repo-local bin/ directory (development fallback)
        """
        from .paths import resolve_ffmpeg
        return resolve_ffmpeg()

    def run_capture(
        self,
        cmd: list[str],
        cwd: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> RunResult:
        """
        Run synchronously and capture stdout/stderr fully.
        """
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return RunResult(
            returncode=p.returncode,
            stdout=p.stdout or "",
            stderr=p.stderr or "",
            cmd=cmd,
        )

    def run_stream(
        self,
        cmd: list[str],
        on_log: Optional[LogCallback] = None,
        cwd: Optional[str | Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> int:
        """
        Run and stream combined stdout/stderr line-by-line to on_log.
        Returns process return code.
        """
        # Merge env
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            if on_log:
                on_log(line)

        rc = self._proc.wait()
        self._proc = None
        return rc

    def cancel(self) -> None:
        """
        Best-effort termination of the running process.
        """
        if not self._proc:
            return

        try:
            self._proc.terminate()
        except Exception:
            pass

        try:
            self._proc.wait(timeout=2)
            self._proc = None
            return
        except Exception:
            pass

        try:
            self._proc.kill()
        except Exception:
            pass
        finally:
            self._proc = None


def python_executable() -> str:
    """
    Always use the current interpreter (e.g. conda env) to run scripts.
    """
    return sys.executable
