from __future__ import annotations

import re
from typing import List

from .ffmpeg import FFmpegRunner


def list_dshow_devices() -> List[str]:
    """
    List DirectShow video devices on Windows using ffmpeg.

    Returns a list of device names that you can pass to:
      ffmpeg -f dshow -i video="<NAME>"
    """
    ffmpeg = FFmpegRunner.resolve_ffmpeg()
    runner = FFmpegRunner()

    cmd = [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    res = runner.run_capture(cmd)

    # ffmpeg prints devices to stderr in many builds
    text = (res.stdout or "") + "\n" + (res.stderr or "")

    devices: List[str] = []
    in_video_section = False

    # Typical lines:
    # [dshow @ ...] DirectShow video devices
    # [dshow @ ...]  "3D USB Camera"
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
            continue

        if in_video_section:
            m = re.search(r'"([^"]+)"', line)
            if m:
                devices.append(m.group(1))

    # de-dup preserve order
    seen = set()
    out = []
    for d in devices:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out
