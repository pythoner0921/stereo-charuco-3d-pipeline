import argparse
import os
import subprocess
import sys
from datetime import datetime

try:
    import yaml
except ImportError:
    yaml = None


def _load_yaml(path: str) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please: pip install pyyaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML root must be a mapping (dict).")
    return data


def _get(cfg: dict, keys: list, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_tool_root() -> str:
    # tools/calig_record_tool/scripts/record_raw.py -> tools/calig_record_tool
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_ffmpeg(ffmpeg_exe: str, tool_root: str) -> str:
    if not ffmpeg_exe:
        return "ffmpeg"
    if os.path.isabs(ffmpeg_exe):
        return ffmpeg_exe
    candidate = os.path.abspath(os.path.join(tool_root, ffmpeg_exe))
    if os.path.exists(candidate):
        return candidate
    return ffmpeg_exe


def run():
    ap = argparse.ArgumentParser(description="Record raw side-by-side stereo video via FFmpeg (DirectShow). Supports configs/default.yaml.")
    ap.add_argument("--config", default="configs/default.yaml", help="path to YAML config (relative to tool root ok)")

    ap.add_argument("--device", default=None, help='DirectShow video device name, e.g. "3D USB Camera"')
    ap.add_argument("--size", default=None, help="video_size, e.g. 3200x1200")
    ap.add_argument("--fps", type=int, default=None, help="framerate, e.g. 60")
    ap.add_argument("--seconds", type=int, default=None, help="record duration in seconds")

    ap.add_argument("--outdir", default=None, help="output directory (base)")
    ap.add_argument("--out", default=None, help="output file path (default: outdir/raw_YYYYmmdd_HHMMSS.avi)")

    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable or relative path, e.g. bin/ffmpeg/ffmpeg.exe")

    args = ap.parse_args()

    tool_root = _resolve_tool_root()

    cfg_path = args.config
    if cfg_path and not os.path.isabs(cfg_path):
        cfg_path = os.path.join(tool_root, cfg_path)

    cfg = _load_yaml(cfg_path)

    # config defaults
    cfg_device = _get(cfg, ["camera", "device_name"], "3D USB Camera")
    cfg_size = _get(cfg, ["camera", "video_size"], "3200x1200")
    cfg_fps = int(_get(cfg, ["camera", "fps"], 60))
    cfg_seconds = int(_get(cfg, ["recording", "default_seconds"], 120))

    cfg_outdir = _get(cfg, ["output", "base_dir"], ".")
    if cfg_outdir and not os.path.isabs(cfg_outdir):
        cfg_outdir = os.path.abspath(os.path.join(tool_root, cfg_outdir))

    cfg_ffmpeg = _get(cfg, ["ffmpeg", "executable"], "ffmpeg")
    cfg_ffmpeg = _resolve_ffmpeg(cfg_ffmpeg, tool_root)

    # apply CLI overrides
    device = args.device if args.device is not None else cfg_device
    size = args.size if args.size is not None else cfg_size
    fps = args.fps if args.fps is not None else cfg_fps
    seconds = args.seconds if args.seconds is not None else cfg_seconds

    outdir = args.outdir if args.outdir is not None else cfg_outdir
    ffmpeg_exe = args.ffmpeg if args.ffmpeg is not None else cfg_ffmpeg
    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_exe, tool_root)

    os.makedirs(outdir, exist_ok=True)

    if args.out:
        out_path = args.out
        if not os.path.isabs(out_path):
            out_path = os.path.abspath(os.path.join(outdir, out_path))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(outdir, f"raw_{ts}.avi")

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-y",
        "-f", "dshow",
        "-video_size", size,
        "-framerate", str(fps),
        "-vcodec", "mjpeg",
        "-i", f'video={device}',
        "-t", str(seconds),
        "-c:v", "copy",
        out_path
    ]

    print("=== Effective Settings ===")
    print(f"config   : {cfg_path}")
    print(f"ffmpeg   : {ffmpeg_exe}")
    print(f"device   : {device}")
    print(f"size/fps : {size} @ {fps}")
    print(f"seconds  : {seconds}")
    print(f"out      : {out_path}")
    print("==========================")

    print("[INFO] Recording raw AVI (MJPEG copy).")
    print("[INFO] Command:")
    print("       " + " ".join(cmd))
    print("[INFO] Watch FFmpeg stats: fps ~ target, speed ~ 1.0x for real-time capture.")

    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"[ERROR] FFmpeg exited with code {r.returncode}")
        sys.exit(r.returncode)

    print("[INFO] Done.")
    print(out_path)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Check default.yaml ffmpeg.executable or pass --ffmpeg.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
