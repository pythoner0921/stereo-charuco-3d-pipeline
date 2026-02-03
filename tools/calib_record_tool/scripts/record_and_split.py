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


def _resolve_ffmpeg(ffmpeg_exe: str) -> str:
    """
    Resolve ffmpeg path:
    - If a relative path is provided, resolve relative to project root (= folder that contains this scripts file's parent).
      Here we assume: tools/calig_record_tool/scripts/record_and_split.py
      Project root is tools/calig_record_tool
    - If not found, fall back to 'ffmpeg' in PATH.
    """
    if not ffmpeg_exe:
        return "ffmpeg"

    # If it's already absolute, keep it
    if os.path.isabs(ffmpeg_exe):
        return ffmpeg_exe

    # Resolve relative to tool root (parent of scripts/)
    tool_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate = os.path.abspath(os.path.join(tool_root, ffmpeg_exe))
    if os.path.exists(candidate):
        return candidate

    # Also allow user to pass plain "ffmpeg"
    return ffmpeg_exe


def run():
    ap = argparse.ArgumentParser(
        description="Record raw (AVI MJPEG copy) then convert to MP4 and split left/right. Supports configs/default.yaml."
    )

    # NEW: config
    ap.add_argument("--config", default="configs/default.yaml", help="path to YAML config (relative to tool root ok)")

    # capture (CLI still exists; if provided explicitly, it overrides config)
    ap.add_argument("--device", default=None)
    ap.add_argument("--size", default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--seconds", type=int, default=None)

    # output
    ap.add_argument("--outdir", default=None, help="base output directory")
    ap.add_argument("--outroot", default=None, help="session folder path (overrides outdir/session_YYYYmmdd_HHMMSS)")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable or relative path, e.g. bin/ffmpeg/ffmpeg.exe")

    # split params
    ap.add_argument("--full_w", type=int, default=None)
    ap.add_argument("--full_h", type=int, default=None)
    ap.add_argument("--xsplit", type=int, default=None)

    # mp4 encode params
    ap.add_argument("--preset", default=None)
    ap.add_argument("--crf", type=int, default=None)

    # keep intermediate AVI
    ap.add_argument("--keep_avi", action="store_true", help="keep raw.avi even if config says otherwise")
    ap.add_argument("--no_keep_avi", action="store_true", help="force delete raw.avi even if config says keep")

    args = ap.parse_args()

    # Resolve tool root for config relative paths
    tool_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = args.config
    if cfg_path and not os.path.isabs(cfg_path):
        cfg_path = os.path.join(tool_root, cfg_path)

    cfg = _load_yaml(cfg_path)

    # ---- defaults from config ----
    cfg_device = _get(cfg, ["camera", "device_name"], "3D USB Camera")
    cfg_size = _get(cfg, ["camera", "video_size"], "3200x1200")
    cfg_fps = int(_get(cfg, ["camera", "fps"], 60))

    cfg_default_seconds = int(_get(cfg, ["recording", "default_seconds"], 120))

    cfg_out_base = _get(cfg, ["output", "base_dir"], ".")
    if cfg_out_base and not os.path.isabs(cfg_out_base):
        cfg_out_base = os.path.abspath(os.path.join(tool_root, cfg_out_base))

    cfg_keep_avi = bool(_get(cfg, ["output", "keep_avi"], True))

    cfg_full_w = int(_get(cfg, ["split", "full_w"], 3200))
    cfg_full_h = int(_get(cfg, ["split", "full_h"], 1200))
    cfg_xsplit = int(_get(cfg, ["split", "xsplit"], cfg_full_w // 2))

    cfg_ffmpeg = _get(cfg, ["ffmpeg", "executable"], "ffmpeg")
    cfg_ffmpeg = _resolve_ffmpeg(cfg_ffmpeg)

    cfg_preset = _get(cfg, ["mp4", "preset"], _get(cfg, ["encode", "preset"], "veryfast"))
    cfg_crf = int(_get(cfg, ["mp4", "crf"], _get(cfg, ["encode", "crf"], 18)))

    # ---- apply CLI overrides (if user provided values) ----
    device = args.device if args.device is not None else cfg_device
    size = args.size if args.size is not None else cfg_size
    fps = args.fps if args.fps is not None else cfg_fps
    seconds = args.seconds if args.seconds is not None else cfg_default_seconds

    outdir = args.outdir if args.outdir is not None else cfg_out_base
    outroot = args.outroot if args.outroot is not None else ""

    ffmpeg_exe = args.ffmpeg if args.ffmpeg is not None else cfg_ffmpeg
    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_exe)

    full_w = args.full_w if args.full_w is not None else cfg_full_w
    full_h = args.full_h if args.full_h is not None else cfg_full_h
    xsplit = args.xsplit if args.xsplit is not None else cfg_xsplit

    preset = args.preset if args.preset is not None else cfg_preset
    crf = args.crf if args.crf is not None else cfg_crf

    # keep_avi resolution: CLI flags override config
    if args.keep_avi and args.no_keep_avi:
        raise ValueError("Cannot use both --keep_avi and --no_keep_avi")

    keep_avi = cfg_keep_avi
    if args.keep_avi:
        keep_avi = True
    if args.no_keep_avi:
        keep_avi = False

    # ---- session dir ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = outroot if outroot else os.path.join(outdir, f"session_{ts}")
    os.makedirs(session_dir, exist_ok=True)

    raw_avi = os.path.join(session_dir, "raw.avi")
    raw_mp4 = os.path.join(session_dir, "raw.mp4")
    left_mp4 = os.path.join(session_dir, "left.mp4")
    right_mp4 = os.path.join(session_dir, "right.mp4")

    # ---- print effective config ----
    print("=== Effective Settings ===")
    print(f"config      : {cfg_path}")
    print(f"ffmpeg      : {ffmpeg_exe}")
    print(f"device      : {device}")
    print(f"size/fps    : {size} @ {fps}")
    print(f"seconds     : {seconds}")
    print(f"split       : full={full_w}x{full_h}, xsplit={xsplit}")
    print(f"mp4 encode  : preset={preset}, crf={crf}")
    print(f"output dir  : {session_dir}")
    print(f"keep_avi    : {keep_avi}")
    print("==========================")

    # STEP 1: record AVI with MJPEG copy
    cmd_record = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-f", "dshow",
        "-video_size", size,
        "-framerate", str(fps),
        "-vcodec", "mjpeg",
        "-i", f'video={device}',
        "-t", str(seconds),
        "-c:v", "copy",
        raw_avi
    ]
    print("[STEP 1] RECORD raw.avi (MJPEG copy, real-time)")
    print("        " + " ".join(cmd_record))
    print("        Watch FFmpeg stats: fps ~ target, speed ~ 1.0x")

    r = subprocess.run(cmd_record, check=False)
    if r.returncode != 0:
        print(f"[ERROR] record failed, code={r.returncode}")
        sys.exit(r.returncode)

    # STEP 2: convert AVI -> MP4
    cmd_convert = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", raw_avi,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        raw_mp4
    ]
    print("[STEP 2] CONVERT raw.avi -> raw.mp4 (H.264)")
    print("        " + " ".join(cmd_convert))

    r = subprocess.run(cmd_convert, check=False)
    if r.returncode != 0:
        print(f"[ERROR] convert failed, code={r.returncode}")
        sys.exit(r.returncode)

    # Optionally remove raw.avi
    if not keep_avi:
        try:
            os.remove(raw_avi)
            print("[INFO] Deleted intermediate:", raw_avi)
        except OSError:
            print("[WARN] Could not delete:", raw_avi)

    # STEP 3: split from raw.mp4 into left/right MP4
    left_crop = f"crop={xsplit}:{full_h}:0:0"
    right_crop = f"crop={full_w - xsplit}:{full_h}:{xsplit}:0"

    cmd_left = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", raw_mp4,
        "-vf", left_crop,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        left_mp4
    ]
    cmd_right = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", raw_mp4,
        "-vf", right_crop,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        right_mp4
    ]

    print("[STEP 3] SPLIT raw.mp4 -> left.mp4 / right.mp4")
    print("        left : " + " ".join(cmd_left))
    print("        right: " + " ".join(cmd_right))

    r1 = subprocess.run(cmd_left, check=False)
    if r1.returncode != 0:
        print(f"[ERROR] left split failed, code={r1.returncode}")
        sys.exit(r1.returncode)

    r2 = subprocess.run(cmd_right, check=False)
    if r2.returncode != 0:
        print(f"[ERROR] right split failed, code={r2.returncode}")
        sys.exit(r2.returncode)

    print("[DONE] Outputs in session folder:")
    print("  " + raw_mp4)
    print("  " + left_mp4)
    print("  " + right_mp4)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Check default.yaml ffmpeg.executable or pass --ffmpeg.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
