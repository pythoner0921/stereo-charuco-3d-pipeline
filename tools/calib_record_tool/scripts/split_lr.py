import argparse
import os
import subprocess
import sys

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
    ap = argparse.ArgumentParser(description="Split side-by-side stereo video into left/right MP4 using FFmpeg crop. Supports configs/default.yaml.")
    ap.add_argument("--config", default="configs/default.yaml", help="path to YAML config (relative to tool root ok)")

    ap.add_argument("--in", dest="inp", required=True, help="input raw file path (AVI or MP4), side-by-side")
    ap.add_argument("--outdir", default=None, help="output directory")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable or relative path")

    ap.add_argument("--w", type=int, default=None, help="full frame width")
    ap.add_argument("--h", type=int, default=None, help="full frame height")
    ap.add_argument("--xsplit", type=int, default=None, help="split x (default: half width)")

    ap.add_argument("--preset", default=None, help="x264 preset for outputs")
    ap.add_argument("--crf", type=int, default=None, help="x264 CRF")

    ap.add_argument("--left_name", default="left.mp4")
    ap.add_argument("--right_name", default="right.mp4")
    args = ap.parse_args()

    tool_root = _resolve_tool_root()

    cfg_path = args.config
    if cfg_path and not os.path.isabs(cfg_path):
        cfg_path = os.path.join(tool_root, cfg_path)
    cfg = _load_yaml(cfg_path)

    cfg_outdir = _get(cfg, ["output", "base_dir"], ".")
    if cfg_outdir and not os.path.isabs(cfg_outdir):
        cfg_outdir = os.path.abspath(os.path.join(tool_root, cfg_outdir))

    cfg_full_w = int(_get(cfg, ["split", "full_w"], 3200))
    cfg_full_h = int(_get(cfg, ["split", "full_h"], 1200))
    cfg_xsplit = int(_get(cfg, ["split", "xsplit"], cfg_full_w // 2))

    cfg_ffmpeg = _get(cfg, ["ffmpeg", "executable"], "ffmpeg")
    cfg_ffmpeg = _resolve_ffmpeg(cfg_ffmpeg, tool_root)

    cfg_preset = _get(cfg, ["mp4", "preset"], _get(cfg, ["encode", "preset"], "veryfast"))
    cfg_crf = int(_get(cfg, ["mp4", "crf"], _get(cfg, ["encode", "crf"], 18)))

    # Apply overrides
    outdir = args.outdir if args.outdir is not None else cfg_outdir
    ffmpeg_exe = args.ffmpeg if args.ffmpeg is not None else cfg_ffmpeg
    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_exe, tool_root)

    full_w = args.w if args.w is not None else cfg_full_w
    full_h = args.h if args.h is not None else cfg_full_h
    xsplit = args.xsplit if args.xsplit is not None else cfg_xsplit

    preset = args.preset if args.preset is not None else cfg_preset
    crf = args.crf if args.crf is not None else cfg_crf

    os.makedirs(outdir, exist_ok=True)

    left_path = os.path.join(outdir, args.left_name)
    right_path = os.path.join(outdir, args.right_name)

    left_crop = f"crop={xsplit}:{full_h}:0:0"
    right_crop = f"crop={full_w - xsplit}:{full_h}:{xsplit}:0"

    cmd_left = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", args.inp,
        "-vf", left_crop,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        left_path
    ]
    cmd_right = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", args.inp,
        "-vf", right_crop,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        right_path
    ]

    print("=== Effective Settings ===")
    print(f"config : {cfg_path}")
    print(f"ffmpeg : {ffmpeg_exe}")
    print(f"in     : {args.inp}")
    print(f"outdir : {outdir}")
    print(f"split  : full={full_w}x{full_h}, xsplit={xsplit}")
    print(f"mp4    : preset={preset}, crf={crf}")
    print("==========================")

    print("[INFO] Splitting to left/right MP4")
    print("[INFO] Left command : " + " ".join(cmd_left))
    print("[INFO] Right command: " + " ".join(cmd_right))

    r1 = subprocess.run(cmd_left, check=False)
    if r1.returncode != 0:
        print(f"[ERROR] left split failed, code={r1.returncode}")
        sys.exit(r1.returncode)

    r2 = subprocess.run(cmd_right, check=False)
    if r2.returncode != 0:
        print(f"[ERROR] right split failed, code={r2.returncode}")
        sys.exit(r2.returncode)

    print("[INFO] Done.")
    print(left_path)
    print(right_path)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Check default.yaml ffmpeg.executable or pass --ffmpeg.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
