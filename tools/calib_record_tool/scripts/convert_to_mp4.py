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
    ap = argparse.ArgumentParser(description="Convert AVI (MJPEG) to MP4 (H.264). Supports configs/default.yaml.")
    ap.add_argument("--config", default="configs/default.yaml", help="path to YAML config (relative to tool root ok)")

    ap.add_argument("--in", dest="inp", required=True, help="input AVI path")
    ap.add_argument("--out", required=True, help="output MP4 path")

    ap.add_argument("--ffmpeg", default=None, help="ffmpeg executable or relative path")
    ap.add_argument("--preset", default=None, help="x264 preset")
    ap.add_argument("--crf", type=int, default=None, help="x264 CRF (lower=better)")

    args = ap.parse_args()

    tool_root = _resolve_tool_root()

    cfg_path = args.config
    if cfg_path and not os.path.isabs(cfg_path):
        cfg_path = os.path.join(tool_root, cfg_path)
    cfg = _load_yaml(cfg_path)

    cfg_ffmpeg = _get(cfg, ["ffmpeg", "executable"], "ffmpeg")
    cfg_ffmpeg = _resolve_ffmpeg(cfg_ffmpeg, tool_root)

    cfg_preset = _get(cfg, ["mp4", "preset"], _get(cfg, ["encode", "preset"], "veryfast"))
    cfg_crf = int(_get(cfg, ["mp4", "crf"], _get(cfg, ["encode", "crf"], 18)))

    ffmpeg_exe = args.ffmpeg if args.ffmpeg is not None else cfg_ffmpeg
    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_exe, tool_root)

    preset = args.preset if args.preset is not None else cfg_preset
    crf = args.crf if args.crf is not None else cfg_crf

    inp = args.inp
    outp = args.out

    # resolve output directory
    outdir = os.path.dirname(os.path.abspath(outp))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    cmd = [
        ffmpeg_exe, "-hide_banner", "-y",
        "-i", inp,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        outp
    ]

    print("=== Effective Settings ===")
    print(f"config : {cfg_path}")
    print(f"ffmpeg : {ffmpeg_exe}")
    print(f"preset: {preset}")
    print(f"crf   : {crf}")
    print(f"in    : {inp}")
    print(f"out   : {outp}")
    print("==========================")

    print("[INFO] Converting AVI -> MP4 (H.264).")
    print("[INFO] Command:")
    print("       " + " ".join(cmd))

    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"[ERROR] convert failed, code={r.returncode}")
        sys.exit(r.returncode)

    print("[INFO] Done.")
    print(outp)


if __name__ == "__main__":
    try:
        run()
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Check default.yaml ffmpeg.executable or pass --ffmpeg.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
