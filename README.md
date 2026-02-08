# Stereo ChArUco 3D Pipeline

Stereo camera 3D motion capture pipeline with auto-calibration. Uses a dual-camera (stereo) setup with ChArUco board calibration, automatic person detection, pose tracking, and 3D triangulation.

## Features

- Auto-calibration using ChArUco board (intrinsic + extrinsic)
- Manual and auto-monitor recording modes
- Multiple pose trackers: MediaPipe (Holistic/Pose/Hand), YOLOv8-Pose
- **ONNX Runtime acceleration** — auto-exports YOLO to ONNX, true parallel per-port inference (v0.5.0)
- **Batch 2D detection** — bypasses caliscope's frame-by-frame streaming for ~3x speedup (v0.5.0)
- **AVI fast-split** — MJPEG re-encode ~5-10x faster than H.264 for pipeline recordings (v0.5.0)
- **VideoPose3D fusion** — optional monocular 3D lifting to fill occluded joints (v0.5.0, requires `torch`)
- 3D triangulation via caliscope
- Interactive 3D skeleton visualization with playback controls
- Configurable reconstruction FPS (10/15/20/30/60)

## Requirements

- **Python**: 3.10 - 3.12
- **OS**: Windows (DirectShow camera access)
- **Hardware**: Stereo USB camera (e.g. "3D USB Camera", 3200x1200 output)
- **FFmpeg**: Required for video processing

---

## Installation

> **Python version**: Must be 3.10, 3.11, or 3.12. Python 3.13+ is **NOT supported**
> (caliscope requires `<3.13`). Always specify `python=3.11` when creating the environment.

### Method 1: pip install (recommended for users)

```bash
# 1. Create conda environment (Python 3.11 required — 3.13 will NOT work)
conda create -n stereo-pipeline python=3.11 pip ffmpeg -c conda-forge -y
conda activate stereo-pipeline

# 2. Install caliscope (must be from GitHub, not PyPI)
pip install git+https://github.com/mprib/caliscope.git@8dc0cd4e

# 3. Install this pipeline
pip install stereo-charuco-pipeline

# 4. (Optional) Install VideoPose3D support for monocular 3D fusion
#    This improves accuracy on occluded joints. Skip if you don't need it.
pip install stereo-charuco-pipeline[videopose3d]

# 5. Fix dependency conflicts (required after every install/upgrade)
pip install pyside6-essentials==6.8.1 shiboken6==6.8.1   # PySide6 6.10+ has DLL issues
pip install --force-reinstall --no-cache-dir opencv-contrib-python>=4.8.0.74  # restore aruco
pip install "numpy<2.4"                                    # numba requires numpy<2.4
```

> **Important**: Step 5 is required every time you install or upgrade.
> These fixes resolve three known conflicts:
> - `ultralytics` installs `opencv-python` which lacks `cv2.aruco`
> - `PySide6 >= 6.10` causes DLL load failures on some Windows machines
> - `opencv-contrib-python` reinstall may pull `numpy >= 2.4` which breaks `numba`
>
> **Step 4 is optional**: Without torch, the pipeline works normally — it just
> skips the VideoPose3D fusion step during reconstruction. ONNX acceleration
> and all other v0.5.0 optimizations work without torch.

### Method 2: Clone from GitHub (for development)

```bash
# 1. Clone
git clone https://github.com/pythoner0921/stereo-charuco-3d-pipeline.git
cd stereo-charuco-3d-pipeline

# 2. Create environment
conda create -n stereo-pipeline python=3.11 pip ffmpeg -c conda-forge -y
conda activate stereo-pipeline

# 3. Install caliscope
pip install git+https://github.com/mprib/caliscope.git@8dc0cd4e

# 4. Install in editable mode
cd tools/calib_record_tool
pip install -e .

# 5. (Optional) Install VideoPose3D support
pip install -e ".[videopose3d]"

# 6. Fix dependency conflicts
pip install pyside6-essentials==6.8.1 shiboken6==6.8.1
pip install --force-reinstall --no-cache-dir opencv-contrib-python>=4.8.0.74
pip install "numpy<2.4"
```

### Conda environment.yml (alternative)

```bash
conda env create -f environment.yml
conda activate stereo-pipeline

# Fix dependency conflicts after environment creation
pip install pyside6-essentials==6.8.1 shiboken6==6.8.1
pip install --force-reinstall --no-cache-dir opencv-contrib-python>=4.8.0.74
pip install "numpy<2.4"
```

> If `conda env create` fails with encoding errors on Japanese/Chinese Windows,
> use Method 1 or Method 2 instead.

---

## Usage

### CLI Commands

After installation, three commands are available:

| Command | Description |
|---------|-------------|
| `stereo-pipeline` | Full workflow (Project Manager + Calibration + Pipeline) |
| `stereo-calibrate` | Calibration UI only |
| `stereo-record` | Pipeline UI only (record + reconstruct) |

```bash
conda activate stereo-pipeline

# Full workflow (recommended)
stereo-pipeline

# With custom config
stereo-pipeline --config path/to/config.yaml

# With custom project base directory
stereo-pipeline --projects-base D:\my_projects

# Calibration only
stereo-calibrate --project-dir D:\my_projects\project_001

# Recording/reconstruction only
stereo-record --project-dir D:\my_projects\project_001
```

### Running from source (clone method)

```bash
conda activate stereo-pipeline
cd tools/calib_record_tool

python scripts/run_unified.py              # Full workflow
python scripts/run_pipeline_ui.py          # Pipeline UI only
python scripts/run_calibration_ui.py       # Calibration UI only
```

---

## Workflow

The pipeline has 4 stages, each corresponding to a panel in the UI:

### Panel 1: Auto Calibration

Runs automatic stereo calibration using ChArUco board videos:
- Intrinsic calibration (per-camera lens parameters)
- Extrinsic calibration (relative camera positions via bundle adjustment)
- Requires `port_1.mp4` and `port_2.mp4` in calibration directories

### Panel 2: Record Session

Two recording modes:

- **Manual**: Start/stop recording with buttons. Uses FFmpeg to capture stereo video.
- **Auto Monitor**: Automatic recording triggered by person detection (HOG-based). Configurable detection interval, absence threshold, and cooldown.

Post-processing splits the stereo AVI into left/right video files (parallel FFmpeg encoding).
Pipeline recordings use **AVI fast-split** (MJPEG, ~5-10x faster). Calibration recordings use MP4 (H.264, required for caliscope frame-accurate seeking).

### Panel 3: 3D Reconstruction

- Select a recording and tracker type
- **Trackers**: YOLOV8_POSE (recommended), HOLISTIC, POSE, HAND
- **Model Size** (YOLOV8 only): Nano (fastest) / Small / Medium (most accurate)
- **Infer Size** (YOLOV8 only): 320px (fastest) / 480px (balanced) / 640px (best accuracy)
- **FPS**: Target framerate for processing (default 10). Lower = faster processing.
  - 60fps recording at FPS=10 processes only 1/6 of frames (~6x speedup)
  - For normal human motion, 10-15fps is sufficient
- **ONNX acceleration**: YOLOV8_POSE auto-exports to ONNX on first run, subsequent runs use cached model. ONNX Runtime releases the GIL, enabling true CPU parallelism across camera ports.
- **Batch detection**: YOLOV8_POSE uses fast batch inference (bypasses caliscope streaming)
- **Outlier filtering**: Per-keypoint IQR + velocity-based filtering removes triangulation drift
- **VideoPose3D fusion** (optional, requires `torch`): After stereo triangulation, fuses monocular 3D predictions from VideoPose3D to fill occluded joints. Automatically skipped if torch is not installed.
- Output: `xyz_{TRACKER}.csv` with 3D coordinates per frame

### Panel 4: 3D Visualization

- Load reconstructed xyz CSV data
- Interactive 3D skeleton rendering with wireframe connections
- Playback controls: play/pause, frame-by-frame, speed (0.25x - 4x)
- Mouse interaction: left-drag to rotate, scroll to zoom

---

## Configuration

Default configuration is built-in. Override with `--config path/to/config.yaml`:

```yaml
camera:
  device_name: "3D USB Camera"
  backend: "dshow"
  video_size: "3200x1200"
  fps: 60

split:
  full_w: 3200
  full_h: 1200
  xsplit: 1600              # Left/right split point

charuco:
  columns: 10
  rows: 16
  board_height_cm: 80.0
  board_width_cm: 50.0
  dictionary: "DICT_4X4_100"
  aruco_scale: 0.7
  square_size_cm: 5.0

mp4:
  preset: "veryfast"
  crf: 18
```

---

## Project Directory Structure

Each project is organized as:

```
project_YYYYMMDD_HHMMSS/
  calibration/
    intrinsic/
      port_1.mp4            # Left camera calibration video (H.264)
      port_2.mp4            # Right camera calibration video (H.264)
    extrinsic/
      port_1.mp4
      port_2.mp4
  recordings/
    session_YYYYMMDD_HHMMSS/
      port_1.avi            # Left camera recording (MJPEG, v0.5.0+)
      port_2.avi            # Right camera recording (MJPEG, v0.5.0+)
      YOLOV8_POSE/
        xy_YOLOV8_POSE.csv  # 2D landmarks
        xyz_YOLOV8_POSE.csv # 3D coordinates (output)
  camera_array.toml         # Calibration results
```

> **Note**: v0.5.0+ pipeline recordings produce `.avi` files (MJPEG fast-split).
> Existing projects with `.mp4` recordings are fully backward-compatible.

---

## Tracker Comparison

| Tracker | Keypoints | Strengths | Weaknesses |
|---------|-----------|-----------|------------|
| **YOLOV8_POSE** | 17 (COCO) | Robust to partial occlusion, side views | Requires `ultralytics` |
| HOLISTIC | 33 (BlazePose) | Full body + hands + face | Needs full body visible, jittery |
| POSE | 33 (BlazePose) | Full body | Same as HOLISTIC without hands/face |
| HAND | 21 per hand | Hand tracking | Hands only |

---

## Troubleshooting

### `requires a different Python: 3.13 not in '<3.13,>=3.10'`

You created the environment with Python 3.13, which is not supported.
Recreate with Python 3.11:
```bash
conda deactivate
conda remove -n stereo-pipeline --all -y
conda create -n stereo-pipeline python=3.11 pip ffmpeg -c conda-forge -y
```

### `No module named 'caliscope.core'`

caliscope must be installed from GitHub (the PyPI version is outdated):
```bash
pip install git+https://github.com/mprib/caliscope.git@8dc0cd4e
```

### `cv2` has no attribute `aruco`

```bash
pip install --force-reinstall --no-cache-dir opencv-contrib-python>=4.8.0.74
pip install "numpy<2.4"
```

### PySide6 DLL load failed (`找不到指定的程序`)

Downgrade PySide6 to a compatible version:
```bash
pip install pyside6-essentials==6.8.1 shiboken6==6.8.1
```

### `Numba needs NumPy 2.3 or less`

```bash
pip install "numpy<2.4"
```

### Bundle adjustment hangs / very slow

This is normal for long calibration videos. The pipeline uses `ftol=1e-4` and
subsamples every 12th frame. If it takes more than a few minutes, the calibration
video may be too long — 30-60 seconds of ChArUco board movement is sufficient.

### Conda environment creation fails on CJK Windows

Use the step-by-step installation (Method 1) instead of `environment.yml`.
Set `chcp 65001` before running conda commands.

---

## Version History

| Version | Changes |
|---------|---------|
| 0.5.0 | ONNX Runtime acceleration, batch 2D detection, AVI fast-split (~5-10x faster post-processing), VideoPose3D fusion (optional torch), .avi/.mp4 interchangeable |
| 0.4.0 | YOLO model/resolution UI controls, skeleton wireframe in 3D viz, IQR+velocity outlier filtering, fix multi-window crash |
| 0.3.6 | Pin numpy<2.4, PySide6 >=6.5.0, startup checks for caliscope + aruco |
| 0.3.3 | Pin caliscope to commit 8dc0cd4e, add missing-dependency error message |
| 0.3.2 | Performance: parallel video split, reconstruction FPS control, faster bundle adjustment |
| 0.3.1 | Bundle adjustment tuning, OpenCV dependency ordering fix |
| 0.3.0 | YOLOv8-Pose tracker integration |
| 0.2.9 | Axis scaling fix (absolute clip + percentile view) |
| 0.2.8 | Disable smoothing, relax outlier filter |
| 0.2.7 | Larger 3D view, bidirectional EMA smoothing |
| 0.2.6 | Proportional 3D aspect ratio |

---

## License

MIT
