"""
Calibration Recording UI Tool

A guided UI for stereo camera calibration video recording.
Guides users through a standardized 2-minute calibration workflow.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import queue
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .config import load_yaml_config
from .ffmpeg import FFmpegRunner, python_executable


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CalibrationPosition:
    """A single calibration position in the workflow."""
    name: str
    instruction: str
    duration_seconds: int


# Default calibration positions (total ~120 seconds)
DEFAULT_POSITIONS: List[CalibrationPosition] = [
    CalibrationPosition("Center", "Hold the board at CENTER of frame", 15),
    CalibrationPosition("Top-Left", "Move board to TOP-LEFT corner", 15),
    CalibrationPosition("Top-Right", "Move board to TOP-RIGHT corner", 15),
    CalibrationPosition("Bottom-Left", "Move board to BOTTOM-LEFT corner", 15),
    CalibrationPosition("Bottom-Right", "Move board to BOTTOM-RIGHT corner", 15),
    CalibrationPosition("Near", "Move board CLOSER to camera", 15),
    CalibrationPosition("Far", "Move board FURTHER from camera", 15),
    CalibrationPosition("Tilt-Left", "TILT board to the LEFT", 10),
    CalibrationPosition("Tilt-Right", "TILT board to the RIGHT", 10),
]

MAX_CALIBRATION_TIME = 120  # 2 minutes max


@dataclass
class CalibConfig:
    """Configuration for calibration recording."""
    device_name: str = "3D USB Camera"
    video_size: str = "3200x1200"
    fps: int = 60

    # Split parameters
    full_w: int = 3200
    full_h: int = 1200
    xsplit: int = 1600

    # Encoding
    preset: str = "veryfast"
    crf: int = 18

    # Output
    output_base: str = ""
    keep_avi: bool = False

    # FFmpeg
    ffmpeg_exe: str = "ffmpeg"

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "CalibConfig":
        """Load configuration from YAML file."""
        data = load_yaml_config(yaml_path)

        cfg = cls()

        # Camera settings
        camera = data.get("camera", {})
        cfg.device_name = camera.get("device_name", cfg.device_name)
        cfg.video_size = camera.get("video_size", cfg.video_size)
        cfg.fps = int(camera.get("fps", cfg.fps))

        # Split settings
        split = data.get("split", {})
        cfg.full_w = int(split.get("full_w", cfg.full_w))
        cfg.full_h = int(split.get("full_h", cfg.full_h))
        cfg.xsplit = int(split.get("xsplit", cfg.xsplit))

        # MP4 encoding
        mp4 = data.get("mp4", {})
        cfg.preset = mp4.get("preset", cfg.preset)
        cfg.crf = int(mp4.get("crf", cfg.crf))

        # Output
        output = data.get("output", {})
        cfg.output_base = output.get("base_dir", cfg.output_base)
        cfg.keep_avi = bool(output.get("keep_avi", cfg.keep_avi))

        # FFmpeg
        ffmpeg = data.get("ffmpeg", {})
        cfg.ffmpeg_exe = ffmpeg.get("executable", cfg.ffmpeg_exe)

        return cfg


# ============================================================================
# Camera Preview Thread
# ============================================================================

class CameraPreview:
    """Handles camera preview using OpenCV."""

    def __init__(self, device_name: str, video_size: str, fps: int,
                 device_index: int = 0):
        self.device_name = device_name
        self.video_size = video_size
        self.fps = fps
        self.device_index = device_index

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()

        # Parse video size
        parts = video_size.split("x")
        self.width = int(parts[0])
        self.height = int(parts[1])

    def start(self) -> bool:
        """Start camera capture. Returns True if successful."""
        if not CV2_AVAILABLE:
            return False

        # Open camera by device index
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            return False

        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (non-blocking)."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        """Background thread for capturing frames."""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # Clear old frames and put new one
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            time.sleep(1.0 / 30)  # Limit to ~30fps for preview


# ============================================================================
# FFmpeg Recording
# ============================================================================

class FFmpegRecorder:
    """Handles FFmpeg-based video recording."""

    def __init__(self, config: CalibConfig, output_path: Path):
        self.config = config
        self.output_path = output_path
        self._proc: Optional[subprocess.Popen] = None
        self._running = False

    def _resolve_ffmpeg(self) -> str:
        """Resolve ffmpeg executable path."""
        exe = self.config.ffmpeg_exe

        if not exe:
            return "ffmpeg"

        if os.path.isabs(exe):
            return exe

        # Resolve relative to tool root
        from .paths import tool_root as _tool_root; tool_root = _tool_root()
        candidate = tool_root / exe
        if candidate.exists():
            return str(candidate)

        # Try system PATH
        which = shutil.which("ffmpeg")
        if which:
            return which

        return exe

    def start(self) -> bool:
        """Start recording. Returns True if successful."""
        ffmpeg = self._resolve_ffmpeg()

        cmd = [
            ffmpeg, "-hide_banner", "-y",
            "-f", "dshow",
            "-video_size", self.config.video_size,
            "-framerate", str(self.config.fps),
            "-vcodec", "mjpeg",
            "-i", f"video={self.config.device_name}",
            "-c:v", "copy",
            str(self.output_path)
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._running = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            return False

    def stop(self) -> bool:
        """Stop recording gracefully."""
        if not self._proc:
            return False

        self._running = False

        try:
            # Send 'q' to FFmpeg to stop gracefully
            if self._proc.stdin:
                self._proc.stdin.write(b'q')
                self._proc.stdin.flush()

            # Wait for process to finish
            self._proc.wait(timeout=5)
            return self._proc.returncode == 0
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            return False
        except Exception as e:
            print(f"[ERROR] Failed to stop recording: {e}")
            return False
        finally:
            self._proc = None

    @property
    def is_running(self) -> bool:
        return self._running and self._proc is not None


# ============================================================================
# Post-Processing Pipeline
# ============================================================================

class PostProcessor:
    """Handles post-processing of recorded video."""

    def __init__(self, config: CalibConfig, session_dir: Path,
                 on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.session_dir = session_dir
        self.on_log = on_log

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)

    def _resolve_ffmpeg(self) -> str:
        """Resolve ffmpeg executable path."""
        exe = self.config.ffmpeg_exe

        if not exe:
            return "ffmpeg"

        if os.path.isabs(exe):
            return exe

        from .paths import tool_root as _tool_root; tool_root = _tool_root()
        candidate = tool_root / exe
        if candidate.exists():
            return str(candidate)

        which = shutil.which("ffmpeg")
        if which:
            return which

        return exe

    def run(self, raw_avi: Path, output_dirs) -> bool:
        """
        Run the full post-processing pipeline.

        Crops AVI directly into left/right MP4 files in parallel
        (no intermediate full-frame MP4).

        Parameters
        ----------
        raw_avi : Path
            Input raw AVI file.
        output_dirs : Path or list[Path]
            One or more directories to receive port_1.mp4 / port_2.mp4.
        """
        # Normalise to list
        if isinstance(output_dirs, (str, Path)):
            output_dirs = [Path(output_dirs)]
        else:
            output_dirs = [Path(d) for d in output_dirs]

        ffmpeg = self._resolve_ffmpeg()

        preset = self.config.preset
        crf = str(self.config.crf)
        xsplit = self.config.xsplit
        full_w = self.config.full_w
        full_h = self.config.full_h

        left_crop = f"crop={xsplit}:{full_h}:0:0"
        right_crop = f"crop={full_w - xsplit}:{full_h}:{xsplit}:0"

        # Ensure output directories exist
        for out_dir in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Use first output dir for primary encoding; copy to others after
        primary_dir = output_dirs[0]
        port1_path = primary_dir / "port_1.mp4"
        port2_path = primary_dir / "port_2.mp4"

        # Step 1: Crop left + right from AVI in parallel
        self._log("[STEP 1] Splitting AVI into left/right MP4 (parallel)...")

        cmd_left = [
            ffmpeg, "-hide_banner", "-y",
            "-i", str(raw_avi),
            "-vf", left_crop,
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p",
            str(port1_path),
        ]
        cmd_right = [
            ffmpeg, "-hide_banner", "-y",
            "-i", str(raw_avi),
            "-vf", right_crop,
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-pix_fmt", "yuv420p",
            str(port2_path),
        ]

        proc_left = subprocess.Popen(cmd_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_right = subprocess.Popen(cmd_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        _, stderr_left = proc_left.communicate()
        _, stderr_right = proc_right.communicate()

        if proc_left.returncode != 0:
            self._log(f"[ERROR] Left split failed: {stderr_left.decode(errors='replace')}")
            return False
        if proc_right.returncode != 0:
            self._log(f"[ERROR] Right split failed: {stderr_right.decode(errors='replace')}")
            return False

        self._log("[STEP 1] Split complete.")
        self._log(f"  → {port1_path}")
        self._log(f"  → {port2_path}")

        # Step 2: Copy to additional output directories (if any)
        if len(output_dirs) > 1:
            self._log("[STEP 2] Copying to additional output directories...")
            for out_dir in output_dirs[1:]:
                shutil.copy2(port1_path, out_dir / "port_1.mp4")
                shutil.copy2(port2_path, out_dir / "port_2.mp4")
                self._log(f"  → {out_dir / 'port_1.mp4'}")
                self._log(f"  → {out_dir / 'port_2.mp4'}")

        self._log("[DONE] Post-processing complete.")

        # Cleanup AVI if not keeping
        if not self.config.keep_avi:
            try:
                raw_avi.unlink()
                self._log(f"[CLEANUP] Deleted {raw_avi}")
            except Exception:
                pass

        return True

    def run_avi_fast(self, raw_avi: Path, output_dirs) -> bool:
        """Fast AVI split using MJPEG re-encode (~5-10x faster than H.264).

        Produces port_1.avi / port_2.avi instead of .mp4.
        Use this for pipeline recordings where H.264 seeking is not needed.

        Parameters
        ----------
        raw_avi : Path
            Input raw AVI file.
        output_dirs : Path or list[Path]
            One or more directories to receive port_1.avi / port_2.avi.
        """
        if isinstance(output_dirs, (str, Path)):
            output_dirs = [Path(output_dirs)]
        else:
            output_dirs = [Path(d) for d in output_dirs]

        ffmpeg = self._resolve_ffmpeg()

        xsplit = self.config.xsplit
        full_w = self.config.full_w
        full_h = self.config.full_h

        left_crop = f"crop={xsplit}:{full_h}:0:0"
        right_crop = f"crop={full_w - xsplit}:{full_h}:{xsplit}:0"

        for out_dir in output_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)

        primary_dir = output_dirs[0]
        port1_path = primary_dir / "port_1.avi"
        port2_path = primary_dir / "port_2.avi"

        self._log("[STEP 1] Splitting AVI into left/right AVI (MJPEG, parallel)...")

        cmd_left = [
            ffmpeg, "-hide_banner", "-y",
            "-i", str(raw_avi),
            "-vf", left_crop,
            "-c:v", "mjpeg", "-q:v", "2",
            str(port1_path),
        ]
        cmd_right = [
            ffmpeg, "-hide_banner", "-y",
            "-i", str(raw_avi),
            "-vf", right_crop,
            "-c:v", "mjpeg", "-q:v", "2",
            str(port2_path),
        ]

        proc_left = subprocess.Popen(cmd_left, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_right = subprocess.Popen(cmd_right, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        _, stderr_left = proc_left.communicate()
        _, stderr_right = proc_right.communicate()

        if proc_left.returncode != 0:
            self._log(f"[ERROR] Left split failed: {stderr_left.decode(errors='replace')}")
            return False
        if proc_right.returncode != 0:
            self._log(f"[ERROR] Right split failed: {stderr_right.decode(errors='replace')}")
            return False

        self._log("[STEP 1] AVI split complete.")
        self._log(f"  -> {port1_path}")
        self._log(f"  -> {port2_path}")

        if len(output_dirs) > 1:
            self._log("[STEP 2] Copying to additional output directories...")
            for out_dir in output_dirs[1:]:
                import shutil
                shutil.copy2(port1_path, out_dir / "port_1.avi")
                shutil.copy2(port2_path, out_dir / "port_2.avi")
                self._log(f"  -> {out_dir / 'port_1.avi'}")
                self._log(f"  -> {out_dir / 'port_2.avi'}")

        self._log("[DONE] Post-processing complete (AVI fast path).")

        if not self.config.keep_avi:
            try:
                raw_avi.unlink()
                self._log(f"[CLEANUP] Deleted {raw_avi}")
            except Exception:
                pass

        return True


# ============================================================================
# Main UI
# ============================================================================

class CalibrationUI(tk.Tk):
    """Main calibration recording UI."""

    def __init__(self, config_path: Optional[Path] = None,
                 project_dir: Optional[Path] = None,
                 on_complete: Optional[Callable[[], None]] = None):
        super().__init__()

        self.geometry("1200x800")
        self.configure(bg="#2b2b2b")

        # Load configuration
        self._load_config(config_path)

        # Dynamic project directory (overrides config-derived path)
        self._project_dir: Optional[Path] = Path(project_dir) if project_dir else None
        self._on_complete = on_complete

        if self._project_dir:
            self.title(f"Stereo Calibration - {self._project_dir.name}")
        else:
            self.title("Stereo Calibration Recording Tool")

        # State
        self._state = "idle"  # idle, preview, recording, processing
        self._camera: Optional[CameraPreview] = None
        self._recorder: Optional[FFmpegRecorder] = None
        self._positions = DEFAULT_POSITIONS.copy()
        self._current_position_idx = 0
        self._recording_start_time: Optional[float] = None
        self._position_start_time: Optional[float] = None
        self._session_dir: Optional[Path] = None
        self._raw_avi_path: Optional[Path] = None

        # Message queue for thread-safe UI updates
        self._msg_queue: queue.Queue = queue.Queue()

        # Build UI
        self._build_ui()

        # Start periodic updates
        self.after(50, self._update_loop)

    def _load_config(self, config_path: Optional[Path]):
        """Load configuration from YAML or use defaults."""
        if config_path and config_path.exists():
            self.config = CalibConfig.from_yaml(config_path)
        else:
            # Try default config path
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            default_config = tool_root / "configs" / "default.yaml"
            if default_config.exists():
                self.config = CalibConfig.from_yaml(default_config)
            else:
                self.config = CalibConfig()

        # Resolve output base
        if self.config.output_base and not os.path.isabs(self.config.output_base):
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            self.config.output_base = str(tool_root / self.config.output_base)

    def _build_ui(self):
        """Build the main UI components."""
        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Style configuration
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 12))
        style.configure("Instruction.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 12), padding=10)

        # Top: Status and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        ttk.Label(top_frame, text="Stereo Camera Calibration",
                  style="Title.TLabel").pack(side=tk.LEFT)

        # Status indicator
        self.status_var = tk.StringVar(value="Camera Ready")
        self.status_label = ttk.Label(top_frame, textvariable=self.status_var,
                                       style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Status dot
        self.status_canvas = tk.Canvas(top_frame, width=20, height=20,
                                        highlightthickness=0)
        self.status_canvas.pack(side=tk.RIGHT)
        self._status_dot = self.status_canvas.create_oval(4, 4, 16, 16,
                                                          fill="#4CAF50", outline="")

        # Middle: Preview and Progress
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Camera preview
        preview_frame = ttk.LabelFrame(middle_frame, text="Camera Preview", padding=5)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas for video preview (scaled down for display)
        self.preview_canvas = tk.Canvas(preview_frame, bg="#1a1a1a",
                                         width=800, height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Right: Calibration guide
        guide_frame = ttk.LabelFrame(middle_frame, text="Calibration Guide", padding=10)
        guide_frame.pack(side=tk.RIGHT, fill=tk.Y, ipadx=20)

        # Current instruction
        ttk.Label(guide_frame, text="Current Position:").pack(anchor=tk.W)
        self.position_var = tk.StringVar(value="--")
        ttk.Label(guide_frame, textvariable=self.position_var,
                  style="Instruction.TLabel", foreground="#2196F3").pack(anchor=tk.W, pady=(5, 15))

        # Instruction text
        ttk.Label(guide_frame, text="Instruction:").pack(anchor=tk.W)
        self.instruction_var = tk.StringVar(value="Click 'Start Camera' to begin")
        self.instruction_label = ttk.Label(guide_frame, textvariable=self.instruction_var,
                                           wraplength=250)
        self.instruction_label.pack(anchor=tk.W, pady=(5, 15))

        # Position countdown
        ttk.Label(guide_frame, text="Position Time:").pack(anchor=tk.W)
        self.position_time_var = tk.StringVar(value="--")
        ttk.Label(guide_frame, textvariable=self.position_time_var,
                  font=("Segoe UI", 24, "bold")).pack(anchor=tk.W, pady=(5, 15))

        # Total time
        ttk.Label(guide_frame, text="Total Time:").pack(anchor=tk.W)
        self.total_time_var = tk.StringVar(value="0:00 / 2:00")
        ttk.Label(guide_frame, textvariable=self.total_time_var,
                  font=("Segoe UI", 14)).pack(anchor=tk.W, pady=(5, 15))

        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Calibration Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=600)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Position indicators
        self.position_frame = ttk.Frame(progress_frame)
        self.position_frame.pack(fill=tk.X)

        self.position_labels = []
        for i, pos in enumerate(self._positions):
            lbl = ttk.Label(self.position_frame, text=pos.name,
                           font=("Segoe UI", 9), foreground="#888888")
            lbl.pack(side=tk.LEFT, expand=True)
            self.position_labels.append(lbl)

        # Bottom: Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Camera index selector
        cam_select_frame = ttk.Frame(button_frame)
        cam_select_frame.pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(cam_select_frame, text="Camera:").pack(side=tk.LEFT, padx=(0, 4))
        self.var_cam_index = tk.IntVar(value=0)
        self.spin_cam = ttk.Spinbox(
            cam_select_frame, from_=0, to=9, width=3,
            textvariable=self.var_cam_index, state="readonly"
        )
        self.spin_cam.pack(side=tk.LEFT)

        self.btn_switch_cam = ttk.Button(
            cam_select_frame, text="Switch",
            command=self._on_switch_camera
        )
        self.btn_switch_cam.pack(side=tk.LEFT, padx=(4, 0))

        self.btn_camera = ttk.Button(button_frame, text="Start Camera",
                                     style="Big.TButton", command=self._on_camera_toggle)
        self.btn_camera.pack(side=tk.LEFT, padx=5)

        self.btn_record = ttk.Button(button_frame, text="Start Calibration Recording",
                                     style="Big.TButton", command=self._on_record_start,
                                     state=tk.DISABLED)
        self.btn_record.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(button_frame, text="Stop",
                                   style="Big.TButton", command=self._on_stop,
                                   state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = tk.Text(log_frame, height=6, bg="#1a1a1a", fg="#ffffff",
                                font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        self._log("Calibration UI initialized.")
        self._log(f"Device: {self.config.device_name}")
        self._log(f"Resolution: {self.config.video_size} @ {self.config.fps}fps")

    def _log(self, msg: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)

    def _set_status(self, text: str, color: str = "#4CAF50"):
        """Update status indicator."""
        self.status_var.set(text)
        self.status_canvas.itemconfig(self._status_dot, fill=color)

    def _open_camera(self, device_index: int) -> bool:
        """Open camera with the given device index. Returns True if successful."""
        if not CV2_AVAILABLE:
            messagebox.showerror("Error", "OpenCV (cv2) is not installed.")
            return False

        self._camera = CameraPreview(
            self.config.device_name,
            self.config.video_size,
            self.config.fps,
            device_index=device_index,
        )

        if self._camera.start():
            self._state = "preview"
            self._set_status(f"Camera {device_index} Active", "#4CAF50")
            self.btn_camera.config(text="Stop Camera")
            self.btn_record.config(state=tk.NORMAL)
            self.instruction_var.set("Camera ready. Click 'Start Calibration Recording' to begin.")
            self._log(f"Camera {device_index} started successfully.")
            return True
        else:
            self._camera = None
            messagebox.showerror("Error", f"Failed to open camera {device_index}.")
            self._log(f"Failed to open camera {device_index}.")
            return False

    def _close_camera(self):
        """Close the current camera."""
        if self._camera:
            self._camera.stop()
            self._camera = None
        self._state = "idle"
        self._set_status("Camera Ready", "#888888")
        self.btn_camera.config(text="Start Camera")
        self.btn_record.config(state=tk.DISABLED)
        self.instruction_var.set("Click 'Start Camera' to begin")

    def _on_camera_toggle(self):
        """Toggle camera preview on/off."""
        if self._camera is None:
            self._open_camera(self.var_cam_index.get())
        else:
            self._close_camera()
            self._log("Camera stopped.")

    def _on_switch_camera(self):
        """Switch to a different camera device index."""
        new_index = self.var_cam_index.get()

        if self._state == "recording":
            messagebox.showwarning("Recording", "Cannot switch camera while recording.")
            return

        # If camera is running, restart with new index
        was_open = self._camera is not None
        if was_open:
            self._close_camera()
            self._log(f"Switching to camera {new_index}...")
            time.sleep(0.3)

        self._open_camera(new_index)

    def _on_record_start(self):
        """Start calibration recording."""
        if self._state != "preview":
            return

        # Stop camera preview first (to release camera for FFmpeg)
        if self._camera:
            self._camera.stop()
            self._camera = None
            self._log("Camera preview stopped for recording.")
            # Small delay to ensure camera is fully released
            time.sleep(0.5)

        # Create session directory under raw_output/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Resolve project root
        if self._project_dir:
            project_root = self._project_dir
        else:
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            if self.config.output_base:
                output_base = Path(self.config.output_base)
                if not output_base.is_absolute():
                    output_base = tool_root / output_base
                project_root = output_base.parent
            else:
                project_root = tool_root

        raw_output_dir = project_root / "raw_output"
        self._session_dir = raw_output_dir / f"calib_session_{ts}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._raw_avi_path = self._session_dir / "raw.avi"

        # Start FFmpeg recording
        self._recorder = FFmpegRecorder(self.config, self._raw_avi_path)

        if not self._recorder.start():
            messagebox.showerror("Error", "Failed to start recording.")
            self._log("Failed to start FFmpeg recording.")
            # Restart camera preview
            self._restart_camera_preview()
            return

        # Initialize recording state
        self._state = "recording"
        self._current_position_idx = 0
        self._recording_start_time = time.time()
        self._position_start_time = time.time()

        # Update UI
        self._set_status("RECORDING", "#f44336")
        self.btn_camera.config(state=tk.DISABLED)
        self.btn_record.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

        self._update_position_display()
        self._log(f"Recording started. Session: {self._session_dir}")

    def _restart_camera_preview(self):
        """Restart camera preview after recording."""
        if not CV2_AVAILABLE:
            return

        self._open_camera(self.var_cam_index.get())

    def _on_stop(self):
        """Stop recording manually."""
        if self._state == "recording":
            self._finish_recording()

    def _finish_recording(self):
        """Finish recording and start post-processing."""
        if self._recorder:
            self._recorder.stop()
            self._recorder = None

        self._state = "processing"
        self._set_status("Processing...", "#FF9800")
        self.btn_stop.config(state=tk.DISABLED)

        self.instruction_var.set("Processing video...")
        self.position_var.set("--")
        self.position_time_var.set("--")

        self._log("Recording stopped. Starting post-processing...")

        # Run post-processing in background thread
        thread = threading.Thread(target=self._run_post_processing, daemon=True)
        thread.start()

    def _run_post_processing(self):
        """Run post-processing pipeline in background."""
        try:
            # Determine project root
            if self._project_dir:
                project_root = self._project_dir
            else:
                from .paths import tool_root as _tool_root; tool_root = _tool_root()
                if self.config.output_base:
                    output_base = Path(self.config.output_base)
                    if not output_base.is_absolute():
                        output_base = tool_root / output_base
                    project_root = output_base.parent
                else:
                    project_root = tool_root

            # Output to both intrinsic and extrinsic
            intrinsic_dir = project_root / "calibration" / "intrinsic"
            extrinsic_dir = project_root / "calibration" / "extrinsic"

            processor = PostProcessor(
                self.config,
                self._session_dir,
                on_log=lambda msg: self._msg_queue.put(("log", msg))
            )

            success = processor.run(self._raw_avi_path, [intrinsic_dir, extrinsic_dir])

            if success:
                self._msg_queue.put(("done", "Calibration recording complete!"))
            else:
                self._msg_queue.put(("error", "Post-processing failed."))

        except Exception as e:
            self._msg_queue.put(("error", f"Error: {e}"))

    def _update_position_display(self):
        """Update the current position display."""
        if self._current_position_idx >= len(self._positions):
            return

        pos = self._positions[self._current_position_idx]
        self.position_var.set(pos.name)
        self.instruction_var.set(pos.instruction)

        # Update position label colors
        for i, lbl in enumerate(self.position_labels):
            if i < self._current_position_idx:
                lbl.config(foreground="#4CAF50")  # Completed - green
            elif i == self._current_position_idx:
                lbl.config(foreground="#2196F3")  # Current - blue
            else:
                lbl.config(foreground="#888888")  # Pending - gray

    def _update_loop(self):
        """Main update loop (called every 50ms)."""
        # Process messages from background threads
        while True:
            try:
                msg_type, msg = self._msg_queue.get_nowait()
                if msg_type == "log":
                    self._log(msg)
                elif msg_type == "done":
                    self._on_processing_complete(msg)
                elif msg_type == "error":
                    self._on_processing_error(msg)
            except queue.Empty:
                break

        # Update preview
        if self._state == "preview" or self._state == "recording":
            self._update_preview()

        # Update recording state
        if self._state == "recording":
            self._update_recording_state()

        # Schedule next update
        self.after(50, self._update_loop)

    def _update_preview(self):
        """Update camera preview."""
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        # During recording, show a placeholder since camera is used by FFmpeg
        if self._state == "recording" and self._camera is None:
            self._draw_recording_placeholder(canvas_w, canvas_h)
            return

        if self._camera is None:
            return

        frame = self._camera.get_frame()
        if frame is None:
            return

        # Maintain aspect ratio
        frame_h, frame_w = frame.shape[:2]
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        frame_resized = cv2.resize(frame, (new_w, new_h))

        # Draw center line (split indicator)
        center_x = new_w // 2
        cv2.line(frame_resized, (center_x, 0), (center_x, new_h), (0, 255, 0), 2)

        # Convert to PhotoImage
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)

        # Update canvas
        self.preview_canvas.delete("all")
        x_offset = (canvas_w - new_w) // 2
        y_offset = (canvas_h - new_h) // 2
        self.preview_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        self.preview_canvas._photo = photo  # Keep reference

    def _draw_recording_placeholder(self, canvas_w: int, canvas_h: int):
        """Draw a placeholder during recording (when camera is used by FFmpeg)."""
        self.preview_canvas.delete("all")

        # Draw background
        self.preview_canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill="#1a1a1a")

        # Draw recording indicator
        center_x = canvas_w // 2
        center_y = canvas_h // 2

        # Flashing red circle
        if int(time.time() * 2) % 2 == 0:
            self.preview_canvas.create_oval(
                center_x - 30, center_y - 80,
                center_x + 30, center_y - 20,
                fill="#f44336", outline=""
            )

        # "RECORDING" text
        self.preview_canvas.create_text(
            center_x, center_y,
            text="RECORDING IN PROGRESS",
            fill="#ffffff",
            font=("Segoe UI", 18, "bold")
        )

        # Instruction
        self.preview_canvas.create_text(
            center_x, center_y + 40,
            text="Follow the calibration guide on the right",
            fill="#888888",
            font=("Segoe UI", 12)
        )

    def _update_recording_state(self):
        """Update recording state (timing, position transitions)."""
        if not self._recording_start_time or not self._position_start_time:
            return

        current_time = time.time()
        total_elapsed = current_time - self._recording_start_time
        position_elapsed = current_time - self._position_start_time

        # Check for timeout
        if total_elapsed >= MAX_CALIBRATION_TIME:
            self._set_status("Time Exceeded!", "#f44336")
            self.instruction_var.set("Calibration time exceeded. Click Stop to finish.")
            return

        # Update total time display
        total_mins = int(total_elapsed) // 60
        total_secs = int(total_elapsed) % 60
        self.total_time_var.set(f"{total_mins}:{total_secs:02d} / 2:00")

        # Check if current position is complete
        if self._current_position_idx < len(self._positions):
            pos = self._positions[self._current_position_idx]
            remaining = max(0, pos.duration_seconds - position_elapsed)

            self.position_time_var.set(f"{int(remaining)}s")

            # Calculate progress
            completed_time = sum(p.duration_seconds for p in self._positions[:self._current_position_idx])
            current_progress = min(position_elapsed, pos.duration_seconds)
            total_progress = completed_time + current_progress
            total_duration = sum(p.duration_seconds for p in self._positions)
            self.progress_var.set((total_progress / total_duration) * 100)

            # Move to next position
            if position_elapsed >= pos.duration_seconds:
                self._current_position_idx += 1
                self._position_start_time = current_time

                if self._current_position_idx >= len(self._positions):
                    # All positions complete
                    self._log("All calibration positions completed!")
                    self._finish_recording()
                else:
                    self._update_position_display()
                    self._log(f"Moving to: {self._positions[self._current_position_idx].name}")

    def _on_processing_complete(self, msg: str):
        """Handle processing completion."""
        self._state = "idle"
        self._set_status("Complete", "#4CAF50")
        self.instruction_var.set(msg)
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_record.config(state=tk.DISABLED)
        self.progress_var.set(100)

        # Reset position labels
        for lbl in self.position_labels:
            lbl.config(foreground="#4CAF50")

        self._log(msg)

        if self._on_complete:
            result = messagebox.askquestion(
                "Recording Complete",
                f"{msg}\n\nContinue to Pipeline for auto-calibration?",
            )
            if result == "yes":
                self._on_complete()
                self.destroy()
        else:
            messagebox.showinfo("Complete", msg)

    def _on_processing_error(self, msg: str):
        """Handle processing error."""
        self._state = "idle"
        self._set_status("Error", "#f44336")
        self.instruction_var.set(msg)
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_record.config(state=tk.DISABLED)

        self._log(msg)
        messagebox.showerror("Error", msg)

    def destroy(self):
        """Clean up before closing."""
        if self._camera:
            self._camera.stop()
        if self._recorder:
            self._recorder.stop()
        super().destroy()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Stereo Calibration Recording UI")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML configuration file")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    app = CalibrationUI(config_path)
    app.mainloop()


if __name__ == "__main__":
    main()
