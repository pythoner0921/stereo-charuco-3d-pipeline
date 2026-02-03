"""
Advanced Calibration Recording UI

Live video preview during recording with:
- Target region overlay for each calibration position
- ArUco marker detection to verify board placement
- Auto-advance when board is held in the correct region
- Manual fallback button
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import queue
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try ArUco availability
ARUCO_AVAILABLE = False
if CV2_AVAILABLE:
    try:
        _test = cv2.aruco.DICT_4X4_100
        ARUCO_AVAILABLE = True
    except AttributeError:
        pass

from .calibration_ui import CalibConfig, PostProcessor, MAX_CALIBRATION_TIME


# ============================================================================
# Position Targets
# ============================================================================

@dataclass
class PositionTarget:
    """Calibration position with a target region on the frame."""
    name: str
    instruction: str
    # Target region in normalised coordinates (0‑1) relative to ONE camera half
    x1: float
    y1: float
    x2: float
    y2: float
    hold_seconds: float = 3.0   # seconds board must stay inside to confirm


CALIBRATION_POSITIONS: List[PositionTarget] = [
    PositionTarget("Center",       "Hold board at CENTER of frame",       0.20, 0.15, 0.80, 0.85, 3.0),
    PositionTarget("Top-Left",     "Move board to TOP-LEFT corner",       0.02, 0.02, 0.48, 0.48, 3.0),
    PositionTarget("Top-Right",    "Move board to TOP-RIGHT corner",      0.52, 0.02, 0.98, 0.48, 3.0),
    PositionTarget("Bottom-Left",  "Move board to BOTTOM-LEFT corner",    0.02, 0.52, 0.48, 0.98, 3.0),
    PositionTarget("Bottom-Right", "Move board to BOTTOM-RIGHT corner",   0.52, 0.52, 0.98, 0.98, 3.0),
    PositionTarget("Near",         "Move board CLOSER to camera",         0.10, 0.05, 0.90, 0.95, 3.0),
    PositionTarget("Far",          "Move board FURTHER from camera",      0.30, 0.25, 0.70, 0.75, 3.0),
    PositionTarget("Tilt-Left",    "TILT board to the LEFT (~30°)",       0.15, 0.10, 0.85, 0.90, 3.0),
    PositionTarget("Tilt-Right",   "TILT board to the RIGHT (~30°)",      0.15, 0.10, 0.85, 0.90, 3.0),
]


# ============================================================================
# ArUco Board Detector
# ============================================================================

class BoardDetector:
    """Lightweight ArUco marker detector for position checking."""

    MIN_MARKERS = 4  # Need at least this many markers to count as "detected"

    def __init__(self, dictionary_name: str = "DICT_4X4_100"):
        if not ARUCO_AVAILABLE:
            raise RuntimeError("cv2.aruco is not available")

        dict_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_100)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

        # Try new API (OpenCV ≥ 4.7)
        try:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary)
            self._use_new_api = True
        except AttributeError:
            self._use_new_api = False

    def detect(self, image: np.ndarray) -> Optional[Tuple[Tuple[float, float, float, float],
                                                           Tuple[float, float], int]]:
        """
        Detect ArUco markers and return normalised bounding‑box + centroid.

        Returns
        -------
        (bbox_norm, centre_norm, num_markers) or None
            bbox_norm  = (x1, y1, x2, y2) in 0‑1
            centre_norm = (cx, cy) in 0‑1
        """
        h, w = image.shape[:2]

        if self._use_new_api:
            corners, ids, _ = self._detector.detectMarkers(image)
        else:
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(image, self._dictionary, parameters=params)

        if ids is None or len(ids) < self.MIN_MARKERS:
            return None

        all_pts = np.concatenate([c.reshape(-1, 2) for c in corners])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)

        bbox = (xmin / w, ymin / h, xmax / w, ymax / h)
        cx = (xmin + xmax) / 2.0 / w
        cy = (ymin + ymax) / 2.0 / h
        return bbox, (cx, cy), int(len(ids))


# ============================================================================
# FFmpeg Pipe Recorder (replaces cv2.VideoWriter)
# ============================================================================

def _resolve_ffmpeg_exe(config_exe: str) -> str:
    """Resolve ffmpeg path: bundled binary → system PATH → raw string."""
    if not config_exe:
        return "ffmpeg"
    if os.path.isabs(config_exe):
        return config_exe
    from .paths import tool_root as _tool_root; tool_root = _tool_root()
    candidate = tool_root / config_exe
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ffmpeg")
    return found if found else config_exe


class PipeRecorder:
    """
    Record video by piping raw BGR frames from OpenCV into an FFmpeg process.

    OpenCV captures → raw bytes via stdin → FFmpeg encodes MJPG → .avi file.
    This completely avoids cv2.VideoWriter and its timestamp/codec issues.
    """

    def __init__(self, ffmpeg_exe: str, output_path: Path,
                 width: int, height: int, fps: int):
        self.width = width
        self.height = height
        cmd = [
            ffmpeg_exe, "-hide_banner", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "mjpeg",
            "-q:v", "2",
            str(output_path),
        ]
        # creationflags: hide the console window on Windows
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )

    def write(self, frame: np.ndarray) -> bool:
        """Write one BGR frame. Returns False if the pipe is broken."""
        if not self._proc or not self._proc.stdin:
            return False
        try:
            self._proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def release(self):
        """Close the pipe and wait for FFmpeg to finish."""
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ============================================================================
# Combined Camera Capture + Record
# ============================================================================

@dataclass
class DetectionSnapshot:
    """Thread‑safe detection result container."""
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    center_norm: Optional[Tuple[float, float]] = None
    num_markers: int = 0
    detected: bool = False


class CameraRecorder:
    """
    OpenCV captures the camera for preview + detection.
    FFmpeg (via PipeRecorder) handles the actual recording.

    Architecture:
      OpenCV cap.read() → frame
         ├─ PipeRecorder.write(frame)   → FFmpeg encodes to .avi
         ├─ BoardDetector.detect(frame) → position check
         └─ frame_queue                 → UI preview
    """

    def __init__(self, device_index: int, width: int, height: int, fps: int,
                 half_width: int, ffmpeg_exe: str = "ffmpeg",
                 dictionary_name: str = "DICT_4X4_100",
                 detect_interval: int = 4):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.half_width = half_width
        self._ffmpeg_exe = ffmpeg_exe
        self._detect_interval = detect_interval

        self._cap: Optional[cv2.VideoCapture] = None
        self._pipe: Optional[PipeRecorder] = None
        self._running = False
        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Detection
        self._detector: Optional[BoardDetector] = None
        if ARUCO_AVAILABLE:
            try:
                self._detector = BoardDetector(dictionary_name)
            except Exception:
                pass

        self._det_lock = threading.Lock()
        self._det_snapshot = DetectionSnapshot()

    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not CV2_AVAILABLE:
            return False
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self.stop_recording()
        if self._cap:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    def start_recording(self, path: Path) -> bool:
        """Start FFmpeg pipe recorder."""
        try:
            self._pipe = PipeRecorder(
                self._ffmpeg_exe, path,
                self.width, self.height, self.fps,
            )
        except Exception as e:
            print(f"[ERROR] PipeRecorder failed to start: {e}")
            self._pipe = None
            return False
        self._recording = True
        return True

    def stop_recording(self):
        self._recording = False
        if self._pipe:
            self._pipe.release()
            self._pipe = None

    # ------------------------------------------------------------------
    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_detection(self) -> DetectionSnapshot:
        with self._det_lock:
            return DetectionSnapshot(
                bbox_norm=self._det_snapshot.bbox_norm,
                center_norm=self._det_snapshot.center_norm,
                num_markers=self._det_snapshot.num_markers,
                detected=self._det_snapshot.detected,
            )

    # ------------------------------------------------------------------
    def _loop(self):
        n = 0
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue

            # Record via FFmpeg pipe (skip bad-sized frames)
            if self._recording and self._pipe:
                fh, fw = frame.shape[:2]
                if fw == self.width and fh == self.height:
                    self._pipe.write(frame)

            # Detect periodically on left half
            n += 1
            if self._detector and n % self._detect_interval == 0:
                try:
                    left_half = frame[:, :self.half_width]
                    result = self._detector.detect(left_half)
                    snap = DetectionSnapshot()
                    if result:
                        snap.bbox_norm, snap.center_norm, snap.num_markers = result
                        snap.detected = True
                    with self._det_lock:
                        self._det_snapshot = snap
                except Exception:
                    pass  # detection failure is non-critical

            # Preview queue – keep only latest
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass


# ============================================================================
# Overlay Renderer
# ============================================================================

class OverlayRenderer:
    """Draws target regions, detection feedback, and text on the preview frame."""

    # Colours (BGR)
    GRAY   = (100, 100, 100)
    GREEN  = (0, 220, 0)
    YELLOW = (0, 220, 220)
    RED    = (0, 0, 220)
    CYAN   = (220, 220, 0)
    WHITE  = (255, 255, 255)

    @staticmethod
    def draw(frame: np.ndarray,
             target: PositionTarget,
             detection: DetectionSnapshot,
             in_position: bool,
             hold_progress: float,
             half_width: int,
             recording: bool) -> np.ndarray:
        """Draw all overlays and return the modified frame."""
        out = frame.copy()
        h, full_w = out.shape[:2]
        hw = half_width  # pixel width of one camera half in original frame

        # Scale factor from original to display (frame may already be original)
        # We work in original pixel coords then let the caller handle scaling.

        for offset in (0, hw):
            OverlayRenderer._draw_target_region(out, target, hw, h, offset,
                                                in_position, hold_progress)

        # Draw detected board bounding box on left half
        if detection.detected and detection.bbox_norm:
            bx1, by1, bx2, by2 = detection.bbox_norm
            px1 = int(bx1 * hw)
            py1 = int(by1 * h)
            px2 = int(bx2 * hw)
            py2 = int(by2 * h)
            color = OverlayRenderer.GREEN if in_position else OverlayRenderer.YELLOW
            cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
            # Mirror on right half
            cv2.rectangle(out, (px1 + hw, py1), (px2 + hw, py2), color, 2)

        # Recording indicator
        if recording:
            if int(time.time() * 2) % 2:
                cv2.circle(out, (40, 40), 18, OverlayRenderer.RED, -1)
            cv2.putText(out, "REC", (68, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, OverlayRenderer.RED, 2, cv2.LINE_AA)

        # Position name
        cv2.putText(out, target.name, (full_w // 2 - 120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, OverlayRenderer.WHITE, 2, cv2.LINE_AA)

        # Hold progress text
        if in_position and hold_progress > 0:
            pct = int(hold_progress * 100)
            txt = f"HOLD STEADY  {pct}%"
            cv2.putText(out, txt, (full_w // 2 - 160, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, OverlayRenderer.GREEN, 2, cv2.LINE_AA)
        elif detection.detected:
            cv2.putText(out, "Board detected - move to target",
                        (full_w // 2 - 250, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, OverlayRenderer.YELLOW, 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "Show the CharUco board to the camera",
                        (full_w // 2 - 280, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, OverlayRenderer.GRAY, 2, cv2.LINE_AA)

        return out

    @staticmethod
    def _draw_target_region(frame, target, hw, h, x_offset,
                            in_position, hold_progress):
        x1 = int(target.x1 * hw) + x_offset
        y1 = int(target.y1 * h)
        x2 = int(target.x2 * hw) + x_offset
        y2 = int(target.y2 * h)

        if in_position:
            # Semi‑transparent green fill showing hold progress
            overlay = frame.copy()
            fill_h = int((y2 - y1) * hold_progress)
            cv2.rectangle(overlay, (x1, y2 - fill_h), (x2, y2), OverlayRenderer.GREEN, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), OverlayRenderer.GREEN, 3)
        else:
            # Dashed‑style rectangle (draw with thinner line)
            cv2.rectangle(frame, (x1, y1), (x2, y2), OverlayRenderer.GRAY, 2)
            # Corner markers for emphasis
            corner_len = min(30, (x2 - x1) // 5, (y2 - y1) // 5)
            c = OverlayRenderer.CYAN
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), c, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), c, 3)


# ============================================================================
# Main UI
# ============================================================================

class CalibrationUIAdvanced(tk.Tk):

    def __init__(self, config_path: Optional[Path] = None,
                 project_dir: Optional[Path] = None,
                 on_complete: Optional[Callable[[], None]] = None):
        super().__init__()

        self._load_config(config_path)

        # Dynamic project directory (overrides config-derived path)
        self._project_dir: Optional[Path] = Path(project_dir) if project_dir else None
        self._on_complete = on_complete

        if self._project_dir:
            self.title(f"Stereo Calibration - Advanced - {self._project_dir.name}")
        else:
            self.title("Stereo Calibration - Advanced (Live Detection)")
        self.geometry("1280x820")
        self.configure(bg="#2b2b2b")

        # State
        self._state = "idle"  # idle | preview | recording | processing
        self._cam: Optional[CameraRecorder] = None
        self._positions = list(CALIBRATION_POSITIONS)
        self._pos_idx = 0
        self._rec_start: Optional[float] = None
        self._pos_start: Optional[float] = None
        self._in_position_since: Optional[float] = None
        self._session_dir: Optional[Path] = None
        self._raw_avi: Optional[Path] = None

        self._msg_q: queue.Queue = queue.Queue()

        self._build_ui()
        self.after(33, self._tick)  # ~30 fps UI refresh

    # ── config ──────────────────────────────────────────────────────
    def _load_config(self, config_path: Optional[Path]):
        if config_path and config_path.exists():
            self.config = CalibConfig.from_yaml(config_path)
        else:
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            default = tool_root / "configs" / "default.yaml"
            self.config = CalibConfig.from_yaml(default) if default.exists() else CalibConfig()

        if self.config.output_base and not os.path.isabs(self.config.output_base):
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            self.config.output_base = str(tool_root / self.config.output_base)

        # Read charuco dictionary from YAML
        self._charuco_dict = "DICT_4X4_100"
        try:
            from .config import load_yaml_config
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            default = tool_root / "configs" / "default.yaml"
            if default.exists():
                data = load_yaml_config(default)
                self._charuco_dict = data.get("charuco", {}).get("dictionary", self._charuco_dict)
        except Exception:
            pass

        # Parse video size
        parts = self.config.video_size.split("x")
        self._frame_w = int(parts[0])
        self._frame_h = int(parts[1])

    # ── build UI ────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 12))
        style.configure("Inst.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 11), padding=8)

        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # ── top bar ──
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(top, text="Stereo Calibration (Advanced)", style="Title.TLabel").pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="Camera Ready")
        self._status_lbl = ttk.Label(top, textvariable=self._status_var, style="Status.TLabel")
        self._status_lbl.pack(side=tk.RIGHT, padx=8)
        self._dot_cvs = tk.Canvas(top, width=18, height=18, highlightthickness=0)
        self._dot_cvs.pack(side=tk.RIGHT)
        self._dot = self._dot_cvs.create_oval(3, 3, 15, 15, fill="#888", outline="")

        # ── middle ──
        mid = ttk.Frame(main)
        mid.pack(fill=tk.BOTH, expand=True)

        # preview
        pf = ttk.LabelFrame(mid, text="Camera Preview (Live)", padding=4)
        pf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self._canvas = tk.Canvas(pf, bg="#111", width=840, height=340)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # guide panel
        gf = ttk.LabelFrame(mid, text="Calibration Guide", padding=10)
        gf.pack(side=tk.RIGHT, fill=tk.Y, ipadx=15)

        ttk.Label(gf, text="Position:").pack(anchor=tk.W)
        self._pos_var = tk.StringVar(value="--")
        ttk.Label(gf, textvariable=self._pos_var, style="Inst.TLabel",
                  foreground="#2196F3").pack(anchor=tk.W, pady=(4, 12))

        ttk.Label(gf, text="Instruction:").pack(anchor=tk.W)
        self._inst_var = tk.StringVar(value="Start camera to begin")
        ttk.Label(gf, textvariable=self._inst_var, wraplength=240).pack(anchor=tk.W, pady=(4, 12))

        ttk.Label(gf, text="Hold Progress:").pack(anchor=tk.W)
        self._hold_bar = ttk.Progressbar(gf, maximum=100, length=200)
        self._hold_bar.pack(anchor=tk.W, pady=(4, 12))

        ttk.Label(gf, text="Position Time:").pack(anchor=tk.W)
        self._pos_time_var = tk.StringVar(value="--")
        ttk.Label(gf, textvariable=self._pos_time_var,
                  font=("Segoe UI", 22, "bold")).pack(anchor=tk.W, pady=(4, 12))

        ttk.Label(gf, text="Total Time:").pack(anchor=tk.W)
        self._total_var = tk.StringVar(value="0:00 / 2:00")
        ttk.Label(gf, textvariable=self._total_var,
                  font=("Segoe UI", 13)).pack(anchor=tk.W, pady=(4, 12))

        # Detection info
        self._det_var = tk.StringVar(value="Detection: --")
        ttk.Label(gf, textvariable=self._det_var,
                  font=("Segoe UI", 9), foreground="#888").pack(anchor=tk.W, pady=(8, 0))

        # ── progress bar ──
        pbar = ttk.LabelFrame(main, text="Calibration Progress", padding=8)
        pbar.pack(fill=tk.X, pady=6)
        self._prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(pbar, variable=self._prog_var, maximum=100).pack(fill=tk.X, pady=(0, 6))
        pf2 = ttk.Frame(pbar)
        pf2.pack(fill=tk.X)
        self._plabels: List[ttk.Label] = []
        for p in self._positions:
            l = ttk.Label(pf2, text=p.name, font=("Segoe UI", 8), foreground="#888")
            l.pack(side=tk.LEFT, expand=True)
            self._plabels.append(l)

        # ── buttons ──
        bf = ttk.Frame(main)
        bf.pack(fill=tk.X, pady=6)

        # Camera selector
        cs = ttk.Frame(bf)
        cs.pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(cs, text="Camera:").pack(side=tk.LEFT, padx=(0, 3))
        self._cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(cs, from_=0, to=9, width=3,
                     textvariable=self._cam_idx, state="readonly").pack(side=tk.LEFT)
        ttk.Button(cs, text="Switch", command=self._on_switch_cam).pack(side=tk.LEFT, padx=(3, 0))

        self._btn_cam = ttk.Button(bf, text="Start Camera", style="Big.TButton",
                                   command=self._on_cam_toggle)
        self._btn_cam.pack(side=tk.LEFT, padx=4)

        self._btn_rec = ttk.Button(bf, text="Start Calibration Recording",
                                   style="Big.TButton", command=self._on_rec_start,
                                   state=tk.DISABLED)
        self._btn_rec.pack(side=tk.LEFT, padx=4)

        self._btn_next = ttk.Button(bf, text="Next Position",
                                    style="Big.TButton", command=self._on_next_pos,
                                    state=tk.DISABLED)
        self._btn_next.pack(side=tk.LEFT, padx=4)

        self._btn_stop = ttk.Button(bf, text="Stop", style="Big.TButton",
                                    command=self._on_stop, state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        # ── log ──
        lf = ttk.LabelFrame(main, text="Log", padding=4)
        lf.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self._log_txt = tk.Text(lf, height=5, bg="#111", fg="#eee", font=("Consolas", 9))
        self._log_txt.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(self._log_txt, command=self._log_txt.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_txt.config(yscrollcommand=sb.set)

        self._log("Advanced Calibration UI ready.")
        det_status = "available" if ARUCO_AVAILABLE else "NOT available (time-based fallback)"
        self._log(f"ArUco detection: {det_status}")
        self._log(f"Device: {self.config.device_name}  |  "
                  f"Resolution: {self.config.video_size} @ {self.config.fps}fps")

    # ── helpers ─────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_txt.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_txt.see(tk.END)

    def _set_status(self, text: str, color: str):
        self._status_var.set(text)
        self._dot_cvs.itemconfig(self._dot, fill=color)

    # ── camera ──────────────────────────────────────────────────────
    def _open_cam(self, idx: int) -> bool:
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            messagebox.showerror("Error", "opencv-python and Pillow are required.")
            return False
        ffmpeg_exe = _resolve_ffmpeg_exe(self.config.ffmpeg_exe)
        self._cam = CameraRecorder(
            device_index=idx,
            width=self._frame_w,
            height=self._frame_h,
            fps=self.config.fps,
            half_width=self.config.xsplit,
            ffmpeg_exe=ffmpeg_exe,
            dictionary_name=self._charuco_dict,
        )
        if self._cam.start():
            self._state = "preview"
            self._set_status(f"Camera {idx} Active", "#4CAF50")
            self._btn_cam.config(text="Stop Camera")
            self._btn_rec.config(state=tk.NORMAL)
            self._inst_var.set("Camera ready. Click 'Start Calibration Recording'.")
            self._log(f"Camera {idx} opened.")
            return True
        self._cam = None
        messagebox.showerror("Error", f"Cannot open camera {idx}.")
        self._log(f"Failed to open camera {idx}.")
        return False

    def _close_cam(self):
        if self._cam:
            self._cam.stop()
            self._cam = None
        self._state = "idle"
        self._set_status("Camera Ready", "#888")
        self._btn_cam.config(text="Start Camera")
        self._btn_rec.config(state=tk.DISABLED)
        self._btn_next.config(state=tk.DISABLED)
        self._inst_var.set("Start camera to begin")

    def _on_cam_toggle(self):
        if self._cam is None:
            self._open_cam(self._cam_idx.get())
        else:
            self._close_cam()
            self._log("Camera stopped.")

    def _on_switch_cam(self):
        if self._state == "recording":
            messagebox.showwarning("Recording", "Cannot switch while recording.")
            return
        was_open = self._cam is not None
        if was_open:
            self._close_cam()
            time.sleep(0.3)
        self._open_cam(self._cam_idx.get())

    # ── recording ───────────────────────────────────────────────────
    def _on_rec_start(self):
        if self._state != "preview" or self._cam is None:
            return

        # Session dir under raw_output/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._project_dir:
            project_root = self._project_dir
        else:
            from .paths import tool_root as _tool_root; tool_root = _tool_root()
            if self.config.output_base:
                out_base = Path(self.config.output_base)
                if not out_base.is_absolute():
                    out_base = tool_root / out_base
                project_root = out_base.parent
            else:
                project_root = tool_root
        raw_output_dir = project_root / "raw_output"
        self._session_dir = raw_output_dir / f"calib_session_{ts}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._raw_avi = self._session_dir / "raw.avi"

        if not self._cam.start_recording(self._raw_avi):
            messagebox.showerror("Error", "Failed to start recording.")
            self._log("VideoWriter open failed.")
            return

        self._state = "recording"
        self._pos_idx = 0
        self._rec_start = time.time()
        self._pos_start = time.time()
        self._in_position_since = None

        self._set_status("RECORDING", "#f44336")
        self._btn_cam.config(state=tk.DISABLED)
        self._btn_rec.config(state=tk.DISABLED)
        self._btn_next.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.NORMAL)
        self._update_pos_labels()
        self._log(f"Recording started → {self._session_dir}")

    def _on_next_pos(self):
        """Manual advance to next position."""
        if self._state != "recording":
            return
        self._advance_position()

    def _on_stop(self):
        if self._state == "recording":
            self._finish_recording()

    def _advance_position(self):
        self._pos_idx += 1
        self._pos_start = time.time()
        self._in_position_since = None
        if self._pos_idx >= len(self._positions):
            self._log("All positions completed!")
            self._finish_recording()
        else:
            self._update_pos_labels()
            self._log(f"→ {self._positions[self._pos_idx].name}")

    def _finish_recording(self):
        if self._cam:
            self._cam.stop_recording()

        self._state = "processing"
        self._set_status("Processing…", "#FF9800")
        self._btn_stop.config(state=tk.DISABLED)
        self._btn_next.config(state=tk.DISABLED)
        self._inst_var.set("Processing video…")
        self._pos_var.set("--")
        self._log("Recording stopped. Post-processing…")
        threading.Thread(target=self._post_process, daemon=True).start()

    # ── post-process ────────────────────────────────────────────────
    def _post_process(self):
        try:
            if self._project_dir:
                project_root = self._project_dir
            else:
                from .paths import tool_root as _tool_root; tool_root = _tool_root()
                if self.config.output_base:
                    out_base = Path(self.config.output_base)
                    if not out_base.is_absolute():
                        out_base = tool_root / out_base
                    project_root = out_base.parent
                else:
                    project_root = tool_root

            # Output to both intrinsic and extrinsic
            intrinsic_dir = project_root / "calibration" / "intrinsic"
            extrinsic_dir = project_root / "calibration" / "extrinsic"

            proc = PostProcessor(self.config, self._session_dir,
                                 on_log=lambda m: self._msg_q.put(("log", m)))
            ok = proc.run(self._raw_avi, [intrinsic_dir, extrinsic_dir])
            tag = "done" if ok else "error"
            msg = "Calibration complete!" if ok else "Post-processing failed."
            self._msg_q.put((tag, msg))
        except Exception as e:
            self._msg_q.put(("error", f"Error: {e}"))

    # ── position helpers ────────────────────────────────────────────
    def _cur_target(self) -> PositionTarget:
        idx = min(self._pos_idx, len(self._positions) - 1)
        return self._positions[idx]

    def _update_pos_labels(self):
        t = self._cur_target()
        self._pos_var.set(t.name)
        self._inst_var.set(t.instruction)
        for i, l in enumerate(self._plabels):
            if i < self._pos_idx:
                l.config(foreground="#4CAF50")
            elif i == self._pos_idx:
                l.config(foreground="#2196F3")
            else:
                l.config(foreground="#888")

    # ── main tick (≈30 fps) ─────────────────────────────────────────
    def _tick(self):
        # Drain messages
        while True:
            try:
                kind, msg = self._msg_q.get_nowait()
                if kind == "log":
                    self._log(msg)
                elif kind == "done":
                    self._on_done(msg)
                elif kind == "error":
                    self._on_err(msg)
            except queue.Empty:
                break

        # Update preview + recording logic
        if self._cam and self._state in ("preview", "recording"):
            self._render_frame()
        if self._state == "recording":
            self._update_recording()

        self.after(33, self._tick)

    # ── render frame ────────────────────────────────────────────────
    def _render_frame(self):
        frame = self._cam.get_frame()
        if frame is None:
            return

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # If recording, draw overlays on the frame BEFORE scaling
        if self._state == "recording":
            det = self._cam.get_detection()
            in_pos, hold_prog = self._check_position(det)
            frame = OverlayRenderer.draw(
                frame, self._cur_target(), det,
                in_pos, hold_prog, self.config.xsplit,
                recording=True,
            )
            self._hold_bar["value"] = hold_prog * 100

            # Detection info
            if det.detected:
                self._det_var.set(f"Detection: {det.num_markers} markers")
            else:
                self._det_var.set("Detection: no board")
        else:
            # Preview mode: just draw centre split line
            hw = self.config.xsplit
            fh = frame.shape[0]
            cv2.line(frame, (hw, 0), (hw, fh), (0, 255, 0), 2)

        # Scale to canvas
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        small = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=img)

        self._canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._canvas.create_image(ox, oy, anchor=tk.NW, image=photo)
        self._canvas._photo = photo  # prevent GC

    def _check_position(self, det: DetectionSnapshot) -> Tuple[bool, float]:
        """Return (in_position, hold_progress 0‑1)."""
        if not det.detected or det.center_norm is None:
            self._in_position_since = None
            return False, 0.0

        t = self._cur_target()
        cx, cy = det.center_norm
        inside = t.x1 <= cx <= t.x2 and t.y1 <= cy <= t.y2

        if not inside:
            self._in_position_since = None
            return False, 0.0

        now = time.time()
        if self._in_position_since is None:
            self._in_position_since = now
        elapsed = now - self._in_position_since
        prog = min(1.0, elapsed / t.hold_seconds)

        if prog >= 1.0:
            self._in_position_since = None
            self._advance_position()
            return True, 1.0

        return True, prog

    # ── update recording timers ─────────────────────────────────────
    def _update_recording(self):
        if not self._rec_start:
            return
        elapsed = time.time() - self._rec_start

        if elapsed >= MAX_CALIBRATION_TIME:
            self._set_status("Time Exceeded", "#f44336")
            self._inst_var.set("Calibration time exceeded. Click Stop or Next.")

        m, s = divmod(int(elapsed), 60)
        self._total_var.set(f"{m}:{s:02d} / 2:00")

        # Per-position time
        if self._pos_start:
            ps = int(time.time() - self._pos_start)
            self._pos_time_var.set(f"{ps}s")

        # Overall progress
        done = self._pos_idx
        total = len(self._positions)
        self._prog_var.set(done / total * 100)

    # ── callbacks ───────────────────────────────────────────────────
    def _on_done(self, msg: str):
        self._state = "idle"
        self._set_status("Complete", "#4CAF50")
        self._inst_var.set(msg)
        self._btn_cam.config(state=tk.NORMAL)
        self._prog_var.set(100)
        for l in self._plabels:
            l.config(foreground="#4CAF50")
        self._log(msg)
        messagebox.showinfo("Complete", msg)

    def _on_err(self, msg: str):
        self._state = "idle"
        self._set_status("Error", "#f44336")
        self._inst_var.set(msg)
        self._btn_cam.config(state=tk.NORMAL)
        self._log(msg)
        messagebox.showerror("Error", msg)

    def destroy(self):
        if self._cam:
            self._cam.stop()
        super().destroy()


# ============================================================================
# Entry point
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Advanced Stereo Calibration UI")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--project-dir", type=str, default=None)
    args = ap.parse_args()
    path = Path(args.config) if args.config else None
    proj = Path(args.project_dir) if args.project_dir else None
    app = CalibrationUIAdvanced(path, project_dir=proj)
    app.mainloop()


if __name__ == "__main__":
    main()
