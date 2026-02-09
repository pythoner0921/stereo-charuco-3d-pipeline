"""
SmartRecorder: Person-detection triggered auto-recording.

Monitors a stereo camera via OpenCV, runs person detection (OpenCV HOG),
and automatically records video segments when a person is present.

State machine: IDLE -> RECORDING -> COOLDOWN -> IDLE

Uses OpenCV VideoWriter (MJPG) for recording so that detection can continue
during recording (avoids Windows DirectShow exclusive access issue with FFmpeg).
"""
from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SmartRecorderConfig:
  """Configuration for SmartRecorder."""

  # Detection parameters
  detection_interval: int = 5        # Run detection every N frames
  detection_scale: float = 0.4       # Downscale factor for detection
  hog_hit_threshold: float = 0.0     # HOG SVM threshold (lower = more sensitive)
  hog_win_stride: int = 8            # HOG window stride
  hog_scale: float = 1.05            # HOG multi-scale step

  # State machine parameters
  absence_threshold: int = 5         # Consecutive detection misses before COOLDOWN
  cooldown_seconds: float = 300.0    # Seconds in COOLDOWN before stopping (5 min)

  # Camera parameters (from CalibConfig)
  device_name: str = "3D USB Camera"
  video_size: str = "3200x1200"
  fps: int = 60
  device_index: int = 0

  # Split parameters
  full_w: int = 3200
  full_h: int = 1200
  xsplit: int = 1600

  @classmethod
  def from_calib_config(cls, calib_config, device_index: int = 0) -> "SmartRecorderConfig":
    """Create SmartRecorderConfig from an existing CalibConfig."""
    return cls(
      device_name=calib_config.device_name,
      video_size=calib_config.video_size,
      fps=calib_config.fps,
      device_index=device_index,
      full_w=calib_config.full_w,
      full_h=calib_config.full_h,
      xsplit=calib_config.xsplit,
    )


# ============================================================================
# SmartRecorder
# ============================================================================

class SmartRecorder:
  """Person-detection triggered auto-recording with state machine.

  Monitors a stereo camera via OpenCV, runs person detection on every Nth frame
  using OpenCV's HOG person detector, and records video when a person is present.
  Recording uses OpenCV VideoWriter (MJPG fourcc), producing raw AVI files
  compatible with PostProcessor.

  Events emitted via on_event callback:
    {"type": "state_change", "state": "idle"|"recording"|"cooldown"}
    {"type": "recording_start", "visit": "visit_YYYYMMDD_HHMMSS"}
    {"type": "recording_stop", "visit": "visit_YYYYMMDD_HHMMSS", "session_dir": str}
    {"type": "error", "message": str}
  """

  def __init__(self, config: SmartRecorderConfig,
               output_base: Path,
               on_event: Optional[Callable[[dict], None]] = None):
    """
    Parameters
    ----------
    config : SmartRecorderConfig
        Camera and detection settings.
    output_base : Path
        Base directory for raw output (e.g. project/raw_output).
    on_event : callable, optional
        Callback for state/event notifications.
    """
    self.config = config
    self.output_base = Path(output_base)
    self.on_event = on_event

    # Parse video dimensions
    parts = config.video_size.split("x")
    self._full_w = int(parts[0])
    self._full_h = int(parts[1])

    # State
    self._state = "idle"  # idle | recording | cooldown
    self._running = False
    self._thread: Optional[threading.Thread] = None

    # Camera
    self._cap: Optional[cv2.VideoCapture] = None

    # Recording
    self._writer: Optional[cv2.VideoWriter] = None
    self._visit_name: Optional[str] = None
    self._session_dir: Optional[Path] = None
    self._recording_start_time: Optional[float] = None
    self._frame_count_recorded: int = 0
    self._visit_count: int = 0

    # Detection state
    self._consecutive_misses: int = 0
    self._cooldown_start: Optional[float] = None
    self._frame_count: int = 0
    self._last_detection_result: bool = False

    # Preview frame queue (thread-safe)
    self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

    # HOG person detector (lazy init)
    self._hog: Optional[cv2.HOGDescriptor] = None

  # ────────────────────────────────────────────────────────────────
  # Public API
  # ────────────────────────────────────────────────────────────────

  def start(self) -> bool:
    """Start monitoring. Opens camera and begins detection loop.

    Returns True if camera opened successfully.
    """
    if self._running:
      return True

    # Open camera
    if sys.platform == "win32":
      self._cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)
    else:
      self._cap = cv2.VideoCapture(self.config.device_index)

    if not self._cap.isOpened():
      self._emit({"type": "error", "message": "Failed to open camera"})
      return False

    # Set camera properties
    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._full_w)
    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._full_h)
    self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Initialize HOG person detector
    self._hog = cv2.HOGDescriptor()
    self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Reset state
    self._state = "idle"
    self._running = True
    self._frame_count = 0
    self._consecutive_misses = 0
    self._visit_count = 0

    self._emit({"type": "state_change", "state": "idle"})

    # Start monitor thread
    self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
    self._thread.start()

    logger.info("SmartRecorder started monitoring")
    return True

  def stop(self):
    """Stop monitoring. Finalizes any active recording."""
    self._running = False

    if self._thread:
      self._thread.join(timeout=3.0)
      self._thread = None

    # Finalize recording if active
    if self._state in ("recording", "cooldown"):
      self._stop_recording()

    # Release camera
    if self._cap:
      self._cap.release()
      self._cap = None

    # Release HOG detector
    self._hog = None

    self._state = "idle"
    logger.info("SmartRecorder stopped")

  def get_frame(self) -> Optional[np.ndarray]:
    """Get the latest preview frame (non-blocking, thread-safe).

    Returns a downscaled left-half frame suitable for UI preview,
    or None if no frame available.
    """
    try:
      return self._frame_queue.get_nowait()
    except queue.Empty:
      return None

  @property
  def state(self) -> str:
    """Current state: 'idle', 'recording', or 'cooldown'."""
    return self._state

  @property
  def is_running(self) -> bool:
    return self._running

  @property
  def visit_count(self) -> int:
    """Number of completed visits in this monitoring session."""
    return self._visit_count

  @property
  def recording_elapsed(self) -> float:
    """Seconds elapsed in current recording, or 0 if not recording."""
    if self._recording_start_time and self._state in ("recording", "cooldown"):
      return time.time() - self._recording_start_time
    return 0.0

  @property
  def cooldown_remaining(self) -> float:
    """Seconds remaining in cooldown, or 0 if not in cooldown."""
    if self._state == "cooldown" and self._cooldown_start:
      elapsed = time.time() - self._cooldown_start
      remaining = self.config.cooldown_seconds - elapsed
      return max(0.0, remaining)
    return 0.0

  # ────────────────────────────────────────────────────────────────
  # Monitor loop (runs in background thread)
  # ────────────────────────────────────────────────────────────────

  def _monitor_loop(self):
    """Main monitoring thread. Reads frames, runs detection, manages state."""
    while self._running and self._cap and self._cap.isOpened():
      ret, frame = self._cap.read()
      if not ret or frame is None:
        # Camera read failure — retry after short delay
        time.sleep(0.01)
        continue

      self._frame_count += 1

      # ── Person detection (every N frames) ──
      is_detection_frame = (self._frame_count % self.config.detection_interval == 0)
      person_detected = self._last_detection_result
      if is_detection_frame:
        person_detected = self._detect_person(frame)
        self._last_detection_result = person_detected

      # ── Write frame if recording ──
      if self._state in ("recording", "cooldown") and self._writer:
        self._writer.write(frame)
        self._frame_count_recorded += 1

      # ── State machine transitions ──
      self._update_state_machine(person_detected, is_detection_frame)

      # ── Update preview frame ──
      self._update_preview(frame)

    # Thread ending — cleanup
    if self._state in ("recording", "cooldown"):
      self._stop_recording()

  def _detect_person(self, frame: np.ndarray) -> bool:
    """Run OpenCV HOG person detector on downscaled left-half frame.

    Returns True if at least one person is detected.
    """
    if self._hog is None:
      return False

    try:
      # Crop left half only (detection on one view)
      xsplit = self.config.xsplit
      left = frame[:, :xsplit]

      # Downscale for faster detection
      scale = self.config.detection_scale
      small_w = int(xsplit * scale)
      small_h = int(self._full_h * scale)
      small = cv2.resize(left, (small_w, small_h))

      # Run HOG person detector
      stride = self.config.hog_win_stride
      boxes, weights = self._hog.detectMultiScale(
        small,
        hitThreshold=self.config.hog_hit_threshold,
        winStride=(stride, stride),
        padding=(4, 4),
        scale=self.config.hog_scale,
      )

      return len(boxes) > 0
    except Exception as e:
      logger.warning(f"Detection error: {e}")
      return False

  def _update_state_machine(self, person_detected: bool,
                            is_detection_frame: bool = True):
    """Handle state transitions based on detection result.

    Only counts misses on actual detection frames so that
    absence_threshold means "N consecutive detection checks
    with no person", not "N raw frames".
    """
    prev_state = self._state

    if self._state == "idle":
      if person_detected and is_detection_frame:
        self._start_recording()
        self._state = "recording"
        self._consecutive_misses = 0

    elif self._state == "recording":
      if not is_detection_frame:
        pass  # skip — only update on detection frames
      elif person_detected:
        self._consecutive_misses = 0
      else:
        self._consecutive_misses += 1
        if self._consecutive_misses >= self.config.absence_threshold:
          self._state = "cooldown"
          self._cooldown_start = time.time()

    elif self._state == "cooldown":
      if person_detected and is_detection_frame:
        # Person returned — resume recording
        self._state = "recording"
        self._consecutive_misses = 0
        self._cooldown_start = None
      elif self._cooldown_start is not None:
        elapsed = time.time() - self._cooldown_start
        if elapsed >= self.config.cooldown_seconds:
          # Cooldown expired — stop recording
          self._stop_recording()
          self._state = "idle"
          self._consecutive_misses = 0
          self._cooldown_start = None

    if self._state != prev_state:
      self._emit({"type": "state_change", "state": self._state})

  def _update_preview(self, frame: np.ndarray):
    """Put a downscaled left-half frame into the preview queue."""
    try:
      # Crop left half and downscale for preview
      xsplit = self.config.xsplit
      left = frame[:, :xsplit]
      preview = cv2.resize(left, (640, 480))

      # Clear old frame and put new one
      try:
        self._frame_queue.get_nowait()
      except queue.Empty:
        pass
      try:
        self._frame_queue.put_nowait(preview)
      except queue.Full:
        pass
    except Exception:
      pass

  # ────────────────────────────────────────────────────────────────
  # Recording management
  # ────────────────────────────────────────────────────────────────

  def _start_recording(self):
    """Begin a new recording segment (visit)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    self._visit_name = f"visit_{ts}"
    self._session_dir = self.output_base / self._visit_name
    self._session_dir.mkdir(parents=True, exist_ok=True)

    raw_avi = self._session_dir / "raw.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    self._writer = cv2.VideoWriter(
      str(raw_avi), fourcc, self.config.fps,
      (self._full_w, self._full_h),
    )

    if not self._writer.isOpened():
      logger.error(f"Failed to open VideoWriter: {raw_avi}")
      self._emit({"type": "error", "message": f"Failed to open VideoWriter: {raw_avi}"})
      self._writer = None
      return

    self._recording_start_time = time.time()
    self._frame_count_recorded = 0

    logger.info(f"Recording started: {self._visit_name}")
    self._emit({
      "type": "recording_start",
      "visit": self._visit_name,
    })

  def _stop_recording(self):
    """Finalize current recording segment."""
    if self._writer:
      self._writer.release()
      self._writer = None

    visit_name = self._visit_name
    session_dir = self._session_dir
    start_time = self._recording_start_time
    frame_count = self._frame_count_recorded

    # Write metadata
    if session_dir and session_dir.exists():
      end_time = time.time()
      duration = end_time - start_time if start_time else 0

      metadata = {
        "visit_id": visit_name,
        "start_time": datetime.fromtimestamp(start_time).isoformat() if start_time else None,
        "end_time": datetime.now().isoformat(),
        "duration_seconds": round(duration, 1),
        "frame_count": frame_count,
        "fps": self.config.fps,
        "resolution": self.config.video_size,
        "detection_interval": self.config.detection_interval,
        "cooldown_seconds": self.config.cooldown_seconds,
      }

      metadata_path = session_dir / "metadata.json"
      try:
        with open(metadata_path, "w") as f:
          json.dump(metadata, f, indent=2)
      except Exception as e:
        logger.warning(f"Failed to write metadata: {e}")

    self._visit_count += 1

    logger.info(f"Recording stopped: {visit_name} ({frame_count} frames)")
    self._emit({
      "type": "recording_stop",
      "visit": visit_name,
      "session_dir": str(session_dir) if session_dir else None,
      "frame_count": frame_count,
    })

    # Reset recording state
    self._visit_name = None
    self._session_dir = None
    self._recording_start_time = None
    self._frame_count_recorded = 0

  # ────────────────────────────────────────────────────────────────
  # Helpers
  # ────────────────────────────────────────────────────────────────

  def _emit(self, event: dict):
    """Emit an event via callback."""
    if self.on_event:
      try:
        self.on_event(event)
      except Exception as e:
        logger.warning(f"Event callback error: {e}")
