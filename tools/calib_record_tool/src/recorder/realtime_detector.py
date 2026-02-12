"""Real-time 2D pose detection without video recording.

Captures stereo camera frames via OpenCV, splits in-memory into
left/right views, runs parallel YOLO ONNX inference per port,
and accumulates 2D keypoints directly into memory.

Supports two operating modes:
  - Manual mode (auto_mode=False): User triggers start/stop, all frames
    are sent to YOLO inference.
  - Auto mode (auto_mode=True): Lightweight YOLO person detection triggers
    full YOLO pose inference only when a person is present. State machine:
    IDLE -> DETECTING -> COOLDOWN -> IDLE.

On stop (or session end in auto mode), writes xy_YOLOV8_POSE.csv
in caliscope ImagePoints format.  No video files are created.

Usage:
  from recorder.realtime_detector import RealtimeDetector, RealtimeDetectorConfig
  config = RealtimeDetectorConfig.from_calib_config(calib_config)
  detector = RealtimeDetector(config, output_base, on_event=print)
  detector.start()
  # ... wait for user to stop ...
  detector.stop()
"""
from __future__ import annotations

import gc
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import pandas as pd

from .yolov8_pose_tracker import MIN_CONFIDENCE, _ensure_onnx_model

logger = logging.getLogger(__name__)


@dataclass
class RealtimeDetectorConfig:
  """Configuration for real-time detection mode."""
  device_name: str = "3D USB Camera"
  video_size: str = "3200x1200"
  fps: int = 60
  device_index: int = 0
  full_w: int = 3200
  full_h: int = 1200
  xsplit: int = 1600
  # YOLO detection parameters
  fps_target: int = 20
  model_size: str = "n"
  imgsz: int = 480
  # Auto-monitoring parameters (only used when auto_mode=True)
  auto_mode: bool = False
  detection_interval: int = 10     # Person detection every N raw frames
  detection_imgsz: int = 192       # YOLO resolution for person detection (smaller = faster)
  detection_confidence: float = 0.3  # Min confidence for person detection
  absence_threshold: int = 15      # Consecutive detection misses before cooldown
  cooldown_seconds: float = 300.0  # Seconds in cooldown before stopping session
  # Low-light / night mode parameters
  low_light_enhance: bool = True       # Enable brightness gate + CLAHE enhancement
  brightness_min: int = 15             # Below this → too dark, skip detection entirely
  brightness_enhance_max: int = 80     # Below this → apply CLAHE before YOLO
  clahe_clip_limit: float = 3.0        # CLAHE contrast limit
  gamma: float = 0.6                   # Gamma correction (< 1.0 = brighten)

  @classmethod
  def from_calib_config(cls, calib_config, device_index: int = 0,
                        fps_target: int = 20, model_size: str = "n",
                        imgsz: int = 480,
                        auto_mode: bool = False,
                        **kwargs) -> "RealtimeDetectorConfig":
    """Create from CalibConfig.

    Extra kwargs are forwarded to the dataclass (e.g. detection_interval,
    absence_threshold, cooldown_seconds for auto_mode).
    """
    return cls(
      device_name=calib_config.device_name,
      video_size=calib_config.video_size,
      fps=calib_config.fps,
      device_index=device_index,
      full_w=calib_config.full_w,
      full_h=calib_config.full_h,
      xsplit=calib_config.xsplit,
      fps_target=fps_target,
      model_size=model_size,
      imgsz=imgsz,
      auto_mode=auto_mode,
      **kwargs,
    )


class RealtimeDetector:
  """Real-time 2D pose detection without video recording.

  Events emitted via on_event callback:
    # Both modes
    {"type": "started"}
    {"type": "stopped", "total_sessions": int}
    {"type": "error", "message": str}
    {"type": "csv_saved", "path": str, "session": str}
    # Manual mode only
    {"type": "session_start", "session": "session_YYYYMMDD_HHMMSS"}
    # Auto mode
    {"type": "auto_state_change", "state": "idle"|"detecting"|"cooldown"}
    {"type": "session_start", "session": "session_YYYYMMDD_HHMMSS"}
    {"type": "session_complete", "session": str, "total_frames": int, "total_keypoints": int}
  """

  def __init__(self, config: RealtimeDetectorConfig,
               output_base: Path,
               on_event: Optional[Callable[[dict], None]] = None):
    self.config = config
    self.output_base = Path(output_base)
    self.on_event = on_event

    # State
    self._running = False
    self._capture_thread: Optional[threading.Thread] = None
    self._session_name: Optional[str] = None
    self._session_dir: Optional[Path] = None

    # Camera
    self._cap: Optional[cv2.VideoCapture] = None

    # Inference threads (one per port)
    self._inference_threads: dict[int, threading.Thread] = {}
    self._frame_queues: dict[int, queue.Queue] = {
      1: queue.Queue(maxsize=4), 2: queue.Queue(maxsize=4),
    }
    self._result_queues: dict[int, queue.Queue] = {
      1: queue.Queue(maxsize=100), 2: queue.Queue(maxsize=100),
    }
    self._inference_running = False

    # Keypoint accumulator
    self._keypoint_rows: list[dict] = []
    self._lock = threading.Lock()
    self.frame_count = 0
    self.keypoint_count = 0

    # Preview frame
    self._preview_queue: queue.Queue = queue.Queue(maxsize=2)

    # Low-light enhancement (pre-built for reuse)
    if config.low_light_enhance:
      inv_gamma = 1.0 / config.gamma
      self._gamma_lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
      ], dtype=np.uint8)
      self._clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8),
      )
    else:
      self._gamma_lut = None
      self._clahe = None

    # Auto-mode state
    self._auto_state = "idle"  # idle | detecting | cooldown
    self._detection_model = None  # Lightweight YOLO for person detection (lazy-load)
    self._consecutive_misses = 0
    self._last_keypoint_time: float = 0.0
    self._cooldown_start: Optional[float] = None
    self._session_count = 0
    self._detecting_start_time: Optional[float] = None
    self._last_detection_result = False
    self._raw_frame_count = 0  # raw frames since start (for detection interval)

  # ── Public properties ──────────────────────────────────────

  @property
  def is_running(self) -> bool:
    return self._running

  @property
  def auto_state(self) -> str:
    """Current auto-mode state: 'idle', 'detecting', or 'cooldown'."""
    return self._auto_state

  @property
  def session_count(self) -> int:
    """Number of completed sessions (auto mode)."""
    return self._session_count

  @property
  def recording_elapsed(self) -> float:
    """Seconds elapsed in current detection session, or 0."""
    if self._detecting_start_time and self._auto_state in ("detecting", "cooldown"):
      return time.time() - self._detecting_start_time
    return 0.0

  @property
  def cooldown_remaining(self) -> float:
    """Seconds remaining in cooldown, or 0."""
    if self._auto_state == "cooldown" and self._cooldown_start:
      elapsed = time.time() - self._cooldown_start
      remaining = self.config.cooldown_seconds - elapsed
      return max(0.0, remaining)
    return 0.0

  # ── Public API ─────────────────────────────────────────────

  def start(self) -> bool:
    """Start capture. Returns True on success."""
    if self._running:
      return False

    # Open camera
    if not self._open_camera():
      return False

    # Reset global counters
    self._session_count = 0
    self._raw_frame_count = 0

    if self.config.auto_mode:
      # Auto mode: detection model is lazy-loaded on first use in capture thread
      self._auto_state = "idle"
      self._consecutive_misses = 0
      self._cooldown_start = None
      self._last_detection_result = False
      self._last_keypoint_time = 0.0
      logger.info("Auto-mode: YOLO person detection (lazy-load on first use)")
    else:
      # Manual mode: create session and start inference immediately
      self._create_session()
      self._start_inference_threads()

    # Start capture thread
    self._running = True
    self._capture_thread = threading.Thread(
      target=self._capture_loop,
      name="rt-capture", daemon=True,
    )
    self._capture_thread.start()

    self._emit({"type": "started"})
    if not self.config.auto_mode:
      self._emit({"type": "session_start", "session": self._session_name})
    logger.info(f"Real-time detector started (auto_mode={self.config.auto_mode})")
    return True

  def stop(self):
    """Stop capture, finalize any active session, cleanup."""
    if not self._running:
      return

    # Stop capture loop
    self._running = False
    if self._capture_thread:
      self._capture_thread.join(timeout=3.0)
      self._capture_thread = None

    # Finalize any active session
    if self.config.auto_mode:
      if self._auto_state in ("detecting", "cooldown"):
        self._finish_session()
      if self._detection_model is not None:
        del self._detection_model
        self._detection_model = None
        gc.collect()
    else:
      self._stop_inference_threads()
      self._drain_results()
      self._write_csv()

    # Release camera
    if self._cap:
      self._cap.release()
      self._cap = None

    self._emit({
      "type": "stopped",
      "total_sessions": self._session_count,
    })
    logger.info(f"Real-time detector stopped ({self._session_count} sessions)")

  def get_frame(self) -> Optional[np.ndarray]:
    """Get latest preview frame (non-blocking, for UI)."""
    try:
      return self._preview_queue.get_nowait()
    except queue.Empty:
      return None

  # ── Internal: camera ──────────────────────────────────────

  def _open_camera(self) -> bool:
    """Open stereo camera via OpenCV DirectShow."""
    try:
      if sys.platform == "win32":
        self._cap = cv2.VideoCapture(self.config.device_index, cv2.CAP_DSHOW)
      else:
        self._cap = cv2.VideoCapture(self.config.device_index)

      self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.full_w)
      self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.full_h)
      self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
      self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

      if not self._cap.isOpened():
        logger.error("Failed to open camera")
        return False

      logger.info(
        f"Camera opened: {self.config.full_w}x{self.config.full_h} "
        f"@{self.config.fps}fps"
      )
      return True
    except Exception as e:
      logger.error(f"Camera open error: {e}")
      return False

  # ── Internal: session management ───────────────────────────

  def _create_session(self):
    """Create a new session directory and reset accumulators."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    self._session_name = f"session_{ts}"
    self._session_dir = self.output_base / self._session_name
    self._session_dir.mkdir(parents=True, exist_ok=True)
    self._keypoint_rows = []
    self.frame_count = 0
    self.keypoint_count = 0
    self._detecting_start_time = time.time()

  def _start_inference_threads(self):
    """Start YOLO inference threads (one per port)."""
    # Clear queues
    for port in [1, 2]:
      while not self._frame_queues[port].empty():
        try:
          self._frame_queues[port].get_nowait()
        except queue.Empty:
          break
      while not self._result_queues[port].empty():
        try:
          self._result_queues[port].get_nowait()
        except queue.Empty:
          break

    self._inference_running = True
    for port in [1, 2]:
      t = threading.Thread(
        target=self._inference_loop, args=(port,),
        name=f"rt-inference-port{port}", daemon=True,
      )
      self._inference_threads[port] = t
      t.start()

  def _stop_inference_threads(self):
    """Send shutdown signal and wait for inference threads."""
    self._inference_running = False
    for port in [1, 2]:
      try:
        self._frame_queues[port].put_nowait(None)
      except queue.Full:
        pass
    for port, t in self._inference_threads.items():
      t.join(timeout=5.0)
    self._inference_threads.clear()

  # ── Internal: capture loop ─────────────────────────────────

  def _capture_loop(self):
    """Read frames at camera FPS, subsample, split, dispatch."""
    step = max(1, self.config.fps // self.config.fps_target)
    raw_frame_idx = 0
    consecutive_failures = 0

    while self._running and self._cap and self._cap.isOpened():
      ret, frame = self._cap.read()
      if not ret or frame is None:
        consecutive_failures += 1
        if consecutive_failures > 30:
          self._emit({"type": "error", "message": "Camera connection lost"})
          self._running = False
          break
        time.sleep(0.001)
        continue
      consecutive_failures = 0
      raw_frame_idx += 1
      self._raw_frame_count = raw_frame_idx

      if self.config.auto_mode:
        self._capture_loop_auto(frame, raw_frame_idx, step)
      else:
        self._capture_loop_manual(frame, raw_frame_idx, step)

  def _capture_loop_manual(self, frame: np.ndarray,
                           raw_frame_idx: int, step: int):
    """Manual mode: subsample and dispatch all frames to YOLO."""
    if raw_frame_idx % step != 0:
      return

    sync_idx = raw_frame_idx // step
    frame_time = raw_frame_idx / self.config.fps

    # Split in memory
    left = frame[:, :self.config.xsplit]
    right = frame[:, self.config.xsplit:]

    # Dispatch to inference threads
    for port, img in [(1, left), (2, right)]:
      try:
        self._frame_queues[port].put_nowait((sync_idx, frame_time, img.copy()))
      except queue.Full:
        logger.debug(f"Port {port} queue full, dropping frame {sync_idx}")

    # Update preview
    self._update_preview(frame)

    # Collect results
    self._drain_results()
    self.frame_count = sync_idx

  def _capture_loop_auto(self, frame: np.ndarray,
                         raw_frame_idx: int, step: int):
    """Auto mode: person detection + state machine + conditional YOLO."""
    # Person detection at configured interval
    is_detection_frame = (raw_frame_idx % self.config.detection_interval == 0)
    person_detected = self._last_detection_result
    if is_detection_frame:
      person_detected = self._detect_person(frame)
      self._last_detection_result = person_detected

    # State machine transition
    self._update_auto_state(person_detected, is_detection_frame)

    # Only dispatch to YOLO when actively detecting
    if self._auto_state in ("detecting", "cooldown"):
      if raw_frame_idx % step == 0:
        sync_idx = raw_frame_idx // step
        frame_time = raw_frame_idx / self.config.fps

        left = frame[:, :self.config.xsplit]
        right = frame[:, self.config.xsplit:]

        for port, img in [(1, left), (2, right)]:
          try:
            self._frame_queues[port].put_nowait((sync_idx, frame_time, img.copy()))
          except queue.Full:
            pass

        self._drain_results()
        self.frame_count = sync_idx

    # Always update preview (for UI display in all states)
    self._update_preview(frame)

  # ── Internal: preview ──────────────────────────────────────

  def _update_preview(self, frame: np.ndarray):
    """Put latest frame into preview queue."""
    try:
      self._preview_queue.get_nowait()
    except queue.Empty:
      pass
    try:
      self._preview_queue.put_nowait(frame)
    except queue.Full:
      pass

  # ── Internal: person detection (auto mode) ───────────────

  def _load_detection_model(self):
    """Load lightweight YOLO model for person detection (runs in capture thread)."""
    from ultralytics import YOLO
    try:
      onnx_path = _ensure_onnx_model(
        self.config.model_size, self.config.detection_imgsz,
      )
      self._detection_model = YOLO(str(onnx_path), task="pose")
      logger.info(f"Person detection model loaded: {onnx_path}")
    except Exception as e:
      logger.warning(f"ONNX detection model failed, fallback to PyTorch: {e}")
      self._detection_model = YOLO(f"yolov8{self.config.model_size}-pose.pt")

  def _enhance_low_light(self, img: np.ndarray) -> np.ndarray:
    """Apply gamma correction + CLAHE to brighten a dark frame.

    Uses pre-built LUT and CLAHE object for speed (~10ms on 1600x1200).
    """
    # Gamma correction (global brightness boost)
    enhanced = cv2.LUT(img, self._gamma_lut)
    # CLAHE on luminance channel (local contrast)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = self._clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

  def _detect_person(self, frame: np.ndarray) -> bool:
    """Run lightweight YOLO on left-half frame to detect persons.

    Much more robust than HOG: works for sitting, walking, bending,
    any angle. Uses a small ONNX model (192px) for speed (~5-15ms).

    Low-light mode: checks frame brightness first to avoid false
    positives in dark conditions. Applies CLAHE enhancement for
    dim (but not black) frames.
    """
    if self._detection_model is None:
      self._load_detection_model()
    if self._detection_model is None:
      return False
    try:
      left = frame[:, :self.config.xsplit]

      # Low-light brightness gate
      if self.config.low_light_enhance:
        gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < self.config.brightness_min:
          # Too dark — skip detection entirely (avoids noise false positives)
          return False
        if brightness < self.config.brightness_enhance_max and self._clahe is not None:
          # Dim but not black — enhance before detection
          left = self._enhance_low_light(left)

      results = self._detection_model(
        left, verbose=False, imgsz=self.config.detection_imgsz,
      )
      if not results or len(results) == 0:
        return False
      result = results[0]
      if result.boxes is None or len(result.boxes) == 0:
        return False
      # Check for person detections with sufficient confidence
      confs = result.boxes.conf
      if hasattr(confs, 'cpu'):
        confs = confs.cpu().numpy()
      else:
        confs = np.asarray(confs)
      return bool(np.any(confs >= self.config.detection_confidence))
    except Exception as e:
      logger.warning(f"Person detection error: {e}")
      return False

  # ── Internal: auto-mode state machine ──────────────────────

  def _update_auto_state(self, person_detected: bool,
                         is_detection_frame: bool):
    """Handle auto-mode state transitions."""
    prev_state = self._auto_state

    if self._auto_state == "idle":
      if person_detected and is_detection_frame:
        # Person appeared — start new session
        self._start_session()
        self._auto_state = "detecting"
        self._consecutive_misses = 0

    elif self._auto_state == "detecting":
      if not is_detection_frame:
        pass
      elif person_detected:
        self._consecutive_misses = 0
      else:
        self._consecutive_misses += 1
        if self._consecutive_misses >= self.config.absence_threshold:
          self._auto_state = "cooldown"
          self._cooldown_start = time.time()

    elif self._auto_state == "cooldown":
      if person_detected and is_detection_frame:
        # Person returned
        self._auto_state = "detecting"
        self._consecutive_misses = 0
        self._cooldown_start = None
      elif self._cooldown_start is not None:
        elapsed = time.time() - self._cooldown_start
        if elapsed >= self.config.cooldown_seconds:
          # Cooldown expired — finish session
          self._finish_session()
          self._auto_state = "idle"
          self._consecutive_misses = 0
          self._cooldown_start = None

    if self._auto_state != prev_state:
      self._emit({"type": "auto_state_change", "state": self._auto_state})

  def _start_session(self):
    """Start a new detection session (auto mode)."""
    self._create_session()
    self._start_inference_threads()
    self._emit({"type": "session_start", "session": self._session_name})
    logger.info(f"Auto session started: {self._session_name}")

  def _finish_session(self):
    """Finish current detection session (auto mode): stop inference, write CSV."""
    self._stop_inference_threads()
    self._drain_results()
    self._write_csv()

    session_name = self._session_name or ""
    frames = self.frame_count
    keypoints = self.keypoint_count
    self._session_count += 1

    self._emit({
      "type": "session_complete",
      "session": session_name,
      "total_frames": frames,
      "total_keypoints": keypoints,
    })
    logger.info(
      f"Auto session complete: {session_name} "
      f"({frames} frames, {keypoints} keypoints)"
    )

    # Reset for next session
    self._session_name = None
    self._session_dir = None
    self._detecting_start_time = None

  # ── Internal: inference loop (one per port) ────────────────

  def _inference_loop(self, port: int):
    """YOLO ONNX inference worker thread."""
    from ultralytics import YOLO

    try:
      onnx_path = _ensure_onnx_model(self.config.model_size, self.config.imgsz)
      model = YOLO(str(onnx_path), task="pose")
      logger.info(f"Port {port}: ONNX model loaded")
    except Exception as e:
      logger.warning(f"Port {port}: ONNX failed, fallback to PyTorch: {e}")
      model = YOLO(f"yolov8{self.config.model_size}-pose.pt")

    in_q = self._frame_queues[port]
    out_q = self._result_queues[port]

    while self._inference_running:
      try:
        item = in_q.get(timeout=0.2)
      except queue.Empty:
        continue

      if item is None:  # Shutdown signal
        break

      sync_idx, frame_time, img = item
      rows = self._detect_keypoints(model, img, sync_idx, frame_time, port)
      try:
        out_q.put_nowait(rows)
      except queue.Full:
        logger.warning(f"Port {port}: result queue full at sync {sync_idx}")

    del model
    gc.collect()
    logger.info(f"Port {port}: inference thread stopped")

  def _detect_keypoints(
    self, model, img: np.ndarray,
    sync_idx: int, frame_time: float, port: int,
  ) -> list[dict]:
    """Run YOLO on a single frame and extract keypoints."""
    results = model(img, verbose=False, imgsz=self.config.imgsz)
    if not results or len(results) == 0:
      return []

    result = results[0]
    if result.keypoints is None or result.keypoints.xy is None:
      return []

    raw_xy = result.keypoints.xy
    kpts_xy = raw_xy.cpu().numpy() if hasattr(raw_xy, 'cpu') else np.asarray(raw_xy)

    raw_conf = result.keypoints.conf
    if raw_conf is not None:
      kpts_conf = raw_conf.cpu().numpy() if hasattr(raw_conf, 'cpu') else np.asarray(raw_conf)
    else:
      kpts_conf = None

    if kpts_xy.shape[0] == 0:
      return []

    # Pick person with most high-confidence keypoints
    if kpts_conf is not None and kpts_xy.shape[0] > 1:
      best_idx = int(np.argmax((kpts_conf > MIN_CONFIDENCE).sum(axis=1)))
    else:
      best_idx = 0

    xy = kpts_xy[best_idx]
    conf = kpts_conf[best_idx] if kpts_conf is not None else np.ones(17)

    mask = conf >= MIN_CONFIDENCE
    point_ids = np.where(mask)[0]
    img_loc = xy[mask].astype(np.float64)

    if len(point_ids) == 0:
      return []

    rows = []
    for j, pid in enumerate(point_ids):
      rows.append({
        "sync_index": sync_idx,
        "port": port,
        "frame_index": sync_idx,
        "frame_time": frame_time,
        "point_id": int(pid),
        "img_loc_x": float(img_loc[j, 0]),
        "img_loc_y": float(img_loc[j, 1]),
      })
    return rows

  # ── Internal: result collection ────────────────────────────

  def _drain_results(self):
    """Collect all pending keypoint results from inference threads."""
    for port in [1, 2]:
      while True:
        try:
          rows = self._result_queues[port].get_nowait()
          with self._lock:
            self._keypoint_rows.extend(rows)
            self.keypoint_count += len(rows)
            if rows:
              self._last_keypoint_time = time.time()
        except queue.Empty:
          break

  # ── Internal: CSV output ───────────────────────────────────

  def _write_csv(self):
    """Write accumulated keypoints to xy_YOLOV8_POSE.csv."""
    if not self._session_dir:
      return

    output_dir = self._session_dir / "YOLOV8_POSE"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "xy_YOLOV8_POSE.csv"

    with self._lock:
      rows = list(self._keypoint_rows)

    if not rows:
      logger.warning("No keypoints detected, writing empty CSV")
      df = pd.DataFrame(columns=[
        "sync_index", "port", "frame_index", "frame_time",
        "point_id", "img_loc_x", "img_loc_y",
      ])
    else:
      df = pd.DataFrame(rows)
      # Re-assign sync_index as sequential counter
      unique_frames = sorted(df["frame_index"].unique())
      sync_map = {fi: si for si, fi in enumerate(unique_frames)}
      df["sync_index"] = df["frame_index"].map(sync_map)

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} keypoints to {csv_path}")

    self._emit({
      "type": "csv_saved",
      "path": str(csv_path),
      "session": self._session_name or "",
    })

  # ── Internal: event emission ───────────────────────────────

  def _emit(self, event: dict):
    """Emit event via callback."""
    if self.on_event:
      try:
        self.on_event(event)
      except Exception as e:
        logger.warning(f"Event callback error: {e}")
