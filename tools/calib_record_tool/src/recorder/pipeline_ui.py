"""
Simplified Pipeline UI: Auto-Calibrate -> Record -> Auto-Reconstruct

One-click workflow that replaces caliscope's manual 5-tab process.
Uses the same dark theme and threading patterns as calibration_ui.py.
"""
from __future__ import annotations

import logging
import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

try:
  import cv2
  import numpy as np
  CV2_AVAILABLE = True
except ImportError:
  CV2_AVAILABLE = False

try:
  import matplotlib
  matplotlib.use("Agg")  # Non-interactive backend; avoids "main thread is not in main loop"
                          # when ultralytics imports matplotlib from a background thread.
                          # FigureCanvasTkAgg works independently of the global backend.
  from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
  from matplotlib.figure import Figure
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
  MPL_AVAILABLE = True
except ImportError:
  MPL_AVAILABLE = False

from .config import load_yaml_config
from .calibration_ui import (
  CalibConfig,
  CameraPreview,
)
from .viz_3d import Viz3DData, load_xyz_csv, load_wireframe_for_tracker, render_frame
from .realtime_detector import RealtimeDetector, RealtimeDetectorConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

BG = "#2b2b2b"
BG_PANEL = "#333333"
FG = "#e0e0e0"
FG_DIM = "#888888"
ACCENT = "#4CAF50"
ACCENT_BLUE = "#2196F3"
ACCENT_ORANGE = "#FF9800"
ACCENT_RED = "#f44336"


# ============================================================================
# Pipeline UI
# ============================================================================

class PipelineUI(tk.Tk):
  """Simplified pipeline: calibrate -> record -> reconstruct."""

  def __init__(self, config_path: Optional[Path] = None,
               project_dir: Optional[Path] = None,
               master=None):
    if master is not None:
      tk.Toplevel.__init__(self, master)
    else:
      tk.Tk.__init__(self)

    self.geometry("1050x960")
    self.configure(bg=BG)

    # Message queue for thread -> UI communication
    self._msg_queue: queue.Queue = queue.Queue()

    # Load configuration
    self._load_config(config_path, project_dir_override=project_dir)

    # State
    self._state = "idle"  # idle | calibrating | calibrated | recording | monitoring | reconstructing
    self._camera: Optional[CameraPreview] = None

    # Real-time detector (used for both manual and auto modes)
    self._detector: Optional[RealtimeDetector] = None
    self._detector_timer_id: Optional[str] = None
    self._detector_event_queue: queue.Queue = queue.Queue()

    # Dining zone state
    self._zone_data: Optional[dict] = None  # {x1, y1, x2, y2} normalized
    self._zone_drawing = False
    self._zone_start: Optional[tuple] = None
    self._preview_scale = 1.0
    self._preview_x_offset = 0
    self._preview_y_offset = 0
    self._preview_left_w = 1600  # left camera width in source image pixels
    self._preview_left_h = 1200  # left camera height in source image pixels

    # Visualization state
    self._viz_data: Optional[Viz3DData] = None
    self._viz_playing = False
    self._viz_frame_idx = 0
    self._viz_speed = 1.0
    self._viz_timer_id: Optional[str] = None

    # Build UI
    self._build_ui()

    # Start update loop
    self.after(50, self._update_loop)

    # Check if already calibrated
    self._check_existing_calibration()

  def _load_config(self, config_path: Optional[Path],
                   project_dir_override: Optional[Path] = None):
    """Load configuration from YAML."""
    if config_path is None:
      from .paths import resolve_default_config
      config_path = resolve_default_config()

    self._yaml_data = load_yaml_config(config_path)
    self.config = CalibConfig.from_yaml(config_path)

    # Use explicit project_dir if provided
    if project_dir_override:
      self._project_dir = Path(project_dir_override)
      self.title(f"Stereo Pipeline - {self._project_dir.name}")
      logger.info(f"Project directory (override): {self._project_dir}")
      return

    self.title("Stereo Pipeline - Calibrate & Record")

    # Resolve project directory from YAML
    project_cfg = self._yaml_data.get("project", {})
    project_dir = project_cfg.get("project_dir", "")
    if not project_dir:
      # Derive from output.base_dir
      output_base = self._yaml_data.get("output", {}).get("base_dir", "")
      if output_base:
        base = Path(output_base)
        if not base.is_absolute():
          base = Path.cwd() / base
        self._project_dir = base.parent  # recordings -> parent = caliscope_project
      else:
        self._project_dir = Path(".")
    else:
      p = Path(project_dir)
      if not p.is_absolute():
        p = Path.cwd() / p
      self._project_dir = p

    logger.info(f"Project directory: {self._project_dir}")

  def _check_existing_calibration(self):
    """Check if calibration data already exists."""
    camera_array_path = self._project_dir / "camera_array.toml"
    charuco_dir = self._project_dir / "calibration" / "extrinsic" / "CHARUCO"
    bundle_exists = (charuco_dir / "camera_array.toml").exists()

    # Load existing dining zone if available
    self._zone_data = self._load_dining_zone()

    if camera_array_path.exists() and bundle_exists:
      self._state = "calibrated"
      self._update_state_display()
      if self._zone_data:
        self._log("Existing calibration and dining zone found. Ready to record.")
      else:
        self._log("Existing calibration found. Mark dining zone before recording.")

  # ========================================================================
  # UI Building
  # ========================================================================

  def _build_ui(self):
    """Build the main UI layout."""
    # Scrollable container
    outer = tk.Frame(self, bg=BG)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas_scroll = tk.Canvas(outer, bg=BG, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas_scroll.yview)
    main_frame = tk.Frame(canvas_scroll, bg=BG)

    main_frame.bind(
      "<Configure>",
      lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
    )
    canvas_scroll.create_window((0, 0), window=main_frame, anchor="nw", tags="inner")
    canvas_scroll.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Bind mousewheel scrolling
    def _on_mousewheel(event):
      canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)

    # Make inner frame fill the canvas width
    def _on_canvas_resize(event):
      canvas_scroll.itemconfig("inner", width=event.width)
    canvas_scroll.bind("<Configure>", _on_canvas_resize)

    self._main_canvas = canvas_scroll
    self._main_inner = main_frame

    # Add padding inside scrollable area
    main_frame.config(padx=10, pady=10)

    # Title
    tk.Label(
      main_frame, text="Stereo Pipeline",
      font=("Segoe UI", 18, "bold"), fg=FG, bg=BG,
    ).pack(pady=(0, 5))

    tk.Label(
      main_frame,
      text=f"Project: {self._project_dir.name}",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG,
    ).pack(pady=(0, 10))

    # ── Panel 1: Calibration ──────────────────────────────────────
    calib_frame = tk.LabelFrame(
      main_frame, text=" 1. Auto Calibration ",
      font=("Segoe UI", 11, "bold"), fg=ACCENT, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    calib_frame.pack(fill=tk.X, pady=(0, 8))

    calib_inner = tk.Frame(calib_frame, bg=BG_PANEL)
    calib_inner.pack(fill=tk.X, padx=10, pady=8)

    # Status indicator
    status_frame = tk.Frame(calib_inner, bg=BG_PANEL)
    status_frame.pack(fill=tk.X)

    self._calib_status_dot = tk.Label(
      status_frame, text="\u25CF", font=("Segoe UI", 14),
      fg=FG_DIM, bg=BG_PANEL,
    )
    self._calib_status_dot.pack(side=tk.LEFT, padx=(0, 5))

    self._calib_status_label = tk.Label(
      status_frame, text="Not calibrated",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
    )
    self._calib_status_label.pack(side=tk.LEFT)

    # Calibrate button
    self.btn_calibrate = tk.Button(
      status_frame, text="Auto Calibrate",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT, fg="white", activebackground="#388E3C",
      relief="flat", padx=15, pady=4,
      command=self._on_calibrate,
    )
    self.btn_calibrate.pack(side=tk.RIGHT)

    # Progress bar
    self._calib_progress = ttk.Progressbar(
      calib_inner, mode="determinate", maximum=100,
    )
    self._calib_progress.pack(fill=tk.X, pady=(8, 0))

    self._calib_step_label = tk.Label(
      calib_inner, text="", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
      anchor="w",
    )
    self._calib_step_label.pack(fill=tk.X)

    # ── Panel 2: Recording ────────────────────────────────────────
    rec_frame = tk.LabelFrame(
      main_frame, text=" 2. Record Session ",
      font=("Segoe UI", 11, "bold"), fg=ACCENT_BLUE, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    rec_frame.pack(fill=tk.X, pady=(0, 8))

    rec_inner = tk.Frame(rec_frame, bg=BG_PANEL)
    rec_inner.pack(fill=tk.X, padx=10, pady=8)

    # ── Dining Zone section ──
    zone_row = tk.Frame(rec_inner, bg=BG_PANEL)
    zone_row.pack(fill=tk.X, pady=(0, 6))

    self._zone_status_dot = tk.Label(
      zone_row, text="\u25CF", font=("Segoe UI", 11),
      fg=FG_DIM, bg=BG_PANEL,
    )
    self._zone_status_dot.pack(side=tk.LEFT, padx=(0, 4))

    self._zone_status_label = tk.Label(
      zone_row, text="Dining Zone: not defined",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
    )
    self._zone_status_label.pack(side=tk.LEFT)

    self.btn_clear_zone = tk.Button(
      zone_row, text="Clear",
      font=("Segoe UI", 9), bg="#555", fg=FG,
      relief="flat", padx=8, pady=2,
      command=self._on_clear_zone, state=tk.DISABLED,
    )
    self.btn_clear_zone.pack(side=tk.RIGHT, padx=2)

    self.btn_mark_zone = tk.Button(
      zone_row, text="Mark Zone",
      font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white",
      relief="flat", padx=10, pady=2,
      command=self._on_mark_zone, state=tk.DISABLED,
    )
    self.btn_mark_zone.pack(side=tk.RIGHT, padx=2)

    # Camera controls row
    cam_row = tk.Frame(rec_inner, bg=BG_PANEL)
    cam_row.pack(fill=tk.X)

    tk.Label(
      cam_row, text="Camera Index:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_cam_index = tk.IntVar(value=0)
    self._cam_spin = tk.Spinbox(
      cam_row, from_=0, to=10, width=4,
      textvariable=self.var_cam_index,
      font=("Segoe UI", 10),
    )
    self._cam_spin.pack(side=tk.LEFT, padx=5)

    self.btn_preview = tk.Button(
      cam_row, text="Preview",
      font=("Segoe UI", 9), bg="#555", fg=FG,
      relief="flat", padx=10, pady=2,
      command=self._on_toggle_preview,
    )
    self.btn_preview.pack(side=tk.LEFT, padx=5)

    # Mode toggle row (Manual / Auto Monitor)
    mode_row = tk.Frame(rec_inner, bg=BG_PANEL)
    mode_row.pack(fill=tk.X, pady=(6, 0))

    tk.Label(
      mode_row, text="Mode:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_rec_mode = tk.StringVar(value="manual")
    tk.Radiobutton(
      mode_row, text="Manual", variable=self.var_rec_mode,
      value="manual", command=self._on_rec_mode_change,
      font=("Segoe UI", 9), fg=FG, bg=BG_PANEL,
      selectcolor=BG, activebackground=BG_PANEL, activeforeground=FG,
    ).pack(side=tk.LEFT, padx=(8, 4))
    tk.Radiobutton(
      mode_row, text="Auto Monitor", variable=self.var_rec_mode,
      value="auto", command=self._on_rec_mode_change,
      font=("Segoe UI", 9), fg=FG, bg=BG_PANEL,
      selectcolor=BG, activebackground=BG_PANEL, activeforeground=FG,
    ).pack(side=tk.LEFT, padx=4)

    # ── Shared YOLO detection settings (used by both modes) ──
    self.var_rt_fps = tk.IntVar(value=20)
    self.var_rt_model = tk.StringVar(value="n")
    self.var_rt_imgsz = tk.IntVar(value=480)

    # ── Manual recording controls ──
    self._manual_frame = tk.Frame(rec_inner, bg=BG_PANEL)
    self._manual_frame.pack(fill=tk.X, pady=(4, 0))

    manual_btn_row = tk.Frame(self._manual_frame, bg=BG_PANEL)
    manual_btn_row.pack(fill=tk.X)

    self.btn_record = tk.Button(
      manual_btn_row, text="Start Recording",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT_RED, fg="white", activebackground="#C62828",
      relief="flat", padx=15, pady=4,
      state=tk.DISABLED,
      command=self._on_record,
    )
    self.btn_record.pack(side=tk.LEFT)

    self.btn_stop = tk.Button(
      manual_btn_row, text="Stop & Save",
      font=("Segoe UI", 10),
      bg="#666", fg=FG,
      relief="flat", padx=10, pady=4,
      state=tk.DISABLED,
      command=self._on_stop,
    )
    self.btn_stop.pack(side=tk.LEFT, padx=5)

    # Manual status display
    manual_status_frame = tk.Frame(self._manual_frame, bg=BG_PANEL)
    manual_status_frame.pack(fill=tk.X, pady=(4, 0))

    self._manual_state_dot = tk.Label(
      manual_status_frame, text="\u25CF", font=("Segoe UI", 12),
      fg=FG_DIM, bg=BG_PANEL,
    )
    self._manual_state_dot.pack(side=tk.LEFT, padx=(0, 4))

    self._manual_state_label = tk.Label(
      manual_status_frame, text="Ready",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
    )
    self._manual_state_label.pack(side=tk.LEFT)

    self._manual_stats_label = tk.Label(
      manual_status_frame, text="",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
    )
    self._manual_stats_label.pack(side=tk.RIGHT)

    # Manual detection settings
    manual_det_frame = tk.LabelFrame(
      self._manual_frame, text=" Detection Settings ",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    manual_det_frame.pack(fill=tk.X, pady=(4, 0))

    m_grid = tk.Frame(manual_det_frame, bg=BG_PANEL)
    m_grid.pack(fill=tk.X, padx=6, pady=4)

    tk.Label(m_grid, text="Target FPS:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=0, sticky="w")
    tk.Spinbox(m_grid, from_=5, to=30, width=4,
               textvariable=self.var_rt_fps,
               font=("Segoe UI", 9)).grid(row=0, column=1, padx=4)

    tk.Label(m_grid, text="Model:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=2, sticky="w", padx=(12, 0))
    tk.Spinbox(m_grid, values=("n", "s", "m"), width=4,
               textvariable=self.var_rt_model,
               font=("Segoe UI", 9)).grid(row=0, column=3, padx=4)

    tk.Label(m_grid, text="Img Size:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=4, sticky="w", padx=(12, 0))
    tk.Spinbox(m_grid, values=(320, 480, 640), width=5,
               textvariable=self.var_rt_imgsz,
               font=("Segoe UI", 9)).grid(row=0, column=5, padx=4)

    # ── Auto monitor controls ──
    self._auto_frame = tk.Frame(rec_inner, bg=BG_PANEL)
    # Hidden by default (manual mode)

    auto_btn_row = tk.Frame(self._auto_frame, bg=BG_PANEL)
    auto_btn_row.pack(fill=tk.X)

    self.btn_start_monitor = tk.Button(
      auto_btn_row, text="Start Monitoring",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT_BLUE, fg="white", activebackground="#1976D2",
      relief="flat", padx=15, pady=4,
      state=tk.DISABLED,
      command=self._on_start_monitor,
    )
    self.btn_start_monitor.pack(side=tk.LEFT)

    self.btn_stop_monitor = tk.Button(
      auto_btn_row, text="Stop Monitoring",
      font=("Segoe UI", 10),
      bg="#666", fg=FG,
      relief="flat", padx=10, pady=4,
      state=tk.DISABLED,
      command=self._on_stop_monitor,
    )
    self.btn_stop_monitor.pack(side=tk.LEFT, padx=5)

    # Auto-monitor status display
    self._auto_status_frame = tk.Frame(self._auto_frame, bg=BG_PANEL)
    self._auto_status_frame.pack(fill=tk.X, pady=(4, 0))

    self._auto_state_dot = tk.Label(
      self._auto_status_frame, text="\u25CF", font=("Segoe UI", 12),
      fg=FG_DIM, bg=BG_PANEL,
    )
    self._auto_state_dot.pack(side=tk.LEFT, padx=(0, 4))

    self._auto_state_label = tk.Label(
      self._auto_status_frame, text="Idle",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
    )
    self._auto_state_label.pack(side=tk.LEFT)

    self._auto_visit_label = tk.Label(
      self._auto_status_frame, text="Visits: 0",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
    )
    self._auto_visit_label.pack(side=tk.RIGHT)

    self._auto_timer_label = tk.Label(
      self._auto_status_frame, text="",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
    )
    self._auto_timer_label.pack(side=tk.RIGHT, padx=(0, 10))

    # Detection settings sub-frame
    det_frame = tk.LabelFrame(
      self._auto_frame, text=" Detection Settings ",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    det_frame.pack(fill=tk.X, pady=(4, 0))

    det_grid = tk.Frame(det_frame, bg=BG_PANEL)
    det_grid.pack(fill=tk.X, padx=6, pady=4)

    # Row 1: detection_interval | absence_threshold
    tk.Label(det_grid, text="Detect every:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=0, sticky="w")
    self.var_det_interval = tk.IntVar(value=10)
    tk.Spinbox(det_grid, from_=1, to=30, width=4,
               textvariable=self.var_det_interval,
               font=("Segoe UI", 9),
               command=lambda: self._on_detection_param_change(
                 "detection_interval", self.var_det_interval.get()),
               ).grid(row=0, column=1, padx=4)
    tk.Label(det_grid, text="frames", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=2, sticky="w")

    tk.Label(det_grid, text="Miss threshold:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=0, column=3, sticky="w", padx=(12, 0))
    self.var_absence = tk.IntVar(value=15)
    tk.Spinbox(det_grid, from_=1, to=30, width=4,
               textvariable=self.var_absence,
               font=("Segoe UI", 9),
               command=lambda: self._on_detection_param_change(
                 "absence_threshold", self.var_absence.get()),
               ).grid(row=0, column=4, padx=4)

    # Row 2: cooldown_seconds
    tk.Label(det_grid, text="Cooldown:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=1, column=0, sticky="w", pady=(4, 0))
    self.var_cooldown = tk.IntVar(value=300)
    tk.Spinbox(det_grid, from_=30, to=600, width=4,
               textvariable=self.var_cooldown,
               font=("Segoe UI", 9),
               command=lambda: self._on_detection_param_change(
                 "cooldown_seconds", float(self.var_cooldown.get())),
               ).grid(row=1, column=1, padx=4, pady=(4, 0))
    tk.Label(det_grid, text="sec", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=1, column=2, sticky="w", pady=(4, 0))

    # Row 3: YOLO settings (FPS / Model / ImgSize)
    tk.Label(det_grid, text="Target FPS:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=2, column=0, sticky="w", pady=(4, 0))
    tk.Spinbox(det_grid, from_=5, to=30, width=4,
               textvariable=self.var_rt_fps,
               font=("Segoe UI", 9)).grid(row=2, column=1, padx=4, pady=(4, 0))

    tk.Label(det_grid, text="Model:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=2, column=2, sticky="w", pady=(4, 0))
    tk.Spinbox(det_grid, values=("n", "s", "m"), width=4,
               textvariable=self.var_rt_model,
               font=("Segoe UI", 9)).grid(row=2, column=3, padx=4, pady=(4, 0))

    tk.Label(det_grid, text="Img Size:", font=("Segoe UI", 9),
             fg=FG_DIM, bg=BG_PANEL).grid(row=2, column=4, sticky="w", padx=(12, 0), pady=(4, 0))
    tk.Spinbox(det_grid, values=(320, 480, 640), width=5,
               textvariable=self.var_rt_imgsz,
               font=("Segoe UI", 9)).grid(row=2, column=5, padx=4, pady=(4, 0))

    # Camera preview canvas
    self.preview_canvas = tk.Canvas(
      rec_inner, bg="#1a1a1a", height=250,
      highlightthickness=0,
    )
    self.preview_canvas.pack(fill=tk.X, pady=(8, 0))

    # Recording timer
    self._rec_timer_label = tk.Label(
      rec_inner, text="", font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
    )
    self._rec_timer_label.pack(fill=tk.X)

    # ── Panel 3: Reconstruction ───────────────────────────────────
    recon_frame = tk.LabelFrame(
      main_frame, text=" 3. 3D Reconstruction ",
      font=("Segoe UI", 11, "bold"), fg=ACCENT_ORANGE, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    recon_frame.pack(fill=tk.X, pady=(0, 8))

    recon_inner = tk.Frame(recon_frame, bg=BG_PANEL)
    recon_inner.pack(fill=tk.X, padx=10, pady=8)

    recon_row = tk.Frame(recon_inner, bg=BG_PANEL)
    recon_row.pack(fill=tk.X)

    tk.Label(
      recon_row, text="Recording:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_recording = tk.StringVar()
    self._rec_combo = ttk.Combobox(
      recon_row, textvariable=self.var_recording,
      state="readonly", width=30,
    )
    self._rec_combo.pack(side=tk.LEFT, padx=5)

    self.btn_refresh_rec = tk.Button(
      recon_row, text="Refresh",
      font=("Segoe UI", 9), bg="#555", fg=FG,
      relief="flat", padx=8, pady=2,
      command=self._refresh_recordings,
    )
    self.btn_refresh_rec.pack(side=tk.LEFT, padx=2)

    # Tracker type selection row
    tracker_row = tk.Frame(recon_inner, bg=BG_PANEL)
    tracker_row.pack(fill=tk.X, pady=(6, 0))

    tk.Label(
      tracker_row, text="Tracker:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_tracker = tk.StringVar(value="HOLISTIC")
    self._tracker_combo = ttk.Combobox(
      tracker_row, textvariable=self.var_tracker,
      state="readonly", width=20,
    )
    self._tracker_combo.pack(side=tk.LEFT, padx=5)
    self._tracker_combo.bind("<<ComboboxSelected>>", self._on_tracker_changed)

    # YOLO model size selection row (visible only for YOLOV8_POSE)
    self._model_row = tk.Frame(recon_inner, bg=BG_PANEL)
    self._model_row.pack(fill=tk.X, pady=(6, 0))

    tk.Label(
      self._model_row, text="Model:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_model_size = tk.StringVar(value="n")
    self._model_combo = ttk.Combobox(
      self._model_row, textvariable=self.var_model_size,
      state="readonly", width=20,
      values=["n  (Nano - fastest)", "s  (Small)", "m  (Medium - accurate)"],
    )
    self._model_combo.pack(side=tk.LEFT, padx=5)
    self._model_combo.current(0)

    # YOLO inference resolution row (visible only for YOLOV8_POSE)
    self._imgsz_row = tk.Frame(recon_inner, bg=BG_PANEL)
    self._imgsz_row.pack(fill=tk.X, pady=(6, 0))

    tk.Label(
      self._imgsz_row, text="Infer Size:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_imgsz = tk.StringVar(value="480")
    self._imgsz_combo = ttk.Combobox(
      self._imgsz_row, textvariable=self.var_imgsz,
      state="readonly", width=20,
      values=["320  (fastest)", "480  (balanced)", "640  (best accuracy)"],
    )
    self._imgsz_combo.pack(side=tk.LEFT, padx=5)
    self._imgsz_combo.current(1)

    # Load tracker options after YOLO rows are created.
    self._load_tracker_options()

    # FPS selection row
    fps_row = tk.Frame(recon_inner, bg=BG_PANEL)
    fps_row.pack(fill=tk.X, pady=(6, 0))

    tk.Label(
      fps_row, text="FPS:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_recon_fps = tk.StringVar(value="10")
    self._fps_combo = ttk.Combobox(
      fps_row, textvariable=self.var_recon_fps,
      state="readonly", width=6,
      values=["10", "15", "20", "30", "60"],
    )
    self._fps_combo.pack(side=tk.LEFT, padx=5)

    self.btn_reconstruct = tk.Button(
      fps_row, text="Reconstruct 3D",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT_ORANGE, fg="white", activebackground="#E65100",
      relief="flat", padx=15, pady=4,
      state=tk.DISABLED,
      command=self._on_reconstruct,
    )
    self.btn_reconstruct.pack(side=tk.RIGHT)

    self._recon_progress = ttk.Progressbar(
      recon_inner, mode="determinate", maximum=100,
    )
    self._recon_progress.pack(fill=tk.X, pady=(8, 0))

    self._recon_label = tk.Label(
      recon_inner, text="", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
      anchor="w",
    )
    self._recon_label.pack(fill=tk.X)

    # ── Panel 4: 3D Visualization ──────────────────────────────────
    self._build_viz_panel(main_frame)

    # ── Log Panel ─────────────────────────────────────────────────
    log_frame = tk.LabelFrame(
      main_frame, text=" Log ",
      font=("Segoe UI", 10), fg=FG_DIM, bg=BG_PANEL,
      bd=1, relief="groove",
    )
    log_frame.pack(fill=tk.X)

    self._log_text = tk.Text(
      log_frame, bg="#1a1a1a", fg="#aaaaaa",
      font=("Consolas", 9), height=8,
      state=tk.DISABLED, wrap=tk.WORD,
      highlightthickness=0, bd=0,
    )
    self._log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Initial recordings refresh
    self._refresh_recordings()

  # ========================================================================
  # State Display
  # ========================================================================

  def _update_state_display(self):
    """Update UI elements based on current state."""
    if self._state == "idle":
      self._calib_status_dot.config(fg=FG_DIM)
      self._calib_status_label.config(text="Not calibrated", fg=FG_DIM)
      self.btn_calibrate.config(state=tk.NORMAL, text="Auto Calibrate")
      self.btn_record.config(state=tk.DISABLED)
      self.btn_reconstruct.config(state=tk.DISABLED)

    elif self._state == "calibrating":
      self._calib_status_dot.config(fg=ACCENT_ORANGE)
      self._calib_status_label.config(text="Calibrating...", fg=ACCENT_ORANGE)
      self.btn_calibrate.config(state=tk.DISABLED, text="Calibrating...")
      self.btn_record.config(state=tk.DISABLED)

    elif self._state == "calibrated":
      self._calib_status_dot.config(fg=ACCENT)
      self._calib_status_label.config(text="Calibrated", fg=ACCENT)
      self.btn_calibrate.config(state=tk.NORMAL, text="Re-calibrate")
      # Enable zone marking
      self.btn_mark_zone.config(state=tk.NORMAL)
      # Gate recording behind zone check
      if self._zone_data:
        self.btn_record.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_start_monitor.config(state=tk.NORMAL)
        self._zone_status_dot.config(fg=ACCENT)
        self._zone_status_label.config(text="Dining Zone: defined", fg=ACCENT)
        self.btn_clear_zone.config(state=tk.NORMAL)
      else:
        self.btn_record.config(state=tk.DISABLED)
        self.btn_start_monitor.config(state=tk.DISABLED)
        self._zone_status_dot.config(fg=FG_DIM)
        self._zone_status_label.config(
          text="Dining Zone: not defined (mark zone first)", fg=ACCENT_ORANGE)
      self._calib_progress["value"] = 100
      self._refresh_recordings()
      # Enable reconstruct if recordings available
      if self._rec_combo["values"]:
        self.btn_reconstruct.config(state=tk.NORMAL)

    elif self._state == "recording":
      self.btn_record.config(state=tk.DISABLED)
      self.btn_stop.config(state=tk.NORMAL)
      self.btn_calibrate.config(state=tk.DISABLED)
      self.btn_start_monitor.config(state=tk.DISABLED)
      self.btn_reconstruct.config(state=tk.DISABLED)

    elif self._state == "monitoring":
      self.btn_calibrate.config(state=tk.DISABLED)
      self.btn_record.config(state=tk.DISABLED)
      self.btn_start_monitor.config(state=tk.DISABLED)
      self.btn_stop_monitor.config(state=tk.NORMAL)
      self.btn_reconstruct.config(state=tk.DISABLED)

    elif self._state == "reconstructing":
      self.btn_reconstruct.config(state=tk.DISABLED)
      self.btn_record.config(state=tk.DISABLED)

  # ========================================================================
  # Calibration
  # ========================================================================

  def _on_calibrate(self):
    """Start auto-calibration in background thread."""
    # Verify calibration videos exist
    intrinsic_dir = self._project_dir / "calibration" / "intrinsic"
    extrinsic_dir = self._project_dir / "calibration" / "extrinsic"

    for port in [1, 2]:
      if not (intrinsic_dir / f"port_{port}.mp4").exists():
        messagebox.showerror("Error", f"Missing intrinsic video: port_{port}.mp4\n\nUse the recording tool first.")
        return
      if not (extrinsic_dir / f"port_{port}.mp4").exists():
        messagebox.showerror("Error", f"Missing extrinsic video: port_{port}.mp4\n\nUse the recording tool first.")
        return

    self._state = "calibrating"
    self._update_state_display()
    self._calib_progress["value"] = 0

    thread = threading.Thread(target=self._worker_calibrate, daemon=True)
    thread.start()

  def _worker_calibrate(self):
    """Background thread: run auto-calibration."""
    try:
      from .auto_calibrate import AutoCalibConfig, run_auto_calibration

      config = AutoCalibConfig.from_yaml(self._yaml_data, self._project_dir)

      def on_progress(stage, msg, pct):
        self._msg_queue.put(("calib_progress", (stage, msg, pct)))

      result = run_auto_calibration(config, on_progress=on_progress)

      if result.success:
        self._msg_queue.put(("calib_done", result))
      else:
        self._msg_queue.put(("calib_error", result.error_message))

    except Exception as e:
      self._msg_queue.put(("calib_error", str(e)))

  # ========================================================================
  # Recording
  # ========================================================================

  def _on_toggle_preview(self):
    """Toggle camera preview on/off."""
    if self._camera and self._camera._running:
      self._camera.stop()
      self._camera = None
      self.btn_preview.config(text="Preview")
      self.preview_canvas.delete("all")
      return

    if not CV2_AVAILABLE:
      messagebox.showerror("Error", "OpenCV not available for camera preview.")
      return

    self._camera = CameraPreview(
      self.config.device_name,
      self.config.video_size,
      self.config.fps,
      device_index=self.var_cam_index.get(),
    )

    if self._camera.start():
      self.btn_preview.config(text="Stop Preview")
    else:
      messagebox.showerror("Error", "Failed to open camera.")
      self._camera = None

  def _on_record(self):
    """Start manual recording (real-time detection, no video)."""
    # Stop preview first (camera needed by detector)
    if self._camera:
      self._camera.stop()
      self._camera = None
      self.btn_preview.config(text="Preview")
      time.sleep(0.3)

    # Read UI parameters in main thread
    fps_target = self.var_rt_fps.get()
    model_size = self.var_rt_model.get()
    imgsz = self.var_rt_imgsz.get()

    rt_config = RealtimeDetectorConfig.from_calib_config(
      self.config,
      device_index=self.var_cam_index.get(),
      fps_target=fps_target,
      model_size=model_size,
      imgsz=imgsz,
      auto_mode=False,
    )

    recordings_base = self._project_dir / "recordings"
    recordings_base.mkdir(parents=True, exist_ok=True)

    def on_event(event):
      self._detector_event_queue.put(event)

    self._detector = RealtimeDetector(rt_config, recordings_base, on_event=on_event)

    if not self._detector.start():
      messagebox.showerror("Error", "Failed to start recording.\nCheck camera connection.")
      self._detector = None
      return

    self._state = "recording"
    self._update_state_display()
    self._log("Recording started (real-time detection, no video)")
    self._manual_detect_tick()

  def _on_stop(self):
    """Stop manual recording, save CSV, and auto-reconstruct."""
    if self._detector_timer_id:
      self.after_cancel(self._detector_timer_id)
      self._detector_timer_id = None

    session_name = None
    if self._detector:
      session_name = self._detector._session_name
      self._detector.stop()
      self._detector = None

    # Reset manual mode status indicators
    self._manual_state_dot.config(fg=FG_DIM)
    self._manual_state_label.config(text="Ready", fg=FG_DIM)
    self._manual_stats_label.config(text="")
    self._rec_timer_label.config(text="")
    self._log("Recording stopped, CSV saved")

    # Auto-start 3D reconstruction
    if session_name:
      self._refresh_recordings()
      self.var_recording.set(session_name)
      self._log(f"Auto-reconstructing {session_name}...")
      self._on_reconstruct()
    else:
      self._state = "calibrated"
      self._update_state_display()

  def _manual_detect_tick(self):
    """Periodic UI update during manual recording (50ms)."""
    if not self._detector or not self._detector.is_running:
      return

    # Process events
    while True:
      try:
        event = self._detector_event_queue.get_nowait()
        self._handle_detector_event(event, mode="manual")
      except queue.Empty:
        break

    # Update status display
    dt = self._detector
    self._manual_state_dot.config(fg=ACCENT_RED)
    self._manual_state_label.config(text="Detecting...", fg=ACCENT_RED)
    self._manual_stats_label.config(
      text=f"Frames: {dt.frame_count}  |  Keypoints: {dt.keypoint_count}",
      fg=FG,
    )

    # Update preview from detector
    frame = dt.get_frame()
    if frame is not None:
      self._display_preview_frame(frame, show_detection=True)

    self._detector_timer_id = self.after(50, self._manual_detect_tick)

  # ========================================================================
  # Auto-Monitor Mode
  # ========================================================================

  def _on_rec_mode_change(self):
    """Switch between manual and auto recording modes."""
    mode = self.var_rec_mode.get()
    self._manual_frame.pack_forget()
    self._auto_frame.pack_forget()
    if mode == "manual":
      self._manual_frame.pack(fill=tk.X, pady=(4, 0))
    elif mode == "auto":
      self._auto_frame.pack(fill=tk.X, pady=(4, 0))

  def _on_start_monitor(self):
    """Start auto-monitoring with YOLO person detection + real-time pose."""
    # Stop any existing preview
    if self._camera:
      self._camera.stop()
      self._camera = None
      self.btn_preview.config(text="Preview")
      time.sleep(0.3)

    # Read UI parameters in main thread
    fps_target = self.var_rt_fps.get()
    model_size = self.var_rt_model.get()
    imgsz = self.var_rt_imgsz.get()
    detection_interval = self.var_det_interval.get()
    absence_threshold = self.var_absence.get()
    cooldown_seconds = float(self.var_cooldown.get())

    rt_config = RealtimeDetectorConfig.from_calib_config(
      self.config,
      device_index=self.var_cam_index.get(),
      fps_target=fps_target,
      model_size=model_size,
      imgsz=imgsz,
      auto_mode=True,
      detection_interval=detection_interval,
      absence_threshold=absence_threshold,
      cooldown_seconds=cooldown_seconds,
    )

    recordings_base = self._project_dir / "recordings"
    recordings_base.mkdir(parents=True, exist_ok=True)

    def on_event(event):
      self._detector_event_queue.put(event)

    self._detector = RealtimeDetector(rt_config, recordings_base, on_event=on_event)

    if not self._detector.start():
      messagebox.showerror("Error", "Failed to start auto-monitoring.\nCheck camera connection.")
      self._detector = None
      return

    self._state = "monitoring"
    self._update_state_display()
    self._log("Auto-monitoring started (YOLO person detection + pose, no video)")
    self._auto_monitor_tick()

  def _on_stop_monitor(self):
    """Stop auto-monitoring and reconstruct any active session."""
    if self._detector_timer_id:
      self.after_cancel(self._detector_timer_id)
      self._detector_timer_id = None

    session_name = None
    if self._detector:
      # If a session was active, save its name for reconstruction
      if self._detector.auto_state in ("detecting", "cooldown"):
        session_name = self._detector._session_name
      self._detector.stop()
      self._detector = None

    self._auto_state_dot.config(fg=FG_DIM)
    self._auto_state_label.config(text="Idle", fg=FG_DIM)
    self._auto_timer_label.config(text="")
    self._log("Auto-monitoring stopped.")

    # Auto-reconstruct the last active session
    if session_name:
      self._refresh_recordings()
      self.var_recording.set(session_name)
      self._log(f"Auto-reconstructing {session_name}...")
      self._on_reconstruct()
    else:
      self._state = "calibrated"
      self._update_state_display()

  def _auto_monitor_tick(self):
    """Periodic UI update during auto-monitoring (50ms interval)."""
    if not self._detector or not self._detector.is_running:
      return

    # Process events from detector
    while True:
      try:
        event = self._detector_event_queue.get_nowait()
        self._handle_detector_event(event, mode="auto")
      except queue.Empty:
        break

    # Update auto-monitor status display
    dt = self._detector
    state = dt.auto_state

    if state == "idle":
      self._auto_state_dot.config(fg=ACCENT)
      self._auto_state_label.config(text="Monitoring (idle)", fg=ACCENT)
      self._auto_timer_label.config(text="")
    elif state == "detecting":
      self._auto_state_dot.config(fg=ACCENT_RED)
      elapsed = int(dt.recording_elapsed)
      self._auto_state_label.config(text="DETECTING", fg=ACCENT_RED)
      self._auto_timer_label.config(
        text=f"{elapsed}s  |  Frames: {dt.frame_count}  |  KP: {dt.keypoint_count}",
        fg=ACCENT_RED,
      )
    elif state == "cooldown":
      remaining = int(dt.cooldown_remaining)
      self._auto_state_dot.config(fg=ACCENT_ORANGE)
      self._auto_state_label.config(text="Cooldown", fg=ACCENT_ORANGE)
      self._auto_timer_label.config(text=f"{remaining}s remaining", fg=ACCENT_ORANGE)

    self._auto_visit_label.config(text=f"Sessions: {dt.session_count}")

    # Update preview from detector
    frame = dt.get_frame()
    if frame is not None:
      self._display_preview_frame(frame, show_detection=(state == "detecting"))

    # Schedule next tick
    self._detector_timer_id = self.after(50, self._auto_monitor_tick)

  # ========================================================================
  # Detector Event Handling (shared by manual and auto modes)
  # ========================================================================

  def _handle_detector_event(self, event: dict, mode: str = "manual"):
    """Handle events from RealtimeDetector."""
    etype = event.get("type", "")
    prefix = "[AUTO]" if mode == "auto" else "[REC]"

    if etype == "started":
      self._log(f"{prefix} Detector started")

    elif etype == "stopped":
      sessions = event.get("total_sessions", 0)
      self._log(f"{prefix} Detector stopped ({sessions} sessions)")

    elif etype == "session_start":
      session = event.get("session", "")
      self._log(f"{prefix} Session started: {session}")

    elif etype == "session_complete":
      session = event.get("session", "")
      frames = event.get("total_frames", 0)
      kp = event.get("total_keypoints", 0)
      self._log(f"{prefix} Session complete: {session} ({frames} frames, {kp} keypoints)")
      self._refresh_recordings()

    elif etype == "auto_state_change":
      new_state = event.get("state", "")
      self._log(f"{prefix} State: {new_state}")

    elif etype == "csv_saved":
      session = event.get("session", "")
      self._log(f"{prefix} CSV saved: {event.get('path', '')}")
      self._refresh_recordings()
      # Auto-reconstruct completed sessions during monitoring
      if mode == "auto" and session:
        self._start_auto_reconstruct(session)

    elif etype == "error":
      msg = event.get("message", "")
      self._log(f"{prefix} Error: {msg}")

  def _start_auto_reconstruct(self, session_name: str):
    """Start 3D reconstruction in background without changing UI state.

    Used by auto-monitor mode when a session completes while monitoring
    is still active.
    """
    tracker_name = self.var_tracker.get()
    fps_target = int(self.var_recon_fps.get())
    model_size = self.var_model_size.get().split()[0]
    imgsz = int(self.var_imgsz.get().split()[0])

    self._recon_progress["value"] = 0
    self._log(f"Auto-reconstructing {session_name}...")
    self.var_recording.set(session_name)

    thread = threading.Thread(
      target=self._worker_reconstruct,
      args=(session_name, tracker_name, fps_target, model_size, imgsz),
      daemon=True,
    )
    thread.start()

  def _display_preview_frame(self, frame: np.ndarray, show_detection: bool = False):
    """Display a preview frame on the preview canvas."""
    canvas_w = self.preview_canvas.winfo_width()
    canvas_h = self.preview_canvas.winfo_height()
    if canvas_w < 10 or canvas_h < 10:
      return

    frame_h, frame_w = frame.shape[:2]
    scale = min(canvas_w / frame_w, canvas_h / frame_h)
    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)

    frame_resized = cv2.resize(frame, (new_w, new_h))

    # Add recording indicator overlay
    if show_detection:
      cv2.circle(frame_resized, (new_w - 20, 20), 8, (0, 0, 255), -1)

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    from PIL import Image, ImageTk
    img = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=img)

    self.preview_canvas.delete("all")
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    self.preview_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
    self.preview_canvas._photo = photo

    # Store preview transform (auto-monitor: frame IS left camera)
    self._preview_scale = scale
    self._preview_x_offset = x_offset
    self._preview_y_offset = y_offset
    self._preview_left_w = frame_w
    self._preview_left_h = frame_h

    # Draw zone overlay
    self._draw_zone_overlay()

  # ========================================================================
  # Dining Zone
  # ========================================================================

  def _on_mark_zone(self):
    """Enter zone-drawing mode. Start preview if needed."""
    # Start camera preview if not running (and not in auto-monitor mode)
    if not (self._camera and self._camera._running) and not self._detector:
      if not CV2_AVAILABLE:
        messagebox.showerror("Error", "OpenCV not available for camera preview.")
        return
      self._camera = CameraPreview(
        self.config.device_name,
        self.config.video_size,
        self.config.fps,
        device_index=self.var_cam_index.get(),
      )
      if self._camera.start():
        self.btn_preview.config(text="Stop Preview")
      else:
        messagebox.showerror("Error", "Failed to open camera for zone marking.")
        self._camera = None
        return

    self._zone_drawing = True
    self.preview_canvas.bind("<Button-1>", self._zone_mouse_press)
    self.preview_canvas.bind("<B1-Motion>", self._zone_mouse_drag)
    self.preview_canvas.bind("<ButtonRelease-1>", self._zone_mouse_release)
    self._zone_status_label.config(text="Dining Zone: draw on preview...", fg=ACCENT_ORANGE)
    self._zone_status_dot.config(fg=ACCENT_ORANGE)
    self._log("Draw dining zone on camera preview (click and drag)")

  def _on_clear_zone(self):
    """Clear the dining zone."""
    self._zone_data = None
    self._zone_drawing = False
    self.preview_canvas.delete("zone_rect")
    self.preview_canvas.delete("zone_overlay")
    # Unbind mouse events
    self.preview_canvas.unbind("<Button-1>")
    self.preview_canvas.unbind("<B1-Motion>")
    self.preview_canvas.unbind("<ButtonRelease-1>")
    # Delete dining_zone.toml
    zone_path = self._project_dir / "config" / "dining_zone.toml"
    if zone_path.exists():
      zone_path.unlink()
    # Update UI
    self._zone_status_dot.config(fg=FG_DIM)
    self._zone_status_label.config(text="Dining Zone: not defined", fg=FG_DIM)
    self.btn_clear_zone.config(state=tk.DISABLED)
    self._update_state_display()
    self._log("Dining zone cleared")

  def _zone_mouse_press(self, event):
    """Handle mouse press for zone drawing."""
    self._zone_start = (event.x, event.y)

  def _zone_mouse_drag(self, event):
    """Handle mouse drag for zone drawing — show dashed rectangle."""
    if self._zone_start is None:
      return
    self.preview_canvas.delete("zone_rect")
    self.preview_canvas.create_rectangle(
      self._zone_start[0], self._zone_start[1], event.x, event.y,
      outline="#4CAF50", width=2, dash=(5, 3), tags="zone_rect",
    )

  def _zone_mouse_release(self, event):
    """Handle mouse release — finalize zone rectangle."""
    if self._zone_start is None:
      return
    x1_c, y1_c = self._zone_start
    x2_c, y2_c = event.x, event.y
    self._zone_start = None
    self._zone_drawing = False

    # Ensure minimum size on canvas
    if abs(x2_c - x1_c) < 10 or abs(y2_c - y1_c) < 10:
      self.preview_canvas.delete("zone_rect")
      self._zone_status_label.config(text="Dining Zone: too small, try again", fg=ACCENT_RED)
      return

    # Normalize order
    if x1_c > x2_c:
      x1_c, x2_c = x2_c, x1_c
    if y1_c > y2_c:
      y1_c, y2_c = y2_c, y1_c

    # Convert canvas coords to source image coords
    img_x1 = (x1_c - self._preview_x_offset) / self._preview_scale
    img_y1 = (y1_c - self._preview_y_offset) / self._preview_scale
    img_x2 = (x2_c - self._preview_x_offset) / self._preview_scale
    img_y2 = (y2_c - self._preview_y_offset) / self._preview_scale

    # Normalize relative to left camera dimensions
    left_w = self._preview_left_w
    left_h = self._preview_left_h
    x1_norm = max(0.0, min(1.0, img_x1 / left_w))
    y1_norm = max(0.0, min(1.0, img_y1 / left_h))
    x2_norm = max(0.0, min(1.0, img_x2 / left_w))
    y2_norm = max(0.0, min(1.0, img_y2 / left_h))

    # Validate normalized size
    if x2_norm - x1_norm < 0.02 or y2_norm - y1_norm < 0.02:
      self.preview_canvas.delete("zone_rect")
      self._zone_status_label.config(text="Dining Zone: too small, try again", fg=ACCENT_RED)
      return

    # Store and save
    self._zone_data = {
      "x1": x1_norm, "y1": y1_norm,
      "x2": x2_norm, "y2": y2_norm,
    }
    self._save_dining_zone()

    # Unbind mouse events
    self.preview_canvas.unbind("<Button-1>")
    self.preview_canvas.unbind("<B1-Motion>")
    self.preview_canvas.unbind("<ButtonRelease-1>")

    # Update UI
    self.preview_canvas.delete("zone_rect")
    self._draw_zone_overlay()
    self._zone_status_dot.config(fg=ACCENT)
    self._zone_status_label.config(text="Dining Zone: defined", fg=ACCENT)
    self.btn_clear_zone.config(state=tk.NORMAL)
    self._update_state_display()
    self._log(f"Dining zone saved: ({x1_norm:.2f},{y1_norm:.2f})-({x2_norm:.2f},{y2_norm:.2f})")

  def _save_dining_zone(self):
    """Save dining zone to project config/dining_zone.toml."""
    if not self._zone_data:
      return
    config_dir = self._project_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    zone_path = config_dir / "dining_zone.toml"

    z = self._zone_data
    xsplit = self.config.xsplit
    full_h = self.config.full_h
    content = (
      "[zone]\n"
      f'name = "dining_area"\n'
      f"x1 = {z['x1']:.4f}\n"
      f"y1 = {z['y1']:.4f}\n"
      f"x2 = {z['x2']:.4f}\n"
      f"y2 = {z['y2']:.4f}\n"
      f"px_x1 = {int(z['x1'] * xsplit)}\n"
      f"px_y1 = {int(z['y1'] * full_h)}\n"
      f"px_x2 = {int(z['x2'] * xsplit)}\n"
      f"px_y2 = {int(z['y2'] * full_h)}\n"
    )
    zone_path.write_text(content)

  def _load_dining_zone(self) -> Optional[dict]:
    """Load dining zone from project config/dining_zone.toml."""
    zone_path = self._project_dir / "config" / "dining_zone.toml"
    if not zone_path.exists():
      return None
    try:
      data = {}
      for line in zone_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("[") and not line.startswith("#"):
          key, val = line.split("=", 1)
          key = key.strip()
          val = val.strip().strip('"')
          if key in ("x1", "y1", "x2", "y2"):
            data[key] = float(val)
      if all(k in data for k in ("x1", "y1", "x2", "y2")):
        return data
    except Exception as e:
      logger.warning(f"Failed to load dining zone: {e}")
    return None

  def _draw_zone_overlay(self):
    """Draw the dining zone rectangle overlay on preview canvas."""
    self.preview_canvas.delete("zone_overlay")
    if not self._zone_data:
      return
    z = self._zone_data
    left_w = self._preview_left_w
    left_h = self._preview_left_h
    scale = self._preview_scale
    x_off = self._preview_x_offset
    y_off = self._preview_y_offset

    # Convert normalized zone coords to canvas coords
    cx1 = z["x1"] * left_w * scale + x_off
    cy1 = z["y1"] * left_h * scale + y_off
    cx2 = z["x2"] * left_w * scale + x_off
    cy2 = z["y2"] * left_h * scale + y_off

    self.preview_canvas.create_rectangle(
      cx1, cy1, cx2, cy2,
      outline="#4CAF50", width=2, tags="zone_overlay",
    )
    self.preview_canvas.create_text(
      cx1 + 4, cy1 + 2, text="Dining Zone", anchor=tk.NW,
      fill="#4CAF50", font=("Segoe UI", 8), tags="zone_overlay",
    )

  def _on_detection_param_change(self, param_name: str, value):
    """Update detector config parameter in real-time."""
    if self._detector:
      try:
        setattr(self._detector.config, param_name, value)
        logger.debug(f"Detection param updated: {param_name}={value}")
      except Exception as e:
        logger.warning(f"Failed to update detection param: {e}")

  # ========================================================================
  # Reconstruction
  # ========================================================================

  def _load_tracker_options(self):
    """Load available tracker types from caliscope's TrackerEnum + custom trackers."""
    try:
      from caliscope.trackers.tracker_enum import TrackerEnum
      names = [t.name for t in TrackerEnum]
    except ImportError:
      names = ["HOLISTIC", "POSE", "HAND", "SIMPLE_HOLISTIC", "CHARUCO", "ARUCO"]
    # Append custom trackers not in caliscope's enum
    if "YOLOV8_POSE" not in names:
      names.append("YOLOV8_POSE")
    self._tracker_combo["values"] = names
    if "YOLOV8_POSE" in names:
      self.var_tracker.set("YOLOV8_POSE")
    self._toggle_yolo_options()

  def _on_tracker_changed(self, event=None):
    """Show/hide YOLO-specific options based on selected tracker."""
    self._toggle_yolo_options()

  def _toggle_yolo_options(self):
    """Show YOLO model/imgsz rows only when YOLOV8_POSE is selected."""
    is_yolo = self.var_tracker.get() == "YOLOV8_POSE"
    if is_yolo:
      self._model_row.pack(fill=tk.X, pady=(6, 0), after=self._tracker_combo.master)
      self._imgsz_row.pack(fill=tk.X, pady=(6, 0), after=self._model_row)
    else:
      self._model_row.pack_forget()
      self._imgsz_row.pack_forget()

  def _refresh_recordings(self):
    """Refresh the recordings dropdown."""
    rec_dir = self._project_dir / "recordings"
    recordings = []
    if rec_dir.exists():
      for d in sorted(rec_dir.iterdir()):
        if d.is_dir():
          # Check for xy CSV (real-time detection) or video files (legacy)
          has_xy_csv = (d / "YOLOV8_POSE" / "xy_YOLOV8_POSE.csv").exists()
          has_videos = (
            (d / "port_1.mp4").exists() or (d / "port_2.mp4").exists()
            or (d / "port_1.avi").exists() or (d / "port_2.avi").exists()
          )
          if has_xy_csv or has_videos:
            recordings.append(d.name)

    self._rec_combo["values"] = recordings
    if recordings:
      self._rec_combo.current(len(recordings) - 1)  # Select latest
      if self._state == "calibrated":
        self.btn_reconstruct.config(state=tk.NORMAL)

    # Also refresh Panel 4 recording list
    if hasattr(self, "_viz_rec_combo"):
      self._refresh_viz_recordings()

  def _on_reconstruct(self):
    """Start 3D reconstruction."""
    recording_name = self.var_recording.get()
    if not recording_name:
      messagebox.showerror("Error", "No recording selected.")
      return

    # Read all Tkinter variables in main thread (not safe from worker thread)
    tracker_name = self.var_tracker.get()
    fps_target = int(self.var_recon_fps.get())
    model_size = self.var_model_size.get().split()[0]
    imgsz = int(self.var_imgsz.get().split()[0])

    self._state = "reconstructing"
    self._update_state_display()
    self._recon_progress["value"] = 0

    thread = threading.Thread(
      target=self._worker_reconstruct,
      args=(recording_name, tracker_name, fps_target, model_size, imgsz),
      daemon=True,
    )
    thread.start()

  def _worker_reconstruct(self, recording_name: str,
                           tracker_name: str, fps_target: int,
                           model_size: str, imgsz: int):
    """Background: run 3D reconstruction."""
    try:
      from .auto_calibrate import run_auto_reconstruction

      def on_progress(stage, msg, pct):
        self._msg_queue.put(("recon_progress", (stage, msg, pct)))

      output_path = run_auto_reconstruction(
        self._project_dir,
        recording_name,
        tracker_name=tracker_name,
        fps_target=fps_target,
        model_size=model_size,
        imgsz=imgsz,
        on_progress=on_progress,
      )

      self._msg_queue.put(("recon_done", str(output_path)))

    except Exception as e:
      self._msg_queue.put(("recon_error", str(e)))

  # ========================================================================
  # 3D Visualization
  # ========================================================================

  def _build_viz_panel(self, parent):
    """Build Panel 4: 3D Visualization with playback controls."""
    viz_frame = tk.LabelFrame(
      parent, text=" 4. 3D Visualization ",
      font=("Segoe UI", 11, "bold"), fg="#9C27B0", bg=BG_PANEL,
      bd=1, relief="groove",
    )
    viz_frame.pack(fill=tk.X, pady=(0, 8))

    viz_inner = tk.Frame(viz_frame, bg=BG_PANEL)
    viz_inner.pack(fill=tk.X, padx=10, pady=8)

    # ── Top row: recording + tracker selectors + Load ──
    sel_row = tk.Frame(viz_inner, bg=BG_PANEL)
    sel_row.pack(fill=tk.X)

    tk.Label(
      sel_row, text="Recording:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_viz_recording = tk.StringVar()
    self._viz_rec_combo = ttk.Combobox(
      sel_row, textvariable=self.var_viz_recording,
      state="readonly", width=22,
    )
    self._viz_rec_combo.pack(side=tk.LEFT, padx=5)

    tk.Label(
      sel_row, text="Tracker:",
      font=("Segoe UI", 10), fg=FG, bg=BG_PANEL,
    ).pack(side=tk.LEFT, padx=(10, 0))

    self.var_viz_tracker = tk.StringVar(value="HOLISTIC")
    self._viz_tracker_combo = ttk.Combobox(
      sel_row, textvariable=self.var_viz_tracker,
      state="readonly", width=16,
    )
    self._viz_tracker_combo.pack(side=tk.LEFT, padx=5)
    # Share tracker options with Panel 3 combo
    self._viz_tracker_combo["values"] = self._tracker_combo["values"]

    self.btn_viz_load = tk.Button(
      sel_row, text="Load",
      font=("Segoe UI", 9, "bold"), bg="#9C27B0", fg="white",
      relief="flat", padx=12, pady=2,
      command=self._on_load_viz,
    )
    self.btn_viz_load.pack(side=tk.LEFT, padx=5)

    # ── Matplotlib 3D canvas ──
    if MPL_AVAILABLE:
      self._viz_fig = Figure(figsize=(10, 7), dpi=90, facecolor="#1a1a1a")
      self._viz_ax = self._viz_fig.add_subplot(111, projection="3d")
      self._viz_ax.set_facecolor("#1a1a1a")
      self._viz_fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

      self._viz_canvas = FigureCanvasTkAgg(self._viz_fig, master=viz_inner)
      self._viz_canvas.get_tk_widget().configure(height=550, bg="#1a1a1a")
      self._viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(8, 0))
      self._viz_canvas.draw()
    else:
      tk.Label(
        viz_inner,
        text="matplotlib not available — install it to enable 3D visualization",
        font=("Segoe UI", 10), fg=ACCENT_RED, bg=BG_PANEL,
      ).pack(fill=tk.X, pady=20)
      self._viz_fig = None
      self._viz_ax = None
      self._viz_canvas = None

    # ── Playback controls row ──
    ctrl_row = tk.Frame(viz_inner, bg=BG_PANEL)
    ctrl_row.pack(fill=tk.X, pady=(6, 0))

    self.btn_viz_prev = tk.Button(
      ctrl_row, text="\u25c0", font=("Segoe UI", 10),
      bg="#555", fg=FG, relief="flat", width=3,
      command=self._on_viz_prev, state=tk.DISABLED,
    )
    self.btn_viz_prev.pack(side=tk.LEFT)

    self.btn_viz_play = tk.Button(
      ctrl_row, text="\u25b6 Play", font=("Segoe UI", 10, "bold"),
      bg=ACCENT, fg="white", relief="flat", padx=10,
      command=self._on_viz_play_pause, state=tk.DISABLED,
    )
    self.btn_viz_play.pack(side=tk.LEFT, padx=4)

    self.btn_viz_next = tk.Button(
      ctrl_row, text="\u25b6", font=("Segoe UI", 10),
      bg="#555", fg=FG, relief="flat", width=3,
      command=self._on_viz_next, state=tk.DISABLED,
    )
    self.btn_viz_next.pack(side=tk.LEFT)

    # Frame slider
    self._viz_slider = tk.Scale(
      ctrl_row, from_=0, to=0, orient=tk.HORIZONTAL,
      bg=BG_PANEL, fg=FG, highlightthickness=0, troughcolor="#555",
      showvalue=False, command=self._on_viz_slider,
    )
    self._viz_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
    self._viz_slider.config(state=tk.DISABLED)

    self._viz_frame_label = tk.Label(
      ctrl_row, text="Frame: 0/0",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL, width=14,
    )
    self._viz_frame_label.pack(side=tk.LEFT)

    # Speed row
    speed_row = tk.Frame(viz_inner, bg=BG_PANEL)
    speed_row.pack(fill=tk.X, pady=(2, 0))

    tk.Label(
      speed_row, text="Speed:",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
    ).pack(side=tk.LEFT)

    self.var_viz_speed = tk.StringVar(value="1.0x")
    speed_combo = ttk.Combobox(
      speed_row, textvariable=self.var_viz_speed,
      state="readonly", width=6,
      values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"],
    )
    speed_combo.pack(side=tk.LEFT, padx=5)
    speed_combo.bind("<<ComboboxSelected>>", self._on_viz_speed_change)

    self._viz_status_label = tk.Label(
      speed_row, text="",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL, anchor="e",
    )
    self._viz_status_label.pack(side=tk.RIGHT)

  def _refresh_viz_recordings(self):
    """Refresh Panel 4 recording dropdown (same list as Panel 3)."""
    rec_dir = self._project_dir / "recordings"
    recordings = []
    if rec_dir.exists():
      for d in sorted(rec_dir.iterdir()):
        if d.is_dir():
          recordings.append(d.name)
    self._viz_rec_combo["values"] = recordings
    if recordings:
      self._viz_rec_combo.current(len(recordings) - 1)

  def _on_load_viz(self):
    """Load xyz CSV for the selected recording + tracker."""
    rec_name = self.var_viz_recording.get()
    tracker = self.var_viz_tracker.get()

    if not rec_name or not tracker:
      messagebox.showerror("Error", "Select a recording and tracker first.")
      return

    csv_path = (
      self._project_dir / "recordings" / rec_name / tracker / f"xyz_{tracker}.csv"
    )

    if not csv_path.exists():
      messagebox.showerror(
        "Error",
        f"No reconstruction found:\n{csv_path}\n\n"
        "Run 3D Reconstruction first (Panel 3).",
      )
      return

    self._log(f"Loading 3D data: {csv_path.name}")
    self.btn_viz_load.config(state=tk.DISABLED, text="Loading...")

    thread = threading.Thread(
      target=self._worker_load_viz,
      args=(csv_path, tracker),
      daemon=True,
    )
    thread.start()

  def _worker_load_viz(self, csv_path: Path, tracker_name: str):
    """Background: load CSV + wireframe data."""
    try:
      viz_data = load_xyz_csv(csv_path)
      segments = load_wireframe_for_tracker(tracker_name)
      viz_data.segments = segments
      self._msg_queue.put(("viz_loaded", viz_data))
    except Exception as e:
      self._msg_queue.put(("viz_error", str(e)))

  def _viz_set_data(self, viz_data: Viz3DData):
    """Set loaded data and enable playback controls."""
    self._viz_data = viz_data
    self._viz_frame_idx = 0
    self._viz_playing = False

    # Enable controls
    self.btn_viz_load.config(state=tk.NORMAL, text="Load")
    self.btn_viz_prev.config(state=tk.NORMAL)
    self.btn_viz_play.config(state=tk.NORMAL, text="\u25b6 Play")
    self.btn_viz_next.config(state=tk.NORMAL)

    self._viz_slider.config(state=tk.NORMAL, from_=0, to=max(0, viz_data.num_frames - 1))
    self._viz_slider.set(0)

    self._viz_frame_label.config(text=f"Frame: 1/{viz_data.num_frames}")
    self._viz_status_label.config(
      text=f"{viz_data.num_frames} frames, {len(viz_data.segments)} segments",
    )

    self._render_current_frame()

  def _render_current_frame(self):
    """Render the current frame on the 3D canvas."""
    if not self._viz_data or not self._viz_canvas:
      return
    render_frame(self._viz_ax, self._viz_data, self._viz_frame_idx)
    self._viz_canvas.draw_idle()
    self._viz_frame_label.config(
      text=f"Frame: {self._viz_frame_idx + 1}/{self._viz_data.num_frames}",
    )

  def _on_viz_prev(self):
    """Go to previous frame."""
    if not self._viz_data:
      return
    self._viz_frame_idx = max(0, self._viz_frame_idx - 1)
    self._viz_slider.set(self._viz_frame_idx)
    self._render_current_frame()

  def _on_viz_next(self):
    """Go to next frame."""
    if not self._viz_data:
      return
    self._viz_frame_idx = min(self._viz_data.num_frames - 1, self._viz_frame_idx + 1)
    self._viz_slider.set(self._viz_frame_idx)
    self._render_current_frame()

  def _on_viz_play_pause(self):
    """Toggle play/pause for animation."""
    if not self._viz_data:
      return

    self._viz_playing = not self._viz_playing

    if self._viz_playing:
      self.btn_viz_play.config(text="\u23f8 Pause", bg=ACCENT_ORANGE)
      # Reset to start if at the end
      if self._viz_frame_idx >= self._viz_data.num_frames - 1:
        self._viz_frame_idx = 0
      self._animation_tick()
    else:
      self.btn_viz_play.config(text="\u25b6 Play", bg=ACCENT)
      if self._viz_timer_id:
        self.after_cancel(self._viz_timer_id)
        self._viz_timer_id = None

  def _animation_tick(self):
    """Advance one frame during playback."""
    if not self._viz_playing or not self._viz_data:
      return

    self._viz_frame_idx += 1
    if self._viz_frame_idx >= self._viz_data.num_frames:
      # Reached the end — stop playback
      self._viz_frame_idx = self._viz_data.num_frames - 1
      self._viz_playing = False
      self.btn_viz_play.config(text="\u25b6 Play", bg=ACCENT)
      self._render_current_frame()
      self._viz_slider.set(self._viz_frame_idx)
      return

    self._render_current_frame()
    self._viz_slider.set(self._viz_frame_idx)

    # Schedule next tick based on speed
    interval_ms = max(1, int(33 / self._viz_speed))  # ~30fps at 1x
    self._viz_timer_id = self.after(interval_ms, self._animation_tick)

  def _on_viz_slider(self, value):
    """Handle slider drag for manual frame seek."""
    if not self._viz_data:
      return
    idx = int(float(value))
    if idx != self._viz_frame_idx:
      self._viz_frame_idx = idx
      self._render_current_frame()

  def _on_viz_speed_change(self, event=None):
    """Handle speed dropdown change."""
    speed_str = self.var_viz_speed.get().rstrip("x")
    try:
      self._viz_speed = float(speed_str)
    except ValueError:
      self._viz_speed = 1.0

  # ========================================================================
  # Update Loop
  # ========================================================================

  def _update_loop(self):
    """Main update loop (called every 50ms)."""
    # Process messages from background threads
    while True:
      try:
        msg_type, msg = self._msg_queue.get_nowait()
        self._handle_message(msg_type, msg)
      except queue.Empty:
        break

    # Update camera preview
    if self._camera and self._camera._running:
      self._update_preview()

    self.after(50, self._update_loop)

  def _handle_message(self, msg_type: str, msg):
    """Handle messages from background threads."""
    if msg_type == "log":
      self._log(msg)

    elif msg_type == "calib_progress":
      stage, text, pct = msg
      if pct >= 0:
        self._calib_progress["value"] = pct
      self._calib_step_label.config(text=text)
      self._log(f"[{stage}] {text}")

    elif msg_type == "calib_done":
      result = msg
      self._state = "calibrated"
      self._update_state_display()
      rmse_info = ", ".join(f"cam{p}={r:.3f}px" for p, r in result.intrinsic_rmse.items())
      self._log(f"Calibration complete! Intrinsic RMSE: {rmse_info}")
      self._log(f"Extrinsic cost: {result.extrinsic_cost:.4f}")
      self._log(f"Origin sync_index: {result.origin_sync_index}")
      self._calib_step_label.config(text="Calibration complete!")
      messagebox.showinfo("Success", "Auto-calibration complete!\nYou can now record sessions.")

    elif msg_type == "calib_error":
      self._state = "idle"
      self._update_state_display()
      self._log(f"[ERROR] Calibration failed: {msg}")
      self._calib_step_label.config(text=f"Error: {msg}")
      messagebox.showerror("Calibration Failed", str(msg))

    elif msg_type == "recon_progress":
      stage, text, pct = msg
      if pct >= 0:
        self._recon_progress["value"] = pct
      self._recon_label.config(text=text)
      self._log(f"[{stage}] {text}")

    elif msg_type == "recon_done":
      # Only change state if we're in "reconstructing" (manual/stop flow).
      # During monitoring, state stays "monitoring".
      if self._state == "reconstructing":
        self._state = "calibrated"
        self._update_state_display()
      self._recon_label.config(text="Reconstruction complete!")
      self._log(f"3D output: {msg}")
      # Auto-load result into Panel 4
      rec_name = self.var_recording.get()
      tracker = self.var_tracker.get()
      if rec_name and tracker:
        self.var_viz_recording.set(rec_name)
        self.var_viz_tracker.set(tracker)
        self._refresh_viz_recordings()
        self._on_load_viz()

    elif msg_type == "recon_error":
      if self._state == "reconstructing":
        self._state = "calibrated"
        self._update_state_display()
      self._log(f"[ERROR] Reconstruction failed: {msg}")
      self._recon_label.config(text=f"Error: {msg}")

    elif msg_type == "viz_loaded":
      viz_data = msg
      self._viz_set_data(viz_data)
      self._log(f"3D visualization loaded: {viz_data.num_frames} frames")

    elif msg_type == "viz_error":
      self.btn_viz_load.config(state=tk.NORMAL, text="Load")
      self._log(f"[ERROR] Viz load failed: {msg}")
      messagebox.showerror("Error", f"Failed to load 3D data:\n{msg}")

  def _update_preview(self):
    """Update camera preview on canvas."""
    if not self._camera:
      return

    frame = self._camera.get_frame()
    if frame is None:
      return

    canvas_w = self.preview_canvas.winfo_width()
    canvas_h = self.preview_canvas.winfo_height()
    if canvas_w < 10 or canvas_h < 10:
      return

    frame_h, frame_w = frame.shape[:2]
    scale = min(canvas_w / frame_w, canvas_h / frame_h)
    new_w = int(frame_w * scale)
    new_h = int(frame_h * scale)

    frame_resized = cv2.resize(frame, (new_w, new_h))

    # Draw center line (stereo split indicator)
    center_x = new_w // 2
    cv2.line(frame_resized, (center_x, 0), (center_x, new_h), (0, 255, 0), 2)

    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    from PIL import Image, ImageTk
    img = Image.fromarray(frame_rgb)
    photo = ImageTk.PhotoImage(image=img)

    self.preview_canvas.delete("all")
    x_offset = (canvas_w - new_w) // 2
    y_offset = (canvas_h - new_h) // 2
    self.preview_canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
    self.preview_canvas._photo = photo  # Keep reference

    # Store preview transform (manual preview: full stereo frame)
    self._preview_scale = scale
    self._preview_x_offset = x_offset
    self._preview_y_offset = y_offset
    self._preview_left_w = self.config.xsplit  # left camera width in source pixels
    self._preview_left_h = frame_h

    # Draw zone overlay
    self._draw_zone_overlay()

  def _log(self, msg: str):
    """Append message to the log panel."""
    ts = datetime.now().strftime("%H:%M:%S")
    self._log_text.config(state=tk.NORMAL)
    self._log_text.insert(tk.END, f"[{ts}] {msg}\n")
    self._log_text.see(tk.END)
    self._log_text.config(state=tk.DISABLED)

  def destroy(self):
    """Cleanup on window close."""
    if self._detector:
      self._detector.stop()
      self._detector = None
    if self._detector_timer_id:
      self.after_cancel(self._detector_timer_id)
      self._detector_timer_id = None
    if self._camera:
      self._camera.stop()
    super().destroy()
