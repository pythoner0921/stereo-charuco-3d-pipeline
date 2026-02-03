"""
Project Manager UI: Select or create a project before launching the pipeline.

Shows existing projects under the projects base directory and allows
creating new timestamped project folders with the correct scaffold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

logger = logging.getLogger(__name__)

# Theme constants (match calibration_ui / pipeline_ui)
BG = "#2b2b2b"
BG_PANEL = "#333333"
BG_LIST = "#1a1a1a"
FG = "#e0e0e0"
FG_DIM = "#888888"
ACCENT = "#4CAF50"
ACCENT_BLUE = "#2196F3"
ACCENT_ORANGE = "#FF9800"


@dataclass
class ProjectContext:
  """Carries dynamic project paths through the unified workflow."""
  project_dir: Path
  project_name: str
  is_new: bool
  needs_calibration: bool  # True if calibration/intrinsic/port_1.mp4 missing


@dataclass
class _ProjectInfo:
  """Internal: scanned project metadata for display."""
  name: str
  path: Path
  has_calib_videos: bool
  calibrated: bool
  num_recordings: int


class ProjectManagerUI(tk.Tk):
  """Project selection/creation dialog."""

  def __init__(self, projects_base: Path):
    super().__init__()

    self.projects_base = projects_base
    self.projects_base.mkdir(parents=True, exist_ok=True)

    self.result: Optional[ProjectContext] = None
    self._projects: list[_ProjectInfo] = []

    self.title("Stereo Pipeline - Project Manager")
    self.geometry("700x450")
    self.configure(bg=BG)
    self.resizable(False, False)

    self._build_ui()
    self._scan_and_display()

    # Center window on screen
    self.update_idletasks()
    x = (self.winfo_screenwidth() - 700) // 2
    y = (self.winfo_screenheight() - 450) // 2
    self.geometry(f"700x450+{x}+{y}")

  # ======================================================================
  # UI
  # ======================================================================

  def _build_ui(self):
    # Title
    tk.Label(
      self, text="Project Manager",
      font=("Segoe UI", 16, "bold"), fg=FG, bg=BG,
    ).pack(pady=(15, 5))

    tk.Label(
      self, text=f"Base: {self.projects_base}",
      font=("Segoe UI", 9), fg=FG_DIM, bg=BG,
    ).pack(pady=(0, 10))

    # Main content: list + details side by side
    content = tk.Frame(self, bg=BG)
    content.pack(fill=tk.BOTH, expand=True, padx=15)

    # ── Left: project list ──
    left = tk.Frame(content, bg=BG)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

    tk.Label(
      left, text="Projects", font=("Segoe UI", 10, "bold"),
      fg=FG, bg=BG, anchor="w",
    ).pack(fill=tk.X)

    list_frame = tk.Frame(left, bg=BG_LIST)
    list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    self._listbox = tk.Listbox(
      list_frame, bg=BG_LIST, fg=FG,
      font=("Consolas", 10), selectmode=tk.SINGLE,
      selectbackground=ACCENT_BLUE, selectforeground="white",
      highlightthickness=0, bd=0,
    )
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                              command=self._listbox.yview)
    self._listbox.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    self._listbox.bind("<<ListboxSelect>>", self._on_select)
    self._listbox.bind("<Double-Button-1>", lambda e: self._on_open())

    # ── Right: details panel ──
    right = tk.Frame(content, bg=BG_PANEL, width=250)
    right.pack(side=tk.RIGHT, fill=tk.Y)
    right.pack_propagate(False)

    tk.Label(
      right, text="Details", font=("Segoe UI", 10, "bold"),
      fg=FG, bg=BG_PANEL, anchor="w",
    ).pack(fill=tk.X, padx=10, pady=(10, 5))

    self._detail_name = tk.Label(
      right, text="(select a project)", font=("Segoe UI", 10),
      fg=FG_DIM, bg=BG_PANEL, anchor="w", wraplength=220,
    )
    self._detail_name.pack(fill=tk.X, padx=10, pady=2)

    self._detail_calib = tk.Label(
      right, text="", font=("Segoe UI", 9),
      fg=FG_DIM, bg=BG_PANEL, anchor="w",
    )
    self._detail_calib.pack(fill=tk.X, padx=10, pady=2)

    self._detail_status = tk.Label(
      right, text="", font=("Segoe UI", 9),
      fg=FG_DIM, bg=BG_PANEL, anchor="w",
    )
    self._detail_status.pack(fill=tk.X, padx=10, pady=2)

    self._detail_recordings = tk.Label(
      right, text="", font=("Segoe UI", 9),
      fg=FG_DIM, bg=BG_PANEL, anchor="w",
    )
    self._detail_recordings.pack(fill=tk.X, padx=10, pady=2)

    self._detail_path = tk.Label(
      right, text="", font=("Segoe UI", 8),
      fg=FG_DIM, bg=BG_PANEL, anchor="w", wraplength=220,
    )
    self._detail_path.pack(fill=tk.X, padx=10, pady=(10, 2))

    # ── Bottom: buttons ──
    btn_frame = tk.Frame(self, bg=BG)
    btn_frame.pack(fill=tk.X, padx=15, pady=15)

    self.btn_new = tk.Button(
      btn_frame, text="New Project",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT, fg="white", activebackground="#388E3C",
      relief="flat", padx=20, pady=6,
      command=self._on_new,
    )
    self.btn_new.pack(side=tk.LEFT)

    self.btn_open = tk.Button(
      btn_frame, text="Open Project",
      font=("Segoe UI", 10, "bold"),
      bg=ACCENT_BLUE, fg="white", activebackground="#1565C0",
      relief="flat", padx=20, pady=6,
      state=tk.DISABLED,
      command=self._on_open,
    )
    self.btn_open.pack(side=tk.LEFT, padx=10)

    self.btn_cancel = tk.Button(
      btn_frame, text="Cancel",
      font=("Segoe UI", 10),
      bg="#555", fg=FG,
      relief="flat", padx=15, pady=6,
      command=self.destroy,
    )
    self.btn_cancel.pack(side=tk.RIGHT)

  # ======================================================================
  # Scanning
  # ======================================================================

  def _scan_projects(self) -> list[_ProjectInfo]:
    """Scan for valid project directories."""
    found: list[_ProjectInfo] = []

    def _check_dir(d: Path):
      if not d.is_dir() or d.name.startswith((".", "__")):
        return
      has_calib = (d / "calibration").exists()
      has_rec = (d / "recordings").exists()
      if not has_calib and not has_rec and not d.name.startswith("project_"):
        return
      has_calib_videos = (
        (d / "calibration" / "intrinsic" / "port_1.mp4").exists()
        and (d / "calibration" / "intrinsic" / "port_2.mp4").exists()
      )
      calibrated = (d / "camera_array.toml").exists()
      num_rec = 0
      rec_dir = d / "recordings"
      if rec_dir.exists():
        num_rec = sum(1 for rd in rec_dir.iterdir()
                      if rd.is_dir() and rd.name.startswith("session_"))
      found.append(_ProjectInfo(
        name=d.name,
        path=d,
        has_calib_videos=has_calib_videos,
        calibrated=calibrated,
        num_recordings=num_rec,
      ))

    # Scan 1 level deep
    for item in sorted(self.projects_base.iterdir()):
      _check_dir(item)
      # Scan 2 levels deep (e.g. caliscope/caliscope_project)
      if item.is_dir() and not item.name.startswith((".", "__")):
        for sub in sorted(item.iterdir()):
          _check_dir(sub)

    return found

  def _scan_and_display(self):
    """Scan projects and update the listbox."""
    self._projects = self._scan_projects()
    self._listbox.delete(0, tk.END)

    for p in self._projects:
      status = ""
      if p.calibrated:
        status = " [calibrated]"
      elif p.has_calib_videos:
        status = " [has videos]"
      else:
        status = " [new]"
      self._listbox.insert(tk.END, f"  {p.name}{status}")

    if not self._projects:
      self._detail_name.config(text="No projects found.\nClick 'New Project' to create one.")

  # ======================================================================
  # Event handlers
  # ======================================================================

  def _on_select(self, event=None):
    """Handle project list selection."""
    sel = self._listbox.curselection()
    if not sel:
      return

    idx = sel[0]
    p = self._projects[idx]

    self.btn_open.config(state=tk.NORMAL)

    self._detail_name.config(text=p.name, fg=FG)

    if p.calibrated:
      self._detail_calib.config(text="Calibration: Complete", fg=ACCENT)
    elif p.has_calib_videos:
      self._detail_calib.config(text="Calibration: Videos ready", fg=ACCENT_ORANGE)
    else:
      self._detail_calib.config(text="Calibration: Not started", fg=FG_DIM)

    if p.calibrated:
      self._detail_status.config(text="Status: Ready to record & reconstruct", fg=ACCENT)
    elif p.has_calib_videos:
      self._detail_status.config(text="Status: Needs auto-calibration", fg=ACCENT_ORANGE)
    else:
      self._detail_status.config(text="Status: Needs calibration videos", fg=FG_DIM)

    self._detail_recordings.config(
      text=f"Recordings: {p.num_recordings}",
      fg=FG if p.num_recordings > 0 else FG_DIM,
    )
    self._detail_path.config(text=f"Path: {p.path}")

  def _on_new(self):
    """Create a new project."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"project_{ts}"
    project_dir = self.projects_base / project_name

    # Create scaffold
    (project_dir / "calibration" / "intrinsic").mkdir(parents=True, exist_ok=True)
    (project_dir / "calibration" / "extrinsic").mkdir(parents=True, exist_ok=True)
    (project_dir / "recordings").mkdir(parents=True, exist_ok=True)
    (project_dir / "raw_output").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created new project: {project_dir}")

    self.result = ProjectContext(
      project_dir=project_dir,
      project_name=project_name,
      is_new=True,
      needs_calibration=True,
    )
    self.destroy()

  def _on_open(self):
    """Open the selected project."""
    sel = self._listbox.curselection()
    if not sel:
      return

    p = self._projects[sel[0]]

    self.result = ProjectContext(
      project_dir=p.path,
      project_name=p.name,
      is_new=False,
      needs_calibration=not p.has_calib_videos,
    )
    self.destroy()
