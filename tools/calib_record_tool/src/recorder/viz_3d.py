"""
3D Visualization module for reconstructed point data.

Loads xyz CSV output from caliscope reconstruction, optionally loads
wireframe skeleton definitions, and renders animated 3D frames using
matplotlib (embedded in Tkinter via FigureCanvasTkAgg).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class WireSegment:
  """A single wireframe connection between two point IDs."""
  name: str
  point_a_id: int
  point_b_id: int
  color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Viz3DData:
  """Container for all data needed to render 3D playback."""

  # {sync_index: {point_id: (x, y, z)}}
  frames: dict[int, dict[int, tuple[float, float, float]]]
  # Sorted list of sync indices for sequential playback
  frame_indices: list[int]
  # Wireframe segments (empty list = points only)
  segments: list[WireSegment] = field(default_factory=list)
  # Axis limits: (x_min, x_max, y_min, y_max, z_min, z_max)
  axis_limits: tuple[float, float, float, float, float, float] = (
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
  )
  # Total frame count
  num_frames: int = 0


# ============================================================================
# Loading functions
# ============================================================================

def load_xyz_csv(csv_path: Path) -> Viz3DData:
  """
  Load a caliscope xyz CSV file and parse into frame-indexed data.

  CSV columns: sync_index, point_id, x_coord, y_coord, z_coord, frame_time
  """
  logger.info(f"Loading xyz CSV: {csv_path}")
  df = pd.read_csv(csv_path)

  required_cols = {"sync_index", "point_id", "x_coord", "y_coord", "z_coord"}
  missing = required_cols - set(df.columns)
  if missing:
    raise ValueError(f"CSV missing columns: {missing}")

  # Drop rows with NaN / inf coordinates
  df = df.dropna(subset=["x_coord", "y_coord", "z_coord"])
  for col in ("x_coord", "y_coord", "z_coord"):
    df = df[np.isfinite(df[col])]

  # Robust outlier removal (IQR-based) to prevent axis blowup
  coord_cols = ["x_coord", "y_coord", "z_coord"]
  n_raw = len(df)
  for col in coord_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 2.0 * iqr
    upper = q3 + 2.0 * iqr
    df = df[(df[col] >= lower) & (df[col] <= upper)]
  if len(df) < n_raw:
    logger.info(f"Viz outlier filter: {n_raw} -> {len(df)} points "
                f"(removed {n_raw - len(df)})")

  # Build frame dictionary
  frames: dict[int, dict[int, tuple[float, float, float]]] = {}
  for sync_idx, group in df.groupby("sync_index"):
    points: dict[int, tuple[float, float, float]] = {}
    for _, row in group.iterrows():
      pid = int(row["point_id"])
      points[pid] = (float(row["x_coord"]), float(row["y_coord"]), float(row["z_coord"]))
    frames[int(sync_idx)] = points

  frame_indices = sorted(frames.keys())

  # Axis limits: use full min/max of already-filtered data (IQR removed outliers)
  all_x = df["x_coord"].values
  all_y = df["y_coord"].values
  all_z = df["z_coord"].values

  x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
  y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
  z_min, z_max = float(np.min(all_z)), float(np.max(all_z))

  # Add 20% padding so the full body is comfortably visible
  pad_x = max((x_max - x_min) * 0.20, 0.10)
  pad_y = max((y_max - y_min) * 0.20, 0.10)
  pad_z = max((z_max - z_min) * 0.20, 0.10)

  axis_limits = (
    x_min - pad_x, x_max + pad_x,
    y_min - pad_y, y_max + pad_y,
    z_min - pad_z, z_max + pad_z,
  )

  logger.info(f"Loaded {len(frame_indices)} frames, "
              f"{len(df)} total points, "
              f"axis range: x=[{x_min:.2f},{x_max:.2f}] "
              f"y=[{y_min:.2f},{y_max:.2f}] z=[{z_min:.2f},{z_max:.2f}]")

  return Viz3DData(
    frames=frames,
    frame_indices=frame_indices,
    axis_limits=axis_limits,
    num_frames=len(frame_indices),
  )


def load_wireframe_for_tracker(tracker_name: str) -> list[WireSegment]:
  """
  Load wireframe segments for a given tracker type.

  Uses caliscope's wireframe TOML files. Falls back to empty list
  (points-only) if no wireframe is defined for the tracker.
  """
  try:
    from caliscope.gui.geometry.wireframe import load_wireframe_config
    import caliscope
    caliscope_root = Path(caliscope.__file__).parent
    toml_path = (
      caliscope_root / "gui" / "geometry" / "wireframes"
      / f"{tracker_name.lower()}_wireframe.toml"
    )

    if not toml_path.exists():
      logger.info(f"No wireframe TOML for tracker '{tracker_name}', using points only")
      return []

    config = load_wireframe_config(toml_path)
    segments = [
      WireSegment(
        name=seg.name,
        point_a_id=seg.point_a_id,
        point_b_id=seg.point_b_id,
        color_rgb=seg.color_rgb,
      )
      for seg in config.segments
    ]
    logger.info(f"Loaded {len(segments)} wireframe segments for '{tracker_name}'")
    return segments

  except ImportError:
    logger.warning("caliscope not available, cannot load wireframe")
    return []
  except Exception as e:
    logger.warning(f"Failed to load wireframe for '{tracker_name}': {e}")
    return []


# ============================================================================
# Rendering
# ============================================================================

def render_frame(ax, viz_data: Viz3DData, frame_idx: int):
  """
  Render a single frame onto a matplotlib 3D Axes.

  Args:
      ax: matplotlib Axes3D instance
      viz_data: loaded visualization data
      frame_idx: index into viz_data.frame_indices (0-based position)
  """
  ax.cla()

  if frame_idx < 0 or frame_idx >= viz_data.num_frames:
    return

  sync_idx = viz_data.frame_indices[frame_idx]
  points = viz_data.frames.get(sync_idx, {})

  if not points:
    _apply_axis_limits(ax, viz_data)
    return

  # Extract coordinates
  pids = list(points.keys())
  coords = np.array([points[pid] for pid in pids])
  xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

  # Draw points (size scales with axis range for consistent visibility)
  ax.scatter(xs, ys, zs, c="#00BFFF", s=30, alpha=0.8, depthshade=True)

  # Draw wireframe segments
  if viz_data.segments:
    point_lookup = points
    for seg in viz_data.segments:
      if seg.point_a_id in point_lookup and seg.point_b_id in point_lookup:
        pa = point_lookup[seg.point_a_id]
        pb = point_lookup[seg.point_b_id]
        ax.plot(
          [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
          color=seg.color_rgb, linewidth=2.0, alpha=0.9,
        )

  _apply_axis_limits(ax, viz_data)

  # Dark theme styling
  ax.set_facecolor("#1a1a1a")
  ax.set_xlabel("X", fontsize=7, color="#888888")
  ax.set_ylabel("Y", fontsize=7, color="#888888")
  ax.set_zlabel("Z", fontsize=7, color="#888888")
  ax.tick_params(labelsize=6, colors="#666666")
  for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_facecolor("#222222")
    pane.set_edgecolor("#444444")
  ax.grid(True, alpha=0.2)


def _apply_axis_limits(ax, viz_data: Viz3DData):
  """Set consistent axis limits."""
  lim = viz_data.axis_limits
  ax.set_xlim(lim[0], lim[1])
  ax.set_ylim(lim[2], lim[3])
  ax.set_zlim(lim[4], lim[5])
