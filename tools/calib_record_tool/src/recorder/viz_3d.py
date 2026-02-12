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

def _smooth_frames_ema(
  frames: dict[int, dict[int, tuple[float, float, float]]],
  frame_indices: list[int],
  alpha: float = 0.25,
) -> None:
  """Bidirectional EMA smoothing per landmark (in-place, no lag).

  Forward pass followed by backward pass, then averaged.  This eliminates
  the lag that single-pass EMA introduces while providing strong smoothing.
  Gaps (frames where a landmark is missing) reset the filter.
  """
  all_pids: set[int] = set()
  for pts in frames.values():
    all_pids.update(pts.keys())

  for pid in all_pids:
    # Collect indices where this landmark exists
    present = [(i, si) for i, si in enumerate(frame_indices) if pid in frames[si]]
    if len(present) < 3:
      continue

    # Extract raw values
    raw_vals = [frames[si][pid] for _, si in present]

    # Forward pass
    fwd = [raw_vals[0]]
    for k in range(1, len(raw_vals)):
      p = fwd[-1]
      r = raw_vals[k]
      fwd.append((
        alpha * r[0] + (1 - alpha) * p[0],
        alpha * r[1] + (1 - alpha) * p[1],
        alpha * r[2] + (1 - alpha) * p[2],
      ))

    # Backward pass
    bwd = [None] * len(raw_vals)
    bwd[-1] = raw_vals[-1]
    for k in range(len(raw_vals) - 2, -1, -1):
      p = bwd[k + 1]
      r = raw_vals[k]
      bwd[k] = (
        alpha * r[0] + (1 - alpha) * p[0],
        alpha * r[1] + (1 - alpha) * p[1],
        alpha * r[2] + (1 - alpha) * p[2],
      )

    # Average forward + backward
    for k, (_, si) in enumerate(present):
      frames[si][pid] = (
        (fwd[k][0] + bwd[k][0]) * 0.5,
        (fwd[k][1] + bwd[k][1]) * 0.5,
        (fwd[k][2] + bwd[k][2]) * 0.5,
      )


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

  # Drop rows with NaN / inf coordinates only (no IQR — auto_calibrate already filtered)
  df = df.dropna(subset=["x_coord", "y_coord", "z_coord"])
  for col in ("x_coord", "y_coord", "z_coord"):
    df = df[np.isfinite(df[col])]

  # Build frame dictionary
  frames: dict[int, dict[int, tuple[float, float, float]]] = {}
  for sync_idx, group in df.groupby("sync_index"):
    points: dict[int, tuple[float, float, float]] = {}
    for _, row in group.iterrows():
      pid = int(row["point_id"])
      points[pid] = (float(row["x_coord"]), float(row["y_coord"]), float(row["z_coord"]))
    frames[int(sync_idx)] = points

  frame_indices = sorted(frames.keys())

  # Axis limits: use full min/max of all data points so nothing is clipped.
  # Outliers are already filtered by auto_calibrate's IQR pass.
  all_x = df["x_coord"].values
  all_y = df["y_coord"].values
  all_z = df["z_coord"].values

  x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
  y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
  z_min, z_max = float(np.min(all_z)), float(np.max(all_z))

  # Add 20% padding so skeleton has breathing room at edges
  pad_x = max((x_max - x_min) * 0.20, 0.30)
  pad_y = max((y_max - y_min) * 0.20, 0.30)
  pad_z = max((z_max - z_min) * 0.20, 0.30)

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

  Checks our bundled configs first, then caliscope's wireframe TOML files.
  Falls back to empty list (points-only) if no wireframe is defined.
  """
  # Check our bundled configs first (for custom trackers like YOLOV8_POSE)
  bundled = Path(__file__).resolve().parent.parent.parent / "configs" / f"{tracker_name.lower()}_wireframe.toml"
  toml_path = None
  if bundled.exists():
    toml_path = bundled
  else:
    try:
      import caliscope
      caliscope_root = Path(caliscope.__file__).parent
      candidate = (
        caliscope_root / "gui" / "geometry" / "wireframes"
        / f"{tracker_name.lower()}_wireframe.toml"
      )
      if candidate.exists():
        toml_path = candidate
    except ImportError:
      pass

  if toml_path is None:
    logger.info(f"No wireframe TOML for tracker '{tracker_name}', trying built-in fallback")
    return _builtin_wireframe(tracker_name)

  segments = _parse_wireframe_toml(toml_path)
  if not segments:
    logger.info("TOML parse returned no segments, trying built-in fallback")
    return _builtin_wireframe(tracker_name)
  return segments


def _builtin_wireframe(tracker_name: str) -> list[WireSegment]:
  """Return hardcoded wireframe segments for known trackers."""
  if tracker_name.upper() != "YOLOV8_POSE":
    return []

  C = (0.2, 0.9, 0.9)  # cyan  — face
  Y = (0.9, 0.9, 0.2)  # yellow — torso
  R = (1.0, 0.2, 0.2)  # red   — left side
  B = (0.2, 0.4, 1.0)  # blue  — right side

  return [
    # Face
    WireSegment("nose_left_eye",        0,  1, C),
    WireSegment("nose_right_eye",       0,  2, C),
    WireSegment("left_eye_left_ear",    1,  3, C),
    WireSegment("right_eye_right_ear",  2,  4, C),
    # Shoulders
    WireSegment("shoulders",            5,  6, Y),
    # Left arm
    WireSegment("left_upper_arm",       5,  7, R),
    WireSegment("left_forearm",         7,  9, R),
    # Right arm
    WireSegment("right_upper_arm",      6,  8, B),
    WireSegment("right_forearm",        8, 10, B),
    # Torso
    WireSegment("left_torso",           5, 11, Y),
    WireSegment("right_torso",          6, 12, Y),
    WireSegment("hips",                11, 12, Y),
    # Left leg
    WireSegment("left_thigh",          11, 13, R),
    WireSegment("left_shin",           13, 15, R),
    # Right leg
    WireSegment("right_thigh",         12, 14, B),
    WireSegment("right_shin",          14, 16, B),
  ]


def _parse_wireframe_toml(toml_path: Path) -> list[WireSegment]:
  """Parse a wireframe TOML file into WireSegment list."""
  try:
    import rtoml
    with open(toml_path) as f:
      data = rtoml.load(f)
  except ImportError:
    try:
      import tomllib
      with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    except ImportError:
      logger.warning("No TOML parser available (rtoml or tomllib)")
      return []

  points_map = data.get("points", {})
  name_to_id = {name: pid for name, pid in points_map.items()}

  color_map = {
    "r": (1.0, 0.2, 0.2), "g": (0.2, 1.0, 0.2), "b": (0.2, 0.4, 1.0),
    "c": (0.2, 0.9, 0.9), "m": (0.9, 0.2, 0.9), "y": (0.9, 0.9, 0.2),
    "k": (0.3, 0.3, 0.3), "w": (1.0, 1.0, 1.0),
  }

  segments = []
  for section, content in data.items():
    if section == "points" or not isinstance(content, dict):
      continue
    seg_points = content.get("points", [])
    color_key = content.get("color", "w")
    if len(seg_points) == 2 and seg_points[0] in name_to_id and seg_points[1] in name_to_id:
      segments.append(WireSegment(
        name=section,
        point_a_id=name_to_id[seg_points[0]],
        point_b_id=name_to_id[seg_points[1]],
        color_rgb=color_map.get(color_key, (1.0, 1.0, 1.0)),
      ))

  logger.info(f"Loaded {len(segments)} wireframe segments for '{toml_path.stem}'")
  return segments


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
  """Set consistent axis limits with proportional aspect ratio."""
  lim = viz_data.axis_limits
  ax.set_xlim(lim[0], lim[1])
  ax.set_ylim(lim[2], lim[3])
  ax.set_zlim(lim[4], lim[5])

  # Proportional box: match real-world room shape instead of forced cube
  x_range = max(lim[1] - lim[0], 0.01)
  y_range = max(lim[3] - lim[2], 0.01)
  z_range = max(lim[5] - lim[4], 0.01)
  ax.set_box_aspect([x_range, y_range, z_range])
