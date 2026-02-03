"""Headless auto-calibration pipeline for caliscope.

Calls caliscope's core APIs directly (no GUI) to run the full
calibration workflow: intrinsic calibration, 2D extraction,
extrinsic calibration (bundle adjustment), and origin alignment.

Usage:
  from recorder.auto_calibrate import AutoCalibConfig, run_auto_calibration
  config = AutoCalibConfig(project_dir=Path("path/to/caliscope_project"))
  result = run_auto_calibration(config, on_progress=print)
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pandas as pd

from caliscope.cameras.camera_array import CameraArray, CameraData
from caliscope.core.calibrate_intrinsics import run_intrinsic_calibration
from caliscope.core.charuco import Charuco
from caliscope.core.point_data import ImagePoints
from caliscope.core.point_data_bundle import PointDataBundle
from caliscope.core.bootstrap_pose.build_paired_pose_network import build_paired_pose_network
from caliscope.persistence import (
  save_camera_array,
  save_charuco,
  save_image_points_csv,
)
from caliscope.recording.frame_source import FrameSource
from caliscope.recording.frame_sync import compute_sync_indices
from caliscope.repositories.point_data_bundle_repository import PointDataBundleRepository
from caliscope.trackers.charuco_tracker import CharucoTracker

logger = logging.getLogger(__name__)

# Type alias for progress callbacks
# (stage_name, message, percent_0_to_100)
ProgressCallback = Callable[[str, str, int], None]

FILTERED_FRACTION = 0.025  # 2.5% outlier removal


@dataclass
class AutoCalibConfig:
  """Configuration for headless auto-calibration pipeline."""
  project_dir: Path
  # Charuco board parameters (must match the physical board)
  charuco_columns: int = 10
  charuco_rows: int = 16
  board_height_cm: float = 80.0
  board_width_cm: float = 50.0
  dictionary: str = "DICT_4X4_100"
  aruco_scale: float = 0.7
  square_size_cm: float = 5.0
  units: str = "cm"
  # Camera setup
  ports: list[int] = field(default_factory=lambda: [1, 2])
  image_size: tuple[int, int] = (1600, 1200)
  # Processing parameters
  intrinsic_subsample: int = 10  # process every Nth frame for intrinsic
  extrinsic_subsample: int = 6   # process every Nth sync index for extrinsic

  @classmethod
  def from_yaml(cls, yaml_data: dict, project_dir: Path) -> "AutoCalibConfig":
    """Create config from YAML dictionary (default.yaml charuco section)."""
    charuco = yaml_data.get("charuco", {})
    split = yaml_data.get("split", {})
    # Derive image size from split config
    xsplit = split.get("xsplit", 1600)
    full_h = split.get("full_h", 1200)
    return cls(
      project_dir=project_dir,
      charuco_columns=charuco.get("columns", 10),
      charuco_rows=charuco.get("rows", 16),
      board_height_cm=charuco.get("board_height_cm", 80.0),
      board_width_cm=charuco.get("board_width_cm", 50.0),
      dictionary=charuco.get("dictionary", "DICT_4X4_100"),
      aruco_scale=charuco.get("aruco_scale", 0.7),
      square_size_cm=charuco.get("square_size_cm", 5.0),
      image_size=(xsplit, full_h),
    )


@dataclass
class CalibrationResult:
  """Result of the auto-calibration pipeline."""
  camera_array: CameraArray
  bundle: PointDataBundle
  origin_sync_index: int
  intrinsic_rmse: dict[int, float]  # port -> RMSE
  extrinsic_cost: float  # final_cost from bundle adjustment
  success: bool = True
  error_message: str = ""


def _emit(on_progress: Optional[ProgressCallback], stage: str, msg: str, pct: int):
  """Helper to safely emit progress."""
  if on_progress:
    on_progress(stage, msg, pct)
  logger.info(f"[{stage}] {msg} ({pct}%)")


def _collect_charuco_points_from_video(
  video_dir: Path,
  port: int,
  tracker: CharucoTracker,
  subsample: int = 10,
  on_progress: Optional[ProgressCallback] = None,
) -> list[tuple[int, object]]:
  """Iterate video frames and collect charuco corner detections.

  Uses sequential decoding (single pass) instead of per-frame seeking
  for ~10x speedup on H.264 video.

  Returns list of (frame_index, PointPacket) tuples.
  """
  video_path = video_dir / f"port_{port}.mp4"
  if not video_path.exists():
    raise FileNotFoundError(f"Video file not found: {video_path}")

  container = av.open(str(video_path))
  stream = container.streams.video[0]
  time_base = float(stream.time_base)
  fps = float(stream.average_rate)
  total_frames = stream.frames
  if total_frames == 0 and container.duration is not None:
    total_frames = int(container.duration / 1_000_000 * fps)

  collected = []
  sampled = 0

  try:
    for frame in container.decode(stream):
      if frame.pts is None:
        continue
      frame_idx = round(frame.pts * time_base * fps)

      # Only process every Nth frame
      if frame_idx % subsample != 0:
        continue

      img = frame.to_ndarray(format="bgr24")
      points = tracker.get_points(img, port, 0)
      sampled += 1

      if points is not None and len(points.point_id) > 0:
        collected.append((frame_idx, points))

      if on_progress and sampled % 20 == 0:
        pct = int(frame_idx / max(total_frames, 1) * 100)
        _emit(on_progress, "intrinsic", f"Port {port}: scanning frame {frame_idx}/{total_frames}", pct)
  finally:
    container.close()

  logger.info(f"Port {port}: collected {len(collected)} frames with charuco detections out of {sampled} sampled")
  return collected


def _build_image_points_from_packets(
  collected_points: list[tuple[int, object]],
  port: int,
) -> ImagePoints:
  """Convert (frame_index, PointPacket) list to ImagePoints DataFrame.

  Replicates the pattern from IntrinsicCalibrationPresenter._build_image_points().
  """
  rows = []
  for frame_index, points in collected_points:
    point_count = len(points.point_id)
    if point_count == 0:
      continue

    obj_loc_x, obj_loc_y, obj_loc_z = points.obj_loc_list

    for i in range(point_count):
      rows.append({
        "sync_index": frame_index,
        "port": port,
        "frame_index": frame_index,
        "frame_time": 0.0,
        "point_id": int(points.point_id[i]),
        "img_loc_x": float(points.img_loc[i, 0]),
        "img_loc_y": float(points.img_loc[i, 1]),
        "obj_loc_x": obj_loc_x[i],
        "obj_loc_y": obj_loc_y[i],
        "obj_loc_z": obj_loc_z[i],
      })

  if not rows:
    df = pd.DataFrame(columns=[
      "sync_index", "port", "frame_index", "frame_time",
      "point_id", "img_loc_x", "img_loc_y",
      "obj_loc_x", "obj_loc_y", "obj_loc_z",
    ])
    return ImagePoints(df)

  df = pd.DataFrame(rows)
  return ImagePoints(df)


def _decode_and_track_port(
  video_path: Path,
  port: int,
  needed_frames: set[int],
  charuco,
  rotation: int,
) -> dict[int, object]:
  """Decode one port's video sequentially and run charuco detection.

  Each thread gets its own CharucoTracker instance to avoid shared state.
  OpenCV and PyAV both release the GIL, so threads run in true parallelism.

  Returns:
    {frame_index: PointPacket} for all needed frames.
  """
  # Each thread gets its own tracker — CharucoTracker is lightweight
  tracker = CharucoTracker(charuco)

  results: dict[int, object] = {}
  needed = set(needed_frames)  # local copy

  container = av.open(str(video_path))
  stream = container.streams.video[0]
  time_base = float(stream.time_base)
  fps = float(stream.average_rate)

  for frame in container.decode(stream):
    if frame.pts is None:
      continue
    frame_idx = round(frame.pts * time_base * fps)

    if frame_idx in needed:
      img = frame.to_ndarray(format="bgr24")
      points = tracker.get_points(img, port, rotation)
      results[frame_idx] = points
      needed.discard(frame_idx)

      if not needed:
        break

  container.close()
  logger.info(f"Port {port}: tracked {len(results)} frames (parallel)")
  return results


def _process_extrinsic_sequential(
  recording_dir: Path,
  cameras: dict[int, CameraData],
  tracker: CharucoTracker,
  subsample: int = 3,
  on_progress: Optional[ProgressCallback] = None,
) -> ImagePoints:
  """Optimized extrinsic 2D extraction.

  Two key optimizations over process_synchronized_recording():
    1. Sequential decode (single pass) instead of per-frame H.264 seeking
    2. All ports processed in parallel threads (OpenCV releases GIL)
  """
  # Load sync map
  timestamps_csv = recording_dir / "frame_timestamps.csv"
  sync_map = compute_sync_indices(timestamps_csv)

  # Load frame times
  timestamps_df = pd.read_csv(timestamps_csv)
  frame_times: dict[tuple[int, int], float] = {}
  for port_key, group in timestamps_df.groupby("port"):
    sorted_group = group.sort_values("frame_time").reset_index(drop=True)
    for frame_index, row in sorted_group.iterrows():
      frame_times[(int(port_key), int(frame_index))] = float(row["frame_time"])

  # Determine which sync indices to process
  all_sync_indices = sorted(sync_map.keys())
  sync_indices_to_process = all_sync_indices[::subsample]
  total = len(sync_indices_to_process)

  logger.info(f"Parallel extrinsic: {total} sync indices (subsample={subsample}, total={len(all_sync_indices)})")

  # Pre-compute needed frame indices per port
  port_needed: dict[int, set[int]] = {port: set() for port in cameras}
  for sync_index in sync_indices_to_process:
    for port, frame_index in sync_map[sync_index].items():
      if frame_index is not None and port in cameras:
        port_needed[port].add(frame_index)

  # ── Parallel decode + track per port ──────────────────────────
  port_results: dict[tuple[int, int], object] = {}

  with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
    futures = {}
    for port in cameras:
      needed = port_needed.get(port, set())
      if not needed:
        continue
      video_path = recording_dir / f"port_{port}.mp4"
      if not video_path.exists():
        logger.warning(f"Video not found: {video_path}")
        continue

      futures[port] = pool.submit(
        _decode_and_track_port,
        video_path, port, needed,
        tracker.charuco,  # pass charuco config, thread creates own tracker
        cameras[port].rotation_count,
      )

    for port, future in futures.items():
      for frame_idx, points in future.result().items():
        port_results[(port, frame_idx)] = points

  # Assemble results in sync index order
  point_rows: list[dict] = []
  for i, sync_index in enumerate(sync_indices_to_process):
    for port, frame_index in sync_map[sync_index].items():
      if frame_index is None or port not in cameras:
        continue
      points = port_results.get((port, frame_index))
      if points is None or len(points.point_id) == 0:
        continue

      obj_loc_x, obj_loc_y, obj_loc_z = points.obj_loc_list
      frame_time = frame_times.get((port, frame_index), 0.0)

      for j in range(len(points.point_id)):
        point_rows.append({
          "sync_index": sync_index,
          "port": port,
          "frame_index": frame_index,
          "frame_time": frame_time,
          "point_id": int(points.point_id[j]),
          "img_loc_x": float(points.img_loc[j, 0]),
          "img_loc_y": float(points.img_loc[j, 1]),
          "obj_loc_x": obj_loc_x[j],
          "obj_loc_y": obj_loc_y[j],
          "obj_loc_z": obj_loc_z[j],
        })

    if on_progress and (i + 1) % 50 == 0:
      on_progress(i + 1, total)

  if on_progress:
    on_progress(total, total)

  if not point_rows:
    df = pd.DataFrame(columns=[
      "sync_index", "port", "frame_index", "frame_time",
      "point_id", "img_loc_x", "img_loc_y",
      "obj_loc_x", "obj_loc_y", "obj_loc_z",
    ])
    return ImagePoints(df)

  return ImagePoints(pd.DataFrame(point_rows))


def _generate_frame_timestamps(
  recording_dir: Path,
  ports: list[int],
) -> None:
  """Generate synthetic frame_timestamps.csv for perfectly synchronized stereo videos.

  Since both views come from a single stereo USB camera,
  frames are inherently synchronized. We create a CSV where
  both ports share identical timestamps based on video FPS.
  """
  rows = []
  for port in ports:
    source = FrameSource(recording_dir, port)
    fps = source.fps
    frame_count = source.frame_count
    source.close()

    for i in range(frame_count):
      rows.append({"port": port, "frame_time": i / fps})

  df = pd.DataFrame(rows)
  csv_path = recording_dir / "frame_timestamps.csv"
  df.to_csv(csv_path, index=False)
  logger.info(f"Generated frame_timestamps.csv at {csv_path} ({len(rows)} rows, {len(ports)} ports)")


def _find_best_origin_sync_index(bundle: PointDataBundle) -> int:
  """Find the sync_index with the most world points for origin alignment.

  Picks the frame where the charuco board is most visible across all cameras.
  """
  world_df = bundle.world_points.df
  counts = world_df.groupby("sync_index").size()
  best_sync_index = int(counts.idxmax())
  logger.info(f"Selected sync_index {best_sync_index} for origin alignment ({counts[best_sync_index]} world points)")
  return best_sync_index


def run_auto_calibration(
  config: AutoCalibConfig,
  on_progress: Optional[ProgressCallback] = None,
) -> CalibrationResult:
  """Execute complete auto-calibration pipeline.

  Steps:
    1. Create charuco board and tracker
    2. Intrinsic calibration for each camera
    3. Generate frame_timestamps.csv for extrinsic videos
    4. 2D extraction from extrinsic videos
    5. Bootstrap extrinsic poses (PnP)
    6. Bundle adjustment optimization
    7. Outlier filtering and re-optimization
    8. Origin alignment
    9. Save all results

  Args:
    config: Pipeline configuration
    on_progress: Optional callback for progress updates

  Returns:
    CalibrationResult with calibrated camera array and bundle
  """
  project = config.project_dir
  intrinsic_dir = project / "calibration" / "intrinsic"
  extrinsic_dir = project / "calibration" / "extrinsic"
  charuco_dir = extrinsic_dir / "CHARUCO"
  charuco_dir.mkdir(parents=True, exist_ok=True)

  intrinsic_rmse = {}

  try:
    # ── Step 1: Create charuco and tracker ──────────────────────────
    _emit(on_progress, "init", "Creating charuco board and tracker...", 0)

    charuco = Charuco(
      columns=config.charuco_columns,
      rows=config.charuco_rows,
      board_height=config.board_height_cm,
      board_width=config.board_width_cm,
      dictionary=config.dictionary,
      units=config.units,
      aruco_scale=config.aruco_scale,
      square_size_overide_cm=config.square_size_cm,
    )
    tracker = CharucoTracker(charuco)

    # Save charuco.toml
    save_charuco(charuco, project / "charuco.toml")
    _emit(on_progress, "init", "Charuco board configured", 5)

    # ── Step 2: Build initial camera array ──────────────────────────
    cameras = {}
    for port in config.ports:
      cameras[port] = CameraData(
        port=port,
        size=config.image_size,
        rotation_count=0,
      )
    camera_array = CameraArray(cameras)

    # ── Step 3: Intrinsic calibration (parallel collection) ────────
    total_ports = len(config.ports)

    # Verify all videos exist first
    for port in config.ports:
      video_path = intrinsic_dir / f"port_{port}.mp4"
      if not video_path.exists():
        raise FileNotFoundError(f"Intrinsic video not found: {video_path}")

    # Collect charuco points from ALL ports in parallel
    _emit(on_progress, "intrinsic",
          f"Scanning {total_ports} cameras in parallel...", 10)

    def _collect_for_port(port):
      """Thread worker: decode + detect charuco for one port."""
      # Each thread gets its own tracker (lightweight, just references charuco)
      t = CharucoTracker(charuco)
      video_path = intrinsic_dir / f"port_{port}.mp4"
      container = av.open(str(video_path))
      stream = container.streams.video[0]
      tb = float(stream.time_base)
      fps_val = float(stream.average_rate)
      collected = []
      sampled = 0
      try:
        for frame in container.decode(stream):
          if frame.pts is None:
            continue
          fidx = round(frame.pts * tb * fps_val)
          if fidx % config.intrinsic_subsample != 0:
            continue
          img = frame.to_ndarray(format="bgr24")
          pts = t.get_points(img, port, 0)
          sampled += 1
          if pts is not None and len(pts.point_id) > 0:
            collected.append((fidx, pts))
      finally:
        container.close()
      logger.info(f"Port {port}: collected {len(collected)} charuco frames out of {sampled} sampled")
      return port, collected

    with ThreadPoolExecutor(max_workers=total_ports) as pool:
      futures = [pool.submit(_collect_for_port, p) for p in config.ports]
      collected_by_port = {}
      for f in futures:
        port, collected = f.result()
        collected_by_port[port] = collected

    _emit(on_progress, "intrinsic", "Parallel scanning complete", 25)

    # Run calibration for each port (fast — uses only ~30 selected frames)
    for idx, port in enumerate(config.ports):
      collected = collected_by_port[port]
      if not collected:
        raise ValueError(f"No charuco corners detected in intrinsic video for port {port}")

      image_points = _build_image_points_from_packets(collected, port)
      camera = camera_array.cameras[port]
      output = run_intrinsic_calibration(camera, image_points)

      camera_array.cameras[port] = output.camera
      intrinsic_rmse[port] = output.report.rmse

      _emit(on_progress, "intrinsic",
            f"Camera {port} calibrated: RMSE={output.report.rmse:.3f}px, "
            f"frames={output.report.frames_used}",
            30 + (idx + 1) * 5)

    # Save intermediate camera array
    save_camera_array(camera_array, project / "camera_array.toml")
    _emit(on_progress, "intrinsic", "All intrinsic calibrations complete", 40)

    # ── Step 4: Generate frame_timestamps.csv ───────────────────────
    _emit(on_progress, "extrinsic_2d", "Generating synchronization timestamps...", 42)
    _generate_frame_timestamps(extrinsic_dir, config.ports)

    # ── Step 5: 2D extraction from extrinsic videos (sequential) ────
    _emit(on_progress, "extrinsic_2d", "Extracting 2D charuco points (sequential decode)...", 45)

    image_points = _process_extrinsic_sequential(
      recording_dir=extrinsic_dir,
      cameras=camera_array.cameras,
      tracker=tracker,
      subsample=config.extrinsic_subsample,
      on_progress=lambda cur, total: _emit(
        on_progress, "extrinsic_2d",
        f"Processing sync index {cur}/{total}",
        45 + int(cur / total * 15),
      ),
    )

    # Save image points
    save_image_points_csv(image_points, charuco_dir / "image_points.csv")
    _emit(on_progress, "extrinsic_2d", f"2D extraction complete: {len(image_points.df)} observations", 60)

    # ── Step 6: Bootstrap extrinsic poses ───────────────────────────
    _emit(on_progress, "extrinsic_3d", "Bootstrapping camera poses (PnP)...", 62)

    pose_network = build_paired_pose_network(image_points, camera_array, method="pnp")
    pose_network.apply_to(camera_array)

    # ── Step 7: Triangulate initial 3D points ───────────────────────
    _emit(on_progress, "extrinsic_3d", "Triangulating 3D points...", 65)

    world_points = image_points.triangulate(camera_array)

    # ── Step 8: Bundle adjustment ───────────────────────────────────
    _emit(on_progress, "extrinsic_3d", "Running bundle adjustment (first pass)...", 70)

    bundle = PointDataBundle(camera_array, image_points, world_points)
    optimized = bundle.optimize(ftol=1e-8, verbose=0)

    initial_cost = optimized.optimization_status.final_cost if optimized.optimization_status else 0.0
    _emit(on_progress, "extrinsic_3d",
          f"First pass cost: {initial_cost:.4f}", 75)

    # ── Step 9: Filter outliers and re-optimize ─────────────────────
    _emit(on_progress, "extrinsic_3d", "Filtering outliers and re-optimizing...", 78)

    filtered = optimized.filter_by_percentile_error(FILTERED_FRACTION * 100)
    final_bundle = filtered.optimize(ftol=1e-8, verbose=0)

    final_cost = final_bundle.optimization_status.final_cost if final_bundle.optimization_status else 0.0
    _emit(on_progress, "extrinsic_3d",
          f"Final cost: {final_cost:.4f}", 85)

    # ── Step 10: Origin alignment ───────────────────────────────────
    _emit(on_progress, "extrinsic_3d", "Aligning coordinate origin...", 88)

    origin_sync_index = _find_best_origin_sync_index(final_bundle)
    aligned_bundle = final_bundle.align_to_object(origin_sync_index)

    # ── Step 11: Save everything ────────────────────────────────────
    _emit(on_progress, "save", "Saving calibration results...", 92)

    # Save final camera array to project root
    save_camera_array(aligned_bundle.camera_array, project / "camera_array.toml")

    # Save bundle to extrinsic/CHARUCO/
    bundle_repo = PointDataBundleRepository(charuco_dir)
    bundle_repo.save(aligned_bundle)

    # Save project settings
    _save_project_settings(project)

    _emit(on_progress, "done", "Calibration complete!", 100)

    return CalibrationResult(
      camera_array=aligned_bundle.camera_array,
      bundle=aligned_bundle,
      origin_sync_index=origin_sync_index,
      intrinsic_rmse=intrinsic_rmse,
      extrinsic_cost=final_cost,
      success=True,
    )

  except Exception as e:
    logger.error(f"Auto-calibration failed: {e}", exc_info=True)
    _emit(on_progress, "error", f"Calibration failed: {e}", -1)
    return CalibrationResult(
      camera_array=CameraArray({}),
      bundle=None,
      origin_sync_index=-1,
      intrinsic_rmse=intrinsic_rmse,
      extrinsic_cost=0.0,
      success=False,
      error_message=str(e),
    )


def _save_project_settings(project_dir: Path):
  """Save minimal project settings for caliscope compatibility."""
  import rtoml
  from datetime import datetime

  settings_path = project_dir / "project_settings.toml"
  settings = {
    "creation_date": datetime.now().isoformat(),
    "save_tracked_points_video": True,
    "fps_sync_stream_processing": 100,
  }

  # Only create if doesn't exist (don't overwrite user settings)
  if not settings_path.exists():
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
      rtoml.dump(settings, f)
    logger.info(f"Created project settings at {settings_path}")


def run_auto_reconstruction(
  project_dir: Path,
  recording_name: str,
  tracker_name: str = "HOLISTIC",
  on_progress: Optional[ProgressCallback] = None,
) -> Path:
  """Run 3D reconstruction on a recording session.

  Args:
    project_dir: Path to caliscope project directory
    recording_name: Name of recording subdirectory under recordings/
    tracker_name: Tracker to use (HOLISTIC, POSE, HAND, etc.)
    on_progress: Optional callback for progress updates

  Returns:
    Path to output xyz CSV file
  """
  from caliscope.persistence import load_camera_array
  from caliscope.reconstruction.reconstructor import Reconstructor
  from caliscope.trackers.tracker_enum import TrackerEnum

  _emit(on_progress, "reconstruction", "Loading calibration data...", 0)

  camera_array = load_camera_array(project_dir / "camera_array.toml")
  recording_path = project_dir / "recordings" / recording_name
  tracker_enum = TrackerEnum[tracker_name]

  if not recording_path.exists():
    raise FileNotFoundError(f"Recording not found: {recording_path}")

  _emit(on_progress, "reconstruction", "Starting 2D landmark detection...", 5)

  reconstructor = Reconstructor(camera_array, recording_path, tracker_enum)

  # Stage 1: 2D detection
  completed = reconstructor.create_xy(include_video=False)
  if not completed:
    raise RuntimeError("2D landmark detection was cancelled or failed")

  _emit(on_progress, "reconstruction", "Starting 3D triangulation...", 80)

  # Stage 2: 3D triangulation
  reconstructor.create_xyz()

  output_path = recording_path / tracker_enum.name / f"xyz_{tracker_enum.name}.csv"
  _emit(on_progress, "reconstruction", f"Reconstruction complete: {output_path}", 100)

  return output_path
