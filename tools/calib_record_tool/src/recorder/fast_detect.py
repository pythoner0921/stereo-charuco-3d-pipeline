"""Fast batch ONNX 2D detection bypassing caliscope's frame-by-frame streaming.

Reads frames from stereo port videos using pyav sequential decode,
batches them, and runs YOLO ONNX inference for significant speedup.
Writes xy_YOLOV8_POSE.csv in caliscope ImagePoints format.
"""
from __future__ import annotations

import logging
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .yolov8_pose_tracker import (
  MIN_CONFIDENCE,
  _ensure_onnx_model,
)

logger = logging.getLogger(__name__)


def fast_create_xy(
  recording_path: Path,
  camera_array,
  model_size: str = "n",
  imgsz: int = 480,
  fps_target: int = 10,
  batch_size: int = 1,
) -> Path:
  """Run batch ONNX 2D detection on stereo recording videos.

  Produces xy_YOLOV8_POSE.csv in caliscope ImagePoints format,
  one file per port under recording_path/YOLOV8_POSE/.

  Args:
    recording_path: Directory containing port_1.{mp4,avi} and port_2.{mp4,avi}
    camera_array: CameraArray with camera metadata (ports, rotation_count)
    model_size: YOLOv8 model size variant (n/s/m)
    imgsz: Inference resolution in pixels
    fps_target: Target FPS for subsampling (lower = fewer frames = faster)
    batch_size: Number of frames per batch for ONNX inference

  Returns:
    Path to the output directory (recording_path/YOLOV8_POSE/)
  """
  from ultralytics import YOLO

  # Prepare ONNX model
  try:
    onnx_path = _ensure_onnx_model(model_size, imgsz)
    model = YOLO(str(onnx_path), task="pose")
    logger.info(f"Fast detect: loaded ONNX model {onnx_path}")
  except Exception as e:
    logger.warning(f"ONNX failed, falling back to PyTorch: {e}")
    model = YOLO(f"yolov8{model_size}-pose.pt")

  output_dir = recording_path / "YOLOV8_POSE"
  output_dir.mkdir(parents=True, exist_ok=True)

  all_rows: list[dict] = []

  for port, cam_data in camera_array.cameras.items():
    video_path = _find_video(recording_path, port)
    if video_path is None:
      logger.warning(f"No video found for port {port} in {recording_path}")
      continue

    logger.info(f"Fast detect port {port}: {video_path}")
    rows = _process_port(
      model, video_path, port, cam_data.rotation_count,
      imgsz, fps_target, batch_size,
    )
    all_rows.extend(rows)
    logger.info(f"Port {port}: detected {len(rows)} keypoint observations")

  if not all_rows:
    logger.warning("No keypoints detected in any port")
    df = pd.DataFrame(columns=[
      "sync_index", "port", "frame_index", "frame_time",
      "point_id", "img_loc_x", "img_loc_y",
    ])
  else:
    df = pd.DataFrame(all_rows)

  csv_path = output_dir / "xy_YOLOV8_POSE.csv"
  df.to_csv(csv_path, index=False)
  logger.info(f"Wrote {len(df)} rows to {csv_path}")

  return output_dir


def _find_video(recording_path: Path, port: int) -> Path | None:
  """Find video file for a port, preferring .avi then .mp4."""
  for ext in (".avi", ".mp4"):
    p = recording_path / f"port_{port}{ext}"
    if p.exists():
      return p
  return None


def _process_port(
  model,
  video_path: Path,
  port: int,
  rotation_count: int,
  imgsz: int,
  fps_target: int,
  batch_size: int,
) -> list[dict]:
  """Decode video, subsample frames, run batch inference, collect results."""
  from caliscope.trackers.helper import apply_rotation, unrotate_points

  container = av.open(str(video_path))
  stream = container.streams.video[0]
  time_base = float(stream.time_base)
  video_fps = float(stream.average_rate)
  total_frames = stream.frames
  if total_frames == 0 and container.duration is not None:
    total_frames = int(container.duration / 1_000_000 * video_fps)

  # Subsample interval
  step = max(1, round(video_fps / fps_target))

  # Collect frames
  frame_batch: list[np.ndarray] = []
  frame_meta: list[tuple[int, float, int, int]] = []  # (frame_idx, frame_time, w, h)

  rows: list[dict] = []

  try:
    for av_frame in container.decode(stream):
      if av_frame.pts is None:
        continue
      frame_idx = round(av_frame.pts * time_base * video_fps)

      if frame_idx % step != 0:
        continue

      frame_time = av_frame.pts * time_base
      img = av_frame.to_ndarray(format="bgr24")

      # Apply rotation (caliscope convention)
      rotated = apply_rotation(img, rotation_count)
      h, w = rotated.shape[:2]

      frame_batch.append(rotated)
      frame_meta.append((frame_idx, frame_time, w, h))

      # Run batch when full
      if len(frame_batch) >= batch_size:
        batch_rows = _run_batch(
          model, frame_batch, frame_meta, port, rotation_count, imgsz,
        )
        rows.extend(batch_rows)
        frame_batch.clear()
        frame_meta.clear()

    # Process remaining frames
    if frame_batch:
      batch_rows = _run_batch(
        model, frame_batch, frame_meta, port, rotation_count, imgsz,
      )
      rows.extend(batch_rows)

  finally:
    container.close()

  # Assign sync_index = sequential frame order (stereo is inherently synced)
  sync_indices = sorted(set(r["frame_index"] for r in rows))
  sync_map = {fi: si for si, fi in enumerate(sync_indices)}
  for r in rows:
    r["sync_index"] = sync_map[r["frame_index"]]

  return rows


def _run_batch(
  model,
  frames: list[np.ndarray],
  meta: list[tuple[int, float, int, int]],
  port: int,
  rotation_count: int,
  imgsz: int,
) -> list[dict]:
  """Run YOLO inference on a batch of frames and extract keypoints."""
  from caliscope.trackers.helper import unrotate_points

  results_list = model(frames, verbose=False, imgsz=imgsz)

  rows: list[dict] = []

  for i, result in enumerate(results_list):
    frame_idx, frame_time, frame_w, frame_h = meta[i]

    if result.keypoints is None or result.keypoints.xy is None:
      continue

    raw_xy = result.keypoints.xy
    if hasattr(raw_xy, 'cpu'):
      kpts_xy = raw_xy.cpu().numpy()
    else:
      kpts_xy = np.asarray(raw_xy)

    raw_conf = result.keypoints.conf
    if raw_conf is not None:
      if hasattr(raw_conf, 'cpu'):
        kpts_conf = raw_conf.cpu().numpy()
      else:
        kpts_conf = np.asarray(raw_conf)
    else:
      kpts_conf = None

    if kpts_xy.shape[0] == 0:
      continue

    # Pick person with most high-confidence keypoints
    if kpts_conf is not None and kpts_xy.shape[0] > 1:
      best_idx = int(np.argmax((kpts_conf > MIN_CONFIDENCE).sum(axis=1)))
    else:
      best_idx = 0

    xy = kpts_xy[best_idx]  # (17, 2)
    conf = kpts_conf[best_idx] if kpts_conf is not None else np.ones(17)

    mask = conf >= MIN_CONFIDENCE
    point_ids = np.where(mask)[0]
    img_loc = xy[mask].astype(np.float64)

    if len(point_ids) == 0:
      continue

    # Undo rotation
    img_loc = unrotate_points(img_loc, rotation_count, frame_w, frame_h)

    for j, pid in enumerate(point_ids):
      rows.append({
        "sync_index": 0,  # filled later
        "port": port,
        "frame_index": frame_idx,
        "frame_time": frame_time,
        "point_id": int(pid),
        "img_loc_x": float(img_loc[j, 0]),
        "img_loc_y": float(img_loc[j, 1]),
      })

  return rows
