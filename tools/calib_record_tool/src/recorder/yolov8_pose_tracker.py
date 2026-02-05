"""YOLOv8-Pose tracker for caliscope-compatible 3D reconstruction.

Uses ultralytics YOLOv8-Pose to detect 17 COCO keypoints per person.
Implements caliscope's Tracker ABC so it can be used with the existing
Reconstructor pipeline via a duck-type enum wrapper.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from threading import Thread
from queue import Queue, Full

import numpy as np

from caliscope.packets import PointPacket
from caliscope.tracker import Tracker
from caliscope.trackers.helper import apply_rotation, unrotate_points

logger = logging.getLogger(__name__)

# COCO 17 keypoint names (index = point_id)
COCO_POINT_NAMES = [
  "nose",            # 0
  "left_eye",        # 1
  "right_eye",       # 2
  "left_ear",        # 3
  "right_ear",       # 4
  "left_shoulder",   # 5
  "right_shoulder",  # 6
  "left_elbow",      # 7
  "right_elbow",     # 8
  "left_wrist",      # 9
  "right_wrist",     # 10
  "left_hip",        # 11
  "right_hip",       # 12
  "left_knee",       # 13
  "right_knee",      # 14
  "left_ankle",      # 15
  "right_ankle",     # 16
]

# Minimum keypoint confidence to include in output
MIN_CONFIDENCE = 0.3


class YoloV8PoseTracker(Tracker):
  """YOLOv8-Pose tracker producing 17 COCO keypoints.

  Each camera port gets its own background thread with a dedicated
  model instance, matching the pattern used by caliscope's PoseTracker.
  """

  def __init__(self, model_size: str = "n", imgsz: int = 480) -> None:
    self.model_size = model_size
    self.imgsz = imgsz
    self.in_queues: dict[int, Queue] = {}
    self.out_queues: dict[int, Queue] = {}
    self.threads: dict[int, Thread] = {}

  # ── Tracker ABC ──────────────────────────────────────────────

  @property
  def name(self) -> str:
    return "YOLOV8_POSE"

  def get_points(
    self, frame: np.ndarray, port: int = 0, rotation_count: int = 0,
  ) -> PointPacket:
    if port not in self.in_queues:
      self._start_worker(port, rotation_count)

    self.in_queues[port].put(frame)
    return self.out_queues[port].get()

  def get_point_name(self, point_id: int) -> str:
    if 0 <= point_id < len(COCO_POINT_NAMES):
      return COCO_POINT_NAMES[point_id]
    return f"point_{point_id}"

  def scatter_draw_instructions(self, point_id: int) -> dict:
    name = self.get_point_name(point_id)
    if name.startswith("left"):
      return {"radius": 5, "color": (0, 0, 220), "thickness": 3}
    elif name.startswith("right"):
      return {"radius": 5, "color": (220, 0, 0), "thickness": 3}
    return {"radius": 5, "color": (220, 0, 220), "thickness": 3}

  def get_connected_points(self) -> set[tuple[int, int]]:
    return {
      (0, 1), (0, 2),       # nose → eyes
      (1, 3), (2, 4),       # eyes → ears
      (5, 6),               # shoulders
      (5, 7), (7, 9),       # left arm
      (6, 8), (8, 10),      # right arm
      (5, 11), (6, 12),     # torso sides
      (11, 12),             # hips
      (11, 13), (13, 15),   # left leg
      (12, 14), (14, 16),   # right leg
    }

  def cleanup(self) -> None:
    for port, q in self.in_queues.items():
      try:
        q.put(None, timeout=1.0)
      except Full:
        logger.warning(f"Timeout sending shutdown to port {port}")
    for port, t in self.threads.items():
      t.join(timeout=3.0)
      if t.is_alive():
        logger.warning(f"Worker thread for port {port} did not exit")
    self.in_queues.clear()
    self.out_queues.clear()
    self.threads.clear()
    gc.collect()

  # ── Internal ─────────────────────────────────────────────────

  def _start_worker(self, port: int, rotation_count: int) -> None:
    self.in_queues[port] = Queue(1)
    self.out_queues[port] = Queue(1)
    t = Thread(
      target=self._worker_loop,
      args=(port, rotation_count),
      daemon=True,
      name=f"YoloV8Pose_Port{port}",
    )
    t.start()
    self.threads[port] = t

  def _worker_loop(self, port: int, rotation_count: int) -> None:
    from ultralytics import YOLO

    model = YOLO(f"yolov8{self.model_size}-pose.pt")
    logger.info(f"YOLOv8{self.model_size}-Pose model loaded for port {port} (imgsz={self.imgsz})")

    in_q = self.in_queues[port]
    out_q = self.out_queues[port]

    while True:
      frame = in_q.get()
      if frame is None:
        break

      rotated = apply_rotation(frame, rotation_count)
      h, w = rotated.shape[:2]

      packet = self._detect(model, rotated, rotation_count, w, h)
      out_q.put(packet)

    del model
    gc.collect()
    logger.info(f"YOLOv8-Pose worker for port {port} shut down")

  def _detect(
    self, model, frame: np.ndarray, rotation_count: int, frame_w: int, frame_h: int,
  ) -> PointPacket:
    results = model(frame, verbose=False, imgsz=self.imgsz)

    if not results or len(results) == 0:
      return PointPacket(np.array([], dtype=np.int32), np.array([]).reshape(0, 2))

    result = results[0]
    if result.keypoints is None or result.keypoints.xy is None:
      return PointPacket(np.array([], dtype=np.int32), np.array([]).reshape(0, 2))

    kpts_xy = result.keypoints.xy.cpu().numpy()    # (N_people, 17, 2)
    kpts_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None  # (N_people, 17)

    if kpts_xy.shape[0] == 0:
      return PointPacket(np.array([], dtype=np.int32), np.array([]).reshape(0, 2))

    # Pick person with most high-confidence keypoints
    if kpts_conf is not None and kpts_xy.shape[0] > 1:
      best_idx = int(np.argmax((kpts_conf > MIN_CONFIDENCE).sum(axis=1)))
    else:
      best_idx = 0

    xy = kpts_xy[best_idx]          # (17, 2)
    conf = kpts_conf[best_idx] if kpts_conf is not None else np.ones(17)

    # Filter low-confidence keypoints
    mask = conf >= MIN_CONFIDENCE
    point_ids = np.where(mask)[0].astype(np.int32)
    img_loc = xy[mask].astype(np.float64)

    if len(point_ids) == 0:
      return PointPacket(np.array([], dtype=np.int32), np.array([]).reshape(0, 2))

    # Undo rotation
    img_loc = unrotate_points(img_loc, rotation_count, frame_w, frame_h)

    return PointPacket(point_ids, img_loc)
