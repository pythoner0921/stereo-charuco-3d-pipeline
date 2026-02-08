"""VideoPose3D monocular 3D lifting + stereo fusion.

Fuses monocular 3D estimates from Facebook Research's VideoPose3D
with stereo triangulation results to improve accuracy under occlusion.

Requires: torch (optional dependency, installed via pip install .[videopose3d])
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

# ── COCO 17 → Human3.6M 17 keypoint mapping ──────────────────────────

# Human3.6M joint order (17 joints):
#  0: pelvis, 1: right_hip, 2: right_knee, 3: right_ankle,
#  4: left_hip, 5: left_knee, 6: left_ankle,
#  7: spine, 8: thorax, 9: neck_base, 10: head_top,
#  11: left_shoulder, 12: left_elbow, 13: left_wrist,
#  14: right_shoulder, 15: right_elbow, 16: right_wrist

# Direct COCO→H36M mapping (index: H36M joint, value: COCO keypoint or None)
_COCO_TO_H36M_DIRECT = {
  1: 12,   # right_hip ← COCO right_hip
  2: 14,   # right_knee ← COCO right_knee
  3: 16,   # right_ankle ← COCO right_ankle
  4: 11,   # left_hip ← COCO left_hip
  5: 13,   # left_knee ← COCO left_knee
  6: 15,   # left_ankle ← COCO left_ankle
  9: 0,    # neck_base ← COCO nose (approximation)
  10: 0,   # head_top ← COCO nose (approximation)
  11: 5,   # left_shoulder ← COCO left_shoulder
  12: 7,   # left_elbow ← COCO left_elbow
  13: 9,   # left_wrist ← COCO left_wrist
  14: 6,   # right_shoulder ← COCO right_shoulder
  15: 8,   # right_elbow ← COCO right_elbow
  16: 10,  # right_wrist ← COCO right_wrist
}

# Computed joints (midpoints)
# 0: pelvis = midpoint(left_hip=11, right_hip=12) in COCO
# 7: spine = midpoint(pelvis, thorax)
# 8: thorax = midpoint(left_shoulder=5, right_shoulder=6) in COCO

N_H36M_JOINTS = 17
N_COCO_JOINTS = 17

# VideoPose3D checkpoint URL (Facebook Research, d-pt-243 model)
_CHECKPOINT_URL = (
  "https://dl.fbaipublicfiles.com/video-pose-3d/d-pt-243.bin"
)
_CHECKPOINT_NAME = "d-pt-243.bin"


def coco_to_h36m_2d(coco_2d: np.ndarray) -> np.ndarray:
  """Convert COCO 17-keypoint 2D array to Human3.6M 17-joint format.

  Args:
    coco_2d: (T, 17, 2) array of COCO keypoints

  Returns:
    (T, 17, 2) array of H36M keypoints
  """
  T = coco_2d.shape[0]
  h36m = np.zeros((T, N_H36M_JOINTS, 2), dtype=np.float32)

  # Direct mappings
  for h36m_idx, coco_idx in _COCO_TO_H36M_DIRECT.items():
    h36m[:, h36m_idx, :] = coco_2d[:, coco_idx, :]

  # Computed joints
  left_hip = coco_2d[:, 11, :]   # COCO left_hip
  right_hip = coco_2d[:, 12, :]  # COCO right_hip
  left_shoulder = coco_2d[:, 5, :]
  right_shoulder = coco_2d[:, 6, :]

  pelvis = (left_hip + right_hip) / 2.0
  thorax = (left_shoulder + right_shoulder) / 2.0
  spine = (pelvis + thorax) / 2.0

  h36m[:, 0, :] = pelvis    # pelvis
  h36m[:, 7, :] = spine     # spine
  h36m[:, 8, :] = thorax    # thorax

  return h36m


def _get_checkpoint_path() -> Path:
  """Download VideoPose3D checkpoint if not cached."""
  cache_dir = Path(user_cache_dir("stereo-pipeline", "stereo-pipeline")) / "videopose3d"
  cache_dir.mkdir(parents=True, exist_ok=True)

  ckpt_path = cache_dir / _CHECKPOINT_NAME
  if ckpt_path.exists():
    return ckpt_path

  logger.info(f"Downloading VideoPose3D checkpoint to {ckpt_path}...")
  urllib.request.urlretrieve(_CHECKPOINT_URL, str(ckpt_path))
  logger.info("Download complete.")
  return ckpt_path


def _build_model():
  """Build VideoPose3D temporal convolution model and load weights.

  Architecture: d-pt-243 (dilated, pretrained, 243-frame receptive field)
  """
  import torch
  import torch.nn as nn

  class TemporalModelOptimized1f(nn.Module):
    """VideoPose3D temporal convolution model (optimized single-frame output)."""

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
      super().__init__()

      self.num_joints_in = num_joints_in
      self.in_features = in_features
      self.num_joints_out = num_joints_out
      self.filter_widths = filter_widths

      self.drop = nn.Dropout(dropout)
      self.relu = nn.ReLU(inplace=True)

      self.pad = []
      self.causal_shift = []
      self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
      self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels,
                                    filter_widths[0], bias=False)
      self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

      layers_conv = []
      layers_bn = []

      next_dilation = filter_widths[0]
      for i in range(1, len(filter_widths)):
        self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
        self.causal_shift.append(filter_widths[i] // 2 * next_dilation if causal else 0)

        layers_conv.append(nn.Conv1d(channels, channels,
                                      filter_widths[i],
                                      dilation=next_dilation, bias=False))
        layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
        layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
        layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

        next_dilation *= filter_widths[i]

      self.layers_conv = nn.ModuleList(layers_conv)
      self.layers_bn = nn.ModuleList(layers_bn)

    def set_bn_momentum(self, momentum):
      self.expand_bn.momentum = momentum
      for bn in self.layers_bn:
        bn.momentum = momentum

    def receptive_field(self):
      frames = 0
      for f in self.filter_widths:
        frames = frames * f + f - 1
      return frames + 1

    def forward(self, x):
      # x: (B, T, J*C)
      x = x.permute(0, 2, 1)  # (B, J*C, T)
      x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

      for i in range(len(self.pad)):
        pad = self.pad[i]
        shift = self.causal_shift[i]
        res = x[:, :, pad + shift: x.shape[2] - pad + shift]

        x = self.drop(self.relu(self.layers_bn[2 * i](
          self.layers_conv[2 * i](x))))
        x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](
          self.layers_conv[2 * i + 1](x))))

      x = self.shrink(x)
      return x.permute(0, 2, 1)  # (B, 1, J*3)

  # d-pt-243: dilated, pretrained, 243-frame receptive field
  # filter_widths=[3,3,3,3,3] with dilation → 243 frames
  model = TemporalModelOptimized1f(
    num_joints_in=N_H36M_JOINTS,
    in_features=2,
    num_joints_out=N_H36M_JOINTS,
    filter_widths=[3, 3, 3, 3, 3],
    causal=False,
    dropout=0.25,
    channels=1024,
  )

  ckpt_path = _get_checkpoint_path()
  checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

  # The checkpoint contains model_pos state_dict
  state_dict = checkpoint.get("model_pos", checkpoint)
  model.load_state_dict(state_dict)
  model.eval()

  return model


def _predict_3d_monocular(model, h36m_2d: np.ndarray) -> np.ndarray:
  """Run VideoPose3D inference on normalized H36M 2D input.

  Args:
    model: VideoPose3D temporal convolution model
    h36m_2d: (T, 17, 2) normalized 2D keypoints

  Returns:
    (T, 17, 3) root-relative 3D predictions
  """
  import torch

  receptive_field = model.receptive_field()
  T = h36m_2d.shape[0]

  # Pad input to match receptive field
  pad = receptive_field // 2
  h36m_padded = np.pad(h36m_2d, ((pad, pad), (0, 0), (0, 0)), mode="edge")

  # Reshape for model: (1, T_padded, J*2)
  inp = h36m_padded.reshape(1, -1, N_H36M_JOINTS * 2)
  inp_tensor = torch.from_numpy(inp).float()

  with torch.no_grad():
    out = model(inp_tensor)  # (1, T, J*3)

  predicted = out.numpy().reshape(-1, N_H36M_JOINTS, 3)

  # Trim to original length
  predicted = predicted[:T]

  return predicted


def _procrustes_align(
  source: np.ndarray,
  target: np.ndarray,
  mask: np.ndarray,
) -> np.ndarray:
  """Procrustes alignment: align source to target using well-triangulated points.

  Args:
    source: (N, 3) monocular predictions
    target: (N, 3) stereo triangulation results
    mask: (N,) boolean mask — True for well-triangulated anchor points

  Returns:
    (N, 3) aligned source points
  """
  if mask.sum() < 3:
    return source

  src = source[mask]
  tgt = target[mask]

  # Center
  src_mean = src.mean(axis=0)
  tgt_mean = tgt.mean(axis=0)
  src_c = src - src_mean
  tgt_c = tgt - tgt_mean

  # Scale
  src_scale = np.sqrt((src_c ** 2).sum() / len(src_c))
  tgt_scale = np.sqrt((tgt_c ** 2).sum() / len(tgt_c))

  if src_scale < 1e-8 or tgt_scale < 1e-8:
    return source

  src_n = src_c / src_scale
  tgt_n = tgt_c / tgt_scale

  # Rotation (Kabsch algorithm)
  H = src_n.T @ tgt_n
  U, S, Vt = np.linalg.svd(H)
  d = np.linalg.det(Vt.T @ U.T)
  sign_mat = np.diag([1, 1, d])
  R = Vt.T @ sign_mat @ U.T

  scale = tgt_scale / src_scale

  # Apply transform to all source points
  aligned = (source - src_mean) * scale @ R.T + tgt_mean
  return aligned


def _h36m_to_coco_indices() -> dict[int, int]:
  """Map H36M joint indices back to COCO indices for fusion."""
  coco_to_h36m = {}
  for h36m_idx, coco_idx in _COCO_TO_H36M_DIRECT.items():
    coco_to_h36m[h36m_idx] = coco_idx
  # Reverse: COCO index → H36M index
  h36m_for_coco = {}
  for h36m_idx, coco_idx in coco_to_h36m.items():
    if coco_idx not in h36m_for_coco:
      h36m_for_coco[coco_idx] = h36m_idx
  return h36m_for_coco


def fuse_stereo_and_monocular(
  xyz_csv_path: Path,
  xy_csv_path: Path,
  port: int = 1,
  conf_threshold: float = 0.5,
  stereo_weight_good: float = 0.85,
  mono_weight_occluded: float = 0.75,
) -> None:
  """Fuse stereo triangulation with VideoPose3D monocular estimates.

  Reads the stereo xyz CSV, runs VideoPose3D on the 2D detections,
  and writes fused results back to the same xyz CSV.

  Args:
    xyz_csv_path: Path to stereo xyz_YOLOV8_POSE.csv
    xy_csv_path: Path to 2D xy_YOLOV8_POSE.csv
    port: Camera port to use for monocular input (default: 1 = left camera)
    conf_threshold: Reprojection error threshold for "well-triangulated"
    stereo_weight_good: Weight for stereo data on well-triangulated joints
    mono_weight_occluded: Weight for monocular data on occluded joints
  """
  import torch  # noqa: F401 — ensure torch is available

  logger.info("Running VideoPose3D fusion...")

  # Load stereo 3D data
  xyz_df = pd.read_csv(xyz_csv_path)
  if xyz_df.empty:
    logger.warning("Empty xyz CSV, skipping fusion")
    return

  # Load 2D detections for monocular input
  xy_df = pd.read_csv(xy_csv_path)
  if xy_df.empty:
    logger.warning("Empty xy CSV, skipping fusion")
    return

  # Filter to selected port
  xy_port = xy_df[xy_df["port"] == port].copy()
  if xy_port.empty:
    logger.warning(f"No 2D detections for port {port}")
    return

  # Build 2D sequence: (T, 17, 2) in COCO order
  sync_indices = sorted(xy_port["sync_index"].unique())
  T = len(sync_indices)
  sync_to_t = {si: t for t, si in enumerate(sync_indices)}

  coco_2d = np.zeros((T, N_COCO_JOINTS, 2), dtype=np.float32)
  coco_valid = np.zeros((T, N_COCO_JOINTS), dtype=bool)

  for _, row in xy_port.iterrows():
    t = sync_to_t.get(row["sync_index"])
    if t is None:
      continue
    pid = int(row["point_id"])
    if 0 <= pid < N_COCO_JOINTS:
      coco_2d[t, pid, 0] = row["img_loc_x"]
      coco_2d[t, pid, 1] = row["img_loc_y"]
      coco_valid[t, pid] = True

  # Normalize 2D coordinates (zero-mean, unit-variance per frame)
  for t in range(T):
    valid = coco_valid[t]
    if valid.sum() >= 2:
      pts = coco_2d[t, valid]
      mean = pts.mean(axis=0)
      std = pts.std()
      if std > 1e-6:
        coco_2d[t] = (coco_2d[t] - mean) / std

  # Convert to H36M format
  h36m_2d = coco_to_h36m_2d(coco_2d)

  # Run VideoPose3D
  logger.info(f"Running VideoPose3D on {T} frames...")
  model = _build_model()
  mono_3d = _predict_3d_monocular(model, h36m_2d)  # (T, 17, 3) H36M
  del model

  # Build stereo 3D array for alignment
  coord_cols = ["x_coord", "y_coord", "z_coord"]
  stereo_sync_indices = sorted(xyz_df["sync_index"].unique())

  # Find overlapping sync indices
  common_syncs = sorted(set(sync_indices) & set(stereo_sync_indices))
  if len(common_syncs) < 5:
    logger.warning(f"Only {len(common_syncs)} common sync indices, skipping fusion")
    return

  # Map COCO↔H36M for fusion
  coco_to_h36m_map = _h36m_to_coco_indices()  # coco_idx → h36m_idx

  # Per-frame fusion
  fused_rows = []
  n_improved = 0

  for sync_idx in common_syncs:
    t = sync_to_t.get(sync_idx)
    if t is None or t >= mono_3d.shape[0]:
      continue

    stereo_frame = xyz_df[xyz_df["sync_index"] == sync_idx]
    mono_frame_h36m = mono_3d[t]  # (17, 3) in H36M order

    # Build stereo array in H36M order for Procrustes
    stereo_h36m = np.full((N_H36M_JOINTS, 3), np.nan)
    stereo_valid = np.zeros(N_H36M_JOINTS, dtype=bool)

    for _, row in stereo_frame.iterrows():
      coco_pid = int(row["point_id"])
      h36m_idx = coco_to_h36m_map.get(coco_pid)
      if h36m_idx is not None:
        stereo_h36m[h36m_idx] = [row["x_coord"], row["y_coord"], row["z_coord"]]
        stereo_valid[h36m_idx] = np.isfinite(row["x_coord"])

    # Procrustes align mono to stereo
    if stereo_valid.sum() >= 3:
      aligned_mono = _procrustes_align(
        mono_frame_h36m, stereo_h36m, stereo_valid,
      )
    else:
      aligned_mono = mono_frame_h36m

    # Fuse per joint (in COCO space for output)
    for coco_pid in range(N_COCO_JOINTS):
      h36m_idx = coco_to_h36m_map.get(coco_pid)
      if h36m_idx is None:
        # No H36M mapping — keep stereo only
        stereo_row = stereo_frame[stereo_frame["point_id"] == coco_pid]
        if not stereo_row.empty:
          fused_rows.append(stereo_row.iloc[0].to_dict())
        continue

      stereo_row = stereo_frame[stereo_frame["point_id"] == coco_pid]
      has_stereo = not stereo_row.empty and np.isfinite(
        stereo_row.iloc[0]["x_coord"])
      mono_xyz = aligned_mono[h36m_idx]

      if has_stereo:
        # Well-triangulated: prefer stereo
        sr = stereo_row.iloc[0]
        stereo_xyz = np.array([sr["x_coord"], sr["y_coord"], sr["z_coord"]])
        w_s = stereo_weight_good
        w_m = 1.0 - w_s
        fused_xyz = w_s * stereo_xyz + w_m * mono_xyz
        row_dict = sr.to_dict()
        row_dict["x_coord"] = fused_xyz[0]
        row_dict["y_coord"] = fused_xyz[1]
        row_dict["z_coord"] = fused_xyz[2]
        fused_rows.append(row_dict)
      else:
        # Occluded: use monocular prediction
        n_improved += 1
        fused_rows.append({
          "sync_index": sync_idx,
          "point_id": coco_pid,
          "x_coord": mono_xyz[0],
          "y_coord": mono_xyz[1],
          "z_coord": mono_xyz[2],
        })

  if not fused_rows:
    logger.warning("No fused rows produced")
    return

  # Keep non-overlapping stereo rows unchanged
  non_common = xyz_df[~xyz_df["sync_index"].isin(common_syncs)]
  fused_df = pd.concat([non_common, pd.DataFrame(fused_rows)], ignore_index=True)
  fused_df = fused_df.sort_values(["sync_index", "point_id"]).reset_index(drop=True)

  fused_df.to_csv(xyz_csv_path, index=False)
  logger.info(
    f"VideoPose3D fusion complete: {n_improved} occluded joints improved, "
    f"{len(fused_df)} total rows written"
  )
