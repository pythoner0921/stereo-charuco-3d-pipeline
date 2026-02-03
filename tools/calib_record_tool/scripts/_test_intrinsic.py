"""Test: verify charuco detection + intrinsic calibration pipeline."""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

tool_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(tool_root / "src"))

from recorder.auto_calibrate import (
    AutoCalibConfig,
    _collect_charuco_points_from_video,
    _build_image_points_from_packets,
)
from recorder.config import load_yaml_config
from caliscope.core.charuco import Charuco
from caliscope.trackers.charuco_tracker import CharucoTracker
from caliscope.cameras.camera_array import CameraData
from caliscope.core.calibrate_intrinsics import run_intrinsic_calibration

yaml_data = load_yaml_config(tool_root / "configs" / "default.yaml")
project_dir = Path(r"D:\partition3\PHD_reserch\stereo-charuco-3d-pipeline\projects\caliscope\caliscope_project")
config = AutoCalibConfig.from_yaml(yaml_data, project_dir)

print("Creating charuco + tracker...")
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
print(f"Charuco board: {charuco.columns}x{charuco.rows}, square_size={charuco.square_size_overide_cm}cm")

# Test charuco detection on port 1 intrinsic video (subsample=50 for speed)
intrinsic_dir = config.project_dir / "calibration" / "intrinsic"
port = 1

print(f"\nCollecting charuco points from port {port} (subsample=50 for test)...")
collected = _collect_charuco_points_from_video(
    intrinsic_dir, port, tracker, subsample=50,
)
print(f"Collected {len(collected)} frames with detections")

if not collected:
    print("ERROR: No charuco corners detected!")
    sys.exit(1)

# Show sample detection
first_frame_idx, first_points = collected[0]
print(f"First detection at frame {first_frame_idx}: {len(first_points.point_id)} corners")
print(f"  point_ids: {first_points.point_id[:5]}...")
print(f"  img_loc shape: {first_points.img_loc.shape}")
print(f"  obj_loc shape: {first_points.obj_loc.shape if first_points.obj_loc is not None else 'None'}")

# Build ImagePoints
print("\nBuilding ImagePoints DataFrame...")
image_points = _build_image_points_from_packets(collected, port)
print(f"ImagePoints: {len(image_points.df)} rows")
print(f"Columns: {list(image_points.df.columns)}")
print(f"Unique sync_indices: {image_points.df['sync_index'].nunique()}")

# Run intrinsic calibration
print("\nRunning intrinsic calibration...")
camera = CameraData(port=port, size=config.image_size, rotation_count=0)
output = run_intrinsic_calibration(camera, image_points)

print(f"\nCalibration result for port {port}:")
print(f"  RMSE: {output.report.rmse:.3f}px")
print(f"  Frames used: {output.report.frames_used}")
print(f"  Coverage: {output.report.coverage_fraction:.0%}")
print(f"  Matrix:\n    {output.camera.matrix}")
print(f"  Distortions: {output.camera.distortions}")

print("\nIntrinsic calibration test PASSED!")
