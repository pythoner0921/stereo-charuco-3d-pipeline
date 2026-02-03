"""Test: run the full auto-calibration pipeline end-to-end."""
import sys
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

tool_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(tool_root / "src"))

from recorder.auto_calibrate import AutoCalibConfig, run_auto_calibration
from recorder.config import load_yaml_config

yaml_data = load_yaml_config(tool_root / "configs" / "default.yaml")
project_dir = Path(r"D:\partition3\PHD_reserch\stereo-charuco-3d-pipeline\projects\caliscope\caliscope_project")
config = AutoCalibConfig.from_yaml(yaml_data, project_dir)

print("=" * 60)
print("FULL AUTO-CALIBRATION PIPELINE TEST")
print("=" * 60)
print(f"Project dir: {config.project_dir}")
print(f"Charuco: {config.charuco_columns}x{config.charuco_rows}")
print(f"Image size: {config.image_size}")
print(f"Ports: {config.ports}")
print(f"Intrinsic subsample: {config.intrinsic_subsample}")
print(f"Extrinsic subsample: {config.extrinsic_subsample}")
print()

# Verify all required videos exist
intrinsic_dir = config.project_dir / "calibration" / "intrinsic"
extrinsic_dir = config.project_dir / "calibration" / "extrinsic"
all_ok = True
for port in config.ports:
    i_path = intrinsic_dir / f"port_{port}.mp4"
    e_path = extrinsic_dir / f"port_{port}.mp4"
    i_ok = i_path.exists()
    e_ok = e_path.exists()
    print(f"  port_{port} intrinsic: {'OK' if i_ok else 'MISSING'}")
    print(f"  port_{port} extrinsic: {'OK' if e_ok else 'MISSING'}")
    if not i_ok or not e_ok:
        all_ok = False

if not all_ok:
    print("\nERROR: Missing video files! Cannot run full pipeline test.")
    sys.exit(1)

print("\nStarting pipeline...")
start_time = time.time()


def progress_callback(stage, msg, pct):
    elapsed = time.time() - start_time
    print(f"  [{elapsed:6.1f}s] [{stage:15s}] ({pct:3d}%) {msg}")


result = run_auto_calibration(config, on_progress=progress_callback)

elapsed = time.time() - start_time
print()
print("=" * 60)
print(f"Pipeline finished in {elapsed:.1f}s")
print(f"Success: {result.success}")

if not result.success:
    print(f"ERROR: {result.error_message}")
    sys.exit(1)

print(f"\nIntrinsic RMSE:")
for port, rmse in result.intrinsic_rmse.items():
    print(f"  Port {port}: {rmse:.3f}px")

print(f"\nExtrinsic cost: {result.extrinsic_cost:.4f}")
print(f"Origin sync index: {result.origin_sync_index}")

print(f"\nCamera array ({len(result.camera_array.cameras)} cameras):")
for port, cam in result.camera_array.cameras.items():
    print(f"  Port {port}:")
    print(f"    Size: {cam.size}")
    print(f"    Matrix:\n      {cam.matrix}")
    print(f"    Distortions: {cam.distortions}")
    if cam.translation is not None:
        print(f"    Translation: {cam.translation.flatten()}")
    if cam.rotation is not None:
        print(f"    Rotation: {cam.rotation.flatten()}")

# Verify saved files
print("\nVerifying saved files:")
saved_files = [
    project_dir / "charuco.toml",
    project_dir / "camera_array.toml",
    extrinsic_dir / "frame_timestamps.csv",
    extrinsic_dir / "CHARUCO" / "image_points.csv",
    extrinsic_dir / "CHARUCO" / "camera_array.toml",
    extrinsic_dir / "CHARUCO" / "world_points.csv",
]
for f in saved_files:
    status = "OK" if f.exists() else "MISSING"
    print(f"  {f.name}: {status}")

print("\nFull pipeline test PASSED!")
