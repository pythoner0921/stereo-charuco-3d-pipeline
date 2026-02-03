"""Quick test: verify AutoCalibConfig from YAML."""
import sys
from pathlib import Path

tool_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(tool_root / "src"))

from recorder.auto_calibrate import AutoCalibConfig
from recorder.config import load_yaml_config

yaml_data = load_yaml_config(tool_root / "configs" / "default.yaml")
project_dir = Path(r"D:\partition3\PHD_reserch\stereo-charuco-3d-pipeline\projects\caliscope\caliscope_project")
config = AutoCalibConfig.from_yaml(yaml_data, project_dir)

print(f"project_dir: {config.project_dir}")
print(f"charuco: {config.charuco_columns}x{config.charuco_rows}")
print(f"image_size: {config.image_size}")
print(f"ports: {config.ports}")
print(f"board: {config.board_width_cm}x{config.board_height_cm}cm")
print(f"dictionary: {config.dictionary}")
print(f"square_size: {config.square_size_cm}cm")

# Verify video files exist
intrinsic_dir = config.project_dir / "calibration" / "intrinsic"
extrinsic_dir = config.project_dir / "calibration" / "extrinsic"
for port in config.ports:
    i_path = intrinsic_dir / f"port_{port}.mp4"
    e_path = extrinsic_dir / f"port_{port}.mp4"
    print(f"port_{port} intrinsic: {'OK' if i_path.exists() else 'MISSING'} ({i_path})")
    print(f"port_{port} extrinsic: {'OK' if e_path.exists() else 'MISSING'} ({e_path})")

print("\nAll checks passed!")
