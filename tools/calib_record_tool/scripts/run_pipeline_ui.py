#!/usr/bin/env python
"""
Launch the simplified pipeline UI: Auto-Calibrate -> Record -> Reconstruct.

Usage:
  conda activate caliscope311
  python scripts/run_pipeline_ui.py
  python scripts/run_pipeline_ui.py --config configs/default.yaml
"""
import sys
import argparse
from pathlib import Path

# Add src/ to path
tool_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(tool_root / "src"))


def main():
  parser = argparse.ArgumentParser(description="Stereo Pipeline UI")
  parser.add_argument(
    "--config", type=str, default=None,
    help="Path to YAML config file (default: configs/default.yaml)",
  )
  args = parser.parse_args()

  config_path = None
  if args.config:
    config_path = Path(args.config)
    if not config_path.is_absolute():
      config_path = tool_root / config_path

  from recorder.pipeline_ui import PipelineUI
  app = PipelineUI(config_path=config_path)
  app.mainloop()


if __name__ == "__main__":
  main()
