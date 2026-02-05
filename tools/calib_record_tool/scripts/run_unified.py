#!/usr/bin/env python
"""
Unified launcher for the stereo pipeline.

  1. Project Manager  -> select or create a project
  2. Calibration UI   -> record calibration videos (if needed)
  3. Pipeline UI      -> calibrate, record, reconstruct, visualize

Usage:
  conda activate stereo-pipeline
  python scripts/run_unified.py
  python scripts/run_unified.py --config configs/default.yaml
"""
import argparse
import logging
import sys
import tkinter as tk
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Add src/ to path
tool_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(tool_root / "src"))

# Default projects base directory
DEFAULT_PROJECTS_BASE = tool_root.parent.parent / "projects"


def main():
  parser = argparse.ArgumentParser(description="Unified Stereo Pipeline")
  parser.add_argument(
    "--config", type=str, default=None,
    help="Path to YAML config file (default: configs/default.yaml)",
  )
  parser.add_argument(
    "--projects-base", type=str, default=None,
    help=f"Base directory for projects (default: {DEFAULT_PROJECTS_BASE})",
  )
  args = parser.parse_args()

  # Resolve config path
  config_path = None
  if args.config:
    config_path = Path(args.config)
    if not config_path.is_absolute():
      config_path = tool_root / config_path

  # Resolve projects base
  projects_base = Path(args.projects_base) if args.projects_base else DEFAULT_PROJECTS_BASE

  # Create a single persistent Tk root.
  # Each stage opens as a Toplevel of this root, avoiding the Windows crash
  # caused by creating/destroying multiple tk.Tk() instances.
  root = tk.Tk()
  root.withdraw()

  # ── Stage 1: Project Manager ──────────────────────────────
  from recorder.project_manager import ProjectManagerUI

  pm = ProjectManagerUI(projects_base, master=root)
  root.wait_window(pm)

  context = pm.result
  if context is None:
    print("No project selected. Exiting.")
    root.destroy()
    return

  print(f"Project: {context.project_dir}")
  print(f"  New: {context.is_new}")
  print(f"  Needs calibration: {context.needs_calibration}")

  # ── Stage 2: Advanced Calibration UI (if needed) ─────────
  if context.needs_calibration:
    print("\nOpening Advanced Calibration Recording UI...")

    from recorder.calibration_ui_advanced import CalibrationUIAdvanced

    calib_app = CalibrationUIAdvanced(
      config_path=config_path,
      project_dir=context.project_dir,
      on_complete=lambda: None,
      master=root,
    )
    root.wait_window(calib_app)

  # ── Stage 3: Pipeline UI ──────────────────────────────────
  print("\nOpening Pipeline UI...")

  from recorder.pipeline_ui import PipelineUI

  pipeline_app = PipelineUI(
    config_path=config_path,
    project_dir=context.project_dir,
    master=root,
  )
  root.wait_window(pipeline_app)

  print("\nPipeline closed.")
  root.destroy()


if __name__ == "__main__":
  main()
