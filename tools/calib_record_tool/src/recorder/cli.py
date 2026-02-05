"""
Command-line entry points for stereo-charuco-pipeline.

After `pip install .`, these commands are available:
  stereo-pipeline     Full workflow (Project Manager -> Calibration -> Pipeline)
  stereo-calibrate    Calibration UI only
  stereo-record       Pipeline UI only (record + reconstruct)
"""
import argparse
import logging
import sys
from pathlib import Path

from .paths import resolve_default_config


def main():
    """Full workflow: Project Manager -> Calibration UI -> Pipeline UI."""
    import tkinter as tk
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Unified Stereo Pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: built-in default.yaml)",
    )
    parser.add_argument(
        "--projects-base", type=str, default=None,
        help="Base directory for projects (default: ./projects)",
    )
    args = parser.parse_args()

    config_path = _resolve_config(args.config)
    projects_base = Path(args.projects_base) if args.projects_base else Path.cwd() / "projects"

    # Create a single persistent Tk root to avoid Windows crash caused by
    # creating/destroying multiple tk.Tk() instances sequentially.
    root = tk.Tk()
    root.withdraw()

    # Stage 1: Project Manager
    from .project_manager import ProjectManagerUI

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

    # Stage 2: Advanced Calibration UI (if needed)
    if context.needs_calibration:
        print("\nOpening Advanced Calibration Recording UI...")
        from .calibration_ui_advanced import CalibrationUIAdvanced

        calib_app = CalibrationUIAdvanced(
            config_path=config_path,
            project_dir=context.project_dir,
            on_complete=lambda: None,
            master=root,
        )
        root.wait_window(calib_app)

    # Stage 3: Pipeline UI
    print("\nOpening Pipeline UI...")
    from .pipeline_ui import PipelineUI

    pipeline_app = PipelineUI(
        config_path=config_path,
        project_dir=context.project_dir,
        master=root,
    )
    root.wait_window(pipeline_app)

    print("\nPipeline closed.")
    root.destroy()


def calibrate_only():
    """Advanced Calibration UI standalone."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Stereo Calibration UI (Advanced)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--project-dir", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_config(args.config)

    from .calibration_ui_advanced import CalibrationUIAdvanced

    app = CalibrationUIAdvanced(
        config_path=config_path,
        project_dir=Path(args.project_dir) if args.project_dir else None,
    )
    app.mainloop()


def pipeline_only():
    """Pipeline UI standalone."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Stereo Pipeline UI")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--project-dir", type=str, default=None)
    args = parser.parse_args()

    config_path = _resolve_config(args.config)

    from .pipeline_ui import PipelineUI

    app = PipelineUI(
        config_path=config_path,
        project_dir=Path(args.project_dir) if args.project_dir else None,
    )
    app.mainloop()


def _resolve_config(user_config):
    """Resolve config path from user arg or built-in default."""
    if user_config:
        p = Path(user_config)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    return resolve_default_config()
