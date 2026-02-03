#!/usr/bin/env python
"""
Launch the Stereo Calibration Recording UI.

Usage:
    python scripts/run_calibration_ui.py
    python scripts/run_calibration_ui.py --config configs/default.yaml
"""
import sys
from pathlib import Path

# Add src to path
tool_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(tool_root / "src"))

from recorder.calibration_ui import main

if __name__ == "__main__":
    main()
