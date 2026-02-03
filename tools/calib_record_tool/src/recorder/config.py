from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RecorderConfig:
    # capture
    device: str = "3D USB Camera"
    size: str = "3200x1200"
    fps: int = 60
    seconds: int = 120

    # output
    outdir: str = "."
    outroot: str = ""  # if empty: script decides session folder

    # optional: pass-through (future)
    extra_args: Optional[list[str]] = None

    def to_args(self) -> list[str]:
        """Convert config to CLI args that match scripts/record_and_split.py."""
        args = [
            "--device", self.device,
            "--size", self.size,
            "--fps", str(int(self.fps)),
            "--seconds", str(int(self.seconds)),
            "--outdir", str(self.outdir),
        ]
        if self.outroot:
            args += ["--outroot", str(self.outroot)]
        if self.extra_args:
            args += list(self.extra_args)
        return args

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML into dict. If PyYAML isn't installed, raise a clear error.
    """
    path = Path(path)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load YAML config. Please `pip install pyyaml`."
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict. Got: {type(data)}")
    return data


def config_from_yaml(path: str | Path) -> RecorderConfig:
    """
    Create RecorderConfig from YAML mapping. Unknown keys are ignored (safe for future expansion).
    """
    data = load_yaml_config(path)
    cfg = RecorderConfig()

    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg
