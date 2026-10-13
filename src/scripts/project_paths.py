"""Resolve project-relative inputs and serialize portable file references."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def relative_path(value, base=PROJECT_ROOT):
    return Path(os.path.relpath(value, base)).as_posix()


os.environ.setdefault("YOLO_CONFIG_DIR", str(project_path("results/runtime_config")))
