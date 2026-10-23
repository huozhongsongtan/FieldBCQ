"""Resolve project-relative inputs and serialize portable file references."""

import os
import sys
from pathlib import Path


# Set before importing OpenCV (including in spawned workers). Keep native errors visible.
# Override with OPENCV_LOG_LEVEL=WARNING when diagnosing image-reader warnings.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = str(PROJECT_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)  # use this fork even when an official package is also installed


def project_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def relative_path(value, base=PROJECT_ROOT):
    return Path(os.path.relpath(value, base)).as_posix()


def relative_or_absolute(value, base):
    """Keep paths inside base relative, without requiring Python 3.9 pathlib APIs."""
    path = Path(value)
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


os.environ.setdefault("YOLO_CONFIG_DIR", str(project_path("results/runtime_config")))
