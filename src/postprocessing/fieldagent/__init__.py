"""FieldAgent: paired parcel boundary vertex-reduction experiments."""
from pathlib import Path
import os
import sys

__version__ = "0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]  # FieldBCQ/src/postprocessing/fieldagent


def activate_engine():
    """Configure native libraries before importing numerical compatibility code."""
    from .native_runtime import configure
    configure()
    engine = str(PACKAGE_ROOT / "_engine")
    if engine not in sys.path:
        sys.path.insert(0, engine)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "results/runtime_config"))

