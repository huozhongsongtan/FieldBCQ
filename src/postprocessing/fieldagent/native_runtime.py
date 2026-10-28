"""Process-local Conda DLL search order; no global changes or package installs."""
import hashlib
import os
from pathlib import Path
import sys

_DLL_HANDLES = []


def configure():
    base = Path(sys.base_prefix).resolve()
    native = base / "Library" / "bin"
    if os.name != "nt" or not native.is_dir():
        return {"changed": False, "base_prefix": str(base)}
    prepend = [str(native), str(base)]
    existing = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(prepend + [p for p in existing if p.lower() not in {s.lower() for s in prepend}])
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(native)))
    return {"changed": True, "scope": "this process and its children only", "base_prefix": str(base),
        "dll_search_prefix": prepend, "package_installations": 0, "scientific_method_changes": False}


def preflight():
    record = configure()
    import numpy as np
    # 512x2 triggered the native delay-load crash in the inherited base PATH.
    sizes = [8, 256, 512, 1024, 4096, 16384]
    for n in sizes:
        a = np.arange(n * 2, dtype=np.float64).reshape(n, 2) / 17.
        v = np.array([1.25, -2.5])
        np.testing.assert_allclose(a @ v, np.sum(a * v, axis=-1), rtol=1e-13, atol=1e-13)
    record.update(numpy_version=np.__version__, numpy_file=np.__file__,
        matrix_vector_sizes_passed=sizes, preflight_passed=True)
    import psutil
    record["loaded_blas_dlls"] = sorted({m.path for m in psutil.Process().memory_maps()
        if Path(m.path).name.lower() in {"libblas.dll", "libcblas.dll", "mkl_rt.2.dll", "mkl_core.2.dll"}})
    record["runtime_helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


if __name__ == "__main__":
    import json
    print(json.dumps(preflight(), ensure_ascii=False, indent=2))
