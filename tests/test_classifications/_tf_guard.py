"""Decide whether TensorFlow can be used in this test session, and load it early.

Established from the CI logs: on macOS arm64 the TensorFlow import segfaults
part-way through a pytest session, while `python -c "import tensorflow"` in a
clean subprocess on the same runner succeeds. The arm64 wheel's dylibs depend on
nothing outside system frameworks, so a symbol clash with conda-forge is ruled
out. What differs between the working and crashing case is how much is already
resident when the ~700 MB of native libraries get mapped in.

Still unproven: that load order is the whole story. Importing TensorFlow before
anything else is the cheapest way to test that, so this module is imported from
tests/conftest.py ahead of any test module, and the subprocess probe is followed
by an immediate in-process import. Treat it as a diagnostic rather than a known
fix - predictions.py imported TensorFlow at module scope before commit 48ce6bc
and still segfaulted, though rasterio, matplotlib, geopandas and pyproj were
imported above it there, so this is a stronger version of an arrangement that has
already failed once.

If the arm job now crashes at session start instead of part-way through, load
order is not sufficient and SHELTERBELTS_SKIP_TF=1 should be set for that job.
"""
import os
import subprocess
import sys


def _probe_tensorflow():
    """Detect whether TensorFlow can be imported safely in this environment."""
    if os.environ.get("SHELTERBELTS_SKIP_TF"):
        return False, "TensorFlow tests disabled via SHELTERBELTS_SKIP_TF."

    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import tensorflow"],
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "TensorFlow import timed out (>300s) in this environment."

    if proc.returncode != 0:
        stderr_lines = proc.stderr.decode(errors="replace").strip().splitlines()
        tail = stderr_lines[-1] if stderr_lines else f"exit code {proc.returncode}"
        reason = (
            f"TensorFlow could not be imported in a subprocess (exit code "
            f"{proc.returncode}: {tail}). Likely a broken TensorFlow build "
        )
        return False, reason

    # The probe only shows TensorFlow loads into a clean process, so try claiming
    # that same state here rather than leaving the real import until the tests run.
    # These match the settings predictions.py applies before importing keras,
    # which by then would be too late to affect an already-loaded TensorFlow.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow  # noqa: F401

    return True, ""

TF_AVAILABLE, TF_SKIP_REASON = _probe_tensorflow()
