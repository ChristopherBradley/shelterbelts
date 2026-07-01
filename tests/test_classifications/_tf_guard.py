import subprocess
import sys


def _probe_tensorflow():
    """Detect whether TensorFlow can be imported safely in this environment."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import tensorflow"],
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "TensorFlow import timed out (>300s) in this environment."

    if proc.returncode == 0:
        return True, ""

    stderr_lines = proc.stderr.decode(errors="replace").strip().splitlines()
    tail = stderr_lines[-1] if stderr_lines else f"exit code {proc.returncode}"
    reason = (
        f"TensorFlow could not be imported in a subprocess (exit code "
        f"{proc.returncode}: {tail}). Likely a broken TensorFlow build "
    )
    return False, reason

TF_AVAILABLE, TF_SKIP_REASON = _probe_tensorflow()
