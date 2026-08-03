"""Load TensorFlow before pdal, which has to happen before anything else is imported.

Both libraries bring their own native threading runtime, and on macOS the order
they load in decides whether they work. With pdal first, every tf.data pipeline -
which is what keras predict and fit build - deadlocks in iterator_get_next on
Apple Silicon, and aborts on Intel. With TensorFlow first, both behave.

pytest imports this before it collects any test module, so it lands ahead of
test_lidar.py pulling in pdal, and fixes the order for the whole session.
"""
import os

# Match what predictions.py sets, which would otherwise come too late to affect
# an already-imported TensorFlow.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow  # noqa: E402,F401
