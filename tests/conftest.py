import os
import sys

# Import the TensorFlow guard before pytest collects any test module, to avoid segfault errors on mac silicon.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test_classifications'))
import _tf_guard  # noqa: E402,F401
