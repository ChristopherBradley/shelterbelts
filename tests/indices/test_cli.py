"""Test command line argument defaults match function signature defaults."""

import sys
import inspect
import pytest

from shelterbelts.indices.tree_categories import tree_categories, parse_arguments as parse_tree_args
from shelterbelts.indices.shelter_categories import shelter_categories, parse_arguments as parse_shelter_args
from shelterbelts.indices.cover_categories import cover_categories, parse_arguments as parse_cover_args
from shelterbelts.indices.buffer_categories import buffer_categories, parse_arguments as parse_buffer_args


@pytest.mark.parametrize('func,parser_func,exclude_args', [
    (tree_categories, parse_tree_args, set()),
    (shelter_categories, parse_shelter_args, set()),
    (cover_categories, parse_cover_args, set()),
    (buffer_categories, parse_buffer_args, set()),
])
def test_cli_defaults_match_function_defaults(func, parser_func, exclude_args):
    """Verify CLI argument defaults match function signature defaults"""
    # Get function signature defaults
    sig = inspect.signature(func)
    func_defaults = {
        name: param.default 
        for name, param in sig.parameters.items() 
        if param.default != inspect.Parameter.empty and name not in exclude_args
    }
    
    # Parse arguments with no CLI args (using defaults)
    original_argv = sys.argv
    try:
        sys.argv = ['test']
        parser = parser_func()
        args = parser.parse_args()
    finally:
        sys.argv = original_argv
    
    # Verify each default matches
    for arg_name, expected_val in func_defaults.items():
        actual_val = getattr(args, arg_name, None)
        assert actual_val == expected_val, \
            f"{func.__name__}.{arg_name}: CLI default {actual_val} != function default {expected_val}"
