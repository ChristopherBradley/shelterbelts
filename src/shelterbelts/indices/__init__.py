"""Shelterbelts indices module - categorize land use and vegetation patterns."""

from .tree_categories import tree_categories
from .shelter_categories import shelter_categories
from .cover_categories import cover_categories

__all__ = [
    'tree_categories',
    'shelter_categories',
    'cover_categories',
]
