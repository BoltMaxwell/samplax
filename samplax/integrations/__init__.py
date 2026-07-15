"""Adapters that make samplax kernels drop into host codebases' seams."""

from .nested import NestedState, nested_correction

__all__ = ["nested_correction", "NestedState"]
