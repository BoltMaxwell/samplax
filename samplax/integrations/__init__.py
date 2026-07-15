"""Adapters that make samplax kernels drop into host codebases' seams."""

from .nested import nested_correction

__all__ = ["nested_correction"]
