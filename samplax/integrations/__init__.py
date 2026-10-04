"""Adapters that make samplax kernels drop into host codebases' seams.

INTERNAL AND UNSTABLE: this package is not part of the samplax public API.
"""

# TODO(pre-1.0.0): move this package out of samplax. It is glue for the ift-sde
# codebase, and a loop-owning driver does not belong in a library whose kernels
# never own the loop. The adapter goes to its consumers, which then import it
# from its new home; samplax keeps kernels, schedules, preconditioners and
# transforms only. Must be done before 1.0.0.

from .nested import NestedState, nested_correction

__all__ = ["nested_correction", "NestedState"]
