"""Core protocol: a sampler is an (init, step) pair over pytree positions.

Every gradient-driven kernel in samplax is a :class:`Kernel`:

- ``init(key, position) -> state`` builds the kernel state. Every state is a
  NamedTuple whose first field is ``position`` (the current sample).
- ``step(key, state, grad, step_size, temperature=1.0) -> state`` advances one
  step. ``grad`` is the **ascent** gradient of the log-density (log-posterior)
  at ``state.position``, as a pytree matching the position; ``step_size`` and
  ``temperature`` are passed per step so schedules and tempering compose from
  the outside (``temperature=0`` disables noise, turning the sampler into its
  optimization counterpart — used by cyclical exploration phases).

Kernels never own the loop: drive them with ``lax.scan``, a Python loop, or
from inside an outer sampler (see ``samplax.integrations``). Samplers that
need more than a per-step gradient (AMAGOLD's amortized Metropolis-Hastings
correction needs energy evaluations) define their own richer protocol in
their module and are not :class:`Kernel` instances.

Dtype policy: kernels inherit the dtype of the position they are given (both
float32 and the x64-enabled float64 workflows are supported).
"""

from typing import Callable, NamedTuple

import jax
from jax.flatten_util import ravel_pytree


class Kernel(NamedTuple):
    init: Callable
    step: Callable


def gaussian_like(key, tree):
    """Standard-normal noise shaped like ``tree`` (one flat draw, unraveled).

    A single flat draw keeps the noise stream independent of the pytree
    structure, so refactoring parameters into different containers does not
    change the sampled trajectory.
    """
    flat, unravel = ravel_pytree(tree)
    return unravel(jax.random.normal(key, flat.shape, flat.dtype))


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(c, a):
    return jax.tree_util.tree_map(lambda x: c * x, a)
