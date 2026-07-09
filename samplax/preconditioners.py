"""Preconditioners for Langevin-family kernels.

A :class:`Preconditioner` supplies the diagonal (or identity) metric used by
pSGLD-style updates:

- ``init(position) -> pstate``
- ``update(pstate, grad) -> pstate``        (e.g. accumulate squared gradients)
- ``drift_scale(pstate, tree) -> tree``     (per-leaf factor for the gradient term, M^-1)
- ``noise_scale(pstate, tree) -> tree``     (per-leaf factor for the noise term, M^-1/2)

``rmsprop`` reproduces the RMSprop-diagonal preconditioner of pSGLD
(Li et al., 2016), the same construction as ift-sde's ``psgld_sampler_fast``
and the ``"rmsprop"`` option of its nested samplers. ``identity`` is the
no-op used by plain SGLD/SGHMC. The Gamma(curvature) correction term of full
pSGLD is omitted, as in the reference implementations.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Preconditioner(NamedTuple):
    init: Callable
    update: Callable
    drift_scale: Callable
    noise_scale: Callable


def identity():
    def init(position):
        return ()

    def update(pstate, grad):
        return pstate

    def one_like(_pstate, tree):
        return jax.tree_util.tree_map(jnp.ones_like, tree)

    return Preconditioner(
        init,
        update,
        drift_scale=lambda pstate, tree: one_like(pstate, tree),
        noise_scale=lambda pstate, tree: one_like(pstate, tree),
    )


def rmsprop(beta=0.99, eps=1e-5):
    """RMSprop-diagonal preconditioner (pSGLD, Li et al. 2016).

    v <- beta v + (1 - beta) g^2;  G = 1 / (eps + sqrt(v));
    drift term scaled by G, noise by sqrt(G).
    """

    def init(position):
        return jax.tree_util.tree_map(jnp.zeros_like, position)

    def update(v, grad):
        return jax.tree_util.tree_map(
            lambda v_, g: beta * v_ + (1.0 - beta) * g * g, v, grad)

    def drift_scale(v, tree):
        return jax.tree_util.tree_map(
            lambda v_: 1.0 / (eps + jnp.sqrt(v_)), v)

    def noise_scale(v, tree):
        return jax.tree_util.tree_map(
            lambda v_: 1.0 / jnp.sqrt(eps + jnp.sqrt(v_)), v)

    return Preconditioner(init, update, drift_scale, noise_scale)
