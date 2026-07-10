"""SGHMC kernel, momentum-buffer form (optionally preconditioned).

Update (ascent gradient g, temperature T, ``alpha`` = friction = 1 - momentum)::

    buf <- (1 - alpha) buf + step_size * G g
           + sqrt(2 (alpha - beta_hat) step_size T) sqrt(G) n
    x   <- x + buf

with beta_hat = 0.5 * v_hat * step_size. With ``v_hat = 0`` (default) this is
exactly the csgmcmc form::

    buf <- (1 - alpha) buf + step_size * g + sqrt(2 alpha step_size T) n
    x   <- x + buf

vendored from csgmcmc-jax ``samplers/jax_backend.py:make_sghmc`` (verified
against the original csgmcmc PyTorch update, and the same update as the
AMAGOLD repo's SGHMC baseline and bayesnn's SGHMCUpdater with the weight
decay folded into ``g``). ``v_hat`` applies Chen et al. (2014)'s
gradient-noise correction: noise variance 2 step_size (alpha - beta_hat) T
with beta_hat = 0.5 * v_hat * step_size, matching SGHMC-jax ``make_sghmc``
under the mapping eta = step_size.

The momentum refresh precedes the position update (the original "pq"
ordering); see the SGHMC-jax verification docs for why this matters.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..base import Kernel, gaussian_like
from ..preconditioners import identity


class SGHMCState(NamedTuple):
    position: jax.Array
    momentum: jax.Array
    precond: tuple = ()


def sghmc(alpha=0.1, v_hat=0.0, preconditioner=None, init_momentum_scale=0.0):
    """Build an SGHMC :class:`~samplax.base.Kernel`.

    ``alpha`` is the friction (1 - momentum decay). ``init_momentum_scale``
    scales the N(0, 1) momentum initialization (0 starts the buffer at rest,
    the csgmcmc convention; sqrt(step_size) reproduces the ML-SGHMC /
    AMAGOLD initialization).
    """
    precond = identity() if preconditioner is None else preconditioner

    def init(key, position):
        momentum = jax.tree_util.tree_map(
            lambda n: init_momentum_scale * n, gaussian_like(key, position))
        return SGHMCState(position, momentum, precond.init(position))

    def step(key, state, grad, step_size, temperature=1.0):
        pstate = precond.update(state.precond, grad)
        drift = precond.drift_scale(pstate, grad)
        nscale = precond.noise_scale(pstate, grad)
        noise = gaussian_like(key, state.position)
        beta_hat = 0.5 * v_hat * step_size
        scale = jnp.sqrt(2.0 * step_size * (alpha - beta_hat) * temperature)
        momentum = jax.tree_util.tree_map(
            lambda buf, g, d, n, s: (1.0 - alpha) * buf + step_size * d * g
            + scale * s * n,
            state.momentum, grad, drift, noise, nscale)
        position = jax.tree_util.tree_map(
            lambda x, buf: x + buf, state.position, momentum)
        return SGHMCState(position, momentum, pstate)

    return Kernel(init, step)
