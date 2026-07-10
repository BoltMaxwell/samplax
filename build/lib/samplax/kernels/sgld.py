"""SGLD kernel (optionally preconditioned = pSGLD).

Update (ascent gradient g, temperature T, preconditioner G):

    x <- x + step_size * G g + sqrt(2 * step_size * T) * sqrt(G) n

Provenance: vendored from csgmcmc-jax ``samplers/jax_backend.py:make_sgld``
(verified against the original csgmcmc PyTorch update) and equivalent to
SGHMC-jax ``make_sgld`` with ``v_hat = 0``; the SGFS-style gradient-noise
correction of Chen et al. (2014) figure 3 is exposed as the ``v_hat``
argument (it rescales the injected noise by ``1 - 0.5 * v_hat * step_size``,
folded into the temperature). RMSprop preconditioning reproduces pSGLD
(Li et al., 2016) and ift-sde's ``psgld_sampler_fast``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..base import Kernel, gaussian_like
from ..preconditioners import identity


class SGLDState(NamedTuple):
    position: jax.Array
    precond: tuple = ()


def sgld(preconditioner=None, v_hat=0.0):
    """Build an SGLD :class:`~samplax.base.Kernel`.

    ``step(key, state, grad, step_size, temperature=1.0)`` with ``grad`` the
    ascent log-density gradient at ``state.position``.
    """
    precond = identity() if preconditioner is None else preconditioner

    def init(key, position):
        del key
        return SGLDState(position, precond.init(position))

    def step(key, state, grad, step_size, temperature=1.0):
        pstate = precond.update(state.precond, grad)
        drift = precond.drift_scale(pstate, grad)
        nscale = precond.noise_scale(pstate, grad)
        noise = gaussian_like(key, state.position)
        # SGFS correction: known gradient-noise variance v_hat shrinks the
        # injected noise (Chen et al. 2014, figure 3 sgld.m)
        eff_temp = temperature * (1.0 - 0.5 * v_hat * step_size)
        scale = jnp.sqrt(2.0 * step_size * eff_temp)
        position = jax.tree_util.tree_map(
            lambda x, g, d, n, s: x + step_size * d * g + scale * s * n,
            state.position, grad, drift, noise, nscale)
        return SGLDState(position, pstate)

    return Kernel(init, step)
