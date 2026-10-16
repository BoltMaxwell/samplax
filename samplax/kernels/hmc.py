"""Leapfrog HMC with optional Metropolis-Hastings test (a full-gradient reference).

Provenance: vendored from SGHMC-jax ``samplers/jax_backend.py:hmc_kernel``
(verified against matlab/figure1/hmc.m from the ML-SGHMC repo). Not a
:class:`~samplax.base.Kernel` — it needs the potential ``u_fn`` and a gradient
*function* (evaluated at intermediate leapfrog points), not a per-step
gradient. Useful as an exact baseline in the simulated experiments.
"""

import jax
import jax.numpy as jnp


def hmc(u_fn, grad_u, *, dt, nstep, m=1.0, mh=True):
    """Returns ``step(key, x) -> new_x``. ``grad_u(key, x)`` is the (possibly
    stochastic) gradient of the potential U (descent direction)."""

    def step(key, x):
        key_p, key_grad, key_mh = jax.random.split(key, 3)
        p = jax.random.normal(key_p, jnp.shape(x)) * jnp.sqrt(m)
        old_energy = jnp.sum(p * m * p) / 2.0 + u_fn(x)
        old_x = x

        def leapfrog(carry, subkey):
            x, p = carry
            k1, k2 = jax.random.split(subkey)
            p = p - grad_u(k1, x) * dt / 2.0
            x = x + p / m * dt
            p = p - grad_u(k2, x) * dt / 2.0
            return (x, p), None

        (x, p), _ = jax.lax.scan(leapfrog, (x, p), jax.random.split(key_grad, nstep))
        p = -p
        if not mh:
            return x
        new_energy = jnp.sum(p * m * p) / 2.0 + u_fn(x)
        accept = jnp.exp(old_energy - new_energy) >= jax.random.uniform(key_mh)
        return jnp.where(accept, x, old_x)

    return step
