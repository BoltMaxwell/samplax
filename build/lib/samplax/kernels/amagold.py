"""AMAGOLD: amortized Metropolis-adjusted SGHMC (Zhang, Cooper, De Sa, 2020).

Provenance: vendored from amagold-jax (verified against the original matlab
and PyTorch code, including the authors' repo-cached simulation samples).

AMAGOLD is not a per-step-gradient :class:`~samplax.base.Kernel`: its
amortized M-H correction needs energy evaluations and an inner leapfrog loop.
Two forms are provided:

- :func:`amagold` — the simulation form (momentum resampled each call,
  potential/gradient functions supplied at build time). ``step(key, x) ->
  (new_x, accepted)``.
- :func:`amagold_minibatch` — the BNN form: persistent momentum (negated on
  rejection), T-1 minibatch gradient steps with half position steps at both
  ends, rho accumulation, and an M-H test against a full-data energy
  difference. ``outer(key, position, momentum, grads_batches..., energy_fn)``
  is exposed as ``step(key, state, batches, grad_fn, energy_fn, step_size)``.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..base import gaussian_like


def amagold(u_fn, grad_u, *, dt, nstep, C, mh=True):
    """Simulation-form AMAGOLD (amagold-jax ``samplers.amagold_kernel``).

    Semi-implicit friction leapfrog with beta = C/2, noise sqrt(2 dt C), and
    acceptance probability exp(U_old - U_new + rho) where rho accumulates
    the kinetic correction along the path.
    Returns ``step(key, x) -> (new_x, accepted)``.
    """
    sigma = jnp.sqrt(2.0 * dt * C)
    beta = 0.5 * C

    def step(key, x):
        key_p, key_steps, key_mh = jax.random.split(key, 3)
        p = jax.random.normal(key_p, jnp.shape(x))
        old_x = x
        old_energy = u_fn(x)
        x = x + p * dt / 2.0

        def leapfrog(carry, inp):
            x, p, rho = carry
            i, subkey = inp
            k_grad, k_noise = jax.random.split(subkey)
            x = jnp.where(i > 0, x + p * dt, x)
            p_old = p
            grad_x = grad_u(k_grad, x)
            p = ((1.0 - dt * beta) * p - grad_x * dt
                 + jax.random.normal(k_noise) * sigma) / (1.0 + dt * beta)
            rho = rho + grad_x * (p + p_old) * dt / 2.0
            return (x, p, rho), None

        (x, p, rho), _ = jax.lax.scan(
            leapfrog, (x, p, jnp.zeros_like(old_energy)),
            (jnp.arange(nstep), jax.random.split(key_steps, nstep)))
        x = x + p * dt / 2.0

        if not mh:
            return x, jnp.asarray(True)
        new_energy = u_fn(x)
        accept = jnp.exp(old_energy - new_energy + rho) >= jax.random.uniform(key_mh)
        return jnp.where(accept, x, old_x), accept

    return step


class AmagoldState(NamedTuple):
    position: jax.Array
    momentum: jax.Array


def amagold_minibatch(*, T, beta, step_size):
    """Minibatch AMAGOLD (amagold-jax ``bnn.train.make_amagold_outer``).

    Build once, then per outer iteration call

        state, accepted, rho = step(key, state, grad_fn, energy_fn, batches)

    where ``grad_fn(position, batch) -> pytree`` is the DESCENT gradient of
    the (sum-scale) potential on one minibatch, ``energy_fn(position) ->
    scalar`` the full-data potential used by the M-H test, and ``batches`` a
    pytree of stacked minibatches with leading axis T (the t = 0 entry is
    unused, matching the original). ``init(key, position)`` draws the
    persistent momentum ~ N(0, step_size).
    """

    def init(key, position):
        buf = jax.tree_util.tree_map(
            lambda n: jnp.sqrt(step_size) * n, gaussian_like(key, position))
        return AmagoldState(position, buf)

    def step(key, state, grad_fn, energy_fn, batches):
        old_position = state.position
        buf_init = state.momentum
        position = jax.tree_util.tree_map(
            lambda p, b: p + 0.5 * b, state.position, state.momentum)

        def leapfrog(carry, inp):
            position, buf, rho = carry
            t, batch, subkey = inp
            d_p = grad_fn(position, batch)
            noise = gaussian_like(subkey, buf)
            buf_new = jax.tree_util.tree_map(
                lambda b, g, n: ((1.0 - beta) * b - step_size * g
                                 + (step_size * beta) ** 0.5 * 2.0 * n) / (1.0 + beta),
                buf, d_p, noise)
            rho = rho + 0.5 * sum(
                jnp.sum(a * (b + c)) for a, b, c in zip(
                    jax.tree_util.tree_leaves(d_p),
                    jax.tree_util.tree_leaves(buf),
                    jax.tree_util.tree_leaves(buf_new)))
            scale = jnp.where(t == T - 1, 0.5, 1.0)
            position = jax.tree_util.tree_map(
                lambda p, b: p + scale * b, position, buf_new)
            return (position, buf_new, rho), None

        ts = jnp.arange(1, T)
        keys = jax.random.split(key, T)
        sub_batches = jax.tree_util.tree_map(lambda b: b[1:], batches)
        (position, buf, rho), _ = jax.lax.scan(
            leapfrog, (position, state.momentum, jnp.zeros(())),
            (ts, sub_batches, keys[: T - 1]))

        u_new = energy_fn(position)
        u_old = energy_fn(old_position)
        accept = jax.random.uniform(keys[T - 1]) <= jnp.exp(u_old - u_new + rho)
        position = jax.tree_util.tree_map(
            lambda new, old: jnp.where(accept, new, old), position, old_position)
        buf = jax.tree_util.tree_map(
            lambda new, init_: jnp.where(accept, new, -init_), buf, buf_init)
        return AmagoldState(position, buf), accept, rho

    return init, step
