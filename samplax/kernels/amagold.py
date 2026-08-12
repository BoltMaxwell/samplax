"""AMAGOLD: amortized Metropolis-adjusted SGHMC (Zhang, Cooper, De Sa, 2020).

Provenance: vendored from amagold-jax (verified against the original matlab
and PyTorch code, including the authors' repo-cached simulation samples).

AMAGOLD is not a per-step-gradient :class:`~samplax.base.Kernel`: its
amortized M-H correction needs energy evaluations and an inner leapfrog loop.
Two forms are provided:

- :func:`amagold` — the simulation form (momentum resampled each call,
  potential/gradient functions supplied at build time). ``step(key, x) ->
  (new_x, accepted)``.
- :func:`amagold_tracked` — the same sampler carrying ``u_fn(position)`` in
  its state, so the amortized M-H test costs one energy evaluation per step
  instead of two. Bit-identical chain; see its docstring.
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


def amagold(u_fn, grad_u, *, dt, nstep, C, mh=True, dt_friction=None):
    """Simulation-form AMAGOLD (amagold-jax ``samplers.amagold_kernel``).

    Semi-implicit friction leapfrog with beta = C/2, noise sqrt(2 dt_f C), and
    acceptance probability exp(U_old - U_new + rho) where rho accumulates
    the kinetic correction along the path.
    Returns ``step(key, x) -> (new_x, accepted)``.

    ``dt`` may be a scalar or a per-coordinate vector (shape of ``x``). A
    vector dt is the diagonal-mass-matrix form, and its exactness reduces to
    the scalar kernel's: with thermostat step ``dt_f`` (scalar) and
    S = diag(dt / dt_f), running THIS kernel on U at vector dt is identical,
    coordinate by coordinate, to running the scalar kernel at step dt_f on
    the rescaled potential ``U_tilde(z) = U(S z)`` (position/gradient/rho
    terms pick up one factor of S each; energies are invariant since
    U_tilde(z) = U(x)). The M-H test is therefore exact for any dt_f > 0.
    ``dt_friction`` sets that scalar thermostat step; default: ``dt`` itself
    when scalar, mean(dt) when vector. The friction/noise factors must stay
    scalar — they discretize the momentum OU process, whose semi-implicit
    update preserves N(0, I) exactly at any scalar step (that identity is
    what beta = C/2 encodes).

    Fix vs the original vendoring (2026-08): the thermostat noise is drawn
    per-coordinate (``jax.random.normal(k_noise, shape(p))``). The vendored
    port drew a SINGLE scalar normal broadcast across all coordinates, which
    correlates momenta within a trajectory and is only correct at d = 1 (the
    original simulation studies). Negligible at tiny dt, wrong in general;
    seeded results differ from pre-fix runs.
    """
    trajectory = _make_trajectory(grad_u, dt=dt, nstep=nstep, C=C,
                                  dt_friction=dt_friction)

    def step(key, x):
        key_p, key_steps, key_mh = jax.random.split(key, 3)
        old_x = x
        # Skipped when mh=False: nothing consumes it, and u_fn is pure.
        old_energy = u_fn(x) if mh else None
        x, rho = trajectory(key_p, key_steps, x)

        if not mh:
            return x, jnp.asarray(True)
        new_energy = u_fn(x)
        accept = jnp.exp(old_energy - new_energy + rho) >= jax.random.uniform(key_mh)
        return jnp.where(accept, x, old_x), accept

    return step


def _make_trajectory(grad_u, *, dt, nstep, C, dt_friction):
    """The semi-implicit friction leapfrog, shared by both simulation forms.

    Returns ``trajectory(key_p, key_steps, x) -> (proposed_x, rho)``. Splitting
    this out keeps :func:`amagold` and :func:`amagold_tracked` on exactly the
    same arithmetic and the same PRNG consumption order, which is what makes
    their chains bit-identical.
    """
    dt = jnp.asarray(dt)
    if dt_friction is None:
        dt_f = dt if dt.ndim == 0 else jnp.mean(dt)
    else:
        dt_f = jnp.asarray(dt_friction)
    sigma = jnp.sqrt(2.0 * dt_f * C)
    beta = 0.5 * C

    def trajectory(key_p, key_steps, x):
        p = jax.random.normal(key_p, jnp.shape(x))
        x = x + p * dt / 2.0

        def leapfrog(carry, inp):
            x, p, rho = carry
            i, subkey = inp
            k_grad, k_noise = jax.random.split(subkey)
            x = jnp.where(i > 0, x + p * dt, x)
            p_old = p
            grad_x = grad_u(k_grad, x)
            p = ((1.0 - dt_f * beta) * p - grad_x * dt
                 + jax.random.normal(k_noise, jnp.shape(p)) * sigma) / (1.0 + dt_f * beta)
            rho = rho + 0.5 * jnp.sum(grad_x * (p + p_old) * dt)
            return (x, p, rho), None

        (x, p, rho), _ = jax.lax.scan(
            leapfrog, (x, p, jnp.zeros(())),
            (jnp.arange(nstep), jax.random.split(key_steps, nstep)))
        return x + p * dt / 2.0, rho

    return trajectory


class AmagoldTrackedState(NamedTuple):
    position: jax.Array
    energy: jax.Array  # u_fn(position), carried so it is never recomputed


def amagold_tracked(u_fn, grad_u, *, dt, nstep, C, dt_friction=None):
    """:func:`amagold` with the current energy carried in the state.

    ``step(key, x)`` in the simulation form evaluates ``u_fn`` twice per call:
    once at the incoming position and once at the proposal. But the energy at
    the incoming position is always a value the previous step already computed
    -- on acceptance it is the proposal's energy, on rejection it is the one
    that was there before -- so half of those evaluations are redundant. This
    form threads it through the state instead:

        init, step = amagold_tracked(u_fn, grad_u, dt=..., nstep=..., C=...)
        state = init(x0)
        state, accepted = step(key, state)

    ``state.energy`` is always exactly ``u_fn(state.position)``, so callers
    that want the log-density for a trace can read it off rather than pay for
    a third evaluation.

    The chain is bit-identical to :func:`amagold` driven with the same keys:
    the saving is a cached float, not an approximation, and the shared
    :func:`_make_trajectory` guarantees identical arithmetic and PRNG order.
    ``tests/test_amagold_tracked.py`` pins that equality.

    This assumes ``u_fn`` is deterministic and unchanged across steps. That is
    not a new requirement -- AMAGOLD's M-H test compares energies from
    different steps, so a drifting ``u_fn`` would already break exactness --
    but it does mean this form cannot be used with an energy that varies per
    iteration (a tempering schedule, a moving normalizing-constant estimate).

    There is no ``mh`` switch: with ``mh=False`` there is no M-H test, nothing
    reads the energy, and tracking it would only add cost. Use :func:`amagold`
    for that case.
    """
    trajectory = _make_trajectory(grad_u, dt=dt, nstep=nstep, C=C,
                                  dt_friction=dt_friction)

    def init(position):
        return AmagoldTrackedState(position, u_fn(position))

    def step(key, state):
        key_p, key_steps, key_mh = jax.random.split(key, 3)
        new_x, rho = trajectory(key_p, key_steps, state.position)
        new_energy = u_fn(new_x)
        accept = jnp.exp(state.energy - new_energy + rho) >= jax.random.uniform(key_mh)
        return AmagoldTrackedState(
            jnp.where(accept, new_x, state.position),
            jnp.where(accept, new_energy, state.energy),
        ), accept

    return init, step


class AmagoldState(NamedTuple):
    position: jax.Array
    momentum: jax.Array
    energy: jax.Array  # energy_fn(position), carried so it is never recomputed


def amagold_minibatch(*, T, beta, step_size):
    """Minibatch AMAGOLD (amagold-jax ``bnn.train.make_amagold_outer``).

    Build once, then per outer iteration call

        state, accepted, rho = step(key, state, grad_fn, energy_fn, batches)

    where ``grad_fn(position, batch) -> pytree`` is the DESCENT gradient of
    the (sum-scale) potential on one minibatch, ``energy_fn(position) ->
    scalar`` the full-data potential used by the M-H test, and ``batches`` a
    pytree of stacked minibatches with leading axis T (the t = 0 entry is
    unused, matching the original). ``init(key, position, energy_fn)`` draws
    the persistent momentum ~ N(0, step_size) and seeds the tracked energy.

    Energy accounting: the M-H test needs the full-data potential at both ends
    of the trajectory, and the potential at the incoming position is always a
    value the previous outer step already computed -- the proposal's energy if
    it was accepted, the unchanged one if it was not. It is therefore carried
    in ``state.energy`` and only the proposal's is evaluated. This is the
    dominant saving for this form: the whole point of amortization here is
    that the O(N) energy is rare relative to the minibatch gradients, so
    halving it halves the part that was not amortized away. For a T=10
    trajectory on 60k MNIST images with batch 2000, the two full-data passes
    outweigh all nine minibatch gradient steps combined.

    As in :func:`amagold_tracked`, this assumes ``energy_fn`` is deterministic
    and the same function on every call -- which the M-H test already requires,
    since it compares energies computed at different outer steps. Do not use
    this form with a potential that drifts (tempering, a moving log-Z estimate,
    Gibbs-resampled hyperparameters entering the energy); recompute instead.
    """

    def init(key, position, energy_fn):
        buf = jax.tree_util.tree_map(
            lambda n: jnp.sqrt(step_size) * n, gaussian_like(key, position))
        return AmagoldState(position, buf, energy_fn(position))

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
        u_old = state.energy  # == energy_fn(old_position), from the previous step
        accept = jax.random.uniform(keys[T - 1]) <= jnp.exp(u_old - u_new + rho)
        position = jax.tree_util.tree_map(
            lambda new, old: jnp.where(accept, new, old), position, old_position)
        buf = jax.tree_util.tree_map(
            lambda new, init_: jnp.where(accept, new, -init_), buf, buf_init)
        return AmagoldState(position, buf, jnp.where(accept, u_new, u_old)), accept, rho

    return init, step
