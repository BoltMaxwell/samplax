"""Persistent-auxiliary-chain (PCD-style) log-Z correction for ``ift_sde``.

Persistent Contrastive Divergence (Tieleman, 2008) trains an energy-based
model by keeping its "negative phase" Markov chain alive across parameter
updates instead of restarting it from scratch each step: because successive
parameter updates are small relative to the chain's own mixing time, the
chain stays approximately equilibrated at the *current* parameters as they
drift. :func:`nested_correction` applies the same idea to
:mod:`samplax.integrations.ift_sde`'s outer ``(theta, x0)`` chain: a second,
persistent auxiliary chain samples

    p(w_tilde | theta, x0) is proportional to exp(-energy(w_tilde, theta, x0))

warm-started from its own previous position at every outer step (never
reinitialized — that persistence is exactly what :class:`NestedState` and
the ``Correction`` protocol's threaded ``cstate`` exist to carry), and its
current draw estimates the intractable log-partition-function gradient that
the outer chain's own drift is missing.

Correction-drift identity
--------------------------
Fix ``(theta, x0)`` and let ``Z(theta, x0) = integral exp(-energy(w, theta,
x0)) dw``. Differentiating under the integral sign,

    grad_{theta,x0} log Z(theta, x0)
        = -E_{w ~ p(.|theta,x0)}[grad_{theta,x0} energy(w, theta, x0)].

So a (approximately) equilibrated draw ``w_tilde`` from the persistent aux
chain gives an estimator of ``-grad log Z`` that is unbiased in expectation
over the aux chain's stationary distribution, via
``+grad_{theta,x0} energy(w_tilde, theta, x0)`` — exactly the quantity this
module returns as the additive correction on the outer drift (the w-block
contribution is exactly zero: the correction only ever touches
``(theta, x0)``).

Staleness and the re-warm policy
---------------------------------
PCD's approximation degrades when the *outer* ``(theta, x0)`` moves fast
relative to the aux chain's own mixing time: the aux chain then estimates
``grad log Z`` at a ``(theta, x0)`` that is no longer close to the current
one, biasing the correction (classic PCD "chain lag" failure mode).
``rewarm_threshold`` guards against this: when the L2 displacement of
``(theta, x0)`` since the previous outer step exceeds the threshold,
``rewarm_iterations`` extra aux-chain steps run first, at the *current*
``(theta, x0)``, before the regular ``aux_iterations`` refinement — a cheap
partial re-equilibration. ``rewarm_threshold=None`` disables the guard
entirely, and the corresponding ``lax.cond`` branch is never traced (a
plain Python ``if`` at build time), so disabled re-warm costs nothing.

Provenance: this is ift-origin integration code — it mirrors the persistent
aux-chain construction of ift-sde's ``methods/npsgld/npsgld.py``
(``run_aux_chain`` / ``log_posterior_surrogate_args``) but is newly written
against samplax's ``Kernel`` / ``Correction`` protocols, not a vendored
kernel port.
"""

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from .ift_sde import Correction
from ..kernels.sgld import sgld
from ..preconditioners import rmsprop
from ..schedules import constant


class NestedState(NamedTuple):
    """Persistent correction state for one outer chain.

    ``aux_state`` is the aux :class:`~samplax.base.Kernel` state (position
    ``w_tilde``, shape ``(d_w,)``, plus any preconditioner accumulator),
    carried and warm-started across outer steps. ``t`` is the outer-step
    counter (int32), used to index ``aux_schedule``. ``prev_params`` is the
    ``(theta, x0)`` vector from the previous outer step, used by the
    re-warm trigger; it is always carried (even with re-warm disabled) so
    the pytree structure of ``NestedState`` is static regardless of
    ``rewarm_threshold``.
    """

    aux_state: object
    t: jax.Array
    prev_params: jax.Array


def _sanitize_grad(g, clip):
    return jnp.clip(jnp.nan_to_num(g, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def nested_correction(
    energy_fn,
    d_w,
    d_theta,
    d_x0,
    *,
    kernel=None,
    aux_iterations=5,
    aux_schedule=None,
    aux_step_size=2e-4,
    grad_clip=1e3,
    rewarm_threshold=None,
    rewarm_iterations=0,
) -> Correction:
    """Build a persistent-PCD-style :class:`~samplax.integrations.ift_sde.Correction`.

    ``energy_fn(w, theta, x0) -> scalar`` follows ift-sde's three-argument
    contract. ``d_w``/``d_theta``/``d_x0`` describe the outer chain's flat
    layout ``z = [w, theta, x0]`` (``d_x0`` is kept only for API symmetry
    and documentation — slicing only ever needs ``d_w`` and ``d_theta``,
    the rest of ``z`` after ``theta`` is ``x0`` by construction).

    ``kernel`` drives the aux chain (default: ``sgld(preconditioner=
    rmsprop())``, matching ift-sde's NPSGLD aux-chain preconditioning).
    ``aux_schedule`` (default: ``constant(aux_step_size)``) supplies the aux
    step size per outer step via ``aux_schedule(t).step_size``; the aux
    chain always samples (temperature fixed at 1.0), regardless of
    ``do_sample`` in the returned :class:`~samplax.schedules.ScheduleState`.
    ``grad_clip`` sanitizes the aux-chain's own gradient
    (``-grad_w energy``) elementwise, matching npsgld's
    ``grad_conditional_path``; the returned correction gradient itself is
    left unsanitized here (``run_sgmcmc`` sanitizes the combined drift as a
    unit after adding it to the main gradient).
    """

    kern = kernel if kernel is not None else sgld(preconditioner=rmsprop())
    sched = aux_schedule if aux_schedule is not None else constant(aux_step_size)

    def _energy_of_w(w, params):
        theta = params[:d_theta]
        x0 = params[d_theta:]
        return energy_fn(w, theta, x0)

    def _aux_grad(w, params):
        g = jax.grad(lambda w_: -_energy_of_w(w_, params))(w)
        return _sanitize_grad(g, grad_clip)

    def init(key, z0):
        w0 = z0[:d_w]
        aux_state = kern.init(key, w0)
        return NestedState(aux_state, jnp.asarray(0, dtype=jnp.int32), z0[d_w:])

    def step(key, z, cstate):
        aux_state, t, prev_params = cstate
        params = z[d_w:]
        step_size = sched(t).step_size

        def aux_kernel_step(k, s):
            g = _aux_grad(s.position, params)
            return kern.step(k, s, g, step_size, 1.0)

        if rewarm_threshold is not None:
            delta = params - prev_params
            do_rewarm = jnp.sqrt(jnp.sum(delta * delta)) > rewarm_threshold
            rewarm_key = jax.random.fold_in(key, 0)

            def run_rewarm(s):
                keys = jax.random.split(rewarm_key, rewarm_iterations)

                def body(carry, k):
                    return aux_kernel_step(k, carry), None

                s_out, _ = jax.lax.scan(body, s, keys)
                return s_out

            aux_state = jax.lax.cond(do_rewarm, run_rewarm, lambda s: s, aux_state)

        keys_main = jax.random.split(key, aux_iterations)

        def body_main(carry, k):
            return aux_kernel_step(k, carry), None

        aux_state, _ = jax.lax.scan(body_main, aux_state, keys_main)

        w_tilde = jax.lax.stop_gradient(aux_state.position)

        def _energy_of_params(p):
            return energy_fn(w_tilde, p[:d_theta], p[d_theta:])

        g_params = jax.grad(_energy_of_params)(params)
        g_corr = jnp.concatenate([jnp.zeros((d_w,), dtype=z.dtype), g_params])

        new_cstate = NestedState(aux_state, t + 1, params)
        return g_corr, new_cstate

    return Correction(init=init, step=step)
