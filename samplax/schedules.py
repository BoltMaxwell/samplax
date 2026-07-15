"""Step-size / temperature schedules. All are jit-friendly: ``fn(step_id)``
accepts Python ints or traced jax integers and returns a :class:`ScheduleState`.

``cyclical`` is vendored from csgmcmc-jax (cSG-MCMC, Zhang et al. 2020): a
cosine-annealed step size per cycle whose first ``exploration_ratio`` fraction
is the exploration (noise-free) phase — combine with any kernel by passing
``temperature = temperature * state.do_sample`` to get cSGLD/cSGHMC/etc.
``polynomial`` is the classic a (b + t)^-gamma SGLD decay (also the ift-sde
``lr_package`` form). ``exponential`` is the geometric interpolation schedule
matching ift-sde's NPSGLD exponential decay.
"""

from typing import NamedTuple

import jax.numpy as jnp


class ScheduleState(NamedTuple):
    step_size: jnp.ndarray
    do_sample: jnp.ndarray  # False during cyclical exploration phases


def constant(step_size):
    def schedule_fn(step_id):
        del step_id
        return ScheduleState(jnp.asarray(step_size), jnp.asarray(True))

    return schedule_fn


def polynomial(a, b, gamma):
    """step_size(t) = a * (b + t)^(-gamma)."""

    def schedule_fn(step_id):
        return ScheduleState(a * (b + step_id) ** (-gamma), jnp.asarray(True))

    return schedule_fn


def exponential(step_size, step_size_final, num_training_steps):
    """Geometric interpolation: step_size → step_size_final over the run.

    Matches ift-sde NPSGLD exponential decay (step_size * exp(log(final/start) * ratio)).
    Both step sizes must be positive; num_training_steps >= 1.
    """
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if step_size_final <= 0.0:
        raise ValueError(f"step_size_final must be positive, got {step_size_final}")
    if num_training_steps < 1:
        raise ValueError(f"num_training_steps must be >= 1, got {num_training_steps}")

    def schedule_fn(step_id):
        if num_training_steps <= 1:
            return ScheduleState(jnp.asarray(step_size_final), jnp.asarray(True))
        ratio = jnp.clip(step_id / (num_training_steps - 1), 0.0, 1.0)
        lr = step_size * jnp.exp(jnp.log(step_size_final / step_size) * ratio)
        return ScheduleState(lr, jnp.asarray(True))

    return schedule_fn


def cyclical(num_training_steps, num_cycles=4, initial_step_size=1e-3,
             exploration_ratio=0.25):
    """Cyclical cosine schedule (Zhang et al., 2020), from csgmcmc-jax."""
    cycle_length = num_training_steps // num_cycles

    def schedule_fn(step_id):
        phase = (step_id % cycle_length) / cycle_length
        do_sample = phase >= exploration_ratio
        step_size = 0.5 * (jnp.cos(jnp.pi * phase) + 1.0) * initial_step_size
        return ScheduleState(step_size, do_sample)

    return schedule_fn
