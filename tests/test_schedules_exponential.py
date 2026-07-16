"""Tests for exponential step-size schedule.

x64 is enabled to pin the reference (ift-sde NPSGLD) values at rtol 1e-12;
float32 already deviates at ~1e-8.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import samplax


def test_exponential_start():
    """At step 0, returns the initial step size."""
    sched = samplax.exponential(3e-4, 1e-6, 1_000_000)
    s0 = sched(0)
    np.testing.assert_allclose(float(s0.step_size), 3e-4, rtol=1e-6)
    assert bool(s0.do_sample)


def test_exponential_end():
    """At final step, returns the final step size."""
    sched = samplax.exponential(3e-4, 1e-6, 1_000_000)
    s_end = sched(999_999)
    np.testing.assert_allclose(float(s_end.step_size), 1e-6, rtol=1e-6)


def test_exponential_mid():
    """At midpoint, returns the geometrically interpolated value."""
    step_size_init = 3e-4
    step_size_final = 1e-6
    num_training_steps = 1_000_000
    mid_step = 499_999

    # Compute expected value using math
    ratio = mid_step / (num_training_steps - 1)
    expected = step_size_init * math.exp(math.log(step_size_final / step_size_init) * ratio)

    sched = samplax.exponential(step_size_init, step_size_final, num_training_steps)
    s_mid = sched(mid_step)
    np.testing.assert_allclose(float(s_mid.step_size), expected, rtol=1e-12)


def test_exponential_beyond_end():
    """Beyond the final step, clipped to final step size."""
    sched = samplax.exponential(3e-4, 1e-6, 1_000_000)
    s_beyond = sched(2_000_000)
    np.testing.assert_allclose(float(s_beyond.step_size), 1e-6, rtol=1e-6)


def test_exponential_num_training_steps_one():
    """With num_training_steps <= 1, always returns final step size."""
    sched = samplax.exponential(1e-3, 1e-5, 1)
    np.testing.assert_allclose(float(sched(0).step_size), 1e-5, rtol=1e-6)
    np.testing.assert_allclose(float(sched(10).step_size), 1e-5, rtol=1e-6)


def test_exponential_negative_step_size_raises():
    """Negative initial step size raises ValueError."""
    with pytest.raises(ValueError):
        samplax.exponential(-1e-3, 1e-5, 10)


def test_exponential_zero_final_step_size_raises():
    """Zero or negative final step size raises ValueError."""
    with pytest.raises(ValueError):
        samplax.exponential(1e-3, 0.0, 10)


def test_exponential_jit_friendly():
    """Exponential schedule is jit-friendly (works with traced step_id)."""
    sched = samplax.exponential(3e-4, 1e-6, 100)

    # JIT-compiled version
    jitted_sched = jax.jit(lambda t: sched(t).step_size)

    # Call with jax array
    step_id_jax = jnp.asarray(50)
    result_jitted = float(jitted_sched(step_id_jax))

    # Call without JIT for comparison
    result_unjitted = float(sched(50).step_size)

    np.testing.assert_allclose(result_jitted, result_unjitted, rtol=1e-6)


def test_exponential_do_sample_always_true():
    """do_sample is always True for exponential schedule."""
    sched = samplax.exponential(3e-4, 1e-6, 1_000_000)
    assert bool(sched(0).do_sample)
    assert bool(sched(500_000).do_sample)
    assert bool(sched(999_999).do_sample)
