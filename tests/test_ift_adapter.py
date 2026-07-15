"""Tests for the ift-sde Family-B adapter (samplax.integrations.ift_sde).

x64 is enabled to match ift-sde's own workflow (jax_enable_x64), like
tests/test_schedules_exponential.py does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.integrations import ift_sde


def test_x0_reaches_likelihood():
    """log_likelihood_fn(w, theta, x0) — x0 must actually feed the likelihood."""
    d_w, d_theta, d_x0 = 4, 1, 1

    def log_likelihood_fn(w, theta, x0):
        return -0.5 * ((x0[0] - 1.7) / 0.1) ** 2 + 0.0 * theta[0] + 0.0 * jnp.sum(w)

    def energy_fn(w, theta, x0):
        return 0.5 * jnp.sum(w**2)

    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=1e-3,
        x0_prior_std=5.0, iterations=4000, burn_in=1000, thinning=10,
        chains=2,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg)
    x0_mean = res.samples["x0_samples"].reshape(-1).mean()
    assert abs(float(x0_mean) - 1.7) < 0.2


def test_init_mean_respected():
    d_w, d_theta, d_x0 = 2, 1, 1
    d_z = d_w + d_theta + d_x0

    def log_likelihood_fn(w, theta, x0):
        return 0.0 * jnp.sum(w) + 0.0 * theta[0] + 0.0 * x0[0]

    def energy_fn(w, theta, x0):
        return 0.0

    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=1e-12,
        init_mean=(9.0,) * d_z, init_std=0.01,
        iterations=10, burn_in=0, thinning=10, chains=2,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg)
    z = res.samples["z_samples"]
    assert z.shape[1] == 1  # one kept chunk
    assert np.all(np.abs(z - 9.0) < 0.1)

    bad_cfg = ift_sde.SGMCMCConfig(init_mean=(9.0,) * (d_z - 1))
    with pytest.raises(ValueError):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                           d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                           energy_fn=energy_fn, config=bad_cfg)


def test_exponential_schedule_wired():
    d_w, d_theta, d_x0 = 2, 1, 1

    def log_likelihood_fn(w, theta, x0):
        return -0.5 * jnp.sum(w**2) - 0.5 * theta[0] ** 2 - 0.5 * x0[0] ** 2

    def energy_fn(w, theta, x0):
        return 0.0

    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="exponential", step_size=1e-3,
        step_size_final=1e-6, iterations=200, burn_in=50, thinning=10, chains=2,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg)
    assert np.all(np.isfinite(res.samples["z_samples"]))

    bad_cfg = ift_sde.SGMCMCConfig(schedule="exponential", step_size_final=None)
    with pytest.raises(ValueError):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                           d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                           energy_fn=energy_fn, config=bad_cfg)


def test_sanitization_keeps_chain_finite():
    """Gradient is NaN outside |z|<2; the chain must never latch onto NaN/inf."""
    d_w, d_theta, d_x0 = 1, 1, 1

    def log_likelihood_fn(w, theta, x0):
        z = jnp.concatenate([w, theta, x0])
        blowup = jnp.where(jnp.sum(z**2) > 4.0, jnp.nan, 0.0)
        return -0.5 * jnp.sum(z**2) + blowup

    def energy_fn(w, theta, x0):
        return 0.0

    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=0.5,
        theta_prior_std=1e6, x0_prior_std=1e6,
        iterations=2000, burn_in=200, thinning=10, chains=4,
        grad_clip=1e3, state_clip=1e6,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(1), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg)
    assert np.all(np.isfinite(res.samples["z_samples"]))


def test_correction_shifts_target():
    """Stationary dist of SGLD with drift grad log p + c is N(c, 1) for p = N(0,1)."""
    d_w, d_theta, d_x0 = 2, 0, 0

    def log_likelihood_fn(w, theta, x0):
        return -0.5 * jnp.sum(w**2)

    def energy_fn(w, theta, x0):
        return 0.0

    correction = ift_sde.Correction(
        init=lambda key, z0: (),
        step=lambda key, z, c: (jnp.full_like(z, 1.5), c),
    )

    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=5e-3,
        theta_prior_std=1.0, x0_prior_std=1.0,
        iterations=20_000, burn_in=10_000, thinning=10, chains=4,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(2), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg, correction=correction)
    w_mean = res.samples["w_samples"].reshape(-1, d_w).mean(axis=0)
    assert np.all(np.abs(w_mean - 1.5) < 0.3)


def test_correction_state_threads():
    d_w, d_theta, d_x0 = 1, 0, 0

    def log_likelihood_fn(w, theta, x0):
        return 0.0 * jnp.sum(w)

    def energy_fn(w, theta, x0):
        return 0.0

    def cstep(key, z, c):
        return jnp.zeros_like(z), c + 1

    correction = ift_sde.Correction(
        init=lambda key, z0: jnp.asarray(0, dtype=jnp.int32),
        step=cstep,
    )

    iterations = 37
    cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=1e-4,
        iterations=iterations, burn_in=0, thinning=iterations, chains=3,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(3), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=log_likelihood_fn,
                             energy_fn=energy_fn, config=cfg, correction=correction)
    counts = np.asarray(res.final_state["correction"])
    assert counts.shape == (cfg.chains,)
    assert np.all(counts == iterations)
