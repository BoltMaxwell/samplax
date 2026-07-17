"""Tests for the AMAGOLD driver branch of run_sgmcmc (samplax.integrations.ift_sde).

x64 is enabled to match ift-sde's own workflow (jax_enable_x64), like
tests/test_ift_adapter.py does.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.integrations import ift_sde


def _std_normal_loglik(w, theta, x0):
    return -0.5 * jnp.sum(w**2)


def _zero_energy(w, theta, x0):
    return 0.0


def test_amagold_unbiased_where_sgld_biased():
    """AMAGOLD's M-H correction should keep a N(0,1) target unbiased at a
    step size that is deliberately too large for plain SGLD.

    d_theta = d_x0 = 0 (supported by the adapter; see test_ift_adapter.py's
    test_correction_shifts_target for precedent).
    """
    d_w, d_theta, d_x0 = 3, 0, 0

    amagold_cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", schedule="constant",
        amagold_dt=0.3, amagold_nstep=5, amagold_C=1.0,
        iterations=20_000, burn_in=5_000, thinning=10, chains=4,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                             energy_fn=_zero_energy, config=amagold_cfg)
    w = res.samples["w_samples"].reshape(-1, d_w)
    assert np.all(np.isfinite(w))
    mean = w.mean(axis=0)
    std = w.std(axis=0)
    assert np.all(np.abs(mean) < 0.15)
    assert np.all((std > 0.85) & (std < 1.15))

    # AMAGOLD's outer step is nstep=5 leapfrog substeps of length dt=0.3
    # each, i.e. a trajectory length of ~1.5; match that displacement scale
    # with a plain SGLD step_size of 1.5. SGLD's (uncorrected) discretization
    # at this step size is analytically biased (Euler-Maruyama stationary
    # variance 2/(2-step_size) = 4 here) -- confirmed empirically to inflate
    # std to ~2.0, well above the N(0,1) target. Either an inflated std or a
    # non-finite chain demonstrates the M-H correction's value over plain
    # SGLD.
    sgld_cfg = ift_sde.SGMCMCConfig(
        kernel="sgld", schedule="constant", step_size=1.5,
        iterations=20_000, burn_in=5_000, thinning=10, chains=4,
    )
    sgld_res = ift_sde.run_sgmcmc(jax.random.key(1), d_w=d_w, d_theta=d_theta,
                                  d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                                  energy_fn=_zero_energy, config=sgld_cfg)
    sgld_w = sgld_res.samples["w_samples"].reshape(-1, d_w)
    if np.all(np.isfinite(sgld_w)):
        sgld_std = sgld_w.std(axis=0)
        assert np.any(sgld_std > 1.2)
    else:
        # sgld blew up at this step size; amagold (asserted finite above)
        # still wins.
        assert np.all(np.isfinite(w))


def test_amagold_acceptance_in_band():
    d_w, d_theta, d_x0 = 2, 0, 0

    # Swept down from 0.3: at dt<=0.4 acceptance stays pinned near 1.0 (the
    # 5-substep leapfrog trajectory is too short relative to the target's
    # curvature to accumulate rejectable error), and only crosses below 0.9
    # around dt~1.0-1.6. dt=1.3 lands comfortably inside [0.15, 0.9]
    # (empirically ~0.75 mean acceptance across chains).
    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", schedule="constant",
        amagold_dt=1.3, amagold_nstep=5, amagold_C=1.0,
        iterations=5_000, burn_in=1_000, thinning=10, chains=4,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(2), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                             energy_fn=_zero_energy, config=cfg)
    accept_rate = res.final_state["accept_rate"]
    assert accept_rate.shape == (cfg.chains,)
    mean_accept = float(np.mean(accept_rate))
    assert 0.15 <= mean_accept <= 0.9
    assert len(res.history["accept_rate"]) == cfg.iterations // cfg.thinning


def test_amagold_rejects_correction():
    d_w, d_theta, d_x0 = 2, 0, 0
    correction = ift_sde.Correction(
        init=lambda key, z0: (),
        step=lambda key, z, c: (jnp.zeros_like(z), c),
    )
    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", amagold_dt=0.1, iterations=10, burn_in=0,
        thinning=10, chains=2,
    )
    with pytest.raises(ValueError):
        ift_sde.run_sgmcmc(jax.random.key(3), d_w=d_w, d_theta=d_theta,
                           d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                           energy_fn=_zero_energy, config=cfg,
                           correction=correction)


def test_amagold_requires_dt():
    d_w, d_theta, d_x0 = 2, 0, 0
    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", amagold_dt=None, iterations=10, burn_in=0,
        thinning=10, chains=2,
    )
    with pytest.raises(ValueError):
        ift_sde.run_sgmcmc(jax.random.key(4), d_w=d_w, d_theta=d_theta,
                           d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                           energy_fn=_zero_energy, config=cfg)


def test_amagold_init_mean_respected():
    d_w, d_theta, d_x0 = 2, 0, 0
    d_z = d_w + d_theta + d_x0

    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", schedule="constant",
        amagold_dt=1e-6, amagold_nstep=2, amagold_C=1.0,
        init_mean=(9.0,) * d_z, init_std=0.01,
        iterations=10, burn_in=0, thinning=10, chains=2,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(5), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=_std_normal_loglik,
                             energy_fn=_zero_energy, config=cfg)
    z = res.samples["z_samples"]
    assert z.shape[1] == 1  # one kept chunk
    assert np.all(np.abs(z - 9.0) < 0.1)
