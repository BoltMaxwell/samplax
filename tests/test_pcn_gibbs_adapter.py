"""pcn-gibbs branch of run_sgmcmc: target correctness, guards, energy probe."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.integrations import ift_sde


def _wprior_energy(w, theta, x0):
    return 0.5 * jnp.sum(w**2)


def test_recovers_gaussian_target_with_theta_block():
    """w likelihood N(0,1)-shaped, theta only via its prior N(0, 3^2):
    both blocks must recover their scales, cross-checked marginally."""
    d_w, d_theta, d_x0 = 3, 2, 0

    def loglik(w, theta, x0):
        return -0.5 * jnp.sum(w**2)  # posterior w ~ N(0, 1/2) marginally... no:
        # target = exp(loglik) * N(w;0,1) => w ~ N(0, 1/sqrt(2)^2? precision 2)

    cfg = ift_sde.SGMCMCConfig(
        kernel="pcn-gibbs", schedule="constant", preconditioner="identity",
        pcn_beta=0.4, pcn_n_pcn=3, pcn_n_mh=3, pcn_mh_scale=1.5,
        theta_prior_std=3.0,
        iterations=30_000, burn_in=6_000, thinning=10, chains=4,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(0), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=loglik,
                             energy_fn=_wprior_energy, config=cfg)
    w = res.samples["w_samples"].reshape(-1, d_w)
    th = res.samples["theta_samples"].reshape(-1, d_theta)
    # w: exp(-w^2/2)*N(0,1) => precision 2 => std 1/sqrt(2)
    np.testing.assert_allclose(w.std(axis=0), 1.0 / np.sqrt(2.0), rtol=0.08)
    np.testing.assert_allclose(th.std(axis=0), 3.0, rtol=0.12)
    assert np.all(np.abs(w.mean(axis=0)) < 0.06)
    assert np.all(np.abs(th.mean(axis=0)) < 0.35)
    aw = res.final_state["accept_rate_w"]
    ar = res.final_state["accept_rate_theta"]
    assert aw.shape == (cfg.chains,) and ar.shape == (cfg.chains,)
    assert 0.05 < float(np.mean(aw)) < 1.0
    assert 0.05 < float(np.mean(ar)) < 1.0
    assert len(res.history["accept_w"]) == cfg.iterations // cfg.thinning


def test_energy_probe_rejects_non_whitened():
    def bad_energy(w, theta, x0):
        return 0.5 * jnp.sum(w**2) * (1.0 + 0.1 * jnp.sum(theta**2))

    cfg = ift_sde.SGMCMCConfig(kernel="pcn-gibbs", schedule="constant",
                               preconditioner="identity",
                               iterations=100, burn_in=0, thinning=10, chains=1)
    with pytest.raises(ValueError, match="whitened"):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=3, d_theta=2, d_x0=0,
                           log_likelihood_fn=lambda w, t, x: 0.0,
                           energy_fn=bad_energy, config=cfg)


def test_preconditioner_and_scales_guards():
    cfg = ift_sde.SGMCMCConfig(kernel="pcn-gibbs", schedule="constant",
                               preconditioner="rmsprop",
                               iterations=100, burn_in=0, thinning=10, chains=1)
    with pytest.raises(ValueError, match="no preconditioner"):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=2, d_theta=1, d_x0=0,
                           log_likelihood_fn=lambda w, t, x: 0.0,
                           energy_fn=_wprior_energy, config=cfg)

    cfg2 = ift_sde.SGMCMCConfig(kernel="pcn-gibbs", schedule="constant",
                                preconditioner="identity",
                                pcn_mh_scales=(0.1, 0.1, 0.1),  # wrong length
                                iterations=100, burn_in=0, thinning=10, chains=1)
    with pytest.raises(ValueError, match="pcn_mh_scales"):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=2, d_theta=1, d_x0=0,
                           log_likelihood_fn=lambda w, t, x: 0.0,
                           energy_fn=_wprior_energy, config=cfg2)
