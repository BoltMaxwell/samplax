"""``run_sgmcmc(..., stochastic_grad_fn=...)``: AMAGOLD with a real stochastic gradient.

AMAGOLD's point (Zhang, Cooper & De Sa, AISTATS 2020) is that the leapfrog may
use *stochastic* gradients while the amortized M-H test on the FULL energy keeps
the chain exact. The kernel has always supported it (``grad_u(key, x)`` gets a
fresh key per leapfrog step), but the ift-sde adapter's ``grad_u`` ignored its
key and returned the full gradient, so no run ever exercised it.

Two properties are pinned here:

1. ``stochastic_grad_fn=None`` (the default) leaves the chain bit-identical to
   what it was, and passing a *full*-gradient callable through the new seam
   reproduces it exactly -- the option cannot perturb existing results.
2. The M-H test keeps using the full energy: with a deliberately WRONG (biased)
   gradient the chain still samples the full-energy target. That is the
   exactness claim, and it is what makes minibatch AMAGOLD worth running.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.integrations.ift_sde import SGMCMCConfig, run_sgmcmc


def _gaussian_target(d=1):
    """Standard normal in d dims, expressed in the adapter's three-arg contract."""

    def log_likelihood_fn(w, theta, x0):
        del theta, x0
        return -0.5 * jnp.sum(w**2)

    def energy_fn(w, theta, x0):
        del w, theta, x0
        return jnp.asarray(0.0)

    return log_likelihood_fn, energy_fn


AMAGOLD = dict(kernel="amagold", amagold_dt=0.3, amagold_nstep=10,
               amagold_C=1.0, schedule="constant")


def test_default_path_unchanged_and_full_gradient_through_the_seam_matches():
    d = 3
    ll, en = _gaussian_target(d)
    cfg = SGMCMCConfig(chains=2, iterations=200, burn_in=50, thinning=10,
                       init_std=0.5, **AMAGOLD)
    base = run_sgmcmc(jax.random.key(0), d_w=d, d_theta=0, d_x0=0,
                      log_likelihood_fn=ll, energy_fn=en, config=cfg)

    # The same gradient, routed through the new argument (key ignored).
    def full_grad(key, z):
        del key
        return -z

    routed = run_sgmcmc(jax.random.key(0), d_w=d, d_theta=0, d_x0=0,
                        log_likelihood_fn=ll, energy_fn=en, config=cfg,
                        stochastic_grad_fn=full_grad)

    np.testing.assert_array_equal(base.samples["w_samples"],
                                  routed.samples["w_samples"])
    np.testing.assert_array_equal(base.final_state["accept_rate"],
                                  routed.final_state["accept_rate"])


def test_mh_test_uses_the_full_energy_under_a_biased_gradient():
    """A gradient for N(0, sqrt(2)) with the energy of N(0, 1) -> samples N(0, 1)."""
    ll, en = _gaussian_target(1)
    cfg = SGMCMCConfig(chains=64, iterations=6_000, burn_in=1_000, thinning=10,
                       init_std=1.0, **AMAGOLD)

    def half_grad(key, z):
        del key
        return -0.5 * z          # gradient of the N(0, sqrt(2)) log-density

    biased = run_sgmcmc(jax.random.key(1), d_w=1, d_theta=0, d_x0=0,
                        log_likelihood_fn=ll, energy_fn=en, config=cfg,
                        stochastic_grad_fn=half_grad)
    std = float(np.std(biased.samples["w_samples"]))
    accept = float(np.mean(biased.final_state["accept_rate"]))
    assert accept > 0.1, f"chain effectively frozen (accept {accept})"
    # The full-energy target has std 1; the gradient alone would give sqrt(2).
    assert abs(std - 1.0) < 0.05, f"std {std} -- M-H is not using the full energy"


def test_a_truly_subsampled_likelihood_gradient_targets_the_full_posterior():
    """Linear-Gaussian model, gradient from 2 of 8 observations, rescaled."""
    rng = np.random.default_rng(0)
    n_obs, sigma_y = 8, 0.5
    y = jnp.asarray(rng.normal(1.0, sigma_y, size=n_obs))

    def log_likelihood_fn(w, theta, x0):
        del theta, x0
        return -0.5 * jnp.sum(((y - w[0]) / sigma_y) ** 2)

    def energy_fn(w, theta, x0):
        del theta, x0
        return 0.5 * jnp.sum(w**2)          # N(0, 1) prior

    # Exact posterior for the mean of a normal with known variance.
    prec = 1.0 + n_obs / sigma_y**2
    post_mean = float(jnp.sum(y) / sigma_y**2 / prec)
    post_std = float(1.0 / np.sqrt(prec))

    m = 2

    def mb_grad(key, z):
        idx = jax.random.choice(key, n_obs, (m,), replace=False)
        scale = n_obs / m

        def lp(zz):
            resid = (y[idx] - zz[0]) / sigma_y
            return -0.5 * scale * jnp.sum(resid**2) - 0.5 * jnp.sum(zz**2)

        return jax.grad(lp)(z)

    cfg = SGMCMCConfig(chains=32, iterations=6_000, burn_in=1_000, thinning=10,
                       init_std=0.2, init_mean=(post_mean,),
                       kernel="amagold", amagold_dt=0.03, amagold_nstep=10,
                       amagold_C=1.0, schedule="constant")
    res = run_sgmcmc(jax.random.key(2), d_w=1, d_theta=0, d_x0=0,
                     log_likelihood_fn=log_likelihood_fn, energy_fn=energy_fn,
                     config=cfg, stochastic_grad_fn=mb_grad)
    w = np.asarray(res.samples["w_samples"])
    assert abs(float(w.mean()) - post_mean) < 0.02
    assert abs(float(w.std()) / post_std - 1.0) < 0.06


def test_stochastic_grad_rejected_by_the_non_amagold_kernels():
    ll, en = _gaussian_target(1)
    cfg = SGMCMCConfig(kernel="sgld", iterations=10, burn_in=0, thinning=10)
    with pytest.raises(ValueError, match="stochastic_grad_fn"):
        run_sgmcmc(jax.random.key(0), d_w=1, d_theta=0, d_x0=0,
                   log_likelihood_fn=ll, energy_fn=en, config=cfg,
                   stochastic_grad_fn=lambda key, z: -z)
