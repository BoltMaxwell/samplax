"""pCN-within-Gibbs kernel: exactness on a coupled linear-Gaussian target.

The toy is chosen to have a strong CROSS-BLOCK correlation (w collectively vs
the rest-block scalar) — the miniature of the sigma-w amplitude ridge that
motivates the kernel. The posterior is Gaussian in closed form, so moments are
checked exactly, not just for sanity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.kernels.pcn_gibbs import pcn_gibbs


def _run(step, key, z0, n):
    def body(carry, k):
        z, aw, ar = carry
        z_new, (a_w, a_r) = step(k, z)
        return (z_new, aw + a_w, ar + a_r), z_new

    (z, aw, ar), zs = jax.lax.scan(body, (z0, jnp.zeros(()), jnp.zeros(())),
                                   jax.random.split(key, n))
    return np.asarray(zs), float(aw) / n, float(ar) / n


def _analytic_posterior(a, s, y):
    """Posterior of z ~ N(0, I_d) with scalar obs y ~ N(a.z, s^2):
    cov = (I + a a^T / s^2)^-1, mean = cov a y / s^2 (Sherman-Morrison)."""
    d = len(a)
    prec = np.eye(d) + np.outer(a, a) / s**2
    cov = np.linalg.inv(prec)
    mean = cov @ a * y / s**2
    return mean, cov


A = np.array([1.0, 0.5, 1.0])  # [c_w1, c_w2, c_theta]
S, Y = 0.5, 1.5
D_W = 2


def _f(z):
    # likelihood + theta prior; w prior EXCLUDED (pCN absorbs it)
    resid = Y - jnp.dot(jnp.asarray(A), z)
    return -0.5 * resid**2 / S**2 - 0.5 * z[2] ** 2


def test_matches_analytic_posterior():
    step = pcn_gibbs(_f, d_w=D_W, beta=0.5, mh_scales=jnp.asarray([0.8]),
                     n_pcn=2, n_mh=2)
    zs, aw, ar = _run(step, jax.random.key(0), jnp.zeros(3), 60_000)
    zs = zs[10_000:]
    mean, cov = _analytic_posterior(A, S, Y)
    got_mean = zs.mean(axis=0)
    got_cov = np.cov(zs.T)
    assert 0.1 < aw < 0.95 and 0.1 < ar < 0.95, (aw, ar)
    np.testing.assert_allclose(got_mean, mean, atol=0.03)
    np.testing.assert_allclose(np.sqrt(np.diag(got_cov)), np.sqrt(np.diag(cov)),
                               rtol=0.06)
    # the cross-block correlation (w vs theta) must be reproduced
    corr_got = got_cov[0, 2] / np.sqrt(got_cov[0, 0] * got_cov[2, 2])
    corr_want = cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2])
    assert abs(corr_got - corr_want) < 0.06, (corr_got, corr_want)


def test_beta_one_is_independence_sampler():
    """beta = 1 proposes w fresh from the prior; still exact."""
    step = pcn_gibbs(_f, d_w=D_W, beta=1.0, mh_scales=jnp.asarray([0.8]))
    zs, aw, _ = _run(step, jax.random.key(1), jnp.zeros(3), 60_000)
    zs = zs[10_000:]
    mean, cov = _analytic_posterior(A, S, Y)
    np.testing.assert_allclose(zs.mean(axis=0), mean, atol=0.04)
    np.testing.assert_allclose(np.sqrt(np.diag(np.cov(zs.T))),
                               np.sqrt(np.diag(cov)), rtol=0.08)
    assert aw < 0.9  # independence proposals do get rejected here


def test_empty_rest_block():
    f = lambda z: -0.5 * jnp.sum((z - 0.5) ** 2) * 0.0  # flat likelihood
    step = pcn_gibbs(f, d_w=3, beta=0.7, mh_scales=jnp.zeros((0,)))
    zs, aw, ar = _run(step, jax.random.key(2), jnp.zeros(3), 20_000)
    # target reduces to the N(0, I) prior; pCN with flat f accepts always
    assert aw == 1.0 and ar == 1.0
    np.testing.assert_allclose(zs[2000:].std(axis=0), 1.0, rtol=0.06)


def test_validation():
    with pytest.raises(ValueError, match="beta"):
        pcn_gibbs(_f, d_w=2, beta=0.0, mh_scales=jnp.asarray([0.1]))
    with pytest.raises(ValueError, match="n_pcn"):
        pcn_gibbs(_f, d_w=2, beta=0.5, mh_scales=jnp.asarray([0.1]), n_pcn=0)
