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
        z_new, (a_w, a_r, _a_s) = step(k, z)
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
    with pytest.raises(ValueError, match="scale_index"):
        pcn_gibbs(_f, d_w=2, beta=0.5, mh_scales=jnp.asarray([0.1]), scale_index=5)


def test_scale_move_preserves_nonridge_target():
    """The ridge-scale move is a valid M-H move for ANY target; adding it to
    the linear-Gaussian toy (whose geometry it does NOT match) must leave the
    posterior unchanged — this pins the acceptance formula (prior ratio +
    Jacobian): a sign error would visibly shift the moments."""
    step = pcn_gibbs(_f, d_w=D_W, beta=0.5, mh_scales=jnp.asarray([0.8]),
                     n_pcn=2, n_mh=2, scale_index=0, scale_step=0.4, n_scale=2)
    zs, _, _ = _run(step, jax.random.key(3), jnp.zeros(3), 60_000)
    zs = zs[10_000:]
    mean, cov = _analytic_posterior(A, S, Y)
    np.testing.assert_allclose(zs.mean(axis=0), mean, atol=0.04)
    np.testing.assert_allclose(np.sqrt(np.diag(np.cov(zs.T))),
                               np.sqrt(np.diag(cov)), rtol=0.08)


def test_scale_move_beats_ridge():
    """Whitened amplitude ridge in miniature: w ~ N(0,1), log s ~ N(0,1),
    y = 2.0 observed with y ~ N(e^{log s} w, 0.05^2). The exact marginals come
    from 2-D grid integration; the scale-move sampler must reproduce them at a
    budget where the likelihood is invariant along the move (Delta f = 0), so
    the ridge is traversed by prior/Jacobian terms alone."""
    s_obs, y = 0.05, 2.0

    def f(z):
        w, ls = z[0], z[1]
        return -0.5 * (y - jnp.exp(ls) * w) ** 2 / s_obs**2 - 0.5 * ls**2

    # exact marginals by grid integration (posterior includes the w prior)
    wg = np.linspace(-8, 8, 1601)
    lg = np.linspace(-5, 5, 1601)
    W, L = np.meshgrid(wg, lg, indexing="ij")
    logp = (-0.5 * (y - np.exp(L) * W) ** 2 / s_obs**2 - 0.5 * L**2
            - 0.5 * W**2)
    p = np.exp(logp - logp.max())
    p /= p.sum()
    mean_ls = (p.sum(axis=0) * lg).sum()
    std_ls = np.sqrt((p.sum(axis=0) * (lg - mean_ls) ** 2).sum())
    mean_w = (p.sum(axis=1) * wg).sum()
    std_w = np.sqrt((p.sum(axis=1) * (wg - mean_w) ** 2).sum())

    step = pcn_gibbs(f, d_w=1, beta=0.05, mh_scales=jnp.asarray([0.05]),
                     n_pcn=2, n_mh=2, scale_index=0, scale_step=0.5, n_scale=2)
    zs, _, _ = _run(step, jax.random.key(4), jnp.asarray([1.5, 0.3]), 80_000)
    zs = zs[20_000:]
    assert abs(zs[:, 1].mean() - mean_ls) < 0.08, (zs[:, 1].mean(), mean_ls)
    assert abs(zs[:, 1].std() - std_ls) < 0.08, (zs[:, 1].std(), std_ls)
    assert abs(zs[:, 0].mean() - mean_w) < 0.1, (zs[:, 0].mean(), mean_w)
    assert abs(zs[:, 0].std() - std_w) < 0.1, (zs[:, 0].std(), std_w)
