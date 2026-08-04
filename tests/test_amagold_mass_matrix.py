"""Vector-dt (diagonal mass matrix) AMAGOLD + the thermostat-noise fix.

The exactness claim for vector dt is inherited from the scalar kernel by the
coordinate-rescaling identity in the kernel docstring; the tests here check
(1) the identity's degenerate case (equal-entry vector == scalar, bitwise),
(2) end-to-end exactness on a strongly anisotropic Gaussian where a scalar-dt
kernel at the same base step cannot mix the slow coordinate, (3) the
integration path (amagold_dt_theta), (4) the loud preconditioner raise, and
(5) a regression for the 2026-08 thermostat-noise-shape fix (per-coordinate
noise, not one scalar broadcast).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from samplax.kernels.amagold import amagold
from samplax.integrations import ift_sde


def _run_chain(step, key, x0, n):
    def body(carry, k):
        x, acc = carry
        x_new, accepted = step(k, x)
        return (x_new, acc + accepted.astype(jnp.float64)), x_new

    keys = jax.random.split(key, n)
    (x, acc), xs = jax.lax.scan(body, (x0, jnp.zeros(())), keys)
    return xs, float(acc) / n


def test_vector_dt_equal_entries_matches_scalar():
    """Equal-entry vector dt == scalar dt: same RNG stream, same trajectory.
    Tolerance is near-machine (XLA fusion reassociates a couple of products;
    the identity is analytic, not bitwise-through-the-compiler)."""
    var = jnp.array([1.0, 4.0])
    u_fn = lambda x: 0.5 * jnp.sum(x**2 / var)
    grad_u = lambda key, x: x / var

    step_s = amagold(u_fn, grad_u, dt=0.3, nstep=5, C=1.0)
    step_v = amagold(u_fn, grad_u, dt=jnp.full((2,), 0.3), nstep=5, C=1.0)
    key = jax.random.key(0)
    x0 = jnp.array([0.7, -1.1])
    xs_s, _ = _run_chain(step_s, key, x0, 200)
    xs_v, _ = _run_chain(step_v, key, x0, 200)
    np.testing.assert_allclose(np.asarray(xs_s), np.asarray(xs_v), rtol=1e-12, atol=1e-13)


def test_vector_dt_exact_on_anisotropic_gaussian():
    """N(0, diag(1, s^2)) with s=30. Vector dt proportional to the coordinate
    scales samples both coordinates correctly; scalar dt at the same base step
    (stable for the fast coordinate) leaves the slow coordinate visibly
    under-mixed in the same budget."""
    s = 30.0
    var = jnp.array([1.0, s**2])
    u_fn = lambda x: 0.5 * jnp.sum(x**2 / var)
    grad_u = lambda key, x: x / var

    n = 30_000
    dt_base = 0.4
    step_v = amagold(u_fn, grad_u, dt=jnp.array([dt_base, dt_base * s]),
                     nstep=5, C=1.0, dt_friction=dt_base)
    xs, accept = _run_chain(step_v, jax.random.key(1), jnp.zeros(2), n)
    xs = np.asarray(xs[n // 5:])
    assert 0.15 <= accept <= 1.0, accept
    assert abs(xs[:, 0].std() - 1.0) < 0.08
    assert abs(xs[:, 1].std() - s) < 0.08 * s
    assert abs(xs[:, 0].mean()) < 0.1
    assert abs(xs[:, 1].mean()) < 0.1 * s

    # Contrast: scalar dt at the fast coordinate's step. The slow coordinate
    # still *reaches* its scale eventually (the toy has no hard stability
    # wall); what the mass matrix buys is DECORRELATION. Compare lag-20
    # autocorrelation of the slow coordinate: near 1 under scalar dt, well
    # below under matched vector dt.
    step_sc = amagold(u_fn, grad_u, dt=dt_base, nstep=5, C=1.0)
    xs_sc, _ = _run_chain(step_sc, jax.random.key(2), jnp.zeros(2), n)
    xs_sc = np.asarray(xs_sc[n // 5:])

    def lag_autocorr(a, lag):
        a = a - a.mean()
        return float(np.dot(a[:-lag], a[lag:]) / np.dot(a, a))

    ac_vec = lag_autocorr(xs[:, 1], 20)
    ac_sc = lag_autocorr(xs_sc[:, 1], 20)
    assert ac_sc > 0.9, ac_sc
    assert ac_vec < 0.5, ac_vec


def test_dt_blocks_through_integration():
    """theta target = its prior N(0, 10^2); w target = N(0, 1) likelihood.
    Per-block dts must recover both scales."""
    d_w, d_theta, d_x0 = 2, 1, 0
    scale = 10.0

    def loglik(w, theta, x0):
        return -0.5 * jnp.sum(w**2)

    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", schedule="constant", preconditioner="identity",
        amagold_dt=0.3, amagold_dt_theta=0.3 * scale, amagold_C=1.0,
        amagold_nstep=5, theta_prior_std=scale,
        iterations=30_000, burn_in=10_000, thinning=10, chains=4,
    )
    res = ift_sde.run_sgmcmc(jax.random.key(3), d_w=d_w, d_theta=d_theta,
                             d_x0=d_x0, log_likelihood_fn=loglik,
                             energy_fn=lambda w, t, x: 0.0, config=cfg)
    w = res.samples["w_samples"].reshape(-1, d_w)
    th = res.samples["theta_samples"].reshape(-1, d_theta)
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(th))
    assert np.all(np.abs(w.std(axis=0) - 1.0) < 0.15)
    assert abs(th.std() - scale) < 0.2 * scale
    accept = float(np.mean(res.final_state["accept_rate"]))
    assert 0.15 <= accept <= 1.0


def test_amagold_preconditioner_raises_loudly():
    cfg = ift_sde.SGMCMCConfig(
        kernel="amagold", schedule="constant", preconditioner="rmsprop",
        amagold_dt=0.1, iterations=100, burn_in=0, thinning=10, chains=1,
    )
    with pytest.raises(ValueError, match="amagold has no preconditioner"):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=2, d_theta=0, d_x0=0,
                           log_likelihood_fn=lambda w, t, x: -0.5 * jnp.sum(w**2),
                           energy_fn=lambda w, t, x: 0.0, config=cfg)


def test_thermostat_noise_is_per_coordinate():
    """Regression for the scalar-broadcast thermostat-noise bug. Free particle
    (U = 0, mh=False), long trajectories: per-step displacements of different
    coordinates must be (nearly) uncorrelated. Under the old bug the injected
    noise was IDENTICAL across coordinates, driving the displacement
    correlation far above zero."""
    u_fn = lambda x: jnp.zeros(())
    grad_u = lambda key, x: jnp.zeros_like(x)
    # Large C and long nstep so thermostat noise dominates the displacement.
    step = amagold(u_fn, grad_u, dt=0.1, nstep=50, C=4.0, mh=False)

    def body(x, k):
        x_new, _ = step(k, x)
        return x_new, x_new - x

    keys = jax.random.split(jax.random.key(4), 4000)
    _, deltas = jax.lax.scan(body, jnp.zeros(2), keys)
    deltas = np.asarray(deltas)
    corr = np.corrcoef(deltas[:, 0], deltas[:, 1])[0, 1]
    assert abs(corr) < 0.1, corr
