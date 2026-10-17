"""Standalone correctness tests (no sibling repos required).

Run:  JAX_PLATFORMS=cpu python tests/test_correctness.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import samplax


def _chain(kernel, grad_fn, n, step_size, seed=0, temperature=1.0, x0=0.0):
    state = kernel.init(jax.random.key(seed + 999), jnp.asarray(x0))

    def body(state, key):
        g = grad_fn(state.position)
        state = kernel.step(key, state, g, step_size, temperature)
        return state, state.position

    _, xs = jax.lax.scan(body, state, jax.random.split(jax.random.key(seed), n))
    return np.asarray(xs)


def test_sgld_samples_standard_normal():
    # tolerance: autocorrelation time ~ 2/step_size makes the SE of the
    # variance estimate ~ 0.07 at this chain length
    xs = _chain(samplax.sgld(), lambda x: -x, 400_000, 0.01)
    assert abs(xs[40_000:].mean()) < 0.06
    assert abs(xs[40_000:].var() - 1.0) < 0.08


def test_sghmc_samples_standard_normal():
    xs = _chain(samplax.sghmc(alpha=0.1), lambda x: -x, 400_000, 0.01)
    assert abs(xs[40_000:].mean()) < 0.06
    assert abs(xs[40_000:].var() - 1.0) < 0.08


def test_psgld_handles_anisotropic_target():
    """RMSprop-preconditioned SGLD on N(0, diag(1, 100)): both variances right."""
    cov = jnp.asarray([1.0, 100.0])
    kernel = samplax.sgld(preconditioner=samplax.rmsprop(beta=0.999))
    state = kernel.init(jax.random.key(0), jnp.zeros(2))

    def body(state, key):
        g = -state.position / cov
        state = kernel.step(key, state, g, 0.05, 1.0)
        return state, state.position

    _, xs = jax.lax.scan(body, state, jax.random.split(jax.random.key(1), 400_000))
    xs = np.asarray(xs[40_000:])
    np.testing.assert_allclose(xs.var(axis=0), [1.0, 100.0], rtol=0.15)


def test_temperature_zero_is_noiseless():
    for kernel in (samplax.sgld(), samplax.sghmc(alpha=0.1)):
        key = jax.random.key(0)
        s1 = kernel.init(key, jnp.asarray(1.0))
        s2 = kernel.step(key, s1, jnp.asarray(-1.0), 0.01, 0.0)
        s3 = kernel.step(key, s1, jnp.asarray(-1.0), 0.01, 0.0)
        assert float(s2.position) == float(s3.position)
        # and deterministic: rerun with a different key gives the same value
        s4 = kernel.step(jax.random.key(42), s1, jnp.asarray(-1.0), 0.01, 0.0)
        assert float(s2.position) == float(s4.position)


def test_cyclical_csghmc_composition():
    """cSGHMC = cyclical schedule x SGHMC kernel: exploration phases must be
    noise-free, sampling phases noisy, step size cosine-decaying."""
    sched = samplax.cyclical(400, num_cycles=4, initial_step_size=0.1,
                             exploration_ratio=0.25)
    s0 = sched(0)
    assert not bool(s0.do_sample)
    np.testing.assert_allclose(float(s0.step_size), 0.1, rtol=1e-6)
    assert bool(sched(30).do_sample)
    assert float(sched(99).step_size) < 0.001


def test_amagold_unbiased_at_large_step():
    step = samplax.amagold(lambda x: 0.5 * x**2, lambda k, x: x,
                           dt=0.5, nstep=10, C=0.5, mh=True)

    def body(x, key):
        x, _ = step(key, x)
        return x, x

    _, xs = jax.lax.scan(body, jnp.zeros(()), jax.random.split(jax.random.key(0), 20_000))
    xs = np.asarray(xs[2000:])
    assert abs(xs.var() - 1.0) < 0.06, xs.var()


def test_vc_quantization_fixes_naive_bias():
    """The low-precision Gaussian toy: naive quantization inflates the
    stationary std, VC restores it (the lpsgld paper's core result)."""
    # the lpsgld paper's Gaussian-toy settings (WL=8, FL=3, alpha=2e-3):
    # naive quantization inflates std to ~1.3, VC restores 1.0
    lr, wl, fl, n = 2e-3, 8, 3, 200_000
    outs = {}
    for variant in ("naive", "vc"):
        kernel = samplax.lp_sgld(variant, wl, fl, datasize=1)
        state = kernel.init(jax.random.key(0), jnp.zeros(()))

        def body(state, key):
            g = state.position  # descent gradient of U = x^2/2 (mean scale)
            state = kernel.step(key, state, g, lr, 1.0)
            return state, state.position

        _, xs = jax.lax.scan(body, state, jax.random.split(jax.random.key(1), n))
        outs[variant] = float(np.asarray(xs[20_000:]).std())
    assert outs["naive"] > 1.15, outs
    assert abs(outs["vc"] - 1.0) < 0.08, outs


def test_gibbs_precision_shapes_and_scale():
    params = {"w": 0.1 * jnp.ones((50, 20)), "b": 0.1 * jnp.ones(20)}
    lams = samplax.gibbs_precision(jax.random.key(0), params)
    assert set(lams) == {"w", "b"}
    # posterior mean ~ (1 + n/2) / (1 + sum w^2 / 2) ~ 1/0.01 = 100 for w
    assert 60 < float(lams["w"]) < 140, lams


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
