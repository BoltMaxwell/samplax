"""``amagold_tracked``: the same chain as ``amagold`` for one third of the energy.

The simulation-form kernel evaluates ``u_fn`` twice per step (incoming position
and proposal), and the ift-sde driver used to evaluate ``log_posterior`` a third
time to record the trace. All three are the same handful of numbers: the
incoming energy is whatever the previous step ended on, and the trace value is
whichever of the two the M-H test selected. The tracked form carries the energy
in the state so only the proposal's is ever computed.

The saving is a cached float, not an approximation, so the chain must come out
bit-identical. That is the first and most important test here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import samplax
from samplax.integrations import ift_sde

U_FN = lambda x: 0.5 * jnp.sum(x**2)
GRAD_U = lambda key, x: x + 0.1 * jax.random.normal(key, jnp.shape(x))
KERNEL = dict(dt=0.25, nstep=10, C=0.5)


def test_tracked_chain_is_bit_identical_to_amagold():
    plain = samplax.amagold(U_FN, GRAD_U, mh=True, **KERNEL)
    init, tracked = samplax.amagold_tracked(U_FN, GRAD_U, **KERNEL)

    x = jnp.asarray(0.3)
    state = init(x)
    for seed in range(25):
        key = jax.random.key(seed)
        x, accepted_plain = plain(key, x)
        state, accepted_tracked = tracked(key, state)
        assert float(state.position) == float(x), f"diverged at seed {seed}"
        assert bool(accepted_tracked) == bool(accepted_plain)


def test_tracked_matches_amagold_in_several_dimensions():
    """The vector path exercises the diagonal-mass-matrix dt as well."""
    for dt in (0.2, jnp.array([0.2, 0.05, 0.1])):
        plain = samplax.amagold(U_FN, GRAD_U, dt=dt, nstep=6, C=1.0, mh=True)
        init, tracked = samplax.amagold_tracked(U_FN, GRAD_U, dt=dt, nstep=6, C=1.0)
        x = jnp.asarray([0.3, -0.4, 1.1])
        state = init(x)
        for seed in range(15):
            key = jax.random.key(seed)
            x, _ = plain(key, x)
            state, _ = tracked(key, state)
            np.testing.assert_array_equal(np.asarray(state.position), np.asarray(x))


def test_state_energy_is_always_the_energy_of_the_state_position():
    """The whole optimisation rests on this invariant; check it every step,
    across both accepted and rejected moves."""
    init, tracked = samplax.amagold_tracked(U_FN, GRAD_U, **KERNEL)
    state = init(jnp.asarray(0.3))
    accepts = 0
    for seed in range(40):
        state, accepted = tracked(jax.random.key(seed), state)
        accepts += int(bool(accepted))
        np.testing.assert_allclose(
            float(state.energy), float(U_FN(state.position)), rtol=0, atol=0
        )
    assert 0 < accepts < 40, "need both branches exercised for this to mean anything"


def test_tracked_evaluates_the_energy_once_per_step():
    """Count the actual evaluations rather than trusting the reasoning."""
    counts = {"plain": 0, "tracked": 0}

    def counting(label):
        def u(x):
            counts[label] += 1
            return U_FN(x)
        return u

    plain = samplax.amagold(counting("plain"), GRAD_U, mh=True, **KERNEL)
    init, tracked = samplax.amagold_tracked(counting("tracked"), GRAD_U, **KERNEL)

    x = jnp.asarray(0.3)
    state = init(x)
    counts["tracked"] = 0  # discount the one-off init
    for seed in range(10):
        x, _ = plain(jax.random.key(seed), x)
        state, _ = tracked(jax.random.key(seed), state)

    assert counts["plain"] == 20
    assert counts["tracked"] == 10


def test_amagold_skips_the_unused_energy_when_mh_is_off():
    """With mh=False nothing reads the energy, so it should not be computed."""
    calls = {"n": 0}

    def u(x):
        calls["n"] += 1
        return U_FN(x)

    step = samplax.amagold(u, GRAD_U, dt=0.1, nstep=5, C=1.0, mh=False)
    step(jax.random.key(0), jnp.asarray(0.3))
    assert calls["n"] == 0


def test_tracked_reproduces_the_unbiased_large_step_result():
    """The property test from test_correctness, through the tracked path."""
    init, step = samplax.amagold_tracked(
        lambda x: 0.5 * x**2, lambda k, x: x, dt=0.5, nstep=10, C=0.5)

    def body(state, key):
        state, _ = step(key, state)
        return state, state.position

    _, xs = jax.lax.scan(
        body, init(jnp.zeros(())), jax.random.split(jax.random.key(0), 20_000))
    xs = np.asarray(xs[2000:])
    assert abs(xs.var() - 1.0) < 0.06, xs.var()


# --- the minibatch (BNN) form -------------------------------------------------


def _minibatch_setup(T=4):
    position = {"w": jnp.array([0.3, -0.2, 0.5]), "b": jnp.array([0.1])}
    data = jnp.arange(T * 5, dtype=jnp.float64).reshape(T, 5) / 20.0

    def grad_fn(p, batch):
        scale = jnp.sum(batch)
        return jax.tree_util.tree_map(lambda x: scale * x, p)

    def energy_fn(p):
        return 0.5 * sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(p))

    return position, data, grad_fn, energy_fn


def test_minibatch_state_energy_tracks_the_position():
    """The invariant that makes the cached u_old exact: if state.energy is
    always energy_fn(state.position), then reusing it in place of a fresh
    evaluation is bit-identical, on both the accepted and rejected branch."""
    position, data, grad_fn, energy_fn = _minibatch_setup()
    init, step = samplax.amagold_minibatch(T=4, beta=0.05, step_size=1e-2)
    state = init(jax.random.key(0), position, energy_fn)
    np.testing.assert_allclose(float(state.energy), float(energy_fn(state.position)),
                               rtol=0, atol=0)

    outcomes = set()
    for seed in range(30):
        state, accepted, _ = step(jax.random.key(seed), state, grad_fn, energy_fn, data)
        outcomes.add(bool(accepted))
        np.testing.assert_allclose(
            float(state.energy), float(energy_fn(state.position)), rtol=0, atol=0
        )
    assert outcomes == {True, False}, f"need both branches, saw {outcomes}"


def test_minibatch_evaluates_the_full_energy_once_per_outer_step():
    position, data, grad_fn, energy_fn = _minibatch_setup()
    calls = {"n": 0}

    def counting(p):
        calls["n"] += 1
        return energy_fn(p)

    init, step = samplax.amagold_minibatch(T=4, beta=0.05, step_size=1e-2)
    state = init(jax.random.key(0), position, counting)
    calls["n"] = 0  # discount the one-off init
    for seed in range(8):
        state, _, _ = step(jax.random.key(seed), state, grad_fn, counting, data)
    assert calls["n"] == 8, "one full-data energy per outer step (was two)"


def test_minibatch_state_is_a_stable_scan_carry():
    """Adding a field to AmagoldState must not break driving it with lax.scan."""
    position, data, grad_fn, energy_fn = _minibatch_setup()
    init, step = samplax.amagold_minibatch(T=4, beta=0.05, step_size=1e-2)
    state = init(jax.random.key(0), position, energy_fn)

    def body(carry, key):
        carry, accepted, _ = step(key, carry, grad_fn, energy_fn, data)
        return carry, accepted

    state, accepts = jax.jit(lambda s, k: jax.lax.scan(body, s, k))(
        state, jax.random.split(jax.random.key(1), 20))
    assert accepts.shape == (20,)
    np.testing.assert_allclose(float(state.energy), float(energy_fn(state.position)),
                               rtol=1e-12)


# --- the ift-sde driver, which is where the third evaluation lived ------------


def _problem():
    def log_likelihood_fn(w, theta, x0):
        return -0.5 * jnp.sum((w - 0.4) ** 2) + 0.0 * theta[0] + 0.0 * x0[0]

    def energy_fn(w, theta, x0):
        return 0.5 * jnp.sum(w**2)

    return log_likelihood_fn, energy_fn


def _config(**kw):
    base = dict(kernel="amagold", schedule="constant", preconditioner="identity",
                amagold_dt=0.05, amagold_nstep=4, amagold_C=1.0,
                iterations=200, burn_in=0, thinning=50, chains=2,
                x0_prior_std=1.0)
    base.update(kw)
    return ift_sde.SGMCMCConfig(**base)


def _run(cfg, seed=0):
    loglik, energy = _problem()
    return ift_sde.run_sgmcmc(jax.random.key(seed), d_w=3, d_theta=1, d_x0=1,
                              log_likelihood_fn=loglik, energy_fn=energy,
                              config=cfg)


def test_driver_trace_log_posterior_is_the_true_log_posterior():
    """The trace used to come from a separate log_posterior pass and now comes
    off the kernel state; it must still be the log-posterior of the position
    the chain actually holds."""
    cfg = _config()
    res = _run(cfg)
    loglik, energy = _problem()

    # Rebuilt from the driver's own pieces, so this checks the substitution
    # rather than a re-derivation of the prior.
    def log_posterior(z):
        w, theta, x0 = ift_sde._split_z(z, 3, 1)
        return (loglik(w, theta, x0) - energy(w, theta, x0)
                + ift_sde._normal_logpdf(theta, jnp.asarray(cfg.theta_prior_std))
                + ift_sde._normal_logpdf(x0, jnp.asarray(cfg.x0_prior_std)))

    final_z = np.asarray(res.final_state["z"])
    recorded = np.asarray(res.history["log_posterior"][-1])
    expected = np.asarray([float(log_posterior(jnp.asarray(z))) for z in final_z])
    np.testing.assert_allclose(recorded, expected, rtol=1e-12)


def test_driver_final_state_is_the_final_position():
    """Regression: threading a state through the scan makes it easy to leave
    `final_state` pointing at the *initial* positions instead."""
    short = _run(_config(iterations=50, thinning=50))
    long = _run(_config(iterations=400, thinning=50))
    assert not np.allclose(
        np.asarray(short.final_state["z"]), np.asarray(long.final_state["z"])
    )


def test_driver_still_refuses_a_drifting_energy():
    """Caching the energy is only sound because the energy cannot drift; the
    guard that enforces that must stay."""
    loglik, energy = _problem()
    with pytest.raises(ValueError, match="true energy"):
        ift_sde.run_sgmcmc(jax.random.key(0), d_w=3, d_theta=1, d_x0=1,
                           log_likelihood_fn=loglik, energy_fn=energy,
                           config=_config(), correction=object())
