samplax
=======

A curated stochastic-gradient MCMC library in pure JAX.

Every kernel in samplax is vendored from a port of the **original paper
authors' code** that was quantitatively verified against that original before
being adopted here (the four source repos below), and the vendored copies are
bit-equivalence-tested against those ports (`tests/test_vendored_equivalence.py`).
This is a personal, curated sampling book — provenance over coverage.

| method | source repo | paper |
|---|---|---|
| SGLD, SGHMC (+ v_hat noise correction) | [SGHMC-jax](https://github.com/BoltMaxwell/SGHMC-jax), [csgmcmc-jax] | Chen, Fox, Guestrin 2014 |
| cyclical schedules (cSGLD/cSGHMC) | csgmcmc-jax | Zhang et al. 2020 |
| pSGLD preconditioning | (Li et al. construction, matches ift-sde) | Li et al. 2016 |
| AMAGOLD (amortized M-H) | [amagold-jax](https://github.com/BoltMaxwell/amagold-jax) | Zhang, Cooper, De Sa 2020 |
| low-precision SGLD (F / naive / VC) | low-precision-sgld-jax | Zhang, Wilson, De Sa 2022 |
| Gibbs Gamma hyperpriors | SGHMC-jax (ML-SGHMC bayesnn/mf) | Chen et al. 2014 |

Integrations
------------

The `samplax.integrations` package holds adapter and composition code of ift-sde origin, not vendored-kernel provenance classes. Currently: `ift_sde` (engine adapter for the ift-sde `run_sgmcmc` sampler seam, with a stateful `Correction` protocol); `nested` (persistent-PCD ∇log Z correction with optional re-warm policy, Tieleman 2008 framing).

How this differs from [jax-sgmc](https://github.com/tummfm/jax-sgmc) /
[sgmcmcjax](https://github.com/jeremiecoullon/SGMCMCJax): samplax is **not a
framework** — no data loaders, potential modules, or solver aliases. You bring
a gradient (or log-density callbacks) and a pytree; kernels never own the
loop, so they embed in nested outer loops (see the ift-sde integration). The
composition axes — schedules x preconditioners x precision x M-H correction —
are cross-cut: cyclical SGHMC, preconditioned SGLD under a cyclical schedule,
variance-corrected low-precision SGLD, all fall out of the same protocol.
Pure JAX: no blackjax dependency (the parameter mappings to blackjax's
diffusions, and the stability caveats of its qp splitting, are documented in
the SGHMC-jax verification notes).

Install
-------

```bash
pip install -e .            # or: pip install git+https://github.com/BoltMaxwell/samplax
```

Quickstart
----------

```python
import jax, jax.numpy as jnp
import samplax

grad_fn = jax.grad(log_posterior)             # ascent gradient, any pytree

kernel = samplax.sghmc(alpha=0.1, preconditioner=samplax.rmsprop())
sched  = samplax.cyclical(num_training_steps=50_000, num_cycles=4,
                          initial_step_size=1e-4)   # -> cyclical pSGHMC
# also available: constant, exponential, polynomial

state = kernel.init(key, position)
def body(state, inp):
    t, key = inp
    s = sched(t)
    g = grad_fn(state.position)
    state = kernel.step(key, state, g, s.step_size,
                        temperature=1.0 * s.do_sample)
    return state, state.position

_, samples = jax.lax.scan(body, state, (jnp.arange(n), jax.random.split(key, n)))
```

The protocol (`samplax.base`): `init(key, position) -> state`;
`step(key, state, grad, step_size, temperature) -> state`; every state's
first field is `position`; `temperature=0` disables noise (exploration
phases / the optimization limit). AMAGOLD and low-precision SGLD have richer,
documented interfaces (energy callbacks / quantization-grid conventions).

ift-sde integration
-------------------

`samplax.integrations.ift_sde` implements both of ift-sde's sampler seams:

- `run_sgmcmc(rng_key, *, d_w, d_theta, d_x0, log_likelihood_fn, energy_fn,
  config)` — the Family-B engine signature (same as `run_nsvi`/`run_npsgld`),
  running any samplax kernel/schedule/preconditioner over `z = [w, theta, x0]`.
  Pass `correction_grad_fn` to add a partition-gradient estimator (e.g.
  ift-sde's auxiliary chain) for calibration targets.
- `make_field_sampler(kernel, schedule, ...)` — the
  `param_loop.sgld_parameter_sampler` inner-sampler boundary
  `f(field_state, param_state, key) -> chain`.

Tests
-----

```bash
JAX_PLATFORMS=cpu python tests/test_correctness.py            # standalone
JAX_PLATFORMS=cpu python tests/test_vendored_equivalence.py   # vs the source repos
```

Docs
----

```bash
pip install -e '.[docs]' && sphinx-build -b html docs docs/_build
```

[csgmcmc-jax]: https://github.com/BoltMaxwell/csgmcmc-jax
