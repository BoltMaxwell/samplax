samplax
=======

A curated stochastic-gradient MCMC library in pure JAX. Kernels are small
`init`/`step` pairs that never own the loop, so you drive them with
`jax.lax.scan`, a Python loop, or inside a larger outer loop. It includes
SGLD, SGHMC, pSGLD preconditioning, cyclical schedules, AMAGOLD, and
low-precision SGLD, with a stated provenance and check for every kernel.
Provenance over coverage.

Install
-------

```bash
pip install git+https://github.com/BoltMaxwell/samplax
```

For development, from a checkout:

```bash
pip install -e '.[dev]'
```

Quickstart
----------

SGLD on a 2-D Gaussian with known mean `[1, -2]` and standard deviation
`[1, 0.5]`. The kernel takes the **ascent** gradient of the log-density at
`state.position`; the sample lives in `state.position`.

```python
import jax, jax.numpy as jnp
import samplax

key = jax.random.PRNGKey(0)
mean = jnp.array([1.0, -2.0])
std = jnp.array([1.0, 0.5])

def log_post(x):
    return -0.5 * jnp.sum(((x - mean) / std) ** 2)

grad_fn = jax.grad(log_post)          # ascent gradient, any pytree

n = 20_000
kernel = samplax.sgld()
sched = samplax.constant(0.01)

key, k_init, k_run = jax.random.split(key, 3)
state = kernel.init(k_init, jnp.zeros(2))

def body(state, inp):
    t, k = inp
    s = sched(t)
    state = kernel.step(k, state, grad_fn(state.position), s.step_size)
    return state, state.position

_, samples = jax.lax.scan(body, state, (jnp.arange(n), jax.random.split(k_run, n)))
samples = samples[n // 4:]            # drop burn-in
print("mean", samples.mean(0), "(true [1, -2])")
print("std ", samples.std(0), "(true [1, 0.5])")
```

Swapping the sampler is one line: preconditioned SGHMC under a cyclical
step-size schedule. Exploration phases of each cycle run at temperature 0
(the optimisation limit); only the sampling phases add noise.

```python
import jax, jax.numpy as jnp
import samplax

key = jax.random.PRNGKey(0)
mean = jnp.array([1.0, -2.0])
std = jnp.array([1.0, 0.5])

def log_post(x):
    return -0.5 * jnp.sum(((x - mean) / std) ** 2)

grad_fn = jax.grad(log_post)

n = 20_000
kernel = samplax.sghmc(alpha=0.1, preconditioner=samplax.rmsprop())
sched = samplax.cyclical(num_training_steps=n, num_cycles=4,
                         initial_step_size=1e-4)
# also available: constant, exponential, polynomial

key, k_init, k_run = jax.random.split(key, 3)
state = kernel.init(k_init, jnp.zeros(2))

def body(state, inp):
    t, k = inp
    s = sched(t)
    temperature = jnp.where(s.do_sample, 1.0, 0.0)
    state = kernel.step(k, state, grad_fn(state.position), s.step_size,
                        temperature=temperature)
    return state, (state.position, s.do_sample)

_, (samples, keep) = jax.lax.scan(
    body, state, (jnp.arange(n), jax.random.split(k_run, n)))
samples = samples[keep]               # keep only sampling-phase draws
print("mean", samples.mean(0), "(true [1, -2])")
# approximate: cyclical draws are autocorrelated
print("std ", samples.std(0), "(true [1, 0.5])")
```

The protocol (`samplax.base`): `init(key, position) -> state`;
`step(key, state, grad, step_size, temperature) -> state`; every state's
first field is `position`. AMAGOLD and low-precision SGLD have richer,
documented interfaces (energy callbacks, quantization-grid conventions).

AMAGOLD
-------

AMAGOLD is the amortized Metropolis-Hastings-corrected SG-HMC sampler. Here
it infers the mean of a 1-D Gaussian from `N = 1000` observations using
minibatch gradients of 100 points, while the M-H test uses the full-data
potential.

```python
import jax, jax.numpy as jnp
import samplax

key = jax.random.PRNGKey(0)
N = 1000
key, k_data = jax.random.split(key)
sigma = 10.0   # large noise keeps dt=0.1 stable at N=1000
data = 2.0 + sigma * jax.random.normal(k_data, (N,))   # y ~ N(2, sigma^2)

def u_fn(x):
    # full-data potential: negative log posterior (flat prior)
    return 0.5 * jnp.sum((data - x) ** 2) / sigma**2

def grad_u(key, x):
    # grad_u(key, x): keyed, so it may use a minibatch; the M-H test uses the full u_fn
    idx = jax.random.choice(key, N, (100,), replace=False)
    return (N / 100) * jnp.sum(x - data[idx]) / sigma**2   # descent gradient of U

init, step = samplax.amagold_tracked(u_fn, grad_u, dt=0.1, nstep=5, C=0.5)
step = jax.jit(step)
state = init(jnp.array(0.0))

accepted = 0
total = 0.0
for i in range(200):
    key, k = jax.random.split(key)
    state, acc = step(k, state)
    accepted += int(acc)
    if i >= 100:
        total += float(state.position)   # average the second 100 steps

mean_x = total / 100
print("posterior mean", mean_x, "(data mean", float(data.mean()), ")")
print("acceptance rate", accepted / 200)
```

Provenance
----------

SGLD, SGHMC, cyclical schedules, AMAGOLD, low-precision SGLD and the HMC reference
are vendored from verified JAX ports of the authors' code and equivalence-tested
against them (`tests/test_vendored_equivalence.py`). pSGLD preconditioning (Li et al.'s
RMSprop construction, checked on a Gaussian target) and the Gibbs hyperpriors (from
SGHMC-jax, checked for shape and scale) are vendored constructions without a
equivalence test. pCN-within-Gibbs is original and checked against closed forms.

| method | source repo | paper | check |
|---|---|---|---|
| SGLD, SGHMC (+ v_hat noise correction) | [SGHMC-jax], [csgmcmc-jax] | Chen, Fox, Guestrin 2014 | equivalence, rtol 1e-7 |
| cyclical schedules (cSGLD/cSGHMC) | [csgmcmc-jax] | Zhang et al. 2020 | equivalence, rtol 1e-7 |
| pSGLD preconditioning | ift-sde `psgld_sampler_fast` (private) | Li et al. 2016 | Gaussian target (test_correctness.py) |
| AMAGOLD (amortized M-H) | [amagold-jax] | Zhang, Cooper, De Sa 2020 | equivalence, rtol 1e-7 |
| low-precision SGLD (F / naive / VC) | [low-precision-sgld-jax] | Zhang, Wilson, De Sa 2022 | equivalence, rtol 1e-7 |
| Gibbs Gamma hyperpriors | [SGHMC-jax] (ML-SGHMC bayesnn/mf) | Chen et al. 2014 | shape & scale (test_correctness.py) |
| HMC reference | [SGHMC-jax] | Chen et al. 2014 | equivalence, rtol 1e-7 |
| pCN-within-Gibbs | original | Cotter et al. 2013 | exactness vs closed form |

How this differs
----------------

Compared with [jax-sgmc](https://github.com/tummfm/jax-sgmc) and
[sgmcmcjax](https://github.com/jeremiecoullon/SGMCMCJax), samplax is not a
framework: no data loaders, potential modules, or solver aliases. You bring a
gradient and a pytree, and the composition axes (schedules, preconditioners,
precision, M-H correction) cross-cut, so cyclical SGHMC or preconditioned SGLD
under a cyclical schedule fall out of the same protocol, with no blackjax
dependency.

Tests
-----

```bash
JAX_PLATFORMS=cpu python -m pytest tests -q --ignore=tests/test_vendored_equivalence.py
```

The equivalence test (`tests/test_vendored_equivalence.py`) looks for the four
source repos in `~/Documents/<repo>` or `$SAMPLAX_SOURCE_REPOS/<repo>`. Missing
repos are reported as skipped, so a run that proves equivalence shows `7 passed`
with no skips. The file enables float64 itself and asserts agreement to rtol
1e-7; in float32 the AMAGOLD chain differs from amagold-jax by one ulp
(operation order).

Docs
----

```bash
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build
```

Cite
----

See `CITATION.cff`.

[SGHMC-jax]: https://github.com/BoltMaxwell/SGHMC-jax
[csgmcmc-jax]: https://github.com/BoltMaxwell/csgmcmc-jax
[amagold-jax]: https://github.com/BoltMaxwell/amagold-jax
[low-precision-sgld-jax]: https://github.com/BoltMaxwell/low-precision-sgld-jax
