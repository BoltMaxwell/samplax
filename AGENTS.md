# AGENTS.md

## What this is

`samplax` is a pure-JAX library of stochastic-gradient MCMC kernels (SGLD, SGHMC, HMC, AMAGOLD, pCN-Gibbs, low-precision SGLD) with schedules and preconditioners. It ships kernels only; you own the sampling loop.

## Install

`pip install git+https://github.com/BoltMaxwell/samplax`, or `pip install -e '.[dev]'` from a checkout.

## The protocol

Defined in `samplax/base.py`:

- `kernel.init(key, position) -> state`
- `kernel.step(key, state, grad, step_size, temperature=1.0) -> state`
- `grad` is the ascent gradient of the log-density.
- `state.position` is the sample.
- `temperature=0` is the optimisation limit.

## Conventions that bite

- The gradient is ascent (of log-density), not descent of a loss.
- For minibatches, pass the sum-scale gradient of the full log-posterior estimated from the batch (rescale by `N / batch_size`). Step sizes are on that scale.
- `amagold`, `amagold_tracked`, `hmc` and `lp_sgld` are not `Kernel`s and have their own shapes. `amagold_tracked(u_fn, grad_u, *, dt, nstep, C)` returns `init(x0)` (no key) and `step(key, state) -> (state, accepted)`. `amagold` gives `step(key, x) -> (new_x, accepted)`; `hmc` gives `step(key, x) -> new_x`. `lp_sgld` returns an `LPKernel(init, step, forward_quant)`.
- `amagold*` and `hmc` take `grad_u`, the gradient of the potential U (descent direction), plus `u_fn`, the full-data potential. `lp_sgld.step(key, state, grad, lr, temperature)` takes the descent mean-loss gradient.

## Run the tests

`JAX_PLATFORMS=cpu python -m pytest tests -q --ignore=tests/test_vendored_equivalence.py`

`tests/test_vendored_equivalence.py` needs SGHMC-jax, csgmcmc-jax, amagold-jax and low-precision-sgld-jax checked out as sibling directories; it skips otherwise.

## Where things are

- `samplax/base.py`: the `Kernel` protocol and state types.
- `samplax/kernels/`: `sgld.py`, `sghmc.py`, `hmc.py`, `amagold.py`, `pcn_gibbs.py`.
- `samplax/schedules.py`: step-size and temperature schedules.
- `samplax/preconditioners.py`: gradient preconditioners.
- `samplax/transforms/`: `lp_sgld.py`, `quant.py`, `vc.py`.
- `samplax/gibbs.py`: Gibbs resampling of Gamma precision hyperpriors.

## What not to do

- Do not add loop-owning drivers; the library provides steps, callers loop.
- Do not add a kernel without a verified source: either vendored from a verified JAX port of the authors' code and bit-equivalence-tested against it, or original and verified by exactness tests against closed-form targets.
