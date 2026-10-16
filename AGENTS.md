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
- For minibatches, pass the sum-scale gradient of the full log-posterior estimated from the batch (rescale the likelihood term by `N / batch_size`, not the prior). Step sizes are on that scale.
- `amagold`, `amagold_tracked`, `hmc` and `lp_sgld` are not `Kernel`s and have their own shapes. `amagold_tracked(u_fn, grad_u, *, dt, nstep, C, dt_friction=None)` returns `init(x0)` (no key) and `step(key, state) -> (state, accepted)`; `grad_u` is called as `grad_u(key, x)` (keyed, so a minibatch estimate can be drawn per call) and `u_fn(x)` is the full-data potential. `amagold` gives `step(key, x) -> (new_x, accepted)`; `hmc` gives `step(key, x) -> new_x`. `lp_sgld` returns an `LPKernel(init, step, forward_quant)`.
- `pcn_gibbs(log_target_no_wprior, *, d_w, beta, mh_scales, ...)` returns `step(key, z) -> (z_new, (acc_w, acc_rest, acc_scale))` over a flat `z = [w | rest]`; gradient-free; `log_target_no_wprior` must EXCLUDE the N(0, I) prior on `z[:d_w]`.
- `amagold_minibatch(*, T, beta, step_size)` returns `init(key, position, energy_fn)` and `step(key, state, grad_fn, energy_fn, batches) -> (state, accepted, rho)`.
- `amagold*` and `hmc` take `grad_u`, the gradient of the potential U (descent direction), plus `u_fn`, the full-data potential. `lp_sgld.step(key, state, grad, lr, temperature)` takes the descent mean-loss gradient.

## Run the tests

`JAX_PLATFORMS=cpu python -m pytest tests -q --ignore=tests/test_vendored_equivalence.py`

`tests/test_vendored_equivalence.py` looks for SGHMC-jax, csgmcmc-jax, amagold-jax and low-precision-sgld-jax in `~/Documents/<repo>` or `$SAMPLAX_SOURCE_REPOS/<repo>`; missing repos are reported as skipped, so a run that proves equivalence shows `7 passed` with no skips. The file enables float64 itself and asserts agreement to rtol 1e-7; in float32 the AMAGOLD chain differs from amagold-jax by one ulp (operation order).

## Where things are

- `samplax/base.py`: the `Kernel` protocol and state types.
- `samplax/kernels/`: `sgld.py`, `sghmc.py`, `hmc.py`, `amagold.py`, `pcn_gibbs.py`.
- `samplax/schedules.py`: step-size and temperature schedules.
- `samplax/preconditioners.py`: gradient preconditioners.
- `samplax/transforms/`: `lp_sgld.py`, `quant.py`, `vc.py`.
- `samplax/gibbs.py`: Gibbs resampling of Gamma precision hyperpriors.

## What not to do

- Do not add loop-owning drivers; the library provides steps, callers loop.
- Do not add a kernel without a verified source: either vendored from a verified JAX port of the authors' code and equivalence-tested against it, or an exact construction from the paper checked against a known target, or original and verified by exactness tests against closed-form targets.
