# Quickstart

## The protocol

A sampler is an `(init, step)` pair ({class}`samplax.Kernel`):

- `state = kernel.init(key, position)` — `position` is any pytree; every
  state's first field is `position`.
- `state = kernel.step(key, state, grad, step_size, temperature)` — `grad`
  is the **ascent** gradient of the log-density at `state.position`;
  `step_size` and `temperature` are per-step so schedules and tempering
  compose from outside. `temperature=0` disables the injected noise
  (the optimization limit, used by cyclical exploration phases).

Kernels never own the loop: drive them from `lax.scan`, a Python loop, or an
outer sampler.

## SGLD on a toy posterior

```python
import jax, jax.numpy as jnp
import samplax

log_post = lambda x: -0.5 * jnp.sum(x**2)
grad_fn = jax.grad(log_post)

kernel = samplax.sgld()
key = jax.random.key(0)
state = kernel.init(key, jnp.zeros(2))

def body(state, key):
    state = kernel.step(key, state, grad_fn(state.position), 1e-2)
    return state, state.position

_, samples = jax.lax.scan(body, state, jax.random.split(key, 100_000))
```

## Composition: cyclical, preconditioned SGHMC

```python
kernel = samplax.sghmc(alpha=0.1, preconditioner=samplax.rmsprop())
sched = samplax.cyclical(50_000, num_cycles=4, initial_step_size=1e-4)

def body(state, inp):
    t, key = inp
    s = sched(t)
    g = grad_fn(state.position)
    state = kernel.step(key, state, g, s.step_size, 1.0 * s.do_sample)
    return state, state.position
```

`samplax.cyclical` + `samplax.sghmc` is cSGHMC (Zhang et al. 2020);
swap in `samplax.sgld()` for cSGLD; add `preconditioner=` for the
preconditioned variants — the axes are independent.

## Minibatch gradients

samplax kernels consume whatever gradient you hand them. For a Bayesian
posterior over `N` data points estimated from a minibatch of size `B`, pass
the *sum-scale* ascent gradient, e.g.
`g = -(N/B) * grad_minibatch_loss - grad_prior_penalty`, and a step size in
the same units. See the source experiments (SGHMC-jax bayesnn/mf,
csgmcmc-jax CIFAR) for worked examples.

## AMAGOLD and low-precision SGLD

These have richer interfaces than the base protocol (an M-H correction needs
energy evaluations; quantization grids are scale-sensitive):

```python
step = samplax.amagold(u_fn, grad_u, dt=0.25, nstep=10, C=0.5)   # simulation form
x, accepted = step(key, x)

kernel = samplax.lp_sgld("vc", wl=8, fl=3, datasize=N)   # variance-corrected
state = kernel.init(key, params)                          # (quantizes params)
state = kernel.step(key, state, mean_loss_grad, lr)       # descent, mean scale
```

See {mod}`samplax.kernels.amagold` and {mod}`samplax.transforms.lp_sgld` for
the exact conventions.
