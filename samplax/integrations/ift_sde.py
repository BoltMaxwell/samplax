"""Adapters for the ift-sde repo's two sampler seams.

Family-B seam (``methods/nsvi``, ``methods/npsgld`` engines):
:func:`run_sgmcmc` has the same keyword-only signature as ``run_nsvi`` /
``run_npsgld`` — ``(rng_key, *, d_w, d_theta, d_x0, log_likelihood_fn,
energy_fn, config)`` — and returns a result object with ``.samples``
(``z/w/theta/x0_samples``) and ``.history``, so it drops into the runner
dispatch tables unchanged. It samples the *relaxed joint posterior*

    log p(z) = log_likelihood(w, theta) - energy(w, theta, x0)
               + log N(theta; 0, theta_prior_std) + log N(x0; 0, x0_prior_std)

with any samplax kernel, schedule, and preconditioner. NOTE: unlike NPSGLD it
does **not** estimate the NIFF prior partition gradient by default; for
calibration targets where log Z(theta, x0) varies, supply
``correction_grad_fn(z, key) -> grad_z`` (e.g. ift-sde's auxiliary-chain
estimator) and it is added to the drift — that is the intended mix-and-match:
their nesting, samplax's kernels. For state estimation (theta fixed) the
partition term is constant and the default target is exact.

Nesting-I seam (``experiments/calibration/param_loop.py``):
:func:`make_field_sampler` builds the inner-field-sampler boundary callable
``f(field_state, param_state, key) -> field_chain`` from a samplax kernel and
a per-(field, param) gradient callback.

x64: ift-sde enables ``jax_enable_x64``; these adapters follow the dtype of
the inputs, so they run in float64 there and float32 elsewhere.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ..kernels.sghmc import sghmc
from ..kernels.sgld import sgld
from ..preconditioners import identity, rmsprop
from ..schedules import constant, cyclical, polynomial


@dataclass(frozen=True)
class SGMCMCConfig:
    kernel: str = "sghmc"            # "sgld" | "sghmc"
    alpha: float = 0.1               # sghmc friction
    preconditioner: str = "identity"  # "identity" | "rmsprop"
    iterations: int = 20_000
    chains: int = 4
    burn_in: int = 5_000
    thinning: int = 10
    step_size: float = 2.0e-4
    schedule: str = "constant"       # "constant" | "cyclical" | "polynomial"
    num_cycles: int = 4              # cyclical
    exploration_ratio: float = 0.25  # cyclical
    poly_b: float = 1.0              # polynomial: step_size * (b + t)^-gamma
    poly_gamma: float = 0.55
    temperature: float = 1.0
    theta_prior_std: float = 1.0
    x0_prior_std: float = 1.0
    init_std: float = 0.5
    grad_clip: float = 1.0e3
    trace_every: int = 1_000


@dataclass
class SGMCMCResult:
    samples: dict
    history: dict
    final_state: dict
    config: SGMCMCConfig


def _split_z(z, d_w, d_theta):
    return z[..., :d_w], z[..., d_w:d_w + d_theta], z[..., d_w + d_theta:]


def _normal_logpdf(x, std):
    return jnp.sum(-0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std) - 0.5 * (x / std) ** 2)


def _clip(g, max_norm):
    norm = jnp.sqrt(jnp.sum(g * g))
    return jnp.where(jnp.isfinite(norm) & (norm > max_norm), g * (max_norm / norm),
                     jnp.nan_to_num(g))


def run_sgmcmc(rng_key, *, d_w, d_theta, d_x0, log_likelihood_fn, energy_fn,
               config: Optional[SGMCMCConfig] = None,
               log_prior_fn: Optional[Callable] = None,
               correction_grad_fn: Optional[Callable] = None) -> SGMCMCResult:
    cfg = config or SGMCMCConfig()
    d_z = d_w + d_theta + d_x0

    def log_posterior(z):
        w, theta, x0 = _split_z(z, d_w, d_theta)
        lp = log_likelihood_fn(w, theta) - energy_fn(w, theta, x0)
        if log_prior_fn is not None:
            return lp + log_prior_fn(z)
        return (lp + _normal_logpdf(theta, jnp.asarray(cfg.theta_prior_std))
                + _normal_logpdf(x0, jnp.asarray(cfg.x0_prior_std)))

    grad_fn = jax.grad(log_posterior)

    precond = {"identity": identity, "rmsprop": rmsprop}[cfg.preconditioner]()
    kernel = (sgld(preconditioner=precond) if cfg.kernel == "sgld"
              else sghmc(alpha=cfg.alpha, preconditioner=precond))
    schedule = {
        "constant": lambda: constant(cfg.step_size),
        "cyclical": lambda: cyclical(cfg.iterations, cfg.num_cycles,
                                     cfg.step_size, cfg.exploration_ratio),
        "polynomial": lambda: polynomial(cfg.step_size, cfg.poly_b, cfg.poly_gamma),
    }[cfg.schedule]()

    key_init, key_run = jax.random.split(jnp.asarray(rng_key))
    z0 = cfg.init_std * jax.random.normal(key_init, (cfg.chains, d_z))
    states = jax.vmap(lambda z: kernel.init(key_init, z))(z0)

    def one_step(carry, inp):
        states, t = carry
        keys = inp
        sched = schedule(t)

        def chain_step(key, state):
            k_corr, k_step = jax.random.split(key)
            g = _clip(grad_fn(state.position), cfg.grad_clip)
            if correction_grad_fn is not None:
                g = g + correction_grad_fn(state.position, k_corr)
            temp = cfg.temperature * jnp.where(sched.do_sample, 1.0, 0.0)
            return kernel.step(k_step, state, g, sched.step_size, temp)

        states = jax.vmap(chain_step)(keys, states)
        lp = jax.vmap(lambda s: log_posterior(s.position))(states)
        return (states, t + 1), lp

    @jax.jit
    def run_chunk(states, t, keys):
        return jax.lax.scan(one_step, (states, t), keys)

    n_chunks = cfg.iterations // cfg.thinning
    kept, trace_t, trace_lp = [], [], []
    t = jnp.asarray(0)
    for c in range(n_chunks):
        key_run, sub = jax.random.split(key_run)
        keys = jax.random.split(sub, (cfg.thinning, cfg.chains))
        (states, t), lps = run_chunk(states, t, keys)
        step_now = (c + 1) * cfg.thinning
        if step_now > cfg.burn_in:
            kept.append(np.asarray(states.position))
        if step_now % cfg.trace_every == 0 or c == n_chunks - 1:
            trace_t.append(step_now)
            trace_lp.append(np.asarray(lps[-1]).tolist())

    z_samples = np.stack(kept, axis=1) if kept else np.zeros((cfg.chains, 0, d_z))
    w_s, th_s, x0_s = (z_samples[..., :d_w],
                       z_samples[..., d_w:d_w + d_theta],
                       z_samples[..., d_w + d_theta:])
    return SGMCMCResult(
        samples={"z_samples": z_samples, "w_samples": w_s,
                 "theta_samples": th_s, "x0_samples": x0_s},
        history={"step": trace_t, "log_posterior": trace_lp},
        final_state={"z": np.asarray(states.position)},
        config=cfg,
    )


def make_field_sampler(kernel, schedule, *, num_steps, keep_every=1, burn=0,
                       temperature=1.0, grad_clip=None):
    """Build the param_loop.py inner-sampler boundary callable.

    Given ``grad_fn(field, param_state, key) -> ascent-gradient pytree``,
    returns ``f(field_state, param_state, key) -> chain`` where ``chain`` has
    a leading sample axis, as ``sgld_parameter_sampler`` expects. Usage::

        inner = make_field_sampler(samplax.sgld(), samplax.constant(1e-4),
                                   num_steps=..., keep_every=..., burn=...)
        posterior_field_sampler_fn = lambda f, p, k: inner(grad_fn, f, p, k)
    """

    def run(grad_fn, field_state, param_state, key):
        state = kernel.init(key, field_state)

        def body(carry, inp):
            state, t = carry
            subkey = inp
            k_grad, k_step = jax.random.split(subkey)
            g = grad_fn(state.position, param_state, k_grad)
            if grad_clip is not None:
                flat = ravel_pytree(g)[0]
                norm = jnp.sqrt(jnp.sum(flat * flat))
                factor = jnp.where(norm > grad_clip, grad_clip / norm, 1.0)
                g = jax.tree_util.tree_map(lambda x: factor * x, g)
            sched = schedule(t)
            temp = temperature * jnp.where(sched.do_sample, 1.0, 0.0)
            state = kernel.step(k_step, state, g, sched.step_size, temp)
            return (state, t + 1), state.position

        (_, _), chain = jax.lax.scan(body, (state, jnp.asarray(0)),
                                     jax.random.split(key, num_steps))
        return jax.tree_util.tree_map(lambda x: x[burn::keep_every], chain)

    return run
